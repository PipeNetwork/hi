//! Build the interactive [`Agent`] from CLI settings, quality, and session state.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hi_agent::{Agent, AgentConfig, CompactionKind};
use hi_ai::Provider;

use crate::config::{Cli, QualitySettings, RsiRequested, Settings, permits_missing_checkpoint};
use crate::goal_drive;
use crate::landing::LoadedAgentSession;
use crate::project_context::{
    load_candidate_project_context_from, load_standing_rules, load_trust_aware_project_context_from,
};
use crate::provider::{LiveModelMetadata, provider_label};

pub(crate) struct BuiltAgent {
    pub agent: Agent,
    pub resume_summary: Option<String>,
}

/// Construct [`AgentConfig`] and resume or create the session agent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_agent(
    cli: &Cli,
    settings: &Settings,
    quality: &QualitySettings,
    workspace_root: PathBuf,
    state_root: PathBuf,
    provider: Arc<dyn Provider>,
    live_metadata: &LiveModelMetadata,
    max_tokens: u32,
    effective_max_steps: u32,
    effective_max_tool_calls: u32,
    planner_model: Option<String>,
    skeptic_model: Option<String>,
    rsi_requested: RsiRequested,
    rsi_control: Option<Arc<dyn hi_agent::RsiControl>>,
    rsi_remote_switch: Option<Arc<std::sync::atomic::AtomicBool>>,
    loaded: Option<LoadedAgentSession>,
    ledger_scan: Option<hi_agent::BackgroundScan>,
    defer_launch_workspace_runtime: bool,
) -> Result<BuiltAgent> {
    hi_tools::configure_browser(settings.browser_enabled, settings.browser_allow_private);
    let measured = session_measured(cli.eval_input.is_some(), cli.report.is_some());
    let agent_config = AgentConfig {
        execution: if cli.subagent || cli.eval_input.is_some() {
            hi_agent::ExecutionMode::Ephemeral
        } else {
            settings.execution
        },
        paths: hi_agent::AgentPaths {
            workspace_root: workspace_root.clone(),
            state_root: state_root.clone(),
        },
        routing: hi_agent::AgentRouting {
            model: settings.model.clone(),
            provider_route: Some(provider_label(settings.provider).to_string()),
            requested_max_tokens: settings.max_tokens,
            max_tokens,
            max_tokens_explicit: settings.max_tokens_explicit,
            temperature: cli.temperature,
            top_p: settings.top_p,
            output_token_parameter: settings.output_token_parameter,
            thinking_budget: settings.thinking_budget,
            reasoning_effort: settings.reasoning_effort,
            tool_mode: settings.tool_mode,
            compat: settings.compat,
            deepseek_compat: settings.deepseek_compat,
            context_window: live_metadata.context_window,
        },
        gates: hi_agent::AgentGates {
            verification: quality.verification.clone(),
            max_verify_repairs: quality.max_verify_repairs,
            review: quality.review,
            allow_unverified: cli.allow_unverified,
            skeptic_fail_open: cli.skeptic_fail_open,
            allow_no_checkpoint: permits_missing_checkpoint(cli),
            lsp_mode: quality.lsp_mode,
            confirm_edits: cli.confirm_edits,
            dry_run: cli.dry_run,
            ..hi_agent::AgentGates::default()
        },
        loop_limits: resolved_loop_limits(cli, effective_max_steps, effective_max_tool_calls),
        harness: settings.harness.clone(),
        harness_session: Some(settings.session_harness.clone()),
        memory: hi_agent::AgentMemory {
            tool_set: quality.tool_set,
            disabled_tools: crate::tool_trim::disabled_tools(&state_root),
            // Env override lets you flip on skill auto-curation without editing a profile.
            // `--eval-input` and `--report` skip it the same way they skip finalize:
            // the extra completion is billed and does not belong in a measured cell.
            // hi-eval uses `--report`, not `--eval-input`.
            curate_skills: session_curate_skills(
                settings.curate_skills,
                std::env::var_os("HI_CURATE_SKILLS").is_some(),
                measured,
            ),
            learning: session_learning_enabled(cli.no_memory, cli.no_save),
            suggest_next_prompt: settings.suggest_next_prompt && !measured,
            // Pausing an autonomous coding turn is opt-in. Even when enabled,
            // the agent-side handler has a bounded wait and resumes with the
            // best available option on timeout.
            offer_ask_user: !measured
                && std::env::var("HI_ENABLE_ASK_USER")
                    .ok()
                    .is_some_and(|value| {
                        matches!(
                            value.trim().to_ascii_lowercase().as_str(),
                            "1" | "true" | "yes" | "on"
                        )
                    }),
            offer_memory: !cli.no_memory && !cli.no_save,
            offer_browser: settings.browser_enabled,
            browser_allow_private: settings.browser_allow_private,
            project_context: if cli.subagent {
                load_candidate_project_context_from(&workspace_root)
            } else {
                load_trust_aware_project_context_from(&workspace_root)
            },
            // Detached write children must not execute repository skills. A
            // project-local pack may shadow a built-in pack with the same
            // slug, so disable automatic pack injection at the Agent boundary.
            inject_review_skill: !cli.subagent,
            inject_stack_skill: !cli.subagent,
            standing_rules: load_standing_rules(),
            context_exclusions: quality.context_exclusions.clone(),
            auto_compact: !cli.no_auto_compact,
            compaction: cli
                .compaction
                .as_deref()
                .and_then(CompactionKind::from_arg)
                .unwrap_or(CompactionKind::Hybrid {
                    keep_recent: hi_agent::DEFAULT_KEEP_RECENT,
                }),
            finalize: !cli.no_finalize && !measured,
            ..hi_agent::AgentMemory::default()
        },
        subagents: hi_agent::AgentSubagents {
            explore_subagents: settings.explore_subagents
                || std::env::var_os("HI_EXPLORE_SUBAGENTS").is_some(),
            // Profile/settings choose Off/Risk/On; HI_WRITE_SUBAGENTS forces On
            // except in a managed worker, whose signed ledger is process-scoped.
            write_subagents: resolved_write_subagent_policy(
                rsi_requested,
                std::env::var_os("HI_WRITE_SUBAGENTS").is_some(),
                settings.write_subagents,
            ),
            // `--subagent` marks a delegate child: no explore/delegate offered (depth ≤ 1).
            is_subagent: cli.subagent,
            planner_model: planner_model.clone(),
            skeptic_model,
            // Opt-in: route the `/goal` skeptic review to a local (or any
            // OpenAI-compatible) endpoint via HI_SKEPTIC_ENDPOINT — e.g. a running
            // hi-local MLX/CUDA server. Requires HI_SKEPTIC_MODEL to name a model it
            // serves. Off unless the env var is set.
            skeptic_endpoint: std::env::var("HI_SKEPTIC_ENDPOINT")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            skeptic_endpoint_key: std::env::var("HI_SKEPTIC_ENDPOINT_KEY")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            // Team-role routes for executors. Env vars seed them at startup
            // (mirroring the skeptic knobs); `/team` adjusts them live.
            delegate_model: env_route("HI_DELEGATE_MODEL"),
            delegate_endpoint: env_route("HI_DELEGATE_ENDPOINT"),
            delegate_endpoint_key: env_route("HI_DELEGATE_ENDPOINT_KEY"),
            explore_model: env_route("HI_EXPLORE_MODEL"),
            explore_endpoint: env_route("HI_EXPLORE_ENDPOINT"),
            explore_endpoint_key: env_route("HI_EXPLORE_ENDPOINT_KEY"),
            editor_model: env_route("HI_EDITOR_MODEL"),
            editor_endpoint: env_route("HI_EDITOR_ENDPOINT"),
            editor_endpoint_key: env_route("HI_EDITOR_ENDPOINT_KEY"),
            // `/goal` is a core CLI contract, not a provider-specific feature.
            // Delegate children receive bounded tasks and therefore keep it off.
            long_horizon: goal_drive::long_horizon_enabled(cli.subagent),
        },
        rsi: hi_agent::AgentRsi {
            enabled: rsi_requested != RsiRequested::Off,
            managed: rsi_requested == RsiRequested::Managed,
            remote_switch: rsi_remote_switch.clone(),
            control: rsi_control,
        },
        suppress_initial_project_hooks: defer_launch_workspace_runtime || cli.subagent,
        defer_initial_lsp: defer_launch_workspace_runtime,
        ..AgentConfig::default()
    };
    let resume_summary = loaded.as_ref().and_then(|l| l.resume_summary.clone());
    let restored_plan = loaded.as_ref().map(|l| l.plan.clone()).unwrap_or_default();
    let restored_plan_drive = loaded.as_ref().map(|l| {
        (
            l.plan_drive_paused,
            l.plan_drive_resume_on_user_input,
            l.plan_drive_stall,
            l.plan_drive_evidence.clone(),
        )
    });
    let restored_plan_approval = loaded.as_ref().map(|l| l.plan_approval_parked);
    let restored_goal_drive = loaded
        .as_ref()
        .map(|l| (l.goal_drive_stall, l.goal_drive_evidence.clone()));
    let declared_provider_capabilities = provider.capabilities();
    let agent_result = match loaded {
        Some(loaded) => Agent::resume(
            provider,
            agent_config,
            loaded.messages,
            loaded.usage,
            loaded.checkpoint_refs,
            loaded.structured_goal,
            loaded.decisions,
        ),
        None => Agent::with_background_scan(provider, agent_config, ledger_scan),
    };
    let mut agent = agent_result.context("initializing workspace runtime")?;
    if let Some(capabilities) = &live_metadata.provider_capabilities {
        let registry = hi_ai::ProviderCapabilityRegistry::default();
        registry.seed_observation(
            hi_ai::CapabilityRoute::new(provider_label(settings.provider), settings.model.clone()),
            declared_provider_capabilities,
            hi_ai::CapabilityProbeObservation {
                capabilities: capabilities.clone(),
                actual_model_revision: capabilities.actual_model_revision.clone(),
            },
        );
        agent.set_provider_capability_registry(registry);
    }
    agent.set_usage_pricing(live_metadata.price);
    agent.restore_plan(restored_plan);
    if std::env::var_os("HI_LOOP_ID").is_some()
        && agent.permission_mode() == hi_agent::PermissionMode::Always
    {
        // Loop children are unattended: Auto so confirms park instead of YOLO.
        agent.set_permission_mode(hi_agent::PermissionMode::Auto);
    }
    if let Some((paused, resume_on_user_input, stall, evidence)) = restored_plan_drive {
        agent.restore_plan_drive_with_policy(paused, resume_on_user_input, stall, evidence);
    }
    if let Some(parked) = restored_plan_approval {
        agent.restore_plan_approval_parked(parked);
    }
    if let Some((stall, evidence)) = restored_goal_drive {
        agent.restore_goal_drive(stall, evidence);
    }

    Ok(BuiltAgent {
        agent,
        resume_summary,
    })
}

/// Read an optional team-role route env var (empty = unset).
fn env_route(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.trim().is_empty())
}

/// `--eval-input` and `--report` are one-shot / harness paths: no human,
/// no extra billed completions after the task. hi-eval uses `--report`.
fn session_measured(eval_input: bool, report: bool) -> bool {
    eval_input || report
}

fn resolved_loop_limits(
    cli: &Cli,
    effective_max_steps: u32,
    effective_max_tool_calls: u32,
) -> hi_agent::AgentLoopLimits {
    hi_agent::AgentLoopLimits {
        max_steps: effective_max_steps,
        max_tool_calls: effective_max_tool_calls,
        turn_soft_deadline: match cli.turn_deadline {
            Some(0) => None,
            Some(seconds) => Some(std::time::Duration::from_secs(seconds)),
            None => hi_agent::AgentLoopLimits::default().turn_soft_deadline,
        },
        ..hi_agent::AgentLoopLimits::default()
    }
}

/// Managed workers cannot hand work to an external delegate process because it
/// would not share their signed budget ledger and evidence trace.
fn resolved_write_subagent_policy(
    rsi_requested: RsiRequested,
    env_force: bool,
    configured: hi_agent::WriteSubagentPolicy,
) -> hi_agent::WriteSubagentPolicy {
    if rsi_requested == RsiRequested::Managed {
        hi_agent::WriteSubagentPolicy::Off
    } else if env_force {
        hi_agent::WriteSubagentPolicy::On
    } else {
        configured
    }
}

/// Skill auto-curation is a follow-up completion after a green mutating turn.
/// Measured cells skip it so the scored model is not billed for a curator call.
pub(crate) fn session_curate_skills(settings_on: bool, env_override: bool, measured: bool) -> bool {
    !measured && (settings_on || env_override)
}

/// Failure findings are durable cross-session learning, so both persistence
/// opt-outs must disable reading and writing them as well as markdown memory.
pub(crate) fn session_learning_enabled(no_memory: bool, no_save: bool) -> bool {
    !no_memory && !no_save
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{
        resolved_loop_limits, resolved_write_subagent_policy, session_curate_skills,
        session_learning_enabled, session_measured,
    };

    #[test]
    fn persistence_opt_outs_disable_failure_learning() {
        assert!(session_learning_enabled(false, false));
        assert!(!session_learning_enabled(true, false));
        assert!(!session_learning_enabled(false, true));
        assert!(!session_learning_enabled(true, true));
    }

    #[test]
    fn eval_disables_skill_curation() {
        assert!(session_curate_skills(true, false, false));
        assert!(session_curate_skills(false, true, false));
        assert!(!session_curate_skills(true, true, true));
        assert!(!session_curate_skills(false, false, false));
    }

    #[test]
    fn report_is_measured_like_eval_input() {
        assert!(!session_measured(false, false));
        assert!(session_measured(true, false));
        assert!(session_measured(false, true));
        assert!(!session_curate_skills(
            true,
            true,
            session_measured(false, true)
        ));
    }

    #[test]
    fn agent_loop_limits_use_the_prevalidated_effective_step_limit() {
        let cli = crate::config::Cli::try_parse_from(["hi"]).unwrap();
        let defaults = resolved_loop_limits(&cli, 12, u32::MAX);
        assert_eq!(defaults.max_steps, 12);
        assert_eq!(defaults.max_tool_calls, u32::MAX);
        assert_eq!(defaults.turn_soft_deadline, None);

        let explicit = crate::config::Cli::try_parse_from([
            "hi",
            "--max-steps",
            "7",
            "--max-tool-calls",
            "9",
            "--turn-deadline",
            "11",
        ])
        .unwrap();
        let explicit_limits = resolved_loop_limits(&explicit, 7, 9);
        assert_eq!(explicit_limits.max_steps, 7);
        assert_eq!(explicit_limits.max_tool_calls, 9);
        assert_eq!(
            explicit_limits.turn_soft_deadline,
            Some(std::time::Duration::from_secs(11))
        );
    }

    #[test]
    fn managed_workers_disable_external_write_subagents_even_when_forced() {
        use crate::config::RsiRequested;
        use hi_agent::WriteSubagentPolicy;

        assert_eq!(
            resolved_write_subagent_policy(RsiRequested::Managed, true, WriteSubagentPolicy::On,),
            WriteSubagentPolicy::Off
        );
        assert_eq!(
            resolved_write_subagent_policy(RsiRequested::Off, true, WriteSubagentPolicy::Off,),
            WriteSubagentPolicy::On
        );
        assert_eq!(
            resolved_write_subagent_policy(RsiRequested::Off, false, WriteSubagentPolicy::Risk,),
            WriteSubagentPolicy::Risk
        );
    }
}
