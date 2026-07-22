//! Build the interactive [`Agent`] from CLI settings, quality, and session state.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hi_agent::{Agent, AgentConfig, CompactionKind};
use hi_ai::Provider;

use crate::config::{Cli, QualitySettings, RsiRequested, Settings, permits_missing_checkpoint};
use crate::goal_drive;
use crate::landing::LoadedAgentSession;
use crate::project_context::load_project_context;
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
    planner_model: Option<String>,
    skeptic_model: Option<String>,
    rsi_requested: RsiRequested,
    rsi_control: Option<Arc<dyn hi_agent::RsiControl>>,
    rsi_remote_switch: Option<Arc<std::sync::atomic::AtomicBool>>,
    loaded: Option<LoadedAgentSession>,
    ledger_scan: Option<hi_agent::BackgroundScan>,
) -> Result<BuiltAgent> {
    let agent_config = AgentConfig {
        paths: hi_agent::AgentPaths {
            workspace_root: workspace_root.clone(),
            state_root: state_root.clone(),
            ..hi_agent::AgentPaths::default()
        },
        routing: hi_agent::AgentRouting {
            model: settings.model.clone(),
            provider_route: Some(provider_label(settings.provider).to_string()),
            requested_max_tokens: settings.max_tokens,
            max_tokens,
            max_tokens_explicit: settings.max_tokens_explicit,
            temperature: cli.temperature,
            thinking_budget: settings.thinking_budget,
            reasoning_effort: settings.reasoning_effort,
            tool_mode: settings.tool_mode,
            compat: settings.compat,
            context_window: live_metadata.context_window,
            ..hi_agent::AgentRouting::default()
        },
        gates: hi_agent::AgentGates {
            verification: quality.verification.clone(),
            max_verify_repairs: quality.max_verify_repairs,
            review: quality.review,
            allow_unverified: cli.allow_unverified,
            allow_no_checkpoint: permits_missing_checkpoint(cli),
            lsp_mode: quality.lsp_mode,
            confirm_edits: cli.confirm_edits,
            ..hi_agent::AgentGates::default()
        },
        loop_limits: hi_agent::AgentLoopLimits {
            max_steps: cli.max_steps.unwrap_or(u32::MAX),
            max_steps_explicit: cli.max_steps.is_some(),
            max_tool_calls: cli.max_tool_calls.unwrap_or(u32::MAX),
            ..hi_agent::AgentLoopLimits::default()
        },
        memory: hi_agent::AgentMemory {
            tool_set: quality.tool_set,
            // Env override lets you flip on skill auto-curation without editing a profile.
            curate_skills: settings.curate_skills || std::env::var_os("HI_CURATE_SKILLS").is_some(),
            project_context: load_project_context(),
            context_exclusions: quality.context_exclusions.clone(),
            auto_compact: !cli.no_auto_compact,
            compaction: cli
                .compaction
                .as_deref()
                .and_then(CompactionKind::from_arg)
                .unwrap_or(CompactionKind::Hybrid {
                    keep_recent: hi_agent::DEFAULT_KEEP_RECENT,
                }),
            finalize: !cli.no_finalize,
            ..hi_agent::AgentMemory::default()
        },
        subagents: hi_agent::AgentSubagents {
            explore_subagents: settings.explore_subagents
                || std::env::var_os("HI_EXPLORE_SUBAGENTS").is_some(),
            // Profile/settings choose Off/Risk/On; HI_WRITE_SUBAGENTS forces On.
            write_subagents: if std::env::var_os("HI_WRITE_SUBAGENTS").is_some() {
                hi_agent::WriteSubagentPolicy::On
            } else {
                settings.write_subagents
            },
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
            // `/goal` is a core CLI contract, not a provider-specific feature.
            // Delegate children receive bounded tasks and therefore keep it off.
            long_horizon: goal_drive::long_horizon_enabled(cli.subagent),
            ..hi_agent::AgentSubagents::default()
        },
        rsi: hi_agent::AgentRsi {
            enabled: rsi_requested != RsiRequested::Off,
            managed: rsi_requested == RsiRequested::Managed,
            remote_switch: rsi_remote_switch.clone(),
            control: rsi_control,
            ..hi_agent::AgentRsi::default()
        },
        ..AgentConfig::default()
    };
    let resume_summary = loaded.as_ref().and_then(|l| l.resume_summary.clone());
    let restored_plan = loaded.as_ref().map(|l| l.plan.clone()).unwrap_or_default();
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
    agent.restore_plan(restored_plan);

    Ok(BuiltAgent {
        agent,
        resume_summary,
    })
}
