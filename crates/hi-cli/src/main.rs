mod bestof;
mod candidate_gate;
mod candidate_merge;
mod child_process;
mod commands;
mod complete;
mod config;
mod delegate;
mod feedback;
mod goal_drive;
mod goal_report;
mod landing;
mod orchestration;
mod project_context;
mod provider;
mod repl;
mod report;
mod review_target;
mod rsi_bootstrap;
mod rsi_observation;
mod rsi_policy;
mod rsi_remote;
mod session;
mod setup;
mod skeptic_review;
mod sync;
mod sync_store;
mod ui;

#[cfg(test)]
mod delegate_tests;

use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::Parser;

use hi_agent::{Agent, AgentConfig, CompactionKind, ObservationSink, VerificationMode};
use hi_ai::Provider;

use config::{Cli, ProviderName, RsiRequested, permits_missing_checkpoint};
use landing::{effective_prompt, print_landing, profile_infos, resolve_session};
use orchestration::{build_sync_config, run_best_of, run_hf_cli, run_mcp_command};
use project_context::{auto_memory_enabled, load_project_context};
use provider::{
    LiveModelMetadata, build_chain, build_provider, default_skeptic_model,
    effective_max_tokens_for_model, provider_label, resolve_live_model_metadata,
};
use repl::repl;
use report::{
    finish_initialization_trace, finish_interactive_trace, finish_turn_trace, one_shot_exit_code,
    pipeline_command, run_one_shot_cancellable, write_initialization_failure_report, write_report,
};
use review_target::{
    absolutize_path, maybe_chdir_to_prompt_review_target, resolve_runtime_roots,
};
use rsi_bootstrap::RsiBootstrap;
use rsi_observation::{ObservedUi, ToolObserver};
use session::JsonlSession;
use skeptic_review::run_skeptic_review;
use ui::PlainUi;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("\x1b[31merror: {error:#}\x1b[0m");
        std::process::exit(top_level_error_code(&error));
    }
}

fn top_level_error_code(error: &anyhow::Error) -> i32 {
    let message = format!("{error:#}").to_ascii_lowercase();
    if message.contains("usage:")
        || message.contains("parsing skeptic-review json")
        || message.contains("invalid configuration")
    {
        2
    } else {
        // Typed turn outcomes use 0/1/130 in the one-shot branch. Anything
        // escaping the top-level dispatcher is unrecovered setup, provider,
        // process-runner, or internal infrastructure failure.
        3
    }
}

async fn run() -> Result<()> {
    let raw_args = std::env::args().collect::<Vec<_>>();
    if raw_args.get(1).map(String::as_str) == Some("hf") {
        return run_hf_cli(&raw_args[2..]).await;
    }

    let cli = Cli::parse();
    if let Some(id) = cli.sync_session_id.as_deref()
        && let Err(err) = sync::validate_session_id(id)
    {
        eprintln!("{err}");
        std::process::exit(2);
    }
    if let Some(id) = cli.attach.as_deref()
        && let Err(err) = sync::validate_session_id(id)
    {
        eprintln!("{err}");
        std::process::exit(2);
    }
    if cli.resume_local && cli.attach.is_none() {
        eprintln!("--resume-local requires --attach <SESSION_ID>");
        std::process::exit(2);
    }
    if cli.attach.is_some() && cli.daemon {
        eprintln!("--attach and --daemon cannot be used together");
        std::process::exit(2);
    }

    if cli.show_config {
        let file = match config::load_config(cli.config.as_deref()) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("{err:#}");
                std::process::exit(2);
            }
        };
        match config::resolve(&cli, &file) {
            Ok(settings) => {
                let live = if settings.provider == ProviderName::Pipenetwork {
                    let provider = build_provider(&settings);
                    resolve_live_model_metadata(provider.as_ref(), &settings.model).await
                } else {
                    LiveModelMetadata {
                        context_window: None,
                        max_output_tokens: None,
                    }
                };
                let effective_max_tokens =
                    effective_max_tokens_for_model(&settings, live.max_output_tokens);
                println!("provider:   {}", provider_label(settings.provider));
                println!("model:      {}", settings.model);
                println!("base_url:   {}", settings.base_url);
                if let Some(mcp_url) = &settings.mcp_url {
                    println!("mcp_url:    {mcp_url}");
                }
                println!("max_tokens: {}", effective_max_tokens);
                if let Some(limit) = live.max_output_tokens {
                    println!("model_max_output_tokens: {limit}");
                }
                println!(
                    "thinking:   {}",
                    settings
                        .thinking_budget
                        .map(|b| b.to_string())
                        .unwrap_or_else(|| "off".into())
                );
                println!(
                    "reasoning:  {}",
                    settings
                        .reasoning_effort
                        .map(|e| e.as_str().to_string())
                        .unwrap_or_else(|| "off".into())
                );
                println!("tool_mode:  {:?}", settings.tool_mode);
                let rsi = config::resolve_rsi(&cli, &file)?;
                println!("rsi_requested: {rsi:?}");
                println!(
                    "rsi_active:    {}",
                    if rsi == RsiRequested::Off {
                        "off"
                    } else {
                        "on"
                    }
                );
                println!("rsi_latest_turn_fully_observed: none");
                println!("compat:     {:?}", settings.compat);
                println!("api_key:    {}", config::mask_key(&settings.api_key));
                return Ok(());
            }
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        }
    }

    if cli.list_sessions {
        return session::list_sessions();
    }

    let mut file = match config::load_config(cli.config.as_deref()) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(2);
        }
    };

    // First run on a real terminal with nothing configured: walk the user
    // through an interactive setup instead of erroring.
    let settings = if cli.prompt.is_none()
        && config::needs_setup(&cli, &file)
        && std::io::stdin().is_terminal()
    {
        setup::run().await?
    } else {
        // Otherwise print config/onboarding guidance plainly (no "Error:" prefix).
        match config::resolve(&cli, &file) {
            Ok(settings) => settings,
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        }
    };

    if cli.prompt.as_deref() == Some("mcp") {
        return run_mcp_command(&settings).await;
    }

    // Fold piped stdin into the one-shot prompt as context.
    let prompt_input = effective_prompt(&cli)?;
    let report_path = cli
        .report
        .as_ref()
        .map(|path| absolutize_path(path.as_path()))
        .transpose()?;
    if let Some(prompt) = prompt_input.as_deref() {
        maybe_chdir_to_prompt_review_target(prompt)?;
    }
    let (workspace_root, state_root) = resolve_runtime_roots()?;
    // Start the workspace file scan in the background immediately — it reads
    // and hashes every tracked file and is the single biggest startup cost.
    // Launching it here lets it overlap with quality resolution, session
    // loading, provider construction, project-context loading, and system
    // prompt building. The agent consumes the result via `from_background_scan`.
    let excluded_roots: Vec<std::path::PathBuf> = if state_root.starts_with(&workspace_root) {
        vec![state_root.clone()]
    } else {
        Vec::new()
    };
    let ledger_scan = hi_agent::BackgroundScan::start(
        &workspace_root,
        &excluded_roots,
        &std::collections::BTreeSet::new(),
    )
    .ok();
    let rsi = RsiBootstrap::initialize(&cli, &file, prompt_input.as_deref())?;
    let rsi_requested = rsi.requested;
    let quality = match config::resolve_quality(&cli, &workspace_root) {
        Ok(quality) => quality,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(2);
        }
    };
    let verify_stages = quality.verification.resolved_stages(&workspace_root);
    if matches!(quality.verification, VerificationMode::Auto) && !verify_stages.is_empty() {
        eprintln!(
            "\x1b[2mverification: auto ({})\x1b[0m",
            verify_stages
                .iter()
                .map(|stage| stage.command.as_str())
                .collect::<Vec<_>>()
                .join(" → ")
        );
    }

    if cli.best_of > 1 {
        let Some(prompt) = prompt_input.as_deref() else {
            eprintln!("--best-of requires a one-shot prompt");
            std::process::exit(2);
        };
        match run_best_of(
            &cli,
            &settings,
            &workspace_root,
            &state_root,
            &verify_stages,
            quality.max_verify_repairs,
            prompt,
            report_path.as_deref(),
        ) {
            Ok(true) => return Ok(()),
            Ok(false) => std::process::exit(1),
            Err(err) => {
                eprintln!("{err:#}");
                std::process::exit(2);
            }
        }
    }

    // Resolve which session file to use and any history to resume.
    let (session_path, loaded) = resolve_session(&cli)?;
    let mut feedback_session_id = feedback::session_id_from_path(&session_path);

    let fallbacks = config::resolve_fallbacks(&cli, &file);
    // Arc so the agent can share it with read-only `explore` subagents.
    let base_provider: std::sync::Arc<dyn Provider> = build_chain(&settings, fallbacks).into();
    let rsi_bundle = rsi_bootstrap::wrap_provider(
        &cli,
        &file,
        &settings,
        workspace_root.clone(),
        state_root.clone(),
        &rsi,
        base_provider,
    )?;
    let provider = rsi_bundle.provider;
    let rsi_control = rsi_bundle.rsi_control;
    let rsi_remote_switch = rsi_bundle.rsi_remote_switch;
    let live_metadata = if settings.provider == ProviderName::Pipenetwork {
        resolve_live_model_metadata(provider.as_ref(), &settings.model).await
    } else {
        LiveModelMetadata {
            context_window: None,
            max_output_tokens: None,
        }
    };
    let max_tokens = effective_max_tokens_for_model(&settings, live_metadata.max_output_tokens);
    rsi_bootstrap::bind_managed_effective(
        rsi.managed_runtime.as_ref(),
        &settings,
        quality.max_verify_repairs,
        quality.tool_set.label(),
        &cli,
        max_tokens,
    )?;
    // The goal planner (glm-5.2 on pipenetwork by default). `HI_PLANNER_MODEL`
    // overrides the profile. Planning is optional; every top-level CLI session
    // supports durable structured goals, falling back to one evolving milestone
    // when no dedicated planner is configured.
    let planner_model = std::env::var("HI_PLANNER_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| settings.planner_model.clone());
    // The `/goal team` skeptic model. `HI_SKEPTIC_MODEL` overrides the
    // profile, which overrides a provider-appropriate default — the gate must
    // work out of the box the moment `/goal team on` is used, with zero
    // configuration. Deliberately does NOT gate `long_horizon` — it's a
    // reviewer of the driver, not the driver.
    let skeptic_model = std::env::var("HI_SKEPTIC_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| settings.skeptic_model.clone())
        .or_else(|| Some(default_skeptic_model(settings.provider, &settings.model)));
    // Offline skeptic detector eval: review one (objective, sub_goal, diff) from
    // stdin and exit, before building the normal turn agent.
    if cli.skeptic_review {
        return run_skeptic_review(provider, &settings, skeptic_model).await;
    }
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
            allow_no_checkpoint: permits_missing_checkpoint(&cli),
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
    let mut agent = match agent_result {
        Ok(agent) => agent,
        Err(error) => {
            if let Some(path) = &report_path {
                let rsi_summary =
                    finish_initialization_trace(rsi.observer.as_ref(), &error)?;
                write_initialization_failure_report(
                    path,
                    &settings.model,
                    provider_label(settings.provider),
                    &error,
                    rsi_summary.as_ref(),
                    cli.max_tool_calls.unwrap_or(u32::MAX),
                )?;
            }
            return Err(error).context("initializing workspace runtime");
        }
    };
    let managed_context = cli
        .rsi_context_json
        .as_deref()
        .map(rsi_remote::load_managed_context)
        .transpose()?;
    agent.set_managed_rsi_context(managed_context);
    agent.restore_plan(restored_plan);
    // Attach the write-`delegate` subagent runner for any top-level agent (a
    // subagent can't delegate), regardless of whether write subagents start on —
    // so `/delegate on` can enable it at runtime. The tool stays gated by the
    // `write_subagents` advertisement; the runner just needs to be ready. It spawns
    // a `hi --subagent` child in an isolated worktree and applies only verified diffs.
    let delegate_runner: Option<std::sync::Arc<dyn hi_agent::DelegateRunner>> = if !cli.subagent
        && let Ok(exe) = std::env::current_exe()
    {
        let runner = delegate::CliDelegateRunner::new(
            exe,
            provider_label(settings.provider).to_string(),
            settings.model.clone(),
            settings.base_url.clone(),
            settings.api_key.clone(),
            pipeline_command(&verify_stages),
            cli.max_steps,
            quality.max_verify_repairs,
            workspace_root.clone(),
            state_root.clone(),
        )?;
        Some(std::sync::Arc::new(runner))
    } else {
        None
    };
    if let Some(runner) = &delegate_runner {
        agent.set_delegate_runner(runner.clone());
    }
    // Build the session sink: local JSONL always (unless --no-save/--subagent),
    // optionally multiplexed with a remote ipop sync sink (--sync or [sync] enabled).
    // When sync is on, also create a RemoteUi for live event streaming.
    // Clone the path before it's moved into JsonlSession — the daemon fallback
    // below may need to create its own session sink.
    let daemon_session_path = session_path.clone();
    let sync_store = sync_store::SyncStore::open()?;
    let legacy_enabled = file.sync.as_ref().is_some_and(|section| section.enabled);
    let mut persisted_sync_mode = sync_store.initialize_mode(legacy_enabled)?;
    if let Some(configured) = file.sync.as_ref().and_then(|section| section.mode) {
        sync_store.set_mode(configured)?;
        persisted_sync_mode = configured;
    }
    // CLI flags are process-only overrides and never rewrite the persisted
    // global policy.
    if cli.sync || cli.sync_session_id.is_some() {
        sync_store::set_process_mode_override(sync_store::SyncMode::On);
    }
    let sync_enabled = cli.sync
        || cli.sync_session_id.is_some()
        || persisted_sync_mode != sync_store::SyncMode::Off;
    let (mut sync_handle, mut remote_ui) = if !cli.no_save && !cli.subagent {
        let sync_config = build_sync_config(&settings, &cli, &file);
        let session_id = cli
            .sync_session_id
            .clone()
            .unwrap_or_else(|| feedback::session_id_from_path(&session_path));
        let remote = sync::RemoteSessionSink::new(sync_config.clone(), session_id.clone());
        let sync_session = sync::SyncSession::new(JsonlSession::new(session_path), remote);
        let handle = sync_session.remote_handle();
        agent.set_session(Box::new(sync_session));
        let remote_ui = std::sync::Arc::new(sync::RemoteUi::new(sync_config, session_id));
        (Some(handle), Some(remote_ui))
    } else {
        (None, None)
    };
    // The fleet launcher: how `/dashboard` spawns worktree-isolated child `hi`
    // runs (one per row turn), each appending to a parent-owned session file.
    let fleet_launcher = hi_tui::FleetLauncher {
        exe: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("hi")),
        provider: provider_label(settings.provider).to_string(),
        model: settings.model.clone(),
        base_url: settings.base_url.clone(),
        api_key: settings.api_key.clone(),
        verify: pipeline_command(&verify_stages),
        max_verify: quality.max_verify_repairs,
        max_steps: cli.max_steps.unwrap_or(0),
        session_path: Box::new(session::new_fleet_session_path),
        sessions: Box::new(|| {
            session::fleet_sessions()
                .into_iter()
                .map(|s| hi_tui::FleetSessionInfo {
                    id: s.id,
                    title: s.title,
                    age: s.age,
                    lines: s.lines,
                })
                .collect()
        }),
        resume_info: Box::new(|id| {
            let id = if id.is_empty() {
                // No id: the most recent fleet session in this project.
                session::fleet_sessions().into_iter().next()?.id
            } else {
                id.to_string()
            };
            let path = session::session_path(&id).ok().filter(|p| p.is_file())?;
            let title = session::fleet_sessions()
                .into_iter()
                .find(|s| s.id == id)
                .map(|s| s.title)
                .unwrap_or_else(|| id.clone());
            let goal = session::session_goal_summary(&path);
            Some(hi_tui::FleetResumeInfo {
                id,
                path,
                title,
                goal_active: goal.as_ref().is_some_and(|g| g.active),
                goal_done: goal.as_ref().map(|g| g.done).unwrap_or(0),
                goal_total: goal.as_ref().map(|g| g.total).unwrap_or(0),
            })
        }),
        loop_session_path: Box::new(session::new_loop_session_path),
        loops_file: session::loops_file(),
    };

    // Headless loop daemon: keep this project's loops firing without the TUI.
    if cli.loops_daemon {
        return hi_tui::run_loops_daemon(fleet_launcher).await;
    }

    // Attach mode: same-user API join. Smart by default —
    //   host alive + accepts_input → steer that runtime over ipop (no SSH)
    //   otherwise → continue the conversation on this machine (portable)
    // `--resume-local` forces portable continue; no flag forces smart.
    if let Some(attach_session_id) = cli.attach.clone() {
        let sync_config = build_sync_config(&settings, &cli, &file);
        if cli.resume_local {
            return sync::run_resume_local(
                sync_config,
                attach_session_id,
                &settings,
                &cli,
                &mut agent,
            )
            .await;
        }
        return sync::run_smart_attach(
            sync_config,
            attach_session_id,
            cli.input_token.clone(),
            &settings,
            &cli,
            &mut agent,
        )
        .await;
    }

    // Daemon mode: hold the agent resident and accept input from remote clients.
    // Requires sync to be enabled.
    if cli.daemon {
        let sync_config = build_sync_config(&settings, &cli, &file);
        let session_id = cli
            .sync_session_id
            .clone()
            .unwrap_or_else(|| feedback::session_id_from_path(&daemon_session_path));
        // Ensure sync handles exist (daemon requires sync).
        let (daemon_sync_handle, daemon_remote_ui) = if sync_handle.is_none() {
            let remote = sync::RemoteSessionSink::new(sync_config.clone(), session_id.clone());
            // Declare before registering: the flag rides in the registration body, and it is what
            // tells a remote client this session can actually be steered.
            remote.set_accepts_input(true);
            let sync_session =
                sync::SyncSession::new(JsonlSession::new(daemon_session_path), remote);
            let handle = sync_session.remote_handle();
            agent.set_session(Box::new(sync_session));
            let rui =
                std::sync::Arc::new(sync::RemoteUi::new(sync_config.clone(), session_id.clone()));
            (Some(handle), Some(rui))
        } else {
            // `--sync --daemon`: the sink already exists from the sync setup above and was built
            // without the flag. Claim it here, before `run_daemon_loop` registers the session.
            if let Some(handle) = sync_handle.as_ref() {
                handle.set_accepts_input(true);
            }
            (sync_handle.clone(), remote_ui.clone())
        };
        return sync::run_daemon_loop(
            agent,
            sync_config,
            session_id,
            daemon_sync_handle,
            daemon_remote_ui,
        )
        .await;
    }

    if let Some(mut prompt) = prompt_input {
        let mut restore_model_state: Option<hi_agent::AgentModelState> = None;
        if let Some(hi_agent::Command::Moa(arg)) = hi_agent::command::parse(&prompt) {
            let arg = arg.trim().to_string();
            if arg.is_empty() {
                eprintln!("usage: /moa <prompt>");
                std::process::exit(2);
            }
            restore_model_state = Some(agent.model_state());
            agent.set_model(hi_ai::MOA_MODEL_CONSERVATIVE.to_string(), None, None);
            prompt = arg;
        }
        // `--goal <objective>` (fleet rows): install a planner-decomposed goal
        // before the turn — but never re-plan when the resumed session already
        // carries one (later fleet turns drive the existing goal).
        if let Some(objective) = cli.goal.as_deref().map(str::trim).filter(|s| !s.is_empty())
            && agent.structured_goal().is_none()
        {
            if !cli.quiet {
                println!("\x1b[2mplanning goal with the planner model…\x1b[0m");
            }
            let steps = match agent.decompose_goal(objective).await {
                Ok(steps) if !steps.is_empty() => steps,
                _ => vec![objective.to_string()],
            };
            let mut goal = hi_agent::Goal::new(objective.to_string(), steps);
            // The skeptic gate is on by default for new goals; HI_GOAL_TEAM is a
            // two-way headless override — `0`/`false`/`off` disables it (e.g. a
            // fleet run that wants raw single-model throughput), anything else
            // (re-)enables it.
            if let Ok(value) = std::env::var("HI_GOAL_TEAM") {
                goal.team = !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off"
                );
            }
            if agent.set_structured_goal(Some(goal)).unwrap_or(false)
                && !cli.quiet
                && let Some(g) = agent.structured_goal()
            {
                println!(
                    "\x1b[2m✓ goal set — {} sub-goal(s)\x1b[0m",
                    g.sub_goals.len()
                );
            }
        }
        let checkpoint_count_before_turn = agent.checkpoint_count();
        let result = if let Some(ref rui) = remote_ui {
            // Multiplex: local UI renders normally, remote UI buffers for sync.
            let primary: Box<dyn hi_agent::Ui> = if cli.quiet {
                Box::new(ui::QuietUi)
            } else {
                Box::new(PlainUi::new())
            };
            let mut multi = sync::MultiplexUi {
                primary,
                remote: rui.clone(),
            };
            let tools = rsi.observer.as_ref().map(|observer| {
                ToolObserver::new(observer.clone() as std::sync::Arc<dyn ObservationSink>)
            });
            let mut observed = ObservedUi::new(&mut multi, tools);
            run_one_shot_cancellable(agent.run_turn(&prompt, &mut observed)).await
        } else {
            let mut plain = PlainUi::new();
            let mut quiet = ui::QuietUi;
            let view: &mut dyn hi_agent::Ui = if cli.quiet { &mut quiet } else { &mut plain };
            let tools = rsi.observer.as_ref().map(|observer| {
                ToolObserver::new(observer.clone() as std::sync::Arc<dyn ObservationSink>)
            });
            let mut observed = ObservedUi::new(view, tools);
            run_one_shot_cancellable(agent.run_turn(&prompt, &mut observed)).await
        };
        if let Some(state) = restore_model_state {
            agent.restore_model_state(state);
        }
        let result = if let Some(result) = result {
            result
        } else {
            agent.kill_background_processes();
            if agent.checkpoint_count() > checkpoint_count_before_turn
                && let Err(err) = agent.undo().await
            {
                eprintln!("\x1b[33mcouldn't roll back cancelled workspace edits: {err:#}\x1b[0m");
            }
            agent.finalize_cancelled_turn()
        };
        let failed_outcome = result.as_ref().err().map(|_| agent.finalize_failed_turn());
        let rsi_summary = finish_turn_trace(
            rsi.observer.as_ref(),
            &agent,
            &prompt,
            result.as_ref().ok().or(failed_outcome.as_ref()),
            result.as_ref().err(),
        );
        let rsi_summary = match rsi_summary {
            Ok(summary) => summary,
            Err(error) if rsi_requested == RsiRequested::Managed => {
                eprintln!("\x1b[31mmanaged RSI trace error: {error:#}\x1b[0m");
                std::process::exit(3);
            }
            Err(error) => {
                eprintln!("\x1b[33mRSI trace warning: {error:#}\x1b[0m");
                None
            }
        };
        agent.set_last_rsi_fully_observed(match rsi_requested {
            RsiRequested::Off => None,
            RsiRequested::Managed => Some(
                rsi_summary
                    .as_ref()
                    .is_some_and(|summary| summary.fully_observed),
            ),
            RsiRequested::Remote => None,
        });
        let report_result = if let Some(path) = &report_path {
            write_report(
                path,
                &agent,
                Some(&prompt),
                result.as_ref().ok().or(failed_outcome.as_ref()),
                result.as_ref().err(),
                rsi_summary.as_ref(),
            )
        } else {
            Ok(())
        };
        if let Err(err) = &result {
            let (kind, guidance) = hi_agent::classify_error(err);
            let suffix = if guidance.is_empty() {
                String::new()
            } else {
                format!(" — {guidance}")
            };
            eprintln!("\x1b[31m{kind}: {err:#}{suffix}\x1b[0m");
        }
        if let Err(err) = &report_result {
            eprintln!("\x1b[33mreport error: {err:#}\x1b[0m");
        }
        // A one-shot turn may have started background processes; don't leak them.
        agent.kill_background_processes();
        // Flush any pending sync records and live events to ipop before exiting.
        if let Some(handle) = &sync_handle {
            if let Err(err) = handle.flush().await {
                eprintln!("\x1b[33msync: {err:#}\x1b[0m");
            }
            handle.end_session().await;
        }
        if let Some(rui) = &remote_ui
            && let Err(err) = rui.flush().await
        {
            eprintln!("\x1b[33msync events: {err:#}\x1b[0m");
        }
        if report_result.is_err() {
            std::process::exit(3);
        }
        let exit_code = match &result {
            Ok(outcome) => one_shot_exit_code(outcome, cli.allow_unverified),
            Err(_) => 3,
        };
        if exit_code == 0 {
            return Ok(());
        }
        std::process::exit(exit_code);
    }

    // Auto-memory at the end of an interactive session (TUI or REPL), unless
    // disabled or the session isn't being saved (memory is a form of persistence).
    // One-shot prompts return above, so scripted/piped/eval runs never write it.
    let auto_memory = auto_memory_enabled(cli.no_memory, cli.no_save);

    let stdout_is_tty = std::io::stdout().is_terminal();
    let stdin_is_tty = std::io::stdin().is_terminal();
    let use_tui = !cli.plain && stdout_is_tty && stdin_is_tty;
    // Prefer the workspace last-session profile (when it still exists) so a
    // mid-session `/provider` switch is what the next bare `hi` resumes with.
    // Explicit `--profile` still wins; otherwise fall back to config default.
    let active_profile = cli.profile.clone().or_else(|| {
        if cli.model.is_none() && cli.provider.is_none() {
            config::load_last_session(std::path::Path::new("."))
                .and_then(|s| s.profile)
                .filter(|name| file.profiles.contains_key(name))
        } else {
            None
        }
    })
    .or_else(|| file.default_profile.clone());

    // Flush durable records and live events after each interactive turn. The
    // callback is synchronous because both frontends own their event loops;
    // the async flush is serialized by the sinks and retried on failure.
    let mut sync_flush_callback: Option<hi_tui::RemoteFlushCallback> =
        if sync_handle.is_some() || remote_ui.is_some() {
            let handle = sync_handle.clone();
            let rui = remote_ui.clone();
            Some(std::sync::Arc::new(move || {
                let handle = handle.clone();
                let rui = rui.clone();
                tokio::spawn(async move {
                    if let Some(handle) = handle {
                        let _ = handle.flush().await;
                    }
                    if let Some(rui) = rui {
                        let _ = rui.flush().await;
                    }
                });
            }))
        } else {
            None
        };

    // The full-screen TUI is the default interactive experience; fall back to
    // the plain REPL when not on a TTY, when --plain is set, or if it errors.
    if use_tui {
        // TUI session switching replaces these handles at runtime. Keeping the
        // indirection here makes live events, per-turn flushes, and shutdown
        // flushing follow the newly selected session instead of the one that
        // happened to be active at process startup.
        let tui_sync_handle = std::sync::Arc::new(std::sync::Mutex::new(sync_handle.clone()));
        let tui_remote_ui = std::sync::Arc::new(std::sync::Mutex::new(remote_ui.clone()));
        let tui_active_session_id =
            std::sync::Arc::new(std::sync::Mutex::new(feedback_session_id.clone()));
        // Build the profile list and resolver for `/provider` in the TUI.
        let profiles: Vec<hi_tui::ProfileInfo> = profile_infos(&file);
        let resolver: hi_tui::ProfileResolver = Box::new({
            let file = file.clone();
            move |name: &str| {
                let settings = config::resolve_named_profile(&file, name)?;
                let label = provider_label(settings.provider).to_string();
                let model = settings.model.clone();
                let provider = build_chain(&settings, Vec::new());
                Ok(hi_tui::SwitchedProvider {
                    provider,
                    model,
                    label,
                    max_tokens: settings.max_tokens,
                    max_tokens_explicit: settings.max_tokens_explicit,
                })
            }
        });
        let saver: hi_tui::ProfileSaver = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            move |data: &hi_tui::ProfileFormData| {
                let provider = data
                    .provider
                    .parse::<ProviderName>()
                    .map_err(|e| anyhow::anyhow!("invalid provider '{}': {e}", data.provider))?;
                let form = config::ProfileForm {
                    name: data.name.clone(),
                    provider,
                    api_key: data.api_key.clone(),
                    store_as_env: data.store_as_env,
                    model: data.model.clone(),
                    base_url: data.base_url.clone(),
                };
                let mut file = file.lock().unwrap();
                // Editing an existing profile must not wipe the fields the form
                // doesn't cover (max_tokens, fallback, tool_mode, …).
                let profile = match file.profiles.get(&data.name) {
                    Some(existing) => form.apply_to(existing),
                    None => form.to_profile(),
                };
                config::upsert_profile(&mut file, &data.name, profile, config_path.as_deref())?;
                // Return the updated profile list.
                Ok(profile_infos(&file))
            }
        });
        let loader: hi_tui::ProfileLoader = Box::new({
            let file = file.clone();
            move |name: &str| {
                let p = file
                    .profiles
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("no profile named '{name}'"))?;
                let form = config::ProfileForm::from_profile(name, p);
                Ok(hi_tui::ProfileFormData {
                    name: form.name,
                    provider: form.provider.as_str().to_string(),
                    api_key: form.api_key,
                    store_as_env: form.store_as_env,
                    model: form.model,
                    base_url: form.base_url,
                })
            }
        });
        let remover: hi_tui::ProfileRemover = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            move |name: &str| {
                let mut file = file.lock().unwrap();
                let existed = config::remove_profile(&mut file, name, config_path.as_deref())?;
                if !existed {
                    anyhow::bail!("no profile named '{name}'");
                }
                Ok(profile_infos(&file))
            }
        });
        let mlx_switcher: hi_tui::MlxProfileSwitcher = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            move |run: &hi_tools::HfMlxRun| {
                let mut file = file.lock().unwrap();
                let profile = config::Profile {
                    provider: Some(ProviderName::Openai),
                    model: Some(run.model_id.clone()),
                    base_url: Some(run.base_url.clone()),
                    api_key: Some("local".to_string()),
                    max_tokens: Some(2048),
                    ..Default::default()
                };
                config::upsert_profile_as_default(
                    &mut file,
                    &run.profile_name,
                    profile,
                    config_path.as_deref(),
                )?;
                let settings = config::resolve_named_profile(&file, &run.profile_name)?;
                let label = provider_label(settings.provider).to_string();
                let model = settings.model.clone();
                let provider = build_chain(&settings, Vec::new());
                Ok(hi_tui::MlxProfileSwitch {
                    switched: hi_tui::SwitchedProvider {
                        provider,
                        model,
                        label,
                        max_tokens: settings.max_tokens,
                        max_tokens_explicit: settings.max_tokens_explicit,
                    },
                    profiles: profile_infos(&file),
                })
            }
        });
        // Snapshot provider/model into `.hi/last_session.toml` so the next bare
        // `hi` in this workspace resumes with the same routing.
        let session_remember: hi_tui::SessionRemember = {
            let root = workspace_root.clone();
            std::sync::Arc::new(move |profile: Option<&str>, provider: &str, model: &str| {
                if let Err(err) = config::remember_session(&root, profile, provider, model) {
                    eprintln!("\x1b[33mcouldn't remember session routing: {err:#}\x1b[0m");
                }
            })
        };
        // Build dynamic live-event and flush callbacks. Session switching swaps the
        // underlying handles, and these callbacks immediately follow them.
        let remote_event_tap: Option<hi_tui::RemoteEventTap> = remote_ui.as_ref().map(|_| {
            let state = tui_remote_ui.clone();
            std::sync::Arc::new(move |event: &hi_tui::event::UiEvent| {
                if let Some(rui) = state.lock().unwrap().as_ref() {
                    rui.push_event(event.clone());
                }
            }) as hi_tui::RemoteEventTap
        });
        let tui_sync_flush_callback: Option<hi_tui::RemoteFlushCallback> =
            sync_handle.is_some().then(|| {
                let handles = tui_sync_handle.clone();
                let events = tui_remote_ui.clone();
                std::sync::Arc::new(move || {
                    let handle = handles.lock().unwrap().clone();
                    let rui = events.lock().unwrap().clone();
                    tokio::spawn(async move {
                        if let Some(handle) = handle {
                            let _ = handle.flush().await;
                        }
                        if let Some(rui) = rui {
                            let _ = rui.flush().await;
                        }
                    });
                }) as hi_tui::RemoteFlushCallback
            });
        // Build the TUI sync config (for /sync, /sessions, /attach commands).
        let tui_sync_config = if sync_handle.is_some() || sync_enabled {
            let cfg = build_sync_config(&settings, &cli, &file);
            Some(hi_tui::SyncConfig {
                base_url: cfg.base_url,
                api_key: cfg.api_key,
                machine_id: cfg.machine_id,
                cwd_digest: cfg.cwd_digest,
            })
        } else {
            None
        };
        let tui_sync_session_id = cli
            .sync_session_id
            .clone()
            .or_else(|| Some(feedback::session_id_from_path(&daemon_session_path)));
        // Build the machine-cache side of the unified `/sessions` list.
        let session_lister: hi_tui::SessionLister = Box::new(|| {
            session::local_sessions()
                .into_iter()
                .map(|s| hi_tui::LocalSessionInfo {
                    id: s.id,
                    title: s.title,
                    age: s.age,
                    lines: s.lines,
                })
                .collect()
        });
        let session_switcher: Option<hi_tui::SessionSwitcher> = (!cli.no_save && !cli.subagent)
            .then(|| {
                let handles = tui_sync_handle.clone();
                let events = tui_remote_ui.clone();
                let active_session_id = tui_active_session_id.clone();
                let switch_sync_config = Some(build_sync_config(&settings, &cli, &file));
                let switcher: hi_tui::SessionSwitcher = Box::new(move |id, agent| {
                    let id = id.to_string();
                    let handles = handles.clone();
                    let events = events.clone();
                    let active_session_id = active_session_id.clone();
                    let switch_sync_config = switch_sync_config.clone();
                    Box::pin(async move {
                        sync::validate_session_id(&id)?;
                        let path = session::session_path(&id)?;
                        if !path.is_file() {
                            let config = switch_sync_config.as_ref().ok_or_else(|| {
                                anyhow!("session '{id}' is unavailable while sync is disabled")
                            })?;
                            let restored = sync::fetch_session_history(config, &id).await?;
                            session::cache_loaded_session(&path, &restored)?;
                        }
                        let loaded = session::load_history(&path)?;
                        let summary = session::resume_summary(&loaded);

                        let previous_handle = handles.lock().unwrap().clone();
                        let previous_events = events.lock().unwrap().clone();
                        let next_sync = if let Some(config) = &switch_sync_config {
                            let remote = sync::RemoteSessionSink::new(config.clone(), id.clone());
                            remote.seed_snapshot(&loaded)?;
                            // Stage the replacement completely, including the
                            // automatic takeover lease, before touching the
                            // live agent or persistence handles.
                            remote.ensure_registered_now().await?;
                            let synced =
                                sync::SyncSession::new(JsonlSession::new(path.clone()), remote);
                            let next_handle = synced.remote_handle();
                            let next_events = std::sync::Arc::new(sync::RemoteUi::new(
                                config.clone(),
                                id.clone(),
                            ));
                            Some((synced, next_handle, next_events))
                        } else {
                            None
                        };

                        agent.apply_loaded_session(
                            loaded.messages,
                            loaded.usage,
                            loaded.checkpoint_refs,
                            loaded.goal,
                            loaded.decisions,
                            loaded.plan,
                        );

                        if let Some((synced, next_handle, next_events)) = next_sync {
                            agent.set_session(Box::new(synced));
                            handles.lock().unwrap().replace(next_handle);
                            events.lock().unwrap().replace(next_events);
                        } else {
                            agent.set_session(Box::new(JsonlSession::new(path)));
                        }
                        *active_session_id.lock().unwrap() = id.clone();

                        if previous_handle.is_some() || previous_events.is_some() {
                            tokio::spawn(async move {
                                if let Some(remote_ui) = previous_events {
                                    let _ = remote_ui.flush().await;
                                }
                                if let Some(handle) = previous_handle {
                                    handle.end_session().await;
                                }
                            });
                        }

                        Ok(hi_tui::SessionSwitchInfo { id, summary })
                    })
                });
                switcher
            });
        let session_renamer: Option<hi_tui::SessionRenamer> =
            (!cli.no_save && !cli.subagent).then(|| {
                let handles = tui_sync_handle.clone();
                let active_session_id = tui_active_session_id.clone();
                Box::new(move |id: &str, name: &str| {
                    sync::validate_session_id(id)?;
                    let name = session::rename_session(id, name)?;
                    if *active_session_id.lock().unwrap() == id
                        && let Some(handle) = handles.lock().unwrap().as_ref()
                    {
                        handle.update_title(&name);
                    }
                    Ok(name)
                }) as hi_tui::SessionRenamer
            });
        // Host mode for the live TUI: advertise accepts_input and stream remote
        // attach prompts into the turn queue without spawning `hi --daemon`.
        let session_host: Option<hi_tui::SessionHostController> =
            (sync_handle.is_some() || sync_enabled).then(|| {
                let handles = tui_sync_handle.clone();
                let active_session_id = tui_active_session_id.clone();
                let host_sync_config = build_sync_config(&settings, &cli, &file);
                let controller: hi_tui::SessionHostController = Box::new(move |enable| {
                    let handles = handles.clone();
                    let active_session_id = active_session_id.clone();
                    let host_sync_config = host_sync_config.clone();
                    Box::pin(async move {
                        let handle = handles.lock().unwrap().clone().ok_or_else(|| {
                            anyhow!(
                                "no active synced session — run `/sessions attach <id>` or start with --sync"
                            )
                        })?;
                        let session_id = active_session_id.lock().unwrap().clone();
                        if enable {
                            handle.publish_accepts_input(true).await?;
                            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                            let poller = sync::spawn_remote_input_poller(
                                host_sync_config,
                                session_id,
                                handle.writer_lease_token(),
                                tx,
                            );
                            Ok(Some((rx, poller.abort_handle())))
                        } else {
                            handle.publish_accepts_input(false).await?;
                            Ok(None)
                        }
                    })
                });
                controller
            });
        let sync_control = hi_tui::SyncControl {
            set_mode: std::sync::Arc::new(|value| {
                let mode = match value {
                    "on" => sync_store::SyncMode::On,
                    "paused" => sync_store::SyncMode::Paused,
                    "off" => sync_store::SyncMode::Off,
                    _ => anyhow::bail!("mode must be on, paused, or off"),
                };
                sync_store::SyncStore::open()?.set_mode(mode)
            }),
            status: std::sync::Arc::new(|session_id| {
                let status = sync_store::SyncStore::open()?.status(session_id)?;
                Ok(format!(
                    "mode={} · queue={} rows/{} bytes · oldest={} · last success={} · error={} · next retry={} · quarantined={} · cursor={} · lease={} ({}) until {} · event drops={}",
                    status.mode.as_str(),
                    status.queue_rows,
                    status.queue_bytes,
                    status
                        .oldest_item_unix
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "none".into()),
                    status
                        .last_success_unix
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "never".into()),
                    status.last_error.as_deref().unwrap_or("none"),
                    status
                        .next_retry_unix
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "none".into()),
                    status.quarantined_records,
                    status.server_cursor,
                    status.lease_generation,
                    status.lease_owner.as_deref().unwrap_or("none"),
                    status.lease_expiry_unix,
                    status.event_drops,
                ))
            }),
            purge: std::sync::Arc::new(|| sync_store::SyncStore::open()?.purge()),
        };
        match hi_tui::run(
            &mut agent,
            hi_tui::RunOptions {
                provider: provider_label(settings.provider).to_string(),
                base_url: settings.base_url.clone(),
                model: settings.model.clone(),
                history_path: session::history_path(),
                auto_memory,
                profiles,
                active_profile: active_profile.clone(),
                resolver,
                saver,
                loader,
                remover,
                mlx_switcher,
                session_remember: Some(session_remember),
                resume_summary: resume_summary.clone(),
                mcp_url: settings.mcp_url.clone(),
                api_key: settings.api_key.clone(),
                fleet_launcher,
                remote_event_tap,
                remote_flush_callback: tui_sync_flush_callback,
                sync_config: tui_sync_config,
                sync_session_id: tui_sync_session_id,
                session_lister: Some(session_lister),
                session_switcher,
                session_renamer,
                session_host,
                sync_control: Some(sync_control),
            },
        )
        .await
        {
            Ok(()) => {
                let active_session_id = tui_active_session_id.lock().unwrap().clone();
                feedback::maybe_prompt_and_submit(&settings, &active_session_id).await;
                let active_handle = tui_sync_handle.lock().unwrap().clone();
                if let Some(handle) = &active_handle {
                    if let Err(err) = handle.flush().await {
                        eprintln!("\x1b[33msync: {err:#}\x1b[0m");
                    }
                    handle.end_session().await;
                }
                let active_remote_ui = tui_remote_ui.lock().unwrap().clone();
                if let Some(rui) = &active_remote_ui
                    && let Err(err) = rui.flush().await
                {
                    eprintln!("\x1b[33msync events: {err:#}\x1b[0m");
                }
                agent.kill_background_processes();
                finish_interactive_trace(rsi.observer.as_ref(), &agent)?;
                return Ok(());
            }
            Err(err) => {
                eprintln!("\x1b[33mTUI error ({err:#}); falling back to plain mode\x1b[0m");
                // A session switch may have replaced every sync handle while
                // the TUI was running. Carry the active handles into fallback
                // mode so subsequent turns and shutdown cannot write to or end
                // the session that was active only at startup.
                sync_handle = tui_sync_handle.lock().unwrap().clone();
                remote_ui = tui_remote_ui.lock().unwrap().clone();
                feedback_session_id = tui_active_session_id.lock().unwrap().clone();
                sync_flush_callback = if sync_handle.is_some() || remote_ui.is_some() {
                    let handle = sync_handle.clone();
                    let rui = remote_ui.clone();
                    Some(std::sync::Arc::new(move || {
                        let handle = handle.clone();
                        let rui = rui.clone();
                        tokio::spawn(async move {
                            if let Some(handle) = handle {
                                let _ = handle.flush().await;
                            }
                            if let Some(rui) = rui {
                                let _ = rui.flush().await;
                            }
                        });
                    }) as hi_tui::RemoteFlushCallback)
                } else {
                    None
                };
            }
        }
    }

    // Plain REPL startup (including TUI fallback): print normal-screen context
    // here, not before TUI launch. The TUI renders its own splash/resume summary
    // inside the alternate screen; printing them before entering TUI leaves a
    // stale banner in scrollback and makes a normal exit look like a crash.
    if let Some(summary) = &resume_summary {
        println!("\x1b[2m{summary}\x1b[0m");
    }
    if stdout_is_tty {
        print_landing(&settings, live_metadata.context_window);
    }

    let repl_result = repl(
        &mut agent,
        &settings,
        &mut file,
        auto_memory,
        active_profile,
        cli.config.clone(),
        sync_flush_callback,
    )
    .await;
    if repl_result.is_ok() {
        feedback::maybe_prompt_and_submit(&settings, &feedback_session_id).await;
    }
    if let Some(handle) = &sync_handle {
        if let Err(err) = handle.flush().await {
            eprintln!("\x1b[33msync: {err:#}\x1b[0m");
        }
        handle.end_session().await;
    }
    if let Some(rui) = &remote_ui
        && let Err(err) = rui.flush().await
    {
        eprintln!("\x1b[33msync events: {err:#}\x1b[0m");
    }
    finish_interactive_trace(rsi.observer.as_ref(), &agent)?;
    repl_result
}



#[cfg(test)]
mod tests {
    use super::top_level_error_code;
    use crate::landing::write_landing;
    use crate::project_context::{auto_memory_enabled, memory_context};
    use crate::provider::{
        default_skeptic_model, effective_max_tokens_for_model,
        resolve_live_model_metadata_with_timeout,
    };
    use crate::report::{
        one_shot_exit_code, report_tool_records, report_verification_stages,
        write_initialization_failure_report,
    };
    use crate::review_target::review_target_dir_from_prompt_at;
    use crate::config::{ProviderName, Settings};
    use anyhow::Result;
    use async_trait::async_trait;
    use hi_agent::VerifyStage;
    use hi_ai::{
        ChatRequest, CompatMode, Completion, Provider, ServedModel, StreamEvent, ToolMode,
    };
    use std::path::PathBuf;

    #[test]
    fn skeptic_defaults_to_glm_on_pipenetwork_and_session_model_elsewhere() {
        assert_eq!(
            default_skeptic_model(ProviderName::Pipenetwork, "ipop/coder-balanced"),
            "pipe/glm-5.2"
        );
        assert_eq!(
            default_skeptic_model(ProviderName::Anthropic, "claude-sonnet-5"),
            "claude-sonnet-5"
        );
        assert_eq!(
            default_skeptic_model(ProviderName::Ollama, "qwen2.5-coder"),
            "qwen2.5-coder"
        );
        // xAI: pin a strong Chat Completions reviewer, not the (often weaker) session model.
        assert_eq!(
            default_skeptic_model(ProviderName::Xai, "grok-4.3"),
            "grok-4.5"
        );
        assert_eq!(
            default_skeptic_model(ProviderName::Xai, "grok-code-fast-1"),
            "grok-4.5"
        );
    }

    struct HangingModelListProvider;

    #[async_trait]
    impl Provider for HangingModelListProvider {
        async fn stream(
            &self,
            _request: ChatRequest,
            _sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            unreachable!("metadata discovery must not start a chat request")
        }

        async fn list_models(&self) -> Result<Vec<ServedModel>> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn hanging_optional_model_metadata_cannot_stall_startup() {
        let started = std::time::Instant::now();
        let metadata = resolve_live_model_metadata_with_timeout(
            &HangingModelListProvider,
            "test-model",
            std::time::Duration::from_millis(20),
        )
        .await;

        assert_eq!(metadata.context_window, None);
        assert_eq!(metadata.max_output_tokens, None);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn auto_memory_off_when_disabled_or_unsaved() {
        assert!(auto_memory_enabled(false, false), "default on");
        assert!(!auto_memory_enabled(true, false), "--no-memory disables");
        assert!(!auto_memory_enabled(false, true), "--no-save disables");
    }

    #[test]
    fn one_shot_exit_codes_follow_v2_outcomes() {
        let outcome = |status, verification| hi_agent::TurnOutcome {
            status,
            verification,
            review: hi_agent::ReviewStatus::NotRequired,
            stop_reason: hi_agent::TurnStopReason::Completed,
            changed_files: Vec::new(),
            verified_workspace_revision: None,
            effective_route: hi_agent::EffectiveModelRoute {
                provider: Some("test".into()),
                model: "model".into(),
            },
        };
        assert_eq!(
            one_shot_exit_code(
                &outcome(
                    hi_agent::TurnStatus::Completed,
                    hi_agent::VerificationStatus::Passed,
                ),
                false,
            ),
            0
        );
        assert_eq!(
            one_shot_exit_code(
                &outcome(
                    hi_agent::TurnStatus::Completed,
                    hi_agent::VerificationStatus::Unverified,
                ),
                true,
            ),
            0
        );
        assert_eq!(
            one_shot_exit_code(
                &outcome(
                    hi_agent::TurnStatus::Incomplete,
                    hi_agent::VerificationStatus::Failed,
                ),
                false,
            ),
            1
        );
        assert_eq!(
            one_shot_exit_code(
                &outcome(
                    hi_agent::TurnStatus::Failed,
                    hi_agent::VerificationStatus::InfrastructureError,
                ),
                false,
            ),
            3
        );
        assert_eq!(
            one_shot_exit_code(
                &outcome(
                    hi_agent::TurnStatus::Cancelled,
                    hi_agent::VerificationStatus::Unverified,
                ),
                false,
            ),
            130
        );
    }

    #[test]
    fn report_stages_prefer_actual_execution_evidence() {
        let execution = hi_agent::VerificationExecution {
            round: 2,
            name: "test".into(),
            command: "cargo test".into(),
            status: hi_tools::ToolStatus::TimedOut,
            process: Some(hi_tools::ProcessOutcome {
                exit_code: None,
                stdout_summary: "partial output".into(),
                stderr_summary: String::new(),
                duration_ms: 30_000,
            }),
            truncation: Some(hi_tools::TruncationState::Truncated {
                original_bytes: 40_000,
                retained_bytes: 8_000,
            }),
        };
        let stages = report_verification_stages(
            &[execution],
            vec![VerifyStage::new("configured", "configured-command")],
        );

        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0]["round"], 2);
        assert_eq!(stages[0]["status"], "timed_out");
        assert_eq!(stages[0]["process"]["duration_ms"], 30_000);
        assert_eq!(stages[0]["truncation"]["state"], "truncated");
        assert_ne!(stages[0]["name"], "configured");
    }

    #[test]
    fn report_stages_do_not_claim_planned_checks_executed() {
        let stages =
            report_verification_stages(&[], vec![VerifyStage::new("check", "cargo check")]);
        assert!(stages.is_empty());
    }

    #[test]
    fn report_tool_records_preserve_typed_evidence() {
        let entry = hi_agent::ToolCallEntry {
            tool: "bash".into(),
            path: String::new(),
            duration_ms: 17,
            status: hi_tools::ToolStatus::Failed,
            background: None,
            process: Some(hi_tools::ProcessOutcome {
                exit_code: Some(9),
                stdout_summary: "partial stdout".into(),
                stderr_summary: "failed".into(),
                duration_ms: 17,
            }),
            effects: hi_tools::ToolEffects {
                mutation_attempted: true,
                mutation_applied: true,
                file_changes: vec![hi_tools::FileChange {
                    path: "src/lib.rs".into(),
                    kind: hi_tools::FileChangeKind::Modify,
                    before_digest: Some("sha256:before".into()),
                    after_digest: Some("sha256:after".into()),
                    before_len: Some(1),
                    after_len: Some(2),
                    before_mode: Some(0o100644),
                    after_mode: Some(0o100644),
                }],
            },
            truncation: hi_tools::TruncationState::Truncated {
                original_bytes: 100,
                retained_bytes: 20,
            },
            error: true,
            progress_kind: "weak".into(),
            progress_reason: "tool returned an error".into(),
            normalized_signature: None,
        };

        let records = report_tool_records(&[entry]);
        assert_eq!(records[0]["status"], "failed");
        assert_eq!(records[0]["process"]["exit_code"], 9);
        assert_eq!(records[0]["effects"]["mutation_applied"], true);
        assert_eq!(
            records[0]["effects"]["file_changes"][0]["path"],
            "src/lib.rs"
        );
        assert_eq!(records[0]["truncation"]["state"], "truncated");
    }

    #[test]
    fn top_level_errors_never_fall_back_to_outcome_exit_one() {
        assert_eq!(top_level_error_code(&anyhow::anyhow!("usage: bad flag")), 2);
        assert_eq!(
            top_level_error_code(&anyhow::anyhow!("workspace runner crashed")),
            3
        );
    }

    #[test]
    fn initialization_failure_still_writes_a_v2_report() {
        let path = std::env::temp_dir().join(format!(
            "hi-init-failure-report-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        write_initialization_failure_report(
            &path,
            "test-model",
            "test-provider",
            &anyhow::anyhow!("state root denied"),
            None,
            7,
        )
        .unwrap();
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(report["schema_version"], 2);
        assert_eq!(report["outcome"]["status"], "failed");
        assert_eq!(report["outcome"]["verification"], "infrastructure_error");
        assert_eq!(report["route"]["provider"], "test-provider");
        assert_eq!(report["changes"], serde_json::json!([]));
        assert_eq!(report["rsi"]["mode"], "off");
        assert_eq!(report["rsi"]["candidate_evidence"], true);
    }

    #[test]
    fn memory_context_wraps_nonempty_and_skips_blank() {
        let section = memory_context("- run cargo fmt before commits").unwrap();
        assert!(section.starts_with("# Memory (from past sessions)"));
        assert!(section.contains("- run cargo fmt before commits"));
        assert!(memory_context("   \n  ").is_none(), "blank → no section");
    }

    fn test_settings() -> Settings {
        Settings {
            provider: ProviderName::Openai,
            model: "gpt-4o".into(),
            base_url: String::new(),
            mcp_url: None,
            api_key: String::new(),
            max_tokens: 4096,
            max_tokens_explicit: true,
            thinking_budget: None,
            reasoning_effort: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
            curate_skills: false,
            explore_subagents: true,
            write_subagents: hi_agent::WriteSubagentPolicy::Risk,
            planner_model: None,
            skeptic_model: None,
            moa: hi_ai::MoaConfig::default(),
            api_unix_socket: None,
        }
    }

    fn pipenetwork_settings(model: &str, max_tokens: u32, explicit: bool) -> Settings {
        Settings {
            provider: ProviderName::Pipenetwork,
            model: model.into(),
            base_url: String::new(),
            mcp_url: None,
            api_key: String::new(),
            max_tokens,
            max_tokens_explicit: explicit,
            thinking_budget: None,
            reasoning_effort: None,
            tool_mode: ToolMode::default(),
            compat: CompatMode::default(),
            curate_skills: false,
            explore_subagents: true,
            write_subagents: hi_agent::WriteSubagentPolicy::Risk,
            planner_model: None,
            skeptic_model: None,
            moa: hi_ai::MoaConfig::default(),
            api_unix_socket: None,
        }
    }

    fn temp_review_dir(name: &str) -> PathBuf {
        let unique = format!(
            "hi-target-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        dir.canonicalize().unwrap()
    }

    #[test]
    fn review_target_detects_absolute_directory() {
        let dir = temp_review_dir("absolute");
        let cwd = std::env::current_dir().unwrap();

        let found = review_target_dir_from_prompt_at(
            &format!("review {} and discuss only", dir.display()),
            &cwd,
            None,
        )
        .unwrap();

        assert_eq!(found, dir);
        let _ = std::fs::remove_dir_all(found);
    }

    #[test]
    fn review_target_expands_home_directory() {
        let home = temp_review_dir("home");
        let repo = home.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let cwd = std::env::current_dir().unwrap();

        let found =
            review_target_dir_from_prompt_at("security review ~/repo read only", &cwd, Some(&home))
                .unwrap();

        assert_eq!(found, repo.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(home);
    }

    #[test]
    fn review_target_ignores_non_review_prompt() {
        let dir = temp_review_dir("non-review");
        let cwd = std::env::current_dir().unwrap();

        let found = review_target_dir_from_prompt_at(&format!("fix {}", dir.display()), &cwd, None);

        assert!(found.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn review_target_ignores_paths_only_in_folded_stdin() {
        let dir = temp_review_dir("stdin");
        let cwd = std::env::current_dir().unwrap();
        let prompt = format!("review codebase\n\nstdin:\n```\n{}\n```", dir.display());

        let found = review_target_dir_from_prompt_at(&prompt, &cwd, None);

        assert!(found.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn pipenetwork_coding_routes_apply_live_output_limits() {
        let balanced = pipenetwork_settings("ipop/coder-balanced", 8192, false);
        assert_eq!(
            effective_max_tokens_for_model(&balanced, Some(131_072)),
            131_072
        );

        let auto_code = pipenetwork_settings("pipe/auto-coder", 8192, false);
        assert_eq!(
            effective_max_tokens_for_model(&auto_code, Some(16_384)),
            16_384
        );
    }

    #[test]
    fn explicit_max_tokens_survive_live_metadata_but_clamp_down() {
        let lower = pipenetwork_settings("ipop/coder-balanced", 4096, true);
        assert_eq!(effective_max_tokens_for_model(&lower, Some(131_072)), 4096);

        let too_high = pipenetwork_settings("pipe/auto-coder", 65_536, true);
        assert_eq!(
            effective_max_tokens_for_model(&too_high, Some(16_384)),
            16_384
        );
    }

    /// `write_landing` renders the ~2x block-letter "PipeNetwork.AI" banner.
    /// We render into a `Vec<u8>`, strip ANSI escapes, and assert the banner
    /// shape (5 figlet rows), the trailing model/cwd lines, and that the raw
    /// output carries the orange SGR escape — no real file descriptors touched.
    #[test]
    fn write_landing_shows_full_pipenetwork_wordmark() {
        let mut buf: Vec<u8> = Vec::new();
        write_landing(&mut buf, &test_settings(), Some(128_000)).expect("render landing");

        let raw = String::from_utf8(buf).expect("utf8");
        let stripped = strip_ansi(&raw);
        let lines: Vec<&str> = stripped.lines().collect();

        // 5 banner rows + model line + cwd line = 7 content rows.
        assert!(
            lines.len() >= 7,
            "expected ≥7 lines (5 banner + model + cwd), got {}: {lines:?}",
            lines.len()
        );

        // The banner rows are the figlet art — they contain block-letter
        // strokes (pipes, underscores, slashes) and span 5 consecutive rows.
        let banner = &lines[0..5];
        // Every banner row is non-empty and carries pipe/underscore strokes.
        for (i, row) in banner.iter().enumerate() {
            assert!(
                row.contains('|') || row.contains('_'),
                "banner row {i} should carry figlet strokes, got: {row:?}"
            );
        }

        // Row 6 (index 5): model + provider + context window.
        let model_line = lines[5];
        assert!(
            model_line.contains("gpt-4o"),
            "model line missing model: {model_line:?}"
        );
        assert!(
            model_line.contains("openai"),
            "model line missing provider: {model_line:?}"
        );
        assert!(
            model_line.contains("128K context"),
            "model line missing context window: {model_line:?}"
        );

        // Row 7 (index 6): cwd — at minimum, non-empty (a path).
        assert!(
            !lines[6].is_empty(),
            "cwd line should be non-empty, got: {:?}",
            lines[6]
        );

        // The raw output must carry the orange SGR escape on banner rows.
        let orange_count = raw.matches("\x1b[38;2;255;140;0m").count();
        assert!(
            orange_count >= 5,
            "expected ≥5 orange SGR escapes (one per banner row), got {orange_count}"
        );
    }

    fn strip_ansi(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                // Skip until we pass a letter (the terminator of a CSI sequence).
                i += 2;
                while i < bytes.len() && !bytes[i].is_ascii_alphabetic() {
                    i += 1;
                }
                i += 1;
            } else {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
        out
    }
}
