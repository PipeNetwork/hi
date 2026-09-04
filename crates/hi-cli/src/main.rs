#![recursion_limit = "256"]
mod agent_build;
mod announcements;
mod approval_store;
mod auth;
mod bench;
mod bestof;
mod bootstrap;
mod browser_cmd;
mod candidate_gate;
mod candidate_merge;
mod child_process;
mod commands;
mod complete;
mod config;
mod delegate;
mod delegate_events;
mod diff_lab;
mod doctor;
mod eval;
mod eval_identity;
mod eval_report;
mod event_store;
mod feedback;
mod goal_drive;
mod goal_report;
mod landing;
mod learning_ledger;
mod local_runtime;
mod mcp_host;
mod mcp_serve;
mod operator_override_audit;
mod orchestration;
mod orchestration_benchmark;
mod orchestration_metrics;
mod outcome_route;
mod project_context;
mod provider;
mod race;
mod repl;
mod report;
mod resource_governor;
mod review_target;
mod rsi_bootstrap;
mod rsi_dev;
mod rsi_observation;
mod rsi_policy;
mod rsi_remote;
mod team_bench;
mod tickets;
mod tool_trim;
mod trace_cmd;
mod tuning_report;
// Wired by the managed RSI entry once descriptor-driven workflow launch lands;
// composition and contracts are complete and tested.
mod pipefs;
#[allow(dead_code)]
mod rsi_stage_model;
mod scheduler_ops;
mod session;
mod session_harness;
mod setup;
mod skeptic_review;
mod sync;
mod sync_store;
mod ui;
mod workflow;
mod workflow_cmd;
mod workspace_cmd;
mod x402;

#[cfg(test)]
mod delegate_tests;
/// Serializes tests that read or mutate the process-wide current directory.
/// `set_current_dir` is global to the test binary, so a test that changes it
/// races every concurrent test that reads it — `cwd_digest`, and anything
/// resolving a relative path. Held across the whole read-or-mutate section,
/// not just the call.
#[cfg(test)]
pub(crate) static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
use std::io::IsTerminal;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

use hi_agent::{ObservationSink, VerificationMode};
use hi_ai::Provider;

pub(crate) use bootstrap::validate_tui_event_trace_request;
use config::{ProviderName, RsiRequested};
use landing::{effective_prompt, print_landing, profile_infos, resolve_session};
use orchestration::{build_sync_config, run_best_of, run_hf_cli, run_mcp_cli};
use project_context::auto_memory_enabled;
use provider::{
    build_chain, default_skeptic_model, effective_max_tokens_for_model, provider_label,
    resolve_startup_route, startup_live_model_metadata,
};
use repl::repl;
use report::{
    finish_initialization_trace, finish_interactive_trace, finish_turn_trace, one_shot_exit_code,
    pipeline_command, run_one_shot_cancellable, write_initialization_failure_report, write_report,
};
use review_target::{absolutize_path, chdir_to_review_target, resolve_runtime_roots};
use rsi_bootstrap::RsiBootstrap;
use rsi_observation::{ObservedUi, ToolObserver};
use session::JsonlSession;
use skeptic_review::run_skeptic_review;
use ui::PlainUi;

fn main() {
    if let Err(error) = run_main() {
        eprintln!("\x1b[31merror: {error:#}\x1b[0m");
        std::process::exit(top_level_error_code(&error));
    }
}

fn run_main() -> Result<()> {
    // Process signal actions must be installed before Tokio creates workers.
    // The install call also registers the main thread's per-thread alt stack.
    let crash_dir = std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".hi/crash"))
        .unwrap_or_else(|_| PathBuf::from(".hi/crash"));
    if let Some(report) = hi_crash_handler::check_previous_crash(&crash_dir) {
        eprintln!(
            "hi crashed during your last session: {} (version {})",
            report.signal_name, report.app_version
        );
        eprintln!("  Report: {}", report.report_path.display());
    }
    hi_crash_handler::install(hi_crash_handler::CrashHandlerConfig {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        crash_dir,
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .on_thread_start(|| {
            if !hi_crash_handler::install_thread_alt_stack() {
                eprintln!(
                    "hi-crash-handler: failed to install alternate signal stack on runtime thread"
                );
            }
        })
        .build()
        .context("building async runtime")?;
    runtime.block_on(run())
}

fn top_level_error_code(error: &anyhow::Error) -> i32 {
    // Typed turn outcomes use 0/1/130 in the one-shot branch. Anything escaping
    // the top-level dispatcher is classified as usage/config (2) or infra (3).
    hi_agent::TopLevelErrorKind::from_anyhow(error).exit_code()
}

fn sync_session_enabled(
    durable_store_available: bool,
    explicitly_requested: bool,
    persisted_mode: Option<sync_store::SyncMode>,
    provider_default_on: bool,
) -> bool {
    durable_store_available
        && (explicitly_requested
            || persisted_mode.map_or(provider_default_on, |mode| {
                mode != sync_store::SyncMode::Off
            }))
}

fn canonical_session_identity(
    explicit_sync_id: Option<&str>,
    persisted_remote_id: Option<&str>,
    local_path: &std::path::Path,
) -> String {
    explicit_sync_id
        .or(persisted_remote_id)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| feedback::session_id_from_path(local_path))
}

fn pipefs_startup_authority_required(
    existing_session: bool,
    local_pipefs_hint: bool,
    persisted_pipefs_authority: bool,
    has_persisted_remote_identity: bool,
    has_explicit_remote_identity: bool,
) -> bool {
    local_pipefs_hint
        || persisted_pipefs_authority
        || (existing_session && (has_persisted_remote_identity || has_explicit_remote_identity))
}

fn completed_session_switch(canonical_id: String, summary: String) -> hi_tui::SessionSwitchInfo {
    hi_tui::SessionSwitchInfo {
        id: canonical_id,
        summary,
    }
}

async fn run() -> Result<()> {
    // `HI_STARTUP_TRACE=1` prints elapsed milestones for startup regressions.
    let startup_began = std::time::Instant::now();
    let startup_trace_on = std::env::var_os("HI_STARTUP_TRACE").is_some();
    macro_rules! startup_trace {
        ($label:expr) => {
            if startup_trace_on {
                eprintln!("[startup {:>9.2?}] {}", startup_began.elapsed(), $label);
            }
        };
    }

    let raw_args = std::env::args().collect::<Vec<_>>();
    match raw_args.get(1).map(String::as_str) {
        Some("workspace") => return workspace_cmd::run_cli(&raw_args[2..]).await,
        Some("announcements") => return announcements::run_cli(&raw_args[2..]).await,
        Some("hf") => return run_hf_cli(&raw_args[2..]).await,
        Some("mcp") => return run_mcp_cli(&raw_args[2..]).await,
        Some("doctor") => return doctor::run_doctor_cli(&raw_args[2..]).await,
        Some("debug") if raw_args.get(2).map(String::as_str) == Some("tui") => {
            return hi_tui::debug_harness::run_cli(&raw_args[3..]);
        }
        _ => {}
    }
    if raw_args.get(1).map(String::as_str) == Some("diff-lab") {
        return diff_lab::run_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("bench") {
        return bench::run_bench_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("eval") {
        return eval::run_eval_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("team-bench") {
        return team_bench::run_team_bench_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("metrics") {
        let (_, state_root) = resolve_runtime_roots()?;
        orchestration_metrics::print_dashboard(&state_root);
        println!("scheduler: {}", scheduler_ops::effective_summary());
        if let Some(data_root) = session::data_root() {
            let sessions = data_root.join("sessions");
            tuning_report::print_tuning_signals(&sessions, &state_root);
            learning_ledger::print_learning_report(&sessions, &state_root);
        }
        return Ok(());
    }
    if raw_args.get(1).map(String::as_str) == Some("intervention") {
        let (_, state_root) = resolve_runtime_roots()?;
        return learning_ledger::run_intervention_cli(&state_root, &raw_args[2..]);
    }
    if raw_args.get(1).map(String::as_str) == Some("tools") {
        let (_, state_root) = resolve_runtime_roots()?;
        let sessions = session::data_root().map(|root| root.join("sessions"));
        return tool_trim::run_tools_cli(&state_root, sessions.as_deref(), &raw_args[2..]);
    }
    if raw_args.get(1).map(String::as_str) == Some("update") {
        return run_update_command().await;
    }
    if raw_args.get(1).map(String::as_str) == Some("workflow") {
        return workflow_cmd::run_workflow_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("runtime") {
        return local_runtime::run_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("trace") {
        return trace_cmd::run_cli(&raw_args[2..]);
    }
    if raw_args.get(1).map(String::as_str) == Some("rsi") {
        return rsi_dev::run(&raw_args[2..]);
    }
    if raw_args.get(1).map(String::as_str) == Some("tickets") {
        return tickets::run_cli(&raw_args[2..]).await;
    }
    // Only the bare `hi setup` — "setup …" is a plausible start to a real
    // prompt, and swallowing it as a subcommand would be worse than not having
    // one. `hi setup fix my nginx config` stays a prompt.
    if raw_args.len() == 2 && raw_args[1] == "setup" {
        return run_setup_command().await;
    }
    if raw_args.get(1).map(String::as_str) == Some("auth") {
        return auth::run_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("browser") {
        return browser_cmd::run_cli(&raw_args[2..]);
    }

    let cli = bootstrap::parse_and_validate_cli();
    validate_tui_event_trace_request(
        &cli,
        std::io::stdin().is_terminal(),
        std::io::stdout().is_terminal(),
    )?;
    startup_trace!("cli parsed");
    // Install before any tool can run: with `--keep-background`, a completed
    // foreground command must not tree-kill the service it just detached.
    hi_tools::preserve_detached_descendants(cli.keep_background);
    if cli.benchmark_orchestration {
        orchestration_benchmark::run();
        return Ok(());
    }
    if let Some(result) = bootstrap::maybe_short_circuit(&cli).await {
        return result;
    }

    let mut file = match config::load_config(cli.config.as_deref()) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(2);
        }
    };

    // First run on a real terminal with nothing configured: walk the user
    // through an interactive setup instead of erroring. This deliberately also
    // covers `hi "some prompt"` — a one-shot prompt is the most natural first
    // command, and answering it with the onboarding text (which points at the
    // wizard) rather than the wizard itself is a dead end. The prompt runs
    // once setup finishes.
    let settings = if config::needs_setup(&cli, &file) && std::io::stdin().is_terminal() {
        let mut settings = setup::run(&mut file).await?;
        // Apply the ordinary contextual default after the wizard so a first
        // saved session is durable while first-run `--no-save` remains valid.
        settings.execution = config::resolve_execution_mode(&cli, None, file.execution)?;
        let session_harness = config::resolve_session_harness(&cli)?;
        let profile = cli
            .profile
            .as_ref()
            .or(file.default_profile.as_ref())
            .and_then(|name| file.profiles.get(name));
        settings.harness = config::resolve_harness(
            &file,
            profile,
            Some(session_harness.clone()),
            &cli.harness_settings,
        )?;
        settings.session_harness = session_harness;
        settings
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
    session::session_shadow::configure(
        settings.harness.features.session_reducer_v2,
        settings.harness.features.session_projection_v2,
    );
    if settings.execution.is_durable() && cli.no_save && !cli.subagent {
        anyhow::bail!(
            "durable execution requires a persisted session; remove --no-save or disable durable mode"
        );
    }
    // A normal interactive session must paint the TUI before any managed MLX
    // model is downloaded or loaded. Plain/one-shot modes retain the
    // synchronous startup behavior so their provider is ready before output.
    let prefer_tui_startup = cli.prompt.is_none()
        && !cli.plain
        && std::io::stdin().is_terminal()
        && std::io::stdout().is_terminal();
    let (settings, startup_local_runtime, startup_local_spec) = if prefer_tui_startup {
        let startup_spec = prepare_managed_local_startup(&settings)?;
        (settings, None, startup_spec)
    } else {
        let (settings, ready) = ensure_managed_local_startup(settings).await?;
        (settings, ready, None)
    };
    outcome_route::install_research_defaults(&file, &settings);

    // Nothing was configured, but a provider key happened to be exported, so
    // `resolve` inferred everything. Say so once — otherwise the session looks
    // configured, writes nothing, and stops working in the next shell that
    // doesn't export that variable.
    if let Some(env_name) = config::auto_selected_env(&cli, &file) {
        eprintln!(
            "\x1b[2musing {env_name} from the environment ({} · {}) — run `hi setup` to save a profile\x1b[0m",
            settings.model,
            provider_label(settings.provider),
        );
    }

    // Fold piped stdin into the one-shot prompt as context.
    let eval_input = cli
        .eval_input
        .as_deref()
        .map(landing::load_eval_input)
        .transpose()?;
    if cli.pipefs
        && (cli.no_save
            || cli.subagent
            || eval_input.is_some()
            || cli.best_of > 1
            || cli.benchmark_orchestration
            || cli.skeptic_review)
    {
        anyhow::bail!(
            "--pipefs requires a persisted HI session and cannot be used with --no-save, subagent, eval, benchmark, or best-of execution"
        );
    }
    if let Some(mode) = cli.eval_output.as_deref()
        && !matches!(mode, "workspace" | "final_message")
    {
        anyhow::bail!("--eval-output must be `workspace` or `final_message`");
    }
    if cli.eval_output.is_some() && eval_input.is_none() {
        anyhow::bail!("--eval-output requires --eval-input");
    }
    let prompt_input = if let Some(input) = &eval_input {
        Some(landing::eval_prompt(input)?)
    } else {
        effective_prompt(&cli)?
    };
    let eval_input_mode = match eval_input.as_ref() {
        Some(hi_eval::EvalInput::Transcript { .. }) => "transcript",
        Some(hi_eval::EvalInput::Prompt { .. }) | None => "prompt",
    };
    let eval_transcript_messages = eval_input.as_ref().and_then(|input| match input {
        hi_eval::EvalInput::Transcript { messages, .. } => Some(messages.len()),
        hi_eval::EvalInput::Prompt { .. } => None,
    });
    let eval_prompt_characters = prompt_input.as_deref().map(|prompt| prompt.chars().count());
    let report_path = cli
        .report
        .as_ref()
        .map(|path| absolutize_path(path.as_path()))
        .transpose()?;
    let report_max_steps =
        std::cell::Cell::new(cli.max_steps.unwrap_or(hi_agent::MAX_MODEL_ROUNDS));
    let report_max_tool_calls =
        std::cell::Cell::new(cli.max_tool_calls.unwrap_or(hi_agent::MAX_TOOL_CALLS));
    // Any failure between here and a constructed agent still leaves a
    // structured report when `--report` was requested: a parent driving this
    // process treats a missing report file as an unexplained crash.
    let report_init_failure = |error: &anyhow::Error, rsi: Option<&hi_trace::TraceSummary>| {
        let Some(path) = &report_path else { return };
        if let Err(report_error) = write_initialization_failure_report(
            path,
            &settings.model,
            provider_label(settings.provider),
            error,
            rsi,
            report_max_steps.get(),
            report_max_tool_calls.get(),
        ) {
            eprintln!("\x1b[33mreport error: {report_error:#}\x1b[0m");
        }
    };
    // A subagent's workspace root is contractual: the spawning machinery
    // (delegate worktrees, the workflow engine) sets the cwd, and the merge
    // gate compares the child's reported paths against that root. Never
    // re-root a child from `--review-target`.
    if !cli.subagent
        && let Some(target) = cli.review_target.as_deref()
    {
        chdir_to_review_target(target).inspect_err(|error| report_init_failure(error, None))?;
    }
    let (workspace_root, state_root) =
        resolve_runtime_roots().inspect_err(|error| report_init_failure(error, None))?;
    startup_trace!("runtime roots resolved");
    operator_override_audit::record_folder_trust_override(&workspace_root, &state_root);
    // Canonical interactive lifecycle events are local-first and best-effort
    // for ordinary progress. The store is independent from the RSI trace
    // path and is safe to omit if state storage is unavailable.
    let event_sink: Option<std::sync::Arc<dyn hi_events::EventSink>> = event_store::open_for_state(
        &state_root,
        session::loops_file()
            .map(|path| path.with_file_name("activity.jsonl"))
            .as_deref(),
    )
    .ok()
    .map(|store| std::sync::Arc::new(store) as std::sync::Arc<dyn hi_events::EventSink>);
    let approval_store: Option<std::sync::Arc<dyn hi_policy::ApprovalStore>> =
        approval_store::open_for_state(&state_root)
            .ok()
            .map(|store| {
                std::sync::Arc::new(store) as std::sync::Arc<dyn hi_policy::ApprovalStore>
            });
    // A prior process may have died after recording an interactive approval.
    // Mark those records abandoned before accepting a new turn; no approved
    // interactive operation can silently execute after restart.
    if let Some(store) = &approval_store {
        let _ = store.abandon_interactive();
    }
    if raw_args.get(1).map(String::as_str) == Some("inbox") {
        return crate::commands::run_inbox_argv(approval_store.as_deref(), &raw_args[2..]);
    }
    // Reconcile local control-plane leases before accepting new work. A
    // crashed worker fences its active effects as unknown; callers may only
    // retry them after an explicit reconciliation or an idempotency check.
    if let Ok(control_store) = hi_control::ControlStore::open_for_state(&state_root)
        && let Ok(recovered) = control_store.recover_expired_attempts(hi_control::now_ms())
        && !recovered.is_empty()
    {
        eprintln!(
            "recovered {} expired control-plane attempt(s)",
            recovered.len()
        );
    }
    let recovered = scheduler_ops::recover_stale_state(&state_root);
    startup_trace!("stale scheduler state recovered");
    if recovered > 0 {
        eprintln!("recovered {recovered} stale scheduler artifact(s)");
    }
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
    startup_trace!("background ledger scan launched");
    let automatic_workflow_plan_path =
        automatic_workflow_plan_path(&cli, &workspace_root, prompt_input.as_deref());
    let rsi = RsiBootstrap::initialize(
        &cli,
        &file,
        prompt_input.as_deref(),
        automatic_workflow_plan_path.is_some(),
    )
    .inspect_err(|error| report_init_failure(error, None))?;
    startup_trace!("rsi bootstrap initialized");
    let rsi_requested = rsi.requested;
    let mut quality = match config::resolve_quality(&cli, &workspace_root) {
        Ok(quality) => quality,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(2);
        }
    };
    quality.max_verify_repairs = rsi_bootstrap::effective_max_verify_repairs(
        quality.max_verify_repairs,
        quality.max_verify_repairs_explicit,
        rsi.managed_runtime.as_ref(),
    );
    startup_trace!("quality resolved");
    let verify_stages = quality.verification.resolved_stages(&workspace_root);
    startup_trace!("verify stages resolved");
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
    let (session_path, resolved_loaded) =
        resolve_session(&cli).inspect_err(|error| report_init_failure(error, None))?;
    let observed_session_harness = config::merge_session_harness(
        resolved_loaded
            .as_ref()
            .map(|loaded| loaded.harness_settings.clone())
            .unwrap_or_else(session_harness::empty_layer),
        &cli.session_harness_settings,
    )?;
    if observed_session_harness != settings.session_harness {
        anyhow::bail!("session harness settings changed during startup; retry the command");
    }
    if !cli.session_harness_settings.is_empty() {
        session_harness::append(&session_path, &observed_session_harness)
            .context("persisting session harness settings")?;
    }
    let existing_session = resolved_loaded.is_some();
    let persisted_remote_session_id = resolved_loaded
        .as_ref()
        .and_then(|loaded| loaded.remote_session_id.clone());
    let persisted_pipefs_enabled = resolved_loaded
        .as_ref()
        .and_then(|loaded| loaded.pipefs_enabled);
    let canonical_session_id = canonical_session_identity(
        cli.sync_session_id.as_deref(),
        persisted_remote_session_id.as_deref(),
        &session_path,
    );
    // Internal subagents/evals and isolated best-of workers deliberately do
    // not inherit the user's PipeFS default: none owns a persistent session
    // whose workspace can be committed and resumed.
    let pipefs_requested_for_new_session = cli.pipefs
        || (file.pipefs.is_enabled()
            && !cli.no_save
            && !cli.subagent
            && eval_input.is_none()
            && cli.best_of <= 1
            && !cli.benchmark_orchestration
            && !cli.skeptic_review);
    let defer_launch_workspace_runtime = existing_session
        || cli.sync_session_id.is_some()
        || cli.attach.is_some()
        || pipefs_requested_for_new_session;
    let loaded = if let Some(input) = &eval_input {
        landing::eval_loaded_session(input)?
    } else {
        resolved_loaded
    };
    startup_trace!("session resolved");
    let mut feedback_session_id = canonical_session_id.clone();

    let fallbacks = config::resolve_fallbacks(&cli, &file);
    let startup_route =
        resolve_startup_route(&settings, &fallbacks, workspace_root.display().to_string()).ok();
    if let Some(route) = &startup_route
        && let Ok(control_store) = hi_control::ControlStore::open_for_state(&state_root)
    {
        let principal = hi_control::Principal {
            id: "local-process".into(),
            kind: "local_cli".into(),
        };
        let _ = control_store.record_audit(&hi_control::AuditRecord {
            audit_id: uuid::Uuid::new_v4().to_string(),
            decision: "route_selected".into(),
            actor: principal.clone(),
            source: "interactive_startup".into(),
            scope: None,
            provenance: Some(hi_control::Provenance {
                principal,
                source: "interactive_startup".into(),
                run_id: None,
                attempt_id: None,
                parent_ref: None,
                correlation_id: None,
                policy_version: None,
            }),
            policy_snapshot: None,
            operation_digest: None,
            approval_id: None,
            route: Some(hi_control::RouteSnapshot {
                harness: Some(route.selected.harness.id.clone()),
                provider: Some(route.selected.model.provider.clone()),
                model: Some(route.selected.model.model.clone()),
                capability_digest: Some(route.capability_digest.clone()),
            }),
            effect_id: None,
            event_id: None,
            detail: (!route.rejected.is_empty()).then(|| {
                serde_json::to_string(&route.rejected)
                    .unwrap_or_else(|_| "fallbacks rejected".into())
            }),
            created_at_ms: hi_control::now_ms(),
        });
    }
    // Arc so the agent can share it with read-only `explore` subagents.
    let base_provider: std::sync::Arc<dyn Provider> = build_chain(&settings, fallbacks).into();
    startup_trace!("provider chain built");
    let rsi_bundle = rsi_bootstrap::wrap_provider(
        &cli,
        &file,
        &settings,
        &quality,
        workspace_root.clone(),
        state_root.clone(),
        &rsi,
        base_provider,
    )
    .inspect_err(|error| report_init_failure(error, None))?;
    startup_trace!("rsi provider wrapped");
    let provider = rsi_bundle.provider;
    let rsi_control = rsi_bundle.rsi_control;
    let rsi_remote_switch = rsi_bundle.rsi_remote_switch;
    // Optional provider metadata must not delay startup. The agent begins with
    // conservative limits; applying a later refresh would require safely
    // reconfiguring the already-built agent.
    let live_metadata = startup_live_model_metadata();
    let max_tokens = effective_max_tokens_for_model(&settings, live_metadata.max_output_tokens);
    let effective_max_steps =
        rsi_bootstrap::effective_max_steps(cli.max_steps, rsi.managed_runtime.as_ref());
    let effective_max_tool_calls =
        rsi_bootstrap::effective_max_tool_calls(cli.max_tool_calls, rsi.managed_runtime.as_ref());
    report_max_steps.set(effective_max_steps);
    report_max_tool_calls.set(effective_max_tool_calls);
    rsi_bootstrap::bind_managed_effective(
        rsi.managed_runtime.as_ref(),
        &settings,
        quality.max_verify_repairs,
        quality.tool_set.label(),
        &cli,
        effective_max_steps,
        effective_max_tool_calls,
        max_tokens,
    )
    .inspect_err(|error| report_init_failure(error, None))?;
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
    let agent_build::BuiltAgent {
        mut agent,
        resume_summary,
    } = match agent_build::build_agent(
        &cli,
        &settings,
        &quality,
        workspace_root.clone(),
        state_root.clone(),
        provider.clone(),
        &live_metadata,
        max_tokens,
        effective_max_steps,
        effective_max_tool_calls,
        planner_model,
        skeptic_model.clone(),
        rsi_requested,
        rsi_control.clone(),
        rsi_remote_switch.clone(),
        loaded,
        ledger_scan,
        defer_launch_workspace_runtime,
    ) {
        Ok(built) => built,
        Err(error) => {
            let rsi_summary = finish_initialization_trace(rsi.observer.as_ref(), &error)
                .unwrap_or_else(|trace_error| {
                    eprintln!("\x1b[33mRSI trace warning: {trace_error:#}\x1b[0m");
                    None
                });
            report_init_failure(&error, rsi_summary.as_ref());
            return Err(error);
        }
    };

    startup_trace!("agent built");
    let pipe_attach = mcp_host::decide_pipe_attach(
        settings.mcp_pipe_enabled,
        settings.mcp_url.as_deref(),
        &settings.api_key,
        settings.mcp_pipe_allow.clone(),
    )
    .ok();
    if !defer_launch_workspace_runtime && !cli.subagent {
        let (mcp, _pipe_status) = mcp_host::connect_workspace_mcp_with_policies(
            &workspace_root,
            &file.mcp_import.to_policy(),
            pipe_attach.as_ref(),
            &file.mcp.server_allowlists(),
        )
        .await;
        if let Some(mcp) = mcp {
            agent.attach_mcp(mcp);
        }
    }
    if !cli.no_memory && !cli.no_save {
        agent.attach_memory(std::sync::Arc::new(hi_agent::MarkdownMemory::new(
            workspace_root.clone(),
            true,
        )));
    }
    if let Some(runtime) = &startup_local_runtime {
        agent.register_driver_local_server(
            runtime.base_url.clone(),
            runtime.model_id.clone(),
            runtime.process_id.clone(),
        );
    }
    let managed_context = cli
        .rsi_context_json
        .as_deref()
        .map(rsi_remote::load_managed_context)
        .transpose()?;
    agent.set_managed_rsi_context(managed_context);
    // Attach the external write-`delegate` runner for ordinary top-level agents,
    // regardless of whether write subagents start on, so `/delegate on` can
    // enable it at runtime. Managed RSI cannot use it: another process would not
    // share the signed budget ledger or evidence trace. The tool stays gated by
    // `write_subagents`; the runner spawns `hi --subagent` in an isolated worktree
    // and applies only verified diffs.
    let delegate_runner: Option<std::sync::Arc<dyn hi_agent::DelegateRunner>> =
        if rsi_bootstrap::external_delegate_allowed(rsi_requested, cli.subagent)
            && let Ok(exe) = std::env::current_exe()
        {
            let runner = delegate::CliDelegateRunner::new(
                exe,
                provider_label(settings.provider).to_string(),
                settings.model.clone(),
                settings.base_url.clone(),
                settings.api_key.clone(),
                pipeline_command(&verify_stages),
                agent.max_steps_limit(),
                agent.max_tool_calls_cap(),
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
    startup_trace!("delegate runner ready");
    let daemon_session_path = session_path.clone();
    let legacy_enabled = file.sync.as_ref().is_some_and(|section| section.enabled);
    let configured_sync_mode = file.sync.as_ref().and_then(|section| section.mode);
    let explicit_sync = cli.sync
        || cli.sync_session_id.is_some()
        || cli.daemon
        || cli.attach.is_some()
        || pipefs_requested_for_new_session;
    let persist_local_session = !cli.no_save && !cli.subagent && cli.eval_input.is_none();
    let pipenetwork_default_on =
        settings.provider == ProviderName::Pipenetwork && !settings.api_key.is_empty();
    let sync_storage_needed = persist_local_session || explicit_sync;
    let durable_sync: Result<_> = if sync_storage_needed {
        (|| {
            let store = sync_store::SyncStore::open()?;
            let mut persisted_mode = store.initialize_mode(legacy_enabled)?;
            if let Some(configured) = configured_sync_mode {
                store.set_mode(configured)?;
                persisted_mode = Some(configured);
            }
            let healed_implicit_off = pipenetwork_default_on && store.heal_implicit_off()?;
            if healed_implicit_off {
                persisted_mode = None;
            }
            Ok((persisted_mode, healed_implicit_off))
        })()
    } else {
        Ok((None, false))
    };
    let (persisted_sync_mode, durable_sync_available) = match durable_sync {
        Ok((mode, healed)) => {
            if sync_storage_needed {
                startup_trace!("sync store opened");
            }
            if healed {
                eprintln!(
                    "hi: session sync is now on by default for pipenetwork — \
                     sessions appear in the dashboard console; /sync off to disable"
                );
            }
            (mode, sync_storage_needed)
        }
        Err(error) if explicit_sync => {
            return Err(error.context("opening required portal sync storage"));
        }
        Err(error) => {
            eprintln!(
                "\x1b[33mwarning: portal sync storage unavailable ({error:#}); \
                 continuing with local session persistence\x1b[0m"
            );
            (None, false)
        }
    };
    // CLI flags are process-only overrides and never rewrite the persisted
    // global policy.
    if cli.sync || cli.sync_session_id.is_some() || pipefs_requested_for_new_session {
        sync_store::set_process_mode_override(sync_store::SyncMode::On);
    }
    // A pipenetwork pairing syncs by default: the console is the product,
    // and the records go to the user's own account on the provider that
    // already serves every prompt. `/sync off` still wins — it persists a
    // user-marked row that beats the default.
    if pipenetwork_default_on {
        sync_store::set_process_mode_default(sync_store::SyncMode::On);
    }
    let sync_enabled = sync_session_enabled(
        durable_sync_available,
        cli.sync || cli.sync_session_id.is_some() || pipefs_requested_for_new_session,
        persisted_sync_mode,
        pipenetwork_default_on,
    );
    let (mut sync_handle, mut remote_ui) = if persist_local_session && sync_enabled {
        let sync_config = build_sync_config(&settings, &cli, &file);
        let session_id = canonical_session_id.clone();
        let remote = sync::RemoteSessionSink::new(sync_config.clone(), session_id.clone());
        startup_trace!("remote sync sink built");
        let mut local_session = JsonlSession::new(session_path);
        if cli.sync_session_id.is_some()
            && persisted_remote_session_id.as_deref() != Some(session_id.as_str())
        {
            local_session
                .record_remote_session_identity(&session_id)
                .context("persisting the canonical remote session identity")?;
        }
        let sync_session = sync::SyncSession::new(local_session, remote);
        startup_trace!("sync session reconciled");
        let handle = sync_session.remote_handle();
        agent.set_session(Box::new(sync_session));
        let remote_ui = std::sync::Arc::new(sync::RemoteUi::new(sync_config, session_id)?);
        (Some(handle), Some(remote_ui))
    } else {
        if persist_local_session {
            agent.set_session(Box::new(JsonlSession::new(session_path)));
        }
        (None, None)
    };
    let pipefs_sync_handle: pipefs::SharedSyncHandle =
        std::sync::Arc::new(std::sync::Mutex::new(sync_handle.clone()));
    let mut pipefs_startup_checked = false;
    let pipefs_host = if persist_local_session {
        let sync_config = build_sync_config(&settings, &cli, &file);
        let session_id = canonical_session_id.clone();
        let host = std::sync::Arc::new(pipefs::PipeFsHost::new(
            sync_config,
            session_id,
            daemon_session_path.clone(),
            pipefs_sync_handle.clone(),
            workspace_root.clone(),
            state_root.clone(),
            pipefs::PipeFsMcpConfig::resolve(&settings, &file),
        )?);
        let local_pipefs_hint = host.local_state_requires_remote_probe();
        let persisted_pipefs_authority = persisted_pipefs_enabled == Some(true);
        // A canonical remote identity is itself an authority signal. Always
        // ask IPOP before activating a launch directory for such a resumed
        // session: another machine may have enabled PipeFS before its
        // best-effort transcript hint reached this cache.
        let must_resolve_remote_pipefs = pipefs_startup_authority_required(
            existing_session,
            local_pipefs_hint,
            persisted_pipefs_authority,
            persisted_remote_session_id.is_some(),
            cli.sync_session_id.is_some(),
        );
        if pipefs_requested_for_new_session || must_resolve_remote_pipefs {
            host.activate_for_startup(
                &mut agent,
                pipefs_requested_for_new_session,
                must_resolve_remote_pipefs,
            )
            .await?;
            // Startup may have lazily upgraded a saved local session to the
            // shared transcript/lease transport after discovering a remote
            // PipeFS head.
            sync_handle = pipefs_sync_handle.lock().unwrap().clone();
            pipefs_startup_checked = true;
        }
        Some(host)
    } else {
        None
    };
    if defer_launch_workspace_runtime && !pipefs_startup_checked && cli.attach.is_none() {
        // A local resume can be a PipeFS startup candidate before persisted
        // sync policy is resolved. If no remote session sink exists, finish
        // the deferred ordinary runtime now rather than leaving LSP, project
        // hooks, and repository MCP silently disabled for the whole session.
        // Do not rebind the same root here: that would discard resumed task
        // state and durable checkpoint references.
        agent.activate_deferred_local_workspace_runtime();
        let (mcp, _) = mcp_host::connect_workspace_mcp_with_policies(
            &workspace_root,
            &file.mcp_import.to_policy(),
            pipe_attach.as_ref(),
            &file.mcp.server_allowlists(),
        )
        .await;
        if let Some(mcp) = mcp {
            agent.attach_mcp(mcp);
        }
    }
    if cli.keep_background
        && let Some(host) = &pipefs_host
        && host.is_active().await
    {
        host.clean_exit(&mut agent)
            .await
            .context("cleaning the PipeFS materialization after rejecting --keep-background")?;
        anyhow::bail!(
            "--keep-background cannot be used with PipeFS because the local materialization is removed on clean exit"
        );
    }
    // Records earlier runs left behind (a one-shot that exited under an open
    // breaker, an interrupted session) belong to sessions no process will
    // ever flush again. Drain them in the background; the one-shot exit path
    // waits briefly for this so a short run still ships its predecessor's
    // backlog.
    let stranded_drain = sync_handle.as_ref().map(|handle| {
        let handle = handle.clone();
        tokio::spawn(async move { handle.drain_stranded_sessions().await })
    });
    // The fleet launcher: how `/dashboard` spawns worktree-isolated child `hi`
    // runs (one per row turn), each appending to a parent-owned session file.
    let fleet_launcher = hi_tui::FleetLauncher {
        exe: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("hi")),
        workspace_root: workspace_root.clone(),
        provider: provider_label(settings.provider).to_string(),
        model: settings.model.clone(),
        base_url: settings.base_url.clone(),
        api_key: settings.api_key.clone(),
        verify: pipeline_command(&verify_stages),
        max_verify: quality.max_verify_repairs,
        max_steps: std::sync::atomic::AtomicU32::new(agent.max_steps_limit().unwrap_or(0)),
        max_tool_calls: std::sync::atomic::AtomicU64::new(
            agent
                .max_tool_calls_cap()
                .map(u64::from)
                .unwrap_or(u64::MAX),
        ),
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

    // Headless workflow execution: `--workflow <name> [args]` runs a script
    // with a stub host and prints the outcome. Does not start the TUI or an
    // agent session. Runs in a blocking thread because the workflow engine
    // uses synchronous channel receives.
    if let Some(workflow_arg) = cli.workflow.clone() {
        tokio::task::spawn_blocking(move || workflow::handle_workflow_command(&workflow_arg))
            .await?;
        return Ok(());
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
        let session_id = canonical_session_id.clone();
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
            let rui = std::sync::Arc::new(sync::RemoteUi::new(
                sync_config.clone(),
                session_id.clone(),
            )?);
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
        // Plain one-shot/headless checklist `plan.md` is the workflow runner's
        // job. Fleet `--session-file` children already have a worktree — they
        // ingest in-process below instead of nesting `workflow run`.
        if let Some(plan_path) = automatic_workflow_plan_path {
            let mut args = vec!["run".to_string(), plan_path];
            if !cli.verify.is_empty() {
                args.push("--verify".into());
                args.push(cli.verify.join(" && "));
            }
            if let Some(max_steps) = agent.max_steps_limit() {
                args.push("--max-steps".into());
                args.push(max_steps.to_string());
            }
            if let Some(max_tool_calls) = agent.max_tool_calls_cap() {
                args.push("--max-tool-calls".into());
                args.push(max_tool_calls.to_string());
            }
            if let Some(max_verify_repairs) = agent.max_verify_repairs_cap() {
                args.push("--max-verify-repairs".into());
                args.push(max_verify_repairs.to_string());
            }
            return workflow_cmd::run_workflow_cli(&args).await;
        }
        // `--goal <objective>` (fleet rows): ingest a checklist or planner-
        // decompose before the turn — but never re-plan when the resumed
        // session already carries one (later fleet turns drive the existing goal).
        if let Some(objective) = cli.goal.as_deref().map(str::trim).filter(|s| !s.is_empty())
            && agent.structured_goal().is_none()
        {
            let mut goal = if let Some(ingested) = agent.try_ingest_goal(objective) {
                ingested
            } else {
                if !cli.quiet {
                    println!("\x1b[2mplanning goal with the planner model…\x1b[0m");
                }
                let steps = match agent.decompose_goal(objective).await {
                    Ok(steps) if !steps.is_empty() => steps,
                    _ => vec![objective.to_string()],
                };
                hi_agent::Goal::new(objective.to_string(), steps)
            };
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
                for warning in g.actionability_warnings() {
                    println!("\x1b[33m  {warning} (driving in-session anyway)\x1b[0m");
                }
            }
        }
        let mut current_prompt = prompt;
        let mut drive_turns = 0u32;
        let mut result;
        let mut failed_outcome;
        loop {
            let kind = hi_agent::DriveKind::from_prompt(&current_prompt);
            let goal_before = agent.structured_goal().cloned();
            let plan_step_before = agent.next_plan_step_title().map(str::to_owned);
            let (turn_result, _interrupt_requested, cancellation_requested) = if let Some(ref rui) =
                remote_ui
            {
                // Multiplex: local UI renders normally, remote UI buffers for sync.
                let primary: Box<dyn hi_agent::Ui> = if cli.quiet {
                    Box::new(ui::QuietUi)
                } else {
                    Box::new(PlainUi::new())
                };
                let primary = delegate_events::wrap_event_ui(primary, cli.events_jsonl.as_deref());
                let mut multi = sync::MultiplexUi {
                    primary,
                    remote: rui.clone(),
                };
                let tools = rsi.observer.as_ref().map(|observer| {
                    ToolObserver::new(
                        observer.clone() as std::sync::Arc<dyn ObservationSink>,
                        observer.full_capture(),
                    )
                });
                let mut observed = ObservedUi::new(&mut multi, tools, approval_store.clone());
                let cancellation = hi_agent::TurnCancellation::new();
                let (result, interrupted) = run_one_shot_cancellable(
                    agent.run_turn_cancellable(
                        &current_prompt,
                        &mut observed,
                        cancellation.clone(),
                    ),
                    cancellation.clone(),
                )
                .await;
                (result, interrupted, cancellation.is_cancelled())
            } else {
                let inner: Box<dyn hi_agent::Ui> = if cli.quiet {
                    Box::new(ui::QuietUi)
                } else {
                    Box::new(PlainUi::new())
                };
                let mut view = delegate_events::wrap_event_ui(inner, cli.events_jsonl.as_deref());
                let tools = rsi.observer.as_ref().map(|observer| {
                    ToolObserver::new(
                        observer.clone() as std::sync::Arc<dyn ObservationSink>,
                        observer.full_capture(),
                    )
                });
                let mut observed = ObservedUi::new(&mut *view, tools, approval_store.clone());
                let cancellation = hi_agent::TurnCancellation::new();
                let (result, interrupted) = run_one_shot_cancellable(
                    agent.run_turn_cancellable(
                        &current_prompt,
                        &mut observed,
                        cancellation.clone(),
                    ),
                    cancellation.clone(),
                )
                .await;
                (result, interrupted, cancellation.is_cancelled())
            };
            result = turn_result;
            failed_outcome = match &result {
                // A hard turn timeout signals the same cancellation token,
                // waits for Agent-owned rollback, then preserves the historic
                // deadline error at the API boundary. Do not immediately run
                // Fail cleanup over that already-finalized Cancel outcome.
                Err(_)
                    if cancellation_requested
                        && agent.last_turn_outcome().is_some_and(|outcome| {
                            outcome.status == hi_agent::TurnStatus::Cancelled
                        }) =>
                {
                    agent.last_turn_outcome().cloned()
                }
                Err(_) => Some(
                    agent
                        .cleanup_turn(hi_agent::TurnCleanupKind::Fail)
                        .await
                        .map(|r| r.outcome)
                        .unwrap_or_else(|_| agent.finalize_failed_turn_snapshot_only()),
                ),
                Ok(_) => None,
            };
            // Ctrl-C and the configured turn timeout both stop synthetic drive.
            // A turn may have committed immediately before either signal and
            // legitimately return Completed, but that is not permission to
            // start another unattended turn after the deadline/interrupt.
            if let Ok(outcome) = &result
                && !cancellation_requested
            {
                if kind == hi_agent::DriveKind::Goal {
                    let made_progress = agent.goal_drive_turn_made_progress(goal_before.as_ref());
                    let progress = agent.note_goal_drive_progress(made_progress);
                    if !cli.quiet {
                        match progress {
                            hi_agent::GoalDriveProgress::Skipped { failed, next } => {
                                println!(
                                    "\x1b[33m{}\x1b[0m",
                                    hi_agent::goal_drive_skip_message(&failed, next.as_deref())
                                );
                            }
                            hi_agent::GoalDriveProgress::Parked => {
                                println!(
                                    "\x1b[33m{}\x1b[0m",
                                    hi_agent::goal_drive_park_message(
                                        agent.leftover_work().as_deref()
                                    )
                                );
                            }
                            _ => {}
                        }
                    }
                } else if kind == hi_agent::DriveKind::Plan {
                    let made_progress =
                        agent.plan_drive_turn_made_progress(plan_step_before.as_deref());
                    agent.note_plan_drive_progress(made_progress);
                }
                if let Some(count) = agent.take_goal_requeue_notice()
                    && !cli.quiet
                {
                    println!(
                        "\x1b[33m{}\x1b[0m",
                        hi_agent::goal_drive_requeue_message(count)
                    );
                }
                if let Some(next) = agent.drive_decision(Some(outcome)).prompt() {
                    let drive_limit = hi_agent::ONE_SHOT_DRIVE_TURN_LIMIT;
                    if drive_limit != u32::MAX && drive_turns >= drive_limit {
                        break;
                    }
                    drive_turns = drive_turns.saturating_add(1);
                    current_prompt = next.to_string();
                    continue;
                }
            }
            break;
        }
        if agent.approval_parked() {
            println!("{}", hi_agent::PARKED_FOR_APPROVAL_STATUS);
        }
        if let Some(state) = restore_model_state {
            agent.restore_model_state(state);
        }
        let rsi_summary = finish_turn_trace(
            rsi.observer.as_ref(),
            &agent,
            &current_prompt,
            result.as_ref().ok().or(failed_outcome.as_ref()),
            result.as_ref().err(),
        );
        let rsi_summary = match rsi_summary {
            Ok(summary) => summary,
            Err(error) if rsi_requested == RsiRequested::Managed => {
                eprintln!("\x1b[31mmanaged RSI trace error: {error:#}\x1b[0m");
                result = Err(error.context("managed RSI trace failed"));
                None
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
                Some(&current_prompt),
                result.as_ref().ok().or(failed_outcome.as_ref()),
                result.as_ref().err(),
                rsi_summary.as_ref(),
                eval_input_mode,
                eval_transcript_messages,
                eval_prompt_characters,
                cli.eval_output.as_deref().unwrap_or("workspace"),
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
        // A one-shot turn may have started background processes; don't leak
        // them — unless the caller asked for the opposite, because the
        // deliverable is a service that must outlive this process.
        if cli.keep_background {
            agent.release_background_services();
            agent.background_task_registry().kill_all().await;
        } else {
            agent.settle_workspace_for_exit().await?;
        }
        // Flush any pending sync records and live events to ipop before
        // exiting. Silent on failure by design: sync is best-effort mirroring
        // of a local-first session — everything unsent stays queued in the
        // durable outbox for a later process, and a portal outage must never
        // surface as an error in the coding workflow.
        if let Some(handle) = &sync_handle {
            let _ = handle.flush().await;
        }
        if let Some(host) = &pipefs_host {
            let result = host.clean_exit(&mut agent).await;
            if let Err(error) = result {
                eprintln!(
                    "\x1b[31mPipeFS exit blocked: {error:#}; recovery cache was retained\x1b[0m"
                );
                std::process::exit(3);
            }
        }
        if let Some(handle) = &sync_handle {
            handle.end_session().await;
        }
        if let Some(rui) = &remote_ui {
            let _ = rui.flush().await;
        }
        if let Some(drain) = stranded_drain {
            // Bounded: a slow portal must not hold a one-shot exit hostage;
            // whatever is left stays queued for the next run.
            let _ = tokio::time::timeout(std::time::Duration::from_secs(20), drain).await;
        }
        if report_result.is_err() {
            std::process::exit(3);
        }
        let exit_code = match &result {
            Ok(outcome) => one_shot_exit_code(
                outcome,
                cli.allow_unverified,
                goal_drive::one_shot_leftover_remains(&agent),
            ),
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
    // Start the announcement fetch now, but display later at a point where the
    // lines are actually visible: printing before/under the TUI lands in the
    // alternate screen and is erased, while still auto-hiding one-shot notices.
    let mut pending_announcements = (stdout_is_tty && stdin_is_tty).then(announcements::spawn_load);
    // Prefer the workspace last-session profile (when it still exists) so a
    // mid-session `/provider` switch is what the next bare `hi` resumes with.
    // Explicit `--profile` still wins. Provider-preset last sessions must NOT
    // fall back to `default_profile` or exit would rewrite last_session under
    // the default and lose the preset on the next launch.
    let active_profile = config::resolve_active_profile(&cli, &file, std::path::Path::new("."));

    // Flush durable records and live events after each interactive turn. The
    // callback is synchronous because both frontends own their event loops;
    // the async flush is serialized by the sinks and retried on failure.
    let mut sync_flush_callback: Option<hi_tui::RemoteFlushCallback> =
        if sync_handle.is_some() || remote_ui.is_some() || pipefs_host.is_some() {
            let handles = pipefs_sync_handle.clone();
            let rui = remote_ui.clone();
            Some(std::sync::Arc::new(move || {
                let handle = handles.lock().unwrap().clone();
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
        let tui_event_trace = cli
            .tui_events_jsonl
            .as_deref()
            .map(hi_tui::TuiEventTrace::open)
            .transpose()
            .context("initializing --tui-events-jsonl")?;
        // TUI session switching replaces these handles at runtime. Keeping the
        // indirection here makes live events, per-turn flushes, and shutdown
        // flushing follow the newly selected session instead of the one that
        // happened to be active at process startup.
        let tui_sync_handle = pipefs_sync_handle.clone();
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
                    tool_mode: settings.tool_mode,
                    local_runtime: None,
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
        let reasoning_effort_saver: hi_tui::ReasoningEffortSaver = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            // Empty profile name = machine-wide only (provider preset / no profile).
            move |name: &str, effort: Option<hi_ai::ReasoningEffort>| {
                let mut file = file.lock().unwrap();
                let profile = (!name.is_empty()).then_some(name);
                config::persist_reasoning_effort(&mut file, profile, effort, config_path.as_deref())
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
                    runtime: Some(config::LocalRuntimeProfile {
                        kind: "mlx".to_string(),
                        repo: run.repo.clone(),
                        backend: Some("mlx".to_string()),
                        autostart: true,
                        model_path: None,
                        quantization: None,
                        context_window: None,
                        tool_mode: Some(hi_ai::ToolMode::ChatOnly),
                    }),
                    ..Default::default()
                };
                config::upsert_profile_project_local(
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
                        tool_mode: settings.tool_mode,
                        local_runtime: Some(hi_tui::LocalRuntimeIdentity {
                            backend: "MLX".into(),
                            model_id: run.model_id.clone(),
                            quantization: None,
                            source: "Hub".into(),
                            endpoint: Some(run.base_url.clone()),
                            ready: true,
                        }),
                    },
                    profiles: profile_infos(&file),
                })
            }
        });
        let local_runtime_switcher: hi_tui::LocalRuntimeSwitcher = Box::new({
            let file = std::sync::Mutex::new(file.clone());
            let config_path = cli.config.clone();
            move |runtime: &hi_agent::local_skeptic::ManagedLocalRuntime| {
                let mut file = file.lock().unwrap();
                let profile = config::Profile {
                    provider: Some(ProviderName::Openai),
                    model: Some(runtime.model_id.clone()),
                    base_url: Some(runtime.base_url.clone()),
                    api_key: Some("local".to_string()),
                    max_tokens: Some(2048),
                    runtime: Some(config::LocalRuntimeProfile {
                        kind: "mlx".to_string(),
                        repo: runtime.repo.clone(),
                        backend: Some(runtime.backend.serve_flag().to_string()),
                        autostart: true,
                        model_path: match &runtime.source {
                            hi_agent::local_skeptic::LocalModelSource::Hub { .. } => None,
                            hi_agent::local_skeptic::LocalModelSource::Directory { path } => {
                                Some(path.clone())
                            }
                        },
                        quantization: runtime.quantization.clone(),
                        context_window: runtime.context_window,
                        tool_mode: Some(match runtime.tool_support {
                            hi_agent::local_skeptic::LocalToolSupport::ToolCapable => {
                                hi_ai::ToolMode::Auto
                            }
                            hi_agent::local_skeptic::LocalToolSupport::ChatOnly
                            | hi_agent::local_skeptic::LocalToolSupport::Unknown => {
                                hi_ai::ToolMode::ChatOnly
                            }
                        }),
                    }),
                    ..Default::default()
                };
                config::upsert_profile_project_local(
                    &mut file,
                    &runtime.profile_name,
                    profile,
                    config_path.as_deref(),
                )?;
                let settings = config::resolve_named_profile(&file, &runtime.profile_name)?;
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
                        tool_mode: settings.tool_mode,
                        local_runtime: Some(hi_tui::LocalRuntimeIdentity {
                            backend: runtime.backend.serve_flag().to_ascii_uppercase(),
                            model_id: runtime.model_id.clone(),
                            quantization: runtime.quantization.clone(),
                            source: match &runtime.source {
                                hi_agent::local_skeptic::LocalModelSource::Hub { repo } => {
                                    repo.clone()
                                }
                                hi_agent::local_skeptic::LocalModelSource::Directory { path } => {
                                    path.display().to_string()
                                }
                            },
                            endpoint: Some(runtime.base_url.clone()),
                            ready: true,
                        }),
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
        // Swappable like the sync handles above: a session switch must republish
        // to the new session's runtime, not keep streaming the new session's
        // events into the old session's socket.
        let tui_runtime_publisher = std::sync::Arc::new(std::sync::Mutex::new(
            local_runtime::Publisher::for_session(feedback_session_id.clone()).ok(),
        ));
        let has_runtime_publisher = tui_runtime_publisher.lock().unwrap().is_some();
        let remote_event_tap: Option<hi_tui::RemoteEventTap> =
            (remote_ui.is_some() || has_runtime_publisher).then(|| {
                let state = tui_remote_ui.clone();
                let publisher_slot = tui_runtime_publisher.clone();
                std::sync::Arc::new(move |event: &hi_tui::event::UiEvent| {
                    if let Some(rui) = state.lock().unwrap().as_ref() {
                        rui.push_event(event.clone());
                    }
                    if let Some(publisher) = publisher_slot.lock().unwrap().as_ref() {
                        publisher.publish_best_effort(event.clone());
                    }
                }) as hi_tui::RemoteEventTap
            });
        let tui_sync_flush_callback: Option<hi_tui::RemoteFlushCallback> =
            (sync_handle.is_some() || pipefs_host.is_some()).then(|| {
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
        let tui_sync_session_id = Some(canonical_session_id.clone());
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
                let runtime_publisher = tui_runtime_publisher.clone();
                let pipefs = pipefs_host.clone();
                let switch_sync_config =
                    sync_enabled.then(|| build_sync_config(&settings, &cli, &file));
                let switcher: hi_tui::SessionSwitcher = Box::new(move |id, agent| {
                    let id = id.to_string();
                    let handles = handles.clone();
                    let events = events.clone();
                    let active_session_id = active_session_id.clone();
                    let runtime_publisher = runtime_publisher.clone();
                    let pipefs = pipefs.clone();
                    let switch_sync_config = switch_sync_config.clone();
                    Box::pin(async move {
                        sync::validate_session_id(&id)?;
                        let path = session::session_path(&id)?;
                        if !path.is_file() {
                            let config = switch_sync_config.as_ref().ok_or_else(|| {
                                anyhow!("session '{id}' is unavailable while sync is disabled")
                            })?;
                            let fetched = sync::fetch_session_history(config, &id).await?;
                            if fetched.pipefs.as_ref().is_some_and(|pipefs| {
                                pipefs.enabled || pipefs.restoration_required
                            }) {
                                anyhow::bail!(
                                    "session '{id}' uses PipeFS; resume it with `hi --attach {id} --resume-local` so its workspace is restored before activation"
                                );
                            }
                            session::cache_loaded_session(&path, &fetched.loaded)?;
                        }
                        let loaded = session::load_history(&path)?;
                        let canonical_remote_identity = loaded.remote_session_id.is_some();
                        let canonical_id = loaded
                            .remote_session_id
                            .clone()
                            .unwrap_or_else(|| id.clone());
                        if let Some(pipefs) = &pipefs {
                            pipefs
                                .prepare_session_switch(
                                    &canonical_id,
                                    canonical_remote_identity,
                                )
                                .await?;
                        }
                        let summary = session::resume_summary(&loaded);

                        let previous_handle = handles.lock().unwrap().clone();
                        let previous_events = events.lock().unwrap().clone();
                        let next_sync = if let Some(config) = &switch_sync_config {
                            let remote =
                                sync::RemoteSessionSink::new(config.clone(), canonical_id.clone());
                            remote.seed_snapshot(&loaded)?;
                            // Stage the replacement completely, including the
                            // automatic takeover lease, before touching the
                            // live agent or persistence handles. Sync is
                            // best-effort: an unreachable portal must not
                            // fail the local session switch — and must not
                            // surface as an error either. Registration is
                            // retried by later flushes.
                            let _ = remote.ensure_registered_now_quiet().await;
                            let synced =
                                sync::SyncSession::new(JsonlSession::new(path.clone()), remote);
                            let next_handle = synced.remote_handle();
                            let next_events = std::sync::Arc::new(sync::RemoteUi::new(
                                config.clone(),
                                canonical_id.clone(),
                            )?);
                            Some((synced, next_handle, next_events))
                        } else {
                            None
                        };

                        session::apply_loaded_session(agent, loaded)?;

                        if let Some((synced, next_handle, next_events)) = next_sync {
                            agent.set_session(Box::new(synced));
                            handles.lock().unwrap().replace(next_handle);
                            events.lock().unwrap().replace(next_events);
                        } else {
                            agent.set_session(Box::new(JsonlSession::new(path.clone())));
                            handles.lock().unwrap().take();
                            events.lock().unwrap().take();
                        }
                        if let Some(pipefs) = &pipefs {
                            pipefs.complete_session_switch(canonical_id.clone(), path.clone());
                        }
                        *active_session_id.lock().unwrap() = canonical_id.clone();
                        *runtime_publisher.lock().unwrap() =
                            local_runtime::Publisher::for_session(canonical_id.clone()).ok();

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

                        Ok(completed_session_switch(canonical_id, summary))
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
        let pipefs_command: Option<hi_tui::PipeFsCommand> = pipefs_host.clone().map(|host| {
            let command: hi_tui::PipeFsCommand =
                std::sync::Arc::new(move |argument: String, agent: &mut hi_agent::Agent| {
                    let host = host.clone();
                    Box::pin(async move { host.command(&argument, agent).await })
                });
            command
        });
        let x402_broker = std::sync::Arc::new(hi_ai::X402ConfirmBroker::new());
        if use_tui && !settings.x402.auto_confirm {
            x402::set_confirmer(x402_broker.clone());
        }
        startup_trace!("handing off to TUI");
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
                reasoning_effort_saver: Some(reasoning_effort_saver),
                mlx_switcher,
                local_runtime_switcher,
                startup_local_runtime: startup_local_spec,
                startup_fallback_profile: file.default_profile.clone(),
                session_remember: Some(session_remember),
                resume_summary: resume_summary.clone(),
                mcp_url: settings.mcp_url.clone(),
                api_key: settings.api_key.clone(),
                diff_api_runner: Some(diff_lab::build_tui_api_runner(file.clone())),
                race_runner: Some(race::build_tui_runner(
                    file.clone(),
                    event_sink.clone(),
                    approval_store.clone(),
                )),
                race_defaults: hi_tui::RaceDefaults {
                    targets: if quality.race.enabled {
                        quality.race.targets.clone()
                    } else {
                        Vec::new()
                    },
                    max_candidates: quality.race.max_candidates,
                    max_concurrency: quality.race.max_concurrency,
                    verify_commands: verify_stages
                        .iter()
                        .map(|stage| stage.command.clone())
                        .collect(),
                    fuzz: quality.race.fuzz.clone(),
                    judge_model: std::env::var("HI_JUDGE")
                        .ok()
                        .is_some_and(|value| value.eq_ignore_ascii_case("model"))
                        && verify_stages.is_empty(),
                },
                race_setup_saver: Some(race::build_setup_saver(workspace_root.clone())),
                event_sink,
                approval_store: approval_store.clone(),
                fleet_launcher,
                tui_event_trace,
                remote_event_tap,
                remote_flush_callback: tui_sync_flush_callback,
                sync_config: tui_sync_config,
                sync_session_id: tui_sync_session_id,
                session_lister: Some(session_lister),
                session_switcher,
                session_renamer,
                session_host,
                sync_control: Some(sync_control),
                pipefs_command,
                x402_broker: Some(x402_broker.clone()),
            },
        )
        .await
        {
            Ok(()) => {
                // Back on the main screen: announcements printed here stay
                // visible, so this is where one-shot notices may be shown and
                // marked seen.
                if let Some(pending) = pending_announcements.take() {
                    announcements::show_after_session(pending).await;
                }
                let active_session_id = tui_active_session_id.lock().unwrap().clone();
                feedback::maybe_prompt_and_submit(&settings, &active_session_id).await;
                // Best-effort exit flush: anything unsent stays in the
                // durable outbox, and portal trouble never surfaces as an
                // error in the coding workflow.
                let active_handle = tui_sync_handle.lock().unwrap().clone();
                if let Some(handle) = &active_handle {
                    let _ = handle.flush().await;
                }
                let active_remote_ui = tui_remote_ui.lock().unwrap().clone();
                if let Some(rui) = &active_remote_ui {
                    let _ = rui.flush().await;
                }
                agent.settle_workspace_for_exit().await?;
                if let Some(host) = &pipefs_host {
                    host.clean_exit(&mut agent)
                        .await
                        .context("persisting PipeFS workspace during TUI shutdown")?;
                }
                if let Some(handle) = &active_handle {
                    handle.end_session().await;
                }
                finish_interactive_trace(rsi.observer.as_ref(), &agent)?;
                return Ok(());
            }
            Err(err) => {
                x402::clear_confirmer();
                if cli.tui_events_jsonl.is_some() {
                    agent.settle_workspace_for_exit().await?;
                    return Err(err.context("interactive TUI trace session failed"));
                }
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
    // here, not before TUI launch. The TUI keeps a quiet empty canvas (and a
    // one-line resume summary when continuing); printing a banner first leaves
    // stale text in scrollback and makes a normal exit look like a crash.
    if let Some(summary) = &resume_summary {
        println!("\x1b[2m{summary}\x1b[0m");
    }
    if stdout_is_tty {
        print_landing(&settings, live_metadata.context_window);
    }
    // The plain REPL never leaves the normal screen, so announcements can
    // interleave with the prompt whenever the fetch completes.
    if let Some(pending) = pending_announcements.take() {
        announcements::show_detached(pending);
    }

    let repl_result = repl(
        &mut agent,
        &settings,
        &mut file,
        auto_memory,
        active_profile,
        cli.config.clone(),
        sync_flush_callback,
        approval_store.clone(),
        pipefs_host.clone(),
    )
    .await;
    sync_handle = pipefs_sync_handle.lock().unwrap().clone();
    if repl_result.is_ok() {
        feedback::maybe_prompt_and_submit(&settings, &feedback_session_id).await;
    }
    // Best-effort exit flush: silent by design — see the TUI exit path.
    if let Some(handle) = &sync_handle {
        let _ = handle.flush().await;
    }
    if let Some(rui) = &remote_ui {
        let _ = rui.flush().await;
    }
    agent.settle_workspace_for_exit().await?;
    if let Some(host) = &pipefs_host {
        host.clean_exit(&mut agent)
            .await
            .context("persisting PipeFS workspace during REPL shutdown")?;
    }
    if let Some(handle) = &sync_handle {
        handle.end_session().await;
    }
    finish_interactive_trace(rsi.observer.as_ref(), &agent)?;
    repl_result
}

fn automatic_workflow_plan_path(
    cli: &config::Cli,
    workspace_root: &std::path::Path,
    prompt_input: Option<&str>,
) -> Option<String> {
    let prompt = prompt_input?;
    let moa_prompt = match hi_agent::command::parse(prompt) {
        Some(hi_agent::Command::Moa(argument)) => {
            let argument = argument.trim();
            if argument.is_empty() {
                return None;
            }
            Some(argument.to_owned())
        }
        _ => None,
    };
    hi_agent::one_shot_workflow_plan_path(
        cli.session_file.is_some(),
        workspace_root,
        moa_prompt.as_deref().unwrap_or(prompt),
        cli.goal.as_deref(),
    )
}

/// Describe an active hi-managed local profile without touching the network,
/// filesystem, or local server. The TUI owns provisioning so a broken or
/// moved profile becomes recoverable UI state instead of a pre-TUI exit.
pub(crate) fn prepare_managed_local_startup(
    settings: &config::Settings,
) -> Result<Option<hi_agent::local_skeptic::LocalRuntimeSpec>> {
    let Some(profile) = settings.runtime.as_ref() else {
        return Ok(None);
    };
    if !profile.autostart
        || profile.kind != "mlx"
        || settings.provider != config::ProviderName::Openai
    {
        return Ok(None);
    }
    let backend = match profile.backend.as_deref() {
        None | Some("mlx") => hi_agent::local_skeptic::LocalBackend::Mlx,
        Some(other) => anyhow::bail!("unsupported managed local backend '{other}'"),
    };
    let tool_support = match profile.tool_mode {
        Some(hi_ai::ToolMode::Auto | hi_ai::ToolMode::Required) => {
            hi_agent::local_skeptic::LocalToolSupport::ToolCapable
        }
        Some(hi_ai::ToolMode::ChatOnly | hi_ai::ToolMode::ReadOnly) => {
            hi_agent::local_skeptic::LocalToolSupport::ChatOnly
        }
        None => hi_agent::local_skeptic::LocalToolSupport::Unknown,
    };
    if let Some(path) = &profile.model_path {
        return Ok(Some(
            hi_agent::local_skeptic::local_runtime_spec_from_directory_source(
                expand_local_profile_path(path),
                settings.model.clone(),
                profile.quantization.clone(),
                profile.context_window,
                tool_support,
            ),
        ));
    }
    match hi_agent::local_skeptic::local_runtime_spec(
        &profile.repo,
        hi_agent::local_skeptic::system_ram_gb(),
        backend,
    ) {
        Ok(mut runtime) => {
            runtime.model_id = settings.model.clone();
            runtime.quantization = profile.quantization.clone().or(runtime.quantization);
            runtime.context_window = profile.context_window.or(runtime.context_window);
            runtime.tool_support = tool_support;
            Ok(Some(runtime))
        }
        Err(_) => Ok(Some(hi_agent::local_skeptic::LocalRuntimeSpec {
            repo: profile.repo.clone(),
            model_id: settings.model.clone(),
            backend,
            model_dir: hi_tools::skeptic_model_dir(&profile.repo),
            profile_name: format!(
                "mlx-{}",
                hi_tools::safe_path(&settings.model).to_ascii_lowercase()
            ),
            source: hi_agent::local_skeptic::LocalModelSource::Hub {
                repo: profile.repo.clone(),
            },
            quantization: profile.quantization.clone(),
            context_window: profile.context_window,
            tool_support,
        })),
    }
}

fn expand_local_profile_path(path: &std::path::Path) -> std::path::PathBuf {
    let text = path.to_string_lossy();
    let expanded = if text == "~" || text.starts_with("~/") {
        std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .map(|home| home.join(text.strip_prefix("~/").unwrap_or("")))
            .unwrap_or_else(|| path.to_path_buf())
    } else {
        path.to_path_buf()
    };
    if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&expanded))
            .unwrap_or(expanded)
    }
}

/// Recreate an active hi-managed local profile before the provider is built.
/// The persisted endpoint is intentionally treated as a cache: ports and
/// processes are ephemeral, so an autostart profile always verifies/restarts
/// its own runtime on a fresh process.
pub(crate) async fn ensure_managed_local_startup(
    mut settings: config::Settings,
) -> Result<(
    config::Settings,
    Option<hi_agent::local_skeptic::ManagedLocalRuntime>,
)> {
    let Some(profile) = settings.runtime.clone() else {
        return Ok((settings, None));
    };
    if !profile.autostart
        || profile.kind != "mlx"
        || settings.provider != config::ProviderName::Openai
    {
        return Ok((settings, None));
    }
    let backend = match profile.backend.as_deref() {
        None | Some("mlx") => hi_agent::local_skeptic::LocalBackend::Mlx,
        Some(other) => anyhow::bail!("unsupported managed local backend '{other}'"),
    };
    if hi_agent::local_skeptic::detect_backend_offload().await != Some(backend) {
        anyhow::bail!(
            "managed MLX profile requires Apple Silicon MLX hardware; select another provider or disable autostart"
        );
    }
    let runtime = prepare_managed_local_startup(&settings)?
        .ok_or_else(|| anyhow!("managed local profile is not active"))?;
    eprintln!(
        "\x1b[2mpreparing managed local MLX runtime for {}…\x1b[0m",
        runtime.model_id
    );
    let (_phase_tx, phase_rx) =
        tokio::sync::watch::channel(hi_agent::local_skeptic::LocalRuntimePhase::Resolving);
    let ready = hi_agent::local_skeptic::provision_local_runtime(runtime, _phase_tx).await?;
    let _ = phase_rx;
    settings.base_url = ready.base_url.clone();
    settings.api_key = "local".to_string();
    Ok((settings, Some(ready)))
}

/// Check for updates. Installation stays manual until a signed manifest
/// contract and embedded verification keys ship in this build
/// (`hi_update::install_update` is fail-closed until then); a successful check
/// that finds an update must exit zero, since the printed guidance is the
/// intended outcome.
async fn run_update_command() -> Result<()> {
    let config = hi_update::UpdateConfig::default();
    let status = hi_update::check_for_update(&config).await;
    hi_update::print_update_status(&status);
    if let Some(error) = status.error.as_deref() {
        return Err(anyhow!("update check failed: {error}"));
    }
    Ok(())
}

/// `hi setup` — run the wizard on demand rather than only on the implicit
/// first-run path. Reachable when a config already exists (to re-run a failed
/// sign-in, switch providers, or replace a key); `setup::save_config` is a
/// read-modify-write of the `default` profile, so an existing config survives.
async fn run_setup_command() -> Result<()> {
    if !std::io::stdin().is_terminal() {
        eprintln!("`hi setup` needs an interactive terminal.\n");
        eprintln!("{}", config::ONBOARDING);
        std::process::exit(2);
    }
    let mut file = match config::load_config(None) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(2);
        }
    };
    let settings = setup::run(&mut file).await?;
    println!(
        "Ready: {} · {}",
        settings.model,
        provider_label(settings.provider)
    );
    println!("Run `hi` to start a session.");
    Ok(())
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
