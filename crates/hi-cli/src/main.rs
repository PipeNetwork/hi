#![recursion_limit = "256"]
mod announcements;
mod auth;
mod bootstrap;
mod browser_cmd;
mod config;
mod paths;
mod pipe_session;
mod prompt;
mod review_target;
mod setup;
mod tickets;
mod trace_cmd;
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

pub(crate) use bootstrap::validate_tui_event_trace_request;
use paths::resolve_runtime_roots;
use prompt::effective_prompt;
use review_target::chdir_to_review_target;

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

fn top_level_error_code(_error: &anyhow::Error) -> i32 {
    1
}

fn canonical_session_identity(
    explicit_sync_id: Option<&str>,
    persisted_remote_id: Option<&str>,
    local_path: &std::path::Path,
) -> String {
    explicit_sync_id
        .or(persisted_remote_id)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            local_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("session")
                .to_string()
        })
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

/// Finish authoritative shutdown before polling optional post-session work.
///
/// The futures are accepted separately so constructing a feedback/flush future
/// cannot accidentally move it back ahead of process reaping during frontend
/// refactors. If settlement fails, optional work is deliberately not polled.
async fn settle_before_post_session_work<S, P>(settlement: S, post_session: P) -> Result<()>
where
    S: std::future::Future<Output = Result<()>>,
    P: std::future::Future<Output = ()>,
{
    settlement.await?;
    post_session.await;
    Ok(())
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
        Some("announcements") => return announcements::run_cli(&raw_args[2..]).await,
        Some("trace") => return trace_cmd::run_cli(&raw_args[2..]),
        Some("tickets") => return tickets::run_cli(&raw_args[2..]).await,
        Some(
            "workspace" | "hf" | "mcp" | "doctor" | "diff-lab" | "bench" | "eval" | "team-bench"
            | "metrics" | "intervention" | "tools" | "workflow" | "runtime" | "rsi",
        ) => {
            anyhow::bail!(
                "`hi {}` was removed with the old harness — use `hi` / `hi login pipenetwork`",
                raw_args[1]
            );
        }
        Some("debug") if raw_args.get(2).map(String::as_str) == Some("tui") => {
            anyhow::bail!("`hi debug tui` was removed with the old harness");
        }
        _ => {}
    }
    if raw_args.get(1).map(String::as_str) == Some("update") {
        return run_update_command().await;
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
    if raw_args.get(1).map(String::as_str) == Some("login") {
        return auth::run_login_cli(&raw_args[2..]).await;
    }
    if raw_args.get(1).map(String::as_str) == Some("logout") {
        return auth::run_logout_cli(&raw_args[2..]);
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

    // First run on a real terminal with nothing configured: sign in to Pipe
    // (same pairing as `hi login pipenetwork`) instead of erroring. This also
    // covers `hi "some prompt"` — a one-shot is a natural first command, and
    // answering it with onboarding text instead of login is a dead end.
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
    if settings.execution.is_durable() && cli.no_save && !cli.subagent {
        anyhow::bail!(
            "durable execution requires a persisted session; remove --no-save or disable durable mode"
        );
    }
    // Nothing was configured, but a provider key happened to be exported, so
    // `resolve` inferred everything. Say so once — otherwise the session looks
    // configured, writes nothing, and stops working in the next shell that
    // doesn't export that variable.
    if let Some(env_name) = config::auto_selected_env(&cli, &file) {
        eprintln!(
            "\x1b[2musing {env_name} from the environment ({} · {}) — run `hi login pipenetwork` to save a profile\x1b[0m",
            settings.model,
            settings.provider.as_str(),
        );
    }

    let prompt_input = effective_prompt(&cli)?;
    if !cli.subagent
        && let Some(target) = cli.review_target.as_deref()
    {
        chdir_to_review_target(target)?;
    }
    let (workspace_root, state_root) = resolve_runtime_roots()?;
    startup_trace!("runtime roots resolved");
    if cli.subagent
        || cli.best_of > 1
        || cli.workflow.is_some()
        || cli.skeptic_review
        || cli.loops_daemon
        || cli.daemon
        || cli.attach.is_some()
    {
        anyhow::bail!(
            "eval, subagent, best-of, workflow, skeptic, daemon, and attach were removed with the old harness — use `hi` / `hi login pipenetwork`"
        );
    }
    pipe_session::run(
        &cli,
        &file,
        &settings,
        workspace_root,
        state_root,
        prompt_input,
    )
    .await
}

async fn run_update_command() -> Result<()> {
    let config = hi_update::UpdateConfig::default();
    let status = hi_update::check_for_update(&config).await;
    hi_update::print_update_status(&status);
    if let Some(error) = status.error.as_deref() {
        return Err(anyhow!("update check failed: {error}"));
    }
    Ok(())
}

/// `hi setup` — Pipe sign-in on demand (same as first run / `hi login`).
/// Reachable when a config already exists so a failed pairing can be retried.
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
    println!("Ready: {} · {}", settings.model, settings.provider.as_str());
    println!("Run `hi` to start a session.");
    Ok(())
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
