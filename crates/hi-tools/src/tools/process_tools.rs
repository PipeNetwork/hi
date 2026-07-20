//! Bash/process tool helpers.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use serde::Deserialize;

use crate::{ProcessRunner, ToolOutcome};

use super::{
    RuntimeResources, background_tool_outcome, mark_effect_inspection_failed,
};

/// Default wall-clock limit for a single `bash` command, used when neither the
/// caller nor `HI_BASH_TIMEOUT_SECS` overrides it. Generous enough for a real
/// `cargo test`/build verify step, bounded so a genuine hang recovers on its own.
pub(crate) const DEFAULT_BASH_TIMEOUT_SECS: u64 = 600;
/// Hard ceiling on any per-command timeout (model- or env-supplied) so a bad
/// value can't reintroduce an unbounded stall.
pub(crate) const MAX_BASH_TIMEOUT_SECS: u64 = 3600;
/// Default wall-clock limit for a single verification command (compile/test
/// gate). Overridable via `HI_VERIFY_TIMEOUT_SECS`; sized to fit a real
/// `cargo test`/build on a mid-size project rather than a toy check.
const PYTHON_TUI_MARKERS: &[&str] = &[
    "from textual",
    "import textual",
    "App().run(",
    "import curses",
    "from curses",
    "curses.wrapper(",
    "import urwid",
    "from urwid",
    "prompt_toolkit",
    "blessed.Terminal",
    "asciimatics",
    "npyscreen",
];
const RUST_TUI_MARKERS: &[&str] = &[
    "ratatui",
    "crossterm",
    "tui =",
    "cursive",
    "termion",
    "termwiz",
];

/// Resolve the effective bash timeout: an explicit per-command request wins,
/// else `HI_BASH_TIMEOUT_SECS`, else the default — always clamped to
/// `[1, MAX_BASH_TIMEOUT_SECS]` so neither side can disable the guard.
pub(crate) fn resolve_bash_timeout(requested: Option<u64>) -> Duration {
    let secs = requested
        .or_else(|| {
            std::env::var("HI_BASH_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.trim().parse().ok())
        })
        .unwrap_or(DEFAULT_BASH_TIMEOUT_SECS)
        .clamp(1, MAX_BASH_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Whether a foreground bash command that outlasts its budget is handed to the
/// background (kept running, returns a handle) instead of being killed. On by
/// default — losing a slow build or a mistakenly-foregrounded server to a kill
/// is exactly the babysitting the agent must avoid; a backgrounded process is
/// fully recoverable (poll or kill it). Disable with `HI_BASH_AUTO_BACKGROUND=0`.
fn auto_background_enabled() -> bool {
    !matches!(
        std::env::var("HI_BASH_AUTO_BACKGROUND")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("0") | Some("false") | Some("off")
    )
}

/// The foreground window before an auto-backgrounded command is handed off.
/// Defaults to the command's full timeout (so blocking time is unchanged from
/// the kill-on-timeout behaviour — only the *outcome* changes from kill to
/// background). Set `HI_BASH_FOREGROUND_BUDGET_SECS` for the snappier
/// "hand control back fast, keep working" behaviour.
fn resolve_foreground_budget(timeout: Duration) -> Duration {
    match std::env::var("HI_BASH_FOREGROUND_BUDGET_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        Some(secs) => Duration::from_secs(secs.clamp(1, MAX_BASH_TIMEOUT_SECS)).min(timeout),
        None => timeout,
    }
}

/// Run a verification process at an explicit root and retain its typed status,
/// separate stdout/stderr summaries, duration, and truncation state.
/// SIGKILL an entire process group by its id. We spawn with `process_group(0)`,
/// so a process's group id equals its pid; signalling the negative pid reaches
/// every descendant. No-op on non-Unix (where `child.kill()` is the best we have).
#[cfg(unix)]
pub(crate) fn kill_group(pgid: i32) {
    crate::process::kill_group(pgid);
}

#[cfg(not(unix))]
pub(crate) fn kill_group(_pgid: i32) {}

#[cfg(test)]
pub(crate) async fn run_bash_streaming_with_timeout(
    command: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
    bash_timeout: Duration,
) -> Result<String> {
    // Refuse operations a checkpoint can't undo or safely contain.
    if let Some(reason) = crate::guard::blocked_op(command) {
        return Ok(format!(
            "⚠ refused: this command {reason}. The per-turn checkpoint can't undo it. \
             If it's genuinely needed, ask the user to run it themselves (or set the \
             documented override env var for this guard)."
        ));
    }
    let runner = ProcessRunner::from_current_dir()?;
    let execution = runner
        .run_shell_streaming(command, bash_timeout, on_line)
        .await?;
    Ok(execution.model_content())
}

#[derive(Deserialize)]
pub(super) struct BashArgs {
    pub command: String,
    /// Optional per-command wall-clock limit in seconds. Omitted → the default
    /// (or `HI_BASH_TIMEOUT_SECS`). Clamped to `[1, MAX_BASH_TIMEOUT_SECS]`.
    /// Ignored when `run_in_background` is set.
    #[serde(default)]
    pub timeout: Option<u64>,
    /// Run detached: return a handle immediately instead of waiting for exit.
    /// Poll it with `bash_output` and stop it with `bash_kill`.
    #[serde(default)]
    pub run_in_background: bool,
}

/// Shared `bash` dispatch for both the streaming and non-streaming entry points.
/// `run_in_background` short-circuits to a detached process and returns its
/// handle; otherwise it runs to completion (streaming output through `on_line`).
pub(super) async fn run_bash_tool(
    root: &Path,
    state_root: &Path,
    resources: RuntimeResources<'_>,
    args: BashArgs,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<ToolOutcome> {
    if let Some(reason) = crate::guard::blocked_op(&args.command) {
        let mut outcome = ToolOutcome::denied(format!(
            "⚠ refused: this command {reason}. The per-turn checkpoint can't undo it."
        ));
        outcome.effects.mutation_attempted = true;
        return Ok(outcome);
    }
    if args.run_in_background {
        let baseline = match crate::effects::workspace_snapshot(root, state_root).await {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let mut outcome = ToolOutcome::failed("Background process was not started.".into());
                mark_effect_inspection_failed(&mut outcome, &error, false);
                return Ok(outcome);
            }
        };
        let runner = ProcessRunner::new(root)?;
        let id = resources.background.spawn_tracked(
            &runner,
            &args.command,
            root,
            state_root,
            baseline,
        )?;
        if let Ok(mut cache) = resources.read_cache.lock() {
            cache.clear();
        }
        let mut outcome = background_tool_outcome(
            format!(
                "Started background process `{id}`: {}\n\
             Check its output with bash_output {{\"id\":\"{id}\"}} and stop it with \
             bash_kill {{\"id\":\"{id}\"}}.",
                args.command
            ),
            crate::BackgroundOutcome {
                id,
                state: crate::BackgroundState::Started,
                exit_code: None,
            },
        );
        outcome.effects.mutation_attempted = true;
        return Ok(outcome);
    }
    if let Some(reason) = foreground_interactive_command_reason_at(root, &args.command) {
        let mut outcome = ToolOutcome::denied(format!(
            "⚠ refused: this command {reason}. Foreground interactive terminal apps can block \
             the agent turn. For a smoke test, wrap it with `timeout 5s ... >/tmp/hi-app.out \
             2>&1` and inspect the captured output, or use import/unit tests for validation. \
             Use run_in_background:true only for long-lived servers or watchers."
        ));
        outcome.effects.mutation_attempted = true;
        return Ok(outcome);
    }
    let before = match crate::effects::workspace_snapshot(root, state_root).await {
        Ok(snapshot) => snapshot,
        Err(error) => {
            let mut outcome = ToolOutcome::failed("Process was not started.".into());
            mark_effect_inspection_failed(&mut outcome, &error, false);
            return Ok(outcome);
        }
    };
    let timeout = resolve_bash_timeout(args.timeout);
    let runner = ProcessRunner::new(root)?;

    // Auto-background-on-timeout: a command still running at its foreground
    // budget is adopted by the background registry (kept alive, handle
    // returned) instead of killed, so no work is lost. Falls back to the
    // classic kill-on-timeout path when disabled.
    if auto_background_enabled() {
        let budget = resolve_foreground_budget(timeout);
        let outcome = runner
            .run_shell_adoptable(&args.command, budget, on_line)
            .await;
        if let Ok(mut cache) = resources.read_cache.lock() {
            cache.clear();
        }
        match outcome {
            Ok(crate::AdoptableOutcome::Completed(execution)) => {
                let mut outcome = ToolOutcome::plain(execution.model_content());
                outcome.status = execution.status;
                outcome.process = Some(execution.outcome);
                outcome.truncation = execution.truncation;
                match crate::effects::workspace_snapshot(root, state_root).await {
                    Ok(after) => outcome.effects = crate::effects::process_effects(&before, &after),
                    Err(error) => mark_effect_inspection_failed(&mut outcome, &error, true),
                }
                return Ok(outcome);
            }
            Ok(crate::AdoptableOutcome::StillRunning(running)) => {
                let id = resources.background.adopt(
                    &args.command,
                    running.child,
                    running.stdout,
                    running.stderr,
                    running.pgid,
                    running.partial_output,
                    (root.to_path_buf(), state_root.to_path_buf(), before),
                );
                let mut outcome = background_tool_outcome(
                    format!(
                        "Command still running after {}s — moved to background as `{id}` (not \
                         killed): {}\n\
                         Read its output with bash_output {{\"id\":\"{id}\"}} and stop it with \
                         bash_kill {{\"id\":\"{id}\"}}.",
                        budget.as_secs(),
                        args.command
                    ),
                    crate::BackgroundOutcome {
                        id,
                        state: crate::BackgroundState::Started,
                        exit_code: None,
                    },
                );
                outcome.effects.mutation_attempted = true;
                return Ok(outcome);
            }
            Err(error) => {
                return Ok(ToolOutcome::failed(format!(
                    "Error: process runner failed: {error:#}"
                )));
            }
        }
    }

    let execution = runner
        .run_shell_streaming(&args.command, timeout, on_line)
        .await;
    // A shell command can mutate any file (sed -i, codegen, git checkout, mv, a
    // formatter, …); a later `read` in the same turn must not serve stale cached
    // content. We don't know which paths it touched — clear the whole read cache.
    // Done HERE (not only in the dispatch arm) so the *streaming* path — the one
    // the live turn loop actually uses (execute_streaming) — invalidates it too.
    if let Ok(mut cache) = resources.read_cache.lock() {
        cache.clear();
    }
    let mut outcome = match execution {
        Ok(execution) => {
            let mut outcome = ToolOutcome::plain(execution.model_content());
            outcome.status = execution.status;
            outcome.process = Some(execution.outcome);
            outcome.truncation = execution.truncation;
            outcome
        }
        Err(error) => ToolOutcome::failed(format!("Error: process runner failed: {error:#}")),
    };
    match crate::effects::workspace_snapshot(root, state_root).await {
        Ok(after) => outcome.effects = crate::effects::process_effects(&before, &after),
        Err(error) => mark_effect_inspection_failed(&mut outcome, &error, true),
    }
    Ok(outcome)
}

#[cfg(test)]
pub(crate) fn foreground_interactive_command_reason(command: &str) -> Option<&'static str> {
    let root = std::env::current_dir().ok()?;
    foreground_interactive_command_reason_at(&root, command)
}

pub(crate) fn foreground_interactive_command_reason_at(root: &Path, command: &str) -> Option<&'static str> {
    if std::env::var_os("HI_ALLOW_INTERACTIVE_BASH").is_some()
        || command_has_timeout_wrapper(command)
    {
        return None;
    }
    let tokens = first_command_tokens(command);
    let (program_idx, program) = first_program_token(&tokens)?;
    let program = basename(program);
    if program == "textual" {
        return Some("appears to launch a Textual terminal UI in the foreground");
    }
    if is_python_program(program) {
        if python_inline_code_looks_interactive(&tokens[program_idx + 1..]) {
            return Some("appears to launch a Python terminal UI in the foreground");
        }
        if let Some(script) = python_script_arg(&tokens[program_idx + 1..])
            && python_script_looks_interactive(root, &script)
        {
            return Some("appears to launch a Python terminal UI in the foreground");
        }
    }
    if program == "cargo" && cargo_run_looks_like_rust_tui(root, &tokens[program_idx + 1..]) {
        return Some("appears to launch a Rust terminal UI in the foreground");
    }
    None
}

fn command_has_timeout_wrapper(command: &str) -> bool {
    let tokens = first_command_tokens(command);
    let Some((_, program)) = first_program_token(&tokens) else {
        return false;
    };
    matches!(basename(program), "timeout" | "gtimeout")
}

fn first_command_tokens(command: &str) -> Vec<String> {
    command
        .split([';', '\n', '|', '&'])
        .next()
        .unwrap_or(command)
        .split_whitespace()
        .map(|s| s.trim_matches(['"', '\'']).to_string())
        .collect()
}

fn first_program_token(tokens: &[String]) -> Option<(usize, &str)> {
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "env" || is_env_assignment(tok) {
            i += 1;
            continue;
        }
        return Some((i, tok));
    }
    None
}

fn python_script_arg(tokens: &[String]) -> Option<String> {
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        match tok {
            "-m" | "-c" => return None,
            "-u" | "-B" | "-q" | "-I" | "-s" | "-S" | "-E" => {
                i += 1;
                continue;
            }
            _ if tok.starts_with("-X") || tok.starts_with("-W") => {
                i += 1;
                continue;
            }
            _ if tok.starts_with('-') => {
                i += 1;
                continue;
            }
            _ => return Some(tok.to_string()),
        }
    }
    None
}

fn python_inline_code_looks_interactive(tokens: &[String]) -> bool {
    let Some(pos) = tokens.iter().position(|tok| tok == "-c") else {
        return false;
    };
    let Some(code) = tokens.get(pos + 1) else {
        return false;
    };
    text_looks_like_python_tui(code)
}

fn python_script_looks_interactive(root: &Path, path: &str) -> bool {
    if !path.ends_with(".py") {
        return false;
    }
    let path = Path::new(path);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    text_looks_like_python_tui(&text)
}

fn cargo_run_looks_like_rust_tui(root: &Path, tokens: &[String]) -> bool {
    if !tokens.iter().any(|token| token == "run") {
        return false;
    }
    if tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "--help" | "-h" | "--version" | "-V" | "help"
        )
    }) {
        return false;
    }
    rust_workspace_looks_like_tui(root)
}

fn rust_workspace_looks_like_tui(root: &Path) -> bool {
    let Ok(manifest) = std::fs::read_to_string(root.join("Cargo.toml")) else {
        return false;
    };
    let lower = manifest.to_ascii_lowercase();
    RUST_TUI_MARKERS.iter().any(|marker| lower.contains(marker))
}

fn text_looks_like_python_tui(text: &str) -> bool {
    PYTHON_TUI_MARKERS
        .iter()
        .any(|marker| text.contains(marker))
}

fn is_python_program(base: &str) -> bool {
    base == "python"
        || base == "python3"
        || base
            .strip_prefix("python3.")
            .is_some_and(|tail| tail.chars().all(|c| c.is_ascii_digit()))
}

fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

fn is_env_assignment(tok: &str) -> bool {
    !tok.starts_with('-')
        && tok.split_once('=').is_some_and(|(k, _)| {
            !k.is_empty() && k.chars().all(|c| c.is_alphanumeric() || c == '_')
        })
}

