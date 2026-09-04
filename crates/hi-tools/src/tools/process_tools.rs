//! Bash/process tool helpers.

use std::path::Path;
use std::time::Duration;
use std::{fs::File, io::Read};

use anyhow::Result;
use serde::Deserialize;

use crate::{ProcessExecution, ProcessRunner, ToolOutcome};

use super::{RuntimeResources, background_tool_outcome, mark_effect_inspection_failed};

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
/// Interactive-command detection is only a heuristic. Do not let a model
/// point it at a multi-gigabyte generated Python file and make the async tool
/// path read the entire file just to decide whether it looks like a TUI.
const INTERACTIVE_SCAN_MAX_BYTES: u64 = 256 * 1024;

/// Resolve an optional bash deadline. Ordinary commands have no implicit
/// lifetime ceiling: they run until completion, explicit cancellation, or the
/// foreground-to-background handoff below. A positive tool argument wins over
/// `HI_BASH_TIMEOUT_SECS`; omitted, invalid, and zero values mean unlimited.
pub(crate) fn resolve_bash_timeout(requested: Option<u64>) -> Option<Duration> {
    let configured = std::env::var("HI_BASH_TIMEOUT_SECS").ok();
    resolve_bash_timeout_from_values(requested, configured.as_deref())
}

pub(crate) fn resolve_bash_timeout_from_values(
    requested: Option<u64>,
    configured: Option<&str>,
) -> Option<Duration> {
    requested
        .or_else(|| configured.and_then(|value| value.trim().parse().ok()))
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
}

/// Whether a foreground bash command that outlasts its budget is handed to the
/// background (kept running, returns a handle) instead of being killed. This is
/// deliberately opt-in: an ordinary finite build/test must stay attached to the
/// active turn until it completes or the user cancels it. Enable with
/// `HI_BASH_AUTO_BACKGROUND=1` when automatic handoff is desired.
fn auto_background_enabled() -> bool {
    let configured = std::env::var("HI_BASH_AUTO_BACKGROUND").ok();
    auto_background_enabled_from_value(configured.as_deref())
}

pub(super) fn auto_background_enabled_from_value(configured: Option<&str>) -> bool {
    configured.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// The foreground window before an auto-backgrounded command is handed off.
/// Defaults to 30s so a hung build hands control back quickly while the process
/// keeps running in the background. Set `HI_BASH_FOREGROUND_BUDGET_SECS` to
/// override (use the full timeout value to restore the old block-until-done
/// behaviour).
fn resolve_foreground_budget(timeout: Option<Duration>) -> Duration {
    let budget = match std::env::var("HI_BASH_FOREGROUND_BUDGET_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
    {
        Some(secs) => Duration::from_secs(secs),
        None => Duration::from_secs(30),
    };
    timeout.map_or(budget, |deadline| budget.min(deadline))
}

/// Preserve colored subprocess output for the UI while keeping the model and
/// serialized process metadata free of terminal control sequences.
fn process_tool_outcome(execution: ProcessExecution, command: Option<&str>) -> ToolOutcome {
    let model = execution.model_content();
    let display = execution.display_content();
    let mut outcome = ToolOutcome::plain(model);
    if display != outcome.content {
        outcome.display = Some(display);
    }
    outcome.status = execution.status;
    outcome.process = Some(execution.model_outcome());
    outcome.truncation = execution.truncation;
    if command.is_some_and(shell_pipeline_has_ambiguous_status_report) {
        const NOTE: &str = "[shell status note: in a pipeline, `$?` reports the final command's status, not an earlier command. Capture the program status before piping.]";
        if !outcome.content.is_empty() {
            outcome.content.push('\n');
        }
        outcome.content.push_str(NOTE);
        if let Some(display) = &mut outcome.display {
            if !display.is_empty() {
                display.push('\n');
            }
            display.push_str(NOTE);
        }
    }
    outcome
}

fn shell_pipeline_has_ambiguous_status_report(command: &str) -> bool {
    command.contains('|') && (command.contains("$?") || command.contains("${?}"))
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
    /// Optional per-command wall-clock limit in seconds. Omitted or zero means
    /// unlimited unless `HI_BASH_TIMEOUT_SECS` supplies a positive value.
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
    run_bash_tool_with_auto_background(
        root,
        state_root,
        resources,
        args,
        on_line,
        auto_background_enabled(),
    )
    .await
}

/// Policy-injected implementation used by focused tests so they can exercise
/// both branches without mutating the process-global environment.
pub(super) async fn run_bash_tool_with_auto_background(
    root: &Path,
    state_root: &Path,
    resources: RuntimeResources<'_>,
    args: BashArgs,
    on_line: &mut (dyn FnMut(&str) + Send),
    auto_background: bool,
) -> Result<ToolOutcome> {
    if let Some(reason) = crate::guard::blocked_op(&args.command) {
        let mut outcome = ToolOutcome::denied(format!(
            "⚠ refused: this command {reason}. The per-turn checkpoint can't undo it."
        ));
        outcome.effects.mutation_attempted = true;
        return Ok(outcome);
    }
    // Reuse the agent-owned runner whenever one was supplied. Constructing a
    // fresh runner here would consult process-global `HI_SANDBOX` again and
    // can make a command fail in a nested Seatbelt even though the owning
    // agent deliberately selected `SandboxPolicy::Off` (or vice versa).
    let owned_runner = resources
        .process_runner
        .is_none()
        .then(|| ProcessRunner::new(root))
        .transpose()?;
    let runner = resources
        .process_runner
        .or(owned_runner.as_ref())
        .expect("process runner is either borrowed or constructed above");
    // DeepSeek Flash prefers `cat`/`sed -n`/`head` for SPEC.md. Those dumps
    // go through the 5k bash condenser and lose the middle of the spec.
    // Workspace file dumps that fit the read cache become numbered `read`
    // pages (64k budget, paging footer, skip-reread) instead.
    if !args.run_in_background
        && let Some(arguments) = file_dump_read_arguments(&args.command)
        && file_dump_is_read_eligible(root, &arguments)
    {
        return crate::read::run_read(root, resources.read_cache, &arguments).await;
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
        let id = resources
            .background
            .spawn_tracked(runner, &args.command, root, state_root, baseline)
            .await?;
        if let Ok(mut cache) = resources.read_cache.lock() {
            cache.clear();
        }
        let title = crate::background::shell_title(&args.command);
        let mut outcome = background_tool_outcome(
            format!(
                "Started {title} ({id}).\n\
Use bash_output with id {id} to read output; bash_kill with id {id} to stop."
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
    let timeout = resolve_bash_timeout(args.timeout);
    // Read-only inspection is common during orientation and verification
    // (`rg`, `git status`, `sed`, ...). Effect accounting for an opaque shell
    // command normally walks and hashes the whole workspace twice. For a
    // deliberately conservative allowlist, the command itself is the only
    // work needed; the turn-level ledger still performs its normal boundary
    // reconciliation before settlement.
    if definitely_read_only_shell(&args.command) {
        let execution = if auto_background {
            let budget = resolve_foreground_budget(timeout);
            match runner
                .run_shell_adoptable(&args.command, budget, on_line)
                .await
            {
                Ok(crate::AdoptableOutcome::Completed(execution)) => Ok(execution),
                Ok(crate::AdoptableOutcome::StillRunning(running)) => {
                    let foreground_registration = running.foreground_registration;
                    let id = resources
                        .background
                        .adopt_read_only(
                            &args.command,
                            running.child,
                            running.stdout,
                            running.stderr,
                            running.pgid,
                            running.partial_output,
                        )
                        .await?;
                    drop(foreground_registration);
                    let title = crate::background::shell_title(&args.command);
                    let mut outcome = background_tool_outcome(
                        format!(
                            "{title} still running after {}s — continued as {id}.\n\
Use bash_output with id {id} to read output; bash_kill with id {id} to stop.",
                            budget.as_secs(),
                        ),
                        crate::BackgroundOutcome {
                            id,
                            state: crate::BackgroundState::Started,
                            exit_code: None,
                        },
                    );
                    outcome.effects.mutation_attempted = true;
                    if let Ok(mut cache) = resources.read_cache.lock() {
                        cache.clear();
                    }
                    return Ok(outcome);
                }
                Err(error) => Err(error),
            }
        } else {
            runner
                .run_shell_streaming_maybe_timeout(&args.command, timeout, on_line)
                .await
        };
        let mut outcome = match execution {
            Ok(execution) => process_tool_outcome(execution, Some(&args.command)),
            Err(error) => ToolOutcome::failed(format!("Error: process runner failed: {error:#}")),
        };
        // Preserve the bash tool's existing effect/cache contract even though
        // this fast path intentionally omits the expensive workspace scan.
        outcome.effects.mutation_attempted = true;
        if let Ok(mut cache) = resources.read_cache.lock() {
            cache.clear();
        }
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
    // Auto-background-on-timeout: a command still running at its foreground
    // budget is adopted by the background registry (kept alive, handle
    // returned) instead of killed, so no work is lost. Falls back to the
    // classic kill-on-timeout path when disabled.
    if auto_background {
        let budget = resolve_foreground_budget(timeout);
        let outcome = runner
            .run_shell_adoptable(&args.command, budget, on_line)
            .await;
        if let Ok(mut cache) = resources.read_cache.lock() {
            cache.clear();
        }
        match outcome {
            Ok(crate::AdoptableOutcome::Completed(execution)) => {
                let mut outcome = process_tool_outcome(execution, Some(&args.command));
                match crate::effects::workspace_snapshot(root, state_root).await {
                    Ok(after) => outcome.effects = crate::effects::process_effects(&before, &after),
                    Err(error) => mark_effect_inspection_failed(&mut outcome, &error, true),
                }
                return Ok(outcome);
            }
            Ok(crate::AdoptableOutcome::StillRunning(running)) => {
                let foreground_registration = running.foreground_registration;
                let id = resources
                    .background
                    .adopt(
                        &args.command,
                        running.child,
                        running.stdout,
                        running.stderr,
                        running.pgid,
                        running.partial_output,
                        (root.to_path_buf(), state_root.to_path_buf(), before),
                    )
                    .await?;
                drop(foreground_registration);
                let title = crate::background::shell_title(&args.command);
                let mut outcome = background_tool_outcome(
                    format!(
                        "{title} still running after {}s — continued as {id}.\n\
Use bash_output with id {id} to read output; bash_kill with id {id} to stop.",
                        budget.as_secs(),
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
        .run_shell_streaming_maybe_timeout(&args.command, timeout, on_line)
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
        Ok(execution) => process_tool_outcome(execution, Some(&args.command)),
        Err(error) => ToolOutcome::failed(format!("Error: process runner failed: {error:#}")),
    };
    match crate::effects::workspace_snapshot(root, state_root).await {
        Ok(after) => outcome.effects = crate::effects::process_effects(&before, &after),
        Err(error) => mark_effect_inspection_failed(&mut outcome, &error, true),
    }
    Ok(outcome)
}

/// Return true only for shell commands whose common invocation cannot mutate
/// the workspace. This intentionally rejects shell composition, redirection,
/// scripts, and ambiguous subcommands; a false negative costs a snapshot,
/// while a false positive would weaken effect attribution.
fn definitely_read_only_shell(command: &str) -> bool {
    crate::shell_policy::classify_shell_command(command).is_proven_read_only()
}

#[cfg(test)]
pub(crate) fn foreground_interactive_command_reason(command: &str) -> Option<&'static str> {
    let root = std::env::current_dir().ok()?;
    foreground_interactive_command_reason_at(&root, command)
}

pub(crate) fn foreground_interactive_command_reason_at(
    root: &Path,
    command: &str,
) -> Option<&'static str> {
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
    let Ok(metadata) = std::fs::metadata(&path) else {
        return false;
    };
    let Ok(mut file) = File::open(&path) else {
        return false;
    };
    let mut bytes = Vec::new();
    if file
        .by_ref()
        .take(metadata.len().min(INTERACTIVE_SCAN_MAX_BYTES) + 1)
        .read_to_end(&mut bytes)
        .is_err()
    {
        return false;
    }
    let Ok(text) = String::from_utf8(bytes) else {
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
            !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}

/// Map `cat FILE`, `sed -n 'N,Mp' FILE`, and `head -n N FILE` onto `read`
/// arguments. Returns JSON the `read` tool already accepts.
///
/// `printf '---\n' && cat SPEC.md` (a Flash habit) is treated as a dump of
/// the last segment when every prefix is a banner (`echo`/`printf`/`true`).
fn file_dump_read_arguments(command: &str) -> Option<String> {
    parse_file_dump_command(file_dump_command(command)?)
}

fn file_dump_command(command: &str) -> Option<&str> {
    let trimmed = command.trim();
    if trimmed.is_empty()
        || trimmed
            .chars()
            .any(|ch| matches!(ch, '\n' | '\r' | '|' | '>' | '<' | '$' | '`'))
    {
        return None;
    }
    if trimmed.contains('&') && !trimmed.contains("&&") {
        return None;
    }
    let mut segments = Vec::new();
    for chunk in trimmed.split("&&") {
        for piece in chunk.split(';') {
            let piece = piece.trim();
            if !piece.is_empty() {
                segments.push(piece);
            }
        }
    }
    let (last, prefixes) = segments.split_last()?;
    if prefixes
        .iter()
        .copied()
        .any(|prefix| !is_banner_shell(prefix))
    {
        return None;
    }
    Some(*last)
}

fn is_banner_shell(command: &str) -> bool {
    let words: Vec<&str> = command.split_whitespace().collect();
    let Some(start) = words
        .iter()
        .position(|word| !is_env_assignment(word) && *word != "env")
    else {
        return true;
    };
    matches!(basename(words[start]), "echo" | "printf" | "true" | ":")
}

fn parse_file_dump_command(trimmed: &str) -> Option<String> {
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    let start = words
        .iter()
        .position(|word| !is_env_assignment(word) && *word != "env")?;
    let program = basename(words[start]);
    let args: Vec<&str> = words[start + 1..].iter().copied().map(unquote).collect();
    match program {
        "cat" => parse_cat_dump(&args),
        "sed" => parse_sed_print_dump(&args),
        "head" => parse_head_dump(&args),
        _ => None,
    }
}

fn unquote(token: &str) -> &str {
    let bytes = token.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'\'' && *bytes.last().unwrap() == b'\'')
            || (bytes[0] == b'"' && *bytes.last().unwrap() == b'"'))
    {
        &token[1..token.len() - 1]
    } else {
        token
    }
}

fn looks_like_glob(path: &str) -> bool {
    path.contains('*') || path.contains('?') || path.contains('[')
}

fn read_args_json(path: &str, offset: Option<usize>, limit: Option<usize>) -> Option<String> {
    if path.is_empty() || path == "-" || looks_like_glob(path) {
        return None;
    }
    let mut map = serde_json::Map::new();
    map.insert("path".into(), serde_json::Value::String(path.to_string()));
    if let Some(offset) = offset {
        map.insert("offset".into(), serde_json::json!(offset));
    }
    if let Some(limit) = limit {
        map.insert("limit".into(), serde_json::json!(limit));
    }
    Some(serde_json::Value::Object(map).to_string())
}

fn parse_cat_dump(args: &[&str]) -> Option<String> {
    let mut paths = Vec::new();
    for arg in args {
        if *arg == "--" {
            continue;
        }
        if matches!(
            *arg,
            "-n" | "-b" | "-s" | "-u" | "--number" | "--number-nonblank" | "--squeeze-blank"
        ) {
            continue;
        }
        if arg.starts_with('-') {
            return None;
        }
        paths.push(*arg);
    }
    match paths.as_slice() {
        [path] => read_args_json(path, None, None),
        paths if (1..=32).contains(&paths.len()) => {
            Some(serde_json::json!({ "paths": paths }).to_string())
        }
        _ => None,
    }
}

fn parse_sed_print_dump(args: &[&str]) -> Option<String> {
    let mut quiet = false;
    let mut script: Option<&str> = None;
    let mut path: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        if arg == "--" {
            i += 1;
            continue;
        }
        if matches!(arg, "-n" | "--quiet" | "--silent") {
            quiet = true;
            i += 1;
            continue;
        }
        if matches!(arg, "-e" | "--expression") {
            script = Some(args.get(i + 1).copied()?);
            i += 2;
            continue;
        }
        if let Some(expr) = arg.strip_prefix("-e")
            && !expr.is_empty()
        {
            script = Some(expr);
            i += 1;
            continue;
        }
        if arg == "-i" || arg == "--in-place" || arg.starts_with("-i") {
            return None;
        }
        if arg.starts_with('-') {
            return None;
        }
        if script.is_none() {
            script = Some(arg);
        } else if path.is_none() {
            path = Some(arg);
        } else {
            return None;
        }
        i += 1;
    }
    if !quiet {
        return None;
    }
    let (offset, limit) = parse_sed_line_range(script?)?;
    read_args_json(path?, Some(offset), Some(limit))
}

fn parse_sed_line_range(script: &str) -> Option<(usize, usize)> {
    let script = unquote(script).strip_suffix('p')?;
    if let Some((start, end)) = script.split_once(',') {
        let start: usize = start.parse().ok()?;
        let end: usize = end.parse().ok()?;
        if start == 0 || end < start {
            return None;
        }
        Some((start, end.saturating_sub(start).saturating_add(1)))
    } else {
        let line: usize = script.parse().ok()?;
        (line > 0).then_some((line, 1))
    }
}

fn parse_head_dump(args: &[&str]) -> Option<String> {
    let mut limit: Option<usize> = None;
    let mut path: Option<&str> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        if arg == "--" {
            i += 1;
            continue;
        }
        if arg == "-n" || arg == "--lines" {
            limit = Some(parse_line_count(args.get(i + 1).copied()?)?);
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--lines=") {
            limit = Some(parse_line_count(rest)?);
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("-n")
            && !rest.is_empty()
            && rest.bytes().all(|b| b.is_ascii_digit())
        {
            limit = Some(parse_line_count(rest)?);
            i += 1;
            continue;
        }
        if arg.starts_with('-') && arg.len() > 1 && arg[1..].bytes().all(|b| b.is_ascii_digit()) {
            limit = Some(parse_line_count(&arg[1..])?);
            i += 1;
            continue;
        }
        if arg.starts_with('-') {
            return None;
        }
        if path.is_some() {
            return None;
        }
        path = Some(arg);
        i += 1;
    }
    read_args_json(path?, None, Some(limit.unwrap_or(10)))
}

fn parse_line_count(token: &str) -> Option<usize> {
    let n: usize = unquote(token).parse().ok()?;
    (n > 0).then_some(n)
}

fn file_dump_is_read_eligible(root: &Path, arguments: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return false;
    };
    let mut paths = Vec::new();
    if let Some(path) = value.get("path").and_then(|v| v.as_str()) {
        paths.push(path);
    }
    if let Some(list) = value.get("paths").and_then(|v| v.as_array()) {
        for path in list {
            let Some(path) = path.as_str() else {
                return false;
            };
            paths.push(path);
        }
    }
    if paths.is_empty() {
        return false;
    }
    paths
        .iter()
        .all(|path| workspace_file_fits_read(root, path))
}

fn workspace_file_fits_read(root: &Path, rel: &str) -> bool {
    if rel.is_empty() || rel.starts_with('/') || rel.split(['/', '\\']).any(|part| part == "..") {
        return false;
    }
    let path = root.join(rel);
    let Ok(meta) = std::fs::metadata(&path) else {
        return false;
    };
    meta.is_file() && meta.len() <= crate::read::MAX_READ_FILE_BYTES
}

#[cfg(test)]
mod tests {
    use super::{definitely_read_only_shell, file_dump_read_arguments, process_tool_outcome};
    use crate::{ProcessExecution, ProcessOutcome, ToolStatus, TruncationState};

    #[test]
    fn read_only_shell_allowlist_is_conservative() {
        for command in ["rg TODO src", "head -20 README.md", "printf 'done\\n'"] {
            assert!(definitely_read_only_shell(command), "{command:?}");
        }
        for command in [
            "echo hi > marker.txt",
            "sed -i s/old/new/ src/lib.rs",
            "find . -exec rm {} \\;",
            "sort -o sorted.txt input.txt",
            // Even observational Git commands can refresh the index, invoke
            // fsmonitor/textconv/external-diff helpers, or start a pager. Keep
            // them on the live-writer reconciliation path unless a future
            // broker can prove the complete invocation hermetic.
            "git status --short",
            "git -C nested/repo diff",
            "git diff --output=patch.txt",
            "git -C /tmp/repo diff",
            "./scripts/check.sh",
            "cargo test",
        ] {
            assert!(!definitely_read_only_shell(command), "{command:?}");
        }
    }

    #[test]
    fn file_dump_commands_map_to_read_arguments() {
        fn parsed(command: &str) -> Option<serde_json::Value> {
            file_dump_read_arguments(command).map(|json| serde_json::from_str(&json).unwrap())
        }
        assert_eq!(
            parsed("cat SPEC.md"),
            Some(serde_json::json!({"path":"SPEC.md"}))
        );
        assert_eq!(
            parsed("sed -n '200,400p' SPEC.md"),
            Some(serde_json::json!({"path":"SPEC.md","offset":200,"limit":201}))
        );
        assert_eq!(
            parsed("head -n 50 crates/api/src/lib.rs"),
            Some(serde_json::json!({"path":"crates/api/src/lib.rs","limit":50}))
        );
        assert!(parsed("cat file | wc -l").is_none());
        assert!(parsed("sed -i s/a/b/ SPEC.md").is_none());
        assert!(parsed("echo hello").is_none());
        assert!(parsed("cat *.md").is_none());
        assert_eq!(
            parsed("printf -- '---\\n' && cat SPEC.md"),
            Some(serde_json::json!({"path":"SPEC.md"}))
        );
        assert_eq!(
            parsed("echo banner; cat SPEC.md"),
            Some(serde_json::json!({"path":"SPEC.md"}))
        );
        assert!(parsed("cat SPEC.md && rm SPEC.md").is_none());
        assert!(parsed("rm SPEC.md && cat SPEC.md").is_none());
    }

    #[test]
    fn process_tool_outcome_separates_model_and_display_text() {
        let execution = ProcessExecution {
            status: ToolStatus::Succeeded,
            outcome: ProcessOutcome {
                exit_code: Some(0),
                stdout_summary: "\u{1b}[31mred\u{1b}[0m".into(),
                stderr_summary: String::new(),
                duration_ms: 1,
            },
            truncation: TruncationState::Complete,
        };

        let outcome = process_tool_outcome(execution, None);
        assert_eq!(outcome.content, "red");
        assert_eq!(outcome.display.as_deref(), Some("\u{1b}[31mred\u{1b}[0m"));
        assert_eq!(
            outcome.process.unwrap().stdout_summary,
            "red",
            "serialized process metadata stays plain"
        );
    }

    #[test]
    fn pipeline_status_warning_prevents_false_exit_zero_inference() {
        let execution = ProcessExecution {
            status: ToolStatus::Succeeded,
            outcome: ProcessOutcome {
                exit_code: Some(0),
                stdout_summary: "exit=0".into(),
                stderr_summary: String::new(),
                duration_ms: 1,
            },
            truncation: TruncationState::Complete,
        };

        let outcome =
            process_tool_outcome(execution, Some("timeout 10 ./app | head -30; echo exit=$?"));
        assert!(outcome.content.contains("final command's status"));
        assert!(outcome.content.contains("Capture the program status"));
    }
}
