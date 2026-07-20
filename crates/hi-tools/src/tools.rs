use std::io::Read;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tokio::process::Command;


use crate::condense::condense;
use crate::edit::{apply_edit, plan_multi_patch};
use crate::paths::cache_key;
use crate::read::{run_glob, run_grep, run_list, run_read};
use crate::transaction::{MutationPlan, PlannedFileMutation};
use crate::{PlanStatus, PlanStep, ProcessRunner, ToolEffects, ToolOutcome};

/// A completely parsed and materialized file-tool invocation.
///
/// The contained [`MutationPlan`] owns the exact postimages shown by
/// [`PreparedMutation::preview`] and the preimage digests that must still match
/// when it is committed. Consuming this value is therefore the only supported
/// way to execute an edit after an interactive confirmation: the tool call is
/// never reparsed or rebuilt after approval.
#[derive(Debug)]
pub struct PreparedMutation {
    plan: MutationPlan,
    kind: PreparedMutationKind,
}

#[derive(Debug)]
enum PreparedMutationKind {
    Write {
        target: std::path::PathBuf,
        path: String,
        after: String,
    },
    Edit {
        target: std::path::PathBuf,
        path: String,
        after: String,
        replacements: usize,
        replace_all: bool,
    },
    MultiEdit {
        target: std::path::PathBuf,
        path: String,
        after: String,
        edit_count: usize,
    },
    ApplyPatch {
        summary: String,
    },
}

impl PreparedMutation {
    /// Render the exact postimages held by this prepared plan.
    pub fn preview(&self) -> String {
        self.plan.preview()
    }
}

/// Default wall-clock limit for a single `bash` command, used when neither the
/// caller nor `HI_BASH_TIMEOUT_SECS` overrides it. Generous enough for a real
/// `cargo test`/build verify step, bounded so a genuine hang recovers on its own.
const DEFAULT_BASH_TIMEOUT_SECS: u64 = 600;
/// Hard ceiling on any per-command timeout (model- or env-supplied) so a bad
/// value can't reintroduce an unbounded stall.
const MAX_BASH_TIMEOUT_SECS: u64 = 3600;
/// Default wall-clock limit for a single verification command (compile/test
/// gate). Overridable via `HI_VERIFY_TIMEOUT_SECS`; sized to fit a real
/// `cargo test`/build on a mid-size project rather than a toy check.
const DEFAULT_CHECK_TIMEOUT_SECS: u64 = 600;

/// The effective verification timeout: `HI_VERIFY_TIMEOUT_SECS` if set to a
/// positive integer, else [`DEFAULT_CHECK_TIMEOUT_SECS`].
fn check_timeout() -> Duration {
    let secs = std::env::var("HI_VERIFY_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .unwrap_or(DEFAULT_CHECK_TIMEOUT_SECS);
    Duration::from_secs(secs)
}
const MAX_UNTRACKED_DIFF_ENTRIES: usize = 200;
const MAX_CREATED_DIFF_FILE_BYTES: usize = 16 * 1024;
const MAX_CREATED_DIFF_TOTAL_BYTES: usize = 64 * 1024;
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
fn resolve_bash_timeout(requested: Option<u64>) -> Duration {
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
pub async fn run_check_in(
    root: &std::path::Path,
    command: &str,
) -> Result<crate::ProcessExecution> {
    prepare_verify_workdir(root);
    ProcessRunner::new(root)?
        .run_shell(command, check_timeout())
        .await
}

/// Best-effort cleanup before running a verification command.
///
/// Python's import cache can otherwise make same-size, same-second edits look
/// unchanged to `python -c "import solution"` checks. Pruning only `__pycache__`
/// directories keeps this narrow and harmless for non-Python checks.
pub fn prepare_verify_workdir(dir: &std::path::Path) {
    fn walk(dir: &std::path::Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_name() == "__pycache__" {
                let _ = std::fs::remove_dir_all(&path);
            } else if path.is_dir() {
                walk(&path);
            }
        }
    }
    walk(dir);
}

/// A per-file "fast check" command for `path`'s language — a quick, file-scoped
/// syntax/lint check that can run in the background right after an edit, so a
/// type/syntax error surfaces while the edit is still the model's focus rather
/// than at turn-end verify. Returns `None` for languages without a genuinely
/// per-file fast check (e.g. Rust, whose `cargo check` is project-wide and is
/// already the turn-end verify) or for unrecognized extensions. The command is
/// run as an argument-vector process with the file path appended. Launch and
/// check failures are non-fatal — no early signal is better than a wrong one.
pub fn fast_check_for(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?;
    match ext {
        // Python: py_compile catches syntax errors per-file, fast.
        "py" => Some("python3 -m py_compile"),
        // Go: gofmt -l lists files that aren't formatted / have syntax issues.
        "go" => Some("gofmt -l"),
        // TypeScript/JavaScript: tsc --noEmit is project-wide but fast enough
        // and the best signal available; only useful when a tsconfig is present
        // (the caller running it is fine even without — it just no-ops).
        "ts" | "tsx" | "js" | "jsx" => Some("npx --no-install tsc --noEmit"),
        // Ruby: `ruby -c` is a fast per-file syntax check.
        "rb" => Some("ruby -c"),
        // Shell: `shellcheck` catches syntax errors and common pitfalls
        // per-file. Widely available; no-ops gracefully if absent. `--shell=bash`
        // is required because the caller appends the file path as the next arg,
        // and a bare `--shell` would consume that path as the shell name.
        "sh" | "bash" => Some("shellcheck --shell=bash"),
        // Lua: `luac -p` is a fast per-file syntax check.
        "lua" => Some("luac -p"),
        // Perl: `perl -c` is a fast per-file syntax check.
        "pl" | "pm" | "t" => Some("perl -c"),
        // PHP: `php -l` is a fast per-file syntax check
        // (`-l` = lint, not list; available since PHP 5).
        "php" => Some("php -l"),
        // Rust, C/C++, and others: no reliable per-file fast check — rely on
        // the turn-end verify (e.g. `cargo check`).
        _ => None,
    }
}

/// Run one of [`fast_check_for`]'s checks without interpolating `path` into a
/// shell command. The boolean is authoritative; the text is bounded diagnostic
/// context for the model/UI.
pub async fn run_fast_check_in(root: &Path, check: &str, path: &Path) -> (bool, String) {
    use std::ffi::OsString;

    let path_arg = path.as_os_str().to_os_string();
    let (program, args): (&str, Vec<OsString>) = match check {
        "python3 -m py_compile" => (
            "python3",
            vec![OsString::from("-m"), OsString::from("py_compile"), path_arg],
        ),
        "gofmt -l" => ("gofmt", vec![OsString::from("-l"), path_arg]),
        "npx --no-install tsc --noEmit" => (
            "npx",
            vec![
                OsString::from("--no-install"),
                OsString::from("tsc"),
                OsString::from("--noEmit"),
            ],
        ),
        "ruby -c" => ("ruby", vec![OsString::from("-c"), path_arg]),
        "shellcheck --shell=bash" => ("shellcheck", vec![OsString::from("--shell=bash"), path_arg]),
        "luac -p" => ("luac", vec![OsString::from("-p"), path_arg]),
        "perl -c" => ("perl", vec![OsString::from("-c"), path_arg]),
        "php -l" => ("php", vec![OsString::from("-l"), path_arg]),
        _ => return (false, format!("unsupported fast check: {check}")),
    };
    let runner = match ProcessRunner::new(root) {
        Ok(runner) => runner,
        Err(error) => return (false, format!("fast-check runner failed: {error:#}")),
    };
    match runner
        .run_program(program, &args, Duration::from_secs(60))
        .await
    {
        Ok(execution) => (
            execution.status == crate::ToolStatus::Succeeded,
            execution.model_content(),
        ),
        Err(error) => (false, format!("fast check failed to start: {error:#}")),
    }
}

/// A human-readable, ANSI-colored summary of what's changed in the working
/// tree versus the last commit — the body of the `/diff` command. Tracked
/// changes come from `git diff HEAD`; bounded text content is included for new
/// files, while binary/generated/vendor/oversized files are summarized. Returns
/// a friendly message when the workspace isn't a git repo or there's nothing
/// to show.
pub async fn working_tree_diff_in(root: &Path) -> String {
    working_tree_diff_impl(root, true).await
}

/// Same as [`working_tree_diff_in`] but without ANSI color codes — for the `diff`
/// tool, so the model gets plain text it can parse.
pub async fn working_tree_diff_plain_in(root: &Path) -> String {
    working_tree_diff_impl(root, false).await
}

async fn working_tree_diff_impl(root: &Path, color: bool) -> String {
    let git = |args: &'static [&'static str]| async move {
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(root);
        if color {
            cmd.arg("-c").arg("color.ui=always");
        }
        cmd.args(args);
        cmd.output().await
    };

    let tracked = match git(&["--no-pager", "diff", "HEAD"]).await {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            // Fresh repo with no commits yet: diff against the empty tree instead.
            if stderr.contains("unknown revision") || stderr.contains("ambiguous argument") {
                git(&["--no-pager", "diff"])
                    .await
                    .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                    .unwrap_or_default()
            } else if git_diff_failed_not_repo(&stderr) {
                return "not a git repository; no git diff available".to_string();
            } else {
                return format!(
                    "not a git repository (or git unavailable): {}",
                    stderr.trim()
                );
            }
        }
        Err(err) => return format!("git not available: {err}"),
    };

    let untracked = git(&["ls-files", "--others", "--exclude-standard", "-z"])
        .await
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let new_files: Vec<&str> = untracked
        .split('\0')
        .filter(|path| !path.is_empty())
        .collect();

    if tracked.trim().is_empty() && new_files.is_empty() {
        return "no changes since HEAD".to_string();
    }

    let mut out = String::new();
    if !tracked.trim().is_empty() {
        out.push_str(tracked.trim_end());
        out.push('\n');
    }
    if !new_files.is_empty() {
        out.push_str("\nnew (untracked) files and bounded contents:\n");
        out.push_str(&render_untracked_files_with_contents(
            root,
            &new_files,
            MAX_UNTRACKED_DIFF_ENTRIES,
        ));
    }
    out
}

fn render_untracked_files_with_contents(root: &Path, files: &[&str], limit: usize) -> String {
    let mut out = String::new();
    let mut retained = 0usize;
    let mut summarized = Vec::new();
    let mut shown = 0usize;

    for path in files {
        if shown >= limit || retained >= MAX_CREATED_DIFF_TOTAL_BYTES {
            break;
        }
        if summarize_created_path(path) {
            summarized.push(*path);
            shown += 1;
            continue;
        }
        let absolute = root.join(path);
        let Ok(metadata) = std::fs::symlink_metadata(&absolute) else {
            summarized.push(*path);
            shown += 1;
            continue;
        };
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            summarized.push(*path);
            shown += 1;
            continue;
        }
        let Ok(file) = std::fs::File::open(&absolute) else {
            summarized.push(*path);
            shown += 1;
            continue;
        };
        let mut bytes = Vec::new();
        if file
            .take((MAX_CREATED_DIFF_FILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .is_err()
            || bytes.len() > MAX_CREATED_DIFF_FILE_BYTES
            || bytes.contains(&0)
        {
            summarized.push(*path);
            shown += 1;
            continue;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            summarized.push(*path);
            shown += 1;
            continue;
        };
        let mut patch = format!("--- /dev/null\n+++ b/{path}\n");
        for line in text.split_inclusive('\n') {
            patch.push('+');
            patch.push_str(line);
        }
        if !text.is_empty() && !text.ends_with('\n') {
            patch.push('\n');
            patch.push_str("\\ No newline at end of file\n");
        }
        if retained.saturating_add(patch.len()) > MAX_CREATED_DIFF_TOTAL_BYTES {
            summarized.push(*path);
        } else {
            retained += patch.len();
            out.push_str(&patch);
        }
        shown += 1;
    }

    if !summarized.is_empty() {
        out.push_str("summarized binary/generated/vendor/oversized files:\n");
        out.push_str(&render_untracked_files(&summarized, limit));
    }
    if files.len() > shown {
        out.push_str(&format!(
            "  ... omitted {} untracked entr{} (entry/content limit)\n",
            files.len() - shown,
            if files.len() - shown == 1 { "y" } else { "ies" }
        ));
    }
    out
}

fn summarize_created_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    normalized.split('/').any(|part| {
        matches!(
            part,
            "vendor" | "node_modules" | "target" | "dist" | "build" | "coverage" | "generated"
        )
    })
}

fn git_diff_failed_not_repo(stderr: &str) -> bool {
    let lower = stderr.to_ascii_lowercase();
    lower.contains("not a git repository")
        || lower.contains("not a git repo")
        || lower.contains("outside repository")
        || lower.contains("outside a work tree")
        || lower.contains("usage: git diff")
}

fn render_untracked_files(files: &[&str], limit: usize) -> String {
    let mut collapsed = std::collections::BTreeMap::<String, usize>::new();
    for file in files {
        let path = file.trim();
        if path.is_empty() {
            continue;
        }
        *collapsed.entry(collapse_untracked_path(path)).or_default() += 1;
    }

    let total = collapsed.len();
    let mut out = String::new();
    for (path, count) in collapsed.into_iter().take(limit) {
        if count > 1 && path.ends_with('/') {
            out.push_str(&format!("  + {path} ({count} entries)\n"));
        } else {
            out.push_str(&format!("  + {path}\n"));
        }
    }
    if total > limit {
        let omitted = total - limit;
        out.push_str(&format!(
            "  ... omitted {omitted} untracked entr{} (limit {limit})\n",
            if omitted == 1 { "y" } else { "ies" }
        ));
    }
    out
}

fn collapse_untracked_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let mut components = normalized.split('/').filter(|part| !part.is_empty());
    match (components.next(), components.next()) {
        (Some(first), Some(_)) => format!("{first}/"),
        (Some(first), None) => first.to_string(),
        _ => path.to_string(),
    }
}

/// Stage all working-tree changes and commit them with an auto-generated
/// message summarizing the changed files. This is the body of the `/commit`
/// slash command. Returns a single human-readable progress line (the message
/// used), or an error message when there's nothing to commit or this isn't a
/// git repo. Phase order: `git add -A` → read the staged diff stat →
/// `git commit -m "<message>"`.
pub async fn commit_in(root: &Path) -> String {
    // 1. Confirm we're inside a work tree before touching anything.
    let in_tree = match Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--is-inside-work-tree"])
        .output()
        .await
    {
        Ok(o) => o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "true",
        Err(err) => return format!("git not available: {err}"),
    };
    if !in_tree {
        return "not a git repository".to_string();
    }

    // 2. Stage all changes (tracked modifications, deletions, untracked adds).
    let add = match Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["add", "-A"])
        .output()
        .await
    {
        Ok(o) => o,
        Err(err) => return format!("git add failed: {err}"),
    };
    if !add.status.success() {
        let stderr = String::from_utf8_lossy(&add.stderr);
        return format!("git add failed: {}", stderr.trim());
    }

    // 3. Summarize the staged changes for the commit message. We list the
    //    changed file names and count them for the "N files" phrasing.
    let stat = match Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["--no-pager", "diff", "--cached", "--name-only"])
        .output()
        .await
    {
        Ok(o) => String::from_utf8_lossy(&o.stdout).into_owned(),
        Err(err) => return format!("git diff failed: {err}"),
    };
    let files: Vec<&str> = stat.lines().filter(|l| !l.trim().is_empty()).collect();
    if files.is_empty() {
        return "nothing to commit (working tree clean)".to_string();
    }

    // Build a message from the file list: keep it to a single subject line plus
    // a short body. Conventional-ish subject ("update N files") so it reads well
    // in `git log` without depending on a model call.
    let n = files.len();
    let subject = if n == 1 {
        format!("update {}", files[0])
    } else {
        format!("update {n} files")
    };
    // Body: list the files, trimmed to a reasonable cap so huge staging sets
    // don't produce a multi-thousand-line commit message.
    const MAX_FILES_IN_BODY: usize = 40;
    let mut body = String::new();
    for f in files.iter().take(MAX_FILES_IN_BODY) {
        body.push_str("  - ");
        body.push_str(f);
        body.push('\n');
    }
    if n > MAX_FILES_IN_BODY {
        body.push_str(&format!("  - … and {} more\n", n - MAX_FILES_IN_BODY));
    }
    let message = if body.trim().is_empty() {
        subject.clone()
    } else {
        format!("{subject}\n\n{body}", body = body.trim_end())
    };

    // 4. Commit. We pass the message via `-m`; embedded newlines cover subject
    //    + body in a single argument.
    let commit = match Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["commit", "-m", &message])
        .output()
        .await
    {
        Ok(o) => o,
        Err(err) => return format!("git commit failed: {err}"),
    };
    if !commit.status.success() {
        let stderr = String::from_utf8_lossy(&commit.stderr);
        let stdout = String::from_utf8_lossy(&commit.stdout);
        let detail = if stderr.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        };
        return format!("git commit failed: {detail}");
    }

    // Echo the summary the way the UI expects: `── … ──`.
    format!(
        "staged {n} file{}\ncommitted: \"{subject}\"",
        if n == 1 { "" } else { "s" }
    )
}

pub use crate::catalog::{
    MINIMAL_TOOL_SPECS, TOOL_CATALOG, TOOL_SPECS, ToolCapability, ToolMetadata,
    delegate_tool_spec, explore_tool_spec, is_coordination, is_filesystem_mutating, is_known_tool,
    is_read_only, target_path, tool_metadata,
};

/// Execute a tool by name. Tool failures are returned as content (not errors)
/// so the model sees them and can recover, rather than aborting the turn.
#[derive(Clone, Copy)]
struct RuntimeResources<'a> {
    lsp: &'a std::sync::Arc<hi_lsp::LspManager>,
    background: &'a crate::BackgroundRegistry,
    read_cache: &'a std::sync::Mutex<crate::ReadCache>,
    repo_map: &'a std::sync::Mutex<crate::RepoMapCache>,
}

#[cfg(test)]
pub(crate) async fn execute(name: &str, arguments: &str) -> ToolOutcome {
    let root = std::env::current_dir().expect("test working directory");
    execute_in(&root, name, arguments).await
}

#[cfg(test)]
pub(crate) async fn execute_in(root: &Path, name: &str, arguments: &str) -> ToolOutcome {
    static NEXT_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = root.canonicalize().expect("canonical test workspace root");
    let state = std::env::temp_dir().join(format!(
        "hi-tools-test-state-{}-{}",
        std::process::id(),
        NEXT_STATE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::create_dir_all(&state);
    let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root));
    let background = crate::BackgroundRegistry::default();
    let read_cache = std::sync::Mutex::new(crate::ReadCache::new());
    let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
    let outcome = execute_in_impl(
        &root,
        &state,
        RuntimeResources {
            lsp: &lsp,
            background: &background,
            read_cache: &read_cache,
            repo_map: &repo_map,
        },
        name,
        arguments,
    )
    .await;
    let _ = std::fs::remove_dir_all(state);
    outcome
}

pub async fn execute_in_runtime(
    root: &Path,
    state_root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    background: &crate::BackgroundRegistry,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    repo_map: &std::sync::Mutex<crate::RepoMapCache>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    execute_in_impl(
        root,
        state_root,
        RuntimeResources {
            lsp,
            background,
            read_cache,
            repo_map,
        },
        name,
        arguments,
    )
    .await
}

/// Refuse `write` overwrites of existing files larger than this (bytes). Forces
/// the model onto `edit` / `multi_edit` / `apply_patch` for real source rewrites.
/// Creates and small-file overwrites still go through `write`.
pub const MAX_WRITE_OVERWRITE_BYTES: u64 = 16 * 1024;

/// Parse and materialize one built-in file mutation without touching its
/// targets. Preparation errors are returned to the caller and must not be
/// discarded before asking for confirmation.
pub async fn prepare_mutation_in_with_state(
    root: &Path,
    state_root: &Path,
    name: &str,
    arguments: &str,
) -> Result<PreparedMutation> {
    match name {
        "write" => {
            let args: WriteArgs = parse(arguments)?;
            let target = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
            refuse_large_write_overwrite(&target, &args.path)?;
            let after = args.content;
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::write(
                    &args.path,
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::Write {
                    target,
                    path: args.path,
                    after,
                },
            })
        }
        "edit" => {
            let args: EditArgs = parse(arguments)?;
            let target = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
            let (before, after, replacements) = apply_edit_with_disk_retry(
                &target,
                &args.path,
                &args.old_string,
                &args.new_string,
                args.replace_all,
            )
            .await
            .with_context(|| format!("editing {}", args.path))?;
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::update_from_preimage(
                    &args.path,
                    before.as_bytes(),
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::Edit {
                    target,
                    path: args.path,
                    after,
                    replacements,
                    replace_all: args.replace_all,
                },
            })
        }
        "multi_edit" => {
            let args: MultiEditArgs = parse(arguments)?;
            let target = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
            if args.edits.is_empty() {
                bail!("no edits provided");
            }
            let (before, after) =
                apply_multi_edit_with_disk_retry(&target, &args.path, &args.edits).await?;
            let edit_count = args.edits.len();
            let plan = MutationPlan::new_with_state(
                root,
                state_root,
                vec![PlannedFileMutation::update_from_preimage(
                    &args.path,
                    before.as_bytes(),
                    after.as_bytes().to_vec(),
                )],
            )?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::MultiEdit {
                    target,
                    path: args.path,
                    after,
                    edit_count,
                },
            })
        }
        "apply_patch" => {
            #[derive(Deserialize)]
            struct PatchArgs {
                patch: String,
            }
            let args: PatchArgs = parse(arguments)?;
            let (plan, summary) =
                plan_multi_patch_with_disk_retry(root, state_root, &args.patch).await?;
            Ok(PreparedMutation {
                plan,
                kind: PreparedMutationKind::ApplyPatch { summary },
            })
        }
        _ => bail!("{name} is not a preparable file mutation"),
    }
}

fn refuse_large_write_overwrite(target: &Path, display_path: &str) -> Result<()> {
    if !target.is_file() {
        return Ok(());
    }
    let meta = std::fs::metadata(target)
        .with_context(|| format!("statting existing file {display_path}"))?;
    if meta.len() > MAX_WRITE_OVERWRITE_BYTES {
        bail!(
            "refusing to overwrite existing `{display_path}` ({} bytes) via `write` — \
             use `edit`, `multi_edit`, or `apply_patch` for in-place changes to large files \
             (limit {} bytes). `write` is for creates and small files only.",
            meta.len(),
            MAX_WRITE_OVERWRITE_BYTES
        );
    }
    Ok(())
}

/// Apply one edit; if the anchor miss looks like a stale disk race, re-read once
/// and retry. Ambiguous matches are never auto-picked.
async fn apply_edit_with_disk_retry(
    target: &Path,
    display_path: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<(String, String, usize)> {
    let path_str = target.to_string_lossy().into_owned();
    let before = crate::read::read_text_file(&path_str).await?;
    match apply_edit(&before, old, new, replace_all) {
        Ok(after) => {
            let replacements = if replace_all {
                before.matches(old).count().max(1)
            } else {
                1
            };
            Ok((before, after, replacements))
        }
        Err(first) if is_retryable_edit_miss(&first) => {
            // Brief yield so a concurrent writer can finish; then re-read.
            tokio::task::yield_now().await;
            let refreshed = crate::read::read_text_file(&path_str).await?;
            if refreshed == before {
                return Err(first).with_context(|| format!("editing {display_path}"));
            }
            let after = apply_edit(&refreshed, old, new, replace_all).with_context(|| {
                format!(
                    "editing {display_path} (retried after on-disk change; \
                     original miss: {first:#})"
                )
            })?;
            let replacements = if replace_all {
                refreshed.matches(old).count().max(1)
            } else {
                1
            };
            Ok((refreshed, after, replacements))
        }
        Err(err) => Err(err).with_context(|| format!("editing {display_path}")),
    }
}

async fn apply_multi_edit_with_disk_retry(
    target: &Path,
    display_path: &str,
    edits: &[EditOp],
) -> Result<(String, String)> {
    let path_str = target.to_string_lossy().into_owned();
    let before = crate::read::read_text_file(&path_str).await?;
    match apply_edit_chain(&before, edits, display_path) {
        Ok(after) => Ok((before, after)),
        Err(first) if is_retryable_edit_miss(&first) => {
            tokio::task::yield_now().await;
            let refreshed = crate::read::read_text_file(&path_str).await?;
            if refreshed == before {
                return Err(first);
            }
            let after = apply_edit_chain(&refreshed, edits, display_path).with_context(|| {
                format!("multi_edit {display_path} retried after on-disk change")
            })?;
            Ok((refreshed, after))
        }
        Err(err) => Err(err),
    }
}

fn apply_edit_chain(before: &str, edits: &[EditOp], display_path: &str) -> Result<String> {
    let mut after = before.to_string();
    for (index, edit) in edits.iter().enumerate() {
        after = apply_edit(&after, &edit.old_string, &edit.new_string, false)
            .with_context(|| format!("editing {display_path} (edit #{})", index + 1))?;
    }
    Ok(after)
}

async fn plan_multi_patch_with_disk_retry(
    root: &Path,
    state_root: &Path,
    patch: &str,
) -> Result<(MutationPlan, String)> {
    match plan_multi_patch(root, state_root, patch) {
        Ok(ok) => Ok(ok),
        Err(first) if is_retryable_patch_miss(&first) => {
            tokio::task::yield_now().await;
            // Re-plan reads files fresh from disk; a second attempt only helps
            // when the underlying files changed underfoot.
            match plan_multi_patch(root, state_root, patch) {
                Ok(ok) => Ok(ok),
                Err(second) => Err(first).with_context(|| format!("apply_patch failed ({second:#})")),
            }
        }
        Err(err) => Err(err),
    }
}

fn is_retryable_edit_miss(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("old_string not found")
        || msg.contains("replace_all found no exact occurrences")
}

fn is_retryable_patch_miss(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    // found 0 → stale; found >1 → ambiguous — only retry stale (0).
    msg.contains("hunk context must match one unique contiguous region (found 0)")
        || msg.contains("addition-only hunk has no unique insertion anchor")
}

/// Commit the exact mutation plan previously displayed for confirmation.
/// Preimage changes made while the confirmation UI was open cause a typed
/// failure and are never overwritten.
pub async fn execute_prepared_in_runtime(
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    prepared: PreparedMutation,
) -> ToolOutcome {
    match run_prepared_mutation(lsp, read_cache, prepared).await {
        Ok(outcome) => outcome,
        Err(error) => {
            // A failed digest precondition means something else changed the
            // workspace while confirmation was open. Do not let a later read
            // reuse content cached before that external edit.
            if let Ok(mut cache) = read_cache.lock() {
                cache.clear();
            }
            let mut outcome = ToolOutcome::failed(format!("Error: {error:#}"));
            outcome.effects.mutation_attempted = true;
            outcome
        }
    }
}

async fn run_prepared_mutation(
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    prepared: PreparedMutation,
) -> Result<ToolOutcome> {
    let display = prepared.preview();
    let changes = prepared.plan.commit()?;
    let mut outcome = match prepared.kind {
        PreparedMutationKind::Write {
            target,
            path,
            after,
        } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.remove(&cache_key(&target));
            }
            sync_lsp_document(lsp, &target, &after).await;
            ToolOutcome::shown(format!("Wrote {} bytes to {path}", after.len()), display)
        }
        PreparedMutationKind::Edit {
            target,
            path,
            after,
            replacements,
            replace_all,
        } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.remove(&cache_key(&target));
            }
            sync_lsp_document(lsp, &target, &after).await;
            let message = if replace_all && replacements > 1 {
                format!("Replaced {replacements} occurrences in {path}")
            } else {
                format!("Edited {path}")
            };
            ToolOutcome::shown(message, display)
        }
        PreparedMutationKind::MultiEdit {
            target,
            path,
            after,
            edit_count,
        } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.remove(&cache_key(&target));
            }
            sync_lsp_document(lsp, &target, &after).await;
            ToolOutcome::shown(format!("Applied {edit_count} edits to {path}"), display)
        }
        PreparedMutationKind::ApplyPatch { summary } => {
            if let Ok(mut cache) = read_cache.lock() {
                cache.clear();
            }
            ToolOutcome::plain(summary)
        }
    };
    outcome.effects = mutation_effects(changes);
    Ok(outcome)
}

async fn execute_in_impl(
    root: &Path,
    state_root: &Path,
    resources: RuntimeResources<'_>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    match run(root, state_root, resources, name, arguments).await {
        Ok(output) => output,
        Err(err) => {
            let mut outcome = ToolOutcome::failed(format!("Error: {err:#}"));
            outcome.effects.mutation_attempted = mutation_attempted_by_tool(name);
            outcome
        }
    }
}

// The callback is intentionally passed separately from the five workspace
// resources so callers can stream without boxing or hiding lifetimes.
#[allow(clippy::too_many_arguments)]
pub async fn execute_streaming_in_runtime(
    root: &Path,
    state_root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    background: &crate::BackgroundRegistry,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    repo_map: &std::sync::Mutex<crate::RepoMapCache>,
    name: &str,
    arguments: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> ToolOutcome {
    execute_streaming_in_impl(
        root,
        state_root,
        RuntimeResources {
            lsp,
            background,
            read_cache,
            repo_map,
        },
        name,
        arguments,
        on_line,
    )
    .await
}

async fn execute_streaming_in_impl(
    root: &Path,
    state_root: &Path,
    resources: RuntimeResources<'_>,
    name: &str,
    arguments: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> ToolOutcome {
    match run_streaming(root, state_root, resources, name, arguments, on_line).await {
        Ok(output) => output,
        Err(err) => {
            let mut outcome = ToolOutcome::failed(format!("Error: {err:#}"));
            outcome.effects.mutation_attempted = mutation_attempted_by_tool(name);
            outcome
        }
    }
}

async fn run_streaming(
    root: &Path,
    state_root: &Path,
    resources: RuntimeResources<'_>,
    name: &str,
    arguments: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<ToolOutcome> {
    if name == "bash" {
        let args: BashArgs = parse(arguments)?;
        return run_bash_tool(root, state_root, resources, args, on_line).await;
    }
    // All other tools: delegate to the normal path (on_line unused).
    run(root, state_root, resources, name, arguments).await
}

async fn run(
    root: &Path,
    state_root: &Path,
    resources: RuntimeResources<'_>,
    name: &str,
    arguments: &str,
) -> Result<ToolOutcome> {
    match name {
        "read" => run_read(root, resources.read_cache, arguments).await,
        "update_plan" => {
            #[derive(Deserialize)]
            struct StepArg {
                title: String,
                #[serde(default)]
                status: String,
            }
            #[derive(Deserialize)]
            struct PlanArgs {
                steps: Vec<StepArg>,
            }
            let args: PlanArgs = parse(arguments)?;
            if args.steps.is_empty() {
                bail!("update_plan needs at least one step");
            }
            let steps: Vec<PlanStep> = args
                .steps
                .into_iter()
                .map(|s| PlanStep {
                    title: s.title,
                    status: PlanStatus::parse(&s.status),
                })
                .collect();
            let done = steps
                .iter()
                .filter(|s| s.status == PlanStatus::Done)
                .count();
            Ok(ToolOutcome::planned(
                format!("Plan recorded: {done}/{} done.", steps.len()),
                steps,
            ))
        }
        "write" | "edit" | "multi_edit" | "apply_patch" => {
            let prepared =
                prepare_mutation_in_with_state(root, state_root, name, arguments).await?;
            run_prepared_mutation(resources.lsp, resources.read_cache, prepared).await
        }
        "bash" => {
            let args: BashArgs = parse(arguments)?;
            // Read-cache invalidation lives inside run_bash_tool, so both this
            // dispatch path and the streaming path (execute_streaming) clear it.
            run_bash_tool(root, state_root, resources, args, &mut |_| {}).await
        }
        "bash_output" => {
            #[derive(Deserialize)]
            struct Args {
                id: String,
            }
            let args: Args = parse(arguments)?;
            let result = resources.background.poll(&args.id)?;
            let background = resources.background.outcome(&args.id)?;
            if let Ok(mut cache) = resources.read_cache.lock() {
                cache.clear();
            }
            let mut outcome = background_tool_outcome(condense(&result), background);
            attach_background_effects(&mut outcome, resources.background, &args.id).await;
            Ok(outcome)
        }
        "bash_kill" => {
            #[derive(Deserialize)]
            struct Args {
                id: String,
            }
            let args: Args = parse(arguments)?;
            let result = resources.background.kill(&args.id)?;
            let background = resources.background.outcome(&args.id)?;
            if let Ok(mut cache) = resources.read_cache.lock() {
                cache.clear();
            }
            let mut outcome = background_tool_outcome(result, background);
            attach_background_effects(&mut outcome, resources.background, &args.id).await;
            Ok(outcome)
        }
        "list" => run_list(root, arguments).await,
        "repo_map" => crate::repo_map::run_repo_map(root, resources.repo_map, arguments).await,
        "find_symbol" => {
            crate::repo_map::run_find_symbol(root, resources.repo_map, arguments).await
        }
        "diff" => {
            // Reuse the working-tree diff summary, but return it as model content
            // (plain text, no ANSI) so the model can review what changed. A
            // tracked diff can be arbitrarily large, so enforce the same
            // context budget as reads/process output at this final tool
            // boundary and surface typed truncation metadata.
            Ok(ToolOutcome::bounded_plain(
                working_tree_diff_plain_in(root).await,
            ))
        }
        "glob" => run_glob(root, arguments).await,
        "grep" => run_grep(root, arguments).await,
        "diagnostics" => run_lsp_diagnostics(root, resources.lsp, arguments).await,
        "definition" => run_lsp_definition(root, resources.lsp, arguments).await,
        "references" => run_lsp_references(root, resources.lsp, arguments).await,
        "hover" => run_lsp_hover(root, resources.lsp, arguments).await,
        "web_search" => crate::web::run_web_search(arguments).await,
        "web_fetch" => crate::web::run_web_fetch(arguments).await,
        "web_download" => {
            crate::web::run_web_download_in(root, resources.background, arguments).await
        }
        other => bail!("unknown tool: {other}"),
    }
}

async fn sync_lsp_document(lsp: &std::sync::Arc<hi_lsp::LspManager>, path: &Path, text: &str) {
    let _ = lsp.sync_document(path, text).await;
}

fn mutation_effects(changes: Vec<crate::FileChange>) -> ToolEffects {
    ToolEffects {
        mutation_attempted: true,
        mutation_applied: !changes.is_empty(),
        file_changes: changes,
    }
}

fn mutation_attempted_by_tool(name: &str) -> bool {
    is_filesystem_mutating(name) || name == "bash"
}

fn mark_effect_inspection_failed(
    outcome: &mut ToolOutcome,
    error: &anyhow::Error,
    mutation_may_have_applied: bool,
) {
    outcome.status = crate::ToolStatus::Failed;
    if !outcome.content.ends_with('\n') {
        outcome.content.push('\n');
    }
    outcome.content.push_str(&format!(
        "[infrastructure failure: could not inspect workspace effects: {error:#}]"
    ));
    outcome.effects = ToolEffects {
        mutation_attempted: true,
        // There is no "unknown" effects state in the public contract. Once a
        // process has run, conservatively report a possible applied mutation;
        // the Failed status and empty exact list make the inspection failure
        // authoritative instead of incorrectly presenting a clean workspace.
        mutation_applied: mutation_may_have_applied,
        file_changes: Vec::new(),
    };
}

/// Attach effect attribution to a background tool outcome. Lifecycle status
/// (Succeeded / Cancelled / Failed from exit) wins over inspection failures —
/// a kill that reaped cleanly must stay Cancelled even if the workspace scan
/// times out under suite load.
async fn attach_background_effects(
    outcome: &mut ToolOutcome,
    background: &crate::BackgroundRegistry,
    id: &str,
) {
    let lifecycle_status = outcome.status;
    match background.effects(id).await {
        Ok(effects) => outcome.effects = effects,
        Err(error) => {
            mark_effect_inspection_failed(outcome, &error, true);
            if matches!(
                lifecycle_status,
                crate::ToolStatus::Cancelled | crate::ToolStatus::Succeeded
            ) {
                outcome.status = lifecycle_status;
            }
        }
    }
}

fn background_tool_outcome(content: String, background: crate::BackgroundOutcome) -> ToolOutcome {
    let status = match background.state {
        crate::BackgroundState::Started | crate::BackgroundState::Running => {
            crate::ToolStatus::Succeeded
        }
        crate::BackgroundState::Exited if background.exit_code == Some(0) => {
            crate::ToolStatus::Succeeded
        }
        crate::BackgroundState::Killed => crate::ToolStatus::Cancelled,
        crate::BackgroundState::Exited | crate::BackgroundState::Failed => {
            crate::ToolStatus::Failed
        }
    };
    let mut outcome = ToolOutcome::plain(content);
    outcome.status = status;
    outcome.background = Some(background);
    outcome
}

/// Compute a human-readable diff of what a mutating tool call *would* change,
/// without writing anything — so `--confirm-edits` can show the change before
/// the user approves it rather than asking blind. `None` for calls that can't be
/// previewed (unparseable args, missing file, or a non-mutating tool).
#[cfg(test)]
pub(crate) async fn preview_edit_in(root: &Path, name: &str, arguments: &str) -> Option<String> {
    prepare_mutation_in_with_state(root, &root.join(".hi-test-state"), name, arguments)
        .await
        .ok()
        .map(|prepared| prepared.preview())
}

// --- LSP tool handlers ---

async fn run_lsp_diagnostics(
    root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    arguments: &str,
) -> Result<ToolOutcome> {
    if !lsp.is_enabled().await {
        return Ok(ToolOutcome::denied("LSP is off (use `/lsp on`).".into()));
    }
    #[derive(Deserialize)]
    struct Args {
        #[serde(default)]
        path: String,
    }
    let args: Args = parse(arguments)?;
    if args.path.is_empty() {
        // No specific file — return diagnostics across all synced documents.
        let all = lsp.diagnostic_states_all().await;
        if all.is_empty() {
            return Ok(ToolOutcome::failed(
                "LSP has no confirmed diagnostic state for any document.".into(),
            ));
        }
        let mut out = String::new();
        let mut failed = false;
        let mut any_diagnostics = false;
        for (path, state) in all {
            match state {
                hi_lsp::DiagnosticState::ConfirmedClean { document_version } => {
                    out.push_str(&format!(
                        "{}: confirmed clean (document version {document_version})\n",
                        path.display()
                    ));
                }
                hi_lsp::DiagnosticState::DiagnosticsPresent {
                    document_version,
                    diagnostics,
                } => {
                    any_diagnostics = true;
                    append_diagnostics(&mut out, &path, document_version, &diagnostics);
                }
                hi_lsp::DiagnosticState::Unavailable { reason, .. } => {
                    failed = true;
                    out.push_str(&format!(
                        "{}: diagnostics unavailable: {reason}\n",
                        path.display()
                    ));
                }
                hi_lsp::DiagnosticState::Failed { error, .. } => {
                    failed = true;
                    out.push_str(&format!(
                        "{}: diagnostics failed: {error}\n",
                        path.display()
                    ));
                }
            }
        }
        if !any_diagnostics && !failed {
            return Ok(ToolOutcome::plain(
                "No diagnostics (confirmed clean).".into(),
            ));
        }
        return Ok(if failed {
            ToolOutcome::failed(out.trim_end().to_string())
        } else {
            ToolOutcome::plain(out.trim_end().to_string())
        });
    }
    let path = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
    // Sync the file first so diagnostics reflect current state.
    if let Ok(text) = tokio::fs::read_to_string(&path).await {
        let _ = lsp.sync_document(&path, &text).await;
    }
    match lsp.diagnostic_state(&path).await {
        hi_lsp::DiagnosticState::ConfirmedClean { document_version } => Ok(ToolOutcome::plain(
            format!("No diagnostics (confirmed clean at document version {document_version})."),
        )),
        hi_lsp::DiagnosticState::DiagnosticsPresent {
            document_version,
            diagnostics,
        } => {
            let mut out = String::new();
            append_diagnostics(&mut out, &path, document_version, &diagnostics);
            Ok(ToolOutcome::plain(out.trim_end().to_string()))
        }
        hi_lsp::DiagnosticState::Unavailable { reason, .. } => Ok(ToolOutcome::failed(format!(
            "Diagnostics unavailable for {}: {reason}",
            path.display()
        ))),
        hi_lsp::DiagnosticState::Failed { error, .. } => Ok(ToolOutcome::failed(format!(
            "Diagnostics failed for {}: {error}",
            path.display()
        ))),
    }
}

fn append_diagnostics(
    out: &mut String,
    path: &Path,
    document_version: u64,
    diagnostics: &[hi_lsp::Diagnostic],
) {
    for diagnostic in diagnostics {
        let source = diagnostic.source.as_deref().unwrap_or("");
        out.push_str(&format!(
            "{}:{}:{}: {} {}{} [document version {}]\n",
            path.display(),
            diagnostic.line + 1,
            diagnostic.col + 1,
            diagnostic.severity,
            diagnostic.message,
            if source.is_empty() {
                String::new()
            } else {
                format!(" ({source})")
            },
            document_version,
        ));
    }
}

async fn run_lsp_definition(
    root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    arguments: &str,
) -> Result<ToolOutcome> {
    run_lsp_locations(root, lsp, "definition", arguments).await
}

async fn run_lsp_references(
    root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    arguments: &str,
) -> Result<ToolOutcome> {
    run_lsp_locations(root, lsp, "references", arguments).await
}

async fn run_lsp_locations(
    root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    kind: &str,
    arguments: &str,
) -> Result<ToolOutcome> {
    if !lsp.is_enabled().await {
        return Ok(ToolOutcome::plain("LSP is off (use `/lsp on`).".into()));
    }
    #[derive(Deserialize)]
    struct Args {
        path: String,
        line: u32,
        column: u32,
    }
    let args: Args = parse(arguments)?;
    let path = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
    if let Ok(text) = tokio::fs::read_to_string(&path).await {
        let _ = lsp.sync_document(&path, &text).await;
    }
    let locs = if kind == "definition" {
        lsp.definition(&path, args.line, args.column).await?
    } else {
        lsp.references(&path, args.line, args.column).await?
    };
    if locs.is_empty() {
        return Ok(ToolOutcome::plain(format!("No {kind} found.")));
    }
    let out = locs
        .iter()
        .map(|l| format!("{}:{}:{}", l.path, l.line + 1, l.col + 1))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(ToolOutcome::plain(out))
}

async fn run_lsp_hover(
    root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    arguments: &str,
) -> Result<ToolOutcome> {
    if !lsp.is_enabled().await {
        return Ok(ToolOutcome::plain("LSP is off (use `/lsp on`).".into()));
    }
    #[derive(Deserialize)]
    struct Args {
        path: String,
        line: u32,
        column: u32,
    }
    let args: Args = parse(arguments)?;
    let path = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
    if let Ok(text) = tokio::fs::read_to_string(&path).await {
        let _ = lsp.sync_document(&path, &text).await;
    }
    match lsp.hover(&path, args.line, args.column).await? {
        Some(text) => Ok(ToolOutcome::plain(text)),
        None => Ok(ToolOutcome::plain("No hover info.".into())),
    }
}

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
async fn run_bash_streaming_with_timeout(
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

pub(crate) fn parse<T: for<'de> Deserialize<'de>>(arguments: &str) -> Result<T> {
    serde_json::from_str(arguments).context("invalid tool arguments")
}

#[derive(Deserialize)]
pub(crate) struct MultiEditArgs {
    pub path: String,
    pub edits: Vec<EditOp>,
}

#[derive(Deserialize)]
pub(crate) struct EditOp {
    pub old_string: String,
    pub new_string: String,
}

#[derive(Deserialize)]
pub(crate) struct WriteArgs {
    pub path: String,
    pub content: String,
}

#[derive(Deserialize)]
pub(crate) struct EditArgs {
    pub path: String,
    pub old_string: String,
    pub new_string: String,
    /// If true, replace every occurrence of `old_string` (default: false).
    #[serde(default)]
    pub replace_all: bool,
}

#[derive(Deserialize)]
pub(crate) struct BashArgs {
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
async fn run_bash_tool(
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
fn foreground_interactive_command_reason(command: &str) -> Option<&'static str> {
    let root = std::env::current_dir().ok()?;
    foreground_interactive_command_reason_at(&root, command)
}

fn foreground_interactive_command_reason_at(root: &Path, command: &str) -> Option<&'static str> {
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

#[cfg(test)]
mod tests {
    use super::{
        MAX_WRITE_OVERWRITE_BYTES, TOOL_SPECS, fast_check_for,
        foreground_interactive_command_reason, foreground_interactive_command_reason_at,
        is_retryable_edit_miss, render_untracked_files, render_untracked_files_with_contents,
        run_bash_streaming_with_timeout, run_check_in, working_tree_diff_plain_in,
    };
    use crate::edit::{apply_edit, sh_quote};
    use std::time::Duration;

    #[test]
    fn diff_untracked_files_are_collapsed_and_capped() {
        let files = [
            "models/a.bin",
            "models/b.bin",
            "scratch/one.txt",
            "scratch/two.txt",
            "top.txt",
            "z.txt",
        ];

        let rendered = render_untracked_files(&files, 3);

        assert!(rendered.contains("  + models/ (2 entries)"));
        assert!(rendered.contains("  + scratch/ (2 entries)"));
        assert!(rendered.contains("  ... omitted 1 untracked entry (limit 3)"));
        assert!(!rendered.contains("models/a.bin"));
    }

    #[test]
    fn diff_untracked_files_include_bounded_text_and_summarize_vendor_and_binary() {
        let dir = std::env::temp_dir().join(format!(
            "hi-created-diff-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join("vendor")).unwrap();
        std::fs::write(dir.join("src/new.rs"), "fn new_file() {}\n").unwrap();
        std::fs::write(dir.join("vendor/library.js"), "do_not_render();\n").unwrap();
        std::fs::write(dir.join("asset.bin"), [0, 1, 2, 3]).unwrap();

        let rendered = render_untracked_files_with_contents(
            &dir,
            &["src/new.rs", "vendor/library.js", "asset.bin"],
            10,
        );
        let _ = std::fs::remove_dir_all(&dir);

        assert!(rendered.contains("+++ b/src/new.rs"));
        assert!(rendered.contains("+fn new_file() {}"));
        assert!(rendered.contains("summarized binary/generated/vendor/oversized files"));
        assert!(rendered.contains("vendor/"));
        assert!(rendered.contains("asset.bin"));
        assert!(!rendered.contains("do_not_render"));
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn diff_in_non_git_directory_is_concise() {
        let dir = std::env::temp_dir().join(format!("hi-diff-non-git-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let output = working_tree_diff_plain_in(&dir).await;
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(output, "not a git repository; no git diff available");
    }

    #[tokio::test]
    async fn diff_tool_bounds_large_tracked_diff_and_reports_truncation() {
        let dir = std::env::temp_dir().join(format!(
            "hi-diff-bounded-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let init = std::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(&dir)
            .status()
            .unwrap();
        assert!(init.success());
        std::fs::write(dir.join("large.txt"), "before\n".repeat(20_000)).unwrap();
        let add = std::process::Command::new("git")
            .args(["add", "large.txt"])
            .current_dir(&dir)
            .status()
            .unwrap();
        assert!(add.success());
        std::fs::write(dir.join("large.txt"), "after\n".repeat(20_000)).unwrap();

        let direct = crate::execute_in(&dir, "diff", "{}").await;

        let state = dir.join(".hi-test-state");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&dir));
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        let mut sink = |_: &str| {};
        let streaming = crate::execute_streaming_in_runtime(
            &dir,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "diff",
            "{}",
            &mut sink,
        )
        .await;
        let _ = std::fs::remove_dir_all(&dir);

        for outcome in [direct, streaming] {
            assert_eq!(outcome.status, crate::ToolStatus::Succeeded);
            assert!(outcome.content.contains("truncated"), "{}", outcome.content);
            assert!(
                outcome.content.chars().count() < 6_000,
                "bounded diff was {} chars",
                outcome.content.chars().count()
            );
            match outcome.truncation {
                crate::TruncationState::Truncated {
                    original_bytes,
                    retained_bytes,
                } => assert!(original_bytes > retained_bytes),
                crate::TruncationState::Complete => panic!("large diff was reported complete"),
            }
        }
    }

    #[test]
    fn bounded_plain_types_just_over_limit_utf8_when_marker_adds_bytes() {
        let max = *crate::condense::MAX_OUTPUT_CHARS;
        let original = "é".repeat(max + 1);
        let original_bytes = original.len() as u64;

        let (content, truncation) = crate::bound_tool_content(original);

        assert!(content.contains("truncated 1 characters"));
        assert_eq!(content.chars().filter(|ch| *ch == 'é').count(), max);
        match truncation {
            crate::TruncationState::Truncated {
                original_bytes: reported_original,
                retained_bytes,
            } => {
                assert_eq!(reported_original, original_bytes);
                assert_eq!(retained_bytes, content.len() as u64);
                assert!(
                    retained_bytes > original_bytes,
                    "the marker must exercise the case where clipped output is byte-larger"
                );
            }
            crate::TruncationState::Complete => {
                panic!("just-over-limit UTF-8 output was reported complete")
            }
        }
    }

    // A command that keeps its stdout pipe open and never exits must still
    // return via the timeout. Before the fix the timeout wrapped only
    // `child.wait()`, reached after the pipes drained — so a process holding
    // its pipes open blocked the reader forever and the timeout never armed.
    #[tokio::test]
    async fn bash_times_out_when_process_holds_pipe_open() {
        let mut sink = |_: &str| {};
        let out = tokio::time::timeout(
            Duration::from_secs(5),
            run_bash_streaming_with_timeout("sleep 600", &mut sink, Duration::from_millis(200)),
        )
        .await
        .expect("must not hang past the outer guard")
        .expect("bash run returns Ok with a timeout notice");
        assert!(out.contains("timed out"), "got: {out:?}");
    }

    // Output produced before a hang is preserved in the returned text.
    #[tokio::test]
    async fn bash_timeout_preserves_partial_output() {
        let mut sink = |_: &str| {};
        let out = tokio::time::timeout(
            Duration::from_secs(5),
            run_bash_streaming_with_timeout(
                "echo before-hang; sleep 600",
                &mut sink,
                Duration::from_millis(300),
            ),
        )
        .await
        .expect("must not hang past the outer guard")
        .expect("bash run returns Ok");
        assert!(out.contains("before-hang"), "got: {out:?}");
        assert!(out.contains("timed out"), "got: {out:?}");
    }

    // The normal path is unchanged: a fast command returns its output and the
    // exit code is appended on failure.
    #[tokio::test]
    async fn bash_normal_command_returns_output() {
        let mut sink = |_: &str| {};
        let out = run_bash_streaming_with_timeout("echo hello", &mut sink, Duration::from_secs(10))
            .await
            .expect("ok");
        assert!(out.contains("hello"), "got: {out:?}");
        assert!(!out.contains("timed out"), "got: {out:?}");
    }

    #[tokio::test]
    async fn bash_marks_hugging_face_agent_harness() {
        let mut sink = |_: &str| {};
        let out = run_bash_streaming_with_timeout(
            "printf '%s' \"$AI_AGENT\"",
            &mut sink,
            Duration::from_secs(10),
        )
        .await
        .expect("ok");
        assert_eq!(out.trim_end(), "hi");
    }

    #[tokio::test]
    async fn verify_marks_hugging_face_agent_harness() {
        let execution = run_check_in(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
            "printf '%s' \"$AI_AGENT\"",
        )
        .await
        .unwrap();
        assert_eq!(execution.status, crate::ToolStatus::Succeeded);
        assert_eq!(execution.model_content(), "hi");
    }

    /// Auto-background-on-timeout: a foreground command still running at its
    /// budget is moved to the background (handle returned) instead of killed.
    /// A unique, isolated `(root, state)` pair under the system temp dir. Auto-
    /// background tests must NOT share `CARGO_MANIFEST_DIR` as the workspace root
    /// — the effect-snapshot walk of one test would race another test's
    /// `remove_dir_all` of a state dir sitting inside that shared root.
    fn isolated_ws(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let base = std::env::temp_dir().join(format!(
            "hi-autobg-{tag}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let root = base.join("ws");
        let state = base.join("state");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        (root, state)
    }

    // Multi-thread flavor: the foreground budget is a real tokio timer, and on a
    // loaded current-thread runtime that timer can be starved by the blocking
    // child, making the handoff timing flaky under CI load. A dedicated worker
    // thread lets the timer fire independently of the process I/O.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bash_moves_to_background_on_timeout_instead_of_killing() {
        let (root, state) = isolated_ws("bg");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root));
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        // timeout:1 → foreground budget is 1s; a 600s sleep outlasts it.
        let outcome = crate::execute_in_runtime(
            &root,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash",
            r#"{"command":"sleep 600","timeout":1}"#,
        )
        .await;
        assert!(
            outcome.content.contains("moved to background"),
            "not killed — backgrounded: {:?}",
            outcome.content
        );
        let bg = outcome.background.expect("a background handle is returned");
        assert_eq!(bg.state, crate::BackgroundState::Started);
        assert!(bg.id.starts_with("bg_"), "got: {}", bg.id);
        assert!(
            outcome.effects.mutation_attempted,
            "a backgrounded command may have mutated the tree"
        );
        // Registry drop kills the adopted process.
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    /// A command that finishes inside its budget takes the normal foreground
    /// path (full output, no background handle).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bash_fast_command_stays_foreground_under_auto_background() {
        let (root, state) = isolated_ws("fast");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root));
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        let outcome = crate::execute_in_runtime(
            &root,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash",
            r#"{"command":"echo fast-hello","timeout":30}"#,
        )
        .await;
        assert!(
            outcome.content.contains("fast-hello"),
            "foreground output returned: {:?}",
            outcome.content
        );
        assert!(outcome.background.is_none(), "no background handle");
        assert_eq!(outcome.status, crate::ToolStatus::Succeeded);
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    /// A shell command can mutate any file, so `bash` must invalidate the read
    /// cache — otherwise a later `read` serves stale pre-bash content.
    #[tokio::test]
    async fn bash_invalidates_the_read_cache() {
        use crate::paths::cache_key;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let state = root.join(".hi-test-state");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(root));
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        let key = cache_key(std::path::Path::new("/tmp/hi-read-cache-probe"));
        cache.lock().unwrap().insert(key.clone(), "stale".into());
        let _ = crate::execute_in_runtime(
            root,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash",
            r#"{"command":"true"}"#,
        )
        .await;
        assert!(
            cache.lock().unwrap().get(&key).is_none(),
            "bash must clear the read cache"
        );
    }

    /// The real bug: the *streaming* entry point (execute_streaming) is what the
    /// live turn loop uses, and it short-circuits to run_bash_tool before the
    /// dispatch arm — so it must clear the cache too. This drives that path
    /// explicitly (the test above only covers non-streaming `execute`).
    #[tokio::test]
    async fn streaming_bash_invalidates_the_read_cache() {
        use crate::paths::cache_key;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let state = root.join(".hi-test-state");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(root));
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        let key = cache_key(std::path::Path::new("/tmp/hi-read-cache-probe-streaming"));
        cache.lock().unwrap().insert(key.clone(), "stale".into());
        let mut sink = |_: &str| {};
        let _ = crate::execute_streaming_in_runtime(
            root,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "bash",
            r#"{"command":"true"}"#,
            &mut sink,
        )
        .await;
        assert!(
            cache.lock().unwrap().get(&key).is_none(),
            "streaming bash must clear the read cache"
        );
    }

    #[tokio::test]
    async fn write_refuses_large_existing_file_overwrite() {
        let dir = std::env::temp_dir().join(format!(
            "hi-write-guard-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let state = dir.join("state");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&state).unwrap();
        let big = "x".repeat((MAX_WRITE_OVERWRITE_BYTES as usize) + 1);
        std::fs::write(dir.join("big.rs"), &big).unwrap();
        let args = serde_json::json!({
            "path": "big.rs",
            "content": "fn tiny() {}\n"
        })
        .to_string();
        let err = crate::prepare_mutation_in_with_state(&dir, &state, "write", &args)
            .await
            .expect_err("large overwrite must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("refusing to overwrite") && msg.contains("edit"),
            "{msg}"
        );
        // Unchanged on disk.
        assert_eq!(std::fs::read_to_string(dir.join("big.rs")).unwrap(), big);
        // Small overwrite still allowed.
        std::fs::write(dir.join("small.rs"), "old\n").unwrap();
        let small_args = r#"{"path":"small.rs","content":"new\n"}"#;
        assert!(
            crate::prepare_mutation_in_with_state(&dir, &state, "write", small_args)
                .await
                .is_ok()
        );
        // Create is always allowed.
        let create = r#"{"path":"brand_new.rs","content":"pub fn ok() {}\n"}"#;
        assert!(
            crate::prepare_mutation_in_with_state(&dir, &state, "write", create)
                .await
                .is_ok()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn edit_retries_once_when_disk_changes_underfoot() {
        // Exercise the retry path directly: first content misses, second hits.
        let dir = std::env::temp_dir().join(format!(
            "hi-edit-retry-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("race.rs");
        std::fs::write(&file, "stale content without anchor\n").unwrap();

        // Simulate the prepare path's retry helper with a controlled flip:
        // first apply fails, we rewrite the file, second apply succeeds.
        let before = std::fs::read_to_string(&file).unwrap();
        let first = apply_edit(&before, "beta", "BETA", false);
        assert!(first.is_err(), "stale content must miss");
        assert!(is_retryable_edit_miss(first.as_ref().unwrap_err()));
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();
        let refreshed = std::fs::read_to_string(&file).unwrap();
        assert_ne!(refreshed, before);
        let after = apply_edit(&refreshed, "beta", "BETA", false).expect("retry should hit");
        assert!(after.contains("BETA"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn edit_miss_without_disk_change_does_not_loop() {
        let dir = std::env::temp_dir().join(format!(
            "hi-edit-miss-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let state = dir.join("state");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&state).unwrap();
        std::fs::write(dir.join("a.rs"), "fn ok() {}\n").unwrap();
        let args = r#"{"path":"a.rs","old_string":"does_not_exist","new_string":"x"}"#;
        let err = crate::prepare_mutation_in_with_state(&dir, &state, "edit", args)
            .await
            .expect_err("miss must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("old_string not found") || msg.contains("not found"),
            "{msg}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `--confirm-edits` must show the real change: `preview_edit` computes the
    /// diff without writing.
    #[tokio::test]
    async fn preview_edit_computes_diff_without_writing() {
        let dir = std::env::temp_dir().join(format!("hi-preview-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let file = dir.join("a.txt");
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();
        let args = r#"{"path":"a.txt","old_string":"beta","new_string":"BETA"}"#;
        let preview = crate::preview_edit_in(&dir, "edit", args)
            .await
            .expect("a preview");
        assert!(
            preview.contains("BETA"),
            "preview shows the change: {preview}"
        );
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "alpha\nbeta\ngamma\n",
            "preview must not write to the file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn prepared_edit_refuses_an_edit_made_after_preview() {
        let dir =
            std::env::temp_dir().join(format!("hi-prepared-preview-race-{}", std::process::id()));
        let state = dir.join("state");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&state).unwrap();
        let file = dir.join("a.txt");
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();
        let args = r#"{"path":"a.txt","old_string":"beta","new_string":"BETA"}"#;
        let prepared = crate::prepare_mutation_in_with_state(&dir, &state, "edit", args)
            .await
            .unwrap();
        assert!(prepared.preview().contains("BETA"));

        // Simulate an editor save while the confirmation prompt is open.
        std::fs::write(&file, "external editor contents\n").unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&dir));
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let outcome = crate::execute_prepared_in_runtime(&lsp, &cache, prepared).await;

        assert_eq!(outcome.status, crate::ToolStatus::Failed);
        assert!(outcome.content.contains("file changed after preview"));
        assert!(outcome.effects.mutation_attempted);
        assert!(!outcome.effects.mutation_applied);
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "external editor contents\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // stdin is detached: a command reading stdin sees EOF immediately rather
    // than blocking on the agent's terminal.
    #[tokio::test]
    async fn bash_stdin_is_closed_not_blocking() {
        let mut sink = |_: &str| {};
        let out = tokio::time::timeout(
            Duration::from_secs(5),
            run_bash_streaming_with_timeout("cat", &mut sink, Duration::from_secs(10)),
        )
        .await
        .expect("must not block on stdin")
        .expect("ok");
        assert!(!out.contains("timed out"), "got: {out:?}");
    }

    #[test]
    fn detects_foreground_python_tui_commands() {
        let dir = std::env::temp_dir().join(format!("hi-tui-detect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let calc = dir.join("calc.py");
        let script = dir.join("script.py");
        std::fs::write(
            &calc,
            "from textual.app import App\n\nclass Calc(App):\n    pass\n\nCalc().run()\n",
        )
        .unwrap();
        assert!(
            foreground_interactive_command_reason(&format!("python3 {}", calc.display())).is_some()
        );
        assert!(
            foreground_interactive_command_reason(&format!(
                "TERM=xterm python3 {}",
                calc.display()
            ))
            .is_some()
        );
        assert!(
            foreground_interactive_command_reason(&format!(
                "timeout 5s python3 {}",
                calc.display()
            ))
            .is_none(),
            "explicit timeout smoke tests are allowed"
        );
        std::fs::write(&script, "print('done')\n").unwrap();
        assert!(
            foreground_interactive_command_reason(&format!("python3 {}", script.display()))
                .is_none()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn bash_refuses_foreground_python_tui() {
        let dir = std::env::temp_dir().join(format!("hi-tui-bash-refuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let calc = dir.join("calc.py");
        std::fs::write(
            &calc,
            "from textual.app import App\n\nclass Calc(App):\n    pass\n\nCalc().run()\n",
        )
        .unwrap();
        let args = serde_json::json!({ "command": format!("python3 {}", calc.display()) });
        let out = crate::execute("bash", &args.to_string()).await;
        assert!(out.content.contains("refused"), "got: {}", out.content);
        assert!(
            out.content.contains("Foreground interactive terminal apps"),
            "got: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn detects_foreground_rust_tui_cargo_run() {
        let dir = std::env::temp_dir().join(format!("hi-rust-tui-detect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nratatui = \"0.28\"\ncrossterm = \"0.28\"\n",
        )
        .unwrap();
        assert!(foreground_interactive_command_reason_at(&dir, "cargo run").is_some());
        assert!(foreground_interactive_command_reason_at(&dir, "TERM=xterm cargo run").is_some());
        assert!(
            foreground_interactive_command_reason_at(&dir, "timeout 5s cargo run").is_none(),
            "explicit timeout smoke tests are allowed"
        );
        assert!(
            foreground_interactive_command_reason_at(&dir, "cargo run -- --help").is_none(),
            "noninteractive help runs are allowed"
        );
        assert!(foreground_interactive_command_reason_at(&dir, "cargo test").is_none());
        assert!(foreground_interactive_command_reason_at(&dir, "cargo build").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // A timeout kills the whole process tree: a child that outlives its `sh`
    // parent (here, a backgrounded `sleep` holding the pipe) is reaped, so the
    // call returns promptly instead of the pipe keeping the reader alive.
    #[cfg(unix)]
    #[tokio::test]
    async fn bash_timeout_kills_descendants_holding_pipe() {
        let mut sink = |_: &str| {};
        // `sleep 600 &` backgrounds a child that inherits stdout; the script
        // then exits, but the pipe stays open via the grandchild. With group
        // kill this still returns at the timeout rather than hanging on read.
        let out = tokio::time::timeout(
            Duration::from_secs(5),
            run_bash_streaming_with_timeout(
                "sleep 600 & echo started; wait",
                &mut sink,
                Duration::from_millis(300),
            ),
        )
        .await
        .expect("must not hang on the orphaned grandchild's pipe")
        .expect("ok");
        assert!(out.contains("started"), "got: {out:?}");
        assert!(out.contains("timed out"), "got: {out:?}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_bash_future_kills_descendants() {
        let pid_file = std::env::temp_dir().join(format!(
            "hi-cancel-bash-child-{}-{}.pid",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&pid_file);
        let pid_path = pid_file.to_string_lossy().to_string();
        let command = format!(
            "trap '' HUP; sleep 600 & echo $! > {}; wait",
            sh_quote(&pid_path)
        );

        {
            let mut sink = |_: &str| {};
            let fut =
                run_bash_streaming_with_timeout(&command, &mut sink, Duration::from_secs(600));
            tokio::pin!(fut);

            let child_started = async {
                for _ in 0..100 {
                    if pid_file.exists() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                panic!("background child pid file was not written");
            };

            tokio::select! {
                result = &mut fut => panic!("command finished before cancellation: {result:?}"),
                _ = child_started => {}
            }
        }

        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("pid file readable")
            .trim()
            .parse()
            .expect("pid parseable");
        for _ in 0..100 {
            if !process_exists(pid) {
                let _ = std::fs::remove_file(&pid_file);
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        let _ = std::fs::remove_file(&pid_file);
        panic!("cancelled bash future left descendant process {pid} running");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn foreground_bash_completion_kills_detached_descendants() {
        let pid_file = std::env::temp_dir().join(format!(
            "hi-fg-bash-child-{}-{}.pid",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&pid_file);
        let pid_path = pid_file.to_string_lossy().to_string();
        let command = format!(
            "trap '' HUP; sleep 600 >/dev/null 2>&1 & echo $! > {}; echo done",
            sh_quote(&pid_path)
        );
        let mut sink = |_: &str| {};

        let out = run_bash_streaming_with_timeout(&command, &mut sink, Duration::from_secs(5))
            .await
            .expect("foreground command returns");
        assert!(out.contains("done"), "got: {out:?}");

        let pid: i32 = std::fs::read_to_string(&pid_file)
            .expect("pid file readable")
            .trim()
            .parse()
            .expect("pid parseable");
        for _ in 0..100 {
            if !process_exists(pid) {
                let _ = std::fs::remove_file(&pid_file);
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        let _ = std::fs::remove_file(&pid_file);
        panic!("foreground bash left detached descendant process {pid} running");
    }

    #[cfg(unix)]
    fn process_exists(pid: i32) -> bool {
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[test]
    fn bash_timeout_resolution_and_clamping() {
        use super::{DEFAULT_BASH_TIMEOUT_SECS, MAX_BASH_TIMEOUT_SECS, resolve_bash_timeout};
        // Explicit request wins and is honored.
        assert_eq!(resolve_bash_timeout(Some(42)).as_secs(), 42);
        // Absurd values clamp to the ceiling, not unbounded.
        assert_eq!(
            resolve_bash_timeout(Some(u64::MAX)).as_secs(),
            MAX_BASH_TIMEOUT_SECS
        );
        // Zero clamps up to 1 so the guard is never disabled.
        assert_eq!(resolve_bash_timeout(Some(0)).as_secs(), 1);
        // No request → the default (env not set in this test).
        assert_eq!(
            resolve_bash_timeout(None).as_secs(),
            DEFAULT_BASH_TIMEOUT_SECS
        );
    }




    #[test]
    fn sh_quote_escapes_single_quotes() {
        assert_eq!(sh_quote("a b"), "'a b'");
        assert_eq!(sh_quote("it's"), "'it'\\''s'");
    }


    #[test]
    fn fast_check_for_targets_per_file_languages() {
        // Python and Go have genuinely per-file fast checks.
        assert!(fast_check_for("src/a.py").is_some());
        assert!(fast_check_for("main.go").is_some());
        // TS/JS get a project-wide tsc (best available).
        assert!(fast_check_for("x.ts").is_some());
        assert!(fast_check_for("x.jsx").is_some());
        // Ruby, Shell, Lua, Perl, PHP have per-file syntax checks
        // (e.g. `ruby -c`, `shellcheck --shell`, `luac -p`, `perl -c`, `php -l`).
        assert!(fast_check_for("app.rb").is_some());
        assert!(fast_check_for("deploy.sh").is_some());
        assert!(fast_check_for("init.lua").is_some());
        assert!(fast_check_for("script.pl").is_some());
        assert!(fast_check_for("page.php").is_some());
        // Rust has no reliable per-file fast check (cargo check is project-wide
        // and already the turn-end verify) → None.
        assert!(fast_check_for("src/lib.rs").is_none());
        // Unknown extension → None.
        assert!(fast_check_for("README.md").is_none());
        assert!(fast_check_for("noext").is_none());
    }

    #[test]
    fn read_schema_requires_a_single_path() {
        let read = TOOL_SPECS
            .iter()
            .find(|s| s.name == "read")
            .expect("read tool present");
        let params = &read.parameters;
        // `path` is required and unambiguous — no `paths`/empty-required schema
        // that measurably degrades small-model tool-calling.
        assert_eq!(params["required"], serde_json::json!(["path"]));
        let props = params["properties"].as_object().unwrap();
        assert!(props.contains_key("path"));
        assert!(!props.contains_key("paths"));
    }






}
