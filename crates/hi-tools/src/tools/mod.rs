//! Tool execution dispatch, process tools, and shared runtime helpers.
//!
//! File mutations live in [`mutations`]. Advertised specs live in [`crate::catalog`].

mod commit;
mod external;
mod mutations;
mod process_tools;

pub use commit::{CommitOutcome, commit_in, commit_in_typed};

pub use external::{
    McpBackend, McpToolInfo, MemoryBackend, MemorySearchResult, SkillBackend, run_memory_forget,
    run_memory_get, run_memory_search, run_memory_update, run_search_tool, run_skill, run_use_tool,
};
#[cfg(test)]
pub(crate) use mutations::preview_edit_in;
pub use mutations::{
    MAX_WRITE_OVERWRITE_BYTES, PreparedMutation, execute_prepared_in_runtime,
    prepare_mutation_in_with_state,
};

pub(crate) use process_tools::kill_group;

use process_tools::{BashArgs, run_bash_tool};

pub use crate::catalog::{
    MINIMAL_TOOL_SPECS, PROTECTED_TOOLS, SpeculationClass, TOOL_CATALOG, TOOL_SPECS, ToolAdmission,
    ToolCapability, ToolCostClass, ToolMetadata, ask_user_tool_spec, browser_exec_tool_spec,
    delegate_tool_spec, explore_tool_spec, get_task_output_tool_spec, is_coordination,
    is_filesystem_mutating, is_known_tool, is_read_only, kill_task_tool_spec,
    memory_forget_tool_spec, memory_get_tool_spec, memory_search_tool_spec,
    memory_update_tool_spec, new_context_tool_spec, research_read_tool_spec, research_tool_spec,
    run_program_tool_spec, search_tool_tool_spec, skill_tool_spec, speculation_class, target_path,
    target_paths, task_tool_spec, tool_metadata, use_tool_tool_spec, wait_tasks_tool_spec,
};

use mutations::run_prepared_mutation;

use std::io::Read;
use std::path::Path;
use std::time::Duration;

use crate::condense::condense;
use crate::read::{run_glob, run_grep_with_runner, run_list};
use crate::{PlanStatus, PlanStep, ProcessRunner, ToolEffects, ToolOutcome};
use anyhow::{Context, Result, bail};
use serde::Deserialize;

pub(crate) fn check_timeout_from_value(value: Option<&str>) -> Option<Duration> {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map(Duration::from_secs)
        .filter(|timeout| std::time::Instant::now().checked_add(*timeout).is_some())
}

/// Optional verification wall-clock cap. A positive
/// `HI_VERIFY_TIMEOUT_SECS` opts in; unset, invalid, or zero leaves productive
/// verification active until completion or caller cancellation.
pub fn check_timeout() -> Option<Duration> {
    let configured = std::env::var("HI_VERIFY_TIMEOUT_SECS").ok();
    check_timeout_from_value(configured.as_deref())
}
const MAX_UNTRACKED_DIFF_ENTRIES: usize = 200;
const MAX_CREATED_DIFF_FILE_BYTES: usize = 16 * 1024;
const MAX_CREATED_DIFF_TOTAL_BYTES: usize = 64 * 1024;
pub async fn run_check_in(
    root: &std::path::Path,
    command: &str,
) -> Result<crate::ProcessExecution> {
    let runner = ProcessRunner::new(root)?;
    run_check_in_with_runner(&runner, command).await
}

/// Run a verification command through an already-configured process runner.
///
/// Embedded agents must use this variant so verification inherits the
/// workspace's explicit sandbox policy instead of re-reading the process
/// environment and silently selecting a different runner.
pub async fn run_check_in_with_runner(
    runner: &ProcessRunner,
    command: &str,
) -> Result<crate::ProcessExecution> {
    run_check_in_with_runner_maybe_timeout(runner, command, check_timeout()).await
}

/// [`run_check_in`] with an explicit budget. Verification uses this for its
/// one cold-build retry: a stage that timed out gets a single re-run with a
/// doubled budget before the turn is declared unverifiable.
pub async fn run_check_in_with_timeout(
    root: &std::path::Path,
    command: &str,
    timeout: std::time::Duration,
) -> Result<crate::ProcessExecution> {
    let runner = ProcessRunner::new(root)?;
    run_check_in_with_runner_timeout(&runner, command, timeout).await
}

/// [`run_check_in_with_timeout`] using the caller's configured process runner.
pub async fn run_check_in_with_runner_timeout(
    runner: &ProcessRunner,
    command: &str,
    timeout: std::time::Duration,
) -> Result<crate::ProcessExecution> {
    run_check_in_with_runner_maybe_timeout(runner, command, Some(timeout)).await
}

/// Run a verification command with an optional operator deadline.
///
/// `None` keeps productive work alive until completion or caller
/// cancellation; dropping the future still kills the entire process group.
pub async fn run_check_in_with_runner_maybe_timeout(
    runner: &ProcessRunner,
    command: &str,
    timeout: Option<std::time::Duration>,
) -> Result<crate::ProcessExecution> {
    // `__pycache__` cleanup only matters for Python. Cargo/go/npm stages would
    // otherwise pay a full-tree walk before every verify command — and that walk
    // runs on the agent future the TUI co-polls, so it freezes the UI.
    if command_needs_pycache_cleanup(command) {
        let root_for_cleanup = runner.root().to_path_buf();
        let _ =
            tokio::task::spawn_blocking(move || prepare_verify_workdir(&root_for_cleanup)).await;
    }
    runner.run_shell_maybe_timeout(command, timeout).await
}

fn command_needs_pycache_cleanup(command: &str) -> bool {
    let lower = command.to_ascii_lowercase();
    lower.contains("python")
        || lower.contains("pytest")
        || lower.contains("uvicorn")
        || lower.contains("django")
        || lower.contains("py_compile")
}

/// Best-effort cleanup before running a verification command.
///
/// Python's import cache can otherwise make same-size, same-second edits look
/// unchanged to `python -c "import solution"` checks. Pruning only `__pycache__`
/// directories keeps this narrow and harmless for non-Python checks.
///
/// Large generated trees (`target/`, `node_modules/`, weight caches, …) are
/// skipped so this cannot stall on multi-gigabyte workspaces.
pub fn prepare_verify_workdir(dir: &std::path::Path) {
    fn should_prune_dir(name: &std::ffi::OsStr) -> bool {
        matches!(
            name.to_str(),
            Some(
                ".git"
                    | ".hg"
                    | ".svn"
                    | ".jj"
                    | ".cargo-home"
                    | "target"
                    | "node_modules"
                    | "vendor"
                    | ".venv"
                    | "venv"
                    | "dist"
                    | "build"
                    | ".next"
                    | ".turbo"
                    | "coverage"
                    | ".cache"
                    | "hi-test-scratch"
            )
        ) || name.to_str().is_some_and(|name| {
            name.starts_with(".venv-")
                || name.starts_with("venv-")
                || name.starts_with("node_modules-")
        })
    }

    fn is_weight_cache(relative: &std::path::Path) -> bool {
        let mut components = relative
            .components()
            .filter_map(|component| match component {
                std::path::Component::Normal(name) => name.to_str(),
                _ => None,
            });
        matches!(
            (components.next(), components.next()),
            (Some("models"), _) | (Some(".hi"), Some("models"))
        )
    }

    fn walk(dir: &std::path::Path, relative: &std::path::Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let path = entry.path();
            let child_relative = relative.join(&name);
            if name == "__pycache__" {
                // Inspect without following symlinks. A workspace may contain
                // a symlink named `__pycache__`; removing through it must not
                // touch an external directory, and following symlinked dirs
                // below would make this cleanup recurse outside the workspace.
                if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                    let _ = std::fs::remove_dir_all(&path);
                }
                continue;
            }
            if !entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
                continue;
            }
            if is_weight_cache(&child_relative) || should_prune_dir(&name) {
                continue;
            }
            walk(&path, &child_relative);
        }
    }
    walk(dir, std::path::Path::new(""));
}

/// A per-file "fast check" command for `path`'s language — a quick, file-scoped
/// syntax/lint check that can run in the background right after an edit, so a
/// type/syntax error surfaces while the edit is still the model's focus rather
/// than at turn-end verify. Returns `None` for languages without a genuinely
/// per-file fast check (e.g. Rust and TypeScript, whose checks are project-wide
/// and are already handled by affected-package verification) or for unrecognized
/// extensions. The command is run as an argument-vector process with the file
/// path appended. Launch and check failures are non-fatal — no early signal is
/// better than a wrong one.
pub fn fast_check_for(path: &str) -> Option<&'static str> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())?;
    match ext {
        // Python: py_compile catches syntax errors per-file, fast.
        "py" => Some("python3 -m py_compile"),
        // Go: gofmt -l lists files that aren't formatted / have syntax issues.
        "go" => Some("gofmt -l"),
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
    run_fast_check_in_maybe_timeout(root, check, path, check_timeout()).await
}

async fn run_fast_check_in_maybe_timeout(
    root: &Path,
    check: &str,
    path: &Path,
    timeout: Option<Duration>,
) -> (bool, String) {
    use std::ffi::OsString;

    let path_arg = path.as_os_str().to_os_string();
    let (program, args): (&str, Vec<OsString>) = match check {
        "python3 -m py_compile" => (
            // `py_compile` writes bytecode even when
            // PYTHONDONTWRITEBYTECODE is set. Compile the source in memory
            // instead so a fast check cannot mutate a cache outside the
            // workspace or fail under a read-only macOS Python cache.
            "python3",
            vec![
                OsString::from("-c"),
                OsString::from(
                    "import pathlib,sys; compile(pathlib.Path(sys.argv[1]).read_text(), sys.argv[1], 'exec')",
                ),
                path_arg,
            ],
        ),
        "gofmt -l" => ("gofmt", vec![OsString::from("-l"), path_arg]),
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
        .run_program_maybe_timeout(program, &args, timeout)
        .await
    {
        Ok(execution) => (
            fast_check_passed(check, &execution),
            execution.model_content(),
        ),
        Err(error) => (false, format!("fast check failed to start: {error:#}")),
    }
}

fn fast_check_passed(check: &str, execution: &crate::ProcessExecution) -> bool {
    // `gofmt -l` reports unformatted files on stdout but exits 0. Treat that
    // diagnostic as a failed fast check instead of silently claiming the edit
    // is clean.
    execution.status == crate::ToolStatus::Succeeded
        && (check != "gofmt -l" || execution.outcome.stdout_summary.trim().is_empty())
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
    let tracked = match run_git_read(root, color, &["--no-pager", "diff", "HEAD"]).await {
        Ok(out) if out.status == crate::ToolStatus::Succeeded => out.outcome.stdout_summary,
        Ok(out) => {
            let stderr = out.outcome.stderr_summary;
            // Fresh repo with no commits yet: diff against the empty tree instead.
            if stderr.contains("unknown revision") || stderr.contains("ambiguous argument") {
                run_git_read(root, color, &["--no-pager", "diff"])
                    .await
                    .ok()
                    .filter(|out| out.status == crate::ToolStatus::Succeeded)
                    .map(|out| out.outcome.stdout_summary)
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

    let untracked = run_git_read(
        root,
        color,
        &["ls-files", "--others", "--exclude-standard", "-z"],
    )
    .await
    .ok()
    .filter(|out| out.status == crate::ToolStatus::Succeeded)
    .map(|out| out.outcome.stdout_summary)
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
            "vendor"
                | "node_modules"
                | ".cargo-home"
                | "target"
                | "dist"
                | "build"
                | "coverage"
                | "generated"
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

async fn run_git_operation(root: &Path, args: Vec<String>) -> Result<crate::ProcessExecution> {
    run_git_operation_maybe_timeout(root, args, None).await
}

async fn run_git_operation_maybe_timeout(
    root: &Path,
    args: Vec<String>,
    timeout: Option<Duration>,
) -> Result<crate::ProcessExecution> {
    ProcessRunner::new(root)?
        .run_program_maybe_timeout("git", args, timeout)
        .await
}

async fn run_git_read(root: &Path, color: bool, args: &[&str]) -> Result<crate::ProcessExecution> {
    let mut command = Vec::with_capacity(args.len() + 2);
    if color {
        command.extend(["-c".to_string(), "color.ui=always".to_string()]);
    }
    command.extend(args.iter().map(|arg| (*arg).to_string()));
    run_git_operation(root, command).await
}

/// Execute a tool by name. Tool failures are returned as content (not errors)
/// so the model sees them and can recover, rather than aborting the turn.
#[derive(Clone, Copy)]
pub(super) struct RuntimeResources<'a> {
    /// The live agent supplies its already-resolved runner here. Compatibility
    /// entry points leave it unset and retain environment-based construction.
    pub(super) process_runner: Option<&'a ProcessRunner>,
    pub(super) lsp: &'a std::sync::Arc<hi_lsp::LspManager>,
    pub(super) background: &'a crate::BackgroundRegistry,
    pub(super) read_cache: &'a std::sync::Mutex<crate::ReadCache>,
    pub(super) repo_map: &'a std::sync::Mutex<crate::RepoMapCache>,
    /// Owned handle used to move repository indexing to a blocking worker.
    /// Compatibility callers only provide the borrowed cache above.
    pub(super) repo_map_arc: Option<&'a std::sync::Arc<std::sync::Mutex<crate::RepoMapCache>>>,
    pub(super) mcp: Option<&'a dyn external::McpBackend>,
    pub(super) memory: Option<&'a dyn external::MemoryBackend>,
    pub(super) skill: Option<&'a dyn external::SkillBackend>,
    /// Optional hunk-tracker for agent-edit attribution. When present,
    /// file mutations are recorded so the session can track which hunks
    /// the agent edited (for review/undo/LOC accounting).
    pub(super) hunk_tracker: Option<&'a hi_hunk_tracker::HunkTrackerHandle>,
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
    let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
    let background = crate::BackgroundRegistry::default();
    let read_cache = std::sync::Mutex::new(crate::ReadCache::new());
    let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
    let outcome = execute_in_impl(
        &root,
        &state,
        RuntimeResources {
            process_runner: None,
            lsp: &lsp,
            background: &background,
            read_cache: &read_cache,
            repo_map: &repo_map,
            repo_map_arc: None,
            mcp: None,
            memory: None,
            skill: None,
            hunk_tracker: None,
        },
        name,
        arguments,
    )
    .await;
    let _ = std::fs::remove_dir_all(state);
    outcome
}

#[allow(
    clippy::too_many_arguments,
    reason = "this compatibility facade mirrors the runtime resources supplied by callers"
)]
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
    execute_in_runtime_with(
        root, state_root, lsp, background, read_cache, repo_map, None, None, None, name, arguments,
    )
    .await
}

/// Runtime facade for agents that own the shared repository-map handle. The
/// shared form lets expensive first-use indexing run on `spawn_blocking` while
/// preserving the cache across concurrent main/side-channel tool calls.
#[allow(clippy::too_many_arguments)]
pub async fn execute_in_runtime_shared(
    root: &Path,
    state_root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    background: &crate::BackgroundRegistry,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    repo_map: &std::sync::Arc<std::sync::Mutex<crate::RepoMapCache>>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    execute_in_runtime_shared_with(
        root, state_root, lsp, background, read_cache, repo_map, None, None, name, arguments,
    )
    .await
}

/// Like [`execute_in_runtime_shared`] with optional MCP and markdown memory backends.
#[allow(clippy::too_many_arguments)]
pub async fn execute_in_runtime_shared_with(
    root: &Path,
    state_root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    background: &crate::BackgroundRegistry,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    repo_map: &std::sync::Arc<std::sync::Mutex<crate::RepoMapCache>>,
    mcp: Option<&dyn external::McpBackend>,
    memory: Option<&dyn external::MemoryBackend>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    execute_in_impl(
        root,
        state_root,
        RuntimeResources {
            process_runner: None,
            lsp,
            background,
            read_cache,
            repo_map: repo_map.as_ref(),
            repo_map_arc: Some(repo_map),
            mcp,
            memory,
            skill: None,
            hunk_tracker: None,
        },
        name,
        arguments,
    )
    .await
}

/// Like [`execute_in_runtime_shared_with`] but uses the caller's already
/// configured process runner. Live agents use this path so shell tools and
/// verification cannot silently resolve different sandbox policies.
#[allow(clippy::too_many_arguments)]
pub async fn execute_in_runtime_shared_with_runner(
    runner: &ProcessRunner,
    root: &Path,
    state_root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    background: &crate::BackgroundRegistry,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    repo_map: &std::sync::Arc<std::sync::Mutex<crate::RepoMapCache>>,
    mcp: Option<&dyn external::McpBackend>,
    memory: Option<&dyn external::MemoryBackend>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    execute_in_impl(
        root,
        state_root,
        RuntimeResources {
            process_runner: Some(runner),
            lsp,
            background,
            read_cache,
            repo_map: repo_map.as_ref(),
            repo_map_arc: Some(repo_map),
            mcp,
            memory,
            skill: None,
            hunk_tracker: None,
        },
        name,
        arguments,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn execute_in_runtime_with(
    root: &Path,
    state_root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    background: &crate::BackgroundRegistry,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    repo_map: &std::sync::Mutex<crate::RepoMapCache>,
    mcp: Option<&dyn external::McpBackend>,
    memory: Option<&dyn external::MemoryBackend>,
    skill: Option<&dyn external::SkillBackend>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    execute_in_runtime_with_hunks(
        root, state_root, lsp, background, read_cache, repo_map, mcp, memory, skill, None, name,
        arguments,
    )
    .await
}

/// Like [`execute_in_runtime_with`] but also accepts an optional hunk-tracker
/// handle so file mutations can be attributed to the agent.
#[allow(clippy::too_many_arguments)]
pub async fn execute_in_runtime_with_hunks(
    root: &Path,
    state_root: &Path,
    lsp: &std::sync::Arc<hi_lsp::LspManager>,
    background: &crate::BackgroundRegistry,
    read_cache: &std::sync::Mutex<crate::ReadCache>,
    repo_map: &std::sync::Mutex<crate::RepoMapCache>,
    mcp: Option<&dyn external::McpBackend>,
    memory: Option<&dyn external::MemoryBackend>,
    skill: Option<&dyn external::SkillBackend>,
    hunk_tracker: Option<&hi_hunk_tracker::HunkTrackerHandle>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    execute_in_impl(
        root,
        state_root,
        RuntimeResources {
            process_runner: None,
            lsp,
            background,
            read_cache,
            repo_map,
            repo_map_arc: None,
            mcp,
            memory,
            skill,
            hunk_tracker,
        },
        name,
        arguments,
    )
    .await
}

async fn execute_in_impl(
    root: &Path,
    state_root: &Path,
    resources: RuntimeResources<'_>,
    name: &str,
    arguments: &str,
) -> ToolOutcome {
    let mut outcome = match run(root, state_root, resources, name, arguments).await {
        Ok(output) => output,
        Err(err) => {
            let mut outcome = ToolOutcome::failed(format!("Error: {err:#}"));
            outcome.effects.mutation_attempted = mutation_attempted_by_tool(name);
            outcome
        }
    };
    redact_tool_output(&mut outcome);
    outcome
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
            process_runner: None,
            lsp,
            background,
            read_cache,
            repo_map,
            repo_map_arc: None,
            mcp: None,
            memory: None,
            skill: None,
            hunk_tracker: None,
        },
        name,
        arguments,
        on_line,
    )
    .await
}

/// Streaming counterpart to [`execute_in_runtime_shared_with_runner`].
#[allow(clippy::too_many_arguments)]
pub async fn execute_streaming_in_runtime_with_runner(
    runner: &ProcessRunner,
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
            process_runner: Some(runner),
            lsp,
            background,
            read_cache,
            repo_map,
            repo_map_arc: None,
            mcp: None,
            memory: None,
            skill: None,
            hunk_tracker: None,
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
    let mut outcome =
        match run_streaming(root, state_root, resources, name, arguments, on_line).await {
            Ok(output) => output,
            Err(err) => {
                let mut outcome = ToolOutcome::failed(format!("Error: {err:#}"));
                outcome.effects.mutation_attempted = mutation_attempted_by_tool(name);
                outcome
            }
        };
    redact_tool_output(&mut outcome);
    outcome
}

/// Final model/UI boundary for tool content. Individual handlers can return
/// concise `plain` results, process summaries, or structured backend output;
/// keeping redaction here prevents a newly added handler from accidentally
/// bypassing secret scrubbing.
fn redact_tool_output(outcome: &mut ToolOutcome) {
    outcome.content = hi_secrets::redact_secrets(&outcome.content).into_owned();
    // Keep the final boundary defensive: not every handler is naturally a
    // file/process result, and a newly added handler must not send an
    // arbitrarily large payload into the next model request. Preserve metadata
    // from handlers that already bounded their content.
    let ceiling = crate::read::result_char_budget(&outcome.content);
    if matches!(outcome.truncation, crate::TruncationState::Complete)
        && outcome.content.chars().count() > ceiling
    {
        let (content, truncation) = crate::bound_tool_content(std::mem::take(&mut outcome.content));
        outcome.content = content;
        outcome.truncation = truncation;
    }
    if let Some(display) = outcome.display.as_mut() {
        *display = hi_secrets::redact_secrets(display).into_owned();
    }
}

const MAX_LSP_SYNC_BYTES: u64 = 16 * 1024 * 1024;

/// Sync a current source document with LSP without loading an unbounded file
/// into the agent task. Oversized files are left to the server's existing state
/// rather than freezing the UI on a giant generated artifact.
async fn sync_lsp_document(path: &Path, lsp: &std::sync::Arc<hi_lsp::LspManager>) {
    let Ok(metadata) = tokio::fs::metadata(path).await else {
        return;
    };
    if metadata.len() > MAX_LSP_SYNC_BYTES {
        return;
    }
    if let Ok(text) = tokio::fs::read_to_string(path).await {
        let _ = lsp.sync_document(path, &text).await;
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
    if name == "bash_output" {
        return run_bash_output(resources, arguments, on_line).await;
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
        "read" => {
            crate::read::run_read_with_mcp(root, resources.read_cache, resources.mcp, arguments)
                .await
        }
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
            // Titles ride in leftover-plan steering and the structured goal, so
            // bound each title's payload. Preserve every submitted step: silently
            // dropping the tail can make a long-running plan settle early.
            const MAX_PLAN_TITLE_CHARS: usize = 160;
            let steps: Vec<PlanStep> = args
                .steps
                .into_iter()
                .map(|s| PlanStep {
                    title: clip_plan_title(&s.title, MAX_PLAN_TITLE_CHARS),
                    status: PlanStatus::parse(&s.status),
                })
                .collect();
            let done = steps
                .iter()
                .filter(|s| s.status == PlanStatus::Done)
                .count();
            let content = format!("Plan recorded: {done}/{} done.", steps.len());
            Ok(ToolOutcome::planned(content, steps))
        }
        "write" | "edit" | "multi_edit" | "apply_patch" => {
            let prepared =
                prepare_mutation_in_with_state(root, state_root, name, arguments).await?;
            run_prepared_mutation(
                resources.lsp,
                resources.read_cache,
                resources.hunk_tracker,
                prepared,
            )
            .await
        }
        "bash" => {
            let args: BashArgs = parse(arguments)?;
            // Read-cache invalidation lives inside run_bash_tool, so both this
            // dispatch path and the streaming path (execute_streaming) clear it.
            run_bash_tool(root, state_root, resources, args, &mut |_| {}).await
        }
        "bash_output" => run_bash_output(resources, arguments, &mut |_| {}).await,
        "bash_kill" => {
            #[derive(Deserialize)]
            struct Args {
                id: String,
            }
            let args: Args = parse(arguments)?;
            let result = resources.background.kill_and_reap(&args.id).await?;
            let background = resources.background.outcome(&args.id)?;
            if let Ok(mut cache) = resources.read_cache.lock() {
                cache.clear();
            }
            let mut outcome = background_tool_outcome(result, background);
            attach_background_effects(&mut outcome, resources.background, &args.id).await;
            Ok(outcome)
        }
        "list" => run_list(root, arguments).await,
        "repo_map" => match resources.repo_map_arc {
            Some(repo_map) => {
                crate::repo_map::run_repo_map_shared(root, (*repo_map).clone(), arguments).await
            }
            None => crate::repo_map::run_repo_map(root, resources.repo_map, arguments).await,
        },
        "find_symbol" => match resources.repo_map_arc {
            Some(repo_map) => {
                crate::repo_map::run_find_symbol_shared(root, (*repo_map).clone(), arguments).await
            }
            None => crate::repo_map::run_find_symbol(root, resources.repo_map, arguments).await,
        },
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
        "grep" => run_grep_with_runner(root, resources.process_runner, arguments).await,
        "diagnostics" => run_lsp_diagnostics(root, resources.lsp, arguments).await,
        "definition" => run_lsp_definition(root, resources.lsp, arguments).await,
        "references" => run_lsp_references(root, resources.lsp, arguments).await,
        "hover" => run_lsp_hover(root, resources.lsp, arguments).await,
        "web_search" => crate::web::run_web_search(arguments).await,
        "web_fetch" => crate::web::run_web_fetch(arguments).await,
        "research" => crate::research::run_research(arguments).await,
        "research_read" => crate::research::run_research_read(arguments).await,
        "web_download" => {
            crate::web::run_web_download_in(root, resources.background, arguments).await
        }
        "search_tool" => external::run_search_tool(resources.mcp, arguments).await,
        "use_tool" => external::run_use_tool(resources.mcp, arguments).await,
        "browser_exec" => run_browser_exec(arguments).await,
        "memory_search" => external::run_memory_search(resources.memory, arguments).await,
        "memory_get" => external::run_memory_get(resources.memory, arguments).await,
        "memory_update" => external::run_memory_update(resources.memory, arguments).await,
        "memory_forget" => external::run_memory_forget(resources.memory, arguments).await,
        "skill" => external::run_skill(resources.skill, arguments),
        other => bail!("unknown tool: {other}"),
    }
}

async fn run_browser_exec(arguments: &str) -> Result<ToolOutcome> {
    let result = hi_browser::run_exec(arguments).await?;
    let mut outcome = ToolOutcome::plain(result.text);
    outcome.images = result
        .images
        .into_iter()
        .map(|image| crate::ToolImage {
            data: image.data,
            media_type: image.media_type,
        })
        .collect();
    Ok(outcome)
}

fn mutation_attempted_by_tool(name: &str) -> bool {
    is_filesystem_mutating(name) || name == "bash"
}

fn clip_plan_title(title: &str, max: usize) -> String {
    let title = title.trim();
    if title.chars().count() <= max {
        return title.to_string();
    }
    let clipped: String = title.chars().take(max.saturating_sub(1)).collect();
    format!("{clipped}…")
}

pub(super) fn mark_effect_inspection_failed(
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
pub(super) async fn attach_background_effects(
    outcome: &mut ToolOutcome,
    background: &crate::BackgroundRegistry,
    id: &str,
) {
    // A running process has no stable after-snapshot yet. Polling its output
    // used to hash the entire workspace on every `bash_output`, even when the
    // process was quiet. Terminal polls and the turn's final reconciliation
    // still capture the eventual effects once the process has stopped.
    if outcome.background.as_ref().is_some_and(|state| {
        matches!(
            state.state,
            crate::BackgroundState::Started | crate::BackgroundState::Running
        )
    }) {
        return;
    }
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

async fn run_bash_output(
    resources: RuntimeResources<'_>,
    arguments: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<ToolOutcome> {
    #[derive(Deserialize)]
    struct Args {
        id: String,
        /// `None` (omitted) → adaptive default wait. `Some(0)` → instant peek.
        /// `Some(n)` → park up to `n` seconds (capped at 600).
        wait_secs: Option<u64>,
    }
    let args: Args = parse(arguments)?;
    let result = match args.wait_secs {
        None => {
            resources
                .background
                .poll_wait_default_streaming(&args.id, on_line)
                .await?
        }
        Some(0) => resources.background.poll(&args.id)?,
        Some(secs) => {
            resources
                .background
                .poll_wait_streaming(
                    &args.id,
                    std::time::Duration::from_secs(secs.min(600)),
                    on_line,
                )
                .await?
        }
    };
    let background = resources.background.outcome(&args.id)?;
    if let Ok(mut cache) = resources.read_cache.lock() {
        cache.clear();
    }
    // Background polls use the same dual-path contract as foreground process
    // results: keep ANSI for the UI, but never persist it in model content.
    let display = condense(&result);
    let content = condense(&crate::process::strip_ansi(&result));
    let mut outcome = background_tool_outcome(content, background);
    if display != outcome.content {
        outcome.display = Some(display);
    }
    attach_background_effects(&mut outcome, resources.background, &args.id).await;
    Ok(outcome)
}

pub(super) fn background_tool_outcome(
    content: String,
    background: crate::BackgroundOutcome,
) -> ToolOutcome {
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
    sync_lsp_document(&path, lsp).await;
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
    #[derive(Deserialize)]
    struct Args {
        path: String,
        line: u32,
        column: u32,
    }
    let args: Args = parse(arguments)?;
    let path = crate::transaction::resolve_workspace_target(root, Path::new(&args.path))?;
    if !lsp.is_enabled().await {
        // LSP is off — fall back to the codebase-graph index if available.
        if let Some(locs) =
            crate::codebase_graph::query(root, &args.path, args.line, args.column, kind).await
        {
            if locs.is_empty() {
                return Ok(ToolOutcome::plain(format!("No {kind} found.")));
            }
            return Ok(ToolOutcome::plain(locs.join("\n")));
        }
        return Ok(ToolOutcome::plain("LSP is off (use `/lsp on`).".into()));
    }
    sync_lsp_document(&path, lsp).await;
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
    sync_lsp_document(&path, lsp).await;
    match lsp.hover(&path, args.line, args.column).await? {
        Some(text) => Ok(ToolOutcome::plain(text)),
        None => Ok(ToolOutcome::plain("No hover info.".into())),
    }
}

pub(crate) fn parse<T: for<'de> Deserialize<'de>>(arguments: &str) -> Result<T> {
    serde_json::from_str(arguments).context("invalid tool arguments")
}

#[cfg(test)]
mod tests {
    use super::mutations::is_retryable_edit_miss;
    use super::process_tools::{
        BashArgs, auto_background_enabled_from_value, foreground_interactive_command_reason,
        foreground_interactive_command_reason_at, run_bash_streaming_with_timeout,
        run_bash_tool_with_auto_background,
    };
    use super::{
        MAX_WRITE_OVERWRITE_BYTES, RuntimeResources, TOOL_SPECS, check_timeout_from_value,
        commit_in_typed, execute_in, fast_check_for, fast_check_passed, redact_tool_output,
        render_untracked_files, render_untracked_files_with_contents, run_check_in,
        run_fast_check_in_maybe_timeout, run_git_operation_maybe_timeout,
        working_tree_diff_plain_in,
    };
    use crate::edit::{apply_edit, sh_quote};
    use crate::paths::cache_key;
    use std::time::Duration;

    #[test]
    fn verification_timeout_is_an_explicit_positive_opt_in() {
        assert_eq!(check_timeout_from_value(None), None);
        assert_eq!(check_timeout_from_value(Some("0")), None);
        assert_eq!(check_timeout_from_value(Some("invalid")), None);
        assert_eq!(
            check_timeout_from_value(Some("11")),
            Some(Duration::from_secs(11))
        );
    }

    #[tokio::test]
    async fn fast_check_has_no_default_deadline_but_accepts_an_explicit_one() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .map(|output| !output.status.success())
            .unwrap_or(true)
        {
            eprintln!("skipping: python3 not on PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "hi-fast-check-deadline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("valid.py");
        std::fs::write(&path, "answer = 42\n").unwrap();

        let (passed, output) =
            run_fast_check_in_maybe_timeout(&dir, "python3 -m py_compile", &path, None).await;
        assert!(passed, "default-unlimited fast check failed: {output}");

        let (passed, output) = run_fast_check_in_maybe_timeout(
            &dir,
            "python3 -m py_compile",
            &path,
            Some(Duration::ZERO),
        )
        .await;
        assert!(!passed, "an explicit zero-duration deadline must fire");
        assert!(output.contains("timed out"), "{output}");
        let _ = std::fs::remove_dir_all(dir);
    }

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

    fn init_commit_test_repo() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hi-commit-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let status = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .status()
            .unwrap();
        assert!(status.success());
        for (key, value) in [("user.email", "t@t"), ("user.name", "t")] {
            let status = std::process::Command::new("git")
                .args(["config", key, value])
                .current_dir(&dir)
                .status()
                .unwrap();
            assert!(status.success());
        }
        dir
    }

    fn git_in(dir: &std::path::Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success(), "git {args:?} failed in {}", dir.display());
    }

    fn git_stdout(dir: &std::path::Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?} failed");
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn internal_git_deadline_is_optional_and_explicit() {
        let dir = init_commit_test_repo();
        let completed = tokio::time::timeout(
            Duration::from_secs(2),
            run_git_operation_maybe_timeout(
                &dir,
                vec![
                    "-c".into(),
                    "alias.pause=!sleep 0.05".into(),
                    "pause".into(),
                ],
                None,
            ),
        )
        .await
        .expect("default-unlimited git operation should complete")
        .unwrap();
        assert_eq!(completed.status, crate::ToolStatus::Succeeded);

        let timed_out = run_git_operation_maybe_timeout(
            &dir,
            vec!["-c".into(), "alias.pause=!sleep 1".into(), "pause".into()],
            Some(Duration::from_millis(25)),
        )
        .await
        .unwrap();
        assert_eq!(timed_out.status, crate::ToolStatus::TimedOut);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn commit_in_stages_only_ledger_paths() {
        let dir = init_commit_test_repo();
        std::fs::write(dir.join("keep.txt"), "baseline keep\n").unwrap();
        std::fs::write(dir.join("skip.txt"), "baseline skip\n").unwrap();
        git_in(&dir, &["add", "keep.txt", "skip.txt"]);
        git_in(&dir, &["commit", "-qm", "baseline"]);
        std::fs::write(dir.join("keep.txt"), "session keep\n").unwrap();
        std::fs::write(dir.join("skip.txt"), "dirty extra\n").unwrap();
        std::fs::write(dir.join("untracked.txt"), "not in ledger\n").unwrap();

        let outcome = commit_in_typed(&dir, &["keep.txt".into()]).await;
        let status = git_stdout(&dir, &["status", "--porcelain"]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(outcome.status, crate::ToolStatus::Succeeded);
        assert!(outcome.content.contains("committed:"), "{outcome:?}");
        assert!(outcome.workspace_may_have_changed);
        assert!(outcome.external_effect_may_have_occurred);
        assert!(
            status.contains("skip.txt"),
            "extra dirty file must stay uncommitted: {status}"
        );
        assert!(
            status.contains("untracked.txt") || status.contains("?? untracked.txt"),
            "untracked extra must stay unstaged: {status}"
        );
        assert!(
            !status.contains("keep.txt"),
            "ledger path committed: {status}"
        );
    }

    #[tokio::test]
    async fn commit_in_refuses_empty_ledger() {
        let dir = init_commit_test_repo();
        std::fs::write(dir.join("keep.txt"), "baseline\n").unwrap();
        git_in(&dir, &["add", "keep.txt"]);
        git_in(&dir, &["commit", "-qm", "baseline"]);
        std::fs::write(dir.join("keep.txt"), "dirty\n").unwrap();

        let outcome = commit_in_typed(&dir, &[]).await;
        let status = git_stdout(&dir, &["status", "--porcelain"]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(outcome.status, crate::ToolStatus::Failed);
        assert!(
            outcome.content.contains("nothing this session changed"),
            "{outcome:?}"
        );
        assert!(!outcome.workspace_may_have_changed);
        assert!(!outcome.external_effect_may_have_occurred);
        assert!(status.contains("keep.txt"), "must not stage: {status}");
    }

    #[tokio::test]
    async fn commit_in_refuses_secret_in_staged_diff() {
        let dir = init_commit_test_repo();
        std::fs::write(dir.join("secret.env"), "placeholder\n").unwrap();
        git_in(&dir, &["add", "secret.env"]);
        git_in(&dir, &["commit", "-qm", "baseline"]);
        std::fs::write(
            dir.join("secret.env"),
            "api_key=sk-abcdefghijklmnopqrstuvwxyz123456\n",
        )
        .unwrap();

        let outcome = commit_in_typed(&dir, &["secret.env".into()]).await;
        let cached = git_stdout(&dir, &["diff", "--cached", "--name-only"]);
        let log = git_stdout(&dir, &["log", "--oneline"]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(outcome.status, crate::ToolStatus::Failed);
        assert!(outcome.content.contains("secrets"), "{outcome:?}");
        assert!(outcome.workspace_may_have_changed);
        assert!(outcome.external_effect_may_have_occurred);
        assert!(
            cached.trim().is_empty(),
            "secret path must be unstaged: {cached}"
        );
        assert!(
            !log.contains("update secret.env"),
            "must not create a commit: {log}"
        );
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
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&dir).unwrap());
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

    #[test]
    fn numbered_read_under_read_budget_is_not_head_tailed() {
        // A SPEC.md-sized numbered page is well under the 64k read budget but
        // over the shared 5k cap. Clipping it at 5k hid the middle of the spec,
        // then skip-reread treated the file as complete.
        let body: String = (1..=300)
            .map(|i| format!("{i:>4}\t## section {i} unique-marker-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chars = body.chars().count();
        assert!(
            chars > *crate::condense::MAX_OUTPUT_CHARS,
            "fixture must exceed the shared cap ({chars} chars)"
        );
        assert!(
            chars < crate::read::read_output_budget(),
            "fixture must fit in the read budget ({chars} chars)"
        );

        let (content, truncation) = crate::bound_tool_content(body.clone());

        assert_eq!(content, body, "numbered read pages must not be head-tailed");
        assert_eq!(truncation, crate::TruncationState::Complete);
        assert!(
            content.contains("unique-marker-150"),
            "middle of the spec must remain visible to the model"
        );
        assert!(
            !content.contains("truncated"),
            "must not inject a truncation marker: {content}"
        );
    }

    #[test]
    fn final_tool_boundary_bounds_plain_handler_output() {
        let max = *crate::condense::MAX_OUTPUT_CHARS;
        let mut outcome = crate::ToolOutcome::plain("x".repeat(max + 1));

        redact_tool_output(&mut outcome);

        assert!(outcome.content.chars().count() > max);
        assert!(outcome.content.contains("truncated"));
        assert!(matches!(
            outcome.truncation,
            crate::TruncationState::Truncated { .. }
        ));
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

    #[test]
    fn bash_auto_background_is_an_explicit_opt_in() {
        assert!(!auto_background_enabled_from_value(None));
        assert!(!auto_background_enabled_from_value(Some("")));
        assert!(!auto_background_enabled_from_value(Some("0")));
        assert!(!auto_background_enabled_from_value(Some("false")));
        assert!(!auto_background_enabled_from_value(Some("unexpected")));
        assert!(auto_background_enabled_from_value(Some("1")));
        assert!(auto_background_enabled_from_value(Some(" true ")));
        assert!(auto_background_enabled_from_value(Some("YES")));
        assert!(auto_background_enabled_from_value(Some("on")));
    }

    /// Explicit auto-background-on-timeout: a foreground command still running
    /// at its budget is moved to the background (handle returned) instead of
    /// killed. Injecting policy keeps this deterministic and avoids mutating the
    /// process-global environment while other tests execute.
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
    async fn explicitly_enabled_bash_auto_backgrounds_instead_of_killing() {
        let (root, state) = isolated_ws("bg");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        let runner = crate::ProcessRunner::new(&root).unwrap();
        // timeout:1 → foreground budget is 1s; a 600s sleep outlasts it.
        let outcome = run_bash_tool_with_auto_background(
            &root,
            &state,
            RuntimeResources {
                process_runner: Some(&runner),
                lsp: &lsp,
                background: &background,
                read_cache: &cache,
                repo_map: &repo_map,
                repo_map_arc: None,
                mcp: None,
                memory: None,
                skill: None,
                hunk_tracker: None,
            },
            BashArgs {
                command: "sleep 600".into(),
                timeout: Some(1),
                run_in_background: false,
            },
            &mut |_| {},
            true,
        )
        .await
        .unwrap();
        assert!(
            outcome.content.contains("continued as")
                || outcome.content.contains("still running after"),
            "not killed — backgrounded: {:?}",
            outcome.content
        );
        assert!(
            !outcome.content.contains('{') && !outcome.content.contains('}'),
            "user/model-facing start text must not embed JSON: {:?}",
            outcome.content
        );
        let bg = outcome.background.expect("a background handle is returned");
        assert_eq!(bg.state, crate::BackgroundState::Started);
        // Handles carry a command-derived name, not an opaque `sh_N`.
        assert!(bg.id.starts_with("sleep_"), "got: {}", bg.id);
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
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
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

    #[tokio::test]
    async fn bash_cat_of_a_workspace_file_returns_a_numbered_read() {
        // Live: DeepSeek Flash `cat SPEC.md` / `sed -n` which the 5k bash
        // condenser head-and-tailed, hiding Phase 1. File dumps now go
        // through `read` so the middle of a spec-sized file survives.
        let dir = std::env::temp_dir().join(format!(
            "hi-file-dump-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut body = String::new();
        for i in 1..=250 {
            if i == 125 {
                body.push_str("PHASE-1-UNIQUE-MARKER\n");
            } else {
                body.push_str(&format!("line-{i}-padding-aaaaaaaaaaaaaaaa\n"));
            }
        }
        std::fs::write(dir.join("SPEC.md"), &body).unwrap();
        let outcome = execute_in(&dir, "bash", r#"{"command":"cat SPEC.md"}"#).await;
        assert_eq!(outcome.status, crate::ToolStatus::Succeeded);
        assert!(
            outcome.content.contains("PHASE-1-UNIQUE-MARKER"),
            "middle of dump was clipped ({} chars): {}",
            outcome.content.chars().count(),
            outcome.content
        );
        assert!(
            outcome.content.contains('\t'),
            "expected a numbered read page, got: {}",
            outcome.content.chars().take(80).collect::<String>()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A shell command can mutate any file, so `bash` must invalidate the read
    /// cache — otherwise a later `read` serves stale pre-bash content.
    #[tokio::test]
    async fn bash_invalidates_the_read_cache() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let state = root.join(".hi-test-state");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(root).unwrap());
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
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let state = root.join(".hi-test-state");
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(root).unwrap());
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
    async fn mutations_reject_internal_elision_placeholder_as_content() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        std::fs::create_dir_all(&state).unwrap();
        let args = serde_json::json!({
            "path": "index.html",
            "content": "[elided — 4467 chars]"
        })
        .to_string();

        let error = crate::prepare_mutation_in_with_state(temp.path(), &state, "write", &args)
            .await
            .expect_err("compaction metadata must never become file content");

        assert!(format!("{error:#}").contains("transcript-elision"));
        assert!(!temp.path().join("index.html").exists());
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
    async fn prepared_approval_path_names_actual_single_target() {
        let dir = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("src")).unwrap();
        std::fs::create_dir(dir.path().join("private")).unwrap();
        std::fs::write(dir.path().join("private/config.txt"), "original\n").unwrap();

        for requested in ["src/../private/config.txt", "private/config.txt"] {
            let args = serde_json::json!({"path": requested, "content": "original\n"});
            let prepared = crate::prepare_mutation_in_with_state(
                dir.path(),
                state.path(),
                "write",
                &args.to_string(),
            )
            .await
            .unwrap();
            assert_eq!(
                prepared.single_target_path().as_deref(),
                Some("private/config.txt")
            );
        }

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("../private/config.txt", dir.path().join("src/alias.txt"))
                .unwrap();
            let prepared = crate::prepare_mutation_in_with_state(
                dir.path(),
                state.path(),
                "write",
                r#"{"path":"src/alias.txt","content":"updated\n"}"#,
            )
            .await
            .unwrap();
            assert_eq!(
                prepared.single_target_path().as_deref(),
                Some("private/config.txt")
            );
        }

        let single = serde_json::json!({"patch": "*** Begin Patch\n*** Update File: src/../private/config.txt\n-original\n+updated\n*** End Patch"});
        let prepared = crate::prepare_mutation_in_with_state(
            dir.path(),
            state.path(),
            "apply_patch",
            &single.to_string(),
        )
        .await
        .unwrap();
        assert_eq!(
            prepared.single_target_path().as_deref(),
            Some("private/config.txt")
        );

        let multiple = serde_json::json!({"patch": "*** Begin Patch\n*** Add File: first.txt\n+first\n*** Add File: second.txt\n+second\n*** End Patch"});
        let prepared = crate::prepare_mutation_in_with_state(
            dir.path(),
            state.path(),
            "apply_patch",
            &multiple.to_string(),
        )
        .await
        .unwrap();
        assert_eq!(prepared.single_target_path(), None);
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
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&dir).unwrap());
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
                // Wait for a *parseable* pid, not just the file's existence —
                // the shell creates the file before flushing `$!` into it, and
                // cancelling the future in that window can tear the write so
                // the later read sees empty content.
                read_pid_when_ready(&pid_file).await;
            };

            tokio::select! {
                result = &mut fut => panic!("command finished before cancellation: {result:?}"),
                _ = child_started => {}
            }
        }

        // The pid was confirmed durable by child_started; read it directly.
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
    async fn preserved_policy_lets_foreground_bash_leave_detached_services() {
        let _guard = crate::background::TEST_LOCK.lock().await;
        crate::preserve_detached_descendants(true);
        let pid_file = std::env::temp_dir().join(format!(
            "hi-fg-bash-keep-{}-{}.pid",
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
        crate::preserve_detached_descendants(false);
        assert!(out.contains("done"), "got: {out:?}");

        let pid: i32 = read_pid_when_ready(&pid_file).await;
        // Give any (unwanted) group kill time to land before asserting life.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let alive = process_exists(pid);
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
        let _ = std::fs::remove_file(&pid_file);
        assert!(
            alive,
            "under the preserve policy a detached service must outlive the foreground command"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn foreground_bash_completion_kills_detached_descendants() {
        let _guard = crate::background::TEST_LOCK.lock().await;
        crate::preserve_detached_descendants(false);
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

        let pid: i32 = read_pid_when_ready(&pid_file).await;
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

    /// Read a pid file the shell is writing asynchronously. The shell creates
    /// the file before the pid is flushed, so poll for parseable content
    /// rather than reading once — a bare read can observe the empty file
    /// mid-write and fail the parse.
    #[cfg(unix)]
    async fn read_pid_when_ready(pid_file: &std::path::Path) -> i32 {
        for _ in 0..100 {
            if let Ok(contents) = std::fs::read_to_string(pid_file)
                && let Ok(pid) = contents.trim().parse::<i32>()
            {
                return pid;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("pid file {} never became readable", pid_file.display())
    }

    #[test]
    fn bash_timeout_resolution_has_no_implicit_or_maximum_ceiling() {
        use super::process_tools::resolve_bash_timeout_from_values;
        assert_eq!(resolve_bash_timeout_from_values(None, None), None);
        assert_eq!(resolve_bash_timeout_from_values(None, Some("0")), None);
        assert_eq!(
            resolve_bash_timeout_from_values(None, Some("invalid")),
            None
        );
        assert_eq!(
            resolve_bash_timeout_from_values(None, Some("86400")).map(|value| value.as_secs()),
            Some(86_400)
        );
        // Explicit positive requests are honored without a one-hour clamp.
        assert_eq!(
            resolve_bash_timeout_from_values(Some(86_400), Some("1")).map(|value| value.as_secs()),
            Some(86_400)
        );
        // Zero explicitly selects the same continual mode as omission.
        assert_eq!(resolve_bash_timeout_from_values(Some(0), Some("1")), None);
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
        // TS/JS are checked once per affected package; launching a project-wide
        // tsc once per edited file made edit-heavy turns needlessly slow.
        assert!(fast_check_for("x.ts").is_none());
        assert!(fast_check_for("x.jsx").is_none());
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
    fn gofmt_listing_is_not_treated_as_a_clean_check() {
        let execution = crate::ProcessExecution {
            status: crate::ToolStatus::Succeeded,
            outcome: crate::ProcessOutcome {
                exit_code: Some(0),
                stdout_summary: "src/main.go\n".into(),
                stderr_summary: String::new(),
                duration_ms: 1,
            },
            truncation: crate::TruncationState::Complete,
        };
        assert!(!fast_check_passed("gofmt -l", &execution));
        assert!(fast_check_passed("python3 -m py_compile", &execution));
    }

    #[test]
    fn read_schema_requires_a_single_or_multi_path() {
        let read = TOOL_SPECS
            .iter()
            .find(|s| s.name == "read")
            .expect("read tool present");
        let params = &read.parameters;
        // Exactly one of `path`/`paths` is required so the executor's batched
        // read path is available without accepting an ambiguous empty object.
        assert!(params["oneOf"].is_array());
        let props = params["properties"].as_object().unwrap();
        assert!(props.contains_key("path"));
        assert!(props.contains_key("paths"));
    }

    fn diagnostic(severity: &str, message: &str) -> hi_lsp::Diagnostic {
        hi_lsp::Diagnostic {
            severity: severity.into(),
            line: 0,
            col: 0,
            message: message.into(),
            source: None,
        }
    }

    #[tokio::test]
    async fn write_attaches_injected_lsp_diagnostics() {
        let (root, state) = isolated_ws("lsp-write");
        std::fs::create_dir_all(root.join("src")).unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
        lsp.inject_diagnostics(
            lsp.root().join("src/lib.rs"),
            hi_lsp::DiagnosticState::DiagnosticsPresent {
                document_version: 1,
                diagnostics: vec![diagnostic("error", "injected boom")],
            },
        );
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
            "write",
            r#"{"path":"src/lib.rs","content":"fn x() {}\n"}"#,
        )
        .await;
        assert_eq!(
            outcome.status,
            crate::ToolStatus::Succeeded,
            "{}",
            outcome.content
        );
        assert!(
            outcome.content.contains("<diagnostics>") && outcome.content.contains("injected boom"),
            "{}",
            outcome.content
        );
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test]
    async fn edit_attaches_injected_lsp_diagnostics() {
        let (root, state) = isolated_ws("lsp-edit");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "fn x() {}\n").unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
        lsp.inject_diagnostics(
            lsp.root().join("src/lib.rs"),
            hi_lsp::DiagnosticState::DiagnosticsPresent {
                document_version: 1,
                diagnostics: vec![diagnostic("error", "edit boom")],
            },
        );
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
            "edit",
            r#"{"path":"src/lib.rs","old_string":"fn x() {}","new_string":"fn y() {}"}"#,
        )
        .await;
        assert_eq!(
            outcome.status,
            crate::ToolStatus::Succeeded,
            "{}",
            outcome.content
        );
        assert!(
            outcome.content.contains("<diagnostics>") && outcome.content.contains("edit boom"),
            "{}",
            outcome.content
        );
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test]
    async fn write_lsp_timeout_does_not_fail_the_tool() {
        let (root, state) = isolated_ws("lsp-timeout");
        std::fs::create_dir_all(root.join("src")).unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
        lsp.inject_diagnostics_with_delay(
            lsp.root().join("src/lib.rs"),
            hi_lsp::DiagnosticState::DiagnosticsPresent {
                document_version: 1,
                diagnostics: vec![diagnostic("error", "too late")],
            },
            Duration::from_secs(5),
        );
        let background = crate::BackgroundRegistry::default();
        let cache = std::sync::Mutex::new(crate::ReadCache::new());
        let repo_map = std::sync::Mutex::new(crate::RepoMapCache::new());
        let started = std::time::Instant::now();
        let outcome = crate::execute_in_runtime(
            &root,
            &state,
            &lsp,
            &background,
            &cache,
            &repo_map,
            "write",
            r#"{"path":"src/lib.rs","content":"fn x() {}\n"}"#,
        )
        .await;
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "timeout should drop the wait, elapsed {:?}",
            started.elapsed()
        );
        assert_eq!(
            outcome.status,
            crate::ToolStatus::Succeeded,
            "{}",
            outcome.content
        );
        assert!(
            !outcome.content.contains("<diagnostics>") && !outcome.content.contains("too late"),
            "{}",
            outcome.content
        );
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test]
    async fn mutation_diagnostics_include_edited_warnings_and_sibling_errors_only() {
        let (root, state) = isolated_ws("lsp-sib");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/other.rs"), "fn other() {}\n").unwrap();
        let lsp = std::sync::Arc::new(hi_lsp::LspManager::new(&root).unwrap());
        lsp.inject_diagnostics(
            lsp.root().join("src/lib.rs"),
            hi_lsp::DiagnosticState::DiagnosticsPresent {
                document_version: 1,
                diagnostics: vec![diagnostic("warning", "edited warning")],
            },
        );
        lsp.inject_diagnostics(
            lsp.root().join("src/other.rs"),
            hi_lsp::DiagnosticState::DiagnosticsPresent {
                document_version: 1,
                diagnostics: vec![
                    diagnostic("error", "sibling error"),
                    diagnostic("warning", "sibling warning"),
                ],
            },
        );
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
            "write",
            r#"{"path":"src/lib.rs","content":"fn x() {}\n"}"#,
        )
        .await;
        assert_eq!(
            outcome.status,
            crate::ToolStatus::Succeeded,
            "{}",
            outcome.content
        );
        assert!(
            outcome.content.contains("edited warning"),
            "{}",
            outcome.content
        );
        assert!(
            outcome.content.contains("sibling error"),
            "{}",
            outcome.content
        );
        assert!(
            !outcome.content.contains("sibling warning"),
            "{}",
            outcome.content
        );
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[tokio::test]
    async fn browser_exec_disabled_fails_closed() {
        hi_browser::configure(hi_browser::BrowserConfig {
            enabled: false,
            allow_private: false,
        });
        let out = crate::execute("browser_exec", r#"{"script":"goto https://example.com"}"#).await;
        hi_browser::configure(hi_browser::BrowserConfig::default());
        assert_eq!(out.status, crate::ToolStatus::Failed, "{}", out.content);
        assert!(
            out.content.to_ascii_lowercase().contains("disabled"),
            "{}",
            out.content
        );
    }
}
