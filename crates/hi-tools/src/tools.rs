use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

use hi_ai::ToolSpec;

use crate::condense::condense;
use crate::edit::{apply_edit, apply_multi_patch, diff};
use crate::paths::{READ_CACHE, cache_key, validate_workspace_path};
use crate::read::{run_glob, run_grep, run_list, run_read};
use crate::{PlanStatus, PlanStep, ToolOutput};

/// Default wall-clock limit for a single `bash` command, used when neither the
/// caller nor `HI_BASH_TIMEOUT_SECS` overrides it. Generous enough for a real
/// `cargo test`/build verify step, bounded so a genuine hang recovers on its own.
const DEFAULT_BASH_TIMEOUT_SECS: u64 = 600;
/// Hard ceiling on any per-command timeout (model- or env-supplied) so a bad
/// value can't reintroduce an unbounded stall.
const MAX_BASH_TIMEOUT_SECS: u64 = 3600;
const CHECK_TIMEOUT: Duration = Duration::from_secs(300);

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

/// Run a verification command (e.g. a test suite) and report `(passed, output)`
/// based on its exit status. Used by the agent's verification loop.
pub async fn run_check(command: &str) -> (bool, String) {
    let future = Command::new("sh").arg("-c").arg(command).output();
    match tokio::time::timeout(CHECK_TIMEOUT, future).await {
        Ok(Ok(output)) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
                text.push_str(&stderr);
            }
            (output.status.success(), condense(&text))
        }
        Ok(Err(err)) => (false, format!("failed to run verification: {err}")),
        Err(_) => (
            false,
            format!("verification timed out after {}s", CHECK_TIMEOUT.as_secs()),
        ),
    }
}

/// A per-file "fast check" command for `path`'s language — a quick, file-scoped
/// syntax/lint check that can run in the background right after an edit, so a
/// type/syntax error surfaces while the edit is still the model's focus rather
/// than at turn-end verify. Returns `None` for languages without a genuinely
/// per-file fast check (e.g. Rust, whose `cargo check` is project-wide and is
/// already the turn-end verify) or for unrecognized extensions. The command is
/// run via `sh -c` with the file path appended; callers should only run it when
/// the relevant tool is on PATH (probe with [`tool_available`] in
/// `read.rs`). Failures are non-fatal — no early signal is better than a wrong
/// one, so unknown/missing tools yield `None`.
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

/// A human-readable, ANSI-colored summary of what's changed in the working
/// tree versus the last commit — the body of the `/diff` command. Tracked
/// changes come from `git diff HEAD`; new files the agent created are listed by
/// name (their contents aren't in `git diff`). Returns a friendly message when
/// the cwd isn't a git repo or there's nothing to show.
pub async fn working_tree_diff() -> String {
    working_tree_diff_impl(true).await
}

/// Same as [`working_tree_diff`] but without ANSI color codes — for the `diff`
/// tool, so the model gets plain text it can parse.
pub async fn working_tree_diff_plain() -> String {
    working_tree_diff_impl(false).await
}

async fn working_tree_diff_impl(color: bool) -> String {
    let git = |args: &'static [&'static str]| async move {
        let mut cmd = Command::new("git");
        cmd.args(args);
        if color {
            cmd.arg("-c").arg("color.ui=always");
        }
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
            } else {
                return format!(
                    "not a git repository (or git unavailable): {}",
                    stderr.trim()
                );
            }
        }
        Err(err) => return format!("git not available: {err}"),
    };

    let untracked = git(&["ls-files", "--others", "--exclude-standard"])
        .await
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let new_files: Vec<&str> = untracked.lines().filter(|l| !l.trim().is_empty()).collect();

    if tracked.trim().is_empty() && new_files.is_empty() {
        return "no changes since HEAD".to_string();
    }

    let mut out = String::new();
    if !tracked.trim().is_empty() {
        out.push_str(tracked.trim_end());
        out.push('\n');
    }
    if !new_files.is_empty() {
        out.push_str("\nnew (untracked) files:\n");
        for f in new_files {
            out.push_str(&format!("  + {f}\n"));
        }
    }
    out
}

/// Stage all working-tree changes and commit them with an auto-generated
/// message summarizing the changed files. This is the body of the `/commit`
/// slash command. Returns a single human-readable progress line (the message
/// used), or an error message when there's nothing to commit or this isn't a
/// git repo. Phase order: `git add -A` → read the staged diff stat →
/// `git commit -m "<message>"`.
pub async fn commit() -> String {
    // 1. Confirm we're inside a work tree before touching anything.
    let in_tree = match Command::new("git")
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
    let add = match Command::new("git").args(["add", "-A"]).output().await {
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

/// The tools advertised to the model each turn.
fn build_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "update_plan".into(),
            description: "Record or update a short task plan, shown to the user as a live checklist. Call it when starting a task that takes several steps — pass the full ordered list of steps — then call it again as you progress, ALWAYS passing the complete list with updated statuses (mark the step you're on `active`, finished steps `done`). Keep titles to a few words. Skip it for trivial one-step tasks.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "The full ordered list of plan steps, resubmitted in its entirety on every call.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": { "type": "string", "description": "Short description of the step (a few words)." },
                                "status": { "type": "string", "enum": ["pending", "active", "done"], "description": "pending (not started), active (in progress now), or done." }
                            },
                            "required": ["title", "status"]
                        }
                    }
                },
                "required": ["steps"]
            }),
        },
        ToolSpec {
            name: "record_decision".into(),
            description: "Record a key design decision so it persists across context compaction and keeps later turns consistent. Call this when you commit to an approach, a convention, or a non-obvious tradeoff (e.g. 'using a BTreeMap for ordered iteration', 'skipping Windows support for now'). Kept verbatim in the system prompt — NOT summarized away — so a long refactor doesn't drift from its own rationale. Use sparingly: only for decisions that matter later.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "summary": { "type": "string", "description": "A short title of the decision (one line)." },
                    "rationale": { "type": "string", "description": "Why this choice — the constraint or tradeoff that drove it." },
                    "files": {
                        "type": "array",
                        "description": "Files the decision most affects (may be empty).",
                        "items": { "type": "string" }
                    }
                },
                "required": ["summary", "rationale"]
            }),
        },
        ToolSpec {
            name: "read".into(),
            description: "Read a UTF-8 text file. Lines are returned numbered (`<n>\\t<text>`). Returns at most 240 lines by default; page with offset/limit instead of assuming you saw everything.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read." },
                    "offset": { "type": "integer", "description": "1-based line to start at (default: first line)." },
                    "limit": { "type": "integer", "description": "Maximum number of lines to return (default: 240)." }
                },
                "required": ["path"]
            }),
        },
        ToolSpec {
            name: "write".into(),
            description: "Create or overwrite a file with the given content. Parent directories are created as needed.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to write." },
                    "content": { "type": "string", "description": "Full content to write." }
                },
                "required": ["path", "content"]
            }),
        },
        ToolSpec {
            name: "edit".into(),
            description: "Replace a unique block of text in a file. old_string must occur once and be the file's literal text WITHOUT the `read` line-number gutter; whitespace and indentation differences are tolerated. Set replace_all=true to replace every occurrence (use with care).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit." },
                    "old_string": { "type": "string", "description": "Exact text to replace; must be unique in the file unless replace_all is set. Do not include line numbers." },
                    "new_string": { "type": "string", "description": "Replacement text." },
                    "replace_all": { "type": "boolean", "description": "If true, replace every occurrence of old_string (default: false, requires uniqueness)." }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        },
        ToolSpec {
            name: "multi_edit".into(),
            description: "Apply several edits to one file atomically, in order. Each edit replaces a unique block (same rules as `edit`); if any fails, none are applied. Prefer this over multiple `edit` calls on the same file.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit." },
                    "edits": {
                        "type": "array",
                        "description": "Edits applied in sequence to the file's evolving content.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": { "type": "string", "description": "Exact text to replace; unique at the time this edit applies. No line numbers." },
                                "new_string": { "type": "string", "description": "Replacement text." }
                            },
                            "required": ["old_string", "new_string"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }),
        },
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command via `sh -c` in the current working directory and return combined stdout/stderr. stdin is closed, so commands never block on input. A foreground command that exceeds its timeout is killed (whole process tree) and reports what it printed so far. For a long-lived or blocking process (a dev server, a file watcher, `tail -f`), set run_in_background:true — it returns a handle id immediately; read its output with bash_output and stop it with bash_kill. For a slow but finite build or test suite, just raise `timeout` instead.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run." },
                    "timeout": { "type": "integer", "description": "Optional wall-clock limit in seconds (default 600, max 3600). Raise it for a slow test/build suite. Ignored when run_in_background is true." },
                    "run_in_background": { "type": "boolean", "description": "Run detached and return a handle id immediately instead of waiting for the command to exit. Use for servers/watchers/long-lived processes." }
                },
                "required": ["command"]
            }),
        },
        ToolSpec {
            name: "bash_output".into(),
            description: "Read new output (stdout+stderr) from a background process started by `bash` with run_in_background, since the last read. Also reports whether it is still running, exited (with code), or was killed. Returns immediately.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The background process handle returned by bash (e.g. `bg_1`)." }
                },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "bash_kill".into(),
            description: "Stop a background process (and its whole process tree) started by `bash` with run_in_background. Idempotent.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "The background process handle to kill (e.g. `bg_1`)." }
                },
                "required": ["id"]
            }),
        },
        ToolSpec {
            name: "list".into(),
            description: "List the project's files (respecting .gitignore), optionally under a subpath. Use this first to get the lay of the codebase before reading files.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory to list, relative to the project root (default: the whole project)." }
                }
            }),
        },
        ToolSpec {
            name: "diff".into(),
            description: "Show what's changed in the working tree versus the last commit (tracked changes as a diff, plus a list of new untracked files). Use this to review your own edits before finishing.".into(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolSpec {
            name: "grep".into(),
            description: "Search file contents for a regular expression (ripgrep if available, else grep), respecting .gitignore. Returns matching `path:line: text`. Use this to find where something is defined or used. Pass `context` to see surrounding lines. Pass `glob` to filter by file name pattern (e.g. `*.rs`).".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regular expression to search for." },
                    "path": { "type": "string", "description": "File or directory to search (default: the whole project)." },
                    "context": { "type": "integer", "description": "Lines of context to show around each match (default: 0)." },
                    "glob": { "type": "string", "description": "File name glob to filter (e.g. `*.rs`, `*.py`). Only files whose name matches are searched." }
                },
                "required": ["pattern"]
            }),
        },
        ToolSpec {
            name: "glob".into(),
            description: "Find files by name pattern (e.g. `**/*.rs`, `src/*.py`). Respects .gitignore. Returns matching paths, up to 500 results.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern to match file paths (e.g. `**/*.rs`, `*.py`)." },
                    "path": { "type": "string", "description": "Directory to search in (default: the whole project)." }
                },
                "required": ["pattern"]
            }),
        },
        ToolSpec {
            name: "apply_patch".into(),
            description: "Apply a multi-file patch. Use for coordinated edits across several files at once. Format: '*** Begin Patch\\n*** Update File: path\\n@@ context @\\n-old\\n+new\\n unchanged\\n*** End Patch'. Also supports '*** Add File: path' and '*** Delete File: path'.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "patch": { "type": "string", "description": "The patch text in Begin/End Patch format." }
                },
                "required": ["patch"]
            }),
        },
    ]
}

/// The tool specifications advertised to the model, cached once.
pub static TOOL_SPECS: LazyLock<Vec<ToolSpec>> = LazyLock::new(build_tool_specs);

/// Whether a tool only observes state, with no side effects — so several can
/// run concurrently within one round, and it's safe to offer in `ReadOnly`
/// tool mode. Tools that mutate the filesystem (`write`, `edit`, `multi_edit`,
/// `apply_patch`) or have ordering-sensitive external effects (`bash`,
/// `bash_kill`) are excluded. `update_plan` and `record_decision` have no
/// side effects beyond in-memory state, so they're read-only here.
/// `bash_output` is a pure poll of an existing buffer.
pub fn is_read_only(name: &str) -> bool {
    matches!(
        name,
        "read" | "list" | "grep" | "glob" | "diff" | "update_plan" | "record_decision"
            | "bash_output"
    )
}

/// Whether a tool mutates the working tree — so the agent should invalidate its
/// snapshot cache and kick off a proactive fast-check after it runs. This is a
/// narrower set than `!is_read_only`: `bash` can mutate files but is handled
/// separately (it always runs alone), and `bash_kill`/`update_plan`/
/// `record_decision` have no filesystem effect even though they're not
/// read-only for parallelization purposes.
pub fn is_filesystem_mutating(name: &str) -> bool {
    matches!(name, "write" | "edit" | "multi_edit" | "apply_patch")
}

/// Best-effort extraction of the primary target path from a tool call's JSON
/// arguments — the `path` field for read/write/edit/list, the `path`/`glob` for
/// grep. Returns `None` for tools without a meaningful single path (e.g.
/// `bash`, or a `grep` with only a pattern). Used by the agent to infer
/// within-batch dependencies: a read of a file a mutating call earlier in the
/// same batch targeted should observe that mutation, so it's serialized after.
/// Tolerant — a failed parse yields `None`, which the caller treats as "no
/// dependency inferred" (safe fallback to emission order).
pub fn target_path(name: &str, arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    match name {
        // read/write/edit/multi_edit carry an explicit `path`.
        "read" | "write" | "edit" | "multi_edit" => value.get("path")?.as_str().map(str::to_string),
        // list's path is optional (defaults to ".").
        "list" => value.get("path")?.as_str().map(str::to_string),
        // grep: prefer an explicit `path`; fall back to `glob` only as a hint
        // (a glob isn't a single file, so return None to avoid over-serializing).
        "grep" => value.get("path")?.as_str().map(str::to_string),
        // apply_patch: the patch text contains `*** Update File: <path>` (or
        // `*** Add File:`/`*** Delete File:`) directives. Return the first path
        // so a read of that file after the patch serializes. Multi-file patches
        // only expose the first path — the safe fallback (serialize after all
        // earlier mutations) covers the rest.
        "apply_patch" => {
            let patch = value.get("patch")?.as_str()?;
            patch.lines().find_map(|l| {
                l.trim()
                    .strip_prefix("*** Update File: ")
                    .or_else(|| l.trim().strip_prefix("*** Add File: "))
                    .map(|p| p.trim().to_string())
            })
        }
        // diff/glob/bash: no single meaningful target path for dep inference.
        _ => None,
    }
}

/// Execute a tool by name. Tool failures are returned as content (not errors)
/// so the model sees them and can recover, rather than aborting the turn.
pub async fn execute(name: &str, arguments: &str) -> ToolOutput {
    match run(name, arguments).await {
        Ok(output) => output,
        Err(err) => ToolOutput::plain(format!("Error: {err:#}")),
    }
}

/// Execute a tool by name, streaming `bash` output line-by-line through `on_line`
/// so the UI can show progress in real time. Other tools behave identically to
/// [`execute`] — `on_line` is only called for `bash`.
pub async fn execute_streaming(
    name: &str,
    arguments: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> ToolOutput {
    match run_streaming(name, arguments, on_line).await {
        Ok(output) => output,
        Err(err) => ToolOutput::plain(format!("Error: {err:#}")),
    }
}

async fn run_streaming(
    name: &str,
    arguments: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<ToolOutput> {
    if name == "bash" {
        let args: BashArgs = parse(arguments)?;
        return run_bash_tool(args, on_line).await;
    }
    // All other tools: delegate to the normal path (on_line unused).
    run(name, arguments).await
}

async fn run(name: &str, arguments: &str) -> Result<ToolOutput> {
    match name {
        "read" => run_read(arguments).await,
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
            Ok(ToolOutput::planned(
                format!("Plan recorded: {done}/{} done.", steps.len()),
                steps,
            ))
        }
        "write" => {
            let args: WriteArgs = parse(arguments)?;
            validate_workspace_path(&args.path)?;
            if let Some(parent) = Path::new(&args.path).parent()
                && !parent.as_os_str().is_empty()
            {
                tokio::fs::create_dir_all(parent)
                    .await
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            let before = tokio::fs::read_to_string(&args.path)
                .await
                .unwrap_or_default();
            tokio::fs::write(&args.path, &args.content)
                .await
                .with_context(|| format!("writing {}", args.path))?;
            // Invalidate the read cache for this path after a write.
            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(&cache_key(&args.path));
            }
            let written_len = args.content.len();
            let after = crate::read::maybe_format(&args.path, args.content).await;
            let msg = if after.len() != written_len {
                format!(
                    "Wrote {} bytes to {} (formatter adjusted to {} bytes)",
                    written_len,
                    args.path,
                    after.len()
                )
            } else {
                format!("Wrote {} bytes to {}", written_len, args.path)
            };
            Ok(ToolOutput::shown(msg, diff(&before, &after)))
        }
        "edit" => {
            let args: EditArgs = parse(arguments)?;
            validate_workspace_path(&args.path)?;
            let before = crate::read::read_text_file(&args.path).await?;
            let after = apply_edit(
                &before,
                &args.old_string,
                &args.new_string,
                args.replace_all,
            )
            .with_context(|| format!("editing {}", args.path))?;
            tokio::fs::write(&args.path, &after)
                .await
                .with_context(|| format!("writing {}", args.path))?;
            // Invalidate the read cache for this path after an edit.
            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(&cache_key(&args.path));
            }
            let after = crate::read::maybe_format(&args.path, after).await;
            let count = if args.replace_all {
                before.matches(&args.old_string).count()
            } else {
                1
            };
            Ok(ToolOutput::shown(
                if args.replace_all && count > 1 {
                    format!("Replaced {count} occurrences in {}", args.path)
                } else {
                    format!("Edited {}", args.path)
                },
                diff(&before, &after),
            ))
        }
        "multi_edit" => {
            let args: MultiEditArgs = parse(arguments)?;
            validate_workspace_path(&args.path)?;
            if args.edits.is_empty() {
                bail!("no edits provided");
            }
            let before = crate::read::read_text_file(&args.path).await?;
            // Apply edits in order against the evolving content; abort (writing
            // nothing) if any fails, so a partial multi-edit never lands.
            let mut after = before.clone();
            for (i, e) in args.edits.iter().enumerate() {
                after = apply_edit(&after, &e.old_string, &e.new_string, false)
                    .with_context(|| format!("editing {} (edit #{})", args.path, i + 1))?;
            }
            tokio::fs::write(&args.path, &after)
                .await
                .with_context(|| format!("writing {}", args.path))?;
            // Invalidate the read cache for this path after a multi-edit.
            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(&cache_key(&args.path));
            }
            let after = crate::read::maybe_format(&args.path, after).await;
            Ok(ToolOutput::shown(
                format!("Applied {} edits to {}", args.edits.len(), args.path),
                diff(&before, &after),
            ))
        }
        "bash" => {
            let args: BashArgs = parse(arguments)?;
            run_bash_tool(args, &mut |_| {}).await
        }
        "bash_output" => {
            #[derive(Deserialize)]
            struct Args {
                id: String,
            }
            let args: Args = parse(arguments)?;
            Ok(ToolOutput::plain(condense(&crate::background::poll(
                &args.id,
            )?)))
        }
        "bash_kill" => {
            #[derive(Deserialize)]
            struct Args {
                id: String,
            }
            let args: Args = parse(arguments)?;
            Ok(ToolOutput::plain(crate::background::kill(&args.id)?))
        }
        "list" => run_list(arguments).await,
        "diff" => {
            // Reuse the working-tree diff summary, but return it as model content
            // (plain text, no ANSI) so the model can review what changed.
            Ok(ToolOutput::plain(working_tree_diff_plain().await))
        }
        "glob" => run_glob(arguments).await,
        "grep" => run_grep(arguments).await,
        "apply_patch" => {
            #[derive(Deserialize)]
            struct PatchArgs {
                patch: String,
            }
            let args: PatchArgs = parse(arguments)?;
            let result = apply_multi_patch(&args.patch).await?;
            Ok(ToolOutput::plain(result))
        }
        other => bail!("unknown tool: {other}"),
    }
}

/// Spawn `sh -c <command>` hardened for unattended use: stdin detached (so it
/// never blocks on input), stdout/stderr piped, its own process group (so the
/// whole tree can be killed), and kill-on-drop. Shared by the foreground and
/// background bash paths so both behave identically.
pub(crate) fn spawn_shell(command: &str) -> Result<tokio::process::Child> {
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(command)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    cmd.spawn().context("failed to spawn command")
}

/// SIGKILL an entire process group by its id. We spawn with `process_group(0)`,
/// so a process's group id equals its pid; signalling the negative pid reaches
/// every descendant. No-op on non-Unix (where `child.kill()` is the best we have).
#[cfg(unix)]
pub(crate) fn kill_group(pgid: i32) {
    // SAFETY: `kill(2)` with a negative pid targets the process group; it has no
    // memory-safety implications and a stale pid simply errors out.
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
pub(crate) fn kill_group(_pgid: i32) {}

/// SIGKILL the child's whole process group, if it hasn't been reaped yet.
#[cfg(unix)]
fn kill_process_group(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        kill_group(pid as i32);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_child: &tokio::process::Child) {}

pub(crate) async fn run_bash(command: &str) -> Result<String> {
    run_bash_streaming(command, &mut |_| {}).await
}

/// Run a shell command, calling `on_line` for each line of output as it arrives
/// (both stdout and stderr). The final assembled output is still returned for the
/// model. Lines are delivered with a trailing newline.
pub(crate) async fn run_bash_streaming(
    command: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<String> {
    run_bash_streaming_with_timeout(command, on_line, resolve_bash_timeout(None)).await
}

async fn run_bash_streaming_with_timeout(
    command: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
    bash_timeout: Duration,
) -> Result<String> {
    // Refuse the handful of irreversible operations a checkpoint can't undo.
    if let Some(reason) = crate::guard::catastrophic_op(command) {
        return Ok(format!(
            "⚠ refused: this command {reason}. It's blocked as irreversible — the per-turn \
             checkpoint can't undo it. If it's genuinely needed, ask the user to run it \
             themselves (they can also set HI_ALLOW_DANGEROUS=1 to disable this guard)."
        ));
    }
    // Hardened spawn (detached stdin, piped output, own process group) so a
    // timeout can SIGKILL the *whole tree* — cargo, the test binary, any leaked
    // daemon — not just the `sh` parent, and nothing blocks waiting on input.
    let mut child = spawn_shell(command)?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Use a shared Mutex<&mut dyn FnMut> so both stdout and stderr readers can
    // call on_line without double-borrowing.
    let cb: &std::sync::Mutex<&mut (dyn FnMut(&str) + Send)> = &std::sync::Mutex::new(on_line);
    // Accumulate lines in outer-scope buffers so partial output survives a
    // timeout (the futures that fill them get dropped when the timeout fires).
    let stdout_buf = std::sync::Mutex::new(Vec::<String>::new());
    let stderr_buf = std::sync::Mutex::new(Vec::<String>::new());

    // The timeout MUST wrap pipe-draining *and* the wait together: a process
    // that hangs while holding its pipes open (a deadlocked test, something
    // waiting on a socket/stdin, or a leaked child inheriting the stdout fd)
    // never reaches EOF, so reading alone would block forever. Wrapping only
    // `child.wait()` — reached after the pipes drain — would never arm at all.
    let combined = async {
        tokio::join!(
            read_lines(stdout, cb, &stdout_buf),
            read_lines(stderr, cb, &stderr_buf),
        );
        child.wait().await
    };
    let status = match tokio::time::timeout(bash_timeout, combined).await {
        Ok(Ok(status)) => Some(status),
        Ok(Err(err)) => bail!("command failed: {err}"),
        Err(_) => {
            // `combined` (holding the &mut child borrow) is dropped by `timeout`
            // before it returns Err, so the child is free to kill here. Kill the
            // whole group first (orphaned grandchildren), then reap `sh` itself.
            kill_process_group(&child);
            let _ = child.kill().await;
            None
        }
    };

    let stdout_lines = stdout_buf.into_inner().unwrap_or_default();
    let stderr_lines = stderr_buf.into_inner().unwrap_or_default();
    let mut out = String::new();
    for line in &stdout_lines {
        out.push_str(line);
    }
    for line in &stderr_lines {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(line);
    }
    match status {
        Some(status) => {
            if let Some(code) = status.code()
                && code != 0
            {
                out.push_str(&format!("\n[exit code {code}]"));
            }
        }
        None => {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&format!(
                "[timed out after {}s — process killed]",
                bash_timeout.as_secs()
            ));
        }
    }
    if out.is_empty() {
        out.push_str("[no output]");
    }
    Ok(out)
}

/// Read lines from an optional child-process pipe, calling `on_line` (behind a
/// Mutex so stdout/stderr can share it) for each line and pushing each into
/// `buf`. Writing into a shared, outer-scope `buf` (rather than returning a
/// `Vec`) is what lets the caller recover partial output if a timeout drops
/// this future mid-read.
async fn read_lines<R: tokio::io::AsyncRead + Unpin>(
    pipe: Option<R>,
    on_line: &std::sync::Mutex<&mut (dyn FnMut(&str) + Send)>,
    buf: &std::sync::Mutex<Vec<String>>,
) {
    let Some(pipe) = pipe else {
        return;
    };
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut reader = BufReader::new(pipe).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        let mut with_nl = line;
        with_nl.push('\n');
        if let Ok(mut cb) = on_line.lock() {
            (*cb)(&with_nl);
        }
        if let Ok(mut buf) = buf.lock() {
            buf.push(with_nl);
        }
    }
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
    args: BashArgs,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<ToolOutput> {
    if args.run_in_background {
        let id = crate::background::spawn(&args.command)?;
        return Ok(ToolOutput::plain(format!(
            "Started background process `{id}`: {}\n\
             Check its output with bash_output {{\"id\":\"{id}\"}} and stop it with \
             bash_kill {{\"id\":\"{id}\"}}.",
            args.command
        )));
    }
    let timeout = resolve_bash_timeout(args.timeout);
    let out = run_bash_streaming_with_timeout(&args.command, on_line, timeout).await?;
    Ok(ToolOutput::plain(condense(&out)))
}

#[cfg(test)]
mod tests {
    use super::{
        fast_check_for, is_filesystem_mutating, is_read_only, run_bash_streaming_with_timeout,
        target_path,
    };
    use crate::edit::sh_quote;
    use std::time::Duration;

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
        let out =
            run_bash_streaming_with_timeout("echo hello", &mut sink, Duration::from_secs(10))
                .await
                .expect("ok");
        assert!(out.contains("hello"), "got: {out:?}");
        assert!(!out.contains("timed out"), "got: {out:?}");
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

    #[test]
    fn bash_timeout_resolution_and_clamping() {
        use super::{
            resolve_bash_timeout, DEFAULT_BASH_TIMEOUT_SECS, MAX_BASH_TIMEOUT_SECS,
        };
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
    fn read_only_tools_are_classified() {
        assert!(is_read_only("read"));
        assert!(is_read_only("list"));
        assert!(is_read_only("grep"));
        assert!(is_read_only("diff"));
        assert!(is_read_only("glob"));
        // No filesystem side effects — safe to parallelize and offer in
        // read-only mode.
        assert!(is_read_only("update_plan"));
        assert!(is_read_only("record_decision"));
        assert!(is_read_only("bash_output"));
        // Mutating / effecting tools are not safe to run concurrently.
        assert!(!is_read_only("write"));
        assert!(!is_read_only("edit"));
        assert!(!is_read_only("multi_edit"));
        assert!(!is_read_only("apply_patch"));
        assert!(!is_read_only("bash"));
        assert!(!is_read_only("bash_kill"));
    }

    #[test]
    fn filesystem_mutating_tools_are_classified() {
        // Only tools that write to the working tree.
        assert!(is_filesystem_mutating("write"));
        assert!(is_filesystem_mutating("edit"));
        assert!(is_filesystem_mutating("multi_edit"));
        assert!(is_filesystem_mutating("apply_patch"));
        // Everything else — including non-read-only tools like bash — does not
        // directly mutate via the tool layer (bash runs alone; bash_kill stops
        // a process; update_plan/record_decision are in-memory only).
        assert!(!is_filesystem_mutating("bash"));
        assert!(!is_filesystem_mutating("bash_kill"));
        assert!(!is_filesystem_mutating("bash_output"));
        assert!(!is_filesystem_mutating("update_plan"));
        assert!(!is_filesystem_mutating("record_decision"));
        assert!(!is_filesystem_mutating("read"));
        assert!(!is_filesystem_mutating("diff"));
    }

    #[test]
    fn sh_quote_escapes_single_quotes() {
        assert_eq!(sh_quote("a b"), "'a b'");
        assert_eq!(sh_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn target_path_extracts_path_field() {
        assert_eq!(
            target_path("read", r#"{"path":"src/a.rs"}"#),
            Some("src/a.rs".into())
        );
        assert_eq!(
            target_path("write", r#"{"path":"b.rs","content":"x"}"#),
            Some("b.rs".into())
        );
        // list's path is optional → None when absent.
        assert_eq!(target_path("list", r#"{}"#), None);
        assert_eq!(target_path("list", r#"{"path":"sub"}"#), Some("sub".into()));
        // bash has no path → None (the safe-fallback case for dep inference).
        assert_eq!(target_path("bash", r#"{"command":"echo hi"}"#), None);
        // Malformed JSON → None (tolerant).
        assert_eq!(target_path("read", "not json"), None);
        // apply_patch: first Update/Add File directive's path is extracted.
        let patch = r#"{"patch":"*** Begin Patch\n*** Update File: src/a.rs\n-old\n+new\n*** End Patch"}"#;
        assert_eq!(
            target_path("apply_patch", patch),
            Some("src/a.rs".into())
        );
        let add_patch = r#"{"patch":"*** Begin Patch\n*** Add File: new.txt\nhello\n*** End Patch"}"#;
        assert_eq!(
            target_path("apply_patch", add_patch),
            Some("new.txt".into())
        );
        // No file directives → None.
        assert_eq!(
            target_path("apply_patch", r#"{"patch":"*** Begin Patch\n*** End Patch"}"#),
            None
        );
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
}
