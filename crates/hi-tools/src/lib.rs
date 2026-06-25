//! Built-in tools: `read`, `write`, `edit`, `bash`.
//!
//! Richer capabilities come from subprocess CLI tools the model invokes via
//! `bash` — not a plugin runtime — so this set stays intentionally small.

pub mod checkpoint;
pub mod guard;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use hi_ai::ToolSpec;
use regex::Regex;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

/// Per-result character budget so a single read or noisy command can't blow the
/// context. Overridable via `HI_TOOL_RESULT_CHARS` — lower it for a tight local
/// window, raise it when the model has room. Read once, at first use. The
/// default is intentionally tight for remote agent loops: ~5k chars is ~1.3k
/// tokens, and repeated tool rounds resend this history.
static MAX_OUTPUT_CHARS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("HI_TOOL_RESULT_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1_000)
        .unwrap_or(5_000)
});
const DEFAULT_READ_LIMIT: usize = 240;
const BASH_TIMEOUT: Duration = Duration::from_secs(120);
const CHECK_TIMEOUT: Duration = Duration::from_secs(300);

/// Validate that a path is inside the workspace root (cwd by default). Returns
/// the canonicalized absolute path if safe, or an error explaining why not.
/// Set `HI_NO_PATH_GUARD=1` to disable (not recommended — the model can then
/// read/write any file on the system).
fn validate_workspace_path(path: &str) -> Result<std::path::PathBuf> {
    if std::env::var_os("HI_NO_PATH_GUARD").is_some() {
        return Ok(Path::new(path).to_path_buf());
    }
    let cwd = std::env::current_dir().context("determining working directory")?;
    let target = Path::new(path);
    // If absolute, canonicalize and check containment. If relative, join to cwd.
    let resolved = if target.is_absolute() {
        target.to_path_buf()
    } else {
        cwd.join(target)
    };
    // For paths that exist, canonicalize to resolve symlinks and `..`.
    let canonical = resolved.canonicalize().unwrap_or(resolved.clone());
    let canonical_cwd = cwd.canonicalize().unwrap_or(cwd.clone());
    if canonical.starts_with(&canonical_cwd) {
        return Ok(canonical);
    }
    // Allow /tmp and macOS /var/folders paths (scratch files, pipes). On macOS,
    // /tmp symlinks to /private/tmp and /var/folders to /private/var/folders,
    // so canonicalize() resolves them.
    if canonical.starts_with("/tmp/")
        || canonical.starts_with("/private/tmp/")
        || canonical.starts_with("/var/folders/")
        || canonical.starts_with("/private/var/folders/")
    {
        return Ok(canonical);
    }
    bail!(
        "path '{}' is outside the workspace ({}). \
         Set HI_NO_PATH_GUARD=1 to allow out-of-workspace paths.",
        path,
        canonical_cwd.display()
    );
}

/// VCS metadata directories that must never reach the model. We walk with
/// `hidden(false)` so the agent can see useful dotfiles (`.github/`,
/// `.env.example`, `.cargo/config.toml`, …), but these internal directories are
/// large, mostly binary, and leak repository internals (loose/packed objects,
/// refs, reflogs, config). Used as a `WalkBuilder::filter_entry` predicate,
/// which prunes the whole subtree so we never even descend into them.
fn is_vcs_metadata_dir(entry: &ignore::DirEntry) -> bool {
    matches!(
        entry.file_name().to_str(),
        Some(".git" | ".hg" | ".svn" | ".jj")
    )
}

/// Maximum number of cached file reads. Beyond this, the cache is cleared
/// entirely (cheap — it refills lazily on the next re-read).
const READ_CACHE_MAX: usize = 50;

/// Per-turn cache of file reads, so re-reading the same file (common when the
/// model is orienting) hits memory instead of disk. Cleared between turns, and
/// bounded to [`READ_CACHE_MAX`] entries to avoid unbounded memory growth.
static READ_CACHE: LazyLock<Mutex<HashMap<String, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Clear the per-turn read cache. Call at the start of each turn.
pub fn clear_read_cache() {
    if let Ok(mut cache) = READ_CACHE.lock() {
        cache.clear();
    }
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
        Ok(o) => {
            o.status.success()
                && String::from_utf8_lossy(&o.stdout).trim() == "true"
        }
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
    format!("staged {n} file{}\ncommitted: \"{subject}\"", if n == 1 { "" } else { "s" })
}

/// The tools advertised to the model each turn.
pub fn tool_specs() -> Vec<ToolSpec> {
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
            description: "Run a shell command via `sh -c` in the current working directory and return combined stdout/stderr.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to run." }
                },
                "required": ["command"]
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
pub static TOOL_SPECS: LazyLock<Vec<ToolSpec>> = LazyLock::new(tool_specs);

/// The result of a tool call, split into `content` shown to the model and an
/// optional richer `display` for the UI (e.g. a colored diff). This keeps
/// edit/write feedback terse for the model while showing the user what changed.
/// `plan`, when set, drives the live plan tracker instead of a transcript echo.
pub struct ToolOutput {
    pub content: String,
    pub display: Option<String>,
    pub plan: Option<Vec<PlanStep>>,
}

impl ToolOutput {
    fn plain(content: String) -> Self {
        Self {
            content,
            display: None,
            plan: None,
        }
    }

    fn shown(content: String, display: String) -> Self {
        Self {
            content,
            display: Some(display),
            plan: None,
        }
    }

    /// A result that updates the user-facing plan checklist. The model sees only
    /// `content` (a terse confirmation); the steps drive the pinned tracker.
    fn planned(content: String, steps: Vec<PlanStep>) -> Self {
        Self {
            content,
            display: None,
            plan: Some(steps),
        }
    }
}

/// One line of the task plan/checklist surfaced by the `update_plan` tool. The
/// model resubmits the whole list (with updated statuses) on every call, so
/// there is no per-step index or state to drift out of sync.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanStep {
    pub title: String,
    pub status: PlanStatus,
}

/// The progress state of a single [`PlanStep`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanStatus {
    Pending,
    Active,
    Done,
}

impl PlanStatus {
    /// Map the model's free-form status string onto a state, tolerating the
    /// common synonyms models reach for ("in_progress", "completed", "todo", …).
    fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "done" | "complete" | "completed" | "finished" => PlanStatus::Done,
            "active" | "in_progress" | "in-progress" | "doing" | "current" | "started" => {
                PlanStatus::Active
            }
            _ => PlanStatus::Pending,
        }
    }
}

/// Whether a tool only observes state, with no side effects — so several can
/// run concurrently within one round. The mutating/effecting tools (`write`,
/// `edit`, `bash`) are excluded, since order and isolation matter for them.
pub fn is_read_only(name: &str) -> bool {
    matches!(name, "read" | "list" | "grep" | "glob" | "diff")
    // `apply_patch`, like `write`/`edit`, is effecting and never runs in a
    // parallel batch — order and isolation matter.
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
        return Ok(ToolOutput::plain(condense(
            &run_bash_streaming(&args.command, on_line).await?,
        )));
    }
    // All other tools: delegate to the normal path (on_line unused).
    run(name, arguments).await
}

async fn run(name: &str, arguments: &str) -> Result<ToolOutput> {
    match name {
        "read" => {
            let args: ReadArgs = parse(arguments)?;
            validate_workspace_path(&args.path)?;
            // Check the per-turn cache first — models often re-read files.
            let content = {
                let cache = READ_CACHE.lock().unwrap();
                if let Some(cached) = cache.get(&args.path) {
                    cached.clone()
                } else {
                    drop(cache);
                    // Read as bytes first so we can detect binary files and
                    // give a clear message instead of an opaque UTF-8 error.
                    let bytes = tokio::fs::read(&args.path)
                        .await
                        .with_context(|| format!("reading {}", args.path))?;
                    if is_binary(&bytes) {
                        bail!(
                            "{} is a binary file ({} bytes) — the `read` tool is for text. \
                             Use `bash` to inspect it (e.g. `file {}`, `xxd {} | head`).",
                            args.path,
                            bytes.len(),
                            sh_quote(&args.path),
                            sh_quote(&args.path)
                        );
                    }
                    let content = String::from_utf8_lossy(&bytes).into_owned();
                    if let Ok(mut cache) = READ_CACHE.lock() {
                        if cache.len() >= READ_CACHE_MAX {
                            cache.clear();
                        }
                        cache.insert(args.path.clone(), content.clone());
                    }
                    content
                }
            };
            Ok(ToolOutput::plain(truncate(&format_read(
                &content,
                args.offset,
                args.limit,
            ))))
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
                cache.remove(&args.path);
            }
            let after = maybe_format(&args.path, args.content).await;
            Ok(ToolOutput::shown(
                format!("Wrote {} bytes to {}", after.len(), args.path),
                diff(&before, &after),
            ))
        }
        "edit" => {
            let args: EditArgs = parse(arguments)?;
            validate_workspace_path(&args.path)?;
            let before = tokio::fs::read_to_string(&args.path)
                .await
                .with_context(|| format!("reading {}", args.path))?;
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
                cache.remove(&args.path);
            }
            let after = maybe_format(&args.path, after).await;
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
            let before = tokio::fs::read_to_string(&args.path)
                .await
                .with_context(|| format!("reading {}", args.path))?;
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
                cache.remove(&args.path);
            }
            let after = maybe_format(&args.path, after).await;
            Ok(ToolOutput::shown(
                format!("Applied {} edits to {}", args.edits.len(), args.path),
                diff(&before, &after),
            ))
        }
        "bash" => {
            let args: BashArgs = parse(arguments)?;
            Ok(ToolOutput::plain(condense(&run_bash(&args.command).await?)))
        }
        "list" => {
            let args: ListArgs = parse(arguments)?;
            let path = args.path.as_deref().unwrap_or(".");
            // Use the `ignore` crate for gitignore-aware directory walking, same
            // semantics as `git ls-files` but without spawning a process.
            let mut out = String::new();
            let mut count = 0u32;
            let walker = ignore::WalkBuilder::new(path)
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .require_git(false) // fall back to all files outside a repo
                .hidden(false)
                .filter_entry(|e| !is_vcs_metadata_dir(e))
                .build();
            for entry in walker {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if !entry.file_type().map_or(false, |ft| ft.is_file()) {
                    continue;
                }
                let rel = entry.path().to_string_lossy();
                out.push_str(&rel);
                out.push('\n');
                count += 1;
                if count >= 1000 {
                    out.push_str("… (truncated at 1000 entries)\n");
                    break;
                }
            }
            let out = if out.is_empty() {
                "(no files found)".to_string()
            } else {
                out
            };
            Ok(ToolOutput::plain(truncate(&out)))
        }
        "diff" => {
            // Reuse the working-tree diff summary, but return it as model content
            // (plain text, no ANSI) so the model can review what changed.
            Ok(ToolOutput::plain(working_tree_diff_plain().await))
        }
        "glob" => {
            #[derive(Deserialize)]
            struct GlobArgs {
                pattern: String,
                path: Option<String>,
            }
            let args: GlobArgs = parse(arguments)?;
            let path = args.path.as_deref().unwrap_or(".");
            let mut out = String::new();
            let mut count = 0u32;
            let mut builder = ignore::WalkBuilder::new(path);
            builder
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .require_git(false)
                .hidden(false)
                .filter_entry(|e| !is_vcs_metadata_dir(e));
            let mut override_builder = ignore::overrides::OverrideBuilder::new(path);
            if let Err(e) = override_builder.add(&args.pattern) {
                return Ok(ToolOutput::plain(format!("invalid glob `{}`: {e}", args.pattern)));
            }
            match override_builder.build() {
                Ok(ov) => {
                    let walker = builder.overrides(ov).build();
                    for entry in walker {
                        let entry = match entry {
                            Ok(e) => e,
                            Err(_) => continue,
                        };
                        if !entry.file_type().map_or(false, |ft| ft.is_file()) {
                            continue;
                        }
                        let rel = entry.path().to_string_lossy();
                        out.push_str(&rel);
                        out.push('\n');
                        count += 1;
                        if count >= 500 {
                            out.push_str("… (truncated at 500 entries)\n");
                            break;
                        }
                    }
                }
                Err(e) => {
                    return Ok(ToolOutput::plain(format!("invalid glob `{}`: {e}", args.pattern)));
                }
            }
            let out = if out.is_empty() {
                format!("no files match `{}`", args.pattern)
            } else {
                out
            };
            Ok(ToolOutput::plain(truncate(&out)))
        }
        "grep" => {
            let args: GrepArgs = parse(arguments)?;
            let pattern = &args.pattern;
            let path = args.path.as_deref().unwrap_or(".");
            let context = args.context.unwrap_or(0);

            // Fast path: shell out to ripgrep when available — 5-20x faster than
            // the inline walker, with built-in .gitignore support and SIMD.
            if tool_available("rg").await {
                let mut cmd_args = vec![
                    "--no-heading".to_string(),
                    "--line-number".to_string(),
                    "--color=never".to_string(),
                    "--max-count=200".to_string(),
                    // Never search VCS metadata, even if the user's ripgrep
                    // config enables --hidden (which would otherwise descend
                    // into .git and leak repository internals to the model).
                    "--glob=!.git".to_string(),
                    "--glob=!.hg".to_string(),
                    "--glob=!.svn".to_string(),
                    "--glob=!.jj".to_string(),
                ];
                if context > 0 {
                    cmd_args.push(format!("--context={context}"));
                }
                if let Some(glob) = &args.glob {
                    cmd_args.push("--glob".to_string());
                    cmd_args.push(glob.clone());
                }
                cmd_args.push("--".to_string());
                cmd_args.push(pattern.clone());
                cmd_args.push(path.to_string());
                let output = Command::new("rg")
                    .args(&cmd_args)
                    .output()
                    .await;
                match output {
                    Ok(o) if o.status.success() || !o.stdout.is_empty() => {
                        let text = String::from_utf8_lossy(&o.stdout);
                        let out = if text.trim().is_empty() {
                            format!("no matches for {}", args.pattern)
                        } else {
                            text.into_owned()
                        };
                        return Ok(ToolOutput::plain(truncate(&out)));
                    }
                    Ok(o) if o.status.code() == Some(1) => {
                        // rg exit 1 = no matches (not an error)
                        return Ok(ToolOutput::plain(
                            format!("no matches for {}", args.pattern),
                        ));
                    }
                    // Fall through to inline walker on other rg errors.
                    _ => {}
                }
            }

            // Fallback: inline walker with the `ignore` crate + `regex`.
            let re = match Regex::new(pattern) {
                Ok(re) => re,
                Err(e) => {
                    return Ok(ToolOutput::plain(format!("invalid regex: {e}")));
                }
            };
            let mut builder = ignore::WalkBuilder::new(path);
            builder
                .git_ignore(true)
                .git_global(true)
                .git_exclude(true)
                .require_git(false)
                .hidden(false)
                .filter_entry(|e| !is_vcs_metadata_dir(e));
            if let Some(glob) = &args.glob {
                match ignore::overrides::OverrideBuilder::new(path).add(glob) {
                    Ok(ovb) => match ovb.build() {
                        Ok(ov) => {
                            builder.overrides(ov);
                        }
                        Err(e) => {
                            return Ok(ToolOutput::plain(format!("invalid glob `{glob}`: {e}")));
                        }
                    },
                    Err(e) => {
                        return Ok(ToolOutput::plain(format!("invalid glob `{glob}`: {e}")));
                    }
                }
            }
            let mut out = String::new();
            let mut count = 0u32;
            let walker = builder.build();
            for entry in walker {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if !entry.file_type().map_or(false, |ft| ft.is_file()) {
                    continue;
                }
                let file_path = entry.path();
                let bytes = match tokio::fs::read(file_path).await {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                if is_binary(&bytes) {
                    continue;
                }
                let content = String::from_utf8_lossy(&bytes);
                let rel = file_path.to_string_lossy();
                let lines: Vec<&str> = content.lines().collect();
                for (i, line) in lines.iter().enumerate() {
                    if re.is_match(line) {
                        let line_no = i + 1;
                        if context > 0 {
                            let start = i.saturating_sub(context);
                            let end = (i + context + 1).min(lines.len());
                            for ctx_i in start..end {
                                let marker = if ctx_i == i { ":" } else { "-" };
                                out.push_str(&format!(
                                    "{rel}{marker}{}: {}\n",
                                    ctx_i + 1,
                                    lines[ctx_i]
                                ));
                            }
                            out.push_str("--\n");
                        } else {
                            out.push_str(&format!("{rel}:{line_no}: {line}\n"));
                        }
                        count += 1;
                        if count >= 200 {
                            out.push_str("… (truncated at 200 matches)\n");
                            break;
                        }
                    }
                }
                if count >= 200 {
                    break;
                }
            }
            let out = if out.is_empty() {
                format!("no matches for {}", args.pattern)
            } else {
                out
            };
            Ok(ToolOutput::plain(truncate(&out)))
        }
        "apply_patch" => {
            #[derive(Deserialize)]
            struct PatchArgs {
                patch: String,
            }
            let args: PatchArgs = parse(arguments)?;
            let result = apply_multi_patch(&args.patch)?;
            Ok(ToolOutput::plain(result))
        }
        other => bail!("unknown tool: {other}"),
    }
}

/// Apply a multi-file patch in Claude's `apply_patch` format. The envelope is
/// `*** Begin Patch … *** End Patch`; inside, each `*** Update File:`,
/// `*** Add File:`, or `*** Delete File:` header introduces a file operation.
///
/// For updates, each line is either context (no prefix), a removal (`-`), or an
/// addition (`+`). The result file is rebuilt as the "after" state: context and
/// added lines are emitted in order, removed lines are dropped, and `@@ … @@`
/// hunk headers are skipped. This works because the model sends the full new
/// region via context+added lines, with `-` marking only what was removed. We
/// read the original only to verify the file exists and to diff against for the
/// UI-friendly result.
fn apply_multi_patch(patch: &str) -> Result<String> {
    let mut results = Vec::new();
    let lines: Vec<&str> = patch.lines().collect();

    // Validate envelope. An empty body is also rejected so the model gets a
    // clear message instead of a confusing "no operations" result.
    if lines.is_empty() {
        bail!("patch is empty");
    }
    if !lines[0].trim().starts_with("*** Begin Patch") {
        bail!("patch must start with '*** Begin Patch'");
    }
    let mut i = 1;

    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with("*** End Patch") || line.is_empty() {
            i += 1;
            continue;
        }

        // Add File: every following line until the next `*** ` directive is the
        // new file's content (verbatim, no +/- prefixes).
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            let path = path.trim();
            validate_workspace_path(path)?;
            let mut content = String::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with("*** ") {
                content.push_str(lines[i]);
                content.push('\n');
                i += 1;
            }
            if let Some(parent) = Path::new(path).parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            std::fs::write(path, &content).with_context(|| format!("writing {path}"))?;
            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(path);
            }
            results.push(format!("+ added {path}"));
            continue;
        }

        // Delete File: remove the file if it exists.
        if let Some(path) = line.strip_prefix("*** Delete File: ") {
            let path = path.trim();
            validate_workspace_path(path)?;
            std::fs::remove_file(path).with_context(|| format!("deleting {path}"))?;
            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(path);
            }
            results.push(format!("- deleted {path}"));
            i += 1;
            continue;
        }

        // Update File: rebuild from context + added lines, dropping removed ones.
        if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = path.trim();
            validate_workspace_path(path)?;
            // Read to verify the file exists (Update File can't create — use
            // Add File for that). The content isn't needed: the rebuilt "after"
            // text comes entirely from the patch's context + added lines.
            std::fs::read_to_string(path)
                .with_context(|| format!("reading {path} (use *** Add File: to create)"))?;
            let mut after = String::new();
            let mut changes = 0u32;
            i += 1;

            while i < lines.len() && !lines[i].trim().starts_with("*** ") {
                let l = lines[i];
                match l.chars().next() {
                    Some('+') => {
                        after.push_str(&l[1..]);
                        after.push('\n');
                        changes += 1;
                        i += 1;
                    }
                    Some('-') => {
                        // Skip removed line — it's dropped from the output.
                        changes += 1;
                        i += 1;
                    }
                    _ => {
                        // Context line or `@@ … @@` hunk header. Include the
                        // line as-is, except for the hunk header, which is just
                        // a delimiter and not part of the file content.
                        if !l.trim_start().starts_with("@@") {
                            after.push_str(l);
                            after.push('\n');
                        }
                        i += 1;
                    }
                }
            }

            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(path);
            }
            std::fs::write(path, &after).with_context(|| format!("writing {path}"))?;
            results.push(format!("~ updated {path} ({changes} change{})", if changes == 1 { "" } else { "s" }));
            continue;
        }

        // Unknown directive — skip it rather than aborting the whole patch, so a
        // minor format variation doesn't lose all the file operations.
        i += 1;
    }

    if results.is_empty() {
        bail!("patch contained no file operations");
    }
    Ok(results.join("\n"))
}

/// Single-quote a string for safe interpolation into an `sh -c` command.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// A compact, readable diff for UI display: a bold one-line summary of how many
/// lines changed, then each changed region shown with a few lines of unchanged
/// context, gutter line numbers, and `±` signs. The context and line numbers let
/// the reader see *where* an edit lands, not just what changed; non-adjacent
/// regions are separated by a dim `⋯`. The model never sees this — it's the
/// `display` half of an edit's [`ToolOutput`], so the extra context costs no
/// tokens. (`/diff` shows the full working-tree diff.)
fn diff(before: &str, after: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    /// Lines of unchanged context shown on each side of a changed region.
    const CONTEXT: usize = 2;

    let tdiff = TextDiff::from_lines(before, after);
    let (mut adds, mut dels) = (0usize, 0usize);
    for change in tdiff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => adds += 1,
            ChangeTag::Delete => dels += 1,
            ChangeTag::Equal => {}
        }
    }
    if adds == 0 && dels == 0 {
        return "(no changes)".to_string();
    }

    let mut body = String::new();
    for (hunk_idx, group) in tdiff.grouped_ops(CONTEXT).iter().enumerate() {
        if hunk_idx > 0 {
            body.push_str("\x1b[2m   ⋯\x1b[0m\n"); // gap between changed regions
        }
        for op in group {
            for change in tdiff.iter_changes(op) {
                // Number each line by its own side of the diff: removed lines by
                // their old position, added/context lines by their new one.
                let (idx, sign, color) = match change.tag() {
                    ChangeTag::Delete => (change.old_index(), '-', "\x1b[31m"),
                    ChangeTag::Insert => (change.new_index(), '+', "\x1b[32m"),
                    ChangeTag::Equal => (change.new_index(), ' ', "\x1b[2m"),
                };
                let gutter = idx
                    .map(|i| format!("{:>4}", i + 1))
                    .unwrap_or_else(|| "    ".to_string());
                let text = change.value();
                let text = text.strip_suffix('\n').unwrap_or(text);
                body.push_str(&format!("{color}{gutter} {sign} {text}\x1b[0m\n"));
            }
        }
    }

    let plural = |n: usize| if n == 1 { "" } else { "s" };
    format!(
        "\x1b[1m{adds} addition{}, {dels} deletion{}\x1b[0m\n{body}",
        plural(adds),
        plural(dels)
    )
}

/// Replace `old` with `new` in `text`, tolerating the whitespace differences
/// that make models' exact-match edits fail. Strategies, in order:
///   1. exact match (unique, or all when `replace_all`);
///   2. line-based match ignoring trailing whitespace (also fixes CRLF);
///   3. line-based match ignoring all indentation, re-indenting `new` to fit.
///
/// Without `replace_all`, each strategy requires a unique match so an edit is
/// never applied ambiguously. With `replace_all`, strategy 1 replaces every
/// exact occurrence; the fuzzy strategies still require uniqueness (they can't
/// safely disambiguate multiple fuzzy matches).
fn apply_edit(text: &str, old: &str, new: &str, replace_all: bool) -> Result<String> {
    if old.is_empty() {
        bail!("old_string is empty");
    }
    // 1. Exact.
    let count = text.matches(old).count();
    if replace_all && count > 0 {
        return Ok(text.replace(old, new));
    }
    match count {
        1 => return Ok(text.replacen(old, new, 1)),
        n if n > 1 => bail!(
            "old_string is not unique ({n} matches); include more surrounding context to pick one, \
             or set replace_all=true to replace every occurrence"
        ),
        _ => {}
    }

    let lines = lines_with_offsets(text);
    let old_lines: Vec<&str> = old
        .split_inclusive('\n')
        .map(|l| l.strip_suffix('\n').unwrap_or(l))
        .collect();

    // 2. Ignore trailing whitespace (catches trailing spaces and CRLF).
    if let Some((start, end, _)) =
        find_unique_window(&lines, text.len(), &old_lines, |l| l.trim_end())
    {
        return Ok(splice(text, start, end, new.to_string()));
    }

    // 3. Ignore all indentation, then re-indent `new` to match the file.
    if let Some((start, end, idx)) =
        find_unique_window(&lines, text.len(), &old_lines, |l| l.trim())
    {
        let file_indent = leading_ws(lines[idx].1);
        let old_indent = leading_ws(old_lines.first().copied().unwrap_or(""));
        let reindented = reindent(new, old_indent, file_indent);
        return Ok(splice(text, start, end, reindented));
    }

    bail!("{}", edit_not_found_help(text, old));
}

/// Build a helpful error when `old_string` doesn't match: point the model at the
/// nearest similar lines (with numbers) so it can copy the exact text, rather
/// than blindly retrying the same string. Falls back to similarity scoring when
/// no line contains the needle — so a model that got a line slightly wrong still
/// gets pointed at the right region instead of "no line resembles".
fn edit_not_found_help(text: &str, old: &str) -> String {
    let mut msg = String::from(
        "old_string not found, even allowing for whitespace differences. \
         (Do not include the line-number gutter from `read` in old_string.) ",
    );
    let lines: Vec<&str> = text.lines().collect();
    let needle = old
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    if needle.is_empty() {
        msg.push_str("Re-read the file and copy the exact text to replace.");
        return msg;
    }
    // Lines equal (ignoring indentation) or containing the first old line.
    let hits: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim() == needle || l.contains(needle))
        .map(|(i, _)| i)
        .take(3)
        .collect();
    // If no direct hit, try similarity: score each line by how many words from
    // the needle it shares, and show the top matches. This catches cases where
    // the model misremembered a line (wrong variable name, typo, etc.) but is
    // still close enough to find the right region.
    let hits = if hits.is_empty() {
        let needles: Vec<&str> = needle.split_whitespace().collect();
        if needles.is_empty() {
            return msg
                + &format!(
                    "No line resembling `{}` is in the {}-line file; re-read it to get the current text.",
                    clip(needle, 60),
                    lines.len()
                );
        }
        let mut scored: Vec<(usize, usize)> = lines
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let lower = l.to_lowercase();
                let score = needles
                    .iter()
                    .filter(|w| {
                        let w = w.to_lowercase();
                        lower.contains(w.as_str())
                    })
                    .count();
                (i, score)
            })
            .filter(|(_, s)| *s > 0)
            .collect();
        scored.sort_by_key(|(_, s)| std::cmp::Reverse(*s));
        scored.into_iter().take(3).map(|(i, _)| i).collect()
    } else {
        hits
    };
    if hits.is_empty() {
        msg.push_str(&format!(
            "No line resembling `{}` is in the {}-line file; re-read it to get the current text.",
            clip(needle, 60),
            lines.len()
        ));
        return msg;
    }
    msg.push_str("The closest matching lines in the file are:\n");
    for i in hits {
        let lo = i.saturating_sub(2);
        let hi = (i + 3).min(lines.len());
        for (off, line) in lines[lo..hi].iter().enumerate() {
            msg.push_str(&format!("{:>6}\t{}\n", lo + off + 1, line));
        }
        msg.push_str("  ---\n");
    }
    msg.push_str("Copy old_string verbatim from one of these regions.");
    msg
}

/// Truncate to `max` chars with an ellipsis (single-line error context).
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

/// Byte offset and trailing-newline-stripped content of each line in `text`.
fn lines_with_offsets(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        out.push((offset, line.strip_suffix('\n').unwrap_or(line)));
        offset += line.len();
    }
    out
}

/// Find the single run of lines matching `old_lines` under `norm`. Returns
/// `(start_byte, end_byte, first_line_index)`, or `None` if absent or ambiguous.
fn find_unique_window(
    lines: &[(usize, &str)],
    text_len: usize,
    old_lines: &[&str],
    norm: impl Fn(&str) -> &str,
) -> Option<(usize, usize, usize)> {
    let n = old_lines.len();
    if n == 0 || lines.len() < n {
        return None;
    }
    let mut found = None;
    for i in 0..=lines.len() - n {
        if (0..n).all(|j| norm(lines[i + j].1) == norm(old_lines[j])) {
            if found.is_some() {
                return None; // ambiguous
            }
            let start = lines[i].0;
            let end = lines.get(i + n).map_or(text_len, |&(off, _)| off);
            found = Some((start, end, i));
        }
    }
    found
}

/// Replace `text[start..end]` with `replacement`, preserving a trailing newline.
fn splice(text: &str, start: usize, end: usize, mut replacement: String) -> String {
    if text[start..end].ends_with('\n') && !replacement.ends_with('\n') {
        replacement.push('\n');
    }
    format!("{}{}{}", &text[..start], replacement, &text[end..])
}

fn leading_ws(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// Rebase `new`'s indentation from `old_indent` to `file_indent`, preserving
/// each line's relative nesting.
fn reindent(new: &str, old_indent: &str, file_indent: &str) -> String {
    if old_indent == file_indent {
        return new.to_string();
    }
    let mut out = String::new();
    for line in new.split_inclusive('\n') {
        let content = line.strip_suffix('\n').unwrap_or(line);
        if content.trim().is_empty() {
            out.push_str(line);
            continue;
        }
        let stripped = content.strip_prefix(old_indent).unwrap_or(content);
        out.push_str(file_indent);
        out.push_str(stripped);
        if line.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

async fn run_bash(command: &str) -> Result<String> {
    run_bash_streaming(command, &mut |_| {}).await
}

/// Run a shell command, calling `on_line` for each line of output as it arrives
/// (both stdout and stderr). The final assembled output is still returned for the
/// model. Lines are delivered with a trailing newline.
async fn run_bash_streaming(
    command: &str,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<String> {
    // Refuse the handful of irreversible operations a checkpoint can't undo.
    if let Some(reason) = guard::catastrophic_op(command) {
        return Ok(format!(
            "⚠ refused: this command {reason}. It's blocked as irreversible — the per-turn \
             checkpoint can't undo it. If it's genuinely needed, ask the user to run it \
             themselves (they can also set HI_ALLOW_DANGEROUS=1 to disable this guard)."
        ));
    }
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("failed to spawn command")?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Use a shared Mutex<&mut dyn FnMut> so both stdout and stderr readers can
    // call on_line without double-borrowing.
    let cb: &std::sync::Mutex<&mut (dyn FnMut(&str) + Send)> = &std::sync::Mutex::new(on_line);
    let (stdout_lines, stderr_lines): (Vec<String>, Vec<String>) =
        tokio::join!(read_lines(stdout, cb), read_lines(stderr, cb),);

    // Wait for the process to finish (with timeout for the remainder).
    let status = match tokio::time::timeout(BASH_TIMEOUT, child.wait()).await {
        Ok(Ok(status)) => status,
        Ok(Err(err)) => bail!("command failed: {err}"),
        Err(_) => {
            let _ = child.kill().await;
            bail!("command timed out after {}s", BASH_TIMEOUT.as_secs())
        }
    };

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
    if let Some(code) = status.code()
        && code != 0
    {
        out.push_str(&format!("\n[exit code {code}]"));
    }
    if out.is_empty() {
        out.push_str("[no output]");
    }
    Ok(out)
}

/// Read lines from an optional child-process pipe, calling `on_line` (behind a
/// Mutex so stdout/stderr can share it) for each line, and collecting them.
async fn read_lines<R: tokio::io::AsyncRead + Unpin>(
    pipe: Option<R>,
    on_line: &std::sync::Mutex<&mut (dyn FnMut(&str) + Send)>,
) -> Vec<String> {
    let Some(pipe) = pipe else {
        return Vec::new();
    };
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut lines = Vec::new();
    let mut reader = BufReader::new(pipe).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        let mut with_nl = line;
        with_nl.push('\n');
        if let Ok(mut cb) = on_line.lock() {
            (*cb)(&with_nl);
        }
        lines.push(with_nl);
    }
    lines
}

fn parse<T: for<'de> Deserialize<'de>>(arguments: &str) -> Result<T> {
    serde_json::from_str(arguments).context("invalid tool arguments")
}

/// Clip output to the configured character budget ([`MAX_OUTPUT_CHARS`]).
fn truncate(s: &str) -> String {
    truncate_to(s, *MAX_OUTPUT_CHARS)
}

/// Whether the diagnostic condenser is enabled. Off falls back to plain head+tail
/// truncation — the knob that lets the eval harness A/B the condenser's value.
/// Read once, at first use.
static CONDENSE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| condense_enabled(std::env::var("HI_CONDENSE").ok().as_deref()));

/// Parse the `HI_CONDENSE` toggle: on by default; `0`/`off`/`false`/`no` disable.
fn condense_enabled(var: Option<&str>) -> bool {
    !matches!(var, Some("0" | "off" | "false" | "no"))
}

/// Condense output to the configured budget (test/diagnostic-aware), unless the
/// condenser is disabled — then just head+tail clip.
fn condense(s: &str) -> String {
    if *CONDENSE_ENABLED {
        condense_diagnostics(s, *MAX_OUTPUT_CHARS)
    } else {
        truncate(s)
    }
}

/// Condense test-runner and compiler output: keep the head, the summary, and
/// every failure — test failures with a little surrounding context, and whole
/// compiler-diagnostic blocks (the `error[...]` line plus its `-->` location,
/// code frame, and notes) — while dropping the long runs of passing `... ok`
/// noise. A 4,000-line green run with three failures collapses to those three
/// failures plus the count. Output that doesn't look like a test/diagnostic run,
/// or that has nothing of interest in its body, falls back to plain head+tail
/// [`truncate_to`] (don't mangle a file dump).
///
/// Deliberately biased to *over*-keep: a stray noise line surviving is harmless,
/// but dropping a real failure would send the model after the wrong thing. This
/// is the "Tier 0" deterministic extractor — no model call, reproducible on the
/// eval harness.
pub fn condense_diagnostics(s: &str, max: usize) -> String {
    if !looks_like_diagnostics(s) {
        return truncate_to(s, max);
    }
    let lines: Vec<&str> = s.lines().collect();
    let n = lines.len();
    const HEAD: usize = 4; // the "running N tests" / session preamble
    const TAIL: usize = 6; // the summary line(s) live at the very end
    const CONTEXT: usize = 2; // lines kept on each side of a test-failure line
    const MAX_BLOCK: usize = 40; // cap on a single compiler-diagnostic block

    let mut keep = vec![false; n];
    // Whether anything in the *body* (past the head, before the tail) was worth
    // keeping. If not, detection was a false positive — clip normally instead of
    // emitting a misleading "everything omitted" digest.
    let mut matched = false;
    for i in 0..n {
        let line = lines[i];
        if starts_diagnostic_block(line) {
            // Keep the whole multi-line block: rustc/gcc/clang print the location,
            // code frame, and notes under the `error:` line, ending at a blank
            // line. Bounded so a pathological block can't run away.
            matched = true;
            let end = (i + MAX_BLOCK).min(n);
            for (off, l) in lines[i..end].iter().enumerate() {
                if off > 0 && l.trim().is_empty() {
                    break;
                }
                keep[i + off] = true;
            }
        } else if is_signal_line(line) {
            matched = true;
            let lo = i.saturating_sub(CONTEXT);
            let hi = (i + CONTEXT + 1).min(n);
            for slot in keep.iter_mut().take(hi).skip(lo) {
                *slot = true;
            }
        }
        // Always keep the head preamble and the trailing summary, wherever the
        // failures fall — but signals are detected everywhere, so errors sitting
        // in the tail still mark this as real diagnostics (not a false positive).
        if i < HEAD || i + TAIL >= n {
            keep[i] = true;
        }
    }
    if !matched {
        return truncate_to(s, max);
    }
    // When almost everything is a signal (a wall of failures), there's no green
    // noise to drop — head+tail clip the original rather than pepper it with
    // omission markers.
    let kept = keep.iter().filter(|&&k| k).count();
    if kept * 10 >= n * 9 {
        return truncate_to(s, max);
    }

    let mut out = String::new();
    let mut i = 0;
    while i < n {
        if keep[i] {
            out.push_str(lines[i]);
            out.push('\n');
            i += 1;
        } else {
            let start = i;
            while i < n && !keep[i] {
                i += 1;
            }
            // A tiny gap (e.g. a blank line between two error blocks) is cheaper
            // to show than to announce, and keeps the output readable.
            let gap = i - start;
            if gap <= 2 {
                for l in &lines[start..i] {
                    out.push_str(l);
                    out.push('\n');
                }
            } else {
                out.push_str(&format!("… {gap} lines omitted …\n"));
            }
        }
    }
    // Even the condensed view honours the char budget.
    truncate_to(out.trim_end(), max)
}

/// Whether output looks like a test run or compiler diagnostics, and so is worth
/// condensing rather than blind-clipping. Specific markers keep this from firing
/// on an ordinary command dump.
fn looks_like_diagnostics(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    // Test runners.
    l.contains("test result:")                             // rust libtest summary
        || l.contains("=== failures ===")                  // pytest
        || l.contains("short test summary")                // pytest
        || l.contains("collected ")                        // pytest "collected N items"
        || (l.contains("running ") && l.contains(" test")) // libtest "running N tests"
        || (l.contains(" passed") && (l.contains(" failed") || l.contains(" error")))
        // Compilers.
        || l.contains("error[")            // rustc, with an error code
        || l.contains("could not compile") // cargo
        || l.contains("error ts")          // tsc: "error TS2322"
        || l.contains(": error:")          // gcc/clang/go: file:line:col: error:
        || l.contains(": warning:")
}

/// Whether a line *begins* a multi-line compiler diagnostic whose whole block
/// (location, code frame, notes) should be kept together.
fn starts_diagnostic_block(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("error[")          // rustc:  error[E0308]: mismatched types
        || t.starts_with("error:")   // rustc/cargo:  error: cannot find value …
        || t.starts_with("warning:") // rustc:  warning: unused variable …
        || line.contains(": error:") // gcc/clang/go:  src/x.c:4:9: error: …
        || line.contains(": warning:")
        || line.contains("error TS") // tsc:  x.ts(1,2): error TS2322: …
}

/// Whether a line carries failure/summary signal worth keeping (test-runner
/// output). Broad on purpose — see [`condense_diagnostics`]'s over-keep bias.
fn is_signal_line(line: &str) -> bool {
    // pytest prints assertion detail on lines that start with `E ` but may carry
    // no keyword of their own.
    if line.starts_with("E ") {
        return true;
    }
    let l = line.to_ascii_lowercase();
    const SIGNALS: [&str; 14] = [
        "fail",     // "test x ... FAILED", "failures:", "N failed"
        "error",    // "error[E..]", "error:", pytest "ERROR"
        "panic",    // rust panic
        "assert",   // assertion failed / pytest assert
        "thread '", // rust panic location
        "left:",    // assert_eq! diff
        "right:",
        "exception", // python
        "traceback", // python
        "expected",  // assertions
        "could not compile",
        "test result:", // libtest summary
        "short test summary",
        "=====", // pytest section rules (FAILURES / summary)
    ];
    SIGNALS.iter().any(|p| l.contains(p))
}

/// Clip to `max` chars keeping *both* ends. For command and test output the tail
/// — failures, summaries, `error[...]` lines — is usually the most useful part,
/// so head-only truncation drops the signal. Splits the budget ~60% head / ~40%
/// tail and notes how much of the middle went. Split out so tests can set `max`.
fn truncate_to(s: &str, max: usize) -> String {
    let total = s.chars().count();
    if total <= max {
        return s.to_string();
    }
    let head_budget = max * 6 / 10;
    let tail_budget = max - head_budget;
    let head: String = s.chars().take(head_budget).collect();
    let tail: String = s.chars().skip(total - tail_budget).collect();
    let dropped = total - head_budget - tail_budget;
    format!("{head}\n… [truncated {dropped} characters] …\n{tail}")
}

/// Render a file for the `read` tool: each line prefixed with its 1-based number
/// and a tab (so the model can cite and edit precisely), optionally restricted
/// to `[offset, offset+limit)`. When no limit is provided, return a bounded
/// page. A footer notes when lines were omitted so the model knows to page a
/// large file with `offset`/`limit` rather than assume it saw everything.
fn format_read(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    if content.is_empty() {
        return "(empty file)".to_string();
    }
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = offset.unwrap_or(1).max(1);
    if start > total {
        return format!("(file has {total} line(s); offset {start} is past the end)");
    }
    let limit = limit.unwrap_or(DEFAULT_READ_LIMIT);
    let end = start.saturating_add(limit).saturating_sub(1).min(total);
    let width = end.to_string().len().max(4);
    let mut out = String::new();
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let n = start + i;
        out.push_str(&format!("{n:>width$}\t{line}\n"));
    }
    if start > 1 || end < total {
        out.push_str(&format!("… showing lines {start}-{end} of {total}"));
        if end < total {
            out.push_str(&format!(" — read more with offset {}", end + 1));
        }
    }
    out
}

/// Best-effort: run a file-scoped formatter if one is installed for this file
/// type, then return the file's final content (for the diff shown to the user).
/// Never fails the edit — a missing formatter, or a formatter that errors on
/// not-yet-valid code, just leaves the file exactly as written.
async fn maybe_format(path: &str, written: String) -> String {
    // Opt-in: formatters churn unrelated lines in repos that aren't
    // formatter-clean. Disabled by default; set `HI_FORMAT=1` to enable.
    if std::env::var_os("HI_FORMAT").is_none() {
        return written;
    }
    let Some((probe, command)) = formatter_for(path) else {
        return written;
    };
    if !tool_available(probe).await {
        return written;
    }
    let _ = run_bash(&format!("{command} {}", sh_quote(path))).await;
    tokio::fs::read_to_string(path).await.unwrap_or(written)
}

/// The (probe binary, command prefix) of a file-scoped formatter for `path`'s
/// extension, if we support one. The command is run as `<prefix> <file>`.
fn formatter_for(path: &str) -> Option<(&'static str, &'static str)> {
    match Path::new(path).extension()?.to_str()? {
        "rs" => Some(("rustfmt", "rustfmt")),
        "go" => Some(("gofmt", "gofmt -w")),
        "py" => Some(("ruff", "ruff format -q")),
        "js" | "jsx" | "ts" | "tsx" | "json" | "css" | "scss" | "md" | "html" | "yaml" | "yml" => {
            Some(("prettier", "prettier --write --log-level warn"))
        }
        _ => None,
    }
}

/// Cached results of `tool_available` probes — the answer never changes within
/// a session, so we avoid a fork+exec per edit.
static TOOL_AVAILABLE_CACHE: LazyLock<Mutex<HashMap<String, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Whether `prog` is on PATH (so we only invoke formatters that exist).
/// Results are cached per-session: the probe is a fork+exec that takes
/// ~5-20ms, and it's called on every write/edit. Cached after first call.
async fn tool_available(prog: &str) -> bool {
    if let Some(&result) = TOOL_AVAILABLE_CACHE.lock().unwrap().get(prog) {
        return result;
    }
    let result = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {}", sh_quote(prog)))
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false);
    TOOL_AVAILABLE_CACHE
        .lock()
        .unwrap()
        .insert(prog.to_string(), result);
    result
}

/// Heuristic: does `bytes` look like a binary file? A NUL byte in the first 8 KB
/// is the classic signal (ripgrep uses the same heuristic). Empty files are not
/// binary. This lets `grep` and `read` skip/guard against non-text files instead
/// of failing opaquely on `read_to_string`.
fn is_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let probe = &bytes[..bytes.len().min(8192)];
    probe.contains(&0)
}

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
    /// 1-based first line to return (default: start of file).
    #[serde(default)]
    offset: Option<usize>,
    /// Max number of lines to return (default: to end of file).
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct MultiEditArgs {
    path: String,
    edits: Vec<EditOp>,
}

#[derive(Deserialize)]
struct EditOp {
    old_string: String,
    new_string: String,
}

#[derive(Deserialize)]
struct WriteArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct EditArgs {
    path: String,
    old_string: String,
    new_string: String,
    /// If true, replace every occurrence of `old_string` (default: false).
    #[serde(default)]
    replace_all: bool,
}

#[derive(Deserialize)]
struct ListArgs {
    #[serde(default)]
    path: Option<String>,
}

#[derive(Deserialize)]
struct GrepArgs {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    /// Lines of context to show around each match (default: 0).
    #[serde(default)]
    context: Option<usize>,
    /// File name glob to filter (e.g. `*.rs`). Only files whose name matches
    /// are searched.
    #[serde(default)]
    glob: Option<String>,
}

#[derive(Deserialize)]
struct BashArgs {
    command: String,
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_READ_LIMIT, apply_edit, apply_multi_patch, condense_diagnostics,
        condense_enabled, diff, edit_not_found_help, format_read, truncate_to,
    };

    #[test]
    fn condense_toggle_defaults_on_and_parses_off_values() {
        assert!(condense_enabled(None), "default on when unset");
        assert!(condense_enabled(Some("1")), "any other value is on");
        assert!(condense_enabled(Some("on")));
        for off in ["0", "off", "false", "no"] {
            assert!(!condense_enabled(Some(off)), "{off} disables");
        }
    }

    /// A realistic libtest run: a `header`, `passing` "... ok" lines, an injected
    /// failure block somewhere in the middle, and the final summary.
    fn cargo_log(passing: usize, fail_at: usize) -> String {
        let mut out = format!("\nrunning {} tests\n", passing + 1);
        for i in 0..passing {
            if i == fail_at {
                out.push_str("test mod::middle_case ... FAILED\n");
            }
            out.push_str(&format!("test mod::case_{i:04} ... ok\n"));
        }
        out.push_str(
            "\nfailures:\n\n---- mod::middle_case stdout ----\n\
             thread 'mod::middle_case' panicked at src/lib.rs:42:5:\n\
             assertion `left == right` failed\n  left: 3\n right: 4\n\n\
             failures:\n    mod::middle_case\n\n\
             test result: FAILED. {passing} passed; 1 failed; 0 ignored\n",
        );
        out
    }

    #[test]
    fn condense_keeps_cargo_failure_and_summary_drops_green() {
        let log = cargo_log(400, 200);
        let out = condense_diagnostics(&log, 50_000);
        // The failure, its panic detail, and the summary all survive…
        assert!(
            out.contains("middle_case ... FAILED"),
            "keeps the failing test"
        );
        assert!(out.contains("left: 3"), "keeps the assertion detail");
        assert!(out.contains("test result: FAILED"), "keeps the summary");
        // …while the green noise is collapsed.
        assert!(out.contains("lines omitted"), "drops passing lines: {out}");
        assert!(
            out.len() < log.len() / 3,
            "much smaller: {} vs {}",
            out.len(),
            log.len()
        );
    }

    #[test]
    fn condense_beats_head_tail_when_failure_is_in_the_middle() {
        // The money case: one failure buried in the middle of a long green run,
        // with a budget tight enough that blind head+tail would clip it out.
        let log = cargo_log(1000, 500);
        let budget = 8_000;
        assert!(
            !truncate_to(&log, budget).contains("middle_case ... FAILED"),
            "head+tail drops the middle failure"
        );
        assert!(
            condense_diagnostics(&log, budget).contains("middle_case ... FAILED"),
            "condense preserves it"
        );
    }

    #[test]
    fn condense_keeps_pytest_failures() {
        let log = "\
collected 50 items

tests/test_a.py ..........................................         [ 96%]
tests/test_b.py F                                                  [100%]

=================================== FAILURES ===================================
________________________________ test_parsing _________________________________

    def test_parsing():
>       assert parse('1+1') == 3
E       assert 2 == 3

tests/test_b.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_b.py::test_parsing - assert 2 == 3
========================= 1 failed, 49 passed in 0.42s =========================
";
        let out = condense_diagnostics(log, 50_000);
        assert!(out.contains("test_parsing"), "keeps the failing test name");
        assert!(
            out.contains("assert 2 == 3"),
            "keeps the assertion (E line)"
        );
        assert!(out.contains("1 failed, 49 passed"), "keeps the summary");
    }

    #[test]
    fn condense_passes_through_non_test_output() {
        // A plain file/command dump (no test markers) is left untouched when it
        // fits — condense must not mangle ordinary output.
        let dump = "fn main() {\n    println!(\"hello\");\n}\n";
        assert_eq!(condense_diagnostics(dump, 50_000), dump);
    }

    #[test]
    fn condense_keeps_whole_rustc_error_block() {
        // A `cargo build` run: a wall of "Compiling …" noise, one multi-line
        // rustc diagnostic (location + code frame + note), then the summary.
        let mut log = String::new();
        for i in 0..40 {
            log.push_str(&format!("   Compiling crate_{i} v0.1.0 (/tmp/crate_{i})\n"));
        }
        log.push_str(
            "error[E0308]: mismatched types\n  \
             --> src/lib.rs:42:18\n   |\n\
             42 |     let x: u32 = \"hi\";\n   \
             |            ---   ^^^^ expected `u32`, found `&str`\n   |\n   \
             = note: expected type `u32`\n\n\
             error: could not compile `app` (lib) due to 1 previous error\n",
        );
        let out = condense_diagnostics(&log, 50_000);
        // The entire diagnostic block survives — code, the caret line, the note…
        assert!(out.contains("error[E0308]"), "keeps the error line");
        assert!(out.contains("--> src/lib.rs:42:18"), "keeps the location");
        assert!(
            out.contains("expected `u32`, found `&str`"),
            "keeps the code frame / caret line"
        );
        assert!(out.contains("= note: expected type"), "keeps the note");
        assert!(out.contains("could not compile"), "keeps the summary");
        // …while the "Compiling …" noise is dropped.
        assert!(
            out.contains("lines omitted"),
            "drops the compile noise: {out}"
        );
    }

    #[test]
    fn condense_keeps_tsc_errors() {
        let mut log = String::from("> tsc --noEmit\n\n");
        for i in 0..40 {
            log.push_str(&format!("  checking module_{i:02}.ts\n"));
        }
        log.push_str(
            "src/index.ts(10,7): error TS2322: Type 'string' is not assignable to type 'number'.\n\
             src/index.ts(15,3): error TS2554: Expected 1 arguments, but got 0.\n\n\
             Found 2 errors in the same file, starting at: src/index.ts:10\n",
        );
        let out = condense_diagnostics(&log, 50_000);
        assert!(out.contains("error TS2322"), "keeps the first tsc error");
        assert!(out.contains("error TS2554"), "keeps the second tsc error");
        assert!(out.contains("Found 2 errors"), "keeps the summary");
        assert!(out.contains("lines omitted"), "drops the checking noise");
    }

    #[test]
    fn truncate_keeps_head_and_tail() {
        // 300 chars, budget 100 → keep 60 head + 40 tail, drop the 200 middle.
        let s = format!("{}{}{}", "A".repeat(100), "M".repeat(100), "Z".repeat(100));
        let out = truncate_to(&s, 100);
        assert!(out.starts_with(&"A".repeat(60)), "keeps the head");
        assert!(out.trim_end().ends_with(&"Z".repeat(40)), "keeps the tail");
        assert!(!out.contains('M'), "drops the middle");
        assert!(
            out.contains("truncated 200 characters"),
            "notes the gap: {out}"
        );
        // Under budget passes through untouched.
        assert_eq!(truncate_to("short", 100), "short");
    }

    #[test]
    fn read_numbers_lines_and_pages() {
        let body = "alpha\nbravo\ncharlie\ndelta\n";
        // Whole file: every line numbered from 1.
        let all = format_read(body, None, None);
        assert!(all.contains("   1\talpha"), "{all}");
        assert!(all.contains("   4\tdelta"), "{all}");
        // A window keeps absolute line numbers and notes there's more below.
        let win = format_read(body, Some(2), Some(2));
        assert!(
            win.contains("   2\tbravo") && win.contains("   3\tcharlie"),
            "{win}"
        );
        assert!(
            !win.contains("alpha") && !win.contains("delta"),
            "windowed: {win}"
        );
        assert!(
            win.contains("lines 2-3 of 4") && win.contains("offset 4"),
            "footer: {win}"
        );
        let large = (1..=DEFAULT_READ_LIMIT + 2)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let page = format_read(&large, None, None);
        assert!(page.contains("   1\tline 1"), "{page}");
        assert!(
            page.contains(&format!(
                "{DEFAULT_READ_LIMIT:>4}\tline {DEFAULT_READ_LIMIT}"
            )),
            "{page}"
        );
        assert!(
            !page.contains(&format!("line {}", DEFAULT_READ_LIMIT + 1)),
            "{page}"
        );
        assert!(
            page.contains(&format!(
                "lines 1-{DEFAULT_READ_LIMIT} of {}",
                DEFAULT_READ_LIMIT + 2
            )) && page.contains(&format!("offset {}", DEFAULT_READ_LIMIT + 1)),
            "footer: {page}"
        );
        // Empty + past-end are handled.
        assert_eq!(format_read("", None, None), "(empty file)");
        assert!(format_read(body, Some(99), None).contains("past the end"));
    }

    #[test]
    fn edit_not_found_points_at_similar_lines() {
        let file = "fn a() {}\nfn target() {\n    do_thing();\n}\nfn b() {}\n";
        let help = edit_not_found_help(file, "fn target() {\n    do_OTHER();");
        assert!(help.contains("not found"), "{help}");
        // It surfaces the real nearby line with its number so the model can copy it.
        assert!(
            help.contains("fn target() {"),
            "shows the candidate: {help}"
        );
        assert!(help.contains("2\t"), "with a line number: {help}");
    }

    #[tokio::test]
    async fn multi_edit_applies_in_order_and_is_atomic() {
        let dir = std::env::temp_dir().join(format!("hi-medit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.txt");
        std::fs::write(&path, "one\ntwo\nthree\n").unwrap();
        let p = path.to_string_lossy();

        // Bypass path guard — temp files live outside the project workspace.
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        unsafe { std::env::set_var("HI_NO_PATH_GUARD", "1"); }

        // Two edits in one atomic call.
        let args = format!(
            r#"{{"path":"{p}","edits":[{{"old_string":"one","new_string":"1"}},{{"old_string":"three","new_string":"3"}}]}}"#
        );
        let out = super::execute("multi_edit", &args).await;
        assert!(out.content.contains("Applied 2 edits"), "{}", out.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "1\ntwo\n3\n");

        // A failing edit in the batch leaves the file untouched (atomic).
        let bad = format!(
            r#"{{"path":"{p}","edits":[{{"old_string":"1","new_string":"X"}},{{"old_string":"nope","new_string":"Y"}}]}}"#
        );
        let out = super::execute("multi_edit", &bad).await;
        assert!(
            out.content.contains("Error"),
            "should fail: {}",
            out.content
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "1\ntwo\n3\n",
            "file unchanged after a failed batch"
        );
        // Restore env.
        unsafe {
            if had_guard.is_none() { std::env::remove_var("HI_NO_PATH_GUARD"); }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn diff_leads_with_a_change_summary() {
        // The diff a write/edit shows the user must say what changed up front,
        // not just trail off into raw +/- lines.
        let out = diff("one\ntwo\n", "one\nTWO\nthree\n");
        let first = out.lines().next().unwrap();
        assert!(first.contains("2 additions"), "summary: {first:?}");
        assert!(first.contains("1 deletion"), "summary: {first:?}");
        // Singular form when exactly one line changes.
        let single = diff("a\n", "a\nb\n");
        assert!(
            single.lines().next().unwrap().contains("1 addition,"),
            "singular: {single:?}"
        );
        assert_eq!(diff("same\n", "same\n"), "(no changes)");
    }

    #[test]
    fn diff_shows_context_and_line_numbers() {
        // A change deep in a file must show its surrounding context with gutter
        // line numbers, so the reader can see *where* it lands — not just the
        // changed line floating context-free.
        let before = "a\nb\nc\nd\ne\nf\ng\n";
        let after = "a\nb\nc\nD\ne\nf\ng\n";
        let plain = strip_ansi(&diff(before, after));
        // Summary still leads.
        assert!(
            plain.lines().next().unwrap().contains("1 addition"),
            "summary: {plain}"
        );
        // Unchanged neighbours appear as context (proves we're not changed-only).
        assert!(plain.contains(" c\n") || plain.contains(" c"), "context: {plain}");
        // The change is on line 4, numbered, with both old and new sides shown.
        assert!(plain.contains("4 - d"), "removed line w/ number: {plain}");
        assert!(plain.contains("4 + D"), "added line w/ number: {plain}");
        // Distant lines (line 1) are NOT shown — only context around the change.
        assert!(!plain.contains("1   a") && !plain.contains("1 + a"), "far context elided: {plain}");
    }

    /// Strip ANSI SGR escapes (`\x1b[…m`) so tests can assert on plain text.
    fn strip_ansi(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c2 in chars.by_ref() {
                    if c2 == 'm' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn read_only_tools_are_classified() {
        assert!(super::is_read_only("read"));
        assert!(super::is_read_only("list"));
        assert!(super::is_read_only("grep"));
        assert!(super::is_read_only("diff"));
        // Mutating / effecting tools are not safe to run concurrently.
        assert!(!super::is_read_only("write"));
        assert!(!super::is_read_only("edit"));
        assert!(!super::is_read_only("bash"));
    }

    #[test]
    fn sh_quote_escapes_single_quotes() {
        assert_eq!(super::sh_quote("a b"), "'a b'");
        assert_eq!(super::sh_quote("it's"), "'it'\\''s'");
    }

    #[tokio::test]
    async fn grep_finds_a_known_symbol() {
        // Searches the repo's own source via the real tool.
        let out = super::execute("grep", r#"{"pattern":"fn tool_specs"}"#).await;
        assert!(out.content.contains("tool_specs"), "grep: {}", out.content);
    }

    #[tokio::test]
    async fn grep_glob_filters_by_extension() {
        // `fn tool_specs` is in lib.rs but not in any .py file. With a `*.py`
        // glob, grep should find no matches; without a glob it finds the .rs hit.
        let py = super::execute("grep", r#"{"pattern":"fn tool_specs","glob":"*.py"}"#).await;
        assert!(
            py.content.contains("no matches"),
            "glob *.py excludes the .rs hit: {}",
            py.content
        );
        let rs = super::execute("grep", r#"{"pattern":"fn tool_specs","glob":"*.rs"}"#).await;
        assert!(
            rs.content.contains("tool_specs"),
            "glob *.rs finds the hit: {}",
            rs.content
        );
    }

    #[tokio::test]
    async fn grep_skips_binary_files() {
        // A file with a NUL byte should be skipped, not error out.
        let dir = std::env::temp_dir().join(format!("hi-grep-bin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "hello world\n").unwrap();
        std::fs::write(dir.join("b.bin"), b"hello\x00world\n").unwrap();
        let p = dir.to_string_lossy();
        let out = super::execute("grep", &format!(r#"{{"pattern":"hello","path":"{p}"}}"#)).await;
        // Should find the match in a.txt but not error on b.bin.
        assert!(
            out.content.contains("a.txt"),
            "found text file: {}",
            out.content
        );
        assert!(
            !out.content.contains("Error"),
            "no error on binary: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn update_plan_records_steps_and_statuses() {
        let args = r#"{"steps":[
            {"title":"find leak","status":"done"},
            {"title":"fix walkers","status":"in_progress"},
            {"title":"add tests","status":"pending"}
        ]}"#;
        let out = super::execute("update_plan", args).await;
        let plan = out.plan.expect("plan is set");
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].status, super::PlanStatus::Done);
        // "in_progress" is a common model synonym for active.
        assert_eq!(plan[1].status, super::PlanStatus::Active);
        assert_eq!(plan[2].status, super::PlanStatus::Pending);
        // Model-facing content is a terse confirmation; the steps drive the
        // pinned tracker, so there's no transcript-echo display.
        assert!(out.content.contains("1/3"), "content: {}", out.content);
        assert!(out.display.is_none(), "plan should not echo to transcript");
    }

    #[tokio::test]
    async fn update_plan_rejects_empty() {
        let out = super::execute("update_plan", r#"{"steps":[]}"#).await;
        assert!(out.content.contains("Error"), "{}", out.content);
        assert!(out.plan.is_none());
    }

    #[tokio::test]
    async fn list_excludes_git_metadata_but_keeps_dotfiles() {
        // Regression: the model called `list` during a review and got flooded
        // with `.git/objects/...` paths. We walk hidden files (so real dotfiles
        // like `.github/` are visible) but must prune VCS metadata directories.
        let dir = std::env::temp_dir().join(format!("hi-list-vcs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".git/objects/ab")).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::create_dir_all(dir.join(".github/workflows")).unwrap();
        std::fs::write(dir.join(".git/config"), "[core]\n").unwrap();
        std::fs::write(dir.join(".git/objects/ab/cdef"), b"\x00\x01obj").unwrap();
        std::fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.join(".github/workflows/ci.yml"), "name: ci\n").unwrap();

        let p = dir.to_string_lossy();
        let out = super::execute("list", &format!(r#"{{"path":"{p}"}}"#)).await;

        assert!(
            !out.content.contains("/.git/"),
            ".git metadata must be pruned: {}",
            out.content
        );
        assert!(
            out.content.contains("main.rs"),
            "normal files still listed: {}",
            out.content
        );
        // Legit dotfiles must survive — we prune VCS dirs, not all hidden entries.
        assert!(
            out.content.contains(".github"),
            "non-VCS dotfiles preserved: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn grep_excludes_git_metadata() {
        // The same needle lives in a tracked file and in `.git` internals; grep
        // must surface only the tracked hit (covers whichever path runs — the
        // `rg` fast-path's exclusion globs or the inline walker's filter).
        let dir = std::env::temp_dir().join(format!("hi-grep-vcs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/lib.rs"), "let UNIQNEEDLE = 1;\n").unwrap();
        std::fs::write(dir.join(".git/config"), "UNIQNEEDLE\n").unwrap();

        let p = dir.to_string_lossy();
        let out =
            super::execute("grep", &format!(r#"{{"pattern":"UNIQNEEDLE","path":"{p}"}}"#)).await;

        assert!(
            out.content.contains("lib.rs"),
            "tracked hit found: {}",
            out.content
        );
        assert!(
            !out.content.contains(".git/config"),
            ".git must not be searched: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_rejects_binary_with_a_clear_message() {
        let dir = std::env::temp_dir().join(format!("hi-read-bin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        std::fs::write(&path, b"\x00\x01\x02binary\xff").unwrap();
        let p = path.to_string_lossy();
        // Bypass path guard — temp files live outside the project workspace.
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        unsafe { std::env::set_var("HI_NO_PATH_GUARD", "1"); }
        let out = super::execute("read", &format!(r#"{{"path":"{p}"}}"#)).await;
        unsafe {
            if had_guard.is_none() { std::env::remove_var("HI_NO_PATH_GUARD"); }
        }
        assert!(
            out.content.contains("binary file"),
            "clear binary message: {}",
            out.content
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn is_binary_detects_nul_bytes() {
        assert!(!super::is_binary(b"plain text\n"), "text is not binary");
        assert!(!super::is_binary(b""), "empty is not binary");
        assert!(super::is_binary(b"text\x00more"), "NUL → binary");
        // NUL beyond the 8 KB probe window is not detected (same as ripgrep).
        let mut big = vec![b'x'; 9000];
        big.push(0);
        assert!(
            !super::is_binary(&big),
            "NUL past 8 KB probe is not detected"
        );
    }

    #[tokio::test]
    async fn list_includes_source_files() {
        // Pass an explicit path instead of relying on the process cwd: other
        // tests mutate the shared cwd via set_current_dir, which makes a
        // default-path `.` listing racy under parallel execution.
        let manifest = env!("CARGO_MANIFEST_DIR");
        let out = super::execute("list", &format!(r#"{{"path":"{manifest}"}}"#)).await;
        assert!(out.content.contains("lib.rs"), "list: {}", out.content);
    }

    #[tokio::test]
    async fn bash_refuses_catastrophic_but_runs_safe() {
        let refused = super::execute("bash", r#"{"command":"rm -rf /"}"#).await;
        assert!(refused.content.contains("refused"), "{}", refused.content);
        let ok = super::execute("bash", r#"{"command":"echo hello-guard"}"#).await;
        assert!(ok.content.contains("hello-guard"), "{}", ok.content);
    }

    #[test]
    fn exact_unique_match() {
        assert_eq!(
            apply_edit("let x = 1;\n", "let x = 1;", "let x = 2;", false).unwrap(),
            "let x = 2;\n"
        );
    }

    #[test]
    fn missing_old_string_errors() {
        assert!(apply_edit("foo\n", "bar", "baz", false).is_err());
    }

    #[test]
    fn ambiguous_exact_match_errors() {
        assert!(apply_edit("x = 1\nx = 1\n", "x = 1", "y", false).is_err());
    }

    #[test]
    fn tolerates_trailing_whitespace() {
        // The file has a stray trailing space the model's old_string lacks.
        assert_eq!(
            apply_edit("a\nb \nc\n", "a\nb\nc", "a\nB\nc", false).unwrap(),
            "a\nB\nc\n"
        );
    }

    #[test]
    fn tolerates_crlf() {
        let out = apply_edit("a\r\nb\r\n", "a\nb", "X\nY", false).unwrap();
        assert!(out.contains('X') && out.contains('Y'));
    }

    #[test]
    fn tolerates_indentation_and_reindents() {
        // File indents 8 spaces; model used 4 — match anyway and re-indent `new`.
        assert_eq!(
            apply_edit(
                "def f():\n        return 0\n",
                "    return 0",
                "    return 1",
                false
            )
            .unwrap(),
            "def f():\n        return 1\n"
        );
    }

    #[test]
    fn ambiguous_flexible_match_errors() {
        // Two lines match once indentation is ignored — refuse rather than guess.
        assert!(apply_edit("  x\n  x\n", "x ", "y", false).is_err());
    }

    #[test]
    fn preserves_trailing_newline() {
        let out = apply_edit("first\nsecond\n", "second", "SECOND", false).unwrap();
        assert_eq!(out, "first\nSECOND\n");
    }

    #[test]
    fn replace_all_swaps_every_occurrence() {
        let out = apply_edit("a\nb\na\nb\n", "a", "X", true).unwrap();
        assert_eq!(out, "X\nb\nX\nb\n");
    }

    #[test]
    fn replace_all_with_no_match_errors() {
        assert!(apply_edit("a\nb\n", "z", "X", true).is_err());
    }

    #[test]
    fn replace_all_unique_still_works() {
        let out = apply_edit("only\n", "only", "once", true).unwrap();
        assert_eq!(out, "once\n");
    }

    #[test]
    fn edit_not_found_help_finds_similar_lines() {
        // The needle has a typo ("funciton" vs "function") — no exact or
        // substring hit, but the similarity fallback should still point at the
        // right line by shared words.
        let text = "fn funciton_add(a, b) {\n    a + b\n}\n";
        let msg = edit_not_found_help(text, "fn function_add(a, b) {");
        assert!(
            msg.contains("funciton_add"),
            "similarity fallback finds the typo'd line: {msg}"
        );
    }

    #[test]
    fn apply_multi_patch_adds_updates_and_deletes() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let orig_cwd = std::env::current_dir().unwrap();
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        let dir = std::env::temp_dir().join(format!(
            "hi-patch-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_current_dir(&dir).unwrap();
        // Set HI_NO_PATH_GUARD so temp-dir paths aren't rejected.
        // (unsafe: env mutation is unsafe in edition 2024.)
        unsafe { std::env::set_var("HI_NO_PATH_GUARD", "1"); }

        std::fs::write(dir.join("update.txt"), "line1\nline2\nline3\n").unwrap();
        std::fs::write(dir.join("delete.txt"), "bye\n").unwrap();

        let patch = "\
*** Begin Patch
*** Update File: update.txt
@@ line1 @@
 line1
 line2
-replaced
+line2b
 line3
*** Add File: created.txt
new content
*** Delete File: delete.txt
*** End Patch";
        let result = apply_multi_patch(patch).unwrap();

        // Update: context + added lines become the new file; `-replaced` is
        // dropped (it wasn't even in the file, but the format skips `-` lines).
        let updated = std::fs::read_to_string(dir.join("update.txt")).unwrap();
        assert!(updated.contains("line1"), "context kept");
        assert!(updated.contains("line2b"), "added line present");
        assert!(!updated.contains("replaced"), "removed line dropped");
        assert!(updated.contains("line3"), "trailing context kept");

        // Add: new file written with the given content.
        let created = std::fs::read_to_string(dir.join("created.txt")).unwrap();
        assert_eq!(created, "new content\n");

        // Delete: file removed.
        assert!(!dir.join("delete.txt").exists(), "deleted file is gone");

        // Result summary mentions all three operations.
        assert!(result.contains("updated"), "{result}");
        assert!(result.contains("added"), "{result}");
        assert!(result.contains("deleted"), "{result}");

        // Restore environment for other tests.
        unsafe {
            if had_guard.is_some() {
                std::env::set_var("HI_NO_PATH_GUARD", "1");
            } else {
                std::env::remove_var("HI_NO_PATH_GUARD");
            }
        }
        std::env::set_current_dir(&orig_cwd).ok();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_multi_patch_rejects_bad_envelope() {
        unsafe { std::env::set_var("HI_NO_PATH_GUARD", "1"); }
        assert!(apply_multi_patch("not a patch").is_err());
        assert!(apply_multi_patch("").is_err());
        unsafe { std::env::remove_var("HI_NO_PATH_GUARD"); }
    }
}
