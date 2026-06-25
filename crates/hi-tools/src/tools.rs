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
use crate::paths::{READ_CACHE, validate_workspace_path};
use crate::read::{run_glob, run_grep, run_list, run_read};
use crate::{PlanStatus, PlanStep, ToolOutput};

const BASH_TIMEOUT: Duration = Duration::from_secs(120);
const CHECK_TIMEOUT: Duration = Duration::from_secs(300);

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

/// Whether a tool only observes state, with no side effects — so several can
/// run concurrently within one round. The mutating/effecting tools (`write`,
/// `edit`, `bash`) are excluded, since order and isolation matter for them.
pub fn is_read_only(name: &str) -> bool {
    matches!(name, "read" | "list" | "grep" | "glob" | "diff")
    // `apply_patch`, like `write`/`edit`, is effecting and never runs in a
    // parallel batch — order and isolation matter.
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
        "read" | "write" | "edit" | "multi_edit" => {
            value.get("path")?.as_str().map(str::to_string)
        }
        // list's path is optional (defaults to ".").
        "list" => value.get("path")?.as_str().map(str::to_string),
        // grep: prefer an explicit `path`; fall back to `glob` only as a hint
        // (a glob isn't a single file, so return None to avoid over-serializing).
        "grep" => value.get("path")?.as_str().map(str::to_string),
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
        return Ok(ToolOutput::plain(condense(
            &run_bash_streaming(&args.command, on_line).await?,
        )));
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
                cache.remove(&args.path);
            }
            let after = crate::read::maybe_format(&args.path, args.content).await;
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
            let after = crate::read::maybe_format(&args.path, after).await;
            Ok(ToolOutput::shown(
                format!("Applied {} edits to {}", args.edits.len(), args.path),
                diff(&before, &after),
            ))
        }
        "bash" => {
            let args: BashArgs = parse(arguments)?;
            Ok(ToolOutput::plain(condense(&run_bash(&args.command).await?)))
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
            let result = apply_multi_patch(&args.patch)?;
            Ok(ToolOutput::plain(result))
        }
        other => bail!("unknown tool: {other}"),
    }
}

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
    // Refuse the handful of irreversible operations a checkpoint can't undo.
    if let Some(reason) = crate::guard::catastrophic_op(command) {
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
}

#[cfg(test)]
mod tests {
    use super::{is_read_only, target_path};
    use crate::edit::sh_quote;

    #[test]
    fn read_only_tools_are_classified() {
        assert!(is_read_only("read"));
        assert!(is_read_only("list"));
        assert!(is_read_only("grep"));
        assert!(is_read_only("diff"));
        // Mutating / effecting tools are not safe to run concurrently.
        assert!(!is_read_only("write"));
        assert!(!is_read_only("edit"));
        assert!(!is_read_only("bash"));
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
        assert_eq!(
            target_path("list", r#"{"path":"sub"}"#),
            Some("sub".into())
        );
        // bash has no path → None (the safe-fallback case for dep inference).
        assert_eq!(target_path("bash", r#"{"command":"echo hi"}"#), None);
        // Malformed JSON → None (tolerant).
        assert_eq!(target_path("read", "not json"), None);
    }
}
