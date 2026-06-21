//! Built-in tools: `read`, `write`, `edit`, `bash`.
//!
//! Richer capabilities come from subprocess CLI tools the model invokes via
//! `bash` — not a plugin runtime — so this set stays intentionally small.

pub mod checkpoint;
pub mod guard;

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use hi_ai::ToolSpec;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

/// Cap tool output so a single read or noisy command can't blow the context.
const MAX_OUTPUT_CHARS: usize = 50_000;
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
            (output.status.success(), truncate(&text))
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
pub fn working_tree_diff() -> String {
    use std::process::Command as SyncCommand;

    let git = |args: &[&str]| SyncCommand::new("git").args(args).output();

    let tracked = match git(&["--no-pager", "-c", "color.ui=always", "diff", "HEAD"]) {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            // Fresh repo with no commits yet: diff against the empty tree instead.
            if stderr.contains("unknown revision") || stderr.contains("ambiguous argument") {
                git(&["--no-pager", "-c", "color.ui=always", "diff"])
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

/// The tools advertised to the model each turn.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "read".into(),
            description: "Read a UTF-8 text file and return its contents.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read." }
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
            description: "Replace a unique block of text in a file. old_string must occur once; whitespace and indentation differences are tolerated.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit." },
                    "old_string": { "type": "string", "description": "Exact text to replace; must be unique in the file." },
                    "new_string": { "type": "string", "description": "Replacement text." }
                },
                "required": ["path", "old_string", "new_string"]
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
            name: "grep".into(),
            description: "Search file contents for a regular expression (ripgrep if available, else grep), respecting .gitignore. Returns matching `path:line: text`. Use this to find where something is defined or used.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regular expression to search for." },
                    "path": { "type": "string", "description": "File or directory to search (default: the whole project)." }
                },
                "required": ["pattern"]
            }),
        },
    ]
}

/// The result of a tool call, split into `content` shown to the model and an
/// optional richer `display` for the UI (e.g. a colored diff). This keeps
/// edit/write feedback terse for the model while showing the user what changed.
pub struct ToolOutput {
    pub content: String,
    pub display: Option<String>,
}

impl ToolOutput {
    fn plain(content: String) -> Self {
        Self {
            content,
            display: None,
        }
    }

    fn shown(content: String, display: String) -> Self {
        Self {
            content,
            display: Some(display),
        }
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

async fn run(name: &str, arguments: &str) -> Result<ToolOutput> {
    match name {
        "read" => {
            let args: ReadArgs = parse(arguments)?;
            let content = tokio::fs::read_to_string(&args.path)
                .await
                .with_context(|| format!("reading {}", args.path))?;
            Ok(ToolOutput::plain(truncate(&content)))
        }
        "write" => {
            let args: WriteArgs = parse(arguments)?;
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
            Ok(ToolOutput::shown(
                format!("Wrote {} bytes to {}", args.content.len(), args.path),
                diff(&before, &args.content),
            ))
        }
        "edit" => {
            let args: EditArgs = parse(arguments)?;
            let before = tokio::fs::read_to_string(&args.path)
                .await
                .with_context(|| format!("reading {}", args.path))?;
            let after = apply_edit(&before, &args.old_string, &args.new_string)
                .with_context(|| format!("editing {}", args.path))?;
            tokio::fs::write(&args.path, &after)
                .await
                .with_context(|| format!("writing {}", args.path))?;
            Ok(ToolOutput::shown(
                format!("Edited {}", args.path),
                diff(&before, &after),
            ))
        }
        "bash" => {
            let args: BashArgs = parse(arguments)?;
            Ok(ToolOutput::plain(truncate(&run_bash(&args.command).await?)))
        }
        "list" => {
            let args: ListArgs = parse(arguments)?;
            let path = sh_quote(args.path.as_deref().unwrap_or("."));
            // git ls-files inside a repo (gitignore-aware); plain find otherwise.
            let cmd = format!(
                "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
                   git ls-files --cached --others --exclude-standard -- {path}; \
                 else find {path} -type f -not -path '*/.git/*'; fi 2>/dev/null \
                 | head -400"
            );
            let out = run_bash(&cmd).await?;
            let out = if out.trim() == "[no output]" {
                "(no files found)".to_string()
            } else {
                out
            };
            Ok(ToolOutput::plain(truncate(&out)))
        }
        "grep" => {
            let args: GrepArgs = parse(arguments)?;
            let pattern = sh_quote(&args.pattern);
            let path = sh_quote(args.path.as_deref().unwrap_or("."));
            // ripgrep (gitignore-aware) when present, else recursive grep.
            let cmd = format!(
                "if command -v rg >/dev/null 2>&1; then \
                   rg -n --no-heading --color=never -e {pattern} -- {path}; \
                 else grep -rnI -e {pattern} -- {path} 2>/dev/null; fi | head -200"
            );
            let out = run_bash(&cmd).await?;
            let out = if out.trim() == "[no output]" {
                format!("no matches for {}", args.pattern)
            } else {
                out
            };
            Ok(ToolOutput::plain(truncate(&out)))
        }
        other => bail!("unknown tool: {other}"),
    }
}

/// Single-quote a string for safe interpolation into an `sh -c` command.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// A compact colored diff (changed lines only) for UI display.
fn diff(before: &str, after: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    let mut out = String::new();
    for change in TextDiff::from_lines(before, after).iter_all_changes() {
        let (sign, color) = match change.tag() {
            ChangeTag::Delete => ("-", "\x1b[31m"),
            ChangeTag::Insert => ("+", "\x1b[32m"),
            ChangeTag::Equal => continue,
        };
        let line = change.value().trim_end_matches('\n');
        out.push_str(&format!("{color}{sign} {line}\x1b[0m\n"));
    }
    if out.is_empty() {
        out.push_str("(no changes)");
    }
    out
}

/// Replace `old` with `new` in `text`, tolerating the whitespace differences
/// that make models' exact-match edits fail. Strategies, in order:
///   1. exact, unique match;
///   2. line-based match ignoring trailing whitespace (also fixes CRLF);
///   3. line-based match ignoring all indentation, re-indenting `new` to fit.
///
/// Each requires a unique match so an edit is never applied ambiguously.
fn apply_edit(text: &str, old: &str, new: &str) -> Result<String> {
    if old.is_empty() {
        bail!("old_string is empty");
    }
    // 1. Exact.
    match text.matches(old).count() {
        1 => return Ok(text.replacen(old, new, 1)),
        n if n > 1 => bail!(
            "old_string is not unique ({n} matches); include more surrounding context to pick one"
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

    bail!(
        "old_string not found, even allowing for whitespace differences; \
         re-read the file and copy the exact text to replace"
    )
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
    // Refuse the handful of irreversible operations a checkpoint can't undo.
    if let Some(reason) = guard::catastrophic_op(command) {
        return Ok(format!(
            "⚠ refused: this command {reason}. It's blocked as irreversible — the per-turn \
             checkpoint can't undo it. If it's genuinely needed, ask the user to run it \
             themselves (they can also set HI_ALLOW_DANGEROUS=1 to disable this guard)."
        ));
    }
    let future = Command::new("sh").arg("-c").arg(command).output();
    let output = match tokio::time::timeout(BASH_TIMEOUT, future).await {
        Ok(result) => result.context("failed to spawn command")?,
        Err(_) => bail!("command timed out after {}s", BASH_TIMEOUT.as_secs()),
    };

    let mut out = String::new();
    out.push_str(&String::from_utf8_lossy(&output.stdout));
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&stderr);
    }
    if let Some(code) = output.status.code()
        && code != 0
    {
        out.push_str(&format!("\n[exit code {code}]"));
    }
    if out.is_empty() {
        out.push_str("[no output]");
    }
    Ok(out)
}

fn parse<T: for<'de> Deserialize<'de>>(arguments: &str) -> Result<T> {
    serde_json::from_str(arguments).context("invalid tool arguments")
}

/// Truncate output to a character budget, noting how much was dropped.
fn truncate(s: &str) -> String {
    if s.chars().count() <= MAX_OUTPUT_CHARS {
        return s.to_string();
    }
    let kept: String = s.chars().take(MAX_OUTPUT_CHARS).collect();
    let dropped = s.chars().count() - MAX_OUTPUT_CHARS;
    format!("{kept}\n… [truncated {dropped} characters]")
}

#[derive(Deserialize)]
struct ReadArgs {
    path: String,
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
}

#[derive(Deserialize)]
struct BashArgs {
    command: String,
}

#[cfg(test)]
mod tests {
    use super::apply_edit;

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
    async fn list_includes_source_files() {
        let out = super::execute("list", "{}").await;
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
            apply_edit("let x = 1;\n", "let x = 1;", "let x = 2;").unwrap(),
            "let x = 2;\n"
        );
    }

    #[test]
    fn missing_old_string_errors() {
        assert!(apply_edit("foo\n", "bar", "baz").is_err());
    }

    #[test]
    fn ambiguous_exact_match_errors() {
        assert!(apply_edit("x = 1\nx = 1\n", "x = 1", "y").is_err());
    }

    #[test]
    fn tolerates_trailing_whitespace() {
        // The file has a stray trailing space the model's old_string lacks.
        assert_eq!(
            apply_edit("a\nb \nc\n", "a\nb\nc", "a\nB\nc").unwrap(),
            "a\nB\nc\n"
        );
    }

    #[test]
    fn tolerates_crlf() {
        let out = apply_edit("a\r\nb\r\n", "a\nb", "X\nY").unwrap();
        assert!(out.contains('X') && out.contains('Y'));
    }

    #[test]
    fn tolerates_indentation_and_reindents() {
        // File indents 8 spaces; model used 4 — match anyway and re-indent `new`.
        assert_eq!(
            apply_edit(
                "def f():\n        return 0\n",
                "    return 0",
                "    return 1"
            )
            .unwrap(),
            "def f():\n        return 1\n"
        );
    }

    #[test]
    fn ambiguous_flexible_match_errors() {
        // Two lines match once indentation is ignored — refuse rather than guess.
        assert!(apply_edit("  x\n  x\n", "x ", "y").is_err());
    }

    #[test]
    fn preserves_trailing_newline() {
        let out = apply_edit("first\nsecond\n", "second", "SECOND").unwrap();
        assert_eq!(out, "first\nSECOND\n");
    }
}
