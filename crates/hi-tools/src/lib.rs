//! Built-in tools: `read`, `write`, `edit`, `bash`.
//!
//! Richer capabilities come from subprocess CLI tools the model invokes via
//! `bash` — not a plugin runtime — so this set stays intentionally small.

pub mod checkpoint;
pub mod guard;

use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use hi_ai::ToolSpec;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

/// Per-result character budget so a single read or noisy command can't blow the
/// context. Overridable via `HI_TOOL_RESULT_CHARS` — lower it for a tight local
/// window, raise it when the model has room. Read once, at first use. The
/// default is conservative on purpose: ~24k chars is ~6k tokens, and a single
/// `cargo test` dump shouldn't swallow a small context.
static MAX_OUTPUT_CHARS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("HI_TOOL_RESULT_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1_000)
        .unwrap_or(24_000)
});
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
            description: "Read a UTF-8 text file. Lines are returned numbered (`<n>\\t<text>`). For large files, page with offset/limit instead of assuming you saw everything.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read." },
                    "offset": { "type": "integer", "description": "1-based line to start at (default: first line)." },
                    "limit": { "type": "integer", "description": "Maximum number of lines to return (default: to end of file)." }
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
            description: "Replace a unique block of text in a file. old_string must occur once and be the file's literal text WITHOUT the `read` line-number gutter; whitespace and indentation differences are tolerated.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to edit." },
                    "old_string": { "type": "string", "description": "Exact text to replace; must be unique in the file. Do not include line numbers." },
                    "new_string": { "type": "string", "description": "Replacement text." }
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

/// Whether a tool only observes state, with no side effects — so several can
/// run concurrently within one round. The mutating/effecting tools (`write`,
/// `edit`, `bash`) are excluded, since order and isolation matter for them.
pub fn is_read_only(name: &str) -> bool {
    matches!(name, "read" | "list" | "grep")
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
            Ok(ToolOutput::plain(truncate(&format_read(
                &content,
                args.offset,
                args.limit,
            ))))
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
            let after = maybe_format(&args.path, args.content).await;
            Ok(ToolOutput::shown(
                format!("Wrote {} bytes to {}", after.len(), args.path),
                diff(&before, &after),
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
            let after = maybe_format(&args.path, after).await;
            Ok(ToolOutput::shown(
                format!("Edited {}", args.path),
                diff(&before, &after),
            ))
        }
        "multi_edit" => {
            let args: MultiEditArgs = parse(arguments)?;
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
                after = apply_edit(&after, &e.old_string, &e.new_string)
                    .with_context(|| format!("editing {} (edit #{})", args.path, i + 1))?;
            }
            tokio::fs::write(&args.path, &after)
                .await
                .with_context(|| format!("writing {}", args.path))?;
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

/// A compact colored diff (changed lines only) for UI display, led by a bold
/// one-line summary of how many lines were added and removed — so a write/edit
/// shows what happened at a glance instead of trailing off into raw diff lines.
fn diff(before: &str, after: &str) -> String {
    use similar::{ChangeTag, TextDiff};
    let mut body = String::new();
    let (mut adds, mut dels) = (0usize, 0usize);
    for change in TextDiff::from_lines(before, after).iter_all_changes() {
        let (sign, color) = match change.tag() {
            ChangeTag::Delete => {
                dels += 1;
                ("-", "\x1b[31m")
            }
            ChangeTag::Insert => {
                adds += 1;
                ("+", "\x1b[32m")
            }
            ChangeTag::Equal => continue,
        };
        let line = change.value().trim_end_matches('\n');
        body.push_str(&format!("{color}{sign} {line}\x1b[0m\n"));
    }
    if adds == 0 && dels == 0 {
        return "(no changes)".to_string();
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

    bail!("{}", edit_not_found_help(text, old));
}

/// Build a helpful error when `old_string` doesn't match: point the model at the
/// nearest similar lines (with numbers) so it can copy the exact text, rather
/// than blindly retrying the same string.
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
        let lo = i.saturating_sub(1);
        let hi = (i + 2).min(lines.len());
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
        "fail",             // "test x ... FAILED", "failures:", "N failed"
        "error",            // "error[E..]", "error:", pytest "ERROR"
        "panic",            // rust panic
        "assert",           // assertion failed / pytest assert
        "thread '",         // rust panic location
        "left:",            // assert_eq! diff
        "right:",
        "exception",        // python
        "traceback",        // python
        "expected",         // assertions
        "could not compile",
        "test result:",     // libtest summary
        "short test summary",
        "=====",            // pytest section rules (FAILURES / summary)
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
/// to `[offset, offset+limit)`. A footer notes when lines were omitted so the
/// model knows to page a large file with `offset`/`limit` rather than assume it
/// saw everything.
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
    let end = match limit {
        Some(n) => start.saturating_add(n).saturating_sub(1).min(total),
        None => total,
    };
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
    // Opt-out: some repos aren't formatter-clean, so reformatting on edit would
    // churn unrelated lines. `HI_NO_FORMAT=1` disables it.
    if std::env::var_os("HI_NO_FORMAT").is_some() {
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

/// Whether `prog` is on PATH (so we only invoke formatters that exist).
async fn tool_available(prog: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {}", sh_quote(prog)))
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
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
    use super::{
        apply_edit, condense_diagnostics, condense_enabled, diff, edit_not_found_help, format_read,
        truncate_to,
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
        assert!(out.contains("middle_case ... FAILED"), "keeps the failing test");
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
        assert!(out.contains("assert 2 == 3"), "keeps the assertion (E line)");
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
        assert!(out.contains("lines omitted"), "drops the compile noise: {out}");
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
    fn read_only_tools_are_classified() {
        assert!(super::is_read_only("read"));
        assert!(super::is_read_only("list"));
        assert!(super::is_read_only("grep"));
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
