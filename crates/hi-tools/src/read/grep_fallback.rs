//! Hermetic, bounded grep fallback used when ripgrep cannot be launched.

use std::io::Read;

use anyhow::{Context, Result, bail};
use regex::Regex;

use crate::ToolOutcome;
use crate::condense::truncate;

use super::discovery::is_searchable_entry;
use super::{MAX_GREP_FILE_BYTES, display_path};

/// `sandbox-exec` reports a missing relative `rg` as a child failure (exit 71
/// + `execvp()`), not as `ErrorKind::NotFound` from `Command::spawn`.
pub(super) fn ripgrep_binary_unavailable(execution: &crate::ProcessExecution) -> bool {
    let text = execution.model_content();
    text.contains("execvp()")
        || (execution.outcome.exit_code == Some(71)
            && (text.contains("No such file") || text.contains("sandbox-exec")))
}

pub(super) fn run_grep_fallback_sync(
    root: &std::path::Path,
    target: &str,
    pattern: &str,
    glob: Option<&str>,
    context: usize,
) -> Result<ToolOutcome> {
    let re = Regex::new(pattern).context("invalid regex")?;
    let mut builder = ignore::WalkBuilder::new(target);
    builder
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .require_git(false)
        .hidden(false)
        .filter_entry(is_searchable_entry);
    if let Some(glob) = glob {
        match ignore::overrides::OverrideBuilder::new(target).add(glob) {
            Ok(ovb) => match ovb.build() {
                Ok(ov) => {
                    builder.overrides(ov);
                }
                Err(e) => bail!("invalid glob `{glob}`: {e}"),
            },
            Err(e) => bail!("invalid glob `{glob}`: {e}"),
        }
    }
    let mut out = String::new();
    let mut count = 0u32;
    // Auto-size the match cap from the output budget: stop once `out` approaches
    // `MAX_OUTPUT_CHARS` rather than at a fixed match count. This adapts to the
    // context window — short matches yield more results, long lines fewer —
    // without a config knob. A floor of 50 ensures we always show *some* matches
    // even when lines are very long (truncate will clip the final string).
    let budget = *crate::condense::MAX_OUTPUT_CHARS;
    let walker = builder.build();
    for entry in walker {
        let entry = entry.with_context(|| format!("walking {target}"))?;
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let file_path = entry.path();
        let rel = display_path(root, file_path);
        // Read a bounded byte window before splitting into lines. Calling
        // `BufRead::read_line` directly lets one newline-free record allocate
        // the entire file before the line-count guard can run.
        let file = std::fs::File::open(file_path)
            .with_context(|| format!("opening {} while searching", file_path.display()))?;
        let mut bytes = Vec::new();
        file.take((MAX_GREP_FILE_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .with_context(|| format!("reading {} while searching", file_path.display()))?;
        if bytes.len() > MAX_GREP_FILE_BYTES {
            out.push_str(&format!(
                "{rel}: (skipped — file exceeds {MAX_GREP_FILE_BYTES} bytes; install ripgrep for full search)\n"
            ));
            continue;
        }
        // A NUL byte means this isn't text. Avoid UTF-8 work on binary blobs.
        if bytes.contains(&0) {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&bytes) else {
            continue;
        };
        // Read lines into a bounded index for context matching. The rg fast
        // path handles larger files without buffering; this fallback only runs
        // when rg isn't installed.
        const MAX_LINES_PER_FILE: usize = 50_000;
        let mut lines: Vec<(usize, &str)> = Vec::new();
        let mut too_large = false;
        for (index, line) in text.lines().enumerate() {
            if index >= MAX_LINES_PER_FILE {
                too_large = true;
                break;
            }
            lines.push((index + 1, line));
        }
        if too_large {
            out.push_str(&format!(
                "{rel}: (skipped — file exceeds {MAX_LINES_PER_FILE} lines; install ripgrep for full search)\n"
            ));
            continue;
        }
        for (idx, (_, line)) in lines.iter().enumerate() {
            if re.is_match(line) {
                let line_no = lines[idx].0;
                if context > 0 {
                    let start = idx.saturating_sub(context);
                    let end = (idx + context + 1).min(lines.len());
                    for (ctx_i, (ctx_no, ctx_line)) in
                        lines.iter().enumerate().take(end).skip(start)
                    {
                        let marker = if ctx_i == idx { ":" } else { "-" };
                        out.push_str(&format!("{rel}{marker}{}: {}\n", ctx_no, ctx_line));
                    }
                    out.push_str("--\n");
                } else {
                    out.push_str(&format!("{rel}:{line_no}: {line}\n"));
                }
                count += 1;
                // Auto-size: stop when we've filled the output budget. The
                // final `truncate` will clip to exactly `budget`, but we stop
                // early so we don't scan needlessly after the cap is reached.
                if out.len() >= budget && count >= 50 {
                    out.push_str("… (truncated — output budget reached)\n");
                    break;
                }
            }
        }
        if out.ends_with("output budget reached)\n") {
            break;
        }
    }
    let out = if out.is_empty() {
        format!("no matches for {pattern}")
    } else {
        out
    };
    Ok(ToolOutcome::plain(truncate(&out)))
}
