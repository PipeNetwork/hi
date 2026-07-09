use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::paths::{READ_CACHE, cache_key, validate_workspace_path};

/// Apply a multi-file patch in Claude's `apply_patch` format. The envelope is
/// `*** Begin Patch … *** End Patch`; inside, each `*** Update File:`,
/// `*** Add File:`, or `*** Delete File:` header introduces a file operation.
///
/// For updates, each line is either context (no prefix), a removal (`-`), or an
/// addition (`+`). Context and removed lines are validated against the original
/// file — each must appear, in order — so a stale patch is rejected rather than
/// silently corrupting the file. Added lines are inserted in place of removed
/// ones. `@@ … @@` hunk headers are skipped. Lines of the original not mentioned
/// in any hunk are preserved unchanged.
pub(crate) async fn apply_multi_patch(patch: &str) -> Result<String> {
    let mut results = Vec::new();
    let mut unknown: Vec<&str> = Vec::new();
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
                tokio::fs::create_dir_all(parent)
                    .await
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
            tokio::fs::write(path, &content)
                .await
                .with_context(|| format!("writing {path}"))?;
            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(&cache_key(path));
            }
            results.push(format!("+ added {path}"));
            continue;
        }

        // Delete File: remove the file if it exists.
        if let Some(path) = line.strip_prefix("*** Delete File: ") {
            let path = path.trim();
            validate_workspace_path(path)?;
            tokio::fs::remove_file(path)
                .await
                .with_context(|| format!("deleting {path}"))?;
            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(&cache_key(path));
            }
            results.push(format!("- deleted {path}"));
            i += 1;
            continue;
        }

        // Update File: apply a hunk-style patch to the file. Context lines (no
        // prefix) and removed lines (`-`) are validated against the original —
        // each must appear, in order, so a stale or wrong context line is
        // rejected rather than silently overwriting real content. Added lines
        // (`+`) are inserted in place of removed ones. `@@ … @@` hunk headers
        // are delimiters, skipped. Lines of the original not mentioned in any
        // hunk are preserved unchanged.
        if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = path.trim();
            validate_workspace_path(path)?;
            let original = crate::read::read_text_file(path)
                .await
                .with_context(|| format!("reading {path} (use *** Add File: to create)"))?;
            let orig_lines: Vec<&str> = original.lines().collect();

            // Collect this file's patch lines (until the next `*** ` directive).
            let mut patch_lines: Vec<&str> = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with("*** ") {
                patch_lines.push(lines[i]);
                i += 1;
            }

            let mut after = apply_hunk_patch(&orig_lines, &patch_lines)
                .with_context(|| format!("patching {path}"))?;
            // Preserve the original's trailing-newline state. `str::lines()` +
            // apply_hunk_patch always re-emit a final '\n', which would silently
            // add one to a file that had none — corrupting its EOF state and
            // showing a spurious "No newline at end of file" churn line in diffs.
            if !original.ends_with('\n') && after.ends_with('\n') {
                after.pop();
            }

            if let Ok(mut cache) = READ_CACHE.lock() {
                cache.remove(&cache_key(path));
            }
            tokio::fs::write(path, &after)
                .await
                .with_context(|| format!("writing {path}"))?;
            let changes = patch_lines
                .iter()
                .filter(|l| l.starts_with('+') || l.starts_with('-'))
                .count();
            results.push(format!(
                "~ updated {path} ({changes} change{})",
                if changes == 1 { "" } else { "s" }
            ));
            continue;
        }

        // Unknown directive — collect it and skip. If no recognized operations
        // were found, we'll include the unknown directives in the error so the
        // model can see *why* (e.g. a typo like `*** UpdateFile:` missing a
        // space) instead of a bare "no operations" message.
        unknown.push(line);
        i += 1;
    }

    if results.is_empty() {
        if unknown.is_empty() {
            bail!("patch contained no file operations");
        } else {
            let preview: Vec<String> = unknown.iter().take(3).map(|d| format!("'{d}'")).collect();
            bail!(
                "patch contained no file operations (unknown directive{}: {})",
                if unknown.len() == 1 { "" } else { "s" },
                preview.join(", ")
            );
        }
    }
    Ok(results.join("\n"))
}

/// Apply a hunk-style patch to a file's lines, validating context. Walks the
/// original lines with a cursor; for each patch line:
/// - context (space prefix) or `-` (removed): advance the cursor to the next
///   matching original line (comparing `trim_end()` so trailing whitespace / CRLF
///   differences don't cause spurious failures). Emit context lines; skip removed
///   lines. Lines the cursor skips over are emitted unchanged (preserved).
/// - `+` (added): emit immediately, cursor doesn't move.
/// - `@@ … @@`: hunk header, skipped.
///
/// If a context or removed line can't be found in the original (ahead of the
/// cursor), the patch is rejected — a stale context line would silently corrupt
/// the file otherwise. The trailing newline of the original is preserved.
fn apply_hunk_patch(orig_lines: &[&str], patch_lines: &[&str]) -> Result<String> {
    let mut out = String::new();
    let mut cursor = 0usize; // next unread line in orig_lines

    for pl in patch_lines {
        if pl.trim_start().starts_with("@@") {
            continue;
        }
        if let Some(added) = pl.strip_prefix('+') {
            out.push_str(added);
            out.push('\n');
            continue;
        }
        // Context line (space prefix) or removed line (`-`). The first char is
        // the prefix; the rest is the line content to match against the original.
        let (expected, is_removal) = match pl.strip_prefix('-') {
            Some(rest) => (rest, true),
            None => {
                // Context line: strip the leading space prefix. If the line is
                // empty or doesn't start with a space, use it as-is (tolerant).
                let content = pl.strip_prefix(' ').unwrap_or(pl);
                (content, false)
            }
        };
        // Search forward from the cursor for a matching line (trim_end tolerates
        // trailing whitespace and CRLF differences).
        let norm_expected = expected.trim_end();
        let found = orig_lines[cursor..]
            .iter()
            .position(|l| l.trim_end() == norm_expected);
        let Some(rel) = found else {
            bail!(
                "context line not found in the original file: {:?} \
                 (searched from line {}). The patch may be stale — re-read \
                 the file and regenerate the patch.",
                expected,
                cursor + 1
            );
        };
        let abs = cursor + rel;
        // Emit any skipped original lines (unchanged context between hunks).
        for line in &orig_lines[cursor..abs] {
            out.push_str(line);
            out.push('\n');
        }
        // For a context line, emit the original line (preserving its exact
        // whitespace); for a `-` line, drop it.
        if !is_removal {
            out.push_str(orig_lines[abs]);
            out.push('\n');
        }
        cursor = abs + 1;
    }

    // Emit any remaining original lines (trailing context not mentioned in the patch).
    for line in &orig_lines[cursor..] {
        out.push_str(line);
        out.push('\n');
    }

    Ok(out)
}

/// Single-quote a string for safe interpolation into an `sh -c` command.
pub(crate) fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// A compact, readable diff for UI display: a bold one-line summary of how many
/// lines changed, then each changed region shown with a few lines of unchanged
/// context, gutter line numbers, and `±` signs. The context and line numbers let
/// the reader see *where* an edit lands, not just what changed; non-adjacent
/// regions are separated by a dim `⋯`. The model never sees this — it's the
/// `display` half of an edit's [`crate::ToolOutput`], so the extra context costs no
/// tokens. (`/diff` shows the full working-tree diff.)
pub(crate) fn diff(before: &str, after: &str) -> String {
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
pub(crate) fn apply_edit(text: &str, old: &str, new: &str, replace_all: bool) -> Result<String> {
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

    // Reaching the fuzzy strategies with `replace_all` means there were zero
    // *exact* matches (the >0 case returned above). The fuzzy strategies below
    // require a unique match and splice a single span, so they cannot honor
    // replace_all — fail loudly rather than silently replacing one fuzzy
    // occurrence and reporting success on a half-edited file.
    if replace_all {
        bail!(
            "replace_all found no exact occurrences of old_string; fuzzy matching can't \
             safely replace every occurrence — re-read the file and use the exact text, \
             or make old_string unique and drop replace_all"
        );
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
pub(crate) fn edit_not_found_help(text: &str, old: &str) -> String {
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
pub(crate) fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

/// Byte offset and trailing-newline-stripped content of each line in `text`.
pub(crate) fn lines_with_offsets(text: &str) -> Vec<(usize, &str)> {
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
pub(crate) fn find_unique_window(
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
pub(crate) fn splice(text: &str, start: usize, end: usize, mut replacement: String) -> String {
    if text[start..end].ends_with('\n') && !replacement.ends_with('\n') {
        replacement.push('\n');
    }
    format!("{}{}{}", &text[..start], replacement, &text[end..])
}

pub(crate) fn leading_ws(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// Rebase `new`'s indentation from `old_indent` to `file_indent`, preserving
/// each line's relative nesting.
pub(crate) fn reindent(new: &str, old_indent: &str, file_indent: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::{apply_edit, apply_hunk_patch, apply_multi_patch, diff, edit_not_found_help};

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
        assert!(
            plain.contains(" c\n") || plain.contains(" c"),
            "context: {plain}"
        );
        // The change is on line 4, numbered, with both old and new sides shown.
        assert!(plain.contains("4 - d"), "removed line w/ number: {plain}");
        assert!(plain.contains("4 + D"), "added line w/ number: {plain}");
        // Distant lines (line 1) are NOT shown — only context around the change.
        assert!(
            !plain.contains("1   a") && !plain.contains("1 + a"),
            "far context elided: {plain}"
        );
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
    fn replace_all_refuses_fuzzy_fallback() {
        // No EXACT match (CRLF differs) but a fuzzy line-match exists. With
        // replace_all we must NOT silently do a single fuzzy replacement and
        // report success — bail so the model re-reads and uses exact text.
        let err = apply_edit("a\r\nb\r\n", "a\nb", "X\nY", true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("replace_all"), "explains the refusal: {err}");
        // Without replace_all the same fuzzy edit still applies (single, unique).
        let out = apply_edit("a\r\nb\r\n", "a\nb", "X\nY", false).unwrap();
        assert!(out.contains('X') && out.contains('Y'), "{out:?}");
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

    #[tokio::test]
    async fn apply_multi_patch_adds_updates_and_deletes() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        let dir = std::env::temp_dir().join(format!(
            "hi-patch-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Set HI_NO_PATH_GUARD so temp-dir paths aren't rejected.
        // (unsafe: env mutation is unsafe in edition 2024.)
        unsafe {
            std::env::set_var("HI_NO_PATH_GUARD", "1");
        }

        std::fs::write(dir.join("update.txt"), "line1\nline2\nline3\n").unwrap();
        std::fs::write(dir.join("delete.txt"), "bye\n").unwrap();

        // Use absolute paths in the patch so the test doesn't depend on cwd
        // (which races with other async tests that also chdir).
        let upd = dir.join("update.txt");
        let cre = dir.join("created.txt");
        let del = dir.join("delete.txt");
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@ line1 @@\n line1\n-line2\n+line2b\n line3\n*** Add File: {}\nnew content\n*** Delete File: {}\n*** End Patch",
            upd.display(),
            cre.display(),
            del.display(),
        );
        let result = apply_multi_patch(&patch).await.unwrap();

        // Update: the `-line2` removal is validated against the original (it
        // must be present), then replaced by `+line2b`. Context lines are
        // preserved; unmentioned lines are kept.
        let updated = std::fs::read_to_string(dir.join("update.txt")).unwrap();
        assert!(updated.contains("line1"), "context kept");
        assert!(updated.contains("line2b"), "added line present");
        assert!(!updated.contains("line2\n"), "removed line dropped");
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn apply_multi_patch_preserves_trailing_newline_state() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        let dir = std::env::temp_dir().join(format!(
            "hi-patch-eof-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe {
            std::env::set_var("HI_NO_PATH_GUARD", "1");
        }

        // A file with NO trailing newline: patching it must NOT add one.
        let no_nl = dir.join("no_nl.txt");
        std::fs::write(&no_nl, "alpha\nbeta").unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n-alpha\n+ALPHA\n beta\n*** End Patch",
            no_nl.display(),
        );
        apply_multi_patch(&patch).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&no_nl).unwrap(),
            "ALPHA\nbeta",
            "no trailing newline is preserved (not silently added)"
        );

        // A file WITH a trailing newline keeps it.
        let with_nl = dir.join("with_nl.txt");
        std::fs::write(&with_nl, "alpha\nbeta\n").unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n-alpha\n+ALPHA\n beta\n*** End Patch",
            with_nl.display(),
        );
        apply_multi_patch(&patch).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&with_nl).unwrap(),
            "ALPHA\nbeta\n",
            "trailing newline preserved"
        );

        unsafe {
            if had_guard.is_some() {
                std::env::set_var("HI_NO_PATH_GUARD", "1");
            } else {
                std::env::remove_var("HI_NO_PATH_GUARD");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn apply_multi_patch_rejects_bad_envelope() {
        unsafe {
            std::env::set_var("HI_NO_PATH_GUARD", "1");
        }
        assert!(apply_multi_patch("not a patch").await.is_err());
        assert!(apply_multi_patch("").await.is_err());
        unsafe {
            std::env::remove_var("HI_NO_PATH_GUARD");
        }
    }

    #[tokio::test]
    async fn apply_multi_patch_reports_unknown_directives() {
        // A patch with only unrecognized directives should name them in the
        // error so the model can see what went wrong (e.g. a typo).
        unsafe {
            std::env::set_var("HI_NO_PATH_GUARD", "1");
        }
        let patch = "*** Begin Patch\n*** UpdateFile: src/a.rs\n-old\n+new\n*** End Patch";
        let err = apply_multi_patch(patch).await.unwrap_err().to_string();
        assert!(
            err.contains("unknown directive"),
            "should mention unknown directive: {err}"
        );
        assert!(
            err.contains("*** UpdateFile:"),
            "should name the offending directive: {err}"
        );
        unsafe {
            std::env::remove_var("HI_NO_PATH_GUARD");
        }
    }

    #[tokio::test]
    async fn apply_multi_patch_rejects_stale_context() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let had_guard = std::env::var_os("HI_NO_PATH_GUARD");
        let dir = std::env::temp_dir().join(format!(
            "hi-patch-stale-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe {
            std::env::set_var("HI_NO_PATH_GUARD", "1");
        }

        std::fs::write(dir.join("f.txt"), "alpha\nbeta\ngamma\n").unwrap();

        // The context line "delta" is not in the file — must be rejected.
        let f = dir.join("f.txt");
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n alpha\n delta\n+new\n*** End Patch",
            f.display(),
        );
        let result = apply_multi_patch(&patch).await;
        assert!(result.is_err(), "stale context should be rejected");
        // The file is untouched.
        assert_eq!(
            std::fs::read_to_string(dir.join("f.txt")).unwrap(),
            "alpha\nbeta\ngamma\n"
        );

        unsafe {
            if had_guard.is_some() {
                std::env::set_var("HI_NO_PATH_GUARD", "1");
            } else {
                std::env::remove_var("HI_NO_PATH_GUARD");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_hunk_patch_preserves_unmentioned_lines() {
        // A patch that only touches the middle of a file must preserve the
        // lines before and after the hunk — not replace the whole file.
        // Context lines are space-prefixed (unified-diff style).
        let orig = vec!["a", "b", "c", "d", "e", "f"];
        let patch = vec![" b", "-c", "+C", " d"];
        let out = apply_hunk_patch(&orig, &patch).unwrap();
        assert_eq!(out, "a\nb\nC\nd\ne\nf\n");
    }
}
