use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use crate::transaction::{MutationPlan, PlannedFileMutation, resolve_workspace_target};

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
#[derive(Debug)]
#[cfg(test)]
pub(crate) struct PatchApplication {
    pub summary: String,
}

/// Parse and materialize the complete patch before committing any operation.
/// Unknown directives, duplicate targets, stale/ambiguous hunks, and a bad
/// envelope therefore fail without touching the workspace.
#[cfg(test)]
pub(crate) async fn apply_multi_patch_at(root: &Path, patch: &str) -> Result<PatchApplication> {
    apply_multi_patch_at_with_state(root, &crate::checkpoint::default_state_root(), patch).await
}

#[cfg(test)]
pub(crate) async fn apply_multi_patch_at_with_state(
    root: &Path,
    state_root: &Path,
    patch: &str,
) -> Result<PatchApplication> {
    let (plan, summary) = plan_multi_patch(root, state_root, patch)?;
    plan.commit()?;
    Ok(PatchApplication { summary })
}

pub(crate) fn plan_multi_patch(
    root: &Path,
    state_root: &Path,
    patch: &str,
) -> Result<(MutationPlan, String)> {
    let lines: Vec<&str> = patch
        .lines()
        .map(|line| line.trim_end_matches('\r'))
        .collect();
    ensure!(!lines.is_empty(), "patch is empty");
    ensure!(
        lines[0] == "*** Begin Patch",
        "patch must start with exact line '*** Begin Patch'"
    );
    ensure!(
        lines.last() == Some(&"*** End Patch"),
        "patch must end with exact line '*** End Patch'"
    );
    ensure!(
        lines[1..lines.len() - 1]
            .iter()
            .all(|line| *line != "*** Begin Patch" && *line != "*** End Patch"),
        "patch contains a duplicate envelope marker"
    );

    let mut mutations = Vec::new();
    let mut summaries = Vec::new();
    let mut i = 1usize;
    while i + 1 < lines.len() {
        let directive = lines[i];
        ensure!(
            !directive.is_empty(),
            "unexpected blank line outside a file operation"
        );
        let (kind, path) = if let Some(path) = directive.strip_prefix("*** Add File: ") {
            ("add", path)
        } else if let Some(path) = directive.strip_prefix("*** Update File: ") {
            ("update", path)
        } else if let Some(path) = directive.strip_prefix("*** Delete File: ") {
            ("delete", path)
        } else {
            bail!("unknown directive '{directive}'")
        };
        ensure!(
            !path.is_empty() && path.trim() == path,
            "invalid {kind} path in '{directive}'"
        );
        i += 1;
        let body_start = i;
        while i + 1 < lines.len() && !lines[i].starts_with("*** ") {
            i += 1;
        }
        let body = &lines[body_start..i];
        match kind {
            "add" => {
                let mut content = String::new();
                let prefixed = body.iter().all(|line| line.starts_with('+'));
                let verbatim = body.iter().all(|line| !line.starts_with('+'));
                ensure!(
                    prefixed || verbatim,
                    "add operation for {path} mixes '+'-prefixed and verbatim lines"
                );
                for line in body {
                    // Accept both the documented verbatim form and the common
                    // unified-diff `+line` form, but never a mixed encoding.
                    content.push_str(line.strip_prefix('+').unwrap_or(line));
                    content.push('\n');
                }
                mutations.push(PlannedFileMutation::add(path, content.into_bytes()));
                summaries.push(format!("+ added {path}"));
            }
            "delete" => {
                ensure!(
                    body.is_empty(),
                    "delete operation for {path} must not have a body"
                );
                mutations.push(PlannedFileMutation::delete(path));
                summaries.push(format!("- deleted {path}"));
            }
            "update" => {
                ensure!(!body.is_empty(), "update operation for {path} has no hunks");
                let target = resolve_workspace_target(root, Path::new(path))?;
                let bytes = std::fs::read(&target)
                    .with_context(|| format!("reading {path} (use *** Add File: to create)"))?;
                ensure!(!crate::read::is_binary(&bytes), "{path} is a binary file");
                let original = String::from_utf8(bytes)
                    .with_context(|| format!("{path} is not valid UTF-8"))?;
                let after = apply_hunk_patch_text(&original, body)
                    .with_context(|| format!("patching {path}"))?;
                let changes = body
                    .iter()
                    .filter(|line| line.starts_with('+') || line.starts_with('-'))
                    .count();
                mutations.push(PlannedFileMutation::update_from_preimage(
                    path,
                    original.as_bytes(),
                    after.into_bytes(),
                ));
                summaries.push(format!(
                    "~ updated {path} ({changes} change{})",
                    if changes == 1 { "" } else { "s" }
                ));
            }
            _ => unreachable!(),
        }
    }
    ensure!(!mutations.is_empty(), "patch contained no file operations");
    let plan = MutationPlan::new_with_state(root, state_root, mutations)?;
    Ok((plan, summaries.join("\n")))
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
#[cfg(test)]
fn apply_hunk_patch(orig_lines: &[&str], patch_lines: &[&str]) -> Result<String> {
    let mut original = orig_lines.join("\n");
    if !orig_lines.is_empty() {
        original.push('\n');
    }
    apply_hunk_patch_text(&original, patch_lines)
}

#[derive(Clone, Copy)]
struct SourceLine<'a> {
    content: &'a str,
    raw: &'a str,
}

fn apply_hunk_patch_text(original: &str, patch_lines: &[&str]) -> Result<String> {
    let source = source_lines(original);
    let mut hunks: Vec<Vec<&str>> = Vec::new();
    let mut current = Vec::new();
    for line in patch_lines {
        if line.starts_with("@@") {
            if !current.is_empty() {
                hunks.push(std::mem::take(&mut current));
            }
            continue;
        }
        ensure!(
            line.starts_with([' ', '+', '-']),
            "invalid hunk line {line:?}; expected a space, '+', '-', or '@@' prefix"
        );
        current.push(*line);
    }
    if !current.is_empty() {
        hunks.push(current);
    }
    ensure!(!hunks.is_empty(), "patch contains no hunk lines");

    let newline = dominant_newline(original);
    let had_trailing_newline = original.ends_with('\n');
    let mut cursor = 0usize;
    let mut out = String::new();
    for hunk in hunks {
        let expected: Vec<&str> = hunk
            .iter()
            .filter_map(|line| line.strip_prefix(' ').or_else(|| line.strip_prefix('-')))
            .collect();
        ensure!(
            !expected.is_empty(),
            "addition-only hunk has no unique insertion anchor"
        );
        let mut matches = Vec::new();
        if source.len() >= expected.len() {
            for start in cursor..=source.len() - expected.len() {
                if expected.iter().enumerate().all(|(offset, expected)| {
                    source[start + offset].content.trim_end() == expected.trim_end()
                }) {
                    matches.push(start);
                    if matches.len() > 1 {
                        break;
                    }
                }
            }
        }
        ensure!(
            matches.len() == 1,
            "hunk context must match one unique contiguous region (found {})",
            matches.len()
        );
        let start = matches[0];
        for line in &source[cursor..start] {
            out.push_str(line.raw);
        }
        let mut probe_index = start;
        let mut removed_endings = Vec::new();
        for line in &hunk {
            if line.starts_with('-') {
                removed_endings.push(source[probe_index].raw);
                probe_index += 1;
            } else if line.starts_with(' ') {
                probe_index += 1;
            }
        }
        let mut replacement_line = 0usize;
        let mut source_index = start;
        for line in hunk {
            if let Some(added) = line.strip_prefix('+') {
                out.push_str(added);
                out.push_str(
                    removed_endings
                        .get(replacement_line)
                        .and_then(|raw| line_ending(raw))
                        .unwrap_or(newline),
                );
                replacement_line += 1;
            } else if line.starts_with(' ') {
                out.push_str(source[source_index].raw);
                source_index += 1;
            } else if line.starts_with('-') {
                source_index += 1;
            }
        }
        cursor = source_index;
    }
    for line in &source[cursor..] {
        out.push_str(line.raw);
    }

    if !had_trailing_newline && out.ends_with('\n') {
        out.pop();
        if out.ends_with('\r') {
            out.pop();
        }
    } else if had_trailing_newline && !out.is_empty() && !out.ends_with('\n') {
        out.push_str(newline);
    }
    Ok(out)
}

fn line_ending(raw: &str) -> Option<&'static str> {
    if raw.ends_with("\r\n") {
        Some("\r\n")
    } else if raw.ends_with('\n') {
        Some("\n")
    } else {
        None
    }
}

fn source_lines(text: &str) -> Vec<SourceLine<'_>> {
    text.split_inclusive('\n')
        .map(|raw| {
            let content = raw
                .strip_suffix('\n')
                .unwrap_or(raw)
                .strip_suffix('\r')
                .unwrap_or_else(|| raw.strip_suffix('\n').unwrap_or(raw));
            SourceLine { content, raw }
        })
        .collect()
}

fn dominant_newline(text: &str) -> &'static str {
    let crlf = text.match_indices("\r\n").count();
    let lf = text.matches('\n').count().saturating_sub(crlf);
    if crlf > lf { "\r\n" } else { "\n" }
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
/// `display` half of an edit's [`crate::ToolOutcome`], so the extra context costs no
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
        return Ok(text.replace(old, &preserve_line_endings(new, old)));
    }
    match count {
        1 => return Ok(text.replacen(old, &preserve_line_endings(new, old), 1)),
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
        let replacement = preserve_line_endings(new, &text[start..end]);
        return Ok(splice(text, start, end, replacement));
    }

    // 3. Ignore all indentation, then re-indent `new` to match the file.
    if let Some((start, end, idx)) =
        find_unique_window(&lines, text.len(), &old_lines, |l| l.trim())
    {
        let file_indent = leading_ws(lines[idx].1);
        let old_indent = leading_ws(old_lines.first().copied().unwrap_or(""));
        let reindented =
            preserve_line_endings(&reindent(new, old_indent, file_indent), &text[start..end]);
        return Ok(splice(text, start, end, reindented));
    }

    bail!("{}", edit_not_found_help(text, old));
}

/// Rewrite only replacement newline bytes to follow the exact sequence in the
/// matched source span. This preserves CRLF and mixed-EOL files without
/// changing the model's textual content.
fn preserve_line_endings(replacement: &str, matched: &str) -> String {
    if !replacement.contains('\n') || !matched.contains('\n') {
        return replacement.to_string();
    }
    let endings: Vec<&str> = matched
        .split_inclusive('\n')
        .filter_map(|line| {
            if line.ends_with("\r\n") {
                Some("\r\n")
            } else if line.ends_with('\n') {
                Some("\n")
            } else {
                None
            }
        })
        .collect();
    if endings.is_empty() {
        return replacement.to_string();
    }
    let fallback = if endings.iter().filter(|ending| **ending == "\r\n").count()
        > endings.iter().filter(|ending| **ending == "\n").count()
    {
        "\r\n"
    } else {
        "\n"
    };
    let mut out = String::new();
    let mut newline_index = 0usize;
    for line in replacement.split_inclusive('\n') {
        if let Some(content) = line.strip_suffix('\n') {
            out.push_str(content.strip_suffix('\r').unwrap_or(content));
            out.push_str(endings.get(newline_index).copied().unwrap_or(fallback));
            newline_index += 1;
        } else {
            out.push_str(line);
        }
    }
    out
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
        if text[start..end].ends_with("\r\n") {
            replacement.push_str("\r\n");
        } else {
            replacement.push('\n');
        }
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
    use super::{apply_edit, apply_hunk_patch, apply_multi_patch_at, diff, edit_not_found_help};

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
        assert_eq!(out, "X\r\nY\r\n");
    }

    #[test]
    fn preserves_mixed_line_endings_in_matched_span() {
        let out = apply_edit("a\r\nb\nc\r\n", "a\nb\nc", "A\nB\nC", false).unwrap();
        assert_eq!(out, "A\r\nB\nC\r\n");
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
        let dir = std::env::temp_dir().join(format!(
            "hi-patch-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
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
        let result = apply_multi_patch_at(&dir, &patch).await.unwrap().summary;

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

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn apply_multi_patch_preserves_trailing_newline_state() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-patch-eof-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // A file with NO trailing newline: patching it must NOT add one.
        let no_nl = dir.join("no_nl.txt");
        std::fs::write(&no_nl, "alpha\nbeta").unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n-alpha\n+ALPHA\n beta\n*** End Patch",
            no_nl.display(),
        );
        apply_multi_patch_at(&dir, &patch).await.unwrap();
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
        apply_multi_patch_at(&dir, &patch).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&with_nl).unwrap(),
            "ALPHA\nbeta\n",
            "trailing newline preserved"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn apply_multi_patch_preserves_mixed_line_endings() {
        let dir = std::env::temp_dir().join(format!("hi-patch-mixed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("mixed.txt");
        std::fs::write(&file, b"one\r\ntwo\nthree\r\n").unwrap();
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n-one\n+ONE\n two\n-three\n+THREE\n*** End Patch",
            file.display()
        );
        let root = dir.clone();
        apply_multi_patch_at(&root, &patch).await.unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"ONE\r\ntwo\nTHREE\r\n");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_multi_patch_rejects_bad_envelope() {
        let dir = std::env::temp_dir().join(format!("hi-patch-envelope-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        assert!(apply_multi_patch_at(&dir, "not a patch").await.is_err());
        assert!(apply_multi_patch_at(&dir, "").await.is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_multi_patch_reports_unknown_directives() {
        // A patch with only unrecognized directives should name them in the
        // error so the model can see what went wrong (e.g. a typo).
        let dir = std::env::temp_dir().join(format!("hi-patch-unknown-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let patch = "*** Begin Patch\n*** UpdateFile: src/a.rs\n-old\n+new\n*** End Patch";
        let err = apply_multi_patch_at(&dir, patch)
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("unknown directive"),
            "should mention unknown directive: {err}"
        );
        assert!(
            err.contains("*** UpdateFile:"),
            "should name the offending directive: {err}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn apply_multi_patch_rejects_stale_context() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "hi-patch-stale-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "alpha\nbeta\ngamma\n").unwrap();

        // The context line "delta" is not in the file — must be rejected.
        let f = dir.join("f.txt");
        let patch = format!(
            "*** Begin Patch\n*** Update File: {}\n alpha\n delta\n+new\n*** End Patch",
            f.display(),
        );
        let result = apply_multi_patch_at(&dir, &patch).await;
        assert!(result.is_err(), "stale context should be rejected");
        // The file is untouched.
        assert_eq!(
            std::fs::read_to_string(dir.join("f.txt")).unwrap(),
            "alpha\nbeta\ngamma\n"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn later_conflict_leaves_every_patch_target_unchanged() {
        let dir = std::env::temp_dir().join(format!("hi-patch-atomic-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a"), "alpha\n").unwrap();
        std::fs::write(dir.join("b"), "beta\n").unwrap();
        let patch = "*** Begin Patch\n*** Update File: a\n-alpha\n+ALPHA\n*** Update File: b\n-stale\n+BETA\n*** End Patch";
        assert!(apply_multi_patch_at(&dir, patch).await.is_err());
        assert_eq!(std::fs::read_to_string(dir.join("a")).unwrap(), "alpha\n");
        assert_eq!(std::fs::read_to_string(dir.join("b")).unwrap(), "beta\n");
        let _ = std::fs::remove_dir_all(dir);
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
