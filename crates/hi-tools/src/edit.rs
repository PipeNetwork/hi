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
                crate::read::refuse_oversized_text(path, content.len())?;
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
                let bytes = crate::read::read_regular_file_bytes(&target)
                    .with_context(|| format!("reading {path} (use *** Add File: to create)"))?;
                ensure!(!crate::read::is_binary(&bytes), "{path} is a binary file");
                let original = String::from_utf8(bytes)
                    .with_context(|| format!("{path} is not valid UTF-8"))?;
                let after = apply_hunk_patch_text(&original, body)
                    .with_context(|| format!("patching {path}"))?;
                crate::read::refuse_oversized_text(path, after.len())?;
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
                // A context anchor may be the original unterminated EOF line.
                // Appending a new line must terminate that anchor first.
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push_str(newline);
                }
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

/// How long a display diff may search for the shortest edit script. Past this
/// the diff stays correct but may be longer than minimal, so rewriting a large
/// file cannot stall the edit that produced it.
const DIFF_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(250);

/// A compact, readable diff for UI display: a bold one-line summary of how many
/// lines changed, then each changed region shown with a few lines of unchanged
/// context, gutter line numbers, and `±` signs. The context and line numbers let
/// the reader see *where* an edit lands, not just what changed; non-adjacent
/// regions are separated by a dim `⋯`. The model never sees this — it's the
/// `display` half of an edit's [`crate::ToolOutcome`], so the extra context costs no
/// tokens. (`/diff` shows the full working-tree diff.)
pub(crate) fn diff(before: &str, after: &str) -> String {
    diff_within(before, after, usize::MAX)
}

/// [`diff`] with the body cut off once it reaches `max_body_chars`. The
/// summary line always counts every changed line, however large the file or
/// the change: the UI reads its `+N -M` from that line alone.
pub(crate) fn diff_within(before: &str, after: &str, max_body_chars: usize) -> String {
    use similar::{ChangeTag, DiffOp, TextDiff};
    /// Lines of unchanged context shown on each side of a changed region.
    const CONTEXT: usize = 2;

    let tdiff = TextDiff::configure()
        .timeout(DIFF_TIMEOUT)
        .diff_lines(before, after);
    let (mut adds, mut dels) = (0usize, 0usize);
    for op in tdiff.ops() {
        match *op {
            DiffOp::Insert { new_len, .. } => adds += new_len,
            DiffOp::Delete { old_len, .. } => dels += old_len,
            DiffOp::Replace {
                old_len, new_len, ..
            } => {
                dels += old_len;
                adds += new_len;
            }
            DiffOp::Equal { .. } => {}
        }
    }
    if adds == 0 && dels == 0 {
        return "(no changes)".to_string();
    }

    let mut body = String::new();
    'hunks: for (hunk_idx, group) in tdiff.grouped_ops(CONTEXT).iter().enumerate() {
        if hunk_idx > 0 {
            body.push_str("\x1b[2m   ⋯\x1b[0m\n"); // gap between changed regions
        }
        for op in group {
            for change in tdiff.iter_changes(op) {
                let room = max_body_chars.saturating_sub(body.len());
                if room == 0 {
                    body.push_str("\x1b[2m   … (diff truncated)\x1b[0m\n");
                    break 'hunks;
                }
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
                let text = clip_to_char_boundary(text.strip_suffix('\n').unwrap_or(text), room);
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

/// The longest prefix of `text` within `max` bytes that ends on a character
/// boundary.
fn clip_to_char_boundary(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
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
    if old == new {
        bail!(
            "old_string and new_string are identical; no edit was made. \
             Provide a different replacement, or omit this edit if the file is already correct."
        );
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

    // 4. Normalize Unicode confusables (typographic quotes, dashes, and exotic
    //    spaces a model routinely emits from copy-pasting code out of a doc or
    //    chat), comparing trailing-trimmed. The mapping is 1:1 at the character
    //    level — never expanding one glyph into several — so line structure and
    //    the file's real byte offsets stay valid for the splice; the located
    //    span (with its original confusable bytes) is replaced wholesale by
    //    `new`, so the edit both lands and cleans up the stray glyphs.
    if let Some((start, end, _)) = find_unique_window_owned(&lines, text.len(), &old_lines, |l| {
        normalize_confusables(l.trim_end())
    }) {
        let replacement = preserve_line_endings(new, &text[start..end]);
        return Ok(splice(text, start, end, replacement));
    }

    bail!("{}", edit_miss_message(text, old, new));
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
///
/// The anchor is the most distinctive line in `old_string`, not the first one.
/// A multi-line edit that opens with `assert!(` or `{` used to cite the first
/// three of those in the file and miss the real site hundreds of lines later.
pub(crate) fn edit_not_found_help(text: &str, old: &str) -> String {
    let mut msg = String::from(
        "old_string not found, even allowing for whitespace differences. \
         (Do not include the line-number gutter from `read` in old_string.) ",
    );
    let lines: Vec<&str> = text.lines().collect();
    let needles: Vec<&str> = old
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if needles.is_empty() || lines.is_empty() {
        msg.push_str("Re-read the file and copy the exact text to replace.");
        return msg;
    }
    let strong: Vec<&str> = needles
        .iter()
        .copied()
        .filter(|line| !weak_edit_needle(line))
        .collect();
    let search: &[&str] = if strong.is_empty() { &needles } else { &strong };
    let mut hits = rank_edit_anchors(&lines, search);
    if hits.is_empty() {
        hits = similarity_anchors(&lines, longest_needle(search));
    }
    if hits.is_empty() {
        msg.push_str(&format!(
            "No line resembling `{}` is in the {}-line file; re-read it to get the current text.",
            clip(longest_needle(&needles), 60),
            lines.len()
        ));
        return msg;
    }
    msg.push_str("The closest matching lines in the file are:\n");
    for i in hits {
        let lo = i.saturating_sub(2);
        let hi = (i + 6).min(lines.len());
        for (off, line) in lines[lo..hi].iter().enumerate() {
            msg.push_str(&format!("{:>6}\t{}\n", lo + off + 1, line));
        }
        msg.push_str("  ---\n");
    }
    msg.push_str("Copy old_string verbatim from one of these regions.");
    msg
}

fn edit_miss_message(text: &str, old: &str, new: &str) -> String {
    let mut msg = edit_not_found_help(text, old);
    if crate::mutation_repeat::new_string_already_present(text, new) {
        msg.push_str(
            "\nYour new_string is already in this file. Do not retry this edit. \
             If this was a plan step, call update_plan: mark the active step \
             done and set the next unfinished step active.",
        );
    }
    msg
}

/// Lines that occur too often to identify an edit site on their own.
fn weak_edit_needle(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.chars().count() < 8 {
        return true;
    }
    if trimmed.chars().all(|c| {
        matches!(
            c,
            '{' | '}' | '(' | ')' | '[' | ']' | ',' | ';' | '.' | '+' | '-' | '|' | '&'
        )
    }) {
        return true;
    }
    matches!(
        trimmed,
        "assert!("
            | "debug_assert!("
            | "assert_eq!("
            | "assert_ne!("
            | "matches!("
            | "if let"
            | "else {"
            | "Ok(())"
            | "todo!()"
            | "unimplemented!()"
    )
}

fn longest_needle<'a>(needles: &[&'a str]) -> &'a str {
    needles
        .iter()
        .copied()
        .max_by_key(|needle| needle.len())
        .unwrap_or("")
}

fn line_contains_needle(line: &str, needle: &str) -> bool {
    line.trim() == needle || line.contains(needle)
}

/// Rank file regions by how many distinctive `old_string` lines they contain.
/// A line that occurs once outranks a line that occurs on every assertion.
fn rank_edit_anchors(lines: &[&str], needles: &[&str]) -> Vec<usize> {
    let weights: Vec<usize> = needles
        .iter()
        .map(|needle| {
            let hits = lines
                .iter()
                .filter(|line| line_contains_needle(line, needle))
                .count();
            match hits {
                0 => 0,
                1 => 8,
                2..=4 => 4,
                _ => 1,
            }
        })
        .collect();
    if weights.iter().all(|weight| *weight == 0) {
        return Vec::new();
    }
    let span = needles.len().saturating_add(4).clamp(4, 24);
    let mut scored = Vec::new();
    for start in 0..lines.len() {
        let end = (start + span).min(lines.len());
        let mut score = 0usize;
        let mut anchor = start;
        let mut anchor_weight = 0usize;
        for (needle, weight) in needles.iter().zip(weights.iter()) {
            if *weight == 0 {
                continue;
            }
            if let Some(pos) = lines[start..end]
                .iter()
                .position(|line| line_contains_needle(line, needle))
            {
                score += weight;
                if *weight > anchor_weight {
                    anchor_weight = *weight;
                    anchor = start + pos;
                }
            }
        }
        if score > 0 {
            scored.push((anchor, score));
        }
    }
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut picked = Vec::new();
    for (idx, _) in scored {
        if picked.iter().any(|prev| idx.abs_diff(*prev) < span) {
            continue;
        }
        picked.push(idx);
        if picked.len() == 3 {
            break;
        }
    }
    picked
}

/// Word-overlap fallback for a mistyped line that is not a substring of the file.
fn similarity_anchors(lines: &[&str], needle: &str) -> Vec<usize> {
    let tokens: Vec<String> = needle
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .filter(|word| word.len() >= 2)
        .collect();
    if tokens.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, usize)> = lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let lower = line.to_lowercase();
            let score = tokens
                .iter()
                .filter(|token| lower.contains(token.as_str()))
                .count();
            (index, score)
        })
        .filter(|(_, score)| *score > 0)
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut picked = Vec::new();
    for (idx, _) in scored {
        if picked.iter().any(|prev| idx.abs_diff(*prev) < 3) {
            continue;
        }
        picked.push(idx);
        if picked.len() == 3 {
            break;
        }
    }
    picked
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

/// Like [`find_unique_window`] but the normalizer returns an owned `String`
/// (for transforms that rewrite characters, e.g. confusable normalization).
pub(crate) fn find_unique_window_owned(
    lines: &[(usize, &str)],
    text_len: usize,
    old_lines: &[&str],
    norm: impl Fn(&str) -> String,
) -> Option<(usize, usize, usize)> {
    let n = old_lines.len();
    if n == 0 || lines.len() < n {
        return None;
    }
    let normalized_old: Vec<String> = old_lines.iter().map(|l| norm(l)).collect();
    let mut found = None;
    for i in 0..=lines.len() - n {
        if (0..n).all(|j| norm(lines[i + j].1) == normalized_old[j]) {
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

/// Map the Unicode confusables models most often emit for code — typographic
/// quotes, dashes, and non-breaking / exotic spaces — to their ASCII
/// equivalents, **1:1 at the character level**. It never expands one glyph into
/// several (an em dash becomes a single `-`, not `--`), so a normalized
/// comparison preserves line and character structure and a match can splice the
/// file's real byte span. The rarer expand-one-into-two case (a model typing
/// `--` for a file's `—`) is deliberately left to fail into the diagnostic help.
pub(crate) fn normalize_confusables(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            // Single quotes / primes: ' ' ‚ ‛ ′
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' | '\u{2032}' => '\'',
            // Double quotes / primes: " " „ ‟ ″
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '\u{2033}' => '"',
            // Hyphens, dashes, and minus: ‐ ‑ ‒ – — ― −
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            // Non-breaking and other fixed-width spaces → a plain space.
            '\u{00A0}' | '\u{2002}' | '\u{2003}' | '\u{2007}' | '\u{2009}' | '\u{200A}'
            | '\u{202F}' | '\u{205F}' => ' ',
            other => other,
        })
        .collect()
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
#[path = "edit_tests.rs"]
mod tests;
