//! Bounded, accurately paginated rendering for model-facing file reads.

use super::{DEFAULT_READ_LIMIT, read_output_budget};

/// Render a file for the `read` tool: each line prefixed with its 1-based number
/// and a tab (so the model can cite and edit precisely), optionally restricted
/// to `[offset, offset+limit)`. When no limit is provided, return a bounded
/// page. A footer notes when lines were omitted so the model knows to page a
/// large file with `offset`/`limit` rather than assume it saw everything.
#[cfg(test)]
pub(super) fn format_read(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    format_read_with_budget(content, offset, limit, None)
}

/// Render a read page without relying on a lossy post-render clip. A model
/// often asks for the default 2,000-line page, while the shared tool-result
/// budget is much smaller than that for ordinary source files. Rendering the
/// whole page and clipping afterward can leave an inaccurate footer (or no
/// usable next offset), which makes models repeatedly read the same file.
/// Select the largest complete line range that fits and report the exact
/// range that was returned.
pub(super) fn format_read_for_output(
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> String {
    format_read_with_budget(content, offset, limit, Some(read_output_budget()))
}

pub(super) fn format_read_with_budget(
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    budget: Option<usize>,
) -> String {
    if content.is_empty() {
        return "(empty file)".to_string();
    }
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = offset.unwrap_or(1).max(1);
    if start > total {
        return format!("(file has {total} line(s); offset {start} is past the end)");
    }
    // Treat zero as the smallest useful page rather than producing the
    // misleading range "lines 1-0" and an empty result.
    let limit = limit.unwrap_or(DEFAULT_READ_LIMIT).max(1);
    let end = start.saturating_add(limit).saturating_sub(1).min(total);
    // Width from the file's total line count (not this page's end) so the gutter
    // is consistent across pages — reading lines 1-240 vs 9900-10000 shouldn't
    // shift the column.
    let width = total.to_string().len().max(4);
    let mut out = String::new();
    let mut rendered_end = start.saturating_sub(1);
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let n = start + i;
        let rendered = format!("{n:>width$}\t{line}\n");
        let footer = if start > 1 || n < total {
            let mut footer = format!("… showing lines {start}-{n} of {total}");
            if n < total {
                footer.push_str(&format!(" — read more with offset {}", n + 1));
            }
            footer
        } else {
            String::new()
        };
        if let Some(budget) = budget
            && !out.is_empty()
            && out
                .chars()
                .count()
                .saturating_add(rendered.chars().count())
                .saturating_add(footer.chars().count())
                > budget
        {
            break;
        }
        out.push_str(&rendered);
        rendered_end = n;
    }
    // A single unusually long line can exceed the budget even when it is the
    // first line. Keep a bounded prefix rather than falling back to the old
    // ambiguous whole-page truncation.
    if rendered_end < start {
        let prefix = format!("{start:>width$}\t");
        let suffix = " … [line truncated]";
        let remaining = budget
            .unwrap_or(usize::MAX)
            .saturating_sub(prefix.chars().count() + suffix.chars().count());
        out.push_str(&prefix);
        out.extend(lines[start - 1].chars().take(remaining));
        out.push_str(suffix);
        rendered_end = start;
    }
    if start > 1 || rendered_end < total {
        out.push_str(&format!(
            "… showing lines {start}-{rendered_end} of {total}"
        ));
        if rendered_end < total {
            out.push_str(&format!(" — read more with offset {}", rendered_end + 1));
        }
    }
    out
}
