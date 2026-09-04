//! Bounded, accurately paginated rendering for model-facing file reads.

use super::{DEFAULT_READ_LIMIT, read_output_budget};

#[cfg(test)]
#[path = "formatting_tests.rs"]
mod tests;

pub(super) struct RenderedRead {
    pub content: String,
    pub truncated: bool,
}

#[cfg(test)]
pub(super) fn format_read(content: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    format_read_with_budget(content, offset, limit, None)
}

#[cfg(test)]
pub(super) fn format_read_with_budget(
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    budget: Option<usize>,
) -> String {
    render_read_with_budget(content, offset, limit, budget).content
}

/// Keep an explicit clipping flag: a single minified line has no next line to
/// page to, but its abbreviated contents must still be reported as truncated.
pub(super) fn format_read_for_output(
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
) -> RenderedRead {
    render_read_with_budget(content, offset, limit, Some(read_output_budget()))
}

pub(super) fn render_read_with_budget(
    content: &str,
    offset: Option<usize>,
    limit: Option<usize>,
    budget: Option<usize>,
) -> RenderedRead {
    if content.is_empty() {
        return small_message("(empty file)".to_owned(), budget);
    }
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len();
    let start = offset.unwrap_or(1).max(1);
    if start > total {
        return small_message(
            format!("(file has {total} line(s); offset {start} is past the end)"),
            budget,
        );
    }
    let limit = limit.unwrap_or(DEFAULT_READ_LIMIT).max(1);
    let end = start.saturating_add(limit).saturating_sub(1).min(total);
    let width = total.to_string().len().max(4);
    let mut out = String::new();
    let mut used_chars = 0usize;
    let mut rendered_end = start.saturating_sub(1);
    for (i, line) in lines[start - 1..end].iter().enumerate() {
        let n = start + i;
        let prefix = format!("{n:>width$}\t");
        let rendered_chars = prefix.chars().count() + line.chars().count() + 1;
        let footer = page_footer(start, n, total);
        if let Some(budget) = budget
            && used_chars
                .saturating_add(rendered_chars)
                .saturating_add(footer.chars().count())
                > budget
        {
            if out.is_empty() {
                return clipped_first_line(&prefix, line, &footer, budget);
            }
            break;
        }
        out.push_str(&prefix);
        out.push_str(line);
        out.push('\n');
        used_chars += rendered_chars;
        rendered_end = n;
    }
    out.push_str(&page_footer(start, rendered_end, total));
    RenderedRead {
        content: out,
        truncated: rendered_end < total,
    }
}

fn page_footer(start: usize, end: usize, total: usize) -> String {
    if start == 1 && end == total {
        return String::new();
    }
    let mut footer = format!("… showing lines {start}-{end} of {total}");
    if end < total {
        footer.push_str(&format!(" — read more with offset {}", end + 1));
    }
    footer
}

fn clipped_first_line(prefix: &str, line: &str, footer: &str, budget: usize) -> RenderedRead {
    let note =
        "\n… [line truncated]; use a bounded shell command to inspect the rest of this line.\n";
    let overhead = prefix.chars().count() + note.chars().count() + footer.chars().count();
    let content = if overhead <= budget {
        let mut out = prefix.to_owned();
        out.extend(line.chars().take(budget - overhead));
        out.push_str(note);
        out.push_str(footer);
        out
    } else {
        // Tiny shares in a many-file read may not fit a line plus instructions.
        // The typed flag still records clipping even if the marker itself clips.
        format!("[line truncated]\n{footer}")
            .chars()
            .take(budget)
            .collect()
    };
    RenderedRead {
        content,
        truncated: true,
    }
}

fn small_message(message: String, budget: Option<usize>) -> RenderedRead {
    let truncated = budget.is_some_and(|budget| message.chars().count() > budget);
    RenderedRead {
        content: message.chars().take(budget.unwrap_or(usize::MAX)).collect(),
        truncated,
    }
}
