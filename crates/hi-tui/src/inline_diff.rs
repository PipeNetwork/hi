//! Grok-build inline diffs: numbered gutters, no `+/-` signs, `…` hunk gaps.
//!
//! Layout (default grok-build `DiffRenderConfig`): two-space indent, a
//! right-aligned line number, two spaces, then content. Add/delete meaning
//! is the number color plus a background band on truecolor themes — not a
//! leading `+`/`-`. File/`@@` headers are omitted; the Edit row already
//! names the path.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::theme::theme;

const INDENT: &str = "  ";
const CONTENT_GAP: &str = "  ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiffBand {
    Add,
    Del,
    Context,
    Meta,
}

#[derive(Clone, Debug)]
enum DiffItem {
    Row {
        band: DiffBand,
        number: Option<u32>,
        text: String,
    },
    Gap {
        unchanged: Option<usize>,
    },
}

/// Paint a unified or compact-numbered diff in grok-build's inline layout.
pub(crate) fn diff_lines(body: &str) -> Vec<Line<'static>> {
    render_items(&parse_diff(body))
}

/// Line indexes that start a hunk in the painted output (first row, then
/// every row after a `…` separator). Used by Ctrl-G n/p navigation.
pub(crate) fn hunk_start_indices(lines: &[Line<'static>]) -> Vec<usize> {
    let mut starts = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if i == 0 && !is_hunk_gap(line) {
            starts.push(0);
            continue;
        }
        if i > 0 && is_hunk_gap(&lines[i - 1]) && !is_hunk_gap(line) {
            starts.push(i);
        }
    }
    starts
}

pub(crate) fn is_hunk_gap(line: &Line<'static>) -> bool {
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    let t = text.trim_start();
    t.starts_with('…') || t.starts_with('⋯')
}

fn parse_diff(body: &str) -> Vec<DiffItem> {
    let mut items = Vec::new();
    let mut old_ln = 0u32;
    let mut new_ln = 0u32;
    let mut seen_hunk = false;
    let mut last_new: Option<u32> = None;

    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("diff --git ")
            || trimmed.starts_with("index ")
            || trimmed.starts_with("---")
            || trimmed.starts_with("+++")
            || trimmed.starts_with('\\')
            || trimmed.contains(" addition")
            || trimmed.contains(" deletion")
        {
            continue;
        }
        if trimmed.starts_with('…') || trimmed.contains('⋯') {
            items.push(DiffItem::Gap {
                unchanged: ellipsis_count(trimmed),
            });
            continue;
        }
        if trimmed.starts_with("@@") {
            let (old, new) = hunk_starts(trimmed);
            if seen_hunk {
                items.push(DiffItem::Gap {
                    unchanged: gap_unchanged(last_new, new),
                });
            }
            seen_hunk = true;
            old_ln = old.unwrap_or(0);
            new_ln = new.unwrap_or(0);
            continue;
        }
        if let Some((sign, gutter, content)) = compact_parts(line) {
            let number = gutter.trim().parse::<u32>().ok().filter(|n| *n > 0);
            let band = match sign {
                '+' => DiffBand::Add,
                '-' => DiffBand::Del,
                _ => DiffBand::Context,
            };
            if matches!(band, DiffBand::Add | DiffBand::Context) {
                last_new = number.or(last_new);
            }
            items.push(DiffItem::Row {
                band,
                number,
                text: content.to_string(),
            });
            continue;
        }
        if line.starts_with('+') && !line.starts_with("+++") {
            let number = (new_ln > 0).then_some(new_ln);
            items.push(DiffItem::Row {
                band: DiffBand::Add,
                number,
                text: line[1..].to_string(),
            });
            last_new = number.or(last_new);
            if new_ln > 0 {
                new_ln += 1;
            }
            continue;
        }
        if line.starts_with('-') && !line.starts_with("---") {
            items.push(DiffItem::Row {
                band: DiffBand::Del,
                number: (old_ln > 0).then_some(old_ln),
                text: line[1..].to_string(),
            });
            if old_ln > 0 {
                old_ln += 1;
            }
            continue;
        }
        let text = if let Some(rest) = line.strip_prefix(' ') {
            rest.to_string()
        } else {
            line.to_string()
        };
        let number = (new_ln > 0).then_some(new_ln);
        items.push(DiffItem::Row {
            band: DiffBand::Context,
            number,
            text,
        });
        last_new = number.or(last_new);
        if new_ln > 0 {
            new_ln += 1;
        }
        if old_ln > 0 {
            old_ln += 1;
        }
    }
    items
}

fn hunk_starts(header: &str) -> (Option<u32>, Option<u32>) {
    let old = header.split('-').nth(1).and_then(leading_u32);
    let new = header.split('+').nth(1).and_then(leading_u32);
    (old, new)
}

fn leading_u32(s: &str) -> Option<u32> {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

fn gap_unchanged(prev_last_new: Option<u32>, next_first_new: Option<u32>) -> Option<usize> {
    let prev = prev_last_new?;
    let next = next_first_new?;
    next.checked_sub(prev)
        .and_then(|d| d.checked_sub(1))
        .filter(|n| *n > 0)
        .map(|n| n as usize)
}

fn ellipsis_count(line: &str) -> Option<usize> {
    let rest = line
        .trim_start_matches(['…', '⋯', ' '])
        .split_whitespace()
        .next()?;
    rest.parse().ok()
}

fn compact_parts(line: &str) -> Option<(char, &str, &str)> {
    let bytes = line.as_bytes();
    if bytes.len() < 7 || bytes[4] != b' ' || bytes[6] != b' ' {
        return None;
    }
    if !bytes[..4].iter().all(|&b| b.is_ascii_digit() || b == b' ') {
        return None;
    }
    let sign = match bytes[5] {
        b'+' | b'-' | b' ' => bytes[5] as char,
        _ => return None,
    };
    Some((sign, &line[..4], &line[7..]))
}

fn render_items(items: &[DiffItem]) -> Vec<Line<'static>> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut width = 1usize;
    for item in items {
        if let DiffItem::Row {
            number: Some(n), ..
        } = item
        {
            width = width.max((*n as usize).max(1).ilog10() as usize + 1);
        }
    }
    items.iter().map(|item| paint_item(item, width)).collect()
}

fn paint_item(item: &DiffItem, num_width: usize) -> Line<'static> {
    let th = theme();
    match item {
        DiffItem::Gap { unchanged } => {
            let label = match unchanged {
                Some(1) => "… 1 unchanged line".to_string(),
                Some(n) => format!("… {n} unchanged lines"),
                None => "…".to_string(),
            };
            Line::from(vec![
                Span::raw(INDENT),
                Span::styled(label, Style::default().fg(th.gray_dim)),
            ])
        }
        DiffItem::Row { band, number, text } => {
            let (num_fg, content_fg, bg) = match band {
                DiffBand::Add => (
                    th.diff_add,
                    if th.paints_backgrounds() {
                        th.text_primary
                    } else {
                        th.diff_add
                    },
                    Some(th.diff_add_bg).filter(|_| th.paints_backgrounds()),
                ),
                DiffBand::Del => (
                    th.diff_del,
                    if th.paints_backgrounds() {
                        th.text_primary
                    } else {
                        th.diff_del
                    },
                    Some(th.diff_del_bg).filter(|_| th.paints_backgrounds()),
                ),
                DiffBand::Context => (th.diff_gutter, th.diff_context, None),
                DiffBand::Meta => (th.diff_gutter, th.diff_hunk, None),
            };
            let num = match number {
                Some(n) => format!("{n:>num_width$}"),
                None => " ".repeat(num_width),
            };
            let content = if text.is_empty() { " " } else { text.as_str() };
            let mut content_style = Style::default().fg(content_fg);
            if *band == DiffBand::Meta {
                content_style = content_style.add_modifier(Modifier::BOLD);
            }
            if let Some(bg) = bg {
                content_style = content_style.bg(bg);
            }
            Line::from(vec![
                Span::raw(INDENT),
                Span::styled(num, Style::default().fg(num_fg)),
                Span::raw(CONTENT_GAP),
                Span::styled(content.to_string(), content_style),
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::line_text;

    fn texts(body: &str) -> Vec<String> {
        diff_lines(body).iter().map(line_text).collect()
    }

    #[test]
    fn grok_build_basic_hunk_layout() {
        let body = "@@ -10,3 +10,4 @@\n let x = 1;\n-let y = 2;\n+let y = 3;\n let z = 4;\n";
        assert_eq!(
            texts(body),
            vec![
                "  10  let x = 1;".to_string(),
                "  11  let y = 2;".to_string(),
                "  11  let y = 3;".to_string(),
                "  12  let z = 4;".to_string(),
            ]
        );
    }

    #[test]
    fn grok_build_hunk_gap_counts_unchanged_new_lines() {
        let body = "@@ -5,2 +5,1 @@\n first hunk context\n-deleted in first\n@@ -49,2 +49,2 @@\n second hunk context\n+inserted in second\n";
        let t = texts(body);
        assert_eq!(t[0], "   5  first hunk context");
        assert!(
            t.iter().any(|l| l.contains("… 43 unchanged lines")),
            "{t:?}"
        );
        assert!(
            t.iter()
                .any(|l| l.contains("49") && l.contains("second hunk context")),
            "{t:?}"
        );
    }

    #[test]
    fn three_digit_gutter_pads_two_digit_numbers() {
        let body = "@@ -99,3 +99,3 @@\n context before\n-old code\n+new code\n context after\n";
        let t = texts(body);
        assert_eq!(t[0], "   99  context before");
        assert_eq!(t[1], "  100  old code");
        assert_eq!(t[2], "  100  new code");
        assert_eq!(t[3], "  101  context after");
    }

    #[test]
    fn strips_file_and_hunk_headers() {
        let body = "--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-old\n+new\n";
        let t = texts(body).join("\n");
        assert!(!t.contains("---"), "{t}");
        assert!(!t.contains("+++"), "{t}");
        assert!(!t.contains("@@"), "{t}");
        assert!(!t.contains("+new"), "{t}");
        assert!(!t.contains("-old"), "{t}");
        assert!(t.contains("old"), "{t}");
        assert!(t.contains("new"), "{t}");
    }

    #[test]
    fn compact_numbered_preview_drops_signs() {
        let body = "--- src/cli.rs\n+++ src/cli.rs\n   1 - old line\n   2 + new line\n";
        let t = texts(body);
        assert_eq!(
            t,
            vec!["  1  old line".to_string(), "  2  new line".to_string()]
        );
    }

    #[test]
    fn hunk_starts_follow_gap_separators() {
        let body = "@@ -1,1 +1,1 @@\n-a\n+b\n@@ -5,1 +5,1 @@\n-c\n+d\n@@ -10,1 +10,1 @@\n-e\n+f\n";
        let lines = diff_lines(body);
        assert_eq!(hunk_start_indices(&lines), vec![0, 3, 6]);
    }
}
