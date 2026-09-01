//! Shared responsive layout and display-width helpers for the full-screen TUI.

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub(crate) fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// The four visual density bands used by the session and secondary dashboards.
/// Breakpoints are deliberately based on terminal columns, not content or
/// user settings, so resizing never changes agent behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UiLayout {
    Wide,
    Standard,
    Narrow,
    Tiny,
}

impl UiLayout {
    pub(crate) const fn from_width(width: u16) -> Self {
        match width {
            100..=u16::MAX => Self::Wide,
            80..=99 => Self::Standard,
            56..=79 => Self::Narrow,
            _ => Self::Tiny,
        }
    }

    pub(crate) const fn show_full_title(self) -> bool {
        matches!(self, Self::Wide | Self::Standard)
    }

    pub(crate) const fn show_secondary_chrome(self) -> bool {
        !matches!(self, Self::Tiny)
    }

    pub(crate) const fn show_dashboard_secondary(self) -> bool {
        matches!(self, Self::Wide | Self::Standard)
    }

    pub(crate) const fn show_dashboard_tertiary(self) -> bool {
        matches!(self, Self::Wide)
    }

    pub(crate) const fn metrics(self) -> UiMetrics {
        match self {
            Self::Wide => UiMetrics {
                panel_padding: 1,
                gutter_width: 2,
                min_transcript_rows: 1,
            },
            Self::Standard => UiMetrics {
                panel_padding: 1,
                gutter_width: 2,
                min_transcript_rows: 1,
            },
            Self::Narrow => UiMetrics {
                panel_padding: 1,
                gutter_width: 2,
                min_transcript_rows: 1,
            },
            Self::Tiny => UiMetrics {
                // Bordered panels still reserve their one-cell inset on tiny
                // terminals; removing it would let cached lines paint over
                // the right border during sticky/selection overlays.
                panel_padding: 1,
                gutter_width: 2,
                min_transcript_rows: 1,
            },
        }
    }
}

/// Shared spacing values. Keeping these together prevents individual views
/// from slowly developing different padding and minimum-height conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UiMetrics {
    pub(crate) panel_padding: u16,
    pub(crate) gutter_width: u16,
    pub(crate) min_transcript_rows: u16,
}

/// Truncate one-line UI text to a terminal display width, preserving Unicode
/// glyph boundaries and reserving one column for the ellipsis when needed.
pub(crate) fn truncate_display(text: &str, max_width: usize) -> String {
    let text = text.replace(['\n', '\r'], " ");
    let text = text.trim();
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 0 {
        return String::new();
    }
    if max_width == 1 {
        return "…".to_string();
    }

    let target = max_width - 1;
    let mut width = 0;
    let mut out = String::new();
    for ch in text.chars() {
        let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + ch_width > target {
            break;
        }
        width += ch_width;
        out.push(ch);
    }
    format!("{out}…")
}

/// Fit an editable one-line value into a bounded display width while keeping
/// the logical cursor visible. The returned cursor column is measured in
/// terminal cells, so wide glyphs and combining marks cannot place the cursor
/// outside the field.
pub(crate) fn cursor_window(text: &str, cursor: usize, max_width: usize) -> (String, usize) {
    if max_width == 0 {
        return (String::new(), 0);
    }
    let chars: Vec<char> = text
        .chars()
        .map(|ch| if matches!(ch, '\n' | '\r') { ' ' } else { ch })
        .collect();
    let cursor = cursor.min(chars.len());
    let mut prefix_widths: Vec<usize> = Vec::with_capacity(chars.len() + 1);
    prefix_widths.push(0usize);
    for ch in &chars {
        prefix_widths.push(
            prefix_widths
                .last()
                .copied()
                .unwrap_or_default()
                .saturating_add(UnicodeWidthChar::width(*ch).unwrap_or(0)),
        );
    }
    let full_width = prefix_widths[chars.len()];
    if full_width <= max_width {
        return (chars.iter().collect(), prefix_widths[cursor]);
    }
    if max_width == 1 {
        return ("…".to_string(), 0);
    }

    let mut start = 0;
    let mut end = chars.len();
    let width_of = |from: usize, to: usize| prefix_widths[to].saturating_sub(prefix_widths[from]);
    while width_of(start, end) + usize::from(start > 0) + usize::from(end < chars.len()) > max_width
    {
        if cursor <= start {
            end = end.saturating_sub(1);
        } else if cursor >= end {
            start = (start + 1).min(end);
        } else if width_of(start, cursor) > width_of(cursor, end) {
            start += 1;
        } else {
            end = end.saturating_sub(1);
        }
        if start == end {
            break;
        }
    }

    let mut out = String::new();
    let leading = start > 0;
    let trailing = end < chars.len();
    if leading {
        out.push('…');
    }
    out.extend(chars[start..end].iter());
    if trailing {
        out.push('…');
    }
    let cursor_col = usize::from(leading) + width_of(start, cursor.min(end));
    (out, cursor_col.min(max_width))
}

#[cfg(test)]
mod tests {
    use super::{UiLayout, cursor_window, truncate_display};

    #[test]
    fn layout_breakpoints_are_stable() {
        assert_eq!(UiLayout::from_width(120), UiLayout::Wide);
        assert_eq!(UiLayout::from_width(100), UiLayout::Wide);
        assert_eq!(UiLayout::from_width(80), UiLayout::Standard);
        assert_eq!(UiLayout::from_width(56), UiLayout::Narrow);
        assert_eq!(UiLayout::from_width(55), UiLayout::Tiny);
    }

    #[test]
    fn narrow_layout_hides_secondary_surfaces() {
        assert!(!UiLayout::Tiny.show_secondary_chrome());
    }

    #[test]
    fn truncation_uses_display_width_and_unicode_boundaries() {
        assert_eq!(truncate_display("short", 10), "short");
        assert_eq!(truncate_display("abcdefghij", 5), "abcd…");
        assert_eq!(truncate_display("界界界", 5), "界界…");
        assert_eq!(truncate_display("  padded\ntext  ", 20), "padded text");
        assert_eq!(truncate_display("text", 1), "…");
        assert_eq!(truncate_display("text", 0), "");
    }

    #[test]
    fn cursor_window_keeps_long_and_wide_input_inside_the_field() {
        let (text, cursor) = cursor_window("dispatch a focused review", 24, 20);
        assert!(unicode_width::UnicodeWidthStr::width(text.as_str()) <= 20);
        assert!(cursor <= 20);

        let (text, cursor) = cursor_window("界🙂e\u{301} prompt", 4, 8);
        assert!(unicode_width::UnicodeWidthStr::width(text.as_str()) <= 8);
        assert!(cursor <= 8);
    }
}
