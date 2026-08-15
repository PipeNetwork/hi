//! Grok-build-style session chrome: full-screen fill, a flat status bar, and
//! a bottom shortcuts row. The transcript itself is unboxed; only the prompt
//! keeps a quiet rounded frame.

use std::path::Path;

use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use unicode_width::UnicodeWidthStr;

use crate::theme::Theme;

/// Horizontal inset for the session stack, matching grok-build's
/// `outer_hpad_left/right = 2`.
pub(crate) const TRANSCRIPT_HPAD: u16 = 2;

/// Vertical inset (top and bottom), matching grok-build's `outer_vpad = 1`.
pub(crate) const OUTER_VPAD: u16 = 1;

/// Drop outer vertical padding at or below this height (grok-build auto-compact).
pub(crate) const AUTO_COMPACT_MAX_ROWS: u16 = 20;

/// Hide the shortcuts row below this terminal height so the prompt is never
/// starved.
pub(crate) const SHORTCUTS_MIN_HEIGHT: u16 = 10;

/// Hide the top status bar below this height.
pub(crate) const STATUS_MIN_HEIGHT: u16 = 6;

fn canvas_bg(theme: &Theme) -> Color {
    if theme.paints_backgrounds() {
        theme.bg_base
    } else {
        Color::Reset
    }
}

pub(crate) fn fill_background(frame: &mut ratatui::Frame, area: Rect, theme: &Theme) {
    if !theme.paints_backgrounds() {
        return;
    }
    frame.render_widget(
        Block::new().style(Style::default().bg(theme.bg_base).fg(theme.text_primary)),
        area,
    );
}

/// Effective outer padding for `area`: 2 columns on each side, plus a blank
/// row above and below on a tall terminal. Auto-compact (height ≤ 20) drops
/// the vertical pad so a short session never starves the prompt.
pub(crate) fn outer_pad(area: Rect) -> (u16, u16, u16) {
    let compact = area.height > 0 && area.height <= AUTO_COMPACT_MAX_ROWS;
    let vpad = if compact { 0 } else { OUTER_VPAD };
    (TRANSCRIPT_HPAD, vpad, vpad)
}

/// Inset `area` on all four sides so chrome floats on the canvas instead of
/// flushing to the terminal edge.
pub(crate) fn inset(area: Rect, hpad: u16, top: u16, bottom: u16) -> Rect {
    let hpad = hpad.min(area.width / 2);
    let top = top.min(area.height);
    let bottom = bottom.min(area.height.saturating_sub(top));
    Rect {
        x: area.x + hpad,
        y: area.y + top,
        width: area.width.saturating_sub(hpad.saturating_mul(2)),
        height: area.height.saturating_sub(top).saturating_sub(bottom),
    }
}

/// Workspace path for the status-bar left side: `~/…` when under `$HOME`,
/// truncated from the left if it would overflow `max_cols`.
pub(crate) fn display_cwd(root: &Path, max_cols: usize) -> String {
    if max_cols == 0 {
        return String::new();
    }
    let raw = if root.as_os_str().is_empty() {
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".into())
    } else {
        root.display().to_string()
    };
    let home = std::env::var("HOME").ok();
    let short = match &home {
        Some(h) if !h.is_empty() && raw.starts_with(h) => format!("~{}", &raw[h.len()..]),
        _ => raw,
    };
    let width = UnicodeWidthStr::width(short.as_str());
    if width <= max_cols {
        return short;
    }
    let mut budget = max_cols.saturating_sub(1);
    let mut chars: Vec<char> = Vec::new();
    for ch in short.chars().rev() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if w > budget {
            break;
        }
        chars.push(ch);
        budget -= w;
    }
    chars.reverse();
    format!("…{}", chars.into_iter().collect::<String>())
}

pub(crate) fn render_status_bar(
    frame: &mut ratatui::Frame,
    area: Rect,
    left: Line<'static>,
    right: Line<'static>,
    theme: &Theme,
) {
    if area.height == 0 {
        return;
    }
    let style = Style::default().fg(theme.gray).bg(canvas_bg(theme));
    frame.render_widget(Block::new().style(style), area);

    let right_w = right.width() as u16;
    let left_budget = area.width.saturating_sub(right_w.saturating_add(1));
    if left_budget > 0 {
        frame.render_widget(
            Paragraph::new(vec![left]).style(style),
            Rect {
                x: area.x,
                y: area.y,
                width: left_budget,
                height: 1,
            },
        );
    }
    if right_w > 0 && right_w <= area.width {
        frame.render_widget(
            Paragraph::new(vec![right])
                .alignment(Alignment::Right)
                .style(style),
            area,
        );
    }
}

/// One shortcuts-bar item: a key (or chord) plus a short verb.
pub(crate) struct ShortcutHint {
    pub key: &'static str,
    pub label: &'static str,
}

pub(crate) fn render_shortcuts_bar(
    frame: &mut ratatui::Frame,
    area: Rect,
    hints: &[ShortcutHint],
    theme: &Theme,
) {
    if area.height == 0 || hints.is_empty() {
        return;
    }
    let bg = canvas_bg(theme);
    let key_style = Style::default()
        .fg(theme.text_secondary)
        .bg(bg)
        .add_modifier(Modifier::BOLD);
    let label_style = Style::default().fg(theme.gray).bg(bg);
    let sep_style = Style::default()
        .fg(theme.gray)
        .bg(bg)
        .add_modifier(Modifier::DIM);
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut used = 0usize;
    for (i, hint) in hints.iter().enumerate() {
        let extra = if i == 0 { 0 } else { 5 };
        let w = extra + UnicodeWidthStr::width(hint.key) + 1 + UnicodeWidthStr::width(hint.label);
        if used + w > area.width as usize {
            break;
        }
        if i > 0 {
            spans.push(Span::styled("  │  ", sep_style));
        }
        spans.push(Span::styled(hint.key.to_string(), key_style));
        spans.push(Span::styled(format!(":{}", hint.label), label_style));
        used += w;
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(bg)),
        area,
    );
}

pub(crate) fn render_turn_status(
    frame: &mut ratatui::Frame,
    area: Rect,
    line: Line<'static>,
    theme: &Theme,
) {
    if area.height == 0 {
        return;
    }
    frame.render_widget(
        Paragraph::new(vec![line]).style(Style::default().bg(canvas_bg(theme))),
        area,
    );
}

/// Dim ` │ ` separator used between status-bar chips, matching grok-build.
pub(crate) fn chip_sep(theme: &Theme) -> Span<'static> {
    Span::styled(
        " │ ",
        Style::default().fg(theme.gray_dim).bg(canvas_bg(theme)),
    )
}

pub(crate) fn push_chip(spans: &mut Vec<Span<'static>>, theme: &Theme, chip: Span<'static>) {
    if !spans.is_empty() {
        spans.push(chip_sep(theme));
    }
    spans.push(chip);
}

/// The "PipeNetwork.AI" wordmark as figlet-style 5-row block letters — the
/// empty-session splash, matching the CLI landing banner.
pub(crate) const WORDMARK: [&str; 5] = [
    " ___ _           _  _     _                  _       _   ___ ",
    "| _ (_)_ __  ___| \\| |___| |___ __ _____ _ _| |__   /_\\ |_ _|",
    "|  _/ | '_ \\/ -_) .` / -_)  _\\ V  V / _ \\ '_| / /_ / _ \\ | | ",
    "|_| |_| .__/\\___|_|\\_\\___|\\__|\\_/\\_/\\___/_| |_\\_(_)_/ \\_\\___|",
    "      |_|                                                    ",
];

/// Brand orange for the wordmark, matching the CLI landing banner.
const WORDMARK_FG: Color = Color::Rgb(255, 140, 0);

/// Empty-session splash: the figlet wordmark when the canvas is wide enough,
/// otherwise a quiet centered "hi". Display-only — never seeded into the
/// transcript, so the first turn replaces it cleanly.
pub(crate) fn welcome_lines(area: Rect, theme: &Theme) -> Vec<Line<'static>> {
    let banner_w = WORDMARK
        .iter()
        .map(|row| UnicodeWidthStr::width(*row) as u16)
        .max()
        .unwrap_or(0);
    if area.width >= banner_w && area.height >= WORDMARK.len() as u16 + 1 {
        let style = Style::default()
            .fg(WORDMARK_FG)
            .add_modifier(Modifier::BOLD);
        let mut lines = vec![Line::raw("")];
        lines.extend(
            WORDMARK
                .iter()
                .map(|row| Line::from(Span::styled((*row).to_string(), style))),
        );
        lines
    } else {
        vec![
            Line::raw(""),
            Line::from(Span::styled(
                "hi",
                Style::default().fg(theme.gray).add_modifier(Modifier::BOLD),
            )),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn outer_pad_matches_grok_build_thresholds() {
        let tall = Rect {
            x: 0,
            y: 0,
            width: 80,
            height: 24,
        };
        assert_eq!(outer_pad(tall), (2, 1, 1));
        let compact = Rect {
            x: 0,
            y: 0,
            width: 80,
            height: 20,
        };
        assert_eq!(outer_pad(compact), (2, 0, 0));
        let inset_tall = inset(tall, 2, 1, 1);
        assert_eq!(inset_tall.x, 2);
        assert_eq!(inset_tall.y, 1);
        assert_eq!(inset_tall.width, 76);
        assert_eq!(inset_tall.height, 22);
    }

    #[test]
    fn display_cwd_shortens_home_and_truncates() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/test".into());
        let nested = Path::new(&home).join("proj/src");
        let shown = display_cwd(&nested, 80);
        assert!(shown.starts_with("~/"), "home-relative cwd: {shown}");
        let squeezed = display_cwd(Path::new("/very/long/workspace/path/here"), 12);
        assert!(
            squeezed.starts_with('…'),
            "overflow truncates from the left: {squeezed}"
        );
        assert!(squeezed.chars().count() <= 12 || UnicodeWidthStr::width(squeezed.as_str()) <= 12);
    }

    #[test]
    fn welcome_uses_figlet_when_the_canvas_fits() {
        let theme = crate::theme::Theme::groknight();
        let wide = welcome_lines(
            Rect {
                x: 0,
                y: 0,
                width: 80,
                height: 12,
            },
            &theme,
        );
        assert!(
            wide.iter()
                .any(|l| crate::render::line_text(l).contains("___ _")),
            "wide canvas shows the wordmark"
        );
        let narrow = welcome_lines(
            Rect {
                x: 0,
                y: 0,
                width: 24,
                height: 8,
            },
            &theme,
        );
        assert!(
            narrow.iter().any(|l| crate::render::line_text(l) == "hi"),
            "narrow canvas falls back to hi"
        );
        assert!(
            !narrow
                .iter()
                .any(|l| crate::render::line_text(l).contains("___ _")),
            "narrow canvas hides the figlet"
        );
    }
}
