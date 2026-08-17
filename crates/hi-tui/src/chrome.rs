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
/// starved. Idle sessions with a transcript also hide it (`?` still opens help).
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

/// Idle `{pct}% ctx`; hover becomes a same-width bar so the status row does
/// not shift.
pub(crate) fn span_hit(area: Rect, line: &Line<'_>, pred: impl Fn(&str) -> bool) -> Rect {
    let right_w = line.width() as u16;
    let x0 = area.x + area.width.saturating_sub(right_w);
    let mut x = 0u16;
    for span in &line.spans {
        let w = span.content.width() as u16;
        if pred(span.content.as_ref()) {
            return Rect {
                x: x0 + x,
                y: area.y,
                width: w.max(1),
                height: 1,
            };
        }
        x = x.saturating_add(w);
    }
    Rect::default()
}

pub(crate) fn context_chip(pct: u64, hovered: bool) -> String {
    let idle = format!("{pct}% ctx");
    if !hovered {
        return idle;
    }
    usage_bar(idle.chars().count().max(1), pct)
}

/// Grok-build header occupancy: `64k / 128k`. Hover is a same-width bar.
pub(crate) fn context_usage_chip(used: u64, window: u64, hovered: bool) -> String {
    let idle = format!(
        "{} / {}",
        crate::util::fmt_count(used),
        crate::util::fmt_count(window)
    );
    if !hovered {
        return idle;
    }
    let pct = if window == 0 {
        0
    } else {
        (used.saturating_mul(100) / window).min(100)
    };
    usage_bar(idle.chars().count().max(1), pct)
}

fn usage_bar(width: usize, pct: u64) -> String {
    let filled = ((pct as usize) * width / 100).min(width);
    let mut out = String::new();
    for i in 0..width {
        out.push(if i < filled { '#' } else { '-' });
    }
    out
}

/// Right-align `label` on `line` when it fits, grok-build timestamp style.
pub(crate) fn overlay_right(line: &mut Line<'static>, label: &str, width: u16, style: Style) {
    use crate::layout::display_width;
    let used: usize = line
        .spans
        .iter()
        .map(|s| display_width(s.content.as_ref()))
        .sum();
    let lw = display_width(label);
    if used + 1 + lw > width as usize {
        return;
    }
    line.spans.push(Span::raw(
        " ".repeat((width as usize).saturating_sub(used + lw)),
    ));
    line.spans.push(Span::styled(label.to_string(), style));
}

/// Status-bar label for the OS write sandbox (`HI_SANDBOX`).
pub(crate) fn sandbox_chip() -> &'static str {
    match hi_tools::sandbox::SandboxPolicy::from_env() {
        Ok(hi_tools::sandbox::SandboxPolicy::Off) => "sandbox off",
        Ok(_) if cfg!(any(target_os = "macos", target_os = "linux")) => "sandbox",
        Ok(_) => "no sandbox",
        Err(_) => "sandbox?",
    }
}

/// The "hi" wordmark as figlet-style 5-row block letters — the empty-session
/// splash, matching the CLI landing banner.
pub(crate) const WORDMARK: [&str; 5] = [
    " _     _ ",
    "| |__ (_)",
    "| '_ \\| |",
    "| | | | |",
    "|_| |_|_|",
];

/// Brand orange for the wordmark, matching the CLI landing banner.
const WORDMARK_FG: Color = Color::Rgb(255, 140, 0);

/// Empty-session home: repo:branch, recent sessions, plan-mode hint.
pub(crate) struct WelcomeHome<'a> {
    pub location: String,
    pub sessions: &'a [crate::LocalSessionInfo],
}

const MAX_WELCOME_SESSIONS: usize = 5;

pub(crate) fn welcome_location(root: &Path, branch: Option<&str>) -> String {
    let repo = root
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("hi");
    match branch.filter(|b| !b.is_empty()) {
        Some(branch) => format!("{repo}:{branch}"),
        None => repo.to_string(),
    }
}

fn welcome_session_label(session: &crate::LocalSessionInfo) -> &str {
    let title = session.title.trim();
    if title.is_empty() || title.starts_with("[hi:context") {
        session.id.as_str()
    } else {
        title
    }
}

/// Empty-session splash: the figlet wordmark when the canvas is wide enough,
/// otherwise a quiet centered "hi". Display-only — never seeded into the
/// transcript, so the first turn replaces it cleanly.
pub(crate) fn welcome_lines(
    area: Rect,
    theme: &Theme,
    home: Option<&WelcomeHome<'_>>,
) -> Vec<Line<'static>> {
    let banner_w = WORDMARK
        .iter()
        .map(|row| UnicodeWidthStr::width(*row) as u16)
        .max()
        .unwrap_or(0);
    let mut lines = Vec::new();
    if let Some(home) = home
        && area.height >= 3
    {
        lines.push(Line::styled(
            home.location.clone(),
            Style::default().fg(theme.gray),
        ));
    }
    if area.width >= banner_w && area.height >= WORDMARK.len() as u16 + 2 {
        let style = Style::default()
            .fg(WORDMARK_FG)
            .add_modifier(Modifier::BOLD);
        lines.push(Line::raw(""));
        lines.extend(
            WORDMARK
                .iter()
                .map(|row| Line::from(Span::styled((*row).to_string(), style))),
        );
        if let Some(home) = home {
            let remaining = area
                .height
                .saturating_sub(lines.len() as u16)
                .saturating_sub(2);
            if remaining > 2 && !home.sessions.is_empty() {
                lines.push(Line::raw(""));
                lines.push(Line::styled(
                    "recent sessions",
                    Style::default().fg(theme.gray),
                ));
                let title_budget = (area.width as usize).saturating_sub(12).clamp(16, 56);
                for session in home
                    .sessions
                    .iter()
                    .take(MAX_WELCOME_SESSIONS.min(remaining.saturating_sub(2) as usize))
                {
                    let title = hi_agent::ui::clip(welcome_session_label(session), title_budget);
                    lines.push(Line::from(vec![
                        Span::styled(title, Style::default().fg(theme.text_secondary)),
                        Span::styled(
                            format!("  {}", session.age),
                            Style::default().fg(theme.gray_dim),
                        ),
                    ]));
                }
            }
            if area.height as usize > lines.len() + 1 {
                lines.push(Line::raw(""));
                lines.push(Line::styled(
                    "Shift-Tab plan mode · /sessions resume · /memory notes",
                    Style::default().fg(theme.gray_dim),
                ));
            }
        }
        lines
    } else {
        lines.push(Line::from(Span::styled(
            "hi",
            Style::default().fg(theme.gray).add_modifier(Modifier::BOLD),
        )));
        lines
    }
}

pub(crate) fn git_branch(root: &Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!name.is_empty() && name != "HEAD").then_some(name)
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
            None,
        );
        assert!(
            wide.iter()
                .any(|l| crate::render::line_text(l).contains("|_| |_|_|")),
            "wide canvas shows the hi wordmark"
        );
        let tiny = welcome_lines(
            Rect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            &theme,
            None,
        );
        assert!(
            tiny.iter().any(|l| crate::render::line_text(l) == "hi"),
            "tiny canvas falls back to hi"
        );
        assert!(
            !tiny
                .iter()
                .any(|l| crate::render::line_text(l).contains("|_| |_|_|")),
            "tiny canvas hides the figlet"
        );
    }

    #[test]
    fn context_chip_hover_keeps_the_same_width() {
        let idle = context_chip(50, false);
        let hover = context_chip(50, true);
        assert_eq!(idle, "50% ctx");
        assert_eq!(idle.chars().count(), hover.chars().count());
        assert_eq!(hover, "###----");
        assert_eq!(
            context_chip(0, true).chars().count(),
            context_chip(0, false).chars().count()
        );
        assert_eq!(
            context_chip(100, true).chars().count(),
            context_chip(100, false).chars().count()
        );
        let usage = context_usage_chip(64_000, 128_000, false);
        let usage_hover = context_usage_chip(64_000, 128_000, true);
        assert_eq!(usage, "64k / 128k");
        assert_eq!(usage.chars().count(), usage_hover.chars().count());
    }

    #[test]
    fn welcome_home_shows_location_sessions_and_plan_hint() {
        let theme = crate::theme::Theme::groknight();
        let sessions = [crate::LocalSessionInfo {
            id: "abc".into(),
            title: "review layout".into(),
            age: "2h".into(),
            lines: 12,
        }];
        let home = WelcomeHome {
            location: "hi:main".into(),
            sessions: &sessions,
        };
        let lines = welcome_lines(
            Rect {
                x: 0,
                y: 0,
                width: 80,
                height: 20,
            },
            &theme,
            Some(&home),
        );
        let text: Vec<String> = lines.iter().map(|l| crate::render::line_text(l)).collect();
        assert!(text.iter().any(|l| l == "hi:main"), "{text:?}");
        assert!(
            text.iter().any(|l| l.contains("|_| |_|_|")),
            "wordmark stays: {text:?}"
        );
        assert!(
            text.iter().any(|l| l.contains("review layout")),
            "recent session: {text:?}"
        );
        assert!(
            text.iter().any(|l| l.contains("Shift-Tab plan")),
            "plan hint: {text:?}"
        );
    }

    #[test]
    fn welcome_location_is_repo_and_branch() {
        assert_eq!(
            welcome_location(Path::new("/Users/david/hi"), Some("main")),
            "hi:main"
        );
        assert_eq!(welcome_location(Path::new("/workspace"), None), "workspace");
    }

    #[test]
    fn welcome_home_does_not_dump_context_blobs() {
        let theme = crate::theme::Theme::groknight();
        let dump = "[hi:context — session state, not instructions] # Memory (from past sessions; task-ranked) Prefer bullets";
        let sessions: Vec<crate::LocalSessionInfo> = (0..12)
            .map(|i| crate::LocalSessionInfo {
                id: format!("sess-{i}"),
                title: dump.into(),
                age: "5h".into(),
                lines: 2,
            })
            .collect();
        let home = WelcomeHome {
            location: "hi:main".into(),
            sessions: &sessions,
        };
        let lines = welcome_lines(
            Rect {
                x: 0,
                y: 0,
                width: 80,
                height: 24,
            },
            &theme,
            Some(&home),
        );
        let text: Vec<String> = lines.iter().map(|l| crate::render::line_text(l)).collect();
        assert!(
            text.iter().all(|l| !l.contains("[hi:context")),
            "context dumps must not be session titles: {text:?}"
        );
        let listed = text.iter().filter(|l| l.contains("sess-")).count();
        assert!(listed <= 5, "cap recent sessions: {text:?}");
        assert!(
            text.iter().any(|l| l.contains("sess-0")),
            "falls back to id: {text:?}"
        );
    }
}
