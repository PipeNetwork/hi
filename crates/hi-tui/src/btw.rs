//! `/btw` side-question overlay, matching grok-build's inline panel.
//!
//! Renders as a compact rounded box in the vertical stack (scrollback → overlay
//! → prompt), not a right-hand column. Esc dismisses it; a finished answer is
//! persisted to the transcript as a collapsed `/btw <question>` block.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph, Wrap};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::layout::display_width;
use crate::render::{markdown_body_lines, wrapped_line_height};
use crate::theme::Theme;
use crate::{BtwEntry, SPINNER};

/// Overlay shown above the prompt while a `/btw` question is in flight or
/// waiting to be dismissed.
#[derive(Debug, Clone)]
pub(crate) enum BtwOverlayState {
    Loading {
        question: String,
    },
    Done {
        question: String,
        answer: String,
        scroll_offset: usize,
    },
    /// Request failed — kept for grok-build parity; hi does not emit this yet.
    #[allow(dead_code)]
    Error {
        question: String,
        error: String,
    },
}

impl BtwOverlayState {
    pub(crate) fn question(&self) -> &str {
        match self {
            Self::Loading { question }
            | Self::Done { question, .. }
            | Self::Error { question, .. } => question,
        }
    }

    pub(crate) fn max_scroll_offset(&self, content_width: u16, max_body_lines: usize) -> usize {
        match self {
            Self::Done { answer, .. } => {
                let total = wrapped_body_rows(&markdown_body(answer), content_width);
                total.saturating_sub(max_body_lines.max(1))
            }
            _ => 0,
        }
    }
}

/// Maximum body lines shown for a Done / Error response (grok-build).
pub(crate) const DONE_MAX_BODY_LINES: u16 = 12;

/// Desired panel height. Loading is always 3 (borders + one body row).
/// Done / Error is 2 + min(wrapped lines, [`DONE_MAX_BODY_LINES`]).
pub(crate) fn btw_panel_height(state: Option<&BtwOverlayState>, panel_width: u16) -> u16 {
    let cw = panel_width.saturating_sub(4);
    match state {
        None => 0,
        Some(BtwOverlayState::Loading { .. }) => 3,
        Some(BtwOverlayState::Error { error, .. }) => {
            2 + wrapped_plain_lines(error, cw as usize, DONE_MAX_BODY_LINES as usize).len() as u16
        }
        Some(BtwOverlayState::Done { answer, .. }) => {
            let total = wrapped_body_rows(&markdown_body(answer), cw).max(1);
            2 + (total as u16).min(DONE_MAX_BODY_LINES)
        }
    }
}

/// Derive the grok-style overlay from the live `/btw` thread.
pub(crate) fn overlay_from_thread(
    show: bool,
    thread: &[BtwEntry],
    scroll_offset: usize,
) -> Option<BtwOverlayState> {
    if !show {
        return None;
    }
    let question = thread
        .iter()
        .rev()
        .find_map(|entry| match entry {
            BtwEntry::Question(q) => Some(q.clone()),
            _ => None,
        })
        .unwrap_or_default();
    match thread.last() {
        Some(BtwEntry::Answer(answer)) => Some(BtwOverlayState::Done {
            question,
            answer: answer.clone(),
            scroll_offset,
        }),
        Some(_) => Some(BtwOverlayState::Loading {
            question: if question.is_empty() {
                "…".into()
            } else {
                question
            },
        }),
        None => None,
    }
}

/// Paint the overlay into `area`. Returns the `[Esc]` hit rect for mouse dismiss.
pub(crate) fn render_btw_panel(
    frame: &mut ratatui::Frame,
    state: &BtwOverlayState,
    area: Rect,
    tick: usize,
    theme: &Theme,
) -> Rect {
    if area.width < 12 || area.height < 3 {
        return Rect::default();
    }
    let bg = if theme.paints_backgrounds() {
        theme.bg_base
    } else {
        ratatui::style::Color::Reset
    };
    let content_width = area.width.saturating_sub(4);
    if content_width == 0 {
        return Rect::default();
    }
    let max_body = area.height.saturating_sub(2) as usize;
    let focus_active = state.max_scroll_offset(content_width, max_body) > 0;
    let border = if focus_active {
        theme.accent_user
    } else {
        theme.gray_dim
    };

    frame.render_widget(Clear, area);
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border).bg(bg));
    if theme.paints_backgrounds() {
        block = block.style(Style::default().bg(bg));
    }
    frame.render_widget(block, area);

    let hint = match state {
        BtwOverlayState::Loading { .. } | BtwOverlayState::Error { .. } => "[Esc]".to_string(),
        BtwOverlayState::Done {
            answer,
            scroll_offset,
            ..
        } => {
            let total = wrapped_body_rows(&markdown_body(answer), content_width);
            if total > max_body {
                let offset = (*scroll_offset).min(total.saturating_sub(max_body));
                let pos = offset + 1;
                let end = (offset + max_body).min(total);
                format!("{pos}-{end}/{total}  [Esc]")
            } else {
                "[Esc]".to_string()
            }
        }
    };
    let hint_text = format!(" {hint} ");
    let mut hint_w = display_width(&hint_text) as u16;
    let title_x = area.x.saturating_add(2);
    let mut hint_x = (area.x + area.width).saturating_sub(1 + hint_w);
    if hint_x < title_x {
        hint_w = display_width(" [Esc] ") as u16;
        hint_x = (area.x + area.width).saturating_sub(1 + hint_w);
    }
    let hint_paint = if hint_x < title_x {
        " [Esc] ".to_string()
    } else {
        hint_text
    };
    if hint_x < title_x {
        hint_w = display_width(&hint_paint) as u16;
        hint_x = (area.x + area.width).saturating_sub(1 + hint_w);
    }

    let max_title = hint_x.saturating_sub(title_x).saturating_sub(2) as usize;
    let full_title = format!("/btw {}", state.question());
    let truncated = truncate_title(&full_title, max_title);
    let title_text = format!(" {truncated} ");
    let title_style = Style::default()
        .fg(theme.accent_user)
        .bg(bg)
        .add_modifier(Modifier::BOLD);
    let title_w = (display_width(&title_text) as u16).min(hint_x.saturating_sub(title_x));
    if title_w > 0 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(title_text, title_style))),
            Rect {
                x: title_x,
                y: area.y,
                width: title_w,
                height: 1,
            },
        );
    }

    let close = if hint_w > 0 && hint_x >= title_x {
        let hint_style = Style::default().fg(theme.gray).bg(bg);
        let close = Rect {
            x: hint_x,
            y: area.y,
            width: hint_w,
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(hint_paint, hint_style))),
            close,
        );
        close
    } else {
        Rect::default()
    };

    let body = Rect {
        x: area.x.saturating_add(2),
        y: area.y.saturating_add(1),
        width: content_width,
        height: area.height.saturating_sub(2),
    };
    match state {
        BtwOverlayState::Loading { .. } => {
            let spinner = SPINNER[tick % SPINNER.len()];
            let style = Style::default().fg(theme.gray).bg(bg);
            frame.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled(format!("{spinner} "), style),
                    Span::styled("Answering\u{2026}", style),
                ])),
                body,
            );
        }
        BtwOverlayState::Done {
            answer,
            scroll_offset,
            ..
        } => {
            let lines = markdown_body(answer);
            let total = wrapped_body_rows(&lines, content_width);
            let offset = (*scroll_offset).min(total.saturating_sub(max_body.max(1))) as u16;
            frame.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .scroll((offset, 0))
                    .style(Style::default().bg(bg).fg(theme.text_primary)),
                body,
            );
        }
        BtwOverlayState::Error { error, .. } => {
            let style = Style::default().fg(theme.accent_error).bg(bg);
            let rows = (body.height as usize).min(DONE_MAX_BODY_LINES as usize);
            let lines: Vec<Line> = wrapped_plain_lines(error, content_width as usize, rows)
                .into_iter()
                .map(|text| Line::from(Span::styled(text, style)))
                .collect();
            frame.render_widget(Paragraph::new(lines), body);
        }
    }
    close
}

pub(crate) fn cell_in(area: Rect, x: u16, y: u16) -> bool {
    area.width > 0
        && area.height > 0
        && x >= area.x
        && y >= area.y
        && x < area.x.saturating_add(area.width)
        && y < area.y.saturating_add(area.height)
}

fn markdown_body(text: &str) -> Vec<Line<'static>> {
    markdown_body_lines(text)
}

fn wrapped_body_rows(lines: &[Line], width: u16) -> usize {
    if width == 0 {
        return 1;
    }
    lines
        .iter()
        .map(|line| wrapped_line_height(line, width).max(1) as usize)
        .sum::<usize>()
        .max(1)
}

fn wrapped_plain_lines(text: &str, content_width: usize, max_lines: usize) -> Vec<String> {
    if content_width == 0 {
        return vec![String::new()];
    }
    let mut lines: Vec<String> = Vec::new();
    for para in text.trim().split('\n') {
        if para.is_empty() {
            lines.push(String::new());
            continue;
        }
        let chars: Vec<char> = para.chars().collect();
        let mut start = 0;
        while start < chars.len() {
            let mut end = start;
            let mut w = 0;
            while end < chars.len() {
                let cw = UnicodeWidthChar::width(chars[end]).unwrap_or(0);
                if end > start && w + cw > content_width {
                    break;
                }
                w += cw;
                end += 1;
            }
            if end == start {
                end += 1;
            }
            lines.push(chars[start..end].iter().collect());
            start = end;
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        if let Some(last) = lines.last_mut() {
            while last.width() >= content_width {
                last.pop();
            }
            last.push('\u{2026}');
        }
    }
    lines
}

fn truncate_title(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if text.width() <= max_width {
        return text.to_string();
    }
    let mut s = String::new();
    let mut w = 0;
    for ch in text.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw + 1 > max_width {
            break;
        }
        s.push(ch);
        w += cw;
    }
    s.push('\u{2026}');
    s
}

/// Split a vertical stack chunk so a 1-row gap sits above the overlay when
/// there is room, matching grok-build's `Length(1)` before the btw panel.
pub(crate) fn gap_before_overlay(btw_h: u16, room_for_gap: bool) -> u16 {
    u16::from(btw_h > 0 && room_for_gap)
}

/// Clamp the desired overlay height so the prompt and a one-row transcript
/// still fit in `available` rows (chrome already subtracted).
pub(crate) fn clamp_overlay_height(desired: u16, available: u16) -> u16 {
    if desired == 0 || available < 3 {
        0
    } else {
        desired.min(available)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn dump(term: &Terminal<TestBackend>) -> String {
        let buf = term.backend().buffer();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn loading_height_is_three() {
        let state = BtwOverlayState::Loading {
            question: "why?".into(),
        };
        assert_eq!(btw_panel_height(Some(&state), 40), 3);
        assert_eq!(btw_panel_height(None, 40), 0);
    }

    #[test]
    fn done_height_caps_at_twelve_body_lines() {
        let answer = (0..40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let state = BtwOverlayState::Done {
            question: "q".into(),
            answer,
            scroll_offset: 0,
        };
        assert_eq!(btw_panel_height(Some(&state), 40), 2 + DONE_MAX_BODY_LINES);
    }

    #[test]
    fn loading_paints_title_hint_and_spinner() {
        let state = BtwOverlayState::Loading {
            question: "why?".into(),
        };
        let mut terminal = Terminal::new(TestBackend::new(48, 5)).unwrap();
        terminal
            .draw(|frame| {
                render_btw_panel(frame, &state, frame.area(), 0, &crate::theme::theme());
            })
            .unwrap();
        let screen = dump(&terminal);
        assert!(screen.contains("/btw why?"), "{screen}");
        assert!(screen.contains("[Esc]"), "{screen}");
        assert!(screen.contains("Answering"), "{screen}");
        assert!(
            screen.lines().any(|l| l.trim_start().starts_with('╭')),
            "rounded top: {screen}"
        );
    }

    #[test]
    fn long_question_keeps_esc_visible() {
        let state = BtwOverlayState::Loading {
            question: "this is an extremely long side question that would hide the close hint"
                .into(),
        };
        let mut terminal = Terminal::new(TestBackend::new(32, 4)).unwrap();
        terminal
            .draw(|frame| {
                render_btw_panel(frame, &state, frame.area(), 0, &crate::theme::theme());
            })
            .unwrap();
        let screen = dump(&terminal);
        assert!(screen.contains("[Esc]"), "{screen}");
        assert!(screen.contains("/btw"), "{screen}");
    }

    #[test]
    fn error_height_wraps_and_caps() {
        let state = BtwOverlayState::Error {
            question: "q".into(),
            error: "boom".into(),
        };
        assert_eq!(btw_panel_height(Some(&state), 40), 3);
    }

    #[test]
    fn overlay_from_thread_maps_thinking_to_loading() {
        let thread = vec![
            BtwEntry::Question("why?".into()),
            BtwEntry::Thinking("answering…".into()),
        ];
        let state = overlay_from_thread(true, &thread, 0).expect("overlay");
        assert!(matches!(state, BtwOverlayState::Loading { .. }));
        assert_eq!(state.question(), "why?");
        assert!(overlay_from_thread(false, &thread, 0).is_none());
    }

    #[test]
    fn overlay_from_thread_maps_answer_to_done() {
        let thread = vec![
            BtwEntry::Question("why?".into()),
            BtwEntry::Answer("because".into()),
        ];
        let state = overlay_from_thread(true, &thread, 3).expect("overlay");
        match state {
            BtwOverlayState::Done {
                question,
                answer,
                scroll_offset,
            } => {
                assert_eq!(question, "why?");
                assert_eq!(answer, "because");
                assert_eq!(scroll_offset, 3);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }
}
