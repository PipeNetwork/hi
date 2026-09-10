//! Grok-build thinking rows: live truncated tail, finished header-only, Ctrl-E full body.

use std::time::Duration;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::render::{markdown_body_lines, with_gutter, wrap_line_to_width};
use crate::theme::theme;

/// Wrap width for thinking body lines (grok-build thinking wrap).
pub(crate) const THINKING_WRAP_COLS: usize = 120;
/// Live truncated tail, matching grok-build `truncated_lines = 3`.
const THINKING_TRUNCATED_LINES: usize = 3;
const BODY_GUTTER_COLS: u16 = 2;

pub(crate) fn thinking_block_lines(
    text: &str,
    elapsed: Duration,
    expanded: bool,
    running: bool,
) -> Vec<Line<'static>> {
    let mut lines = vec![header_line(elapsed, running)];
    if text.trim().is_empty() || (!expanded && !running) {
        return lines;
    }
    let mut wrapped = Vec::new();
    for line in markdown_body_lines(text) {
        wrapped.extend(wrap_line_to_width(
            &line,
            (THINKING_WRAP_COLS as u16).saturating_sub(BODY_GUTTER_COLS),
        ));
    }
    let body = if expanded || wrapped.len() <= THINKING_TRUNCATED_LINES {
        wrapped
    } else {
        let start = wrapped.len().saturating_sub(THINKING_TRUNCATED_LINES);
        let mut tail = vec![ellipsis_line()];
        tail.extend(wrapped[start..].iter().cloned());
        tail
    };
    lines.extend(body.into_iter().map(dim_thinking_body));
    lines
}

fn header_line(elapsed: Duration, running: bool) -> Line<'static> {
    let th = theme();
    let diamond = if running {
        th.accent_thinking
    } else {
        th.gray_dim
    };
    let label = Style::default().fg(th.gray).add_modifier(Modifier::BOLD);
    let detail = Style::default().fg(th.gray_dim);
    let mut spans = vec![Span::styled("◆ ", Style::default().fg(diamond))];
    if running {
        spans.push(Span::styled("Thinking…", label));
    } else if let Some(time) = format_thought_time(elapsed) {
        spans.push(Span::styled("Thought", label));
        spans.push(Span::styled(format!(" for {time}"), detail));
    } else {
        spans.push(Span::styled("Thought", label));
    }
    Line::from(spans)
}

fn ellipsis_line() -> Line<'static> {
    let th = theme();
    Line::from(Span::styled("…", Style::default().fg(th.gray_dim)))
}

fn dim_thinking_body(line: Line<'static>) -> Line<'static> {
    let th = theme();
    let mut line = line;
    for span in &mut line.spans {
        span.style = span.style.fg(th.gray_dim);
    }
    let mut lined = with_gutter(&line, th.accent_thinking);
    if let Some(gutter) = lined.spans.first_mut() {
        gutter.style = gutter.style.add_modifier(Modifier::DIM);
    }
    lined
}

fn format_thought_time(elapsed: Duration) -> Option<String> {
    let secs = elapsed.as_secs_f64();
    if secs < 0.05 {
        None
    } else if secs < 60.0 {
        Some(format!("{secs:.1}s"))
    } else {
        let mins = (secs / 60.0).floor() as u32;
        let remaining = secs - f64::from(mins) * 60.0;
        Some(format!("{mins}m{remaining:.0}s"))
    }
}
