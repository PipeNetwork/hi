//! Turn timeline rail: one tick per user prompt in the transcript gutter.

use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::theme::theme;
use crate::view_cache::TranscriptViewCache;

pub(crate) const RAIL_WIDTH: u16 = 2;
pub(crate) const MIN_TERMINAL_WIDTH: u16 = 60;
pub(crate) const MIN_TURNS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimelineHit {
    Tick(usize),
    Up,
    Down,
}

#[derive(Clone, Debug)]
pub(crate) struct TimelineRail {
    pub rect: Rect,
    pub hits: Vec<(u16, TimelineHit)>,
    pub lines: Vec<Line<'static>>,
}

pub(crate) fn visible(enabled: bool, area: Rect, turn_count: usize) -> bool {
    enabled && area.width >= MIN_TERMINAL_WIDTH && turn_count >= MIN_TURNS && area.height >= 3
}

pub(crate) fn compute(
    area: Rect,
    cache: &TranscriptViewCache,
    scroll: u16,
    height: u16,
) -> TimelineRail {
    let th = theme();
    let mut lines = vec![Line::raw("  "); height as usize];
    let mut hits = Vec::new();
    let starts = &cache.prompt_line_starts;
    if starts.is_empty() || cache.prefix.len() < 2 {
        return TimelineRail {
            rect: area,
            hits,
            lines,
        };
    }
    let view_top = scroll as u32;
    let view_bot = view_top.saturating_add(height as u32);
    let active = starts
        .iter()
        .enumerate()
        .rev()
        .find(|&(_, &idx)| cache.prefix.get(idx).copied().unwrap_or(0) <= view_top)
        .map(|(i, _)| i);
    let tick_style = Style::default().fg(th.gray);
    let active_style = Style::default()
        .fg(th.accent_running)
        .add_modifier(Modifier::BOLD);
    for (i, &idx) in starts.iter().enumerate() {
        let row = cache.prefix.get(idx).copied().unwrap_or(0);
        if row < view_top || row >= view_bot {
            continue;
        }
        let y_off = (row - view_top) as u16;
        if y_off >= height {
            continue;
        }
        let y = area.y + y_off;
        let style = if Some(i) == active {
            active_style
        } else {
            tick_style
        };
        let glyph = if Some(i) == active { "● " } else { "· " };
        if let Some(line) = lines.get_mut(y_off as usize) {
            *line = Line::from(Span::styled(glyph, style));
        }
        hits.push((y, TimelineHit::Tick(i)));
    }
    let above = starts
        .iter()
        .any(|&idx| cache.prefix.get(idx).copied().unwrap_or(0) < view_top);
    let below = starts
        .iter()
        .any(|&idx| cache.prefix.get(idx).copied().unwrap_or(0) >= view_bot);
    if above && height > 0 {
        lines[0] = Line::from(Span::styled("▲ ", Style::default().fg(th.accent_running)));
        hits.retain(|(y, _)| *y != area.y);
        hits.push((area.y, TimelineHit::Up));
    }
    if below && height > 1 {
        let last = height as usize - 1;
        lines[last] = Line::from(Span::styled("▼ ", Style::default().fg(th.accent_running)));
        let y = area.y + height - 1;
        hits.retain(|(row, _)| *row != y);
        hits.push((y, TimelineHit::Down));
    }
    TimelineRail {
        rect: area,
        hits,
        lines,
    }
}

pub(crate) fn render(frame: &mut ratatui::Frame, rail: &TimelineRail) {
    let th = theme();
    let mut para = Paragraph::new(rail.lines.clone());
    if th.paints_backgrounds() {
        para = para.style(Style::default().bg(th.bg_base));
    }
    frame.render_widget(para, rail.rect);
}

pub(crate) fn hit_at(hits: &[(u16, TimelineHit)], row: u16) -> Option<TimelineHit> {
    hits.iter().find(|(y, _)| *y == row).map(|(_, h)| *h)
}
