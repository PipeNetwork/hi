//! Grok-build turn-status strip: activity on the left, timers and `[stop]` on the right.
//!
//! Hidden when idle and the last turn succeeded. Warnings, failures, background
//! subagents, and an in-flight turn each take the one row between scrollback
//! and the prompt.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::layout::display_width;
use crate::render::{dim, flash_weight, lerp_color, pulse_color};
use crate::theme::theme;
use crate::util::{clip_reason, fmt_count, fmt_elapsed, fmt_rate_limits};
use crate::{App, SPINNER, TurnState};

/// Build the status row for `width` columns, or `None` when the row should hide.
pub(crate) fn build(app: &App, width: u16) -> Option<Line<'static>> {
    let th = theme();
    if app.confirmation.is_some() {
        return Some(waiting_on_you(app, width));
    }
    if app.plan_approval.is_some() {
        return Some(waiting_on_plan(app, width));
    }
    if app.working {
        return Some(working_line(app, width));
    }
    let bg_live = live_background_subagents(app);
    if bg_live > 0 {
        let noun = if bg_live == 1 {
            "subagent still running"
        } else {
            "subagents still running"
        };
        let pulse = pulse_color(th.gray_dim, th.accent_running, app.spinner);
        return Some(Line::from(vec![Span::styled(
            format!("○ {bg_live} {noun}"),
            Style::default().fg(pulse),
        )]));
    }
    settled_line(app)
}

fn waiting_on_plan(app: &App, width: u16) -> Line<'static> {
    let th = theme();
    let diamond = pulse_color(th.bg_base, th.accent_plan, app.spinner);
    let parked = app.plan_approval.as_ref().is_some_and(|card| card.parked);
    let label = if parked {
        "Waiting on plan approval  ·  /view-plan"
    } else {
        "Waiting on plan approval"
    };
    let left = vec![
        Span::styled(
            "◆ ",
            Style::default().fg(diamond).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            label,
            Style::default()
                .fg(th.accent_plan)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    pad_ends(left, Vec::new(), width)
}

fn waiting_on_you(app: &App, width: u16) -> Line<'static> {
    let th = theme();
    let diamond = pulse_color(th.bg_base, th.accent_user, app.spinner);
    let extra = if app.confirmation_waiting > 0 {
        format!("  ·  {} waiting", app.confirmation_waiting)
    } else {
        String::new()
    };
    let left = vec![
        Span::styled(
            "◆ ",
            Style::default().fg(diamond).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("Waiting on you{extra}"),
            Style::default()
                .fg(th.accent_user)
                .add_modifier(Modifier::BOLD),
        ),
    ];
    pad_ends(left, Vec::new(), width)
}

fn working_line(app: &App, width: u16) -> Line<'static> {
    let th = theme();
    let running = th.accent_running;
    let frame_ch = SPINNER[app.spinner % SPINNER.len()];
    let mut left = Vec::new();
    if let Some((started_at, _)) = blocking_subagent(app) {
        left.push(Span::styled(
            format!("{frame_ch} "),
            Style::default().fg(running).add_modifier(Modifier::BOLD),
        ));
        left.push(Span::styled(
            format!(
                "Waiting on subagent… {}",
                fmt_elapsed(started_at.elapsed().as_secs())
            ),
            Style::default().fg(running).add_modifier(Modifier::BOLD),
        ));
    } else {
        let activity = app.activity_line();
        let is_working_wave = activity.starts_with("Working");
        let glyph_fg = if is_working_wave {
            pulse_color(th.gray_dim, running, app.spinner)
        } else {
            running
        };
        left.push(Span::styled(
            format!("{frame_ch} "),
            Style::default().fg(glyph_fg).add_modifier(Modifier::BOLD),
        ));
        if is_working_wave {
            left.extend(app.working_spans());
            if let Some(rest) = activity.strip_prefix("Working") {
                left.push(Span::styled(
                    rest.to_string(),
                    Style::default().fg(running).add_modifier(Modifier::BOLD),
                ));
            }
        } else {
            left.push(Span::styled(
                activity,
                Style::default().fg(running).add_modifier(Modifier::BOLD),
            ));
        }
        if !app.queue.is_empty() {
            left.push(Span::styled(
                format!(" · {} queued", app.queue.len()),
                dim(),
            ));
        }
    }

    let mut right = Vec::new();
    if let Some(started) = app.started {
        right.push(Span::styled(
            fmt_elapsed(started.elapsed().as_secs()),
            dim(),
        ));
    }
    let out = app.usage.1;
    if out > 0 {
        if !right.is_empty() {
            right.push(Span::raw(" "));
        }
        right.push(Span::styled(format!("⇣{}", fmt_count(out)), dim()));
    }
    if let Some(limits) = fmt_rate_limits(app.rate_limits) {
        if !right.is_empty() {
            right.push(Span::raw(" "));
        }
        right.push(Span::styled(limits, dim()));
    }
    if !right.is_empty() {
        right.push(Span::raw(" "));
    }
    right.push(Span::styled(
        "[stop]",
        Style::default()
            .fg(th.accent_error)
            .add_modifier(Modifier::BOLD),
    ));
    pad_ends(left, right, width)
}

fn settled_line(app: &App) -> Option<Line<'static>> {
    let th = theme();
    let latency = app
        .last_turn_latency
        .map(|elapsed| format!(" · latency {}", fmt_elapsed(elapsed.as_secs())))
        .unwrap_or_default();
    let (text, tone) = match &app.last_turn_state {
        TurnState::Warning(s) => (format!("last: warning ({s}){latency}"), th.warning),
        TurnState::Failed(s) => (
            format!("last: failed — {}{latency} · /retry", clip_reason(s)),
            th.accent_error,
        ),
        TurnState::Cancelled => (format!("last: cancelled{latency}"), th.gray_bright),
        TurnState::Done(_) | TurnState::Idle | TurnState::Running => return None,
    };
    let flash = app
        .finished_at
        .map(|t| flash_weight(t.elapsed().as_millis()))
        .filter(|&w| w > 0.0);
    let fg = match flash {
        Some(w) => lerp_color(tone, th.gray_bright, w),
        None => tone,
    };
    Some(Line::styled(text, Style::default().fg(fg)))
}

fn pad_ends(left: Vec<Span<'static>>, right: Vec<Span<'static>>, width: u16) -> Line<'static> {
    let lw: usize = left.iter().map(|s| display_width(s.content.as_ref())).sum();
    let rw: usize = right
        .iter()
        .map(|s| display_width(s.content.as_ref()))
        .sum();
    let pad = (width as usize).saturating_sub(lw).saturating_sub(rw);
    let mut spans = left;
    if pad > 0 {
        spans.push(Span::raw(" ".repeat(pad)));
    }
    spans.extend(right);
    Line::from(spans)
}

fn blocking_subagent(app: &App) -> Option<(std::time::Instant, String)> {
    app.subagents.values().find_map(|info| {
        if !info.background && info.live() {
            Some((info.started_at, info.id.clone()))
        } else {
            None
        }
    })
}

fn live_background_subagents(app: &App) -> usize {
    app.subagents
        .values()
        .filter(|info| info.background && info.live())
        .count()
}
