//! Observational inspect view and `/tasks` list for child subagents.

use std::collections::HashMap;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

use crate::App;
use crate::render::dim;
use crate::theme::theme;
use crate::util::fmt_elapsed;

const MAX_INSPECT_LINES: usize = 200;

#[derive(Clone, Debug)]
pub(crate) struct SubagentInfo {
    pub id: String,
    pub kind: String,
    pub description: String,
    pub background: bool,
    pub activity: String,
    pub started_at: Instant,
    pub finished: Option<(String, u64)>,
    pub summary: String,
    pub lines: Vec<String>,
}

impl SubagentInfo {
    pub(crate) fn live(&self) -> bool {
        self.finished.is_none()
    }

    fn elapsed_secs(&self) -> u64 {
        if let Some((_, ms)) = self.finished {
            ms / 1000
        } else {
            self.started_at.elapsed().as_secs()
        }
    }
}

pub(crate) struct InspectOverlay {
    pub id: String,
    pub scroll: u16,
}

pub(crate) struct TasksOverlay {
    pub selected: usize,
    pub rows: Vec<TaskRow>,
}

#[derive(Clone, Debug)]
pub(crate) struct TaskRow {
    pub id: String,
    pub label: String,
    pub live: bool,
    pub killable: bool,
}

pub(crate) enum OverlayOutcome {
    Continue,
    Close,
    Inspect(String),
    Kill(String),
}

pub(crate) fn open_inspect(app: &mut App, id: &str) {
    if app.subagents.contains_key(id) {
        app.inspect_subagent = Some(InspectOverlay {
            id: id.to_string(),
            scroll: 0,
        });
        app.tasks_overlay = None;
    }
}

/// Mark a tracked subagent cancelled after `/tasks` kill so the feed row
/// and chrome stop showing it as live even if the worker is aborted first.
pub(crate) fn mark_cancelled(app: &mut App, id: &str) {
    let elapsed_ms = app
        .subagents
        .get(id)
        .map(|info| info.started_at.elapsed().as_millis() as u64)
        .unwrap_or(0);
    app.apply(crate::event::UiEvent::SubagentFinished {
        id: id.to_string(),
        status: "cancelled".into(),
        elapsed_ms,
        summary: "cancelled".into(),
    });
}

pub(crate) fn open_tasks(app: &mut App, process_ids: &[String], bg_task_ids: &[String]) {
    let mut rows = Vec::new();
    let mut seen = HashMap::new();
    for (id, info) in &app.subagents {
        seen.insert(id.clone(), ());
        let status = if let Some((st, _)) = &info.finished {
            st.as_str()
        } else {
            "running"
        };
        let kind = if info.background {
            "task"
        } else {
            info.kind.as_str()
        };
        rows.push(TaskRow {
            id: id.clone(),
            label: format!(
                "{status:<10} {kind:<10} {} · {}",
                clip(&info.description, 40),
                fmt_elapsed(info.elapsed_secs())
            ),
            live: info.live(),
            killable: info.background && info.live(),
        });
    }
    for id in bg_task_ids {
        if seen.contains_key(id) {
            continue;
        }
        rows.push(TaskRow {
            id: id.clone(),
            label: format!("running    task       {id}"),
            live: true,
            killable: true,
        });
    }
    for id in process_ids {
        rows.push(TaskRow {
            id: id.clone(),
            label: format!("running    process    {id}"),
            live: true,
            killable: false,
        });
    }
    rows.sort_by(|a, b| b.live.cmp(&a.live).then_with(|| a.id.cmp(&b.id)));
    app.tasks_overlay = Some(TasksOverlay { selected: 0, rows });
}

fn clip(text: &str, max: usize) -> String {
    let count = text.chars().count();
    let clipped: String = text.chars().take(max).collect();
    if count > max {
        format!("{clipped}…")
    } else {
        clipped
    }
}

pub(crate) fn handle_inspect_key(app: &mut App, key: &KeyEvent) -> OverlayOutcome {
    let Some(overlay) = app.inspect_subagent.as_mut() else {
        return OverlayOutcome::Close;
    };
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => OverlayOutcome::Close,
        KeyCode::Up | KeyCode::Char('k') => {
            overlay.scroll = overlay.scroll.saturating_sub(1);
            OverlayOutcome::Continue
        }
        KeyCode::Down | KeyCode::Char('j') => {
            overlay.scroll = overlay.scroll.saturating_add(1);
            OverlayOutcome::Continue
        }
        KeyCode::PageUp => {
            overlay.scroll = overlay.scroll.saturating_sub(10);
            OverlayOutcome::Continue
        }
        KeyCode::PageDown => {
            overlay.scroll = overlay.scroll.saturating_add(10);
            OverlayOutcome::Continue
        }
        _ => OverlayOutcome::Continue,
    }
}

pub(crate) fn handle_tasks_key(app: &mut App, key: &KeyEvent) -> OverlayOutcome {
    let Some(overlay) = app.tasks_overlay.as_mut() else {
        return OverlayOutcome::Close;
    };
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => OverlayOutcome::Close,
        KeyCode::Up => {
            overlay.selected = overlay.selected.saturating_sub(1);
            OverlayOutcome::Continue
        }
        KeyCode::Down => {
            let last = overlay.rows.len().saturating_sub(1);
            overlay.selected = (overlay.selected + 1).min(last);
            OverlayOutcome::Continue
        }
        KeyCode::Enter => overlay
            .rows
            .get(overlay.selected)
            .map(|row| OverlayOutcome::Inspect(row.id.clone()))
            .unwrap_or(OverlayOutcome::Continue),
        KeyCode::Char('k') => overlay
            .rows
            .get(overlay.selected)
            .filter(|row| row.killable)
            .map(|row| OverlayOutcome::Kill(row.id.clone()))
            .unwrap_or(OverlayOutcome::Continue),
        _ => OverlayOutcome::Continue,
    }
}

pub(crate) fn render_inspect(frame: &mut ratatui::Frame, area: Rect, app: &App) {
    let Some(overlay) = &app.inspect_subagent else {
        return;
    };
    let th = theme();
    let info = app.subagents.get(&overlay.id);
    let title = if let Some(info) = info {
        let activity = if let Some((status, _)) = &info.finished {
            status.clone()
        } else if info.activity.is_empty() {
            "running".into()
        } else {
            info.activity.clone()
        };
        format!(
            " {} · {} · {activity} · {} ",
            info.kind,
            clip(&info.description, 48),
            fmt_elapsed(info.elapsed_secs())
        )
    } else {
        format!(" {} ", overlay.id)
    };
    let mut lines: Vec<Line<'static>> = Vec::new();
    if let Some(info) = info {
        if info.lines.is_empty() {
            lines.push(Line::styled(
                "No retained child output yet.".to_string(),
                dim(),
            ));
        }
        for line in info.lines.iter().take(MAX_INSPECT_LINES) {
            lines.push(Line::raw(line.clone()));
        }
        if !info.summary.is_empty() {
            lines.push(Line::default());
            lines.push(Line::styled(info.summary.clone(), dim()));
        }
    } else {
        lines.push(Line::styled(
            "Subagent is no longer tracked.".to_string(),
            dim(),
        ));
    }
    lines.push(Line::default());
    lines.push(Line::styled("Esc/q close", dim()));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(th.accent_running))
        .title(Span::styled(
            title,
            Style::default()
                .fg(th.text_primary)
                .add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((overlay.scroll, 0)),
        area,
    );
}

pub(crate) fn render_tasks(frame: &mut ratatui::Frame, area: Rect, overlay: &TasksOverlay) {
    let th = theme();
    let mut lines = vec![
        Line::styled(
            "TASKS",
            Style::default()
                .fg(th.text_primary)
                .add_modifier(Modifier::BOLD),
        ),
        Line::styled("status     kind       description", dim()),
    ];
    if overlay.rows.is_empty() {
        lines.push(Line::styled(
            "No subagents or background tasks.".to_string(),
            dim(),
        ));
    }
    for (index, row) in overlay.rows.iter().enumerate() {
        let marker = if index == overlay.selected {
            "›"
        } else {
            " "
        };
        let style = if index == overlay.selected {
            Style::default()
                .fg(th.text_primary)
                .add_modifier(Modifier::BOLD)
        } else {
            dim()
        };
        lines.push(Line::styled(format!("{marker} {}", row.label), style));
    }
    lines.push(Line::styled(
        "↑/↓ select · Enter inspect · k kill · Esc close",
        dim(),
    ));
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(th.accent_assistant))
        .title(" tasks ");
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}
