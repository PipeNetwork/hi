//! `/memory` browser: project + global markdown files, list + preview.

use std::path::Path;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{BorderType, Paragraph, Wrap};

use crate::App;
use crate::render::dim;
use crate::theme::{UiTone, theme};

const PREVIEW_CAP: usize = 80;

#[derive(Clone, Debug)]
pub(crate) struct MemoryFile {
    pub source: &'static str,
    pub path: std::path::PathBuf,
    pub body: String,
}

#[derive(Clone, Debug)]
pub(crate) struct MemoryBrowser {
    pub files: Vec<MemoryFile>,
    pub selected: usize,
    pub scroll: usize,
}

impl MemoryBrowser {
    pub(crate) fn open(workspace: &Path) -> Self {
        let project = hi_agent::memory_file_at(workspace);
        let global = hi_agent::global_memory_file();
        let files = [("project", project), ("global", global)]
            .into_iter()
            .map(|(source, path)| {
                let body = std::fs::read_to_string(&path).unwrap_or_default();
                MemoryFile { source, path, body }
            })
            .collect();
        Self {
            files,
            selected: 0,
            scroll: 0,
        }
    }
}

pub(crate) fn handle_key(app: &mut App, key: &KeyEvent) -> bool {
    let Some(browser) = app.memory_browser.as_mut() else {
        return false;
    };
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') if !ctrl => {
            app.memory_browser = None;
            true
        }
        KeyCode::Up | KeyCode::Char('k') if !ctrl => {
            browser.selected = browser.selected.saturating_sub(1);
            browser.scroll = 0;
            true
        }
        KeyCode::Down | KeyCode::Char('j') if !ctrl => {
            let last = browser.files.len().saturating_sub(1);
            browser.selected = (browser.selected + 1).min(last);
            browser.scroll = 0;
            true
        }
        KeyCode::PageUp => {
            browser.scroll = browser.scroll.saturating_sub(10);
            true
        }
        KeyCode::PageDown => {
            browser.scroll = browser.scroll.saturating_add(10);
            true
        }
        _ => true,
    }
}

pub(crate) fn render(frame: &mut ratatui::Frame, area: Rect, browser: &MemoryBrowser) {
    let th = theme();
    let chunks =
        Layout::vertical([Constraint::Percentage(35), Constraint::Percentage(65)]).split(area);
    let mut list = vec![Line::styled(
        "Memory  ·  j/k select  ·  Esc close  ·  /remember to add",
        dim(),
    )];
    for (i, file) in browser.files.iter().enumerate() {
        let exists = file.path.exists();
        let label = format!("{}  {}", file.source, file.path.display());
        if i == browser.selected {
            list.push(Line::styled(
                format!("▶ {label}"),
                Style::default()
                    .fg(th.accent_plan)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            list.push(Line::styled(format!("  {label}"), dim()));
        }
        if !exists {
            list.push(Line::styled("    (empty — not written yet)", dim()));
        }
    }
    let list_block = th
        .panel_block(" memory ", UiTone::Info)
        .border_type(BorderType::Rounded);
    frame.render_widget(Paragraph::new(list).block(list_block), chunks[0]);

    let file = browser.files.get(browser.selected);
    let mut preview: Vec<Line> = Vec::new();
    if let Some(file) = file {
        let body = file.body.trim();
        if body.is_empty() {
            preview.push(Line::styled("No notes in this file yet.", dim()));
        } else {
            preview.extend(
                crate::render::markdown_body_lines(body)
                    .into_iter()
                    .skip(browser.scroll)
                    .take(PREVIEW_CAP),
            );
        }
    }
    let title = file
        .map(|f| format!(" {} ", f.source))
        .unwrap_or_else(|| " preview ".into());
    let preview_block = th
        .panel_block(&title, UiTone::Muted)
        .border_type(BorderType::Rounded)
        .title_bottom(Line::from(Span::styled(" PgUp/PgDn scroll ", dim())));
    frame.render_widget(
        Paragraph::new(preview)
            .block(preview_block)
            .wrap(Wrap { trim: false }),
        chunks[1],
    );
}
