//! Fullscreen viewer for one foldable transcript block (Ctrl-F).
//!
//! Opens the selected Read/Edit/Run/tool row with thinking and hunks expanded,
//! then lets you search and copy without dumping every explore path into the feed.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

use crate::render::line_text;
use crate::theme::theme;
use crate::{App, Density, TranscriptEntry};

pub(crate) struct BlockViewer {
    pub title: String,
    pub lines: Vec<Line<'static>>,
    pub texts: Vec<String>,
    pub scroll: u16,
    pub search: Option<String>,
    pub typing: bool,
    pub match_idx: usize,
    pub matches: Vec<usize>,
}

pub(crate) enum ViewerOutcome {
    Continue,
    Close,
}

impl BlockViewer {
    pub(crate) fn open(app: &App, ord: usize) -> Option<Self> {
        let (title, lines) = block_content(app, ord)?;
        let texts: Vec<String> = lines.iter().map(line_text).collect();
        Some(Self {
            title,
            lines,
            texts,
            scroll: 0,
            search: None,
            typing: false,
            match_idx: 0,
            matches: Vec::new(),
        })
    }

    fn max_scroll(&self, height: u16) -> u16 {
        self.lines
            .len()
            .saturating_sub(height.saturating_sub(2) as usize)
            .min(u16::MAX as usize) as u16
    }

    fn recompute_matches(&mut self) {
        let Some(q) = self.search.as_deref().filter(|s| !s.is_empty()) else {
            self.matches.clear();
            self.match_idx = 0;
            return;
        };
        let needle = q.to_ascii_lowercase();
        self.matches = self
            .texts
            .iter()
            .enumerate()
            .filter(|(_, t)| t.to_ascii_lowercase().contains(&needle))
            .map(|(i, _)| i)
            .collect();
        self.match_idx = 0;
        if let Some(&i) = self.matches.first() {
            self.scroll = i.min(u16::MAX as usize) as u16;
        }
    }

    fn next_match(&mut self, dir: i32) {
        if self.matches.is_empty() {
            return;
        }
        let n = self.matches.len();
        if dir >= 0 {
            self.match_idx = (self.match_idx + 1) % n;
        } else {
            self.match_idx = (self.match_idx + n - 1) % n;
        }
        if let Some(&i) = self.matches.get(self.match_idx) {
            self.scroll = i.min(u16::MAX as usize) as u16;
        }
    }
}

pub(crate) fn handle_key(app: &mut App, key: &KeyEvent) -> ViewerOutcome {
    let mut copy_text: Option<String> = None;
    let outcome = {
        let Some(viewer) = app.block_viewer.as_mut() else {
            return ViewerOutcome::Close;
        };
        if viewer.typing {
            match key.code {
                KeyCode::Esc => {
                    viewer.typing = false;
                    viewer.search = None;
                    viewer.matches.clear();
                }
                KeyCode::Enter => {
                    viewer.typing = false;
                    viewer.recompute_matches();
                }
                KeyCode::Backspace => {
                    if let Some(q) = viewer.search.as_mut() {
                        q.pop();
                    }
                }
                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                    viewer.search.get_or_insert_with(String::new).push(c);
                }
                _ => {}
            }
            return ViewerOutcome::Continue;
        }
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => ViewerOutcome::Close,
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                ViewerOutcome::Close
            }
            KeyCode::Up | KeyCode::Char('k') => {
                viewer.scroll = viewer.scroll.saturating_sub(1);
                ViewerOutcome::Continue
            }
            KeyCode::Down | KeyCode::Char('j') => {
                viewer.scroll = viewer.scroll.saturating_add(1);
                ViewerOutcome::Continue
            }
            KeyCode::PageUp => {
                viewer.scroll = viewer.scroll.saturating_sub(10);
                ViewerOutcome::Continue
            }
            KeyCode::PageDown => {
                viewer.scroll = viewer.scroll.saturating_add(10);
                ViewerOutcome::Continue
            }
            KeyCode::Home | KeyCode::Char('g') => {
                viewer.scroll = 0;
                ViewerOutcome::Continue
            }
            KeyCode::End | KeyCode::Char('G') => {
                viewer.scroll = u16::MAX;
                ViewerOutcome::Continue
            }
            KeyCode::Char('/') => {
                viewer.typing = true;
                viewer.search = Some(String::new());
                ViewerOutcome::Continue
            }
            KeyCode::Char('n') => {
                viewer.next_match(1);
                ViewerOutcome::Continue
            }
            KeyCode::Char('N') => {
                viewer.next_match(-1);
                ViewerOutcome::Continue
            }
            KeyCode::Char('y') => {
                copy_text = Some(viewer.texts.join("\n"));
                ViewerOutcome::Continue
            }
            _ => ViewerOutcome::Continue,
        }
    };
    if let Some(text) = copy_text {
        match crate::util::copy_to_clipboard(&text) {
            Ok(()) => app.status = format!("copied {} chars", text.len()),
            Err(err) => app.status = format!("copy failed: {err}"),
        }
    }
    outcome
}

pub(crate) fn render(frame: &mut ratatui::Frame, area: Rect, app: &mut App) {
    let Some(viewer) = app.block_viewer.as_mut() else {
        return;
    };
    let th = theme();
    let max = viewer.max_scroll(area.height);
    viewer.scroll = viewer.scroll.min(max);
    let search_hint = match (viewer.typing, viewer.search.as_deref()) {
        (true, Some(q)) => format!("  /{q}█"),
        (false, Some(q)) if !q.is_empty() => {
            format!(
                "  /{q}  {}/{}",
                viewer.match_idx.saturating_add(1).min(viewer.matches.len()),
                viewer.matches.len()
            )
        }
        _ => String::new(),
    };
    let title = format!(
        " {}{search_hint} · j/k scroll · / search · y copy · Esc close ",
        viewer.title
    );
    let highlight = viewer
        .matches
        .get(viewer.match_idx)
        .copied()
        .filter(|_| !viewer.typing);
    let mut lines = viewer.lines.clone();
    if let Some(i) = highlight
        && let Some(line) = lines.get_mut(i)
    {
        line.style = line.style.bg(th.selection_bg);
    }
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
            .scroll((viewer.scroll, 0)),
        area,
    );
}

fn block_content(app: &App, ord: usize) -> Option<(String, Vec<Line<'static>>)> {
    let mut o = 0usize;
    for entry in &app.transcript {
        if !entry.is_foldable() {
            continue;
        }
        if o != ord {
            o += 1;
            continue;
        }
        let lines = match entry {
            TranscriptEntry::Activity(block) => {
                let mut clone = block.clone();
                clone.expanded = true;
                clone.flatten(true, true, Density::Verbose)
            }
            TranscriptEntry::ToolOutput { body, .. } => body.clone(),
            TranscriptEntry::Btw {
                question, answer, ..
            } => {
                let mut lines = vec![Line::from(Span::styled(
                    format!("/btw {question}"),
                    Style::default().add_modifier(Modifier::BOLD),
                ))];
                lines.push(Line::raw(""));
                lines.extend(crate::render::markdown_body_lines(answer));
                lines
            }
            other => other.flatten(true, true, Density::Verbose),
        };
        let title = lines
            .first()
            .map(line_text)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "block".into());
        let title: String = title.chars().take(48).collect();
        return Some((title, lines));
    }
    None
}

fn subagent_id_at(app: &App, target: usize) -> Option<String> {
    let mut ord = 0usize;
    for entry in &app.transcript {
        if !entry.is_foldable() {
            continue;
        }
        if ord == target {
            if let TranscriptEntry::Activity(block) = entry {
                return block.subagent_id().map(str::to_string);
            }
            return None;
        }
        ord += 1;
    }
    None
}

/// Open the fullscreen viewer for the block-nav selection, or the latest
/// foldable block when not in block-nav. Subagent rows open inspect instead.
pub(crate) fn open_selected(app: &mut App) {
    let n = app.tool_block_count();
    if n == 0 {
        app.status = "no foldable block to view".into();
        return;
    }
    let ord = if app.mode.is_block_nav() {
        app.selected_block_ord()
    } else {
        n - 1
    };
    if let Some(id) = subagent_id_at(app, ord) {
        crate::subagent_overlay::open_inspect(app, &id);
        return;
    }
    match BlockViewer::open(app, ord) {
        Some(viewer) => app.block_viewer = Some(viewer),
        None => app.status = "couldn't open that block".into(),
    }
}
