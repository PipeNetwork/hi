//! `/jump` and `/rewind` pickers over user turns.

use crossterm::event::{KeyCode, KeyEvent};
use hi_agent::UserTurn;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use crate::render::dim;
use crate::theme::theme;
use crate::{App, TranscriptEntry};

pub(crate) struct JumpPicker {
    pub previews: Vec<String>,
    pub selected: usize,
    pub restore_scroll: u16,
    pub restore_following: bool,
}

pub(crate) struct RewindPicker {
    pub turns: Vec<UserTurn>,
    pub selected: usize,
    pub confirm: bool,
}

pub(crate) enum PickerOutcome {
    Continue,
    Close,
    /// `/rewind n` — apply through the command path so agent + transcript stay aligned.
    Rewind(usize),
}

impl JumpPicker {
    pub(crate) fn from_app(app: &App) -> Option<Self> {
        let previews = user_prompt_previews(app);
        if previews.is_empty() {
            return None;
        }
        let selected = previews.len().saturating_sub(1);
        Some(Self {
            previews,
            selected,
            restore_scroll: app.scroll,
            restore_following: app.following,
        })
    }
}

impl RewindPicker {
    pub(crate) fn new(turns: Vec<UserTurn>) -> Option<Self> {
        if turns.is_empty() {
            return None;
        }
        let selected = turns.len().saturating_sub(1);
        Some(Self {
            turns,
            selected,
            confirm: false,
        })
    }

    fn current_n(&self) -> Option<usize> {
        self.turns.get(self.selected).map(|t| t.n)
    }
}

pub(crate) fn handle_jump_key(app: &mut App, key: &KeyEvent) -> PickerOutcome {
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            if let Some(picker) = app.jump_picker.take() {
                app.scroll = picker.restore_scroll;
                app.following = picker.restore_following;
            }
            PickerOutcome::Close
        }
        KeyCode::Enter => {
            app.jump_picker = None;
            PickerOutcome::Close
        }
        KeyCode::Up | KeyCode::Char('k') => {
            let sel = app.jump_picker.as_mut().map(|p| {
                p.selected = p.selected.saturating_sub(1);
                p.selected
            });
            if let Some(sel) = sel {
                app.scroll_to_user_prompt(sel);
            }
            PickerOutcome::Continue
        }
        KeyCode::Down | KeyCode::Char('j') => {
            let sel = app.jump_picker.as_mut().map(|p| {
                let len = p.previews.len();
                if len > 0 {
                    p.selected = (p.selected + 1).min(len - 1);
                }
                p.selected
            });
            if let Some(sel) = sel {
                app.scroll_to_user_prompt(sel);
            }
            PickerOutcome::Continue
        }
        _ => PickerOutcome::Continue,
    }
}

pub(crate) fn handle_rewind_key(app: &mut App, key: &KeyEvent) -> PickerOutcome {
    let Some(picker) = app.rewind_picker.as_mut() else {
        return PickerOutcome::Close;
    };
    let len = picker.turns.len();
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            if picker.confirm {
                picker.confirm = false;
                PickerOutcome::Continue
            } else {
                PickerOutcome::Close
            }
        }
        KeyCode::Enter => {
            if picker.confirm {
                if let Some(n) = picker.current_n() {
                    PickerOutcome::Rewind(n)
                } else {
                    PickerOutcome::Close
                }
            } else {
                picker.confirm = true;
                PickerOutcome::Continue
            }
        }
        KeyCode::Up | KeyCode::Char('k') if !picker.confirm => {
            picker.selected = picker.selected.saturating_sub(1);
            PickerOutcome::Continue
        }
        KeyCode::Down | KeyCode::Char('j') if !picker.confirm => {
            if len > 0 {
                picker.selected = (picker.selected + 1).min(len - 1);
            }
            PickerOutcome::Continue
        }
        _ => PickerOutcome::Continue,
    }
}

pub(crate) fn render_jump(frame: &mut ratatui::Frame, area: Rect, picker: &JumpPicker) {
    render_list(
        frame,
        area,
        " jump · live-scroll · Enter stay · Esc restore ",
        &picker
            .previews
            .iter()
            .enumerate()
            .map(|(i, p)| format!("{:>3}. {p}", i + 1))
            .collect::<Vec<_>>(),
        picker.selected,
        None,
    );
}

pub(crate) fn render_rewind(frame: &mut ratatui::Frame, area: Rect, picker: &RewindPicker) {
    let rows: Vec<String> = picker
        .turns
        .iter()
        .map(|t| format!("{:>3}. {}", t.n, t.preview))
        .collect();
    let footer = if picker.confirm {
        picker
            .current_n()
            .map(|n| format!("rewind conversation before turn {n}? Enter confirm · Esc cancel"))
    } else {
        Some("Enter confirm · j/k move · Esc close — files unchanged, /undo reverts edits".into())
    };
    render_list(
        frame,
        area,
        " rewind · truncate conversation before this turn ",
        &rows,
        picker.selected,
        footer.as_deref(),
    );
}

fn render_list(
    frame: &mut ratatui::Frame,
    area: Rect,
    title: &str,
    rows: &[String],
    selected: usize,
    footer: Option<&str>,
) {
    let th = theme();
    let mut lines: Vec<Line<'static>> = Vec::new();
    let visible = area.height.saturating_sub(3) as usize;
    let start = selected.saturating_sub(visible.saturating_sub(1).min(selected));
    for (i, row) in rows.iter().enumerate().skip(start).take(visible.max(1)) {
        let style = if i == selected {
            Style::default()
                .fg(th.text_primary)
                .bg(th.selection_bg)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(th.text_secondary)
        };
        lines.push(Line::styled(row.clone(), style));
    }
    if let Some(footer) = footer {
        lines.push(Line::raw(""));
        lines.push(Line::styled(footer.to_string(), dim()));
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(th.accent_plan))
        .title(Span::styled(
            title.to_string(),
            Style::default()
                .fg(th.text_primary)
                .add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn user_prompt_previews(app: &App) -> Vec<String> {
    app.transcript
        .iter()
        .filter_map(|e| match e {
            TranscriptEntry::UserPrompt { line, .. } => {
                let t = crate::render::line_text(line);
                let t = t.trim().trim_start_matches('❯').trim();
                Some(t.chars().take(72).collect())
            }
            _ => None,
        })
        .collect()
}

impl App {
    pub(crate) fn open_jump_picker(&mut self) {
        self.rewind_picker = None;
        match JumpPicker::from_app(self) {
            Some(picker) => {
                let sel = picker.selected;
                self.jump_picker = Some(picker);
                let _ = self.scroll_to_user_prompt(sel);
            }
            None => self.status = "no user prompts to jump to".into(),
        }
    }

    pub(crate) fn open_rewind_picker(&mut self, agent: &hi_agent::Agent) {
        self.jump_picker = None;
        let turns = hi_agent::list_user_turns(agent.messages());
        match RewindPicker::new(turns) {
            Some(picker) => self.rewind_picker = Some(picker),
            None => self.status = "no user turns yet".into(),
        }
    }

    pub(crate) fn scroll_to_user_prompt(&mut self, ord: usize) -> bool {
        let width = self.view_cache.width.max(40);
        let nav = self.mode.is_block_nav().then(|| self.selected_block_ord());
        self.ensure_view_cache(width, nav);
        if let Some(&idx) = self.view_cache.prompt_line_starts.get(ord)
            && let Some(&row) = self.view_cache.prefix.get(idx)
        {
            self.scroll_to(row.min(u16::MAX as u32) as u16);
            true
        } else {
            false
        }
    }

    /// Drop the chosen user prompt and everything after it from the TUI feed.
    pub(crate) fn rewind_transcript_to_user_turn(&mut self, n: usize) {
        let mut seen = 0usize;
        let mut cut = None;
        for (i, entry) in self.transcript.iter().enumerate() {
            if matches!(entry, TranscriptEntry::UserPrompt { .. }) {
                seen += 1;
                if seen == n {
                    cut = Some(i);
                    break;
                }
            }
        }
        if let Some(i) = cut {
            self.transcript.truncate(i);
            self.freeze_verb_group();
            self.pending = None;
            self.following = true;
            self.bump_transcript();
        }
    }
}
