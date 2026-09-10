//! One turn picker for `/jump` and `/rewind`.

use crossterm::event::{KeyCode, KeyEvent};
use hi_agent::UserTurn;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph};

use crate::render::dim;
use crate::theme::theme;
use crate::{App, TranscriptEntry};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnPickerMode {
    /// Live-scroll; Enter keeps the new position.
    Jump,
    /// Truncate conversation; Enter confirms.
    Rewind,
}

pub(crate) struct TurnRow {
    pub n: usize,
    pub preview: String,
}

pub(crate) struct TurnPicker {
    pub rows: Vec<TurnRow>,
    pub selected: usize,
    pub mode: TurnPickerMode,
    pub restore_scroll: u16,
    pub restore_following: bool,
    pub confirm: bool,
}

pub(crate) enum PickerOutcome {
    Continue,
    Close,
    /// `/rewind n` — apply through the command path so agent + transcript stay aligned.
    Rewind(usize),
}

impl TurnPicker {
    fn jump_from_app(app: &App) -> Option<Self> {
        let rows = user_prompt_rows(app);
        if rows.is_empty() {
            return None;
        }
        let selected = rows.len().saturating_sub(1);
        Some(Self {
            rows,
            selected,
            mode: TurnPickerMode::Jump,
            restore_scroll: app.scroll,
            restore_following: app.following,
            confirm: false,
        })
    }

    pub(crate) fn rewind(turns: Vec<UserTurn>) -> Option<Self> {
        if turns.is_empty() {
            return None;
        }
        let selected = turns.len().saturating_sub(1);
        Some(Self {
            rows: turns
                .into_iter()
                .map(|t| TurnRow {
                    n: t.n,
                    preview: t.preview,
                })
                .collect(),
            selected,
            mode: TurnPickerMode::Rewind,
            restore_scroll: 0,
            restore_following: true,
            confirm: false,
        })
    }

    fn current_n(&self) -> Option<usize> {
        self.rows.get(self.selected).map(|t| t.n)
    }
}

pub(crate) fn handle_turn_picker_key(app: &mut App, key: &KeyEvent) -> PickerOutcome {
    let mode = match app.turn_picker.as_ref() {
        Some(p) => p.mode,
        None => return PickerOutcome::Close,
    };
    let confirming = app
        .turn_picker
        .as_ref()
        .is_some_and(|p| p.confirm && p.mode == TurnPickerMode::Rewind);
    match key.code {
        KeyCode::Esc | KeyCode::Char('q') => {
            if confirming {
                if let Some(picker) = app.turn_picker.as_mut() {
                    picker.confirm = false;
                }
                PickerOutcome::Continue
            } else if mode == TurnPickerMode::Jump {
                if let Some(picker) = app.turn_picker.take() {
                    app.scroll = picker.restore_scroll;
                    app.following = picker.restore_following;
                }
                PickerOutcome::Close
            } else {
                app.turn_picker = None;
                PickerOutcome::Close
            }
        }
        KeyCode::Enter => match mode {
            TurnPickerMode::Jump => {
                app.turn_picker = None;
                PickerOutcome::Close
            }
            TurnPickerMode::Rewind if confirming => {
                let n = app.turn_picker.as_ref().and_then(TurnPicker::current_n);
                if let Some(n) = n {
                    PickerOutcome::Rewind(n)
                } else {
                    PickerOutcome::Close
                }
            }
            TurnPickerMode::Rewind => {
                if let Some(picker) = app.turn_picker.as_mut() {
                    picker.confirm = true;
                }
                PickerOutcome::Continue
            }
        },
        KeyCode::Up | KeyCode::Char('k') if !confirming => {
            if let Some(picker) = app.turn_picker.as_mut() {
                picker.selected = picker.selected.saturating_sub(1);
            }
            let sel = app.turn_picker.as_ref().map(|p| p.selected);
            if mode == TurnPickerMode::Jump
                && let Some(sel) = sel
            {
                app.scroll_to_user_prompt(sel);
            }
            PickerOutcome::Continue
        }
        KeyCode::Down | KeyCode::Char('j') if !confirming => {
            if let Some(picker) = app.turn_picker.as_mut() {
                let len = picker.rows.len();
                if len > 0 {
                    picker.selected = (picker.selected + 1).min(len - 1);
                }
            }
            let sel = app.turn_picker.as_ref().map(|p| p.selected);
            if mode == TurnPickerMode::Jump
                && let Some(sel) = sel
            {
                app.scroll_to_user_prompt(sel);
            }
            PickerOutcome::Continue
        }
        _ => PickerOutcome::Continue,
    }
}

pub(crate) fn render_turn_picker(frame: &mut ratatui::Frame, area: Rect, picker: &TurnPicker) {
    let rows: Vec<String> = picker
        .rows
        .iter()
        .map(|t| format!("{:>3}. {}", t.n, t.preview))
        .collect();
    match picker.mode {
        TurnPickerMode::Jump => render_list(
            frame,
            area,
            " jump · live-scroll · Enter stay · Esc restore ",
            &rows,
            picker.selected,
            None,
        ),
        TurnPickerMode::Rewind => {
            let footer = if picker.confirm {
                picker.current_n().map(|n| {
                    format!("rewind conversation before turn {n}? Enter confirm · Esc cancel")
                })
            } else {
                Some(
                    "Enter confirm · j/k move · Esc close — files unchanged, /undo reverts edits"
                        .into(),
                )
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
    }
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

fn user_prompt_rows(app: &App) -> Vec<TurnRow> {
    app.transcript
        .iter()
        .filter_map(|e| match e {
            TranscriptEntry::UserPrompt { line, .. } => {
                let t = crate::render::line_text(line);
                let t = t.trim().trim_start_matches('❯').trim();
                Some(t.chars().take(72).collect::<String>())
            }
            _ => None,
        })
        .enumerate()
        .map(|(i, preview)| TurnRow { n: i + 1, preview })
        .collect()
}

impl App {
    pub(crate) fn open_jump_picker(&mut self) {
        match TurnPicker::jump_from_app(self) {
            Some(picker) => {
                let sel = picker.selected;
                self.turn_picker = Some(picker);
                let _ = self.scroll_to_user_prompt(sel);
            }
            None => self.status = "no user prompts to jump to".into(),
        }
    }

    pub(crate) fn open_rewind_picker(&mut self, agent: &hi_agent::Agent) {
        let turns = hi_agent::list_user_turns(agent.messages());
        match TurnPicker::rewind(turns) {
            Some(picker) => self.turn_picker = Some(picker),
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
