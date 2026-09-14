//! Grok-style `/dashboard` fullscreen roster.

use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hi_harness::{Dashboard, DashboardKnobs, DispatchOpts, Harness, PIPE_GPT6, RowState, RowView};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph, Wrap};

use crate::chrome::{self, ShortcutHint, display_cwd};
use crate::input::InputLine;
use crate::layout::{display_width, truncate_display};
use crate::theme::UiTone;
use crate::{App, SPINNER};

fn cell(text: &str, width: usize) -> String {
    let text = truncate_display(text, width);
    let pad = width.saturating_sub(display_width(&text));
    format!("{text}{}", " ".repeat(pad))
}

fn hint(key: &'static str, label: &'static str) -> ShortcutHint {
    ShortcutHint { key, label }
}

fn roster_hints(peek: bool, draft_empty: bool, cancel_armed: bool) -> Vec<ShortcutHint> {
    if cancel_armed {
        return vec![
            hint("ctrl+x", "delete"),
            hint("esc", "keep"),
            hint("?", "help"),
        ];
    }
    if peek {
        let mut hints = vec![
            hint("enter", if draft_empty { "attach" } else { "reply" }),
            hint("ctrl+s", "open"),
            hint("ctrl+x", "stop"),
        ];
        if draft_empty {
            hints.push(hint("↑/↓", "select"));
        }
        hints.extend([hint("tab", "list"), hint("?", "help"), hint("esc", "back")]);
        return hints;
    }
    vec![
        hint("enter", if draft_empty { "sub-agent" } else { "dispatch" }),
        hint("ctrl+s", "open"),
        hint("ctrl+m", "sub model"),
        hint("ctrl+w", "worktree"),
        hint("tab", "list"),
        hint("?", "help"),
        hint("esc", "close"),
    ]
}

fn attached_hints() -> Vec<ShortcutHint> {
    vec![
        hint("enter", "send"),
        hint("ctrl+x", "cancel"),
        hint("esc", "back"),
        hint("ctrl+\\", "dashboard"),
        hint("?", "help"),
    ]
}

fn help_hints() -> Vec<ShortcutHint> {
    vec![hint("esc", "close"), hint("?", "close")]
}

const CANCEL_DELETE_WINDOW: Duration = Duration::from_secs(2);
const TOAST_TTL: Duration = Duration::from_secs(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Focus {
    List,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DashAction {
    None,
    Close,
    /// Spawn a new agent from the dispatch box.
    Dispatch {
        attach: bool,
    },
    /// Idle untitled agent (`+ New Agent` + empty Enter).
    SpawnIdle,
    Reply {
        attach: bool,
    },
    Attach,
    Cancel,
    Delete,
    ToggleWorktree,
    CycleModel,
}

pub(crate) struct DashboardOverlay {
    pub runtime: Dashboard,
    pub focus: Focus,
    /// Selected row id. `None` = `+ New sub-agent` cursor.
    pub selected: Option<String>,
    pub draft: InputLine,
    pub worktree_next: bool,
    /// When true, next dispatch uses Pipe `pipe/gpt-6` instead of the session model.
    pub gpt6_next: bool,
    pub attached: Option<String>,
    pub help: bool,
    /// False when the roster is hidden but workers must keep running.
    pub visible: bool,
    cancel_armed: Option<(String, Instant)>,
    toast: Option<(String, Instant)>,
}

impl DashboardOverlay {
    pub fn new(runtime: Dashboard) -> Self {
        let empty = runtime.roster().is_empty();
        Self {
            runtime,
            focus: if empty { Focus::Input } else { Focus::List },
            selected: None,
            draft: InputLine::default(),
            worktree_next: false,
            gpt6_next: false,
            attached: None,
            help: false,
            visible: true,
            cancel_armed: None,
            toast: None,
        }
    }

    fn toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), Instant::now()));
    }

    fn selected_id(&self) -> Option<String> {
        self.selected.clone()
    }

    fn selected_index(&self) -> Option<usize> {
        let id = self.selected.as_ref()?;
        self.runtime.roster().iter().position(|r| &r.id == id)
    }

    pub fn handle_key(&mut self, key: &KeyEvent) -> DashAction {
        if self.help {
            match key.code {
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q') => {
                    self.help = false;
                    DashAction::None
                }
                _ => DashAction::None,
            }
        } else if self.attached.is_some() {
            self.handle_attached_key(key)
        } else {
            self.handle_roster_key(key)
        }
    }

    fn handle_attached_key(&mut self, key: &KeyEvent) -> DashAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if is_dashboard_toggle(key) || (key.code == KeyCode::Esc && self.draft.is_empty()) {
            self.attached = None;
            self.focus = Focus::List;
            return DashAction::None;
        }
        match key.code {
            KeyCode::Char('?') if !ctrl && self.draft.is_empty() => {
                self.help = true;
                DashAction::None
            }
            KeyCode::Char('c' | 'x') if ctrl => DashAction::Cancel,
            KeyCode::Enter if !self.draft.is_empty() => DashAction::Reply { attach: false },
            KeyCode::Char(c) if !ctrl => {
                self.draft.insert(c);
                DashAction::None
            }
            KeyCode::Backspace => {
                self.draft.backspace();
                DashAction::None
            }
            _ => DashAction::None,
        }
    }

    fn handle_roster_key(&mut self, key: &KeyEvent) -> DashAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let n = self.runtime.roster().len();

        if is_dashboard_toggle(key) {
            return DashAction::Close;
        }
        if matches!(key.code, KeyCode::Char('c')) && ctrl {
            if self.selected.is_some() {
                return DashAction::Cancel;
            }
            return DashAction::Close;
        }
        if matches!(key.code, KeyCode::Char('?')) && self.draft.is_empty() && !ctrl {
            self.help = true;
            return DashAction::None;
        }
        if matches!(key.code, KeyCode::Char('w')) && ctrl {
            return DashAction::ToggleWorktree;
        }
        if matches!(key.code, KeyCode::Char('m')) && ctrl {
            return DashAction::CycleModel;
        }
        if matches!(key.code, KeyCode::Char('x')) && ctrl {
            if let Some(id) = self.selected_id() {
                if let Some((armed, at)) = &self.cancel_armed
                    && armed == &id
                    && at.elapsed() < CANCEL_DELETE_WINDOW
                {
                    self.cancel_armed = None;
                    return DashAction::Delete;
                }
                self.cancel_armed = Some((id, Instant::now()));
                return DashAction::Cancel;
            }
            return DashAction::None;
        }
        if matches!(key.code, KeyCode::Char('s')) && ctrl {
            if self.selected_index().is_some() {
                if self.draft.is_empty() {
                    return DashAction::Attach;
                }
                return DashAction::Reply { attach: true };
            }
            if self.draft.is_empty() {
                return DashAction::SpawnIdle;
            }
            return DashAction::Dispatch { attach: true };
        }
        match key.code {
            KeyCode::Esc => {
                if !self.draft.is_empty() {
                    self.draft = InputLine::default();
                    DashAction::None
                } else if self.selected.is_some() {
                    self.selected = None;
                    DashAction::None
                } else {
                    DashAction::Close
                }
            }
            KeyCode::Tab if !shift => {
                self.focus = match self.focus {
                    Focus::List => Focus::Input,
                    Focus::Input => Focus::List,
                };
                DashAction::None
            }
            KeyCode::Up => {
                self.move_sel(-1, n);
                DashAction::None
            }
            KeyCode::Down => {
                self.move_sel(1, n);
                DashAction::None
            }
            KeyCode::Char('k') if self.focus == Focus::List => {
                self.move_sel(-1, n);
                DashAction::None
            }
            KeyCode::Char('j') if self.focus == Focus::List => {
                self.move_sel(1, n);
                DashAction::None
            }
            KeyCode::Enter if alt || shift => {
                self.draft.insert('\n');
                DashAction::None
            }
            KeyCode::Enter => {
                if self.selected_index().is_some() {
                    if self.draft.is_empty() {
                        DashAction::Attach
                    } else {
                        DashAction::Reply { attach: false }
                    }
                } else if self.draft.is_empty() {
                    DashAction::SpawnIdle
                } else {
                    DashAction::Dispatch { attach: false }
                }
            }
            KeyCode::Char(c) if !ctrl && !alt => {
                self.focus = Focus::Input;
                self.draft.insert(c);
                DashAction::None
            }
            KeyCode::Backspace => {
                self.draft.backspace();
                DashAction::None
            }
            KeyCode::Left => {
                self.draft.left();
                DashAction::None
            }
            KeyCode::Right => {
                self.draft.right();
                DashAction::None
            }
            _ => DashAction::None,
        }
    }

    fn move_sel(&mut self, dir: i32, n: usize) {
        if n == 0 {
            self.selected = None;
            return;
        }
        let roster = self.runtime.roster();
        let cur = self
            .selected
            .as_ref()
            .and_then(|id| roster.iter().position(|r| &r.id == id));
        let next = match cur {
            None if dir > 0 => Some(0),
            None => Some(n - 1),
            Some(i) => {
                let v = i as i32 + dir;
                if (0..n as i32).contains(&v) {
                    Some(v as usize)
                } else {
                    None
                }
            }
        };
        self.selected = next.and_then(|i| roster.get(i).map(|r| r.id.clone()));
        if self.selected.is_some() {
            self.draft = InputLine::default();
        }
    }
}

pub(crate) fn is_dashboard_toggle(key: &KeyEvent) -> bool {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('\\') if ctrl => true,
        KeyCode::Char('\u{1c}') => true, // some terminals encode Ctrl+\ as FS
        _ => false,
    }
}

pub(crate) fn is_open(app: &App) -> bool {
    app.dashboard.as_ref().is_some_and(|d| d.visible)
}

pub(crate) fn hide(app: &mut App) {
    if let Some(overlay) = app.dashboard.as_mut() {
        overlay.visible = false;
        overlay.help = false;
    }
}

pub(crate) fn open_from_harness(app: &mut App, harness: &Harness) -> Result<(), String> {
    app.api_key = harness.api_key().to_string();
    app.pipe_base_url = harness.base_url().to_string();
    app.model = harness.model();
    app.workspace_root = harness.workspace_root().to_path_buf();
    show_from_app(app)
}

pub(crate) fn toggle_from_app(app: &mut App) -> Result<(), String> {
    if is_open(app) {
        hide(app);
        Ok(())
    } else {
        show_from_app(app)
    }
}

pub(crate) fn open_from_app(app: &mut App) -> Result<(), String> {
    show_from_app(app)
}

pub(crate) fn show_from_app(app: &mut App) -> Result<(), String> {
    if let Some(overlay) = app.dashboard.as_mut() {
        overlay.visible = true;
        overlay.runtime.update_session(
            app.api_key.clone(),
            app.pipe_base_url.clone(),
            app.model.clone(),
        );
        return Ok(());
    }
    let mut knobs = DashboardKnobs::for_workspace(
        app.workspace_root.clone(),
        app.api_key.clone(),
        app.pipe_base_url.clone(),
        app.model.clone(),
    );
    knobs.max_working = knobs.max_working();
    knobs.openai_api_key = app.openai_api_key.clone();
    knobs.openai_base_url = app.openai_base_url.clone();
    match Dashboard::open(knobs) {
        Ok(runtime) => {
            app.dashboard = Some(DashboardOverlay::new(runtime));
            Ok(())
        }
        Err(err) => Err(format!("dashboard: {err:#}")),
    }
}

pub(crate) fn apply_action(overlay: &mut DashboardOverlay, action: DashAction) {
    match action {
        DashAction::None | DashAction::Close => {}
        DashAction::ToggleWorktree => {
            if !hi_tools::worktree::in_git_repo(&overlay.runtime.knobs().workspace_root) {
                overlay.worktree_next = false;
                overlay.toast("not a git repository — worktree dispatch is disabled");
            } else {
                overlay.worktree_next = !overlay.worktree_next;
                overlay.toast(if overlay.worktree_next {
                    "next dispatch uses a git worktree"
                } else {
                    "next dispatch uses the shared workspace"
                });
            }
        }
        DashAction::CycleModel => {
            overlay.gpt6_next = !overlay.gpt6_next;
            overlay.toast(if overlay.gpt6_next {
                "next sub-agent: pipe/gpt-6 (Pipe)"
            } else {
                "next sub-agent: same model as the manager (this session)"
            });
        }
        DashAction::SpawnIdle => match overlay.runtime.spawn_row(
            None,
            DispatchOpts {
                model: overlay.next_model(),
                worktree: overlay.worktree_next,
            },
        ) {
            Ok(_) => {
                overlay.worktree_next = false;
                // Stay on + New sub-agent so the next prompt dispatches, not replies.
                overlay.selected = None;
            }
            Err(err) => overlay.toast(err.to_string()),
        },
        DashAction::Dispatch { attach } => {
            let prompt = overlay.draft.text();
            overlay.draft = InputLine::default();
            match overlay.runtime.dispatch_with(
                &prompt,
                DispatchOpts {
                    model: overlay.next_model(),
                    worktree: overlay.worktree_next,
                },
            ) {
                Ok(id) => {
                    overlay.worktree_next = false;
                    if attach {
                        overlay.attached = Some(id);
                    } else {
                        overlay.selected = None;
                    }
                }
                Err(err) => overlay.toast(err.to_string()),
            }
        }
        DashAction::Reply { attach } => {
            let prompt = overlay.draft.text();
            overlay.draft = InputLine::default();
            let id = overlay.attached.clone().or_else(|| overlay.selected_id());
            match id {
                Some(id) => match overlay.runtime.reply(&id, &prompt) {
                    Ok(()) => {
                        if attach {
                            overlay.attached = Some(id);
                        }
                    }
                    Err(err) => overlay.toast(err.to_string()),
                },
                None => overlay.toast("no agent selected"),
            }
        }
        DashAction::Attach => {
            if let Some(id) = overlay.selected_id() {
                overlay.attached = Some(id);
                overlay.focus = Focus::Input;
            }
        }
        DashAction::Cancel => {
            if let Some(id) = overlay.attached.clone().or_else(|| overlay.selected_id()) {
                match overlay.runtime.cancel(&id) {
                    Ok(()) => overlay.toast("cancelled"),
                    Err(err) => overlay.toast(err.to_string()),
                }
            }
        }
        DashAction::Delete => {
            if let Some(id) = overlay.selected_id() {
                if overlay.attached.as_deref() == Some(id.as_str()) {
                    overlay.attached = None;
                }
                match overlay.runtime.remove(&id) {
                    Ok(()) => {
                        overlay.selected = None;
                        overlay.toast("removed");
                    }
                    Err(err) => overlay.toast(err.to_string()),
                }
            }
        }
    }
}

impl DashboardOverlay {
    fn next_model(&self) -> Option<String> {
        if self.gpt6_next {
            Some(PIPE_GPT6.into())
        } else {
            None
        }
    }
}

pub(crate) fn render(
    frame: &mut ratatui::Frame,
    area: Rect,
    overlay: &mut DashboardOverlay,
    spinner: usize,
) {
    let th = crate::theme::theme();
    chrome::fill_background(frame, area, &th);
    frame.render_widget(Clear, area);

    if overlay.help {
        render_help(frame, area);
        return;
    }
    if overlay.attached.is_some() {
        render_attached(frame, area, overlay, spinner);
        return;
    }

    let toast = overlay
        .toast
        .as_ref()
        .filter(|(_, at)| at.elapsed() < TOAST_TTL)
        .map(|(m, _)| m.as_str());
    let footer_h = if toast.is_some() { 2 } else { 1 };
    let chunks = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(4),
        Constraint::Length(4),
        Constraint::Length(footer_h),
    ])
    .split(chrome::inset(area, 2, 3, 1, 1));

    let roster = overlay.runtime.roster();
    let cwd = display_cwd(&overlay.runtime.knobs().workspace_root, 40);
    let branch = chrome::git_branch(&overlay.runtime.knobs().workspace_root)
        .unwrap_or_else(|| "no-git".into());
    let working = roster
        .iter()
        .filter(|r| r.state == RowState::Working)
        .count();
    let idle = roster.iter().filter(|r| r.state == RowState::Idle).count();
    let failed = roster
        .iter()
        .filter(|r| r.state == RowState::Failed)
        .count();
    let manager_model = overlay.runtime.knobs().model.as_str();
    let sub_model = if overlay.gpt6_next {
        PIPE_GPT6
    } else {
        manager_model
    };
    let new_sel = overlay.selected.is_none();
    let new_label = if overlay.worktree_next {
        "+ New sub-agent in worktree"
    } else {
        "+ New sub-agent"
    };
    let identity = vec![
        Line::from(vec![
            Span::styled(
                format!("{}  ", cell("role", 10)),
                Style::default().fg(th.gray_dim),
            ),
            Span::styled(
                format!("{}  ", cell("model", 32)),
                Style::default().fg(th.gray_dim),
            ),
            Span::styled(
                format!("{}  {branch} {cwd}", cell("where", 22)),
                Style::default().fg(th.gray_dim),
            ),
            Span::raw("  "),
            Span::styled(
                format!("◆ {working} working"),
                Style::default().fg(th.accent_running),
            ),
            Span::raw("  "),
            Span::styled(format!("○ {idle} idle"), Style::default().fg(th.gray)),
            Span::raw("  "),
            Span::styled(
                format!("● {failed} failed"),
                Style::default().fg(th.accent_error),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                format!("{}  ", cell("manager", 10)),
                Style::default()
                    .fg(th.accent_system)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{}  ", cell(manager_model, 32)),
                Style::default()
                    .fg(th.accent_model)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                cell("this session · Esc", 22),
                Style::default().fg(th.text_primary),
            ),
        ]),
        Line::from(vec![
            Span::styled(
                format!("{}  ", cell("sub-agent", 10)),
                Style::default()
                    .fg(th.accent_model)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("{}  ", cell(sub_model, 32)),
                Style::default()
                    .fg(th.accent_model)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                cell(
                    if overlay.worktree_next {
                        "next dispatch · Ctrl+M · wt"
                    } else {
                        "next dispatch · Ctrl+M"
                    },
                    28,
                ),
                Style::default().fg(th.text_primary),
            ),
            Span::raw("  "),
            Span::styled(
                new_label,
                if new_sel {
                    Style::default()
                        .fg(th.text_primary)
                        .bg(th.selection_bg)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(th.accent_system)
                },
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(identity), chunks[0]);

    let mut rows: Vec<Line> = Vec::new();
    rows.push(roster_header());
    if roster.is_empty() {
        rows.push(Line::styled(
            "  no sub-agents yet — you are the manager. Enter dispatches one with the model above.",
            Style::default().fg(th.text_secondary),
        ));
        rows.push(Line::styled(
            "  Ctrl+M switches that model. Rows below will list each configured sub-agent.",
            Style::default().fg(th.gray_dim),
        ));
    }
    for (i, row) in roster.iter().enumerate() {
        rows.push(row_line(
            i,
            row,
            overlay.selected.as_deref() == Some(row.id.as_str()),
            spinner,
        ));
    }
    frame.render_widget(Paragraph::new(rows), chunks[1]);

    let peek = overlay
        .selected
        .as_ref()
        .and_then(|id| roster.iter().find(|r| &r.id == id).cloned());
    render_input_box(frame, chunks[2], overlay, peek.as_ref());

    let footer = chunks[3];
    let (toast_area, shortcuts_area) = if toast.is_some() && footer.height >= 2 {
        let split = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(footer);
        (Some(split[0]), split[1])
    } else {
        (None, footer)
    };
    if let (Some(area), Some(msg)) = (toast_area, toast) {
        frame.render_widget(
            Paragraph::new(Line::styled(
                msg.to_string(),
                Style::default().fg(th.warning),
            )),
            area,
        );
    }
    let cancel_armed = overlay.cancel_armed.as_ref().is_some_and(|(id, at)| {
        at.elapsed() < CANCEL_DELETE_WINDOW && overlay.selected_id().as_deref() == Some(id.as_str())
    });
    chrome::render_shortcuts_bar(
        frame,
        shortcuts_area,
        &roster_hints(peek.is_some(), overlay.draft.is_empty(), cancel_armed),
        &th,
    );
}

fn roster_header() -> Line<'static> {
    let th = crate::theme::theme();
    let dim = Style::default().fg(th.gray_dim);
    Line::from(vec![
        Span::styled(format!(" {} ", cell("#", 2)), dim),
        Span::styled(format!("{}  ", cell("role", 10)), dim),
        Span::styled(format!("{}  ", cell("model", 32)), dim),
        Span::styled(format!("{}  ", cell("state", 10)), dim),
        Span::styled(cell("task", 28), dim),
    ])
}

fn row_line(index: usize, row: &RowView, selected: bool, spinner: usize) -> Line<'static> {
    let th = crate::theme::theme();
    let (glyph, state_label, state_color) = match row.state {
        RowState::Working => (
            SPINNER[spinner % SPINNER.len()],
            "working",
            th.accent_running,
        ),
        RowState::Idle => ("○", "idle", th.gray),
        RowState::Completed => ("●", "done", th.accent_success),
        RowState::Failed => ("●", "failed", th.accent_error),
    };
    let task = if row.last_text.trim().is_empty() {
        row.title.as_str()
    } else {
        row.last_text.as_str()
    };
    let wt = if row.is_worktree { " wt" } else { "" };
    let mark = if selected { "▌" } else { " " };
    let num = if index < 9 {
        format!("{}", index + 1)
    } else {
        "·".into()
    };
    let selected_bg = if selected {
        Style::default().bg(th.selection_bg)
    } else {
        Style::default()
    };
    Line::from(vec![
        Span::styled(
            format!("{mark}{} ", cell(&format!("{num}{glyph}"), 3)),
            Style::default().fg(state_color).patch(selected_bg),
        ),
        Span::styled(
            format!("{}  ", cell("sub-agent", 10)),
            Style::default().fg(th.text_secondary).patch(selected_bg),
        ),
        Span::styled(
            format!("{}  ", cell(&row.model, 32)),
            Style::default()
                .fg(th.accent_model)
                .add_modifier(Modifier::BOLD)
                .patch(selected_bg),
        ),
        Span::styled(
            format!("{}  ", cell(state_label, 10)),
            Style::default().fg(state_color).patch(selected_bg),
        ),
        Span::styled(
            format!("{}{wt}", cell(task, 28)),
            Style::default().fg(th.text_primary).patch(selected_bg),
        ),
    ])
}

fn render_input_box(
    frame: &mut ratatui::Frame,
    area: Rect,
    overlay: &DashboardOverlay,
    peek: Option<&RowView>,
) {
    let th = crate::theme::theme();
    let focused = overlay.focus == Focus::Input;
    let title = if let Some(row) = peek {
        format!(" reply to sub-agent · {} ", clip(&row.title, 32))
    } else {
        " dispatch a sub-agent ".into()
    };
    let bottom = if peek.is_some() {
        if overlay.draft.is_empty() {
            " empty Enter attaches · typed Enter replies (queued if busy) "
        } else {
            " Enter replies · Ctrl+S replies and opens "
        }
    } else if overlay.draft.is_empty() {
        " you are the manager · type a prompt · Enter starts a sub-agent "
    } else {
        " Enter dispatches a sub-agent · Ctrl+S dispatches and opens "
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(th.input_border(focused))
        .title(title)
        .title_bottom(Line::styled(bottom, Style::default().fg(th.gray_dim)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let prefix = if peek.is_some() { "❯ reply " } else { "❯ " };
    let text = format!("{prefix}{}", overlay.draft.text());
    frame.render_widget(
        Paragraph::new(text).style(Style::default().fg(th.text_primary)),
        inner,
    );
    if overlay.focus == Focus::Input && inner.width > 0 && inner.height > 0 {
        let before: String = overlay
            .draft
            .chars
            .iter()
            .take(overlay.draft.cursor())
            .collect();
        let col = (display_width(prefix) + display_width(&before)) as u16;
        frame.set_cursor_position((
            inner
                .x
                .saturating_add(col.min(inner.width.saturating_sub(1))),
            inner.y,
        ));
    }
}

fn render_attached(
    frame: &mut ratatui::Frame,
    area: Rect,
    overlay: &DashboardOverlay,
    spinner: usize,
) {
    let th = crate::theme::theme();
    let id = overlay.attached.as_deref().unwrap_or("");
    let row = overlay.runtime.roster().into_iter().find(|r| r.id == id);
    let chunks = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(4),
        Constraint::Length(1),
    ])
    .split(chrome::inset(area, 2, 3, 1, 1));
    let header = match &row {
        Some(r) => format!(
            "sub-agent · {} │ {}  [{}]   [manager: dashboard]",
            r.title,
            r.model,
            match r.state {
                RowState::Working => SPINNER[spinner % SPINNER.len()],
                RowState::Idle => "idle",
                RowState::Completed => "done",
                RowState::Failed => "failed",
            }
        ),
        None => "agent closed".into(),
    };
    frame.render_widget(
        Paragraph::new(Line::styled(
            header,
            Style::default()
                .fg(th.text_primary)
                .add_modifier(Modifier::BOLD),
        )),
        chunks[0],
    );
    let body = row
        .as_ref()
        .map(|r| {
            if let Some(err) = &r.error {
                format!("{err}\n{}", r.last_text)
            } else {
                r.last_text.clone()
            }
        })
        .unwrap_or_default();
    frame.render_widget(
        Paragraph::new(body)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(th.text_secondary)),
        chunks[1],
    );
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(th.input_border(true))
        .title(" reply ")
        .title_bottom(Line::styled(
            " Enter sends · empty Esc / Ctrl+\\ returns to the dashboard ",
            Style::default().fg(th.gray_dim),
        ));
    let inner = block.inner(chunks[2]);
    frame.render_widget(block, chunks[2]);
    frame.render_widget(Paragraph::new(format!("❯ {}", overlay.draft.text())), inner);
    if inner.width > 0 && inner.height > 0 {
        let before: String = overlay
            .draft
            .chars
            .iter()
            .take(overlay.draft.cursor())
            .collect();
        let col = (display_width("❯ ") + display_width(&before)) as u16;
        frame.set_cursor_position((
            inner
                .x
                .saturating_add(col.min(inner.width.saturating_sub(1))),
            inner.y,
        ));
    }
    chrome::render_shortcuts_bar(frame, chunks[3], &attached_hints(), &th);
}

fn render_help(frame: &mut ratatui::Frame, area: Rect) {
    let th = crate::theme::theme();
    let panel = chrome::inset(area, 4, 6, 2, 2);
    let block = th.panel_block(" agent dashboard ", UiTone::Info);
    let inner = block.inner(panel);
    frame.render_widget(Clear, area);
    frame.render_widget(block, panel);
    let body_shortcuts = Layout::vertical([Constraint::Min(4), Constraint::Length(1)]).split(inner);
    let lines = [
        "Manager vs sub-agent",
        "  Manager   — you. This dashboard, and the hi session behind Esc.",
        "              Change the manager model with /model in that session.",
        "  Sub-agent — each row. The dispatch box always creates a new one.",
        "              There is no parent picker; rows do not call each other.",
        "  You manage them by selecting a row and typing a reply (queued if busy).",
        "",
        "Set the next sub-agent's model with Ctrl+M (Pipe gpt-6, or the manager's",
        "model). Prefix a prompt with /model pipe/gpt-6 … to set it for one dispatch.",
        "",
        "Keys",
        "  Enter          dispatch a sub-agent, or reply to the selected row",
        "  empty Enter    attach to the selected sub-agent",
        "  Ctrl+S         dispatch/reply and attach",
        "  Ctrl+W         next sub-agent uses a git worktree (no auto-merge)",
        "  Ctrl+M         next sub-agent model: pipe/gpt-6 ↔ manager model",
        "  Ctrl+X         cancel the selected turn; twice in 2s deletes the row",
        "  Tab            list ↔ dispatch/peek input",
        "  Ctrl+\\         close, or return from attach",
        "  Esc            clear draft → unselect → close",
        "",
        "gpt-6 uses Pipe Network. An openai profile in config is optional",
        "and only used for models prefixed openai/.",
    ];
    frame.render_widget(
        Paragraph::new(lines.iter().map(|l| Line::raw(*l)).collect::<Vec<_>>())
            .style(Style::default().fg(th.text_primary)),
        body_shortcuts[0],
    );
    chrome::render_shortcuts_bar(frame, body_shortcuts[1], &help_hints(), &th);
}

fn clip(s: &str, max: usize) -> String {
    crate::util::clip(s, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn ui(n: usize) -> DashUi {
        DashUi {
            focus: if n == 0 { Focus::Input } else { Focus::List },
            selected: None,
            row_count: n,
            draft: String::new(),
            attached: false,
        }
    }

    /// Minimal table for key-dispatch tests (no live Dashboard).
    struct DashUi {
        focus: Focus,
        selected: Option<usize>,
        row_count: usize,
        draft: String,
        attached: bool,
    }

    fn apply(ui: &mut DashUi, key: &KeyEvent) -> DashAction {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if is_dashboard_toggle(key) {
            return if ui.attached {
                ui.attached = false;
                DashAction::None
            } else {
                DashAction::Close
            };
        }
        if matches!(key.code, KeyCode::Char('w')) && ctrl {
            return DashAction::ToggleWorktree;
        }
        if matches!(key.code, KeyCode::Char('m')) && ctrl {
            return DashAction::CycleModel;
        }
        if matches!(key.code, KeyCode::Char('x')) && ctrl {
            return DashAction::Cancel;
        }
        if matches!(key.code, KeyCode::Char('s')) && ctrl {
            if ui.selected.is_some() {
                return if ui.draft.is_empty() {
                    DashAction::Attach
                } else {
                    DashAction::Reply { attach: true }
                };
            }
            return if ui.draft.is_empty() {
                DashAction::SpawnIdle
            } else {
                DashAction::Dispatch { attach: true }
            };
        }
        match key.code {
            KeyCode::Tab => {
                ui.focus = match ui.focus {
                    Focus::List => Focus::Input,
                    Focus::Input => Focus::List,
                };
                DashAction::None
            }
            KeyCode::Enter => {
                if ui.selected.is_some() {
                    if ui.draft.is_empty() {
                        DashAction::Attach
                    } else {
                        DashAction::Reply { attach: false }
                    }
                } else if ui.draft.is_empty() {
                    DashAction::SpawnIdle
                } else {
                    DashAction::Dispatch { attach: false }
                }
            }
            KeyCode::Down => {
                if ui.row_count > 0 {
                    ui.selected = Some(0);
                }
                DashAction::None
            }
            KeyCode::Char(c) if !ctrl => {
                ui.focus = Focus::Input;
                ui.draft.push(c);
                DashAction::None
            }
            KeyCode::Esc => {
                if !ui.draft.is_empty() {
                    ui.draft.clear();
                    DashAction::None
                } else if ui.selected.is_some() {
                    ui.selected = None;
                    DashAction::None
                } else {
                    DashAction::Close
                }
            }
            _ => DashAction::None,
        }
    }

    #[test]
    fn dispatch_enter_creates_a_member_action() {
        let mut u = ui(0);
        assert_eq!(apply(&mut u, &key(KeyCode::Char('f'))), DashAction::None);
        assert_eq!(
            apply(&mut u, &key(KeyCode::Enter)),
            DashAction::Dispatch { attach: false }
        );
    }

    #[test]
    fn peek_reply_enqueues_on_selected_row() {
        let mut u = ui(2);
        apply(&mut u, &key(KeyCode::Down));
        assert_eq!(u.selected, Some(0));
        apply(&mut u, &key(KeyCode::Char('h')));
        apply(&mut u, &key(KeyCode::Char('i')));
        assert_eq!(
            apply(&mut u, &key(KeyCode::Enter)),
            DashAction::Reply { attach: false }
        );
    }

    #[test]
    fn empty_enter_on_row_attaches() {
        let mut u = ui(1);
        apply(&mut u, &key(KeyCode::Down));
        assert_eq!(apply(&mut u, &key(KeyCode::Enter)), DashAction::Attach);
    }

    #[test]
    fn chords_match_grok_table() {
        let mut u = ui(1);
        assert_eq!(apply(&mut u, &ctrl('w')), DashAction::ToggleWorktree);
        assert_eq!(apply(&mut u, &ctrl('m')), DashAction::CycleModel);
        apply(&mut u, &key(KeyCode::Down));
        assert_eq!(apply(&mut u, &ctrl('x')), DashAction::Cancel);
        assert_eq!(apply(&mut u, &ctrl('\\')), DashAction::Close);
        assert_eq!(apply(&mut u, &key(KeyCode::Tab)), DashAction::None);
        assert_eq!(u.focus, Focus::Input);
    }

    #[test]
    fn aliases_are_not_removed() {
        assert_eq!(
            hi_harness::parse_command("/dashboard"),
            Some(hi_harness::Command::Dashboard)
        );
        assert_eq!(
            hi_harness::parse_command("/fleet"),
            Some(hi_harness::Command::Dashboard)
        );
        assert_eq!(
            hi_harness::parse_command("/agents-dashboard"),
            Some(hi_harness::Command::Dashboard)
        );
    }

    fn hint_keys(hints: &[ShortcutHint]) -> Vec<&str> {
        hints.iter().map(|h| h.key).collect()
    }

    #[test]
    fn shortcut_bar_follows_the_screen() {
        let dispatch = roster_hints(false, true, false);
        assert_eq!(dispatch[0].key, "enter");
        assert_eq!(dispatch[0].label, "sub-agent");
        assert!(hint_keys(&dispatch).contains(&"ctrl+m"));
        assert!(hint_keys(&dispatch).contains(&"ctrl+w"));
        assert!(hint_keys(&dispatch).contains(&"?"));

        let peek = roster_hints(true, true, false);
        assert_eq!(peek[0].label, "attach");
        assert!(hint_keys(&peek).contains(&"ctrl+x"));

        let reply = roster_hints(true, false, false);
        assert_eq!(reply[0].label, "reply");

        let armed = roster_hints(true, true, true);
        assert_eq!(armed[0].key, "ctrl+x");
        assert_eq!(armed[0].label, "delete");

        let attached = attached_hints();
        assert!(hint_keys(&attached).contains(&"enter"));
        assert!(hint_keys(&attached).contains(&"esc"));
        assert!(hint_keys(&attached).contains(&"ctrl+\\"));
    }

    #[test]
    fn roster_render_paints_shortcuts_and_first_run_copy() {
        let dir = tempfile::tempdir().unwrap();
        let knobs = hi_harness::DashboardKnobs {
            api_key: "pk_test".into(),
            base_url: "http://127.0.0.1:9".into(),
            model: "pipe/test".into(),
            workspace_root: dir.path().to_path_buf(),
            sessions_dir: dir.path().join("sessions"),
            store_path: dir.path().join("dashboard").join("workspace.db"),
            max_working: 8,
            openai_api_key: None,
            openai_base_url: None,
        };
        let mut overlay = DashboardOverlay::new(hi_harness::Dashboard::open(knobs).unwrap());
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 28)).unwrap();
        term.draw(|f| render(f, f.area(), &mut overlay, 0)).unwrap();
        let screen = crate::tests::dump(&term);
        assert!(
            screen.contains("no sub-agents yet") && screen.contains("sub-agent"),
            "first-run copy:\n{screen}"
        );
        assert!(
            screen.contains("enter:sub-agent") || screen.contains("enter:dispatch"),
            "dispatch shortcuts:\n{screen}"
        );
        assert!(
            screen.contains("ctrl+m:sub model") && screen.contains("?:help"),
            "model and help on the bar:\n{screen}"
        );
        assert!(
            screen.contains("manager") && screen.contains("this session"),
            "manager chip:\n{screen}"
        );
    }

    #[test]
    fn roster_lists_each_configured_sub_agent() {
        let dir = tempfile::tempdir().unwrap();
        let knobs = hi_harness::DashboardKnobs {
            api_key: "pk_test".into(),
            base_url: "http://127.0.0.1:9".into(),
            model: "pipe/manager-model".into(),
            workspace_root: dir.path().to_path_buf(),
            sessions_dir: dir.path().join("sessions"),
            store_path: dir.path().join("dashboard").join("workspace.db"),
            max_working: 8,
            openai_api_key: None,
            openai_base_url: None,
        };
        let mut overlay = DashboardOverlay::new(hi_harness::Dashboard::open(knobs).unwrap());
        overlay
            .runtime
            .dispatch("fix login", Some("pipe/gpt-6".into()))
            .unwrap();
        overlay
            .runtime
            .dispatch("review tests", Some("pipe/deepseek-v4-flash-0731".into()))
            .unwrap();
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 28)).unwrap();
        term.draw(|f| render(f, f.area(), &mut overlay, 0)).unwrap();
        let screen = crate::tests::dump(&term);
        assert!(
            screen.contains("pipe/manager-model") && screen.contains("this session"),
            "manager model on the configured table:\n{screen}"
        );
        assert!(
            screen.contains("pipe/gpt-6") && screen.contains("pipe/deepseek-v4-flash-0731"),
            "each sub-agent model must be visible:\n{screen}"
        );
        let sub_agent_rows = screen.matches("sub-agent").count();
        assert!(
            sub_agent_rows >= 3,
            "header slot + two live rows should say sub-agent:\n{screen}"
        );
        assert!(
            screen.contains("role") && screen.contains("model") && screen.contains("state"),
            "column headers:\n{screen}"
        );
    }

    fn test_overlay() -> (tempfile::TempDir, DashboardOverlay) {
        let dir = tempfile::tempdir().unwrap();
        let knobs = hi_harness::DashboardKnobs {
            api_key: "pk_test".into(),
            base_url: "http://127.0.0.1:9".into(),
            model: "pipe/test".into(),
            workspace_root: dir.path().to_path_buf(),
            sessions_dir: dir.path().join("sessions"),
            store_path: dir.path().join("dashboard").join("workspace.db"),
            max_working: 8,
            openai_api_key: None,
            openai_base_url: None,
        };
        let overlay = DashboardOverlay::new(hi_harness::Dashboard::open(knobs).unwrap());
        (dir, overlay)
    }

    #[test]
    fn dispatch_keeps_the_box_on_new_sub_agent() {
        let (_dir, mut overlay) = test_overlay();
        overlay.draft.insert_str("fix login");
        apply_action(&mut overlay, DashAction::Dispatch { attach: false });
        assert!(
            overlay.selected.is_none(),
            "next Enter must dispatch another sub-agent, not reply"
        );
        assert_eq!(overlay.runtime.roster().len(), 1);
    }

    #[test]
    fn typing_jk_in_the_dispatch_box_is_not_navigation() {
        let dir = tempfile::tempdir().unwrap();
        let knobs = hi_harness::DashboardKnobs {
            api_key: "pk_test".into(),
            base_url: "http://127.0.0.1:9".into(),
            model: "pipe/test".into(),
            workspace_root: dir.path().to_path_buf(),
            sessions_dir: dir.path().join("sessions"),
            store_path: dir.path().join("dashboard").join("workspace.db"),
            max_working: 8,
            openai_api_key: None,
            openai_base_url: None,
        };
        let mut overlay = DashboardOverlay::new(hi_harness::Dashboard::open(knobs).unwrap());
        overlay.focus = Focus::Input;
        overlay.handle_key(&key(KeyCode::Char('k')));
        overlay.handle_key(&key(KeyCode::Char('j')));
        assert_eq!(overlay.draft.text(), "kj");
        assert!(overlay.selected.is_none());
    }

    #[test]
    fn idle_session_bar_advertises_dashboard() {
        let mut app = crate::tests::test_app("pipenetwork", "pipe/test");
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 24)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = crate::tests::dump(&term);
        assert!(
            screen.contains("ctrl+\\:dashboard") || screen.contains("ctrl+\\:dashboard"),
            "session bar should advertise the dashboard:\n{screen}"
        );
    }
}
