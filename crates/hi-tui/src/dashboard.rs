//! `/dashboard` — control a fleet, not an agent.
//!
//! A full-screen mode over the same terminal: a table of dispatched agents
//! (one row each), a dispatch box that always spawns a *new* session on Enter,
//! and a peek panel for the selected row — its latest output plus a live reply
//! input, so you can answer an agent's question with a single keystroke
//! (`1`–`9`) or queue a follow-up without opening the full conversation.
//! `Ctrl+S` dispatches *and* attaches (a full-screen focus view of that row).
//!
//! Concurrency: each dispatched turn is a future that *owns* its `Agent` and
//! hands it back on completion (`FuturesUnordered` driven by this loop's
//! `select!`) — no spawn, no locks, same single-threaded model as the rest of
//! the TUI. Every fleet agent writes its own session file, so anything you
//! dispatch here is individually resumable later with `--resume`.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use futures_util::stream::FuturesUnordered;
use hi_agent::Agent;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};
use tokio::sync::mpsc;

use crate::event::{ChannelUi, UiEvent};
use crate::input::InputLine;
use crate::render::dim;
use crate::{App, FleetSpawner, SPINNER};

/// Lines of output kept per row for the peek/attach panels.
const TAIL_CAP: usize = 200;
/// Max table rows shown before the list scrolls with the selection.
const TABLE_ROWS: usize = 8;

/// A dispatched fleet agent: one row in the dashboard, one session on disk.
pub(crate) struct FleetRow {
    /// Display id (stable, 1-based, never reused within a session).
    pub(crate) id: usize,
    /// The dispatch prompt (shown truncated as the row title).
    pub(crate) title: String,
    /// The agent, present while idle; `None` while a turn future owns it.
    pub(crate) agent: Option<Agent>,
    /// Interrupt handle for the in-flight turn (Esc = skip the current tool).
    pub(crate) interrupt: Option<Arc<AtomicBool>>,
    pub(crate) state: RowState,
    /// Live activity lead while working ("running bash", "responding…").
    pub(crate) activity: String,
    /// Recent plain-text output lines (peek/attach panel body).
    pub(crate) tail: Vec<String>,
    /// Unterminated last streamed-text line (flushed on newline/turn end).
    partial: String,
    /// Follow-ups typed while a turn was running; dispatched FIFO on idle.
    pub(crate) pending: VecDeque<String>,
    /// The per-row reply input (peek panel).
    pub(crate) reply: InputLine,
    /// Cumulative (input, output) tokens for this agent.
    pub(crate) usage: (u64, u64),
    /// Current turn start (for the elapsed column).
    pub(crate) started: Option<Instant>,
    pub(crate) turns: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowState {
    Working,
    Idle,
    Failed,
}

impl FleetRow {
    fn new(id: usize, title: String, agent: Agent) -> Self {
        Self {
            id,
            title,
            agent: Some(agent),
            interrupt: None,
            state: RowState::Idle,
            activity: String::new(),
            tail: Vec::new(),
            partial: String::new(),
            pending: VecDeque::new(),
            reply: InputLine::default(),
            usage: (0, 0),
            started: None,
            turns: 0,
        }
    }

    fn push_line(&mut self, line: String) {
        self.tail.push(line);
        if self.tail.len() > TAIL_CAP {
            let drop = self.tail.len() - TAIL_CAP;
            self.tail.drain(..drop);
        }
    }

    /// Append streamed assistant text, splitting into tail lines on newlines.
    fn push_text(&mut self, text: &str) {
        self.partial.push_str(text);
        while let Some(nl) = self.partial.find('\n') {
            let line: String = self.partial.drain(..=nl).collect();
            let line = line.trim_end().to_string();
            if !line.is_empty() {
                self.push_line(line);
            }
        }
    }

    fn flush_partial(&mut self) {
        if !self.partial.trim().is_empty() {
            let line = std::mem::take(&mut self.partial).trim_end().to_string();
            self.push_line(line);
        } else {
            self.partial.clear();
        }
    }

    /// Route one turn event into the row's state (a miniature `App::apply`).
    fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Text(t) => {
                self.activity = "responding…".to_string();
                self.push_text(&t);
            }
            UiEvent::Reasoning(_) => self.activity = "thinking…".to_string(),
            UiEvent::AssistantEnd => self.flush_partial(),
            UiEvent::ToolStarted(name, args) => {
                self.activity = format!("running {}", hi_agent::ui::tool_label(&name, &args));
            }
            UiEvent::ToolCall(name, args) => {
                self.push_line(format!("⚙ {}", hi_agent::ui::tool_label(&name, &args)));
            }
            UiEvent::Status(s) => self.push_line(s),
            UiEvent::Usage { input, output, .. } => self.usage = (input, output),
            UiEvent::TurnEnd(summary) => {
                self.flush_partial();
                if !summary.trim().is_empty() {
                    self.push_line(format!("✓ {summary}"));
                }
            }
            UiEvent::TurnError(kind, message, _guidance) => {
                self.flush_partial();
                self.push_line(format!("✗ {kind}: {message}"));
            }
            UiEvent::ChangedFiles(files) => {
                if !files.is_empty() {
                    self.push_line(format!("changed: {}", files.join(", ")));
                }
            }
            // Tool results/streams, plans, and rate limits are noise at fleet
            // altitude — the attach view is still just the tail; open the
            // session with --resume for the full conversation.
            UiEvent::ToolResult(..)
            | UiEvent::ToolStream(..)
            | UiEvent::Plan(_)
            | UiEvent::RateLimits(_) => {}
        }
    }
}

/// Which input owns keystrokes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    /// The bottom dispatch box (default) — Enter spawns a new agent.
    Dispatch,
    /// The selected row's reply input (peek panel).
    Reply,
    /// Full-screen view of the selected row (bigger tail + reply input).
    Attach,
}

/// A completed turn hands its agent back: (row index, agent, result).
type TurnDone = (usize, Agent, Result<()>);
type TurnFut = Pin<Box<dyn Future<Output = TurnDone>>>;

/// Run the fleet dashboard until the user leaves it. Rows persist on
/// `app.fleet` across open/close; in-flight turns must finish or be abandoned
/// (double-Esc) before leaving, since their futures live in this loop.
pub(crate) async fn run_dashboard(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    input_rx: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    spawner: &FleetSpawner,
) -> Result<()> {
    let (tagged_tx, mut events_rx) = mpsc::unbounded_channel::<(usize, UiEvent)>();
    let mut in_flight: FuturesUnordered<TurnFut> = FuturesUnordered::new();
    let mut selected: usize = app.fleet.len().saturating_sub(1);
    let mut focus = Focus::Dispatch;
    let mut dispatch = InputLine::default();
    let mut exit_armed = false;
    let mut flash: Option<String> = None;

    loop {
        terminal.draw(|f| {
            render_dashboard(
                f,
                app,
                selected,
                focus,
                &dispatch,
                in_flight.len(),
                exit_armed,
                flash.as_deref(),
            )
        })?;

        tokio::select! {
            // A turn finished — take the agent back and run any queued follow-up.
            Some((idx, agent, result)) = in_flight.next(), if !in_flight.is_empty() => {
                if let Some(row) = app.fleet.get_mut(idx) {
                    row.agent = Some(agent);
                    row.interrupt = None;
                    row.turns += 1;
                    row.started = None;
                    row.flush_partial();
                    row.activity.clear();
                    match result {
                        Ok(()) => row.state = RowState::Idle,
                        Err(err) => {
                            row.state = RowState::Failed;
                            row.push_line(format!("✗ turn failed: {err:#}"));
                        }
                    }
                    if let Some(next) = row.pending.pop_front() {
                        start_turn(app, idx, next, &tagged_tx, &mut in_flight);
                    }
                }
            }
            Some((idx, event)) = events_rx.recv() => {
                if let Some(row) = app.fleet.get_mut(idx) {
                    row.apply(event);
                }
                // Drain whatever else already arrived so a chatty turn can't
                // starve the render loop one-event-per-frame.
                while let Ok((idx, event)) = events_rx.try_recv() {
                    if let Some(row) = app.fleet.get_mut(idx) {
                        row.apply(event);
                    }
                }
            }
            _ = ticker.tick() => {
                app.spinner = app.spinner.wrapping_add(1);
            }
            maybe = input_rx.recv() => {
                let Some(event) = maybe else { return Ok(()) };
                match event {
                    Event::Paste(text) => {
                        flash = None;
                        focused_input(app, selected, focus, &mut dispatch)
                            .map(|input| input.insert_str(&text));
                    }
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        flash = None;
                        // Anything but a second Esc disarms the exit warning.
                        if !matches!(key.code, KeyCode::Esc) {
                            exit_armed = false;
                        }
                        match key.code {
                            KeyCode::Esc | KeyCode::Char('c') if matches!(key.code, KeyCode::Esc) || ctrl => {
                                if focus != Focus::Dispatch {
                                    focus = Focus::Dispatch;
                                    exit_armed = false;
                                    continue;
                                }
                                if in_flight.is_empty() {
                                    return Ok(());
                                }
                                if exit_armed {
                                    // Abandon in-flight turns: their futures drop
                                    // with this loop; the sessions on disk stay
                                    // resumable.
                                    for row in app.fleet.iter_mut() {
                                        if row.state == RowState::Working {
                                            row.state = RowState::Failed;
                                            row.started = None;
                                            row.activity.clear();
                                            row.push_line(
                                                "⚠ abandoned mid-turn — session remains resumable"
                                                    .to_string(),
                                            );
                                        }
                                    }
                                    return Ok(());
                                }
                                exit_armed = true;
                            }
                            KeyCode::Up => {
                                selected = selected.saturating_sub(1);
                            }
                            KeyCode::Down => {
                                if !app.fleet.is_empty() {
                                    selected = (selected + 1).min(app.fleet.len() - 1);
                                }
                            }
                            KeyCode::Tab => {
                                focus = match focus {
                                    Focus::Dispatch if !app.fleet.is_empty() => Focus::Reply,
                                    _ => Focus::Dispatch,
                                };
                            }
                            // Ctrl+S: dispatch AND attach (or attach the selected
                            // row when the dispatch box is empty).
                            KeyCode::Char('s') if ctrl => {
                                let text = dispatch.submit();
                                let text = text.trim().to_string();
                                if !text.is_empty() {
                                    match dispatch_new(app, text, spawner, &tagged_tx, &mut in_flight) {
                                        Ok(idx) => {
                                            selected = idx;
                                            focus = Focus::Attach;
                                        }
                                        Err(err) => flash = Some(format!("dispatch failed: {err:#}")),
                                    }
                                } else if !app.fleet.is_empty() {
                                    focus = Focus::Attach;
                                }
                            }
                            KeyCode::Enter => match focus {
                                Focus::Dispatch => {
                                    let text = dispatch.submit();
                                    let text = text.trim().to_string();
                                    if !text.is_empty() {
                                        match dispatch_new(app, text, spawner, &tagged_tx, &mut in_flight) {
                                            Ok(idx) => selected = idx,
                                            Err(err) => {
                                                flash = Some(format!("dispatch failed: {err:#}"))
                                            }
                                        }
                                    }
                                }
                                Focus::Reply | Focus::Attach => {
                                    if let Some(row) = app.fleet.get_mut(selected) {
                                        let text = row.reply.submit().trim().to_string();
                                        if !text.is_empty() {
                                            send_reply(app, selected, text, &tagged_tx, &mut in_flight);
                                        }
                                    }
                                }
                            },
                            // Single-keystroke answer: on an idle row with an
                            // empty reply box, 1–9 replies with that digit —
                            // enough to answer "1) do X or 2) do Y?" instantly.
                            KeyCode::Char(c @ '1'..='9')
                                if focus != Focus::Dispatch
                                    && app
                                        .fleet
                                        .get(selected)
                                        .is_some_and(|r| r.reply.is_empty() && r.state != RowState::Working) =>
                            {
                                send_reply(app, selected, c.to_string(), &tagged_tx, &mut in_flight);
                            }
                            // Ctrl+K: interrupt the selected row's in-flight tool
                            // (the model sees "interrupted by user" and adapts —
                            // same semantics as Esc during a chat turn).
                            KeyCode::Char('k') if ctrl => {
                                if let Some(row) = app.fleet.get(selected)
                                    && row.state == RowState::Working
                                    && let Some(flag) = &row.interrupt
                                {
                                    flag.store(true, Ordering::SeqCst);
                                }
                            }
                            KeyCode::Char('u') if ctrl => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::kill_to_start);
                            }
                            KeyCode::Char('a') if ctrl => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::home);
                            }
                            KeyCode::Char('e') if ctrl => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::end);
                            }
                            KeyCode::Char(c) if !ctrl => {
                                focused_input(app, selected, focus, &mut dispatch)
                                    .map(|input| input.insert(c));
                            }
                            KeyCode::Backspace => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::backspace);
                            }
                            KeyCode::Left => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::left);
                            }
                            KeyCode::Right => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::right);
                            }
                            KeyCode::Home => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::home);
                            }
                            KeyCode::End => {
                                focused_input(app, selected, focus, &mut dispatch).map(InputLine::end);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// The input that currently owns typed characters.
fn focused_input<'a>(
    app: &'a mut App,
    selected: usize,
    focus: Focus,
    dispatch: &'a mut InputLine,
) -> Option<&'a mut InputLine> {
    match focus {
        Focus::Dispatch => Some(dispatch),
        Focus::Reply | Focus::Attach => app.fleet.get_mut(selected).map(|r| &mut r.reply),
    }
}

/// Spawn a fresh agent for `prompt` and start its first turn. Returns the new
/// row's index.
fn dispatch_new(
    app: &mut App,
    prompt: String,
    spawner: &FleetSpawner,
    tagged_tx: &mpsc::UnboundedSender<(usize, UiEvent)>,
    in_flight: &mut FuturesUnordered<TurnFut>,
) -> Result<usize> {
    let agent = spawner()?;
    app.fleet_next_id += 1;
    let row = FleetRow::new(app.fleet_next_id, prompt.clone(), agent);
    app.fleet.push(row);
    let idx = app.fleet.len() - 1;
    start_turn(app, idx, prompt, tagged_tx, in_flight);
    Ok(idx)
}

/// Send `text` to the selected row: run it now if the agent is idle, else
/// queue it to run when the current turn finishes.
fn send_reply(
    app: &mut App,
    idx: usize,
    text: String,
    tagged_tx: &mpsc::UnboundedSender<(usize, UiEvent)>,
    in_flight: &mut FuturesUnordered<TurnFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    if row.state == RowState::Working || row.agent.is_none() {
        row.push_line(format!("⧗ queued: {text}"));
        row.pending.push_back(text);
    } else {
        start_turn(app, idx, text, tagged_tx, in_flight);
    }
}

/// Move the row's agent into a turn future (it comes back on completion).
fn start_turn(
    app: &mut App,
    idx: usize,
    prompt: String,
    tagged_tx: &mpsc::UnboundedSender<(usize, UiEvent)>,
    in_flight: &mut FuturesUnordered<TurnFut>,
) {
    let Some(row) = app.fleet.get_mut(idx) else {
        return;
    };
    let Some(mut agent) = row.agent.take() else {
        return;
    };
    row.interrupt = Some(agent.interrupt_handle());
    row.state = RowState::Working;
    row.started = Some(Instant::now());
    row.activity = "Working…".to_string();
    row.push_line(format!("› {prompt}"));

    // Per-turn channel, forwarded into the shared tagged stream. The forwarder
    // is a real spawn (UiEvent is Send); the turn future itself stays local.
    let (tx, mut rx) = mpsc::unbounded_channel::<UiEvent>();
    let fwd = tagged_tx.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if fwd.send((idx, event)).is_err() {
                break;
            }
        }
    });
    in_flight.push(Box::pin(async move {
        let mut sink = ChannelUi { tx };
        let result = agent.run_turn(&prompt, &mut sink).await;
        (idx, agent, result)
    }));
}

#[allow(clippy::too_many_arguments)]
fn render_dashboard(
    frame: &mut ratatui::Frame,
    app: &App,
    selected: usize,
    focus: Focus,
    dispatch: &InputLine,
    working: usize,
    exit_armed: bool,
    flash: Option<&str>,
) {
    let area = frame.area();
    let attach = focus == Focus::Attach;
    let table_height = if attach {
        0
    } else {
        app.fleet.len().clamp(1, TABLE_ROWS) as u16 + 2
    };
    let rows = Layout::vertical([
        Constraint::Length(1),            // header
        Constraint::Length(table_height), // fleet table (hidden in attach)
        Constraint::Min(3),               // peek / attach panel
        Constraint::Length(3),            // focused input
        Constraint::Length(1),            // footer hints
    ])
    .split(area);

    // --- Header ---
    let title = format!(
        " hi fleet · {} agent(s) · {} working{} ",
        app.fleet.len(),
        working,
        if exit_armed {
            " — turns in flight! Esc again to abandon them (sessions stay resumable)"
        } else {
            ""
        },
    );
    let header_style = if exit_armed {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD)
    };
    frame.render_widget(Paragraph::new(Line::styled(title, header_style)), rows[0]);

    // --- Table ---
    if !attach {
        render_table(frame, app, selected, rows[1]);
    }

    // --- Peek / attach panel ---
    render_peek(frame, app, selected, rows[2], attach);

    // --- Focused input ---
    render_input(frame, app, selected, focus, dispatch, rows[3]);

    // --- Footer ---
    let hint = match flash {
        Some(msg) => Line::styled(msg.to_string(), Style::default().fg(Color::Yellow)),
        None => Line::styled(
            match focus {
                Focus::Dispatch => {
                    "Enter dispatch new · Ctrl+S dispatch+attach · ↑↓ select · Tab reply · Esc back"
                }
                Focus::Reply => {
                    "Enter send · 1-9 quick answer · ↑↓ select · Tab dispatch · Esc back"
                }
                Focus::Attach => "Enter send · 1-9 quick answer · Esc table",
            }
            .to_string(),
            dim(),
        ),
    };
    frame.render_widget(Paragraph::new(hint), rows[4]);
}

fn render_table(frame: &mut ratatui::Frame, app: &App, selected: usize, area: Rect) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Magenta))
        .title(" fleet ");
    let inner_rows = area.height.saturating_sub(2) as usize;
    // Keep the selection visible: scroll the window over the fleet.
    let start = selected.saturating_sub(inner_rows.saturating_sub(1));
    let mut lines: Vec<Line> = Vec::new();
    if app.fleet.is_empty() {
        lines.push(Line::styled(
            "no agents yet — type a prompt below and press Enter to dispatch one".to_string(),
            dim(),
        ));
    }
    for (i, row) in app.fleet.iter().enumerate().skip(start).take(inner_rows) {
        let (glyph, glyph_style) = match row.state {
            RowState::Working => (
                SPINNER[app.spinner % SPINNER.len()].to_string(),
                Style::default().fg(Color::Cyan),
            ),
            RowState::Idle => ("·".to_string(), Style::default().fg(Color::Green)),
            RowState::Failed => ("✗".to_string(), Style::default().fg(Color::Red)),
        };
        let elapsed = row
            .started
            .map(|t| {
                let s = t.elapsed().as_secs();
                format!("{}m{:02}s", s / 60, s % 60)
            })
            .unwrap_or_else(|| format!("{} turn(s)", row.turns));
        let lead = if row.state == RowState::Working && !row.activity.is_empty() {
            &row.activity
        } else {
            &row.title
        };
        let queued = if row.pending.is_empty() {
            String::new()
        } else {
            format!(" ⧗{}", row.pending.len())
        };
        let text = format!(
            "#{:<2} {:>8} ↓{}{} {}",
            row.id,
            elapsed,
            crate::util::fmt_count(row.usage.1),
            queued,
            lead,
        );
        let style = if i == selected {
            Style::default().add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {glyph} "), glyph_style),
            Span::styled(text, style),
        ]));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_peek(frame: &mut ratatui::Frame, app: &App, selected: usize, area: Rect, attach: bool) {
    let Some(row) = app.fleet.get(selected) else {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(dim())
            .title(" peek ");
        frame.render_widget(
            Paragraph::new(Line::styled(
                "select a row to peek at its output".to_string(),
                dim(),
            ))
            .block(block),
            area,
        );
        return;
    };
    let title = format!(
        " #{} · {} {} ",
        row.id,
        truncate(&row.title, 48),
        if attach { "(attached)" } else { "" },
    );
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if attach { Color::Cyan } else { Color::DarkGray }))
        .title(title);
    let inner = area.height.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = Vec::new();
    let shown = row.tail.len().saturating_sub(inner.saturating_sub(1));
    for line in row.tail.iter().skip(shown) {
        let style = if line.starts_with('⚙') || line.starts_with('›') {
            dim()
        } else if line.starts_with('✗') || line.starts_with('⚠') {
            Style::default().fg(Color::Red)
        } else if line.starts_with('✓') {
            Style::default().fg(Color::Green)
        } else {
            Style::default()
        };
        lines.push(Line::styled(line.clone(), style));
    }
    if !row.partial.is_empty() {
        lines.push(Line::styled(row.partial.clone(), Style::default()));
    }
    if row.state == RowState::Working {
        lines.push(Line::styled(
            format!(
                "{} {}",
                SPINNER[app.spinner % SPINNER.len()],
                if row.activity.is_empty() {
                    "Working…"
                } else {
                    &row.activity
                }
            ),
            Style::default().fg(Color::Cyan),
        ));
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_input(
    frame: &mut ratatui::Frame,
    app: &App,
    selected: usize,
    focus: Focus,
    dispatch: &InputLine,
    area: Rect,
) {
    let (title, input, accent) = match focus {
        Focus::Dispatch => (
            " dispatch — Enter spawns a new agent · Ctrl+S spawns and attaches ".to_string(),
            dispatch,
            Color::Magenta,
        ),
        Focus::Reply | Focus::Attach => {
            let id = app.fleet.get(selected).map(|r| r.id).unwrap_or_default();
            let state = app
                .fleet
                .get(selected)
                .map(|r| {
                    if r.state == RowState::Working {
                        " (working — reply will queue)"
                    } else {
                        ""
                    }
                })
                .unwrap_or_default();
            (
                format!(" reply → #{id}{state} "),
                app.fleet
                    .get(selected)
                    .map(|r| &r.reply)
                    .unwrap_or(dispatch),
                Color::Cyan,
            )
        }
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .title(title);
    let text = input.text();
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", dim()),
            Span::raw(text.clone()),
        ]))
        .block(block),
        area,
    );
    // Place the terminal cursor inside the focused input.
    let cursor_col = input.cursor().min(text.chars().count()) as u16;
    frame.set_cursor_position((area.x + 3 + cursor_col, area.y + 1));
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> FleetRow {
        // A test row never touches the agent, so a dummy is unnecessary —
        // construct piecewise around it.
        FleetRow {
            id: 1,
            title: "test".into(),
            agent: None,
            interrupt: None,
            state: RowState::Working,
            activity: String::new(),
            tail: Vec::new(),
            partial: String::new(),
            pending: VecDeque::new(),
            reply: InputLine::default(),
            usage: (0, 0),
            started: None,
            turns: 0,
        }
    }

    #[test]
    fn streamed_text_splits_into_tail_lines() {
        let mut r = row();
        r.apply(UiEvent::Text("first li".into()));
        r.apply(UiEvent::Text("ne\nsecond line\npart".into()));
        assert_eq!(r.tail, vec!["first line", "second line"]);
        assert_eq!(r.partial, "part");
        r.apply(UiEvent::AssistantEnd);
        assert_eq!(r.tail.last().map(String::as_str), Some("part"));
        assert!(r.partial.is_empty());
    }

    #[test]
    fn tail_is_capped() {
        let mut r = row();
        for i in 0..(TAIL_CAP + 50) {
            r.push_line(format!("line {i}"));
        }
        assert_eq!(r.tail.len(), TAIL_CAP);
        assert_eq!(r.tail.first().map(String::as_str), Some("line 50"));
    }

    #[test]
    fn events_map_to_row_state() {
        let mut r = row();
        r.apply(UiEvent::ToolStarted(
            "bash".into(),
            "{\"command\":\"ls\"}".into(),
        ));
        assert!(r.activity.starts_with("running "), "{}", r.activity);
        r.apply(UiEvent::Usage {
            input: 100,
            output: 42,
            ctx_used: 0,
            ctx_window: None,
        });
        assert_eq!(r.usage, (100, 42));
        r.apply(UiEvent::TurnEnd("did the thing".into()));
        assert_eq!(r.tail.last().map(String::as_str), Some("✓ did the thing"));
        r.apply(UiEvent::TurnError(
            "auth".into(),
            "401".into(),
            String::new(),
        ));
        assert!(r.tail.last().unwrap().starts_with("✗ auth"));
    }
}
