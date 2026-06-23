//! Full-screen terminal UI for `hi`.
//!
//! A ratatui application on the alternate screen: a bordered, scrollable
//! conversation transcript with a title/status bar, and an input box with a
//! "working" spinner. The agent runs behind an mpsc channel ([`ChannelUi`]) so
//! the event loop can keep redrawing — spinner, streaming output, scrolling —
//! while a turn is in flight, and can cancel it with Ctrl-C.

use std::collections::{HashMap, VecDeque};
use std::io::{self, Write};
use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, DisableFocusChange, EnableBracketedPaste, EnableFocusChange, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use hi_agent::ui::tool_label;
use hi_agent::{Agent, Command, CompactionKind, Ui, command};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use tokio::sync::mpsc;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// How many model rows the `/model` picker shows at once.
const PICKER_ROWS: usize = 12;
const TICK: Duration = Duration::from_millis(120);
/// Only flag a possibly-stuck provider after a long, genuinely silent wait — the
/// working line already shows the live elapsed time, so an earlier notice just
/// reads as alarming when the model/provider is merely slow.
const WATCHDOG_STUCK: Duration = Duration::from_secs(60);
/// On terminals that don't report focus, notify after a turn at least this long
/// (a proxy for "you probably stepped away").
const NOTIFY_THRESHOLD: Duration = Duration::from_secs(30);

/// Run the full-screen TUI until the user quits. `history_path`, if given, is
/// the file used to persist input history across sessions (shared with the
/// plain REPL).
pub async fn run(
    agent: &mut Agent,
    provider: &str,
    model: &str,
    registry: &hi_ai::Registry,
    history_path: Option<std::path::PathBuf>,
    auto_memory: bool,
) -> Result<()> {
    enable_raw_mode().context("entering raw mode")?;
    execute!(io::stdout(), EnterAlternateScreen).context("entering alternate screen")?;
    // Bracketed paste: the terminal wraps a paste so it arrives as one
    // Event::Paste instead of per-line Enter keys (which would submit each line).
    let _ = execute!(io::stdout(), EnableBracketedPaste);
    // Focus reporting: lets us tell when you've switched away, so a finished turn
    // can ping you only when you're not looking. Harmless if unsupported.
    let _ = execute!(io::stdout(), EnableFocusChange);
    // Deliberately NOT capturing the mouse: capture would route wheel events to
    // us (enabling in-app scroll) but the terminal would then stop doing native
    // click-drag selection, breaking copy/paste. Native selection wins; scroll
    // with PageUp/PageDown.
    let _restore = Restore;
    let mut terminal =
        Terminal::new(CrosstermBackend::new(io::stdout())).context("creating terminal")?;

    let mut app = App::new(provider, model);
    // Seed the context-fill gauge with the model's window so it reads 0% before
    // the first turn (it refreshes from real usage after each round).
    app.context_window = registry.metadata(model).1;
    // The catalog, for inline `/model <id>` completion (the picker fetches the
    // live list on demand; this is the synchronous type-ahead source).
    app.model_ids = registry.model_ids();
    if let Some(path) = &history_path
        && let Ok(text) = std::fs::read_to_string(path)
    {
        app.input.history = text
            .lines()
            .map(str::to_string)
            .filter(|l| !l.trim().is_empty())
            .collect();
    }
    app.push(Line::styled(
        "Welcome to hi. Enter to send; keep typing while it works to queue the next; \
         Ctrl-C interrupts, /exit quits.",
        dim(),
    ));
    // Read terminal events in a dedicated task and forward them over a channel.
    // A channel receiver is fully cancel-safe, so the per-tick redraws in the
    // loops below can't drop or delay a keystroke — which repeatedly cancelling
    // an `EventStream::next()` future inside `select!` can.
    let (input_tx, mut input_rx) = mpsc::unbounded_channel::<Event>();
    tokio::spawn(async move {
        let mut events = EventStream::new();
        loop {
            match events.next().await {
                Some(Ok(event)) => {
                    if input_tx.send(event).is_err() {
                        break; // main loop gone — stop reading
                    }
                }
                Some(Err(_)) => continue, // skip a malformed event, keep reading
                None => break,            // stdin closed
            }
        }
    });
    let mut ticker = tokio::time::interval(TICK);
    let mut startup_check = app.context_window.is_none();

    'session: loop {
        // Run a queued command first (typed while the previous turn ran);
        // otherwise edit the input line until the user submits.
        let line = match app.queue.pop_front() {
            Some(queued) => queued,
            None => 'input: loop {
                terminal.draw(|f| app.render(f))?;
                // Block on input, with a one-shot startup metadata check racing
                // in the background. Input always wins immediately; the check is
                // best-effort and never delays typing into the prompt.
                let event = if startup_check {
                    enum StartupRace {
                        Metadata(Result<Vec<hi_ai::ServedModel>>),
                        Input(Option<Event>),
                    }
                    let race = {
                        let fut = agent.list_models();
                        tokio::pin!(fut);
                        tokio::select! {
                            result = &mut fut => StartupRace::Metadata(result),
                            maybe = input_rx.recv() => StartupRace::Input(maybe),
                        }
                    };
                    startup_check = false;
                    match race {
                        StartupRace::Metadata(Ok(served)) if !served.is_empty() => {
                            app.served = served.into_iter().map(|m| (m.id.clone(), m)).collect();
                            let model_id = app.model.clone();
                            if let Some(health) = app.apply_model(agent, registry, &model_id) {
                                app.warn_degraded(&model_id, &health);
                            }
                            continue;
                        }
                        StartupRace::Metadata(Ok(_)) => {
                            app.startup_notice =
                                Some("provider metadata unavailable; using catalog".to_string());
                            continue;
                        }
                        StartupRace::Metadata(Err(err)) => {
                            app.startup_notice =
                                Some(format!("provider metadata check failed: {err:#}"));
                            continue;
                        }
                        StartupRace::Input(Some(event)) => event,
                        StartupRace::Input(None) => break 'session,
                    }
                } else {
                    let Some(event) = input_rx.recv().await else {
                        break 'session; // input channel closed (stdin gone)
                    };
                    event
                };
                match event {
                    // A paste arrives as one event — insert it literally
                    // (newlines and all) instead of submitting each line.
                    Event::Paste(text) => app.input.insert_str(&text),
                    // While the model picker is open, keys drive it.
                    Event::Key(key) if key.kind == KeyEventKind::Press && app.picker.is_some() => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Enter => app.pick_model(agent, registry),
                            KeyCode::Esc => app.picker = None,
                            KeyCode::Char('c') if ctrl => app.picker = None,
                            // Navigation/filter: a fresh borrow, no app-level action.
                            code => {
                                let picker = app.picker.as_mut().unwrap();
                                match code {
                                    KeyCode::Up => picker.up(),
                                    KeyCode::Down => picker.down(),
                                    KeyCode::PageUp => picker.page_up(),
                                    KeyCode::PageDown => picker.page_down(),
                                    KeyCode::Backspace => picker.backspace(),
                                    KeyCode::Char(c) if !ctrl => picker.insert(c),
                                    _ => {}
                                }
                            }
                        }
                    }
                    // When the `/`-command menu is open, navigation/accept keys
                    // drive it; anything else edits the input and re-syncs it.
                    Event::Key(key)
                        if key.kind == KeyEventKind::Press && app.completion.is_some() =>
                    {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Char('c') if ctrl => app.input.clear(),
                            KeyCode::Esc => app.completion = None,
                            KeyCode::Up => app.completion_move(-1),
                            KeyCode::Down => app.completion_move(1),
                            KeyCode::Tab => {
                                // Completing a command that takes arguments fills
                                // `/name ` — re-sync so its value menu opens next.
                                app.accept_completion(false);
                                app.sync_completion();
                            }
                            KeyCode::Enter => {
                                if let Some(line) = app.accept_completion(true) {
                                    break 'input line;
                                }
                            }
                            _ => {
                                if let Some(line) = app.edit_key(&key) {
                                    break 'input line;
                                }
                                app.sync_completion();
                            }
                        }
                    }
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Char('d') if ctrl && app.input.is_empty() => break 'session,
                            KeyCode::Char('c') if ctrl => app.input.clear(),
                            KeyCode::Esc if app.input.is_empty() => break 'session,
                            KeyCode::Esc => app.input.clear(),
                            _ => {
                                if let Some(line) = app.edit_key(&key) {
                                    break 'input line;
                                }
                                app.sync_completion();
                            }
                        }
                    }
                    Event::FocusGained => app.set_focus(true),
                    Event::FocusLost => app.set_focus(false),
                    _ => {}
                }
            },
        };
        // A line is committed — drop any lingering completion menu state.
        app.completion = None;

        // Slash commands. Most are handled inline; `/compact` runs a model call
        // (driven like a turn so the spinner shows); `/retry` yields the prompt
        // to re-run in the turn phase below.
        let run_line = if let Some(cmd) = command::parse(&line) {
            match cmd {
                Command::Quit => break,
                Command::Compact(arg) => {
                    let kind =
                        CompactionKind::from_arg(&arg).unwrap_or_else(|| agent.compaction_kind());
                    app.set_working(true);
                    app.follow();
                    let (tx, rx) = mpsc::unbounded_channel();
                    let mut sink = ChannelUi { tx };
                    {
                        let fut = agent.compact_with(kind, &mut sink);
                        drive(
                            &mut terminal,
                            &mut input_rx,
                            &mut ticker,
                            &mut app,
                            rx,
                            fut,
                            false,
                        )
                        .await?;
                    }
                    app.set_working(false);
                    app.follow();
                    continue;
                }
                Command::Retry => match app.last_prompt.clone() {
                    Some(prompt) => {
                        agent.truncate_messages(app.last_turn_start);
                        let note = match app.last_turn_state {
                            TurnState::Warning(_) => {
                                if app.last_turn_had_file_edits {
                                    "retrying from the last safe message checkpoint; file edits already made stay in the working tree and may be replayed if the model repeats them"
                                } else {
                                    "retrying from the last safe message checkpoint; no file edits were recorded in the last turn"
                                }
                            }
                            TurnState::Failed(_) => {
                                "retrying after failure from the last safe message checkpoint"
                            }
                            _ => "retrying from the last safe message checkpoint",
                        };
                        app.push(Line::styled(note.to_string(), dim()));
                        app.push(Line::styled(format!("retrying: {prompt}"), dim()));
                        prompt
                    }
                    None => {
                        app.push(Line::styled("nothing to retry yet".to_string(), dim()));
                        continue;
                    }
                },
                Command::Init => {
                    app.push(Line::styled(
                        "scanning the project to write HI.md…".to_string(),
                        dim(),
                    ));
                    command::INIT_PROMPT.to_string()
                }
                Command::Undo => {
                    let checkpoints = agent.checkpoint_count();
                    if checkpoints > 0 {
                        app.push(Line::styled(
                            format!(
                                "undo: restoring latest checkpoint ({checkpoints} available); non-file side effects cannot be reverted"
                            ),
                            dim(),
                        ));
                    }
                    let msg = match agent.undo().await {
                        Ok(Some(0)) => "nothing changed to undo".to_string(),
                        Ok(Some(n)) => format!("↩ undid the last turn — restored {n} file(s)"),
                        Ok(None) => "nothing to undo".to_string(),
                        Err(err) => format!("undo failed: {err:#}"),
                    };
                    app.push(Line::styled(msg, dim()));
                    app.follow();
                    continue;
                }
                // Open the picker on the provider's *live* model list (what this
                // endpoint actually serves), falling back to the static catalog.
                // The fetch runs behind a spinner so the UI stays responsive and
                // Esc/Ctrl-C can cancel a slow or hung endpoint.
                Command::Model(id) if id.is_empty() => {
                    app.fetching = Some(Instant::now());
                    let mut fetched: Option<Result<Vec<hi_ai::ServedModel>>> = None;
                    let mut cancelled = false;
                    {
                        let fut = agent.list_models();
                        tokio::pin!(fut);
                        loop {
                            terminal.draw(|f| app.render(f))?;
                            tokio::select! {
                                result = &mut fut => { fetched = Some(result); break; }
                                _ = ticker.tick() => app.spinner = app.spinner.wrapping_add(1),
                                maybe = input_rx.recv() => {
                                    if let Some(Event::Key(key)) = maybe
                                        && key.kind == KeyEventKind::Press
                                    {
                                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                        if matches!(key.code, KeyCode::Esc)
                                            || (ctrl && matches!(key.code, KeyCode::Char('c')))
                                        {
                                            cancelled = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    app.fetching = None;
                    if cancelled {
                        continue;
                    }
                    let ids = match fetched {
                        Some(Ok(served)) if !served.is_empty() => {
                            // Remember the live metadata (window/price/health) so
                            // selecting a model can apply it and tag its health.
                            app.served = served.into_iter().map(|m| (m.id.clone(), m)).collect();
                            let mut ids: Vec<String> = app.served.keys().cloned().collect();
                            ids.sort();
                            ids
                        }
                        Some(Ok(_)) => {
                            app.push(Line::styled(
                                "provider listed no models; showing the catalog".to_string(),
                                dim(),
                            ));
                            registry.model_ids()
                        }
                        _ => {
                            let note = match fetched {
                                Some(Err(err)) => {
                                    format!("couldn't fetch models ({err:#}); showing the catalog")
                                }
                                _ => "showing the catalog".to_string(),
                            };
                            app.push(Line::styled(note, dim()));
                            registry.model_ids()
                        }
                    };
                    let current = app.model.clone();
                    let tags = app.served_tags();
                    app.picker = Some(ModelPicker::new(ids, &current, tags));
                    continue;
                }
                other => {
                    app.handle_command(agent, other, registry);
                    continue;
                }
            }
        } else {
            line
        };

        // --- Turn phase: run the agent behind a channel, staying responsive. ---
        app.push(Line::styled(
            format!("› {run_line}"),
            Style::default().fg(Color::Blue),
        ));
        app.set_working(true);
        app.follow();
        let checkpoint = agent.messages().len();
        app.last_turn_start = checkpoint;
        app.last_prompt = Some(run_line.clone());
        let (tx, rx) = mpsc::unbounded_channel();
        let mut sink = ChannelUi { tx };
        let cancelled = {
            let fut = agent.run_turn(&run_line, &mut sink);
            drive(
                &mut terminal,
                &mut input_rx,
                &mut ticker,
                &mut app,
                rx,
                fut,
                true,
            )
            .await?
        };

        if cancelled {
            agent.truncate_messages(checkpoint);
            app.last_turn_state = TurnState::Cancelled;
            let dropped = app.queue.len();
            app.queue.clear();
            let msg = if dropped > 0 {
                format!("^C interrupted; turn discarded ({dropped} queued command(s) dropped)")
            } else {
                "^C interrupted; turn discarded".to_string()
            };
            app.push(Line::styled(msg, Style::default().fg(Color::Yellow)));
        } else {
            // Turn finished on its own — ping if you've likely stepped away.
            app.maybe_notify_done();
        }
        app.set_working(false);
        // No follow() at turn end: if the user scrolled up to read mid-turn, leave
        // them there (the "↓ N new" hint shows the summary is below). A new turn
        // re-pins to the bottom.
    }

    // Session ending: distill durable lessons into .hi/memory.md (loaded next
    // session), shown live so the user sees what's saved. Only if work happened.
    if hi_agent::should_distill_memory(auto_memory, agent.totals().output_tokens) {
        app.set_working(true);
        app.follow();
        let (tx, rx) = mpsc::unbounded_channel();
        let mut sink = ChannelUi { tx };
        {
            let fut = async {
                agent.update_memory(&mut sink).await;
                Ok::<(), anyhow::Error>(())
            };
            let _ = drive(
                &mut terminal,
                &mut input_rx,
                &mut ticker,
                &mut app,
                rx,
                fut,
                false,
            )
            .await;
        }
        app.set_working(false);
    }

    // Persist input history for next time.
    if let Some(path) = &history_path {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, app.input.history.join("\n"));
    }

    Ok(())
}

/// Drive a model future (a turn or a compaction) to completion while keeping
/// the UI live: redraw + spin every tick, drain the agent's events, let the
/// user scroll/queue/cancel. Returns whether the user cancelled with Ctrl-C.
async fn drive(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    input: &mut mpsc::UnboundedReceiver<Event>,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    mut rx: mpsc::UnboundedReceiver<UiEvent>,
    fut: impl std::future::Future<Output = Result<()>>,
    expect_turn_end: bool,
) -> Result<bool> {
    tokio::pin!(fut);
    let mut cancelled = false;
    let mut saw_turn_end = false;
    let mut last_activity = Instant::now();
    let mut watchdog_stuck = false;
    loop {
        terminal.draw(|f| app.render(f))?;
        tokio::select! {
            result = &mut fut => {
                while let Ok(event) = rx.try_recv() {
                    if matches!(event, UiEvent::TurnEnd(_)) {
                        saw_turn_end = true;
                    }
                    app.apply(event);
                }
                if let Err(err) = result {
                    app.note_turn_failed(&format!("{err:#}"));
                    app.record_model_issue();
                } else if expect_turn_end && !cancelled && !saw_turn_end {
                    app.note_turn_completed_without_summary();
                }
                break;
            }
            Some(event) = rx.recv() => {
                if matches!(event, UiEvent::TurnEnd(_)) {
                    saw_turn_end = true;
                }
                last_activity = Instant::now();
                app.apply(event);
            }
            _ = ticker.tick() => {
                app.spinner = app.spinner.wrapping_add(1);
                let idle = last_activity.elapsed();
                app.waiting_for = Some(idle);
                // Only warn about a stuck *model/provider* — not while a tool is
                // legitimately running (its own timer shows in the working line),
                // which would otherwise fire a false alarm and mark the model
                // degraded for what is really a slow `cargo test`.
                if expect_turn_end
                    && !watchdog_stuck
                    && app.current_tool.is_none()
                    && idle >= WATCHDOG_STUCK
                {
                    watchdog_stuck = true;
                    app.push(Line::styled(
                        "⚠ no response for 60s; the model/provider may be stuck. Ctrl-C to cancel, then try /model or /retry",
                        Style::default().fg(Color::Yellow),
                    ));
                    app.follow();
                    app.record_model_issue();
                }
            },
            maybe = input.recv() => {
                match maybe {
                    Some(Event::Paste(text)) => app.input.insert_str(&text),
                    Some(Event::Key(key)) if key.kind == KeyEventKind::Press => {
                        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                        match key.code {
                            KeyCode::Char('c') if ctrl => { cancelled = true; break; }
                            // Esc clears a half-typed queued command, or — when the
                            // input is empty — interrupts the running turn (a second
                            // way out besides Ctrl-C).
                            KeyCode::Esc if app.input.is_empty() => { cancelled = true; break; }
                            KeyCode::Esc => app.input.clear(),
                            // Typing while a turn runs queues the next command —
                            // except `/tokens`, which reads the live counter and
                            // runs in sync so you can watch usage climb mid-turn.
                            _ => if let Some(submitted) = app.edit_key(&key) {
                                match command::parse(&submitted) {
                                    Some(Command::Tokens) => app.report_tokens(),
                                    Some(Command::Copy(arg)) => app.copy(&arg),
                                    _ => app.queue.push_back(submitted),
                                }
                            }
                        }
                    }
                    Some(Event::FocusGained) => app.set_focus(true),
                    Some(Event::FocusLost) => app.set_focus(false),
                    _ => {}
                }
            }
        }
    }
    app.waiting_for = None;
    Ok(cancelled)
}

/// Restores the terminal on drop (covers early returns and panics).
struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            DisableFocusChange,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
    }
}

/// Events the agent emits; drained by the event loop into [`App`].
enum UiEvent {
    Text(String),
    Reasoning(String),
    AssistantEnd,
    ToolCall(String, String),
    ToolResult(String),
    Status(String),
    Usage {
        input: u64,
        output: u64,
        ctx_used: u64,
        ctx_window: Option<u32>,
    },
    TurnEnd(String),
}

/// The [`Ui`] handed to the agent: forwards everything over a channel so the
/// turn never borrows the live [`App`].
struct ChannelUi {
    tx: mpsc::UnboundedSender<UiEvent>,
}

impl ChannelUi {
    fn send(&self, event: UiEvent) {
        let _ = self.tx.send(event);
    }
}

impl Ui for ChannelUi {
    fn assistant_text(&mut self, text: &str) {
        self.send(UiEvent::Text(text.to_string()));
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.send(UiEvent::Reasoning(text.to_string()));
    }
    fn assistant_end(&mut self) {
        self.send(UiEvent::AssistantEnd);
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::ToolCall(name.to_string(), arguments.to_string()));
    }
    fn tool_result(&mut self, result: &str) {
        self.send(UiEvent::ToolResult(result.to_string()));
    }
    fn status(&mut self, text: &str) {
        self.send(UiEvent::Status(text.to_string()));
    }
    fn usage(
        &mut self,
        input_tokens: u64,
        output_tokens: u64,
        context_used: u64,
        context_window: Option<u32>,
    ) {
        self.send(UiEvent::Usage {
            input: input_tokens,
            output: output_tokens,
            ctx_used: context_used,
            ctx_window: context_window,
        });
    }
    fn turn_end(&mut self, summary: &str) {
        self.send(UiEvent::TurnEnd(summary.to_string()));
    }
}

struct App {
    provider: String,
    model: String,
    transcript: Vec<Line<'static>>,
    /// The in-progress streamed line: (style, markdown?, text). Committed on
    /// newline/end. `markdown` is set for assistant prose so it's rendered with
    /// light markdown styling; reasoning and other streams stay literal.
    pending: Option<(Style, bool, String)>,
    /// The language of the ``` fence the streamed assistant text is currently
    /// inside (empty string if the fence gave none); `None` when not in a fence.
    /// Carries across streamed lines so code interiors highlight consistently.
    code_lang: Option<String>,
    input: InputLine,
    /// Transcript scroll state. `following` pins the view to the latest output
    /// (the default); scrolling up unpins it and `scroll` holds the absolute
    /// offset (wrapped lines hidden above the viewport). It re-pins once scrolled
    /// back to the bottom, so streaming output never yanks a reader downward.
    following: bool,
    scroll: u16,
    /// Cached each render so scroll events (which fire outside render and don't
    /// know the wrapped height) can clamp and detect the bottom.
    view_max_scroll: u16,
    view_total: u16,
    /// Wrapped-line total at the moment the view last left the bottom — drives
    /// the "↓ N new" indicator while scrolled up.
    total_when_unpinned: u16,
    working: bool,
    spinner: usize,
    /// When the current turn started, for the elapsed-time readout.
    started: Option<Instant>,
    /// The tool currently executing (its display label) and when it started, so
    /// the working line can name the in-flight action with its own timer. `None`
    /// while the model — not a tool — is the active party.
    current_tool: Option<String>,
    current_tool_started: Option<Instant>,
    /// Lines typed while a turn was running, to run once it finishes (FIFO).
    queue: VecDeque<String>,
    /// The last message actually sent to the model, for `/retry`.
    last_prompt: Option<String>,
    /// Message-history length just before the last turn started, so `/retry`
    /// can drop that turn before re-running.
    last_turn_start: usize,
    /// Active model picker (`/model` with no argument), if any.
    picker: Option<ModelPicker>,
    /// When set, a model-list fetch is in flight (start time, for the spinner).
    fetching: Option<Instant>,
    status: String,
    /// Cumulative session token usage (input, output), mirrored from the agent
    /// so the working line and `/tokens` can show it live while a turn runs.
    usage: (u64, u64),
    /// Current context occupancy (tokens of the last request) and the model's
    /// window, for the live context-fill gauge.
    context_used: u64,
    context_window: Option<u32>,
    /// Live per-model metadata (window/price/health) learned from the endpoint's
    /// `/models`, keyed by id — used to apply a model's settings and flag health.
    served: HashMap<String, hi_ai::ServedModel>,
    /// The model catalog (ids), for inline `/model <id>` type-ahead completion.
    model_ids: Vec<String>,
    /// Assistant prose currently streaming. Tool output is intentionally not
    /// included; `/copy` copies the assistant's answer, not command logs.
    current_assistant: String,
    /// Last completed assistant prose, copied by `/copy`.
    last_assistant: String,
    /// Last event type applied during the active turn, for better fallback
    /// diagnostics when the provider stops without a final turn-end event.
    last_turn_event: Option<TurnEventKind>,
    /// Whether the current/last turn invoked file-editing tools.
    last_turn_had_file_edits: bool,
    waiting_for: Option<Duration>,
    last_turn_state: TurnState,
    last_error: Option<String>,
    event_log: Vec<String>,
    model_issues: HashMap<String, u32>,
    startup_notice: Option<String>,
    /// Active `/`-command completion menu: the query it's synced to and the
    /// highlighted row. `None` when the input isn't a slash-command prefix.
    completion: Option<CompletionState>,
    /// Whether the terminal currently has focus (best-effort, via focus-change
    /// reporting). Stays `true` on terminals that don't report it.
    focused: bool,
    /// Set once we've seen any focus event — i.e. the terminal reports focus, so
    /// `focused` is trustworthy.
    focus_known: bool,
}

/// State of the slash-command completion menu.
struct CompletionState {
    /// What the menu is completing — a command name, or the argument of a known
    /// command — and the prefix it's filtered to.
    ctx: CompletionContext,
    /// Index of the highlighted match.
    selected: usize,
}

/// What the completion menu is offering, derived from the input line.
#[derive(Clone, Debug, PartialEq, Eq)]
enum CompletionContext {
    /// Typing the command name itself (`/`, `/co`) — the lowercased prefix.
    Command(String),
    /// Typing the argument of a command that has enumerable values (`/compact `,
    /// `/compact hy`) — the canonical command name and the lowercased value prefix.
    Arg { cmd: &'static str, prefix: String },
}

/// One row in the completion menu — a command name or an argument value, already
/// resolved to what shows and what gets inserted.
struct CompletionItem {
    /// Left column: `/compact` for a command, `hybrid` for an argument value.
    label: String,
    /// Right-column hint.
    help: String,
    /// Text the input becomes when this row is accepted.
    insert: String,
    /// Whether accepting with Enter submits the line. Command names that take
    /// arguments fill `/name ` and wait; everything else (no-arg commands, fully
    /// chosen argument values) is a complete line that runs.
    submit_on_enter: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TurnEventKind {
    Assistant,
    Reasoning,
    AssistantEnd,
    ToolCall,
    ToolResult,
    Status,
    Usage,
    TurnEnd,
}


#[derive(Clone, Debug, Eq, PartialEq)]
enum TurnState {
    Idle,
    Running,
    Done(String),
    Warning(String),
    Failed(String),
    Cancelled,
}

impl App {
    fn new(provider: &str, model: &str) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            transcript: Vec::new(),
            pending: None,
            code_lang: None,
            input: InputLine::default(),
            following: true,
            scroll: 0,
            view_max_scroll: 0,
            view_total: 0,
            total_when_unpinned: 0,
            working: false,
            spinner: 0,
            started: None,
            current_tool: None,
            current_tool_started: None,
            queue: VecDeque::new(),
            last_prompt: None,
            last_turn_start: 0,
            picker: None,
            fetching: None,
            status: String::new(),
            usage: (0, 0),
            context_used: 0,
            context_window: None,
            served: HashMap::new(),
            model_ids: Vec::new(),
            current_assistant: String::new(),
            last_assistant: String::new(),
            last_turn_event: None,
            last_turn_had_file_edits: false,
            waiting_for: None,
            last_turn_state: TurnState::Idle,
            last_error: None,
            event_log: Vec::new(),
            model_issues: HashMap::new(),
            startup_notice: None,
            completion: None,
            focused: true,
            focus_known: false,
        }
    }

    /// Record a focus-change report from the terminal (and that it reports them).
    fn set_focus(&mut self, focused: bool) {
        self.focused = focused;
        self.focus_known = true;
    }

    /// Ping the terminal when a turn finishes and you're likely away: when the
    /// terminal reports it's unfocused, or — on terminals that don't report
    /// focus — when the turn ran long enough that you probably stepped away.
    fn maybe_notify_done(&self) {
        let elapsed = self.started.map(|t| t.elapsed()).unwrap_or_default();
        let away = if self.focus_known {
            !self.focused
        } else {
            elapsed >= NOTIFY_THRESHOLD
        };
        if away {
            notify_done();
        }
    }

    /// Rows for a completion context — model ids from the live catalog, every
    /// other command's values from the static table.
    fn items_for_ctx(&self, ctx: &CompletionContext) -> Vec<CompletionItem> {
        if let CompletionContext::Arg { cmd, prefix } = ctx
            && *cmd == MODEL_CMD
        {
            return self.model_completion_items(prefix);
        }
        completion_items_for(ctx)
    }

    /// Up to [`MODEL_COMPLETION_MAX`] catalog ids starting with `prefix` (already
    /// lowercased), as `/model <id>` rows — inline type-ahead for `/model`.
    fn model_completion_items(&self, prefix: &str) -> Vec<CompletionItem> {
        self.model_ids
            .iter()
            .filter(|id| id.to_lowercase().starts_with(prefix))
            .take(MODEL_COMPLETION_MAX)
            .map(|id| CompletionItem {
                label: id.clone(),
                help: String::new(),
                insert: format!("/{MODEL_CMD} {id}"),
                submit_on_enter: true,
            })
            .collect()
    }

    /// The rows the completion menu currently offers (empty when closed).
    fn completion_items(&self) -> Vec<CompletionItem> {
        match &self.completion {
            Some(c) => self.items_for_ctx(&c.ctx),
            None => Vec::new(),
        }
    }

    /// Re-sync the completion menu to the current input: open/refresh it when the
    /// input is a slash-command name being typed (`/`, `/mo`, …) or the argument
    /// of a command with enumerable values (`/compact `, `/model gp`), with
    /// matches; otherwise close it. Called after every edit to the input line.
    fn sync_completion(&mut self) {
        match completion_context(&self.input.text()) {
            Some(ctx) if !self.items_for_ctx(&ctx).is_empty() => {
                // Reset the highlight only when the context actually changed, so
                // navigation survives unrelated redraws.
                if self.completion.as_ref().map(|c| &c.ctx) != Some(&ctx) {
                    self.completion = Some(CompletionState { ctx, selected: 0 });
                }
            }
            _ => self.completion = None,
        }
    }

    /// Move the completion highlight by `delta`, clamped to the match list.
    fn completion_move(&mut self, delta: isize) {
        let len = self.completion_items().len();
        if let Some(c) = &mut self.completion
            && len > 0
        {
            let last = len - 1;
            c.selected = match delta {
                d if d < 0 => c.selected.saturating_sub(1),
                _ => (c.selected + 1).min(last),
            };
        }
    }

    /// Accept the highlighted completion: replace the input with the row's
    /// insertion (`/name`, `/name ` for an arg-taking command, or `/cmd value`)
    /// and close the menu. When `submit` is set and the row is a complete line,
    /// return it to run immediately; otherwise leave it in the input.
    fn accept_completion(&mut self, submit: bool) -> Option<String> {
        let items = self.completion_items();
        let c = self.completion.as_ref()?;
        let item = items.get(c.selected)?;
        let submit_on_enter = item.submit_on_enter;
        self.input.set(&item.insert);
        self.completion = None;
        if submit && submit_on_enter {
            Some(self.input.submit())
        } else {
            None
        }
    }

    /// Health tags (id → label) for the models we have live metadata on, for the
    /// `/model` picker. Healthy models are omitted.
    fn served_tags(&self) -> HashMap<String, String> {
        self.served
            .iter()
            .map(|(id, m)| {
                let tag = match (
                    m.health().map(str::to_string),
                    self.model_issues.get(id).copied().unwrap_or(0),
                ) {
                    (Some(endpoint), issues) if issues > 0 => {
                        format!("{endpoint}; degraded in-session")
                    }
                    (Some(endpoint), _) => endpoint,
                    (None, issues) if issues > 0 => "degraded in-session".to_string(),
                    (None, _) => String::new(),
                };
                (id.clone(), tag)
            })
            .filter_map(|(id, tag)| (!tag.is_empty()).then_some((id, tag)))
            .collect()
    }

    /// Apply `id` as the model: prefer live endpoint metadata (window/price) when
    /// we have it, else the catalog. Updates the agent and the gauge. Returns the
    /// model's health label if the endpoint flags it as not fully available.
    fn apply_model(
        &mut self,
        agent: &mut Agent,
        registry: &hi_ai::Registry,
        id: &str,
    ) -> Option<String> {
        let (cat_price, cat_window) = registry.metadata(id);
        let served = self.served.get(id);
        let price = served.and_then(|m| m.price).or(cat_price);
        let window = served.and_then(|m| m.context_window).or(cat_window);
        agent.set_model(id.to_string(), price, window);
        self.model = id.to_string();
        self.context_window = window;
        served.and_then(|m| m.health()).map(str::to_string)
    }

    /// Push a yellow line warning that `id` is in a non-healthy state.
    fn warn_degraded(&mut self, id: &str, health: &str) {
        self.push(Line::styled(
            format!(
                "⚠ {id} is reported {health} on this endpoint — responses may be slow or flaky; \
                 /model to pick another"
            ),
            Style::default().fg(Color::Yellow),
        ));
    }

    /// Percent of the context window currently occupied, when the window is known.
    fn context_pct(&self) -> Option<u64> {
        let window = u64::from(self.context_window?);
        (window > 0).then(|| (self.context_used * 100 / window).min(100))
    }

    /// Apply the picker's current selection as the model, then close it.
    fn pick_model(&mut self, agent: &mut Agent, registry: &hi_ai::Registry) {
        let id = self
            .picker
            .as_ref()
            .and_then(|p| p.current())
            .map(str::to_string);
        if let Some(id) = id {
            let health = self.apply_model(agent, registry, &id);
            self.push(Line::styled(format!("model set to {id}"), dim()));
            if let Some(h) = health {
                self.warn_degraded(&id, &h);
            }
        }
        self.picker = None;
        self.follow();
    }

    /// Mark the turn as running (or done), stamping the start time so the
    /// prompt bar can show elapsed seconds.
    fn set_working(&mut self, working: bool) {
        self.working = working;
        self.started = working.then(Instant::now);
        self.current_tool = None;
        self.current_tool_started = None;
        if working {
            self.last_turn_event = None;
            self.last_turn_had_file_edits = false;
            self.waiting_for = Some(Duration::ZERO);
            self.last_turn_state = TurnState::Running;
        } else if matches!(self.last_turn_state, TurnState::Running) {
            self.last_turn_state = TurnState::Idle;
            self.waiting_for = None;
        }
    }

    /// The live "what's happening now" lead for the working line: the in-flight
    /// tool named with its own elapsed timer, otherwise the model phase —
    /// `thinking…` (reasoning), `responding…` (streaming text), or `waiting for
    /// the model…` (the round's model call is in flight but nothing's streamed
    /// yet). Lets you tell a slow tool from a slow model at a glance.
    fn activity_line(&self) -> String {
        if let (Some(tool), Some(started)) = (&self.current_tool, self.current_tool_started) {
            return format!("running {tool} · {}", fmt_elapsed(started.elapsed().as_secs()));
        }
        let secs = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
        let verb = match self.last_turn_event {
            Some(TurnEventKind::Reasoning) => "thinking",
            Some(TurnEventKind::Assistant) => "responding",
            _ => "waiting for the model",
        };
        format!("{verb}… {}", fmt_elapsed(secs))
    }

    /// Apply a pure editing/navigation key to the input line, shared by the
    /// idle input phase and the in-turn queue-entry path. Returns the submitted
    /// text on Enter (when non-empty); the caller decides whether to run it now
    /// or queue it. Phase-specific control keys (Ctrl-C/Ctrl-D/Esc) are handled
    /// by the caller, not here.
    fn edit_key(&mut self, key: &KeyEvent) -> Option<String> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            // Alt+Enter inserts a newline (multi-line prompt without pasting); so
            // does a trailing backslash, for terminals that can't send Alt+Enter.
            KeyCode::Enter if alt => self.input.insert('\n'),
            KeyCode::Enter if self.input.continue_line() => {}
            KeyCode::Enter => {
                let line = self.input.submit();
                if !line.trim().is_empty() {
                    return Some(line);
                }
            }
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Char(c) if !ctrl => self.input.insert(c),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Up => self.input.history_prev(),
            KeyCode::Down => self.input.history_next(),
            KeyCode::PageUp => self.scroll_up(5),
            KeyCode::PageDown => self.scroll_down(5),
            _ => {}
        }
        None
    }

    fn push(&mut self, line: Line<'static>) {
        self.transcript.push(line);
    }

    fn note_turn_completed_without_summary(&mut self) {
        match self.last_turn_event {
            Some(TurnEventKind::ToolCall | TurnEventKind::ToolResult) => {
                self.status = "stopped after tool output".to_string();
                self.last_turn_state = TurnState::Warning("stopped after tool output".to_string());
                self.last_error = Some("turn stopped after tool output".to_string());
                self.push(Line::styled(
                    "⚠ turn stopped after tool output without a final assistant response; try /retry",
                    Style::default().fg(Color::Yellow),
                ));
                self.record_model_issue();
            }
            _ => {
                self.status = "done · no usage reported".to_string();
                self.last_turn_state = TurnState::Done("no usage reported".to_string());
                self.push(Line::styled("✓ done · no usage reported", dim()));
            }
        }
        self.follow();
    }

    fn note_turn_failed(&mut self, error: &str) {
        self.status = "failed".to_string();
        self.last_turn_state = TurnState::Failed(error.to_string());
        self.last_error = Some(error.to_string());
        self.push(Line::styled(
            format!("✗ failed · {error}"),
            Style::default().fg(Color::Red),
        ));
        self.follow();
    }

    fn record_model_issue(&mut self) {
        let count = {
            let entry = self.model_issues.entry(self.model.clone()).or_insert(0);
            *entry += 1;
            *entry
        };
        // Note: don't touch `last_error` here — it holds the actual failure
        // reason set by the caller; the per-model count lives in `model_issues`
        // and surfaces via `/status` model health.
        if count == 1 {
            self.push(Line::styled(
                format!(
                    "⚠ {} returned an incomplete turn; it is now marked degraded in-session. Consider /model",
                    self.model
                ),
                Style::default().fg(Color::Yellow),
            ));
        } else if count >= 2 {
            self.push(Line::styled(
                format!(
                    "⚠ {} has had {count} reliability issue(s) this session and is degraded; consider /model",
                    self.model
                ),
                Style::default().fg(Color::Yellow),
            ));
        }
    }

    /// Re-pin the view to the latest output. Called on explicit user actions (a
    /// new turn, a command's output) — not on streaming appends, so a reader who
    /// scrolled up stays put.
    fn follow(&mut self) {
        self.following = true;
    }

    /// Push the cumulative-usage line from the live counters. Works mid-turn —
    /// when the agent itself is borrowed by the running turn — because it reads
    /// the mirrored `usage` rather than the agent.
    fn report_tokens(&mut self) {
        let (input, output) = self.usage;
        let mut line = format!(
            "cumulative: {input} in · {output} out · {} total",
            input + output
        );
        if let Some(pct) = self.context_pct() {
            line.push_str(&format!("  ·  context {pct}% full"));
        }
        self.push(Line::styled(line, dim()));
        self.follow();
    }

    fn report_status(&mut self, agent: &Agent) {
        let (input, output) = self.usage;
        let state = match &self.last_turn_state {
            TurnState::Idle => "idle".to_string(),
            TurnState::Running => "running".to_string(),
            TurnState::Done(s) => format!("done ({s})"),
            TurnState::Warning(s) => format!("warning ({s})"),
            TurnState::Failed(s) => format!("failed ({s})"),
            TurnState::Cancelled => "cancelled".to_string(),
        };
        let ctx = self
            .context_pct()
            .map(|p| format!("{p}%"))
            .unwrap_or_else(|| "unknown".to_string());
        let goal = agent.goal().unwrap_or("off");
        let verify = agent.verify_summary();
        let error = self.last_error.as_deref().unwrap_or("none");
        let model_issues = self.model_issues.get(&self.model).copied().unwrap_or(0);
        let model_health = if model_issues >= 2 {
            format!("degraded ({model_issues} issue(s))")
        } else if model_issues == 1 {
            "degraded (1 issue)".to_string()
        } else {
            "ok".to_string()
        };
        for line in [
            format!("status: {state}"),
            format!("provider/model: {} · {}", self.provider, self.model),
            format!(
                "context: {ctx}; usage: {input} in · {output} out · {} total",
                input + output
            ),
            format!("model health: {model_health}"),
            format!("goal: {goal}"),
            format!("verify: {verify}"),
            format!("last error: {error}"),
            format!(
                "startup notice: {}",
                self.startup_notice.as_deref().unwrap_or("none")
            ),
            format!(
                "queued: {}; checkpoints: {}",
                self.queue.len(),
                agent.checkpoint_count()
            ),
        ] {
            self.push(Line::styled(line, dim()));
        }
        self.follow();
    }

    fn write_debug_log(&mut self) {
        let path = std::path::Path::new(".hi-debug.log");
        let mut body = String::new();
        body.push_str("# hi debug log\n\n");
        body.push_str("## status\n");
        body.push_str(&format!(
            "provider: {}\nmodel: {}\n",
            self.provider, self.model
        ));
        body.push_str(&format!("status: {}\n\n", self.status));
        body.push_str(&format!(
            "last_error: {}\nwaiting_for: {:?}\nstartup_notice: {}\nlast_turn_file_edits: {}\n\n",
            self.last_error.as_deref().unwrap_or("none"),
            self.waiting_for,
            self.startup_notice.as_deref().unwrap_or("none"),
            self.last_turn_had_file_edits
        ));
        body.push_str("## events\n");
        for event in &self.event_log {
            body.push_str(event);
            body.push('\n');
        }
        body.push_str("\n## transcript\n");
        body.push_str(&self.transcript_text());
        match std::fs::write(path, body) {
            Ok(()) => self.push(Line::styled("wrote debug log: .hi-debug.log", dim())),
            Err(err) => self.push(Line::styled(
                format!("log failed: {err}"),
                Style::default().fg(Color::Yellow),
            )),
        }
        self.follow();
    }

    fn copy(&mut self, arg: &str) {
        let text = match arg.trim() {
            "all" | "transcript" => self.transcript_text(),
            _ => self.last_assistant.trim().to_string(),
        };
        if text.is_empty() {
            self.push(Line::styled("nothing to copy yet", dim()));
        } else {
            match copy_to_clipboard(&text) {
                Ok(()) => self.push(Line::styled(format!("copied {} chars", text.len()), dim())),
                Err(err) => self.push(Line::styled(
                    format!("copy failed: {err}"),
                    Style::default().fg(Color::Yellow),
                )),
            }
        }
        self.follow();
    }

    fn handle_goal(&mut self, agent: &mut Agent, arg: &str) {
        // Apply the change first, then describe the resulting state.
        match arg.trim() {
            "" => {} // no argument — just report the current goal
            "clear" | "off" | "none" => agent.set_goal(None),
            goal => agent.set_goal(Some(goal.to_string())),
        }
        let (msg, prominent) = goal_feedback(arg, agent.goal());
        // A set/clear is an applied change — show it plainly (green ✓), not dim,
        // so it's obvious it took effect. A bare `/goal` is just a read-out.
        let style = if prominent {
            Style::default().fg(Color::Green)
        } else {
            dim()
        };
        self.push(Line::styled(msg, style));
        self.follow();
    }

    fn transcript_text(&self) -> String {
        self.transcript
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn scroll_up(&mut self, n: u16) {
        self.scroll_by(-(n as i32));
    }

    fn scroll_down(&mut self, n: u16) {
        self.scroll_by(n as i32);
    }

    /// Move the viewport by `delta` wrapped lines (negative = toward older
    /// output). Re-pins to the bottom when scrolled all the way down; snapshots
    /// the line count when first leaving the bottom (for the "↓ N new" hint).
    /// Uses the metrics cached by the last render.
    fn scroll_by(&mut self, delta: i32) {
        let max = self.view_max_scroll as i32;
        let cur = if self.following {
            max
        } else {
            (self.scroll as i32).min(max)
        };
        let next = (cur + delta).clamp(0, max);
        if next >= max {
            self.following = true;
        } else {
            if self.following {
                self.total_when_unpinned = self.view_total;
            }
            self.following = false;
            self.scroll = next as u16;
        }
    }

    /// Commit the in-progress streamed line, if any.
    fn flush_pending(&mut self) {
        if let Some((style, markdown, text)) = self.pending.take() {
            let line = if markdown {
                markdown_line(&text, &mut self.code_lang)
            } else {
                Line::styled(text, style)
            };
            self.transcript.push(line);
        }
    }

    /// Append streamed text under `style`, committing complete lines. When
    /// `markdown` is set, committed lines are rendered with light markdown
    /// styling (headings, bullets, code fences, inline emphasis).
    fn stream(&mut self, style: Style, markdown: bool, chunk: &str) {
        // A style/kind change ends the current line.
        if let Some((prev, prev_md, _)) = &self.pending
            && (*prev != style || *prev_md != markdown)
        {
            self.flush_pending();
        }
        let (_, _, buf) = self
            .pending
            .get_or_insert_with(|| (style, markdown, String::new()));
        buf.push_str(chunk);
        while let Some(idx) = buf.find('\n') {
            let committed: String = buf[..idx].to_string();
            buf.drain(..=idx);
            let line = if markdown {
                markdown_line(&committed, &mut self.code_lang)
            } else {
                Line::styled(committed, style)
            };
            self.transcript.push(line);
        }
        // No follow() here: streaming must not yank a reader who scrolled up.
        // While following, the view already tracks the growing bottom.
    }

    fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Text(t) => {
                self.event_log
                    .push(format!("assistant_text {} chars", t.len()));
                self.last_turn_event = Some(TurnEventKind::Assistant);
                self.current_assistant.push_str(&t);
                self.stream(Style::default(), true, &t);
            }
            UiEvent::Reasoning(t) => {
                self.event_log.push(format!("reasoning {} chars", t.len()));
                self.last_turn_event = Some(TurnEventKind::Reasoning);
                self.stream(dim(), false, &t);
            }
            UiEvent::AssistantEnd => {
                self.event_log.push("assistant_end".to_string());
                self.last_turn_event = Some(TurnEventKind::AssistantEnd);
                self.flush_pending();
                if !self.current_assistant.trim().is_empty() {
                    self.last_assistant = self.current_assistant.trim().to_string();
                }
                self.current_assistant.clear();
                // Fences don't span messages; reset so a stray ``` can't bleed
                // code styling into the next response.
                self.code_lang = None;
            }
            UiEvent::ToolCall(name, args) => {
                let label = tool_label(&name, &args);
                self.event_log.push(format!("tool_call {label}"));
                self.last_turn_event = Some(TurnEventKind::ToolCall);
                if matches!(name.as_str(), "write" | "edit") {
                    self.last_turn_had_file_edits = true;
                }
                // Mark this tool as the active party so the working line can name
                // it with its own timer until the result lands.
                self.current_tool = Some(label.clone());
                self.current_tool_started = Some(Instant::now());
                self.flush_pending();
                self.push(Line::styled(
                    format!("⏺ {label}"),
                    Style::default().fg(Color::Cyan),
                ));
            }
            UiEvent::ToolResult(result) => {
                self.event_log
                    .push(format!("tool_result {} chars", result.len()));
                self.last_turn_event = Some(TurnEventKind::ToolResult);
                // The tool finished — back to the model being the active party.
                self.current_tool = None;
                self.current_tool_started = None;
                self.flush_pending();
                self.push_result(&result);
            }
            UiEvent::Status(s) => {
                self.event_log.push(format!("status {s}"));
                self.last_turn_event = Some(TurnEventKind::Status);
                self.flush_pending();
                self.push(Line::styled(s, Style::default().fg(Color::Blue)));
            }
            // Live counters only — no transcript line; the working/title bars read them.
            UiEvent::Usage {
                input,
                output,
                ctx_used,
                ctx_window,
            } => {
                self.event_log
                    .push(format!("usage {input} in {output} out"));
                self.last_turn_event = Some(TurnEventKind::Usage);
                self.usage = (input, output);
                self.context_used = ctx_used;
                self.context_window = ctx_window;
            }
            UiEvent::TurnEnd(summary) => {
                self.event_log.push(format!("turn_end {summary}"));
                self.last_turn_event = Some(TurnEventKind::TurnEnd);
                self.flush_pending();
                // Tokens/cost go to the status bar; a dim marker in the transcript
                // makes the end of a turn unmistakable (so a long run doesn't just
                // trail off with no clear "done").
                self.status = summary.trim_matches(['[', ']']).to_string();
                self.last_turn_state = TurnState::Done(self.status.clone());
                self.push(Line::styled(format!("✓ done · {}", self.status), dim()));
                // No follow(): respect a reader who scrolled up — the "↓ N new"
                // hint tells them the summary landed below.
            }
        }
    }

    /// Render a tool result, clipped to a handful of lines and indented.
    /// Preserves any ANSI colors (e.g. edit/write diffs); for *plain* unified
    /// diff output from a shell command (`git diff`, `diff -u`) — which CLIs
    /// emit without color when piped — adds diff coloring so it's readable.
    fn push_result(&mut self, result: &str) {
        const MAX: usize = 14;
        if result.trim().is_empty() {
            self.push(Line::styled("  (no output)", dim()));
            return;
        }
        let body: String = result.lines().take(MAX).collect::<Vec<_>>().join("\n");
        let lines: Vec<Line<'static>> = if !body.contains('\u{1b}') && looks_like_diff(result) {
            diff_lines(&body)
        } else {
            // ANSI (already-colored) or non-diff text: parse escapes as before.
            body.into_text()
                .unwrap_or_else(|_| Text::from(body.clone()))
                .lines
        };
        for mut line in lines {
            line.spans.insert(0, "  ".into());
            self.transcript.push(line);
        }
        let extra = result.lines().count().saturating_sub(MAX);
        if extra > 0 {
            self.push(Line::styled(format!("  … {extra} more lines"), dim()));
        }
    }

    fn handle_command(&mut self, agent: &mut Agent, command: Command, registry: &hi_ai::Registry) {
        match command {
            Command::Quit => {}
            Command::Help => {
                for line in command::help_text().lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            Command::Tokens => {
                // Sync the live counter from the authoritative totals, then show it.
                let t = agent.totals();
                self.usage = (t.input_tokens, t.output_tokens);
                self.report_tokens();
            }
            Command::Status => self.report_status(agent),
            Command::Log => self.write_debug_log(),
            Command::Model(id) => {
                if id.is_empty() {
                    // Open the interactive picker (filter + arrow-select).
                    let current = self.model.clone();
                    let tags = self.served_tags();
                    self.picker = Some(ModelPicker::new(registry.model_ids(), &current, tags));
                } else {
                    let health = self.apply_model(agent, registry, &id);
                    self.push(Line::styled(format!("model set to {id}"), dim()));
                    if let Some(h) = health {
                        self.warn_degraded(&id, &h);
                    }
                }
            }
            Command::Clear => {
                agent.clear_history();
                self.transcript.clear();
                self.pending = None;
                self.code_lang = None;
                self.current_assistant.clear();
                self.last_assistant.clear();
                self.status.clear();
                self.last_turn_state = TurnState::Idle;
                self.push(Line::styled("conversation cleared", dim()));
            }
            Command::Verify(arg) => {
                let msg = match arg.trim() {
                    "" if agent.verify_is_on() => format!("verify: {}", agent.verify_summary()),
                    "" => "verify: off (set one with /verify <cmd>)".to_string(),
                    "off" | "none" | "clear" | "disable" => {
                        agent.set_verify_command(None);
                        "verification disabled".to_string()
                    }
                    cmd => {
                        agent.set_verify_command(Some(cmd.to_string()));
                        format!(
                            "verification on: `{cmd}` — runs after each turn, iterates on failure"
                        )
                    }
                };
                self.push(Line::styled(msg, dim()));
            }
            Command::Diff => {
                let out = hi_tools::working_tree_diff();
                let text = out.into_text().unwrap_or_else(|_| Text::from(out.clone()));
                for line in text.lines {
                    self.push(line);
                }
            }
            Command::Copy(arg) => self.copy(&arg),
            Command::Goal(arg) => self.handle_goal(agent, &arg),
            // Handled in the event loop (async / runs a turn); never reach here.
            Command::Compact(_) | Command::Retry | Command::Undo | Command::Init => {}
            Command::Unknown(name) => {
                self.push(Line::styled(
                    format!("unknown command /{name}; try /help"),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    /// The editable input rendered as one or more lines (the prompt may hold a
    /// pasted multi-line block), plus the cursor's (row, col) within them. Long
    /// inputs show only their last [`MAX_INPUT_ROWS`] lines with a "… more above"
    /// note so they can't swallow the screen.
    fn input_view(&self) -> (Vec<Line<'static>>, u16, u16) {
        const MAX_INPUT_ROWS: usize = 10;
        let text = self.input.text();
        let before: String = text.chars().take(self.input.cursor()).collect();
        let cursor_row_full = before.matches('\n').count();
        let cursor_col = before.chars().rev().take_while(|&c| c != '\n').count() as u16;

        let all: Vec<&str> = text.split('\n').collect();
        let truncated = all.len() > MAX_INPUT_ROWS;
        let start = if truncated {
            all.len() - MAX_INPUT_ROWS
        } else {
            0
        };

        let mut lines: Vec<Line<'static>> = Vec::new();
        if truncated {
            lines.push(Line::styled(
                format!("  ⋮ {start} more line(s) above"),
                dim(),
            ));
        }
        for (i, seg) in all[start..].iter().enumerate() {
            let prefix = if i == 0 && !truncated { "› " } else { "  " };
            lines.push(Line::from(format!("{prefix}{seg}")));
        }
        let cursor_row = u16::from(truncated) + cursor_row_full.saturating_sub(start) as u16;
        (lines, cursor_row, 2 + cursor_col)
    }

    fn render(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        // The input box grows to fit a spinner status line (while working), the
        // (possibly multi-line) input, and up to three queued commands.
        let status_lines = 1usize;
        let queued_shown = self.queue.len().min(3);
        let queue_extra = usize::from(self.queue.len() > 3);
        let (input_lines, cursor_row, cursor_col) = self.input_view();
        let completion_rows = self.completion_items().len();
        let input_h = if self.fetching.is_some() {
            3
        } else if let Some(p) = &self.picker {
            // filter line + visible model rows + borders, bounded by the screen.
            let rows = p.matches.len().clamp(1, PICKER_ROWS) as u16;
            (rows + 3).min(area.height.saturating_sub(3))
        } else {
            (status_lines + completion_rows + input_lines.len() + queued_shown + queue_extra + 2)
                as u16
        };
        let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(input_h)]).split(area);

        // --- Transcript ---
        let title = format!(" hi · {} · {} ", self.provider, self.model);
        // Right-aligned: a persistent context-fill gauge, then the last status.
        let mut info_parts: Vec<String> = Vec::new();
        if let Some(pct) = self.context_pct() {
            info_parts.push(format!("{pct}% ctx"));
        }
        if !self.status.is_empty() {
            info_parts.push(self.status.clone());
        }
        let info = if info_parts.is_empty() {
            String::new()
        } else {
            format!(" {} ", info_parts.join(" · "))
        };
        let mut lines = self.transcript.clone();
        if let Some((style, _markdown, text)) = &self.pending {
            // The in-progress line shows literally; markdown styling is applied
            // once the line is committed on its newline.
            lines.push(Line::styled(text.clone(), *style));
        }
        let inner_w = rows[0].width.saturating_sub(2);
        let inner_h = rows[0].height.saturating_sub(2);
        let total = wrapped_height(&lines, inner_w);
        let max_scroll = total.saturating_sub(inner_h);
        // Cache the geometry so scroll events (which fire outside render) can clamp
        // and detect the bottom.
        self.view_max_scroll = max_scroll;
        self.view_total = total;
        // Pinned to the bottom while following; otherwise hold the user's absolute
        // offset, re-pinning if the content shrank back to within one screen.
        let scroll = if self.following || self.scroll >= max_scroll {
            self.following = true;
            max_scroll
        } else {
            self.scroll
        };

        let mut block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(dim())
            .title(title)
            .title_top(Line::from(info).right_aligned());
        // While scrolled up, a bottom-right hint shows how much is below — new
        // lines that arrived since you left the bottom, else how far down it is.
        if !self.following {
            let new = total.saturating_sub(self.total_when_unpinned);
            let label = if new > 0 {
                format!(" ↓ {new} new ")
            } else {
                format!(" ↓ {} below ", max_scroll.saturating_sub(scroll))
            };
            block = block.title_bottom(
                Line::from(Span::styled(
                    label,
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ))
                .right_aligned(),
            );
        }
        let para = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block)
            .scroll((scroll, 0));
        frame.render_widget(para, rows[0]);

        // --- Bottom region: a fetch spinner, the model picker, or the input bar. ---
        if let Some(started) = self.fetching {
            let frame_ch = SPINNER[self.spinner % SPINNER.len()];
            let elapsed = fmt_elapsed(started.elapsed().as_secs());
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan));
            let body = Line::from(vec![
                Span::styled(
                    format!("{frame_ch} fetching models from {}… {elapsed}", self.provider),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Esc to cancel", dim()),
            ]);
            frame.render_widget(Paragraph::new(body).block(block), rows[1]);
        } else if let Some(p) = &self.picker {
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" select a model ")
                .title_top(
                    Line::from(format!(" {}/{} ", p.selected + 1, p.matches.len().max(1)))
                        .right_aligned(),
                );
            let mut plines: Vec<Line> = vec![Line::from(vec![
                Span::raw(format!("filter: {}", p.filter)),
                Span::styled(
                    "   ↑↓ move · type to filter · Enter select · Esc cancel",
                    dim(),
                ),
            ])];
            let (_, visible) = p.visible();
            if visible.is_empty() {
                plines.push(Line::styled("  (no matches)".to_string(), dim()));
            }
            for (id, selected) in visible {
                let mut tag = String::new();
                if id == p.current {
                    tag.push_str(" (current)");
                }
                if let Some(health) = p.tags.get(id) {
                    tag.push_str(&format!(" [{health}]"));
                }
                if selected {
                    plines.push(Line::from(vec![
                        Span::styled(
                            format!("▶ {id}"),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(tag, dim()),
                    ]));
                } else {
                    plines.push(Line::from(vec![
                        Span::raw(format!("  {id}")),
                        Span::styled(tag, dim()),
                    ]));
                }
            }
            frame.render_widget(Paragraph::new(plines).block(block), rows[1]);
            // Cursor on the filter line, just after "filter: <text>".
            let cx = rows[1].x + 1 + 8 + p.filter.chars().count() as u16;
            frame.set_cursor_position((cx.min(rows[1].right().saturating_sub(2)), rows[1].y + 1));
        } else {
            // The border turns cyan and the top inner line becomes a bold
            // spinner + elapsed seconds while a turn runs; the prompt stays
            // editable so you can type the next command (it queues below).
            let input_block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(if self.working {
                    Style::default().fg(Color::Cyan)
                } else {
                    dim()
                });

            let mut ilines: Vec<Line> = Vec::new();
            if let Some(notice) = &self.startup_notice {
                ilines.push(Line::styled(
                    notice.clone(),
                    Style::default().fg(Color::Yellow),
                ));
            }
            if self.working {
                let frame_ch = SPINNER[self.spinner % SPINNER.len()];
                let (input, output) = self.usage;
                // Show the running token total once the first round reports it.
                let mut stats = String::new();
                if input + output > 0 {
                    stats.push_str(&format!(" · ↑{} ↓{}", fmt_count(input), fmt_count(output)));
                }
                if let Some(pct) = self.context_pct() {
                    stats.push_str(&format!(" · {pct}% ctx"));
                }
                // The activity lead (named tool + timer, or thinking/responding)
                // replaces the old coarse "working… · last: <event>"; its own timer
                // and the watchdog notices cover the "is it stalled?" signal.
                ilines.push(Line::from(vec![
                    Span::styled(
                        format!("{frame_ch} {}{stats}", self.activity_line()),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("   Ctrl-C to interrupt", dim()),
                ]));
            } else {
                let line = match &self.last_turn_state {
                    TurnState::Idle => "ready".to_string(),
                    TurnState::Running => "working".to_string(),
                    TurnState::Done(s) => format!("ready · last: done ({s})"),
                    TurnState::Warning(s) => format!("ready · last: warning ({s})"),
                    // Show the failure reason inline so you don't have to scroll
                    // the transcript to learn what went wrong.
                    TurnState::Failed(s) => {
                        format!(
                            "ready · last: failed — {} · /retry to rerun",
                            clip_reason(s)
                        )
                    }
                    TurnState::Cancelled => "ready · last: cancelled".to_string(),
                };
                ilines.push(Line::styled(line, dim()));
            }
            // The `/`-command completion menu sits just above the input line. Rows
            // are command names (`/compact`) or, past the name, argument values
            // (`hybrid`, `full`, `elide`).
            let items = self.completion_items();
            let selected = self.completion.as_ref().map(|c| c.selected).unwrap_or(0);
            let label_w = items.iter().map(|i| i.label.len()).max().unwrap_or(0);
            for (i, item) in items.iter().enumerate() {
                let label = format!("{:<width$}", item.label, width = label_w);
                if i == selected {
                    ilines.push(Line::from(vec![
                        Span::styled(
                            format!("▶ {label}"),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                } else {
                    ilines.push(Line::from(vec![
                        Span::raw(format!("  {label}")),
                        Span::styled(format!("  {}", item.help), dim()),
                    ]));
                }
            }
            ilines.extend(input_lines);
            for q in self.queue.iter().take(3) {
                ilines.push(Line::styled(format!("⏳ {q}"), dim()));
            }
            if self.queue.len() > 3 {
                ilines.push(Line::styled(
                    format!("   … +{} more queued", self.queue.len() - 3),
                    dim(),
                ));
            }
            frame.render_widget(Paragraph::new(ilines).block(input_block), rows[1]);

            // Cursor sits within the editable input — below the optional startup
            // notice, the status line, and the completion menu.
            let above = usize::from(self.startup_notice.is_some())
                + status_lines
                + self.completion_items().len();
            let cx = rows[1].x + 1 + cursor_col;
            let cy = rows[1].y + 1 + above as u16 + cursor_row;
            frame.set_cursor_position((
                cx.min(rows[1].right().saturating_sub(2)),
                cy.min(rows[1].bottom().saturating_sub(2)),
            ));
        }
    }
}

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

/// The `/goal` feedback line, and whether it's a prominent applied-change
/// confirmation (set/clear) versus a dim status read-out (a bare `/goal`).
/// `goal` is the agent's goal *after* the action, so a set echoes exactly what's
/// stored. Pure, so the wording is unit-testable without an agent.
fn goal_feedback(arg: &str, goal: Option<&str>) -> (String, bool) {
    match arg.trim() {
        "" => match goal {
            Some(g) => (format!("goal: {g}"), false),
            None => ("goal: off (set one with /goal <text>)".to_string(), false),
        },
        "clear" | "off" | "none" => ("✓ goal cleared".to_string(), true),
        _ => (
            format!(
                "✓ goal set — steers every turn until cleared: \"{}\"",
                goal.unwrap_or_default()
            ),
            true,
        ),
    }
}

fn line_text(line: &Line) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

/// Nudge the terminal that a turn finished: the BEL (which most terminals turn
/// into a dock bounce / taskbar flash / audible ping when unfocused) plus an
/// OSC 9 desktop notification for terminals that support it (iTerm2, WezTerm, …).
/// Written straight to the tty; both are non-printing, so they don't disturb the
/// ratatui frame.
fn notify_done() {
    let mut out = io::stdout().lock();
    let _ = write!(out, "\x07\x1b]9;hi — turn complete\x07");
    let _ = out.flush();
}

fn copy_to_clipboard(text: &str) -> io::Result<()> {
    let encoded = base64_encode(text.as_bytes());
    let mut out = io::stdout().lock();
    write!(out, "\x1b]52;c;{encoded}\x07")?;
    out.flush()
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Whether `s` looks like unified-diff output (a hunk header, a git header, or
/// a `---`/`+++` file-header pair) — so we can colorize plain `git diff` /
/// `diff -u` output the model runs via the shell.
fn looks_like_diff(s: &str) -> bool {
    let (mut minus, mut plus) = (false, false);
    for line in s.lines() {
        if line.starts_with("@@") || line.starts_with("diff --git ") {
            return true;
        }
        minus |= line.starts_with("--- ");
        plus |= line.starts_with("+++ ");
    }
    minus && plus
}

/// Render a unified diff with coloring and a new-file line-number gutter:
/// additions green, removals red, hunk headers cyan, file headers bold, context
/// muted. The line number (tracked from each `@@` header) is shown for context
/// and added lines; removed lines and headers get a blank gutter.
fn diff_lines(body: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut new_line: Option<u32> = None;
    for line in body.lines() {
        let (style, gutter, advance) = if line.starts_with("+++") || line.starts_with("---") {
            (Style::default().add_modifier(Modifier::BOLD), None, false)
        } else if line.starts_with("@@") {
            new_line = parse_hunk_new_start(line);
            (Style::default().fg(Color::Cyan), None, false)
        } else if line.starts_with('+') {
            (Style::default().fg(Color::Green), new_line, true)
        } else if line.starts_with('-') {
            (Style::default().fg(Color::Red), None, false)
        } else {
            (dim(), new_line, true)
        };
        let num = match gutter {
            Some(n) => format!("{n:>4} "),
            None => "     ".to_string(),
        };
        out.push(Line::from(vec![
            Span::styled(num, dim()),
            Span::styled(line.to_string(), style),
        ]));
        if advance && let Some(n) = new_line.as_mut() {
            *n += 1;
        }
    }
    out
}

/// Parse the new-file start line from a unified-diff hunk header
/// `@@ -old,n +new,m @@` → `new`.
fn parse_hunk_new_start(header: &str) -> Option<u32> {
    let plus = header.split('+').nth(1)?;
    let num: String = plus.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

/// Light syntax highlighting for one line of fenced code: whole-line comments
/// (by the fence language) are dimmed and string literals are greened; the rest
/// stays in the default color. Deliberately minimal — no keyword tables — so it
/// reads as intentional on every language and never mis-colors an unknown one.
fn highlight_code(line: &str, lang: &str) -> Vec<Span<'static>> {
    if let Some(marker) = line_comment_marker(lang)
        && line.trim_start().starts_with(marker)
    {
        return vec![Span::styled(line.to_string(), dim())];
    }
    highlight_strings(line)
}

/// The line-comment marker for a fence language, if we know it. Unknown
/// languages return `None` (no comment dimming) rather than guess.
fn line_comment_marker(lang: &str) -> Option<&'static str> {
    match lang.to_lowercase().as_str() {
        "rust" | "rs" | "c" | "cpp" | "c++" | "h" | "hpp" | "js" | "javascript" | "jsx" | "ts"
        | "typescript" | "tsx" | "go" | "java" | "kotlin" | "kt" | "swift" | "scala" | "zig"
        | "dart" | "php" => Some("//"),
        "python" | "py" | "sh" | "bash" | "shell" | "zsh" | "fish" | "ruby" | "rb" | "yaml"
        | "yml" | "toml" | "ini" | "conf" | "r" | "perl" | "pl" | "makefile" | "make"
        | "dockerfile" | "elixir" | "ex" => Some("#"),
        "sql" | "lua" | "haskell" | "hs" => Some("--"),
        _ => None,
    }
}

/// Split a code line into spans, greening `"…"` / `'…'` string literals (honoring
/// `\` escapes) and leaving everything else in the default style.
fn highlight_strings(line: &str) -> Vec<Span<'static>> {
    let chars: Vec<char> = line.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut plain = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' || c == '\'' {
            // Find the matching close on this line, skipping escaped quotes.
            let mut j = i + 1;
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == '\\' {
                    j += 2;
                    continue;
                }
                if chars[j] == c {
                    closed = true;
                    break;
                }
                j += 1;
            }
            if closed {
                if !plain.is_empty() {
                    spans.push(Span::raw(std::mem::take(&mut plain)));
                }
                let s: String = chars[i..=j].iter().collect();
                spans.push(Span::styled(s, Style::default().fg(Color::Green)));
                i = j + 1;
                continue;
            }
        }
        plain.push(c);
        i += 1;
    }
    if !plain.is_empty() {
        spans.push(Span::raw(plain));
    }
    if spans.is_empty() {
        spans.push(Span::raw(String::new()));
    }
    spans
}

/// What the completion menu should offer for `input`, or `None` to close it:
/// the command name while it's being typed (`/`, `/co`), or — once past the name
/// — the argument of a command that has enumerable values (`/compact `,
/// `/compact hy`). A freeform-argument command (`/model <id>`) or a second arg
/// token closes the menu, as does any non-slash input.
/// The one command whose argument values come from live state (the model
/// catalog) rather than the static table.
const MODEL_CMD: &str = "model";
/// Cap on inline `/model` id completions, so a large catalog can't flood the menu.
const MODEL_COMPLETION_MAX: usize = 8;

fn completion_context(input: &str) -> Option<CompletionContext> {
    let rest = input.strip_prefix('/')?;
    match rest.split_once(char::is_whitespace) {
        // No space yet → still choosing the command name.
        None => Some(CompletionContext::Command(rest.to_lowercase())),
        // Past the name, on the first argument token.
        Some((name, arg)) => {
            if arg.contains(char::is_whitespace) {
                return None; // a second token — past the single argument
            }
            let spec = command::COMMANDS
                .iter()
                .find(|c| c.name.eq_ignore_ascii_case(name))?;
            let prefix = arg.to_lowercase();
            if !spec.arg_values.is_empty() {
                // A static enumerable set (compact, copy, verify, goal).
                if spec.arg_values.iter().any(|(v, _)| *v == prefix) {
                    return None; // a full valid value is typed — nothing left to pick
                }
                return Some(CompletionContext::Arg { cmd: spec.name, prefix });
            }
            if spec.name == MODEL_CMD {
                // Model ids are dynamic — the catalog is filtered at render time,
                // so emptiness is resolved there, not here.
                return Some(CompletionContext::Arg { cmd: spec.name, prefix });
            }
            None // freeform or no argument — nothing to enumerate
        }
    }
}

/// Resolve a completion context to the menu rows it offers.
fn completion_items_for(ctx: &CompletionContext) -> Vec<CompletionItem> {
    match ctx {
        CompletionContext::Command(prefix) => command::matching(prefix)
            .into_iter()
            .map(|spec| {
                let takes_args = spec.takes_args();
                CompletionItem {
                    label: format!("/{}", spec.name),
                    help: spec.help.to_string(),
                    insert: if takes_args {
                        format!("/{} ", spec.name)
                    } else {
                        format!("/{}", spec.name)
                    },
                    submit_on_enter: !takes_args,
                }
            })
            .collect(),
        CompletionContext::Arg { cmd, prefix } => command::arg_matching(cmd, prefix)
            .into_iter()
            .map(|(value, hint)| CompletionItem {
                label: value.to_string(),
                help: hint.to_string(),
                insert: format!("/{cmd} {value}"),
                submit_on_enter: true,
            })
            .collect(),
    }
}

/// One-line, length-capped form of an error message for the status bar:
/// whitespace/newlines collapsed, clipped with an ellipsis.
fn clip_reason(s: &str) -> String {
    let one_line = s.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 60;
    if one_line.chars().count() > MAX {
        format!("{}…", one_line.chars().take(MAX).collect::<String>())
    } else {
        one_line
    }
}

/// Compact token count for the working line: `1234` → `1.2k`, `45000` → `45k`.
/// The live working line and the settled usage summary share one humanizer (in
/// `hi-agent`), so the same count never renders two different ways.
fn fmt_count(n: u64) -> String {
    hi_agent::humanize_count(n)
}

/// Format an elapsed-seconds count compactly: `45s`, `14m 28s`, `1h 02m`.
fn fmt_elapsed(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h {m:02}m")
    } else if m > 0 {
        format!("{m}m {s:02}s")
    } else {
        format!("{s}s")
    }
}

/// The style for code — inline spans and fenced blocks.
fn code_style() -> Style {
    Style::default().fg(Color::Cyan)
}

/// Render one committed line of assistant markdown into a styled [`Line`].
/// Block-level constructs (headings, lists, fences, rules, quotes) are detected
/// per line; `code_lang` carries the ``` fence state across calls (`Some(lang)`
/// while inside a fence) so code interiors are highlighted for that language.
/// Anything else gets inline emphasis/code styling.
fn markdown_line(text: &str, code_lang: &mut Option<String>) -> Line<'static> {
    let trimmed = text.trim_start();

    // Fenced code: ``` toggles the block; the fence line becomes a dim gutter
    // (with the language as a caption when opening).
    if trimmed.starts_with("```") {
        let lang = trimmed.trim_start_matches('`').trim();
        let caption = if code_lang.is_none() { lang } else { "" };
        *code_lang = if code_lang.is_none() {
            Some(lang.to_string())
        } else {
            None
        };
        return Line::from(vec![
            Span::styled("▏ ", dim()),
            Span::styled(caption.to_string(), dim().add_modifier(Modifier::ITALIC)),
        ]);
    }
    if let Some(lang) = code_lang.as_deref() {
        let mut spans = vec![Span::styled("▏ ", dim())];
        spans.extend(highlight_code(text, lang));
        return Line::from(spans);
    }

    // Horizontal rule.
    if is_hr(trimmed) {
        return Line::styled("─".repeat(40), dim());
    }

    // Headings: # … ###### → bold, markers stripped.
    if let Some(rest) = heading_text(trimmed) {
        return Line::from(inline_spans(
            rest,
            Style::default().add_modifier(Modifier::BOLD),
        ));
    }

    // Blockquote.
    if let Some(rest) = trimmed
        .strip_prefix("> ")
        .or_else(|| trimmed.strip_prefix('>'))
    {
        let mut spans = vec![Span::styled("▏ ", dim())];
        spans.extend(inline_spans(rest, dim()));
        return Line::from(spans);
    }

    // List items keep their original indentation.
    let indent = &text[..text.len() - trimmed.len()];
    if let Some(rest) = bullet_text(trimmed) {
        let mut spans = vec![Span::raw(format!("{indent}• "))];
        spans.extend(inline_spans(rest, Style::default()));
        return Line::from(spans);
    }
    if let Some((num, rest)) = numbered_text(trimmed) {
        let mut spans = vec![Span::styled(
            format!("{indent}{num}. "),
            Style::default().add_modifier(Modifier::BOLD),
        )];
        spans.extend(inline_spans(rest, Style::default()));
        return Line::from(spans);
    }

    // Plain paragraph (keep leading whitespace) with inline formatting.
    Line::from(inline_spans(text, Style::default()))
}

/// `---`, `***`, or `___` (3+ of one char) — a horizontal rule.
fn is_hr(s: &str) -> bool {
    let s = s.trim_end();
    s.len() >= 3 && ['-', '*', '_'].iter().any(|&m| s.chars().all(|c| c == m))
}

/// Strip a leading `#`..`###### `, returning the heading text.
fn heading_text(s: &str) -> Option<&str> {
    let hashes = s.len() - s.trim_start_matches('#').len();
    if (1..=6).contains(&hashes) {
        return s[hashes..].strip_prefix(' ').map(str::trim_end);
    }
    None
}

/// Strip a leading `- `, `* `, or `+ ` bullet marker.
fn bullet_text(s: &str) -> Option<&str> {
    ['-', '*', '+']
        .iter()
        .find_map(|&m| s.strip_prefix(m)?.strip_prefix(' '))
}

/// Split a leading `N. ` / `N) ` ordered-list marker into (number, rest).
fn numbered_text(s: &str) -> Option<(&str, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit())?;
    if end == 0 {
        return None;
    }
    let rest = s[end..]
        .strip_prefix(". ")
        .or_else(|| s[end..].strip_prefix(") "))?;
    Some((&s[..end], rest))
}

/// Parse inline `**bold**`, `*italic*`/`_italic_`, and `` `code` `` into styled
/// spans over `base`. Unmatched markers fall through as literal text.
fn inline_spans(text: &str, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut plain = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // `code`
        if c == '`'
            && let Some(close) = find_char(&chars, i + 1, '`')
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(slice(&chars, i + 1, close), code_style()));
            i = close + 1;
            continue;
        }
        // **bold**
        if c == '*'
            && chars.get(i + 1) == Some(&'*')
            && let Some(close) = find_double_star(&chars, i + 2)
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(
                slice(&chars, i + 2, close),
                base.add_modifier(Modifier::BOLD),
            ));
            i = close + 2;
            continue;
        }
        // *italic* (not ** and not an empty/space-led run)
        if c == '*'
            && chars.get(i + 1) != Some(&'*')
            && chars.get(i + 1) != Some(&' ')
            && let Some(close) = find_char(&chars, i + 1, '*')
            && close > i + 1
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(
                slice(&chars, i + 1, close),
                base.add_modifier(Modifier::ITALIC),
            ));
            i = close + 1;
            continue;
        }
        // _italic_ — word-boundary guarded so snake_case is left alone.
        if c == '_'
            && (i == 0 || !chars[i - 1].is_alphanumeric())
            && let Some(close) = find_char(&chars, i + 1, '_')
            && close > i + 1
            && chars.get(close + 1).is_none_or(|c| !c.is_alphanumeric())
        {
            flush_plain(&mut spans, &mut plain, base);
            spans.push(Span::styled(
                slice(&chars, i + 1, close),
                base.add_modifier(Modifier::ITALIC),
            ));
            i = close + 1;
            continue;
        }
        plain.push(c);
        i += 1;
    }
    flush_plain(&mut spans, &mut plain, base);
    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base));
    }
    spans
}

fn slice(chars: &[char], from: usize, to: usize) -> String {
    chars[from..to].iter().collect()
}

fn flush_plain(spans: &mut Vec<Span<'static>>, plain: &mut String, base: Style) {
    if !plain.is_empty() {
        spans.push(Span::styled(std::mem::take(plain), base));
    }
}

fn find_char(chars: &[char], from: usize, target: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == target)
}

fn find_double_star(chars: &[char], from: usize) -> Option<usize> {
    (from..chars.len().saturating_sub(1)).find(|&j| chars[j] == '*' && chars[j + 1] == '*')
}

/// Approximate the number of terminal rows `lines` occupy when wrapped to
/// `width` — used to keep the transcript scrolled to the bottom.
fn wrapped_height(lines: &[Line], width: u16) -> u16 {
    if width == 0 {
        return lines.len() as u16;
    }
    let width = width as usize;
    lines
        .iter()
        .map(|line| {
            let len: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
            if len == 0 {
                1
            } else {
                len.div_ceil(width) as u16
            }
        })
        .sum()
}

/// Interactive `/model` picker: a filterable, arrow-navigable list of model ids.
struct ModelPicker {
    all: Vec<String>,
    /// The model in use when the picker opened — pre-selected and marked.
    current: String,
    /// Health label per id (e.g. "degraded"), when the endpoint reported one.
    tags: HashMap<String, String>,
    filter: String,
    /// Indices into `all` matching the current filter.
    matches: Vec<usize>,
    /// Index into `matches` of the highlighted row.
    selected: usize,
}

impl ModelPicker {
    fn new(all: Vec<String>, current: &str, tags: HashMap<String, String>) -> Self {
        let matches: Vec<usize> = (0..all.len()).collect();
        // Open with the current model highlighted (and scrolled into view).
        let selected = all.iter().position(|id| id == current).unwrap_or(0);
        Self {
            all,
            current: current.to_string(),
            tags,
            filter: String::new(),
            matches,
            selected,
        }
    }

    /// Recompute matches (case-insensitive substring) after the filter changes.
    fn refilter(&mut self) {
        let needle = self.filter.to_lowercase();
        self.matches = self
            .all
            .iter()
            .enumerate()
            .filter(|(_, id)| needle.is_empty() || id.to_lowercase().contains(&needle))
            .map(|(i, _)| i)
            .collect();
        self.selected = 0;
    }

    fn insert(&mut self, c: char) {
        self.filter.push(c);
        self.refilter();
    }
    fn backspace(&mut self) {
        self.filter.pop();
        self.refilter();
    }
    fn down(&mut self) {
        if self.selected + 1 < self.matches.len() {
            self.selected += 1;
        }
    }
    fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }
    fn page_down(&mut self) {
        self.selected = (self.selected + PICKER_ROWS).min(self.matches.len().saturating_sub(1));
    }
    fn page_up(&mut self) {
        self.selected = self.selected.saturating_sub(PICKER_ROWS);
    }
    fn current(&self) -> Option<&str> {
        self.matches
            .get(self.selected)
            .map(|&i| self.all[i].as_str())
    }

    /// The visible window of (id, is_selected) rows, scrolled to keep the
    /// selection in view.
    fn visible(&self) -> (usize, Vec<(&str, bool)>) {
        let offset = if self.selected >= PICKER_ROWS {
            self.selected + 1 - PICKER_ROWS
        } else {
            0
        };
        let end = (offset + PICKER_ROWS).min(self.matches.len());
        let rows = (offset..end)
            .map(|vi| (self.all[self.matches[vi]].as_str(), vi == self.selected))
            .collect();
        (offset, rows)
    }
}

/// Terminal-free input line: text + cursor + history. Unit-tested below.
#[derive(Default)]
struct InputLine {
    chars: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
}

impl InputLine {
    fn text(&self) -> String {
        self.chars.iter().collect()
    }
    fn cursor(&self) -> usize {
        self.cursor
    }
    fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }
    fn insert(&mut self, c: char) {
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
    }
    /// If the character just before the cursor is a backslash, replace it with a
    /// newline and report `true` — so a line ending in `\` continues instead of
    /// submitting (a universal fallback for terminals without Alt+Enter).
    fn continue_line(&mut self) -> bool {
        if self.cursor > 0 && self.chars[self.cursor - 1] == '\\' {
            self.chars[self.cursor - 1] = '\n';
            true
        } else {
            false
        }
    }
    /// Insert a (possibly multi-line) string at the cursor — used for pastes.
    /// Line endings are normalized to `\n` so the text submits as one prompt.
    fn insert_str(&mut self, s: &str) {
        let normalized = s.replace("\r\n", "\n").replace('\r', "\n");
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();
        self.chars.splice(self.cursor..self.cursor, chars);
        self.cursor += n;
        self.history_pos = None;
    }
    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.chars.remove(self.cursor - 1);
            self.cursor -= 1;
        }
    }
    fn kill_to_start(&mut self) {
        self.chars.drain(..self.cursor);
        self.cursor = 0;
    }
    fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }
    fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
    }
    fn home(&mut self) {
        self.cursor = 0;
    }
    fn end(&mut self) {
        self.cursor = self.chars.len();
    }
    fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.history_pos = None;
    }
    fn submit(&mut self) -> String {
        let line = self.text();
        self.clear();
        if !line.trim().is_empty() && self.history.last() != Some(&line) {
            self.history.push(line.clone());
        }
        line
    }
    fn set(&mut self, text: &str) {
        self.chars = text.chars().collect();
        self.cursor = self.chars.len();
    }
    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let pos = match self.history_pos {
            Some(0) => 0,
            Some(p) => p - 1,
            None => self.history.len() - 1,
        };
        self.history_pos = Some(pos);
        self.set(&self.history[pos].clone());
    }
    fn history_next(&mut self) {
        match self.history_pos {
            Some(p) if p + 1 < self.history.len() => {
                self.history_pos = Some(p + 1);
                self.set(&self.history[p + 1].clone());
            }
            Some(_) => {
                self.history_pos = None;
                self.set("");
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn dump(term: &Terminal<TestBackend>) -> String {
        let buf = term.backend().buffer();
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn sticky_scroll_unpins_on_scroll_up_and_repins_at_bottom() {
        let mut app = App::new("openai", "gpt-4o");
        // Simulate what render() caches for a transcript taller than the viewport.
        app.view_max_scroll = 100;
        app.view_total = 120;
        assert!(app.following, "starts pinned to the bottom");

        // Scrolling up unpins, holds an absolute offset, and snapshots the count.
        app.scroll_up(10);
        assert!(!app.following, "scroll up unpins");
        assert_eq!(app.scroll, 90, "offset = max_scroll - 10");
        assert_eq!(app.total_when_unpinned, 120);

        // Streaming output below must NOT yank a scrolled-up reader back down.
        app.apply(UiEvent::Text("a fresh streamed line\n".into()));
        assert!(!app.following, "new output leaves the scrolled-up reader put");

        // Scrolling back past the bottom re-pins so output follows again.
        app.scroll_down(1000);
        assert!(app.following, "reaching the bottom re-pins");
    }

    #[test]
    fn working_line_names_the_inflight_tool_and_model_phase() {
        let mut app = App::new("openai", "gpt-4o");
        app.set_working(true);
        // Model phase: reasoning then text stream distinctly.
        app.apply(UiEvent::Reasoning("hmm".into()));
        assert!(
            app.activity_line().starts_with("thinking…"),
            "{}",
            app.activity_line()
        );
        app.apply(UiEvent::Text("here".into()));
        assert!(
            app.activity_line().starts_with("responding…"),
            "{}",
            app.activity_line()
        );
        // A tool starts → the line names it (with its own timer)…
        app.apply(UiEvent::ToolCall(
            "bash".into(),
            "{\"command\":\"cargo test\"}".into(),
        ));
        assert!(
            app.activity_line().starts_with("running bash cargo test"),
            "{}",
            app.activity_line()
        );
        // …and clears back to the model once the result lands.
        app.apply(UiEvent::ToolResult("ok".into()));
        assert!(
            app.activity_line().starts_with("waiting for the model"),
            "{}",
            app.activity_line()
        );
    }

    #[test]
    fn renders_tool_call_diff_and_spinner() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::ToolCall(
            "edit".into(),
            "{\"path\":\"src/cli.rs\",\"old_string\":\"a\",\"new_string\":\"b\"}".into(),
        ));
        // ANSI-colored diff line (from the edit tool) must render as text.
        app.apply(UiEvent::ToolResult(
            "\u{1b}[32m+ pub json: bool\u{1b}[0m".into(),
        ));
        app.apply(UiEvent::TurnEnd("[1234 in · 56 out · 1290 total]".into()));
        app.working = true;
        app.spinner = 2;

        let mut term = Terminal::new(TestBackend::new(56, 13)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        // The header reads as "edit <path>", not a raw JSON dump.
        assert!(screen.contains("⏺ edit src/cli.rs"), "readable tool header");
        assert!(
            !screen.contains("old_string"),
            "header must not dump JSON args"
        );
        assert!(
            screen.contains("pub json: bool"),
            "ANSI diff rendered as text"
        );
        assert!(screen.contains("1290 total"), "status bar shows usage");
        assert!(
            screen.contains(SPINNER[2]) && screen.contains("0s"),
            "prompt bar shows the spinner + an elapsed timer while working: {screen}"
        );
        assert!(
            screen.contains("Ctrl-C to interrupt"),
            "prompt bar shows the interrupt hint while working"
        );
    }

    #[test]
    fn looks_like_diff_detects_unified_and_ignores_lists() {
        assert!(looks_like_diff("@@ -1,2 +1,2 @@\n-a\n+b"));
        assert!(looks_like_diff("--- a/x\n+++ b/x\n context"));
        assert!(looks_like_diff("diff --git a/x b/x\n..."));
        // A bullet list or a flag line must not be mistaken for a diff.
        assert!(!looks_like_diff("- one\n- two\n+ three"));
        assert!(!looks_like_diff("plain output\nno diff here"));
    }

    #[test]
    fn colorizes_plain_diff_tool_output() {
        let mut app = App::new("openai", "gpt-4o");
        let diff = "--- a/x.rs\n+++ b/x.rs\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n";
        app.apply(UiEvent::ToolResult(diff.into()));
        // The content span (after the "  " indent) carries the diff color.
        let colored: Vec<(String, Option<Color>)> = app
            .transcript
            .iter()
            .map(|l| {
                let text: String = l.spans.iter().map(|s| s.content.as_ref()).collect();
                (text, l.spans.last().map(|s| s.style.fg).unwrap_or(None))
            })
            .collect();
        assert!(
            colored
                .iter()
                .any(|(t, fg)| t.contains("+new") && *fg == Some(Color::Green)),
            "added line is green: {colored:?}"
        );
        assert!(
            colored
                .iter()
                .any(|(t, fg)| t.contains("-old") && *fg == Some(Color::Red)),
            "removed line is red"
        );
        assert!(
            colored
                .iter()
                .any(|(t, fg)| t.contains("@@") && *fg == Some(Color::Cyan)),
            "hunk header is cyan"
        );
    }

    #[test]
    fn non_diff_tool_output_is_not_colorized() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::ToolResult("- item one\n- item two\n".into()));
        let any_red = app
            .transcript
            .iter()
            .any(|l| l.spans.last().map(|s| s.style.fg) == Some(Some(Color::Red)));
        assert!(!any_red, "a plain list must not be colorized as a diff");
    }

    #[test]
    fn fmt_count_humanizes() {
        assert_eq!(fmt_count(0), "0");
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1234), "1.2k");
        assert_eq!(fmt_count(45000), "45k");
    }

    #[test]
    fn working_and_summary_share_one_humanizer() {
        // The live working line (fmt_count) and the settled usage summary
        // (hi_agent::humanize_count) must format a count identically — else the
        // same number renders two ways as a turn finishes (the regression fixed).
        for n in [0u64, 999, 1234, 22_864, 12_000, 1_000_000, 1_500_000] {
            assert_eq!(fmt_count(n), hi_agent::humanize_count(n), "diverged at {n}");
        }
    }

    #[test]
    fn fmt_elapsed_shows_minutes_and_seconds() {
        assert_eq!(fmt_elapsed(0), "0s");
        assert_eq!(fmt_elapsed(45), "45s");
        assert_eq!(fmt_elapsed(60), "1m 00s");
        assert_eq!(fmt_elapsed(868), "14m 28s"); // the reported "868s"
        assert_eq!(fmt_elapsed(3600), "1h 00m");
        assert_eq!(fmt_elapsed(3661), "1h 01m");
    }

    #[test]
    fn usage_event_updates_live_counter_and_working_line() {
        let mut app = App::new("openai", "gpt-4o");
        app.set_working(true);
        app.apply(UiEvent::Usage {
            input: 1234,
            output: 340,
            ctx_used: 64_000,
            ctx_window: Some(128_000),
        });
        assert_eq!(app.usage, (1234, 340));
        assert_eq!(app.context_pct(), Some(50));

        let mut term = Terminal::new(TestBackend::new(72, 8)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(screen.contains(SPINNER[0]), "spinner shown: {screen}");
        assert!(screen.contains("↑1.2k"), "live input tokens: {screen}");
        assert!(screen.contains("↓340"), "live output tokens: {screen}");
        assert!(screen.contains("50% ctx"), "live context fill: {screen}");
    }

    #[test]
    fn report_tokens_pushes_cumulative_line() {
        // `/tokens` mid-turn reads the mirrored counter (the agent is borrowed).
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::Usage {
            input: 1000,
            output: 250,
            ctx_used: 0,
            ctx_window: None,
        });
        app.report_tokens();
        let line: String = app
            .transcript
            .last()
            .unwrap()
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert_eq!(line, "cumulative: 1000 in · 250 out · 1250 total");
    }

    #[test]
    fn renders_queued_commands_while_working() {
        let mut app = App::new("openai", "gpt-4o");
        app.set_working(true);
        app.queue.push_back("run the tests".into());
        app.queue.push_back("then commit".into());
        app.input.set("typing a third");

        let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        assert!(screen.contains(SPINNER[0]), "spinner shown while working");
        assert!(
            screen.contains("run the tests"),
            "first queued command shown"
        );
        assert!(
            screen.contains("then commit"),
            "second queued command shown"
        );
        assert!(
            screen.contains("typing a third"),
            "input stays editable while working"
        );
    }

    #[test]
    fn edit_key_submits_on_enter_and_clears() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.set("queue me");
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(app.edit_key(&enter).as_deref(), Some("queue me"));
        assert!(app.input.is_empty(), "input cleared after submit");
        // An empty Enter submits nothing.
        assert_eq!(app.edit_key(&enter), None);
    }

    #[test]
    fn renders_title_transcript_and_input() {
        let mut app = App::new("openai", "gpt-4o");
        app.push(Line::raw("› hello"));
        app.apply(UiEvent::Text("hi there\n".into()));
        app.apply(UiEvent::AssistantEnd);
        app.input.set("next question");

        let mut term = Terminal::new(TestBackend::new(50, 12)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);

        assert!(screen.contains("gpt-4o"), "title shows model");
        assert!(screen.contains("hello"), "user line");
        assert!(screen.contains("hi there"), "assistant line");
        assert!(screen.contains("next question"), "input box");
    }

    #[test]
    fn turn_end_sets_status_and_marks_transcript_done() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::TurnEnd("[10 in · 2 out · 12 total]".into()));
        // Usage in the title bar...
        assert!(app.status.contains("12 total"));
        // ...and a clear "done" marker in the transcript so the turn's end shows.
        assert_eq!(app.transcript.len(), 1);
        let line: String = app.transcript[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(line.contains("✓ done"), "got: {line}");
    }

    #[test]
    fn assistant_text_becomes_copy_target() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::Text("first ".into()));
        app.apply(UiEvent::Text("answer\n".into()));
        app.apply(UiEvent::AssistantEnd);
        assert_eq!(app.last_assistant, "first answer");

        app.apply(UiEvent::ToolCall(
            "bash".into(),
            r#"{"command":"echo noisy"}"#.into(),
        ));
        app.apply(UiEvent::ToolResult("noisy output".into()));
        assert_eq!(
            app.last_assistant, "first answer",
            "tool logs are not copied as the assistant response"
        );
    }

    #[test]
    fn transcript_text_serializes_lines() {
        let mut app = App::new("openai", "gpt-4o");
        app.push(Line::raw("one"));
        app.push(Line::from(vec![Span::raw("t"), Span::raw("wo")]));
        assert_eq!(app.transcript_text(), "one\ntwo");
    }

    #[test]
    fn base64_encoder_handles_padding() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
    }

    #[test]
    fn completed_turn_without_summary_is_visible() {
        let mut app = App::new("openai", "gpt-4o");
        app.note_turn_completed_without_summary();
        let line = line_text(app.transcript.last().unwrap());
        assert!(line.contains("✓ done"), "got: {line}");
        assert!(line.contains("no usage reported"), "got: {line}");
        assert_eq!(app.status, "done · no usage reported");
    }

    #[test]
    fn stopped_after_tool_output_without_turn_end_is_visible() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::ToolCall(
            "edit".into(),
            r#"{"path":"src/main.rs"}"#.into(),
        ));
        app.apply(UiEvent::ToolResult("19 additions, 3 deletions".into()));
        app.note_turn_completed_without_summary();

        let lines: Vec<String> = app.transcript.iter().map(line_text).collect();
        assert!(
            lines
                .iter()
                .any(|line| line.contains("stopped after tool output")),
            "transcript: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("degraded in-session")),
            "transcript: {lines:?}"
        );
        assert_eq!(app.status, "stopped after tool output");
    }

    #[test]
    fn failed_turn_is_visible() {
        let mut app = App::new("openai", "gpt-4o");
        app.note_turn_failed("provider disconnected");
        let line = line_text(app.transcript.last().unwrap());
        assert!(line.contains("✗ failed"), "got: {line}");
        assert!(line.contains("provider disconnected"), "got: {line}");
        assert_eq!(app.status, "failed");
    }

    #[test]
    fn empty_tool_result_is_visible() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::ToolCall(
            "bash".into(),
            r#"{"command":"true"}"#.into(),
        ));
        app.apply(UiEvent::ToolResult(String::new()));
        let rendered: Vec<String> = app.transcript.iter().map(line_text).collect();
        assert!(
            rendered.iter().any(|line| line.contains("(no output)")),
            "transcript: {rendered:?}"
        );
    }

    #[test]
    fn input_editing_and_history() {
        let mut input = InputLine::default();
        for c in "helo".chars() {
            input.insert(c);
        }
        input.left();
        input.insert('l');
        assert_eq!(input.text(), "hello");
        input.submit();
        for c in "two".chars() {
            input.insert(c);
        }
        input.submit();
        input.history_prev();
        assert_eq!(input.text(), "two");
        input.history_prev();
        assert_eq!(input.text(), "hello");
    }

    #[test]
    fn paste_inserts_multiline_as_one_prompt() {
        // The bug: a pasted block used to submit each line. It must instead
        // become one multi-line input that submits whole on Enter.
        let mut input = InputLine::default();
        input.insert_str("line one\nline two\nline three");
        assert_eq!(input.text(), "line one\nline two\nline three");
        assert_eq!(input.submit(), "line one\nline two\nline three");
    }

    #[test]
    fn paste_normalizes_crlf() {
        let mut input = InputLine::default();
        input.insert_str("a\r\nb\rc");
        assert_eq!(input.text(), "a\nb\nc");
    }

    #[test]
    fn model_picker_filters_and_navigates() {
        let mut p = ModelPicker::new(
            vec![
                "anthropic/claude-sonnet-4".into(),
                "openai/gpt-4o".into(),
                "openai/gpt-4o-mini".into(),
                "google/gemini".into(),
            ],
            "google/gemini",
            HashMap::new(),
        );
        // Opens with the current model pre-selected.
        assert_eq!(p.current(), Some("google/gemini"));
        assert_eq!(p.matches.len(), 4);
        for c in "gpt".chars() {
            p.insert(c);
        }
        assert_eq!(p.matches.len(), 2, "only gpt-* match");
        assert_eq!(p.current(), Some("openai/gpt-4o"));
        p.down();
        assert_eq!(p.current(), Some("openai/gpt-4o-mini"));
        p.down(); // clamped at the end
        assert_eq!(p.current(), Some("openai/gpt-4o-mini"));
        p.up();
        assert_eq!(p.current(), Some("openai/gpt-4o"));
        p.backspace(); // "gp"
        p.backspace(); // "g" → matches both gpt-* and google
        assert_eq!(p.filter, "g");
        assert_eq!(p.matches.len(), 3);
    }

    #[test]
    fn renders_fetching_spinner() {
        let mut app = App::new("terminaili", "ipop/coder-balanced");
        app.fetching = Some(Instant::now());
        let mut term = Terminal::new(TestBackend::new(60, 10)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("fetching models from terminaili"),
            "fetch spinner: {screen}"
        );
        assert!(screen.contains("Esc to cancel"), "cancel hint: {screen}");
    }

    #[test]
    fn renders_model_picker() {
        let mut app = App::new("openai", "openai/gpt-4o");
        app.picker = Some(ModelPicker::new(
            vec!["anthropic/claude-sonnet-4".into(), "openai/gpt-4o".into()],
            "openai/gpt-4o",
            HashMap::new(),
        ));
        let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(screen.contains("select a model"), "title: {screen}");
        assert!(screen.contains("filter:"), "filter line: {screen}");
        assert!(screen.contains("claude-sonnet-4"), "lists models: {screen}");
        assert!(screen.contains("▶"), "highlights a selection: {screen}");
        // The active model is marked and pre-selected.
        assert!(
            screen.contains("(current)"),
            "marks current model: {screen}"
        );
    }

    #[test]
    fn picker_shows_health_tag() {
        let mut app = App::new("terminaili", "ipop/coder-balanced");
        let tags = HashMap::from([("claude-sonnet-4.6".to_string(), "degraded".to_string())]);
        app.picker = Some(ModelPicker::new(
            vec!["claude-sonnet-4.6".into(), "ipop/coder-balanced".into()],
            "ipop/coder-balanced",
            tags,
        ));
        let mut term = Terminal::new(TestBackend::new(60, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("[degraded]"),
            "degraded tag shown: {screen}"
        );
    }

    /// Concatenated text of a rendered line.
    fn line_text(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// True if any span carrying `needle` has the given modifier.
    fn span_has(line: &Line, needle: &str, m: Modifier) -> bool {
        line.spans
            .iter()
            .any(|s| s.content.contains(needle) && s.style.add_modifier.contains(m))
    }

    #[test]
    fn markdown_headings_bullets_and_rules() {
        let mut code: Option<String> = None;
        let h = markdown_line("#### 5. visited reset", &mut code);
        assert_eq!(
            line_text(&h),
            "5. visited reset",
            "heading markers stripped"
        );
        assert!(span_has(&h, "visited", Modifier::BOLD), "heading is bold");

        let b = markdown_line("- Threefold repetition", &mut code);
        assert_eq!(line_text(&b), "• Threefold repetition", "bullet rewritten");

        let n = markdown_line("7. parse_move accepts", &mut code);
        assert_eq!(line_text(&n), "7. parse_move accepts", "numbered list kept");

        assert_eq!(line_text(&markdown_line("---", &mut code)), "─".repeat(40));
    }

    #[test]
    fn markdown_code_fence_renders_interior_verbatim() {
        let mut code: Option<String> = None;
        let open = markdown_line("```rust", &mut code);
        assert!(code.is_some(), "fence opens a code block");
        assert!(line_text(&open).contains("rust"), "lang caption shown");

        // Markdown markers inside a fence are NOT interpreted.
        let inner = markdown_line("visited[tr][tc] = **true**;", &mut code);
        assert!(
            line_text(&inner).contains("**true**"),
            "code interior is verbatim: {:?}",
            line_text(&inner)
        );

        markdown_line("```", &mut code);
        assert!(code.is_none(), "closing fence ends the block");
    }

    #[test]
    fn markdown_inline_emphasis_and_code() {
        let mut code: Option<String> = None;
        let line = markdown_line("Use **mut** and `Vec` not _that_", &mut code);
        assert_eq!(
            line_text(&line),
            "Use mut and Vec not that",
            "markers consumed"
        );
        assert!(span_has(&line, "mut", Modifier::BOLD), "**bold**");
        assert!(span_has(&line, "that", Modifier::ITALIC), "_italic_");
        assert!(
            line.spans
                .iter()
                .any(|s| s.content == "Vec" && s.style.fg == Some(Color::Cyan)),
            "`code` styled"
        );
        // A bare underscore in an identifier must not start italics.
        let id = markdown_line("call is_empty here", &mut code);
        assert_eq!(line_text(&id), "call is_empty here");
        assert!(
            !span_has(&id, "is_empty", Modifier::ITALIC),
            "snake_case spared"
        );
    }

    #[test]
    fn renders_multiline_input() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.insert_str("first\nsecond\nthird");
        let mut term = Terminal::new(TestBackend::new(40, 14)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("› first"),
            "first line with prompt: {screen}"
        );
        assert!(screen.contains("second"), "second line: {screen}");
        assert!(screen.contains("third"), "third line: {screen}");
    }

    #[test]
    fn code_block_highlights_strings_and_comments() {
        let mut code: Option<String> = None;
        markdown_line("```rust", &mut code);
        // A whole-line comment is dimmed.
        let c = markdown_line("    // a note", &mut code);
        assert!(
            c.spans
                .iter()
                .any(|s| s.content.contains("// a note")
                    && s.style.add_modifier.contains(Modifier::DIM)),
            "comment dimmed"
        );
        // A string literal is greened; the rest is not.
        let s = markdown_line("let x = \"hi\";", &mut code);
        assert!(
            s.spans
                .iter()
                .any(|sp| sp.content == "\"hi\"" && sp.style.fg == Some(Color::Green)),
            "string greened: {:?}",
            s.spans
                .iter()
                .map(|sp| (sp.content.as_ref(), sp.style.fg))
                .collect::<Vec<_>>()
        );
        // Unknown language → no comment marker, so a `#`-line isn't dimmed away.
        let mut code2 = Some(String::new());
        let u = markdown_line("# this is a heading-ish line", &mut code2);
        assert!(
            !u.spans
                .iter()
                .skip(1)
                .all(|s| s.style.add_modifier.contains(Modifier::DIM)),
            "unknown lang doesn't treat # as a comment"
        );
    }

    #[test]
    fn diff_lines_number_the_new_file() {
        let body = "--- a/x\n+++ b/x\n@@ -10,3 +10,4 @@\n ctx\n-old\n+new\n+more\n";
        let lines = diff_lines(body);
        let text: Vec<String> = lines.iter().map(line_text).collect();
        // Context line is numbered from the hunk's new-file start (10).
        assert!(
            text.iter().any(|t| t.contains("10") && t.contains("ctx")),
            "{text:?}"
        );
        // Additions continue the new-file numbering (11, 12); removals don't advance it.
        assert!(
            text.iter().any(|t| t.contains("11") && t.contains("+new")),
            "{text:?}"
        );
        assert!(
            text.iter().any(|t| t.contains("12") && t.contains("+more")),
            "{text:?}"
        );
        // The removed line carries no number (blank gutter before the '-').
        let removed = text.iter().find(|t| t.contains("-old")).unwrap();
        assert!(
            !removed.chars().any(|c| c.is_ascii_digit()),
            "removed line has no number: {removed:?}"
        );
    }

    #[test]
    fn alt_enter_and_backslash_insert_newline_instead_of_submitting() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.set("line one");
        let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        assert_eq!(app.edit_key(&alt_enter), None, "alt+enter does not submit");
        assert_eq!(app.input.text(), "line one\n");

        // Trailing backslash + Enter continues the line (universal fallback).
        app.input.set("a\\");
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        assert_eq!(app.edit_key(&enter), None, "backslash continues");
        assert_eq!(app.input.text(), "a\n");

        // A normal Enter still submits.
        app.input.set("go");
        assert_eq!(app.edit_key(&enter).as_deref(), Some("go"));
    }

    #[test]
    fn failed_turn_shows_reason_and_keeps_error() {
        let mut app = App::new("openai", "gpt-4o");
        app.note_turn_failed("API error 401: invalid or expired session");
        // record_model_issue runs next in the real flow; it must NOT clobber the
        // real error with a reliability-count message.
        app.record_model_issue();
        assert_eq!(
            app.last_error.as_deref(),
            Some("API error 401: invalid or expired session"),
            "the real error is preserved for /status and /log"
        );
        // The bottom bar shows the reason inline, not a bare "failed".
        let mut term = Terminal::new(TestBackend::new(80, 8)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(
            screen.contains("last: failed — API error 401"),
            "reason inline: {screen}"
        );
        assert!(screen.contains("/retry"), "recovery hint: {screen}");
    }

    #[test]
    fn clip_reason_collapses_and_truncates() {
        assert_eq!(clip_reason("a\n  b   c"), "a b c");
        assert!(clip_reason(&"x".repeat(200)).ends_with('…'));
    }

    #[test]
    fn completion_context_tracks_name_then_argument() {
        use CompletionContext::{Arg, Command};
        // The command name, until a space is typed.
        assert_eq!(completion_context("/"), Some(Command(String::new())));
        assert_eq!(completion_context("/mo"), Some(Command("mo".to_string())));
        assert_eq!(completion_context("/MODEL"), Some(Command("model".to_string())));
        // Past the name, on the argument of a command with enumerable values.
        assert_eq!(
            completion_context("/compact "),
            Some(Arg { cmd: "compact", prefix: String::new() })
        );
        assert_eq!(
            completion_context("/compact hy"),
            Some(Arg { cmd: "compact", prefix: "hy".to_string() })
        );
        // The single-keyword commands and the dynamic model command, too.
        assert_eq!(
            completion_context("/verify "),
            Some(Arg { cmd: "verify", prefix: String::new() })
        );
        assert_eq!(
            completion_context("/model gp"),
            Some(Arg { cmd: "model", prefix: "gp".to_string() })
        );
        // A fully-typed valid static value has nothing left to complete → no menu.
        assert_eq!(completion_context("/compact hybrid"), None);
        assert_eq!(completion_context("/verify off"), None);
        // A command that takes no argument, with a trailing space → no menu.
        assert_eq!(completion_context("/diff "), None);
        // A second argument token is past the single arg → no menu.
        assert_eq!(completion_context("/compact hybrid x"), None);
        // Not a slash command at all.
        assert_eq!(completion_context("hello"), None);
    }

    #[test]
    fn completion_opens_filters_and_closes() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.set("/");
        app.sync_completion();
        assert_eq!(
            app.completion_items().len(),
            hi_agent::command::COMMANDS.len(),
            "bare slash lists every command"
        );
        app.input.set("/co");
        app.sync_completion();
        let labels: Vec<String> = app.completion_items().iter().map(|i| i.label.clone()).collect();
        assert!(
            labels.contains(&"/copy".to_string()) && labels.contains(&"/compact".to_string()),
            "got {labels:?}"
        );
        assert!(labels.iter().all(|n| n.starts_with("/co")));
        // A space after a command that takes no argument closes the menu.
        app.input.set("/diff ");
        app.sync_completion();
        assert!(app.completion.is_none());
    }

    #[test]
    fn goal_feedback_is_prominent_on_change_quiet_on_read() {
        // Setting echoes the stored goal as a prominent ✓ confirmation that says
        // it persists — the visibility fix for "/goal seemed to do nothing".
        let (msg, prominent) = goal_feedback("ship it", Some("ship it"));
        assert!(prominent, "a set is an applied change, shown plainly");
        assert!(msg.starts_with("✓ goal set"), "got: {msg}");
        assert!(
            msg.contains("ship it") && msg.contains("until cleared"),
            "echoes the goal and that it persists: {msg}"
        );
        // Clearing is also a prominent ✓.
        assert_eq!(
            goal_feedback("clear", None),
            ("✓ goal cleared".to_string(), true)
        );
        // A bare /goal is a quiet read-out, not a ✓ confirmation.
        let (read, prominent) = goal_feedback("", Some("ship it"));
        assert_eq!((read.as_str(), prominent), ("goal: ship it", false));
        assert!(!goal_feedback("", None).1, "the off read-out stays dim too");
    }

    #[test]
    fn completion_offers_verify_and_goal_keywords() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.set("/verify ");
        app.sync_completion();
        let labels: Vec<String> = app.completion_items().iter().map(|i| i.label.clone()).collect();
        assert_eq!(labels, vec!["off"], "verify offers its disable keyword");
        app.input.set("/goal cl");
        app.sync_completion();
        let labels: Vec<String> = app.completion_items().iter().map(|i| i.label.clone()).collect();
        assert_eq!(labels, vec!["clear"], "goal offers its clear keyword");
        assert_eq!(app.accept_completion(true).as_deref(), Some("/goal clear"));
    }

    #[test]
    fn completion_offers_live_model_ids() {
        let mut app = App::new("openai", "gpt-4o");
        app.model_ids = vec!["gpt-4o".into(), "gpt-4o-mini".into(), "claude-opus".into()];
        app.input.set("/model gp");
        app.sync_completion();
        let labels: Vec<String> = app.completion_items().iter().map(|i| i.label.clone()).collect();
        assert_eq!(labels, vec!["gpt-4o", "gpt-4o-mini"], "filters the catalog by prefix");
        // Accepting a row runs the full command.
        app.completion.as_mut().unwrap().selected = 1;
        assert_eq!(app.accept_completion(true).as_deref(), Some("/model gpt-4o-mini"));

        // With no catalog loaded, there's no inline menu — the picker still
        // handles `/model` (so the feature degrades, it doesn't break).
        let mut bare = App::new("openai", "gpt-4o");
        bare.input.set("/model gp");
        bare.sync_completion();
        assert!(bare.completion.is_none());
    }

    #[test]
    fn completion_offers_then_fills_compact_kinds() {
        let mut app = App::new("openai", "gpt-4o");
        // The space that used to kill the menu now offers the kinds.
        app.input.set("/compact ");
        app.sync_completion();
        let labels: Vec<String> = app.completion_items().iter().map(|i| i.label.clone()).collect();
        assert_eq!(labels, vec!["hybrid", "full", "elide"], "offers every kind");
        // Typing narrows by prefix.
        app.input.set("/compact e");
        app.sync_completion();
        let labels: Vec<String> = app.completion_items().iter().map(|i| i.label.clone()).collect();
        assert_eq!(labels, vec!["elide"]);
        // Accepting a kind fills the whole command and runs it on Enter.
        assert_eq!(app.accept_completion(true).as_deref(), Some("/compact elide"));
        assert!(app.completion.is_none(), "menu closes after accept");
    }

    #[test]
    fn completing_compact_name_opens_its_kind_menu() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.set("/compact");
        app.sync_completion();
        // Tab accepts the command name, leaving `/compact `…
        app.accept_completion(false);
        assert_eq!(app.input.text(), "/compact ");
        // …and the re-sync the Tab handler performs opens the kind menu.
        app.sync_completion();
        let labels: Vec<String> = app.completion_items().iter().map(|i| i.label.clone()).collect();
        assert!(labels.contains(&"hybrid".to_string()), "got {labels:?}");
    }

    #[test]
    fn completion_navigation_and_accept() {
        let mut app = App::new("openai", "gpt-4o");
        // No-arg command: Enter accepts and submits immediately.
        app.input.set("/hel");
        app.sync_completion();
        let line = app.accept_completion(true);
        assert_eq!(line.as_deref(), Some("/help"));
        assert!(app.completion.is_none(), "menu closes after accept");

        // Arg-taking command: accept leaves a trailing space, does not submit.
        app.input.set("/mod");
        app.sync_completion();
        assert_eq!(
            app.accept_completion(true),
            None,
            "arg command waits for input"
        );
        assert_eq!(app.input.text(), "/model ");

        // Tab never submits, even for a no-arg command.
        app.input.set("/dif");
        app.sync_completion();
        assert_eq!(app.accept_completion(false), None);
        assert_eq!(app.input.text(), "/diff");
    }

    #[test]
    fn completion_move_clamps() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.set("/co"); // [copy, compact]
        app.sync_completion();
        app.completion_move(-1); // already at 0, stays
        assert_eq!(app.completion.as_ref().unwrap().selected, 0);
        app.completion_move(1);
        assert_eq!(app.completion.as_ref().unwrap().selected, 1);
        app.completion_move(1); // clamp at last
        assert_eq!(app.completion.as_ref().unwrap().selected, 1);
    }

    #[test]
    fn renders_completion_menu() {
        let mut app = App::new("openai", "gpt-4o");
        app.input.set("/");
        app.sync_completion();
        let mut term = Terminal::new(TestBackend::new(72, 20)).unwrap();
        term.draw(|f| app.render(f)).unwrap();
        let screen = dump(&term);
        assert!(screen.contains("/help"), "lists help: {screen}");
        assert!(screen.contains("/model"), "lists model: {screen}");
        assert!(screen.contains("▶"), "highlights a row: {screen}");
    }
}
