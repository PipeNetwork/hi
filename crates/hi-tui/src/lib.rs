//! Full-screen terminal UI for `hi`.
//!
//! A ratatui application on the alternate screen: a bordered, scrollable
//! conversation transcript with a title/status bar, and an input box with a
//! "working" spinner. The agent runs behind an mpsc channel ([`ChannelUi`]) so
//! the event loop can keep redrawing — spinner, streaming output, scrolling —
//! while a turn is in flight, and can cancel it with Ctrl-C.

use std::collections::VecDeque;
use std::io::{self};
use std::time::{Duration, Instant};

use ansi_to_tui::IntoText;
use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use hi_agent::ui::preview_args;
use hi_agent::{Agent, Command, Ui, command};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Paragraph, Wrap};
use tokio::sync::mpsc;

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const TICK: Duration = Duration::from_millis(120);

/// Run the full-screen TUI until the user quits. `history_path`, if given, is
/// the file used to persist input history across sessions (shared with the
/// plain REPL).
pub async fn run(
    agent: &mut Agent,
    provider: &str,
    model: &str,
    registry: &hi_ai::Registry,
    history_path: Option<std::path::PathBuf>,
) -> Result<()> {
    enable_raw_mode().context("entering raw mode")?;
    execute!(io::stdout(), EnterAlternateScreen).context("entering alternate screen")?;
    let _restore = Restore;
    let mut terminal =
        Terminal::new(CrosstermBackend::new(io::stdout())).context("creating terminal")?;

    let mut app = App::new(provider, model);
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
    let mut events = EventStream::new();
    let mut ticker = tokio::time::interval(TICK);

    'session: loop {
        // Run a queued command first (typed while the previous turn ran);
        // otherwise edit the input line until the user submits.
        let line = match app.queue.pop_front() {
            Some(queued) => queued,
            None => 'input: loop {
                terminal.draw(|f| app.render(f))?;
                tokio::select! {
                    maybe = events.next() => {
                        let Some(Ok(Event::Key(key))) = maybe else { continue };
                        if key.kind != KeyEventKind::Press { continue; }
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
                            }
                        }
                    }
                    _ = ticker.tick() => {}
                }
            },
        };

        // Slash commands. Most are handled inline; `/compact` runs a model call
        // (driven like a turn so the spinner shows); `/retry` yields the prompt
        // to re-run in the turn phase below.
        let run_line = if let Some(cmd) = command::parse(&line) {
            match cmd {
                Command::Quit => break,
                Command::Compact => {
                    app.set_working(true);
                    app.follow();
                    let (tx, rx) = mpsc::unbounded_channel();
                    let mut sink = ChannelUi { tx };
                    {
                        let fut = agent.compact(&mut sink);
                        drive(&mut terminal, &mut events, &mut ticker, &mut app, rx, fut).await?;
                    }
                    app.set_working(false);
                    app.follow();
                    continue;
                }
                Command::Retry => match app.last_prompt.clone() {
                    Some(prompt) => {
                        agent.truncate_messages(app.last_turn_start);
                        app.push(Line::styled(format!("retrying: {prompt}"), dim()));
                        prompt
                    }
                    None => {
                        app.push(Line::styled("nothing to retry yet".to_string(), dim()));
                        continue;
                    }
                },
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
            drive(&mut terminal, &mut events, &mut ticker, &mut app, rx, fut).await?
        };

        if cancelled {
            agent.truncate_messages(checkpoint);
            let dropped = app.queue.len();
            app.queue.clear();
            let msg = if dropped > 0 {
                format!("^C interrupted; turn discarded ({dropped} queued command(s) dropped)")
            } else {
                "^C interrupted; turn discarded".to_string()
            };
            app.push(Line::styled(msg, Style::default().fg(Color::Yellow)));
        }
        app.set_working(false);
        app.follow();
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
    events: &mut EventStream,
    ticker: &mut tokio::time::Interval,
    app: &mut App,
    mut rx: mpsc::UnboundedReceiver<UiEvent>,
    fut: impl std::future::Future<Output = Result<()>>,
) -> Result<bool> {
    tokio::pin!(fut);
    let mut cancelled = false;
    loop {
        terminal.draw(|f| app.render(f))?;
        tokio::select! {
            result = &mut fut => {
                while let Ok(event) = rx.try_recv() { app.apply(event); }
                if let Err(err) = result {
                    app.push(Line::styled(
                        format!("error: {err:#}"),
                        Style::default().fg(Color::Red),
                    ));
                }
                break;
            }
            Some(event) = rx.recv() => app.apply(event),
            _ = ticker.tick() => app.spinner = app.spinner.wrapping_add(1),
            maybe = events.next() => {
                if let Some(Ok(Event::Key(key))) = maybe
                    && key.kind == KeyEventKind::Press
                {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    match key.code {
                        KeyCode::Char('c') if ctrl => { cancelled = true; break; }
                        KeyCode::Esc => app.input.clear(),
                        // Typing while a turn runs queues the next command.
                        _ => if let Some(queued) = app.edit_key(&key) {
                            app.queue.push_back(queued);
                        }
                    }
                }
            }
        }
    }
    Ok(cancelled)
}

/// Restores the terminal on drop (covers early returns and panics).
struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
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
    fn turn_end(&mut self, summary: &str) {
        self.send(UiEvent::TurnEnd(summary.to_string()));
    }
}

struct App {
    provider: String,
    model: String,
    transcript: Vec<Line<'static>>,
    /// The in-progress streamed line: (style, text). Committed on newline/end.
    pending: Option<(Style, String)>,
    input: InputLine,
    /// Lines scrolled up from the bottom (0 = following the latest output).
    scroll_up: u16,
    working: bool,
    spinner: usize,
    /// When the current turn started, for the elapsed-time readout.
    started: Option<Instant>,
    /// Lines typed while a turn was running, to run once it finishes (FIFO).
    queue: VecDeque<String>,
    /// The last message actually sent to the model, for `/retry`.
    last_prompt: Option<String>,
    /// Message-history length just before the last turn started, so `/retry`
    /// can drop that turn before re-running.
    last_turn_start: usize,
    status: String,
}

impl App {
    fn new(provider: &str, model: &str) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            transcript: Vec::new(),
            pending: None,
            input: InputLine::default(),
            scroll_up: 0,
            working: false,
            spinner: 0,
            started: None,
            queue: VecDeque::new(),
            last_prompt: None,
            last_turn_start: 0,
            status: String::new(),
        }
    }

    /// Mark the turn as running (or done), stamping the start time so the
    /// prompt bar can show elapsed seconds.
    fn set_working(&mut self, working: bool) {
        self.working = working;
        self.started = working.then(Instant::now);
    }

    /// Apply a pure editing/navigation key to the input line, shared by the
    /// idle input phase and the in-turn queue-entry path. Returns the submitted
    /// text on Enter (when non-empty); the caller decides whether to run it now
    /// or queue it. Phase-specific control keys (Ctrl-C/Ctrl-D/Esc) are handled
    /// by the caller, not here.
    fn edit_key(&mut self, key: &KeyEvent) -> Option<String> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Enter => {
                let line = self.input.submit();
                if !line.trim().is_empty() {
                    return Some(line);
                }
            }
            KeyCode::Char('u') if ctrl => self.input.kill_to_start(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
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

    fn follow(&mut self) {
        self.scroll_up = 0;
    }

    fn scroll_up(&mut self, n: u16) {
        self.scroll_up = self.scroll_up.saturating_add(n);
    }

    fn scroll_down(&mut self, n: u16) {
        self.scroll_up = self.scroll_up.saturating_sub(n);
    }

    /// Commit the in-progress streamed line, if any.
    fn flush_pending(&mut self) {
        if let Some((style, text)) = self.pending.take() {
            self.transcript.push(Line::styled(text, style));
        }
    }

    /// Append streamed text under `style`, committing complete lines.
    fn stream(&mut self, style: Style, chunk: &str) {
        // A style change ends the current line.
        if let Some((prev, _)) = &self.pending
            && *prev != style
        {
            self.flush_pending();
        }
        let (_, buf) = self.pending.get_or_insert_with(|| (style, String::new()));
        buf.push_str(chunk);
        while let Some(idx) = buf.find('\n') {
            let committed: String = buf[..idx].to_string();
            buf.drain(..=idx);
            self.transcript.push(Line::styled(committed, style));
        }
        self.follow();
    }

    fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Text(t) => self.stream(Style::default(), &t),
            UiEvent::Reasoning(t) => self.stream(dim(), &t),
            UiEvent::AssistantEnd => self.flush_pending(),
            UiEvent::ToolCall(name, args) => {
                self.flush_pending();
                self.push(Line::styled(
                    format!("⏺ {name}({})", preview_args(&args)),
                    Style::default().fg(Color::Cyan),
                ));
            }
            UiEvent::ToolResult(result) => {
                self.flush_pending();
                self.push_result(&result);
            }
            UiEvent::Status(s) => {
                self.flush_pending();
                self.push(Line::styled(s, Style::default().fg(Color::Blue)));
            }
            UiEvent::TurnEnd(summary) => {
                // Surface tokens/cost in the status bar rather than the transcript.
                self.status = summary.trim_matches(['[', ']']).to_string();
            }
        }
    }

    /// Render a tool result, preserving any ANSI colors (e.g. edit diffs),
    /// clipped to a handful of lines and indented.
    fn push_result(&mut self, result: &str) {
        const MAX: usize = 14;
        let body: String = result.lines().take(MAX).collect::<Vec<_>>().join("\n");
        let text: Text = body
            .into_text()
            .unwrap_or_else(|_| Text::from(body.clone()));
        for mut line in text.lines {
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
                for line in command::HELP.lines() {
                    self.push(Line::styled(line.to_string(), dim()));
                }
            }
            Command::Tokens => {
                let t = agent.totals();
                self.push(Line::styled(
                    format!(
                        "cumulative: {} in · {} out · {} total",
                        t.input_tokens,
                        t.output_tokens,
                        t.total()
                    ),
                    dim(),
                ));
            }
            Command::Model(id) => {
                if id.is_empty() {
                    self.push(Line::styled(format!("model: {}", self.model), dim()));
                } else {
                    let (price, context_window) = registry.metadata(&id);
                    agent.set_model(id.clone(), price, context_window);
                    self.model = id.clone();
                    self.push(Line::styled(format!("model set to {id}"), dim()));
                }
            }
            Command::Clear => {
                agent.clear_history();
                self.transcript.clear();
                self.pending = None;
                self.status.clear();
                self.push(Line::styled("conversation cleared", dim()));
            }
            Command::Verify(arg) => {
                let msg = match arg.trim() {
                    "" => match agent.verify_command() {
                        Some(c) => format!("verify: `{c}`"),
                        None => "verify: off (set one with /verify <cmd>)".to_string(),
                    },
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
            // Handled in the event loop (async / runs a turn); never reach here.
            Command::Compact | Command::Retry => {}
            Command::Unknown(name) => {
                self.push(Line::styled(
                    format!("unknown command /{name}; try /help"),
                    dim(),
                ));
            }
        }
        self.follow();
    }

    fn render(&self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        // The input box grows to fit a spinner status line (while working) and
        // up to three queued commands (plus a "+N more" line).
        let status_lines = usize::from(self.working);
        let queued_shown = self.queue.len().min(3);
        let queue_extra = usize::from(self.queue.len() > 3);
        let input_h = (status_lines + 1 + queued_shown + queue_extra + 2) as u16;
        let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(input_h)]).split(area);

        // --- Transcript ---
        let title = format!(" hi · {} · {} ", self.provider, self.model);
        let info = if self.status.is_empty() {
            String::new()
        } else {
            format!(" {} ", self.status)
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(dim())
            .title(title)
            .title_top(Line::from(info).right_aligned());

        let mut lines = self.transcript.clone();
        if let Some((style, text)) = &self.pending {
            lines.push(Line::styled(text.clone(), *style));
        }
        let inner_w = rows[0].width.saturating_sub(2);
        let inner_h = rows[0].height.saturating_sub(2);
        let total = wrapped_height(&lines, inner_w);
        let max_scroll = total.saturating_sub(inner_h);
        let scroll = max_scroll.saturating_sub(self.scroll_up);
        let para = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block)
            .scroll((scroll, 0));
        frame.render_widget(para, rows[0]);

        // --- Input bar. While a turn runs the border turns cyan and the top
        // inner line becomes a bold spinner + elapsed seconds; the prompt stays
        // editable so you can type the next command, which queues (listed
        // below) and runs when the current turn finishes. ---
        let input_block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(if self.working {
                Style::default().fg(Color::Cyan)
            } else {
                dim()
            });

        let mut ilines: Vec<Line> = Vec::new();
        if self.working {
            let frame_ch = SPINNER[self.spinner % SPINNER.len()];
            let secs = self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
            ilines.push(Line::from(vec![
                Span::styled(
                    format!("{frame_ch} working… {secs}s"),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   Ctrl-C to interrupt", dim()),
            ]));
        }
        ilines.push(Line::from(format!("› {}", self.input.text())));
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

        // Cursor sits on the editable prompt line (below the spinner line, if shown).
        let cx = rows[1].x + 1 + 2 + self.input.cursor() as u16;
        let cy = rows[1].y + 1 + status_lines as u16;
        frame.set_cursor_position((cx.min(rows[1].right().saturating_sub(2)), cy));
    }
}

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
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
    fn renders_tool_call_diff_and_spinner() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::ToolCall(
            "edit".into(),
            "{\"path\":\"src/cli.rs\"}".into(),
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

        assert!(screen.contains("⏺ edit"), "tool call line");
        assert!(
            screen.contains("pub json: bool"),
            "ANSI diff rendered as text"
        );
        assert!(screen.contains("1290 total"), "status bar shows usage");
        assert!(
            screen.contains("working… 0s"),
            "prompt bar shows spinner + elapsed while working"
        );
        assert!(
            screen.contains("Ctrl-C to interrupt"),
            "prompt bar shows the interrupt hint while working"
        );
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

        assert!(screen.contains("working…"), "spinner shown while working");
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
    fn turn_end_summary_goes_to_status_not_transcript() {
        let mut app = App::new("openai", "gpt-4o");
        app.apply(UiEvent::TurnEnd("[10 in · 2 out · 12 total]".into()));
        assert!(app.status.contains("12 total"));
        assert!(app.transcript.is_empty());
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
}
