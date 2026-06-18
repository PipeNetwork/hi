//! Inline terminal UI built on ratatui.
//!
//! Uses an *inline* viewport (not the alternate screen), so finalized
//! transcript lines are pushed into the real scrollback via `insert_before`
//! while a small fixed region at the bottom shows a status line and the input
//! box. Streaming output is line-buffered: complete lines are committed as they
//! form; the trailing partial line lands when the message ends.
//!
//! Requires a real terminal (the inline viewport queries cursor position).
//! Known v1 limitations: no sub-line live streaming, and no mid-turn
//! cancellation — both are straightforward follow-ups.

use std::io::{self, Stdout};

use anyhow::Result;
use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use futures_util::StreamExt;
use pi_agent::ui::clip;
use pi_agent::{Agent, Command, Ui, command, preview_args};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{TerminalOptions, Viewport};

const VIEWPORT_HEIGHT: u16 = 2;

/// Run the interactive TUI loop until the user quits. `provider` and `model`
/// label the status line (e.g. "openai · gpt-4o").
pub async fn run(
    agent: &mut Agent,
    provider: &str,
    model: &str,
    registry: &pi_ai::Registry,
) -> Result<()> {
    enable_raw_mode()?;
    let _guard = RawGuard;

    let terminal = Terminal::with_options(
        CrosstermBackend::new(io::stdout()),
        TerminalOptions {
            viewport: Viewport::Inline(VIEWPORT_HEIGHT),
        },
    )?;

    let mut tui = Tui {
        terminal,
        provider: provider.to_string(),
        model: model.to_string(),
        status: String::new(),
        working: false,
        width: 80,
        pending_text: String::new(),
        pending_think: String::new(),
        input: InputLine::default(),
    };
    tui.emit_line("hi — Ctrl-D to quit, Enter to send.", dim());

    let mut events = EventStream::new();
    loop {
        tui.draw()?;
        let Some(event) = events.next().await else {
            break;
        };
        let Ok(Event::Key(key)) = event else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Enter => {
                let line = tui.input.submit();
                if line.trim().is_empty() {
                    continue;
                }
                if let Some(cmd) = command::parse(&line) {
                    if matches!(cmd, Command::Quit) {
                        break;
                    }
                    tui.handle_command(agent, cmd, registry);
                    continue;
                }
                tui.emit_line(&format!("› {line}"), Style::default().fg(Color::Blue));
                tui.working = true;
                let result = agent.run_turn(&line, &mut tui).await;
                tui.working = false;
                if let Err(err) = result {
                    tui.emit_line(&format!("error: {err:#}"), Style::default().fg(Color::Red));
                }
            }
            KeyCode::Char('d') if ctrl && tui.input.is_empty() => break,
            KeyCode::Char('c') if ctrl => tui.input.clear(),
            KeyCode::Char('u') if ctrl => tui.input.kill_to_start(),
            KeyCode::Char('a') if ctrl => tui.input.home(),
            KeyCode::Char('e') if ctrl => tui.input.end(),
            KeyCode::Char(c) if !ctrl => tui.input.insert(c),
            KeyCode::Backspace => tui.input.backspace(),
            KeyCode::Left => tui.input.left(),
            KeyCode::Right => tui.input.right(),
            KeyCode::Home => tui.input.home(),
            KeyCode::End => tui.input.end(),
            KeyCode::Up => tui.input.history_prev(),
            KeyCode::Down => tui.input.history_next(),
            KeyCode::Esc => {
                if tui.input.is_empty() {
                    break;
                }
                tui.input.clear();
            }
            _ => {}
        }
    }

    // Leave the cursor on a fresh line below the viewport.
    tui.emit_line("", Style::default());
    Ok(())
}

struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    provider: String,
    model: String,
    status: String,
    working: bool,
    width: u16,
    pending_text: String,
    pending_think: String,
    input: InputLine,
}

impl Tui {
    fn handle_command(&mut self, agent: &mut Agent, command: Command, registry: &pi_ai::Registry) {
        match command {
            Command::Quit => {}
            Command::Help => self.emit_line(command::HELP, dim()),
            Command::Tokens => {
                let t = agent.totals();
                let text = format!(
                    "cumulative: {} in · {} out · {} total",
                    t.input_tokens,
                    t.output_tokens,
                    t.total()
                );
                self.emit_line(&text, dim());
            }
            Command::Model(id) => {
                if id.is_empty() {
                    self.emit_line(&format!("model: {}", self.model), dim());
                } else {
                    let (price, context_window) = registry.metadata(&id);
                    agent.set_model(id.clone(), price, context_window);
                    self.model = id.clone();
                    self.emit_line(&format!("model set to {id}"), dim());
                }
            }
            Command::Clear => {
                agent.clear_history();
                self.emit_line("conversation cleared", dim());
            }
            Command::Unknown(name) => {
                self.emit_line(&format!("unknown command /{name}; try /help"), dim());
            }
        }
    }

    fn draw(&mut self) -> io::Result<()> {
        let header = format!("{} · {}", self.provider, self.model);
        let status = if self.working {
            format!("{header}  ·  working…  {}", self.status)
        } else if self.status.is_empty() {
            header
        } else {
            format!("{header}  ·  {}", self.status)
        };
        let input = self.input.text();
        let cursor_col = (2 + self.input.cursor()).min(u16::MAX as usize) as u16;

        self.terminal.draw(|frame| {
            let area = frame.area();
            self.width = area.width;
            let status_area = Rect::new(area.x, area.y, area.width, 1);
            let input_area = Rect::new(area.x, area.y.saturating_add(1), area.width, 1);
            frame.render_widget(Paragraph::new(Line::styled(status, dim())), status_area);
            frame.render_widget(Paragraph::new(format!("› {input}")), input_area);
            let col = input_area
                .x
                .saturating_add(cursor_col)
                .min(area.right().saturating_sub(1));
            frame.set_cursor_position((col, input_area.y));
        })?;
        Ok(())
    }

    /// Push pre-styled, width-wrapped lines into scrollback above the viewport.
    fn emit(&mut self, text: &str, style: Style) -> io::Result<()> {
        let wrapped = wrap(text, self.width as usize);
        let height = wrapped.len().max(1) as u16;
        let lines: Vec<Line> = wrapped.into_iter().map(|l| Line::styled(l, style)).collect();
        self.terminal.insert_before(height, |buf| {
            let area = buf.area;
            Paragraph::new(Text::from(lines)).render(area, buf);
        })
    }

    fn emit_line(&mut self, text: &str, style: Style) {
        let _ = self.emit(text, style);
    }
}

impl Ui for Tui {
    fn assistant_text(&mut self, text: &str) {
        self.pending_text.push_str(text);
        while let Some(idx) = self.pending_text.find('\n') {
            let line: String = self.pending_text[..idx].to_string();
            self.emit_line(&line, Style::default());
            self.pending_text.drain(..=idx);
        }
    }

    fn assistant_reasoning(&mut self, text: &str) {
        self.pending_think.push_str(text);
        while let Some(idx) = self.pending_think.find('\n') {
            let line: String = self.pending_think[..idx].to_string();
            self.emit_line(&line, dim());
            self.pending_think.drain(..=idx);
        }
    }

    fn assistant_end(&mut self) {
        if !self.pending_think.is_empty() {
            let line = std::mem::take(&mut self.pending_think);
            self.emit_line(&line, dim());
        }
        if !self.pending_text.is_empty() {
            let line = std::mem::take(&mut self.pending_text);
            self.emit_line(&line, Style::default());
        }
    }

    fn tool_call(&mut self, name: &str, arguments: &str) {
        let text = format!("⏺ {name}({})", preview_args(arguments));
        self.emit_line(&text, Style::default().fg(Color::Cyan));
    }

    fn tool_result(&mut self, result: &str) {
        const MAX_LINES: usize = 12;
        let lines: Vec<&str> = result.lines().collect();
        for line in lines.iter().take(MAX_LINES) {
            self.emit_line(&format!("  {}", clip(line, 200)), dim());
        }
        if lines.len() > MAX_LINES {
            self.emit_line(&format!("  … {} more lines", lines.len() - MAX_LINES), dim());
        }
    }

    fn status(&mut self, text: &str) {
        self.emit_line(text, Style::default().fg(Color::Blue));
    }

    fn turn_end(&mut self, summary: &str) {
        self.status = summary.trim_matches(['[', ']']).to_string();
        self.emit_line(summary, dim());
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

    /// Take the current text, clearing the line and recording non-empty input.
    fn submit(&mut self) -> String {
        let line = self.text();
        self.clear();
        if !line.trim().is_empty() {
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

fn dim() -> Style {
    Style::default().add_modifier(Modifier::DIM)
}

/// Hard-wrap each line to `width` characters (char-based, never panics).
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        let chars: Vec<char> = raw.chars().collect();
        if chars.is_empty() {
            out.push(String::new());
            continue;
        }
        for chunk in chars.chunks(width) {
            out.push(chunk.iter().collect());
        }
    }
    out
}

/// Restores the terminal even if the loop exits early or errors.
struct RawGuard;

impl Drop for RawGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::{InputLine, wrap};

    #[test]
    fn wrap_hard_wraps_and_keeps_blank_lines() {
        assert_eq!(wrap("abcdef", 3), vec!["abc", "def"]);
        assert_eq!(wrap("a\nb", 10), vec!["a", "b"]);
        assert_eq!(wrap("", 10), vec![""]);
        assert_eq!(wrap("x\n\ny", 10), vec!["x", "", "y"]);
    }

    #[test]
    fn input_editing() {
        let mut input = InputLine::default();
        for c in "helo".chars() {
            input.insert(c);
        }
        input.left();
        input.insert('l');
        assert_eq!(input.text(), "hello");
        input.home();
        assert_eq!(input.cursor(), 0);
        input.end();
        assert_eq!(input.cursor(), 5);
        input.backspace();
        assert_eq!(input.text(), "hell");
        input.kill_to_start();
        assert!(input.is_empty());
    }

    #[test]
    fn history_navigation() {
        let mut input = InputLine::default();
        for c in "one".chars() {
            input.insert(c);
        }
        input.submit();
        for c in "two".chars() {
            input.insert(c);
        }
        input.submit();

        input.history_prev();
        assert_eq!(input.text(), "two");
        input.history_prev();
        assert_eq!(input.text(), "one");
        input.history_next();
        assert_eq!(input.text(), "two");
        input.history_next();
        assert_eq!(input.text(), "");
    }
}
