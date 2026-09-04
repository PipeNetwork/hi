//! Deterministic, terminal-free driver for the production TUI.
//!
//! The JSONL protocol is intentionally small: callers mutate the real [`App`]
//! and render it through ratatui's [`TestBackend`]. No alternate-screen or
//! provider setup is involved, so UI automation can run in CI over stdio.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::{Backend, TestBackend};
use ratatui::text::Line;
use serde::Serialize;
use serde_json::{Value, json};
use unicode_width::UnicodeWidthStr;

use crate::activity_feed::ActivityKind;
use crate::dispatch::DispatchResult;
use crate::mode::UiMode;
use crate::{App, RaceDefaults, TranscriptEntry};

#[path = "debug_harness/protocol.rs"]
mod protocol;
use protocol::{Command, HarnessError, WireRequest, WireResponse};
#[path = "debug_harness/tree.rs"]
mod tree;

pub const TUI_STDIO_PROTOCOL_VERSION: u16 = 1;
pub const TUI_COMPONENT_TREE_SCHEMA_VERSION: u16 = 1;

const DEFAULT_WIDTH: u16 = 80;
const DEFAULT_HEIGHT: u16 = 24;
const MAX_WIDTH: u16 = 512;
const MAX_HEIGHT: u16 = 256;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;

const HELP: &str = "\
Deterministic TUI harness

Usage:
  hi debug tui --stdio
  hi debug tui --help

--stdio reads one JSON object per line and writes one response per line.
Protocol schema: docs/tui-stdio-harness.md
";

/// Direct CLI entry used before normal config/provider bootstrap.
pub fn run_cli(args: &[String]) -> Result<()> {
    match args {
        [arg] if arg == "--stdio" => {
            let stdin = std::io::stdin();
            let stdout = std::io::stdout();
            run_jsonl(stdin.lock(), stdout.lock())
        }
        _ if args.is_empty() || args == ["--help"] || args == ["-h"] => {
            print!("{HELP}");
            Ok(())
        }
        _ => bail!("usage: hi debug tui --stdio (or --help)"),
    }
}

/// Drive one harness for the lifetime of a JSONL stream.
#[doc(hidden)]
pub fn run_jsonl<R: BufRead, W: Write>(reader: R, mut writer: W) -> Result<()> {
    let mut harness = Harness::default();
    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading JSONL request at line {}", index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let response = if line.len() > MAX_REQUEST_BYTES {
            WireResponse::error(None, "request_too_large", "request exceeds 1 MiB")
        } else {
            match serde_json::from_str::<WireRequest>(&line) {
                Ok(request) => {
                    let id = request.id.clone();
                    match harness.handle(request.command) {
                        Ok(result) => WireResponse::success(id, result),
                        Err(error) => WireResponse::error(id, error.code, &error.message),
                    }
                }
                Err(error) => WireResponse::error(None, "invalid_json", &error.to_string()),
            }
        };
        serde_json::to_writer(&mut writer, &response).context("writing JSONL response")?;
        writer
            .write_all(b"\n")
            .context("terminating JSONL response")?;
        writer.flush().context("flushing JSONL response")?;
    }
    Ok(())
}

struct Harness {
    app: App,
    width: u16,
    height: u16,
    revision: u64,
}

impl Default for Harness {
    fn default() -> Self {
        Self::new(
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
            default_provider(),
            default_model(),
        )
    }
}

impl Harness {
    fn new(width: u16, height: u16, provider: String, model: String) -> Self {
        let mut app = harness_app(&provider, &model);
        app.configure_session_projection_v2(true);
        Self {
            app,
            width,
            height,
            revision: 0,
        }
    }

    fn handle(&mut self, command: Command) -> Result<Value, HarnessError> {
        match command {
            Command::Hello => Ok(json!({
                "commands": ["hello", "reset", "resize", "focus", "key", "paste",
                    "transcript", "clear_transcript", "session_event", "session_patch",
                    "session_snapshot", "render", "inspect"],
                "component_tree_schema_version": TUI_COMPONENT_TREE_SCHEMA_VERSION,
                "session_event_schema_version": hi_agent::SESSION_EVENT_SCHEMA_VERSION,
                "session_projection_schema_version": hi_agent::SESSION_PROJECTION_SCHEMA_VERSION,
                "session_reducer_version": hi_agent::SESSION_REDUCER_VERSION,
            })),
            Command::Reset {
                width,
                height,
                provider,
                model,
            } => {
                validate_dimensions(width, height)?;
                *self = Self::new(width, height, provider, model);
                Ok(self.ack())
            }
            Command::Resize { width, height } => {
                validate_dimensions(width, height)?;
                self.width = width;
                self.height = height;
                self.changed();
                Ok(self.ack())
            }
            Command::Focus { focused } => {
                self.app.set_focus(focused);
                self.changed();
                Ok(self.ack())
            }
            Command::Paste { text } => {
                self.app.input.insert_str(&text);
                self.changed();
                Ok(self.ack())
            }
            Command::Key {
                key,
                ctrl,
                alt,
                shift,
            } => {
                let submitted = self.apply_key(&key, ctrl, alt, shift)?;
                self.changed();
                Ok(json!({"revision": self.revision, "submitted": submitted}))
            }
            Command::Transcript { event } => {
                self.normalize_live_clocks();
                self.app
                    .try_apply(event)
                    .map_err(|error| HarnessError::new("invalid_session_projection", error))?;
                self.normalize_live_clocks();
                self.changed();
                Ok(self.ack())
            }
            Command::ClearTranscript => {
                clear_transcript(&mut self.app);
                self.changed();
                Ok(self.ack())
            }
            Command::SessionEvent { event } => {
                let patch = self
                    .app
                    .prepare_session_projection_patch(vec![event])
                    .map_err(|error| HarnessError::new("invalid_session_projection", error))?;
                self.app
                    .apply_session_projection_patch(patch)
                    .map_err(|error| HarnessError::new("invalid_session_projection", error))?;
                self.changed();
                Ok(json!({
                    "revision": self.revision,
                    "session_projection": self.app.session_projection_snapshot(),
                }))
            }
            Command::SessionPatch { patch } => {
                self.app
                    .apply_session_projection_patch(patch)
                    .map_err(|error| HarnessError::new("invalid_session_projection", error))?;
                self.changed();
                Ok(json!({
                    "revision": self.revision,
                    "session_projection": self.app.session_projection_snapshot(),
                }))
            }
            Command::SessionSnapshot { snapshot } => {
                self.app
                    .install_session_projection_snapshot(*snapshot)
                    .map_err(|error| HarnessError::new("invalid_session_projection", error))?;
                self.changed();
                Ok(json!({
                    "revision": self.revision,
                    "session_projection": self.app.session_projection_snapshot(),
                }))
            }
            Command::Render => {
                let snapshot = self
                    .render_snapshot()
                    .map_err(|error| HarnessError::new("render_failed", format!("{error:#}")))?;
                serde_json::to_value(snapshot)
                    .map_err(|error| HarnessError::new("serialization_failed", error.to_string()))
            }
            Command::Inspect => {
                let snapshot = self
                    .render_snapshot()
                    .map_err(|error| HarnessError::new("render_failed", format!("{error:#}")))?;
                Ok(json!({
                    "revision": self.revision,
                    "render_digest": snapshot.digest,
                    "component_tree": tree::build(&self.app, self.width, self.height, self.revision),
                    "session_projection": self.app.session_projection_snapshot(),
                }))
            }
        }
    }

    fn ack(&self) -> Value {
        json!({"revision": self.revision})
    }

    fn changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    fn apply_key(
        &mut self,
        name: &str,
        ctrl: bool,
        alt: bool,
        shift: bool,
    ) -> Result<Option<String>, HarnessError> {
        let code = parse_key(name)?;
        if ctrl
            && matches!(
                code,
                KeyCode::Char('x') | KeyCode::Char('y') | KeyCode::Char(' ')
            )
        {
            return Err(HarnessError::new(
                "external_input_disabled",
                "clipboard, editor, and voice chords are disabled in the stdio harness",
            ));
        }
        let mut modifiers = KeyModifiers::NONE;
        modifiers.set(KeyModifiers::CONTROL, ctrl);
        modifiers.set(KeyModifiers::ALT, alt);
        modifiers.set(KeyModifiers::SHIFT, shift);
        let key = KeyEvent::new(code, modifiers);

        if let Some(palette) = self.app.palette.as_mut() {
            return Ok(match palette.handle_key(&key) {
                crate::palette::PaletteOutcome::Continue => None,
                crate::palette::PaletteOutcome::Closed => {
                    self.app.palette = None;
                    None
                }
                crate::palette::PaletteOutcome::Accept(command) => {
                    self.app.palette = None;
                    self.app.input.set(&command);
                    None
                }
            });
        }
        match self.app.dispatch_key(&key) {
            DispatchResult::Handled => return Ok(None),
            DispatchResult::OpenPalette => {
                self.app.palette = Some(crate::palette::CommandPalette::open());
                return Ok(None);
            }
            DispatchResult::Fallthrough => {}
        }
        if self.app.mode.is_normal() {
            crate::app::handle_normal_mode(&mut self.app, &key);
            return Ok(None);
        }
        if ctrl && code == KeyCode::Char('r') && !self.app.input.history.is_empty() {
            let mut search = crate::input::HistorySearch::default();
            search.refilter(&self.app.input.history);
            if let Some(index) = search.current() {
                self.app.input.set(&self.app.input.history[index].clone());
            }
            self.app.mode = UiMode::HistorySearch(search);
            return Ok(None);
        }
        if code == KeyCode::Esc {
            if self.app.input.is_empty() {
                self.app.mode = UiMode::Normal { search: None };
            } else {
                self.app.input.clear();
            }
            return Ok(None);
        }
        let submitted = self.app.edit_key(&key);
        if let Some(text) = &submitted {
            self.app.push_user_prompt(Line::styled(
                format!("❯ {text}"),
                ratatui::style::Style::default().fg(crate::theme::theme().accent_user),
            ));
            self.app.last_prompt = Some(text.clone());
            self.app.follow();
        }
        Ok(submitted)
    }

    fn normalize_live_clocks(&mut self) {
        let now = Instant::now();
        if self.app.reasoning_started.is_some() {
            self.app.reasoning_started = Some(now);
        }
        if self.app.current_tool_started.is_some() {
            self.app.current_tool_started = Some(now);
        }
        if self.app.started.is_some() {
            self.app.started = Some(now);
        }
        for entry in &mut self.app.transcript {
            if let TranscriptEntry::Activity(block) = entry
                && let ActivityKind::Subagent { started_at, .. } = &mut block.kind
            {
                *started_at = now;
            }
        }
    }

    fn render_snapshot(&mut self) -> Result<RenderSnapshot> {
        validate_dimensions(self.width, self.height)
            .map_err(|error| anyhow::anyhow!(error.message))?;
        self.normalize_live_clocks();
        let backend = TestBackend::new(self.width, self.height);
        let mut terminal = Terminal::new(backend)?;
        terminal.draw(|frame| self.app.render(frame))?;
        let lines = buffer_lines(terminal.backend().buffer(), self.width);
        let cursor = terminal
            .backend_mut()
            .get_cursor_position()
            .ok()
            .map(Into::into);
        let mut digest = blake3::Hasher::new();
        digest.update(b"hi-tui-render-v1\0");
        digest.update(&self.width.to_le_bytes());
        digest.update(&self.height.to_le_bytes());
        for line in &lines {
            digest.update(&(line.len() as u64).to_le_bytes());
            digest.update(line.as_bytes());
        }
        Ok(RenderSnapshot {
            schema_version: 1,
            revision: self.revision,
            width: self.width,
            height: self.height,
            lines,
            cursor,
            digest: digest.finalize().to_hex().to_string(),
        })
    }
}

#[derive(Serialize)]
struct RenderSnapshot {
    schema_version: u16,
    revision: u64,
    width: u16,
    height: u16,
    lines: Vec<String>,
    cursor: Option<CursorSnapshot>,
    digest: String,
}

#[derive(Serialize)]
struct CursorSnapshot {
    x: u16,
    y: u16,
}

impl From<ratatui::layout::Position> for CursorSnapshot {
    fn from(position: ratatui::layout::Position) -> Self {
        Self {
            x: position.x,
            y: position.y,
        }
    }
}

fn harness_app(provider: &str, model: &str) -> App {
    let unavailable = || anyhow::anyhow!("unavailable in deterministic TUI harness");
    let mut app = App::new(
        provider,
        model,
        Vec::new(),
        None,
        Box::new(move |_| Err(unavailable())),
        Box::new(move |_| Err(unavailable())),
        Box::new(move |_| Err(unavailable())),
        Box::new(move |_| Err(unavailable())),
        None,
        Box::new(move |_| Err(unavailable())),
        Box::new(move |_| Err(unavailable())),
        None,
        String::new(),
        None,
        None,
        RaceDefaults::default(),
        None,
    );
    app.workspace_root = PathBuf::from("/workspace");
    app.timestamps_enabled = false;
    app
}

fn clear_transcript(app: &mut App) {
    app.transcript.clear();
    app.pending = None;
    app.reasoning_buffer.clear();
    app.reasoning_started = None;
    app.current_assistant.clear();
    app.current_assistant_streamed_bytes = 0;
    app.assistant_message_open = false;
    app.current_tool = None;
    app.current_tool_started = None;
    app.event_log.clear();
    app.last_assistant.clear();
    app.block_cursor = 0;
    app.scroll = 0;
    app.following = true;
    app.bump_transcript();
}

fn validate_dimensions(width: u16, height: u16) -> Result<(), HarnessError> {
    if width == 0 || height == 0 || width > MAX_WIDTH || height > MAX_HEIGHT {
        return Err(HarnessError::new(
            "invalid_dimensions",
            format!("dimensions must be within 1..={MAX_WIDTH} by 1..={MAX_HEIGHT}"),
        ));
    }
    Ok(())
}

fn parse_key(name: &str) -> Result<KeyCode, HarnessError> {
    let normalized = name.trim().to_ascii_lowercase().replace('-', "_");
    let named = match normalized.as_str() {
        "enter" => Some(KeyCode::Enter),
        "esc" | "escape" => Some(KeyCode::Esc),
        "backspace" => Some(KeyCode::Backspace),
        "delete" => Some(KeyCode::Delete),
        "left" => Some(KeyCode::Left),
        "right" => Some(KeyCode::Right),
        "up" => Some(KeyCode::Up),
        "down" => Some(KeyCode::Down),
        "home" => Some(KeyCode::Home),
        "end" => Some(KeyCode::End),
        "page_up" | "pageup" => Some(KeyCode::PageUp),
        "page_down" | "pagedown" => Some(KeyCode::PageDown),
        "tab" => Some(KeyCode::Tab),
        "back_tab" | "backtab" => Some(KeyCode::BackTab),
        "space" => Some(KeyCode::Char(' ')),
        _ => None,
    };
    if let Some(key) = named {
        return Ok(key);
    }
    let literal = name.strip_prefix("char:").unwrap_or(name);
    let mut chars = literal.chars();
    match (chars.next(), chars.next()) {
        (Some(character), None) => Ok(KeyCode::Char(character)),
        _ => Err(HarnessError::new(
            "invalid_key",
            format!("unknown key {name:?}; use a named key, one character, or char:<character>"),
        )),
    }
}

fn buffer_lines(buffer: &ratatui::buffer::Buffer, width: u16) -> Vec<String> {
    buffer
        .content()
        .chunks(width as usize)
        .map(|row| {
            let mut line = String::new();
            let mut skip = 0usize;
            for cell in row {
                if skip == 0 {
                    line.push_str(cell.symbol());
                }
                skip = skip
                    .max(UnicodeWidthStr::width(cell.symbol()))
                    .saturating_sub(1);
            }
            line.trim_end().to_owned()
        })
        .collect()
}

fn default_width() -> u16 {
    DEFAULT_WIDTH
}

fn default_height() -> u16 {
    DEFAULT_HEIGHT
}

fn default_provider() -> String {
    "debug".to_owned()
}

fn default_model() -> String {
    "debug-model".to_owned()
}

#[cfg(test)]
#[path = "debug_harness/tests.rs"]
mod tests;
