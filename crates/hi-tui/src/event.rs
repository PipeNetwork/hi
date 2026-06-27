//! The agent→UI event channel: the agent emits [`UiEvent`]s over an mpsc
//! channel so the event loop can keep redrawing while a turn is in flight.

use std::io;

use crossterm::event::{DisableBracketedPaste, DisableFocusChange};
use crossterm::execute;
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};
use hi_agent::{PlanStep, Ui};
use tokio::sync::mpsc;

/// Events the agent emits; drained by the event loop into `App`.
pub(crate) enum UiEvent {
    Text(String),
    Reasoning(String),
    AssistantEnd,
    ToolStarted(String, String),
    ToolCall(String, String),
    ToolResult(String, String),
    /// A live line of output from a running tool (e.g. bash stdout).
    ToolStream(String, String),
    Status(String),
    Plan(Vec<PlanStep>),
    Usage {
        input: u64,
        output: u64,
        ctx_used: u64,
        ctx_window: Option<u32>,
    },
    TurnEnd(String),
    /// A classified turn failure: (kind slug, raw message, guidance hint).
    TurnError(String, String, String),
    /// Files changed during the turn.
    ChangedFiles(Vec<String>),
}

/// The [`Ui`] handed to the agent: forwards everything over a channel so the
/// turn never borrows the live `App`.
pub(crate) struct ChannelUi {
    pub tx: mpsc::UnboundedSender<UiEvent>,
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
    fn tool_started(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::ToolStarted(
            name.to_string(),
            arguments.to_string(),
        ));
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::ToolCall(name.to_string(), arguments.to_string()));
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.send(UiEvent::ToolResult(name.to_string(), result.to_string()));
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        self.send(UiEvent::ToolStream(name.to_string(), line.to_string()));
    }
    // confirm_edit: the TUI's ChannelUi is async-by-channel and can't
    // synchronously wait for a user response. --confirm-edits is primarily
    // a REPL/plain-mode feature. The TUI auto-approves (returns true via
    // the default impl). A future improvement could add a confirmation
    // overlay with a oneshot channel.
    fn status(&mut self, text: &str) {
        self.send(UiEvent::Status(text.to_string()));
    }
    fn plan(&mut self, steps: &[PlanStep]) {
        self.send(UiEvent::Plan(steps.to_vec()));
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
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.send(UiEvent::TurnError(
            kind.to_string(),
            message.to_string(),
            guidance.to_string(),
        ));
    }
    fn changed_files(&mut self, files: &[String]) {
        self.send(UiEvent::ChangedFiles(files.to_vec()));
    }
}

/// Restores the terminal on drop (covers early returns and panics).
pub(crate) struct Restore;
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
