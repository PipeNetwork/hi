//! The agent→UI event channel: the agent emits [`UiEvent`]s over an mpsc
//! channel so the event loop can keep redrawing while a turn is in flight.

use std::io;

use crossterm::event::{DisableBracketedPaste, DisableFocusChange, DisableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};
use hi_agent::{ConfirmationFuture, ConfirmationRequest, ConfirmationResult, PlanStep, Ui};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Events the agent emits; drained by the event loop into `App`.
///
/// `pub` and `Serialize` so they can be relayed over the network to a remote
/// viewer (Phase 2 live streaming). The `#[serde(tag = "kind")]` makes each
/// event a self-describing JSON object: `{"kind":"text","text":"..."}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UiEvent {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    AssistantEnd,
    ToolStarted {
        name: String,
        arguments: String,
    },
    ToolCall {
        name: String,
        arguments: String,
    },
    ToolResult {
        name: String,
        result: String,
    },
    /// A live line of output from a running tool (e.g. bash stdout).
    ToolStream {
        name: String,
        line: String,
    },
    Status {
        text: String,
    },
    CheckpointWarning {
        text: String,
    },
    Plan {
        steps: Vec<PlanStep>,
    },
    Usage {
        prompt: u64,
        generated: u64,
        ctx_used: u64,
        ctx_window: Option<u32>,
        #[serde(default)]
        estimated: bool,
    },
    RateLimits {
        rate_limits: Option<hi_ai::RateLimitState>,
    },
    TurnEnd {
        summary: String,
    },
    /// A classified turn failure: (error_kind slug, raw message, guidance hint).
    TurnError {
        error_kind: String,
        message: String,
        guidance: String,
    },
    /// Files changed during the turn.
    ChangedFiles {
        files: Vec<String>,
    },
}

/// The [`Ui`] handed to the agent: forwards everything over a channel so the
/// turn never borrows the live `App`.
pub(crate) struct ChannelUi {
    pub tx: mpsc::UnboundedSender<UiEvent>,
    pub confirmations: mpsc::UnboundedSender<ConfirmationControl>,
}

/// Local-only control message. Confirmation responses are deliberately not
/// serialized as UiEvents or mirrored to remote viewers.
pub(crate) struct ConfirmationControl {
    pub request: ConfirmationRequest,
    pub response: tokio::sync::oneshot::Sender<ConfirmationResult>,
}

impl ChannelUi {
    fn send(&self, event: UiEvent) {
        let _ = self.tx.send(event);
    }
}

impl Ui for ChannelUi {
    fn assistant_text(&mut self, text: &str) {
        self.send(UiEvent::Text {
            text: text.to_string(),
        });
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.send(UiEvent::Reasoning {
            text: text.to_string(),
        });
    }
    fn assistant_end(&mut self) {
        self.send(UiEvent::AssistantEnd);
    }
    fn tool_started(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::ToolStarted {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.send(UiEvent::ToolResult {
            name: name.to_string(),
            result: result.to_string(),
        });
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        self.send(UiEvent::ToolStream {
            name: name.to_string(),
            line: line.to_string(),
        });
    }
    fn confirm(&mut self, request: ConfirmationRequest) -> ConfirmationFuture<'_> {
        let (response, answer) = tokio::sync::oneshot::channel();
        if self
            .confirmations
            .send(ConfirmationControl { request, response })
            .is_err()
        {
            return Box::pin(async { ConfirmationResult::Unavailable });
        }
        Box::pin(async move { answer.await.unwrap_or(ConfirmationResult::Cancelled) })
    }
    fn status(&mut self, text: &str) {
        self.send(UiEvent::Status {
            text: text.to_string(),
        });
    }
    fn checkpoint_warning(&mut self, text: &str) {
        self.send(UiEvent::CheckpointWarning {
            text: text.to_string(),
        });
    }
    fn plan(&mut self, steps: &[PlanStep]) {
        self.send(UiEvent::Plan {
            steps: steps.to_vec(),
        });
    }
    fn usage(
        &mut self,
        prompt_tokens: u64,
        generated_tokens: u64,
        context_used: u64,
        context_window: Option<u32>,
        usage_estimated: bool,
    ) {
        self.send(UiEvent::Usage {
            prompt: prompt_tokens,
            generated: generated_tokens,
            ctx_used: context_used,
            ctx_window: context_window,
            estimated: usage_estimated,
        });
    }
    fn turn_end(&mut self, summary: &str) {
        self.send(UiEvent::TurnEnd {
            summary: summary.to_string(),
        });
    }
    fn rate_limits(&mut self, rate_limits: Option<hi_ai::RateLimitState>) {
        self.send(UiEvent::RateLimits { rate_limits });
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.send(UiEvent::TurnError {
            error_kind: kind.to_string(),
            message: message.to_string(),
            guidance: guidance.to_string(),
        });
    }
    fn changed_files(&mut self, files: &[String]) {
        self.send(UiEvent::ChangedFiles {
            files: files.to_vec(),
        });
    }
}

/// Restores the terminal on drop (covers early returns and panics).
pub(crate) struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            io::stdout(),
            DisableMouseCapture,
            DisableFocusChange,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
    }
}
