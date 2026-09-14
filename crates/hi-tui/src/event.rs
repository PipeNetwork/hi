//! The harness→UI event channel: the turn loop emits [`UiEvent`]s over an mpsc
//! channel so the event loop can keep redrawing while a turn is in flight.

use std::{io, sync::Arc};

use crossterm::event::{DisableBracketedPaste, DisableFocusChange, DisableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{LeaveAlternateScreen, disable_raw_mode};
use hi_harness::{ConfirmationRequest, ConfirmationResult, Ui};
use hi_tools::PlanStep;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Events the harness emits; drained by the event loop into `App`.
///
/// `pub` and `Serialize` so they can be relayed over the network to a remote
/// viewer (Phase 2 live streaming). The `#[serde(tag = "kind")]` makes each
/// event a self-describing JSON object: `{"kind":"text","text":"..."}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UiEvent {
    ProviderAttempt {
        event: hi_ai::ProviderAttemptEvent,
    },
    ProviderProgress,
    ProviderRequest {
        audit: serde_json::Value,
    },
    Text {
        text: String,
    },
    BtwAnswer {
        text: String,
    },
    BtwQuestion {
        question: String,
    },
    BtwToolStarted {
        name: String,
        arguments: String,
    },
    BtwToolResult {
        name: String,
        result: String,
    },
    BtwEnd,
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
    ToolStream {
        name: String,
        line: String,
    },
    Status {
        text: String,
    },
    TopStatus {
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
    SessionUsage {
        usage: hi_ai::Usage,
    },
    RateLimits {
        rate_limits: Option<hi_ai::RateLimitState>,
    },
    TurnEnd {
        summary: String,
    },
    TurnError {
        error_kind: String,
        message: String,
        guidance: String,
    },
    ChangedFiles {
        files: Vec<String>,
    },
    SuggestedPrompt {
        text: String,
    },
    SubagentSpawned {
        id: String,
        subagent_kind: String,
        description: String,
        background: bool,
    },
    SubagentProgress {
        id: String,
        activity: String,
        #[serde(default)]
        line: Option<String>,
    },
    SubagentFinished {
        id: String,
        status: String,
        elapsed_ms: u64,
        summary: String,
    },
    WorkflowUpdated {
        snapshot: hi_workflow::WorkflowRunSnapshot,
    },
    DiffRunUpdated {
        snapshot: hi_diff::DiffRunSnapshot,
    },
}

/// The [`Ui`] handed to the harness: forwards everything over a channel so the
/// turn never borrows the live `App`.
pub(crate) struct ChannelUi {
    pub tx: mpsc::UnboundedSender<UiEvent>,
    pub confirmations: mpsc::UnboundedSender<ConfirmationControl>,
    pub event_sink: Option<Arc<dyn hi_events::EventSink>>,
    pub approval_store: Option<Arc<dyn hi_policy::ApprovalStore>>,
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

    fn semantic(&self, event: hi_events::RunEvent) {
        if let Some(ui_event) = canonical_to_ui_event(&event) {
            self.send(ui_event);
        }
        if let Some(sink) = &self.event_sink {
            let _ = sink.publish(event);
        }
    }

    fn tool_event(
        &self,
        kind: hi_events::EventKind,
        verb: hi_events::ActivityVerb,
        state: hi_events::ActivityState,
        id: &str,
        name: &str,
    ) {
        self.semantic(hi_events::RunEvent::new(
            kind,
            hi_events::EventContext::default(),
            hi_events::SemanticActivity {
                verb,
                object: hi_events::ActivityObject::Tool,
                state,
                group_key: format!("tool:{id}"),
                title: name.to_string(),
                detail: Some(format!("tool {name}")),
                refs: vec![hi_events::ActivityRef {
                    kind: "tool".into(),
                    id: id.into(),
                }],
                progress: None,
            },
        ));
    }

    fn terminal_tool_event(&self, id: &str, name: &str, status: hi_tools::ToolStatus) {
        let (kind, verb, state) = match status {
            hi_tools::ToolStatus::Succeeded => (
                hi_events::EventKind::ToolCompleted,
                hi_events::ActivityVerb::Complete,
                hi_events::ActivityState::Succeeded,
            ),
            hi_tools::ToolStatus::TimedOut => (
                hi_events::EventKind::ToolTimedOut,
                hi_events::ActivityVerb::Fail,
                hi_events::ActivityState::TimedOut,
            ),
            hi_tools::ToolStatus::Denied => (
                hi_events::EventKind::ToolDenied,
                hi_events::ActivityVerb::Deny,
                hi_events::ActivityState::Denied,
            ),
            _ => (
                hi_events::EventKind::ToolCompleted,
                hi_events::ActivityVerb::Fail,
                hi_events::ActivityState::Failed,
            ),
        };
        self.tool_event(kind, verb, state, id, name);
    }
}

/// Presentation adapter kept separate from the durable event contract. It
/// intentionally emits only compact status/turn markers; raw detail remains in
/// the originating session or workflow artifact.
pub(crate) fn canonical_to_ui_event(event: &hi_events::RunEvent) -> Option<UiEvent> {
    let text = event.activity.title.clone();
    match event.kind {
        hi_events::EventKind::RunStarted
        | hi_events::EventKind::AttemptClaimed
        | hi_events::EventKind::AttemptRenewed
        | hi_events::EventKind::AttemptLeaseLost
        | hi_events::EventKind::AttemptCompleted
        | hi_events::EventKind::AttemptFailed => None,
        hi_events::EventKind::RunWaiting
        | hi_events::EventKind::RunResumed
        | hi_events::EventKind::ApprovalDecided
        | hi_events::EventKind::ApprovalConsumed
        | hi_events::EventKind::PolicyEvaluated
        | hi_events::EventKind::RouteSelected
        | hi_events::EventKind::EffectPlanned
        | hi_events::EventKind::EffectStarted
        | hi_events::EventKind::EffectCompleted
        | hi_events::EventKind::EffectFailed
        | hi_events::EventKind::EffectDenied
        | hi_events::EventKind::EffectUnknown
        | hi_events::EventKind::EffectReconciled
        | hi_events::EventKind::AuditRecorded
        | hi_events::EventKind::WorkflowStarted
        | hi_events::EventKind::WorkflowPaused
        | hi_events::EventKind::WorkflowResumed
        | hi_events::EventKind::PhaseStarted
        | hi_events::EventKind::PhaseCompleted
        | hi_events::EventKind::LoopFired
        | hi_events::EventKind::TriggerAccepted
        | hi_events::EventKind::TriggerSkipped
        | hi_events::EventKind::TriggerStarted
        | hi_events::EventKind::TriggerCompleted
        | hi_events::EventKind::TriggerFailed
        | hi_events::EventKind::RaceStarted
        | hi_events::EventKind::RaceCandidateStarted
        | hi_events::EventKind::RaceCandidateCompleted
        | hi_events::EventKind::RaceCandidateScored
        | hi_events::EventKind::RaceWinnerReady
        | hi_events::EventKind::RaceApplied
        | hi_events::EventKind::RaceCancelled
        | hi_events::EventKind::RaceWorkspaceConflict => Some(UiEvent::Status { text }),
        hi_events::EventKind::GitChanged => Some(UiEvent::Status { text }),
        hi_events::EventKind::RunCompleted => None,
        hi_events::EventKind::RunCancelled => Some(UiEvent::Status { text }),
        hi_events::EventKind::RunFailed
        | hi_events::EventKind::WorkflowCompleted
        | hi_events::EventKind::WorkflowFailed
        | hi_events::EventKind::ToolDenied
        | hi_events::EventKind::ToolTimedOut => Some(UiEvent::Status { text }),
        hi_events::EventKind::ToolRequested
        | hi_events::EventKind::ToolStarted
        | hi_events::EventKind::ToolCompleted
        | hi_events::EventKind::CapabilityRequested
        | hi_events::EventKind::VerificationStarted
        | hi_events::EventKind::VerificationCompleted => None,
    }
}

/// Redacted remote-observation payload. A remote viewer can render this, but
/// no approval response is represented in the payload or accepted here.
pub fn remote_sync_payload(event: &hi_events::RunEvent) -> serde_json::Value {
    serde_json::to_value(event).unwrap_or_else(|_| {
        serde_json::json!({
            "schema_version": hi_events::EVENT_SCHEMA_VERSION,
            "event_id": event.event_id,
            "kind": "serialization_error",
        })
    })
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
    fn tool_started_id(&mut self, id: &str, name: &str, arguments: &str) {
        self.tool_event(
            hi_events::EventKind::ToolStarted,
            hi_events::ActivityVerb::Execute,
            hi_events::ActivityState::Running,
            id,
            name,
        );
        self.send(UiEvent::ToolStarted {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        self.send(UiEvent::ToolStream {
            name: name.to_string(),
            line: line.to_string(),
        });
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_call_id(&mut self, id: &str, name: &str, arguments: &str) {
        self.tool_event(
            hi_events::EventKind::ToolRequested,
            hi_events::ActivityVerb::Execute,
            hi_events::ActivityState::Pending,
            id,
            name,
        );
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
    fn tool_result_id(&mut self, id: &str, name: &str, result: &str, status: hi_tools::ToolStatus) {
        self.terminal_tool_event(id, name, status);
        self.send(UiEvent::ToolResult {
            name: name.to_string(),
            result: result.to_string(),
        });
    }
    fn plan(&mut self, steps: &[PlanStep]) {
        self.send(UiEvent::Plan {
            steps: steps.to_vec(),
        });
    }
    fn plan_result_id(
        &mut self,
        id: &str,
        name: &str,
        _result: &str,
        status: hi_tools::ToolStatus,
        steps: &[PlanStep],
    ) {
        self.terminal_tool_event(id, name, status);
        self.send(UiEvent::Plan {
            steps: steps.to_vec(),
        });
    }
    fn status(&mut self, text: &str) {
        self.send(UiEvent::Status {
            text: text.to_string(),
        });
    }
    fn top_status(&mut self, text: &str) {
        self.send(UiEvent::TopStatus {
            text: text.to_string(),
        });
    }
    fn checkpoint_warning(&mut self, text: &str) {
        self.send(UiEvent::CheckpointWarning {
            text: text.to_string(),
        });
    }
    fn usage(
        &mut self,
        prompt: u64,
        generated: u64,
        ctx_used: u64,
        ctx_window: Option<u32>,
        estimated: bool,
    ) {
        self.send(UiEvent::Usage {
            prompt,
            generated,
            ctx_used,
            ctx_window,
            estimated,
        });
    }
    fn session_usage(&mut self, usage: hi_ai::Usage) {
        self.send(UiEvent::SessionUsage { usage });
    }
    fn turn_end(&mut self, summary: &str) {
        self.send(UiEvent::TurnEnd {
            summary: summary.to_string(),
        });
    }
    fn turn_error(&mut self, error_kind: &str, message: &str, guidance: &str) {
        self.send(UiEvent::TurnError {
            error_kind: error_kind.to_string(),
            message: message.to_string(),
            guidance: guidance.to_string(),
        });
    }
    fn changed_files(&mut self, files: Vec<String>) {
        self.send(UiEvent::ChangedFiles { files });
    }
    fn confirm(&mut self, request: ConfirmationRequest) -> hi_harness::ConfirmationFuture<'_> {
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
}

/// Restores the terminal on drop (covers early returns and panics).
pub(crate) struct Restore;
impl Restore {
    pub(crate) fn restore_now(self) {
        Self::restore_now_static();
    }

    pub(crate) fn restore_now_static() {
        let _ = disable_raw_mode();
        let _ = write_leave_session_screen(&mut io::stdout());
    }
}

/// Crossterm bytes that drop the alternate screen. `exec(2)` skips Drop, so
/// `/autoharnessfix on` must emit these before replacing the process.
pub(crate) fn write_leave_session_screen(out: &mut impl io::Write) -> io::Result<()> {
    execute!(
        out,
        DisableMouseCapture,
        DisableFocusChange,
        DisableBracketedPaste,
        LeaveAlternateScreen
    )
}
impl Drop for Restore {
    fn drop(&mut self) {
        Self::restore_now_static();
    }
}

#[cfg(test)]
#[path = "event_tests.rs"]
mod tests;
