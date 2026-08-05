//! The agent→UI event channel: the agent emits [`UiEvent`]s over an mpsc
//! channel so the event loop can keep redrawing while a turn is in flight.

use std::{io, sync::Arc};

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
    /// Assistant text answering a `/btw` side question — rendered in the BTW
    /// side pane (and optionally as a dim main-transcript stub).
    BtwAnswer {
        text: String,
    },
    /// User `/btw` question — opens/feeds the BTW pane thread.
    BtwQuestion {
        question: String,
    },
    /// Read-only tool started inside a `/btw` side loop (pane timeline only).
    BtwToolStarted {
        name: String,
        arguments: String,
    },
    /// Read-only tool finished inside a `/btw` side loop.
    BtwToolResult {
        name: String,
        result: String,
    },
    /// Current `/btw` answer stream finished.
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
    /// Revisioned workflow lifecycle state. Receivers must ignore stale
    /// revisions, including updates for runs already tombstoned by a terminal
    /// snapshot.
    WorkflowUpdated {
        snapshot: hi_workflow::WorkflowRunSnapshot,
    },
    /// Bounded Diff Lab progress; full traces remain in the run artifact store.
    DiffRunUpdated {
        snapshot: hi_diff::DiffRunSnapshot,
    },
}

/// The [`Ui`] handed to the agent: forwards everything over a channel so the
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

    fn approval_request(
        &self,
        request: &ConfirmationRequest,
    ) -> Option<(hi_policy::CapabilityRequest, hi_policy::OperationDigest)> {
        let workspace_id = std::env::current_dir()
            .ok()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "interactive".into());
        let (capability, scope, tool, arguments, detail) = match request {
            ConfirmationRequest::FileEdit { path, diff } => (
                hi_policy::CapabilityKind::WorkspaceWrite,
                hi_policy::ResourceScope::Paths {
                    workspace_id: workspace_id.clone(),
                    paths: vec![path.clone()],
                },
                "edit",
                serde_json::json!({ "path": path, "diff": diff }),
                format!("file edit requested for {path}"),
            ),
            ConfirmationRequest::ShellMutation { command, cwd } => (
                hi_policy::CapabilityKind::ProcessExecution,
                hi_policy::ResourceScope::Command {
                    workspace_id: workspace_id.clone(),
                    command: hi_policy::normalize_command(command),
                    cwd: hi_policy::normalize_cwd(cwd),
                },
                "bash",
                serde_json::json!({
                    "command": hi_policy::normalize_command(command),
                    "cwd": hi_policy::normalize_cwd(cwd)
                }),
                format!("shell mutation requested in {cwd}"),
            ),
            ConfirmationRequest::DelegateApply { summary, diff } => (
                hi_policy::CapabilityKind::DelegateApplication,
                hi_policy::ResourceScope::Operation {
                    workspace_id: workspace_id.clone(),
                    label: summary.clone(),
                },
                "delegate",
                serde_json::json!({ "summary": summary, "diff": diff }),
                "verified delegated application requested".into(),
            ),
        };
        let digest = hi_policy::OperationDigest::calculate(
            &capability,
            tool,
            &arguments,
            &workspace_id,
            &scope,
            match request {
                ConfirmationRequest::FileEdit { diff, .. }
                | ConfirmationRequest::DelegateApply { diff, .. } => Some(diff.as_str()),
                ConfirmationRequest::ShellMutation { .. } => None,
            },
        );
        let capability_request = hi_policy::approval_request(
            capability,
            scope,
            digest.clone(),
            tool,
            None,
            None,
            request.title(),
            detail,
        );
        Some((capability_request, digest))
    }

    fn approval_event(
        &self,
        kind: hi_events::EventKind,
        state: hi_events::ActivityState,
        id: &str,
        title: &str,
    ) {
        self.semantic(hi_events::RunEvent::new(
            kind,
            hi_events::EventContext::default(),
            hi_events::SemanticActivity {
                verb: match state {
                    hi_events::ActivityState::Denied => hi_events::ActivityVerb::Deny,
                    hi_events::ActivityState::Succeeded => hi_events::ActivityVerb::Approve,
                    _ => hi_events::ActivityVerb::Wait,
                },
                object: hi_events::ActivityObject::Approval,
                state,
                group_key: format!("approval:{id}"),
                title: title.into(),
                detail: None,
                refs: vec![hi_events::ActivityRef {
                    kind: "approval".into(),
                    id: id.into(),
                }],
                progress: None,
            },
        ));
    }
}

/// Presentation adapter kept separate from the durable event contract. It
/// intentionally emits only compact status/turn markers; raw detail remains in
/// the originating session or workflow artifact.
pub(crate) fn canonical_to_ui_event(event: &hi_events::RunEvent) -> Option<UiEvent> {
    let text = event.activity.title.clone();
    match event.kind {
        hi_events::EventKind::RunStarted
        | hi_events::EventKind::RunWaiting
        | hi_events::EventKind::RunResumed
        | hi_events::EventKind::VerificationStarted
        | hi_events::EventKind::ApprovalDecided
        | hi_events::EventKind::ApprovalConsumed
        | hi_events::EventKind::CapabilityRequested
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
        | hi_events::EventKind::TriggerFailed => Some(UiEvent::Status { text }),
        hi_events::EventKind::VerificationCompleted | hi_events::EventKind::GitChanged => {
            Some(UiEvent::Status { text })
        }
        hi_events::EventKind::RunCompleted => Some(UiEvent::TurnEnd { summary: text }),
        hi_events::EventKind::RunCancelled => Some(UiEvent::Status { text }),
        hi_events::EventKind::RunFailed
        | hi_events::EventKind::WorkflowCompleted
        | hi_events::EventKind::WorkflowFailed
        | hi_events::EventKind::ToolDenied
        | hi_events::EventKind::ToolTimedOut => Some(UiEvent::Status { text }),
        hi_events::EventKind::ToolRequested
        | hi_events::EventKind::ToolStarted
        | hi_events::EventKind::ToolCompleted => None,
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
    fn semantic_event(&mut self, event: hi_events::RunEvent) {
        self.semantic(event);
    }
    fn assistant_text(&mut self, text: &str) {
        self.send(UiEvent::Text {
            text: text.to_string(),
        });
    }
    fn btw_answer(&mut self, text: &str) {
        self.send(UiEvent::BtwAnswer {
            text: text.to_string(),
        });
    }
    fn btw_question(&mut self, question: &str) {
        self.send(UiEvent::BtwQuestion {
            question: question.to_string(),
        });
    }
    fn btw_tool_started(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::BtwToolStarted {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn btw_tool_result(&mut self, name: &str, result: &str) {
        self.send(UiEvent::BtwToolResult {
            name: name.to_string(),
            result: result.to_string(),
        });
    }
    fn btw_end(&mut self) {
        self.send(UiEvent::BtwEnd);
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
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.send(UiEvent::ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_result_id(&mut self, id: &str, name: &str, result: &str, status: hi_tools::ToolStatus) {
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
        self.send(UiEvent::ToolResult {
            name: name.to_string(),
            result: result.to_string(),
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
        let durable = self
            .approval_store
            .as_ref()
            .and_then(|_| self.approval_request(&request));
        let approval_store = self.approval_store.clone();
        let event_sink = self.event_sink.clone();
        let request_for_control = request.clone();
        let (approval_id, digest) =
            if let (Some(store), Some((request, digest))) = (approval_store.as_ref(), durable) {
                match store.create(request.clone()) {
                    Ok(record) => {
                        let id = record.request.approval_id.0.clone();
                        self.approval_event(
                            hi_events::EventKind::CapabilityRequested,
                            hi_events::ActivityState::Waiting,
                            &id,
                            &record.request.title,
                        );
                        (Some(record.request.approval_id), Some(digest))
                    }
                    Err(_) => return Box::pin(async { ConfirmationResult::Unavailable }),
                }
            } else {
                (None, None)
            };
        let (response, answer) = tokio::sync::oneshot::channel();
        if self
            .confirmations
            .send(ConfirmationControl {
                request: request_for_control,
                response,
            })
            .is_err()
        {
            return Box::pin(async { ConfirmationResult::Unavailable });
        }
        Box::pin(async move {
            let decision = answer.await.unwrap_or(ConfirmationResult::Cancelled);
            let Some(id) = approval_id else {
                return decision;
            };
            let Some(store) = approval_store else {
                return ConfirmationResult::Unavailable;
            };
            let mapped = match decision {
                ConfirmationResult::Approved => hi_policy::ApprovalDecision::Approved,
                ConfirmationResult::Rejected => hi_policy::ApprovalDecision::Denied,
                ConfirmationResult::Cancelled => hi_policy::ApprovalDecision::Cancelled,
                ConfirmationResult::Unavailable => hi_policy::ApprovalDecision::Unavailable,
            };
            if store.decide(&id, mapped).is_err() {
                return ConfirmationResult::Unavailable;
            }
            let Some(digest) = digest else {
                return decision;
            };
            if decision != ConfirmationResult::Approved {
                if let Some(sink) = &event_sink {
                    let _ = sink.publish(hi_events::RunEvent::new(
                        hi_events::EventKind::ApprovalDecided,
                        hi_events::EventContext::default(),
                        hi_events::SemanticActivity {
                            verb: hi_events::ActivityVerb::Deny,
                            object: hi_events::ActivityObject::Approval,
                            state: hi_events::ActivityState::Denied,
                            group_key: format!("approval:{}", id.0),
                            title: "Approval denied".into(),
                            detail: None,
                            refs: vec![],
                            progress: None,
                        },
                    ));
                }
                return decision;
            }
            if store.claim(&id, &digest).is_err() {
                return ConfirmationResult::Unavailable;
            }
            if let Some(sink) = &event_sink {
                let _ = sink.publish(hi_events::RunEvent::new(
                    hi_events::EventKind::ApprovalConsumed,
                    hi_events::EventContext::default(),
                    hi_events::SemanticActivity {
                        verb: hi_events::ActivityVerb::Approve,
                        object: hi_events::ActivityObject::Approval,
                        state: hi_events::ActivityState::Succeeded,
                        group_key: format!("approval:{}", id.0),
                        title: "Approval consumed".into(),
                        detail: None,
                        refs: vec![],
                        progress: None,
                    },
                ));
            }
            ConfirmationResult::Approved
        })
    }
    fn status(&mut self, text: &str) {
        let Some(text) = hi_agent::ui::user_facing_status(text) else {
            return;
        };
        self.send(UiEvent::Status { text });
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
