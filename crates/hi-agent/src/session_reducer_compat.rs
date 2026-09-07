//! Compatibility decoding and inference for pre-reducer session JSONL.

use hi_ai::{Message, Role};
use serde::Deserialize;

use crate::session_reducer::{
    SESSION_EVENT_SCHEMA_VERSION, SessionEvent, SessionEventKind, SessionReduceError,
};
use crate::{
    Decision, Goal, PlanStep, TranscriptBlockId, TranscriptBlockKind, TranscriptBlockTerminal,
    TurnStatus, TurnStopReason,
};

impl SessionEvent {
    /// Strict decoding for restoration. Torn/unknown legacy records remain
    /// opaque; recognized safety/recovery records and canonical envelopes must
    /// not silently disappear and reset durable control state.
    pub fn decode_legacy_json(line: &str) -> Result<Option<Self>, SessionReduceError> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(None);
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => return Ok(Some(Self::opaque_boundary())),
        };
        if value.get("kind").is_some() && value.get("schema_version").is_some() {
            let schema = value["schema_version"].as_u64().unwrap_or(u64::MAX);
            if schema != u64::from(SESSION_EVENT_SCHEMA_VERSION) {
                return Err(SessionReduceError::InvalidRecord(format!(
                    "unsupported session event schema {schema}; this binary supports {SESSION_EVENT_SCHEMA_VERSION}"
                )));
            }
            return serde_json::from_value(value).map(Some).map_err(|error| {
                SessionReduceError::InvalidRecord(format!(
                    "invalid canonical session event: {error}"
                ))
            });
        }
        let settlement_state = value.get("type").and_then(serde_json::Value::as_str)
            == Some("turn_outcome")
            && (value.get("task_recovery").is_some() || value.get("settled_goal").is_some());
        let critical = settlement_state
            || matches!(
                value.get("type").and_then(serde_json::Value::as_str),
                Some(
                    "task_recovery"
                        | "workspace_execution_staged"
                        | "workspace_execution_settled"
                        | "remote_session_identity"
                )
            );
        if value.get("type").is_some() {
            return match serde_json::from_value::<LegacySessionMeta>(value) {
                Ok(meta) => Ok(Some(Self::new(meta.into()))),
                Err(error) if critical => Err(SessionReduceError::InvalidRecord(format!(
                    "invalid durable session record: {error}"
                ))),
                Err(_) => Ok(Some(Self::opaque_boundary())),
            };
        }
        Ok(Some(Self::new(
            serde_json::from_value::<Message>(value)
                .map(|message| SessionEventKind::Message { message })
                .unwrap_or(SessionEventKind::OpaqueBoundary),
        )))
    }

    /// Decode an existing remote record. Remote messages are bare messages;
    /// all metadata retains the tagged local wire representation.
    pub fn decode_remote_record(
        record_type: &str,
        payload_json: &str,
    ) -> Result<Self, SessionReduceError> {
        if record_type == "message" {
            return Ok(Self::new(
                serde_json::from_str::<Message>(payload_json)
                    .map(|message| SessionEventKind::Message { message })
                    .unwrap_or(SessionEventKind::OpaqueBoundary),
            ));
        }
        let mut event =
            Self::decode_legacy_json(payload_json)?.unwrap_or_else(Self::opaque_boundary);
        let wrong_critical_type = match record_type {
            "task_recovery" => !matches!(event.kind, SessionEventKind::TaskRecovery { .. }),
            "turn_outcome" => !matches!(event.kind, SessionEventKind::TurnOutcome { .. }),
            "workspace_execution_staged" => !matches!(
                event.kind,
                SessionEventKind::WorkspaceExecutionStaged { .. }
            ),
            "workspace_execution_settled" => !matches!(
                event.kind,
                SessionEventKind::WorkspaceExecutionSettled { .. }
            ),
            "remote_session_identity" => {
                !matches!(event.kind, SessionEventKind::RemoteSessionIdentity { .. })
            }
            _ => false,
        };
        if wrong_critical_type {
            return Err(SessionReduceError::InvalidRecord(format!(
                "invalid remote {record_type} record"
            )));
        }
        if record_type != "session_event" && matches!(event.kind, SessionEventKind::Message { .. })
        {
            event = Self::opaque_boundary();
        }
        Ok(event)
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, Deserialize)]
pub(super) struct LegacyPlanPauseMigration {
    cancellation_candidate: bool,
    inferred_interruption_chain: bool,
    inferred_pause_active: bool,
    pending_real_user_turn: Option<bool>,
}

impl LegacyPlanPauseMigration {
    pub(super) fn note_state_replacement(&mut self, before: &[Message], replacement: &[Message]) {
        self.cancellation_candidate = replacement.len() < before.len()
            && before[replacement.len()..].iter().any(|message| {
                message.role == Role::User && message.text().contains(crate::PLAN_DRIVE_PROMPT)
            });
        self.inferred_interruption_chain = false;
        self.pending_real_user_turn = None;
    }

    pub(super) fn clear_boundary(&mut self) {
        self.cancellation_candidate = false;
        self.inferred_interruption_chain = false;
    }

    pub(super) fn invalidate(&mut self) {
        self.clear_boundary();
        self.inferred_pause_active = false;
        self.pending_real_user_turn = None;
    }

    pub(super) fn resolve(&mut self, paused: bool, explicit: Option<bool>) -> bool {
        let inferred = explicit.is_none()
            && paused
            && (self.cancellation_candidate || self.inferred_interruption_chain);
        let resume_on_user_input = explicit.unwrap_or(inferred);
        self.cancellation_candidate = false;
        self.inferred_interruption_chain = paused && resume_on_user_input;
        self.inferred_pause_active = inferred;
        self.pending_real_user_turn = None;
        resume_on_user_input
    }

    pub(super) fn note_message(&mut self, message: &Message) {
        self.clear_boundary();
        if self.inferred_pause_active
            && self.pending_real_user_turn.is_none()
            && message.role == Role::User
        {
            let text = message.text();
            let synthetic = text.contains(crate::PLAN_DRIVE_PROMPT)
                || text.contains(crate::GOAL_CONTINUE_PROMPT);
            self.pending_real_user_turn = Some(!synthetic);
        }
    }

    pub(super) fn completed_user_turn_consumes_pause(
        &mut self,
        status: TurnStatus,
        stop_reason: TurnStopReason,
    ) -> bool {
        let successful = status == TurnStatus::Completed
            && !matches!(
                stop_reason,
                TurnStopReason::Cancelled
                    | TurnStopReason::TurnLimit
                    | TurnStopReason::InfrastructureFailure
                    | TurnStopReason::NoProgress
            );
        let consume =
            self.inferred_pause_active && self.pending_real_user_turn == Some(true) && successful;
        self.pending_real_user_turn = None;
        if consume {
            self.inferred_pause_active = false;
            self.inferred_interruption_chain = false;
            self.cancellation_candidate = false;
        }
        consume
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum LegacySessionMeta {
    RemoteSessionIdentity {
        session_id: String,
    },
    PipeFsMode {
        enabled: bool,
    },
    Name {
        name: String,
    },
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        #[serde(default)]
        cache_read_tokens: u64,
        #[serde(default)]
        cache_creation_tokens: u64,
        #[serde(default)]
        estimated: bool,
    },
    WorkspaceExecutionStaged {
        #[serde(default)]
        visible_on_resume: bool,
        execution: crate::WorkspaceTranscriptExecution,
    },
    WorkspaceExecutionSettled {
        operation_id: hi_workspace::OperationId,
    },
    Checkpoints {
        refs: Vec<String>,
    },
    Compaction {
        messages: Vec<Message>,
    },
    Goal {
        goal: Goal,
    },
    GoalCleared,
    TaskRecovery {
        state: crate::TaskRecoveryState,
    },
    Decisions {
        decisions: Vec<Decision>,
    },
    Plan {
        steps: Vec<PlanStep>,
    },
    PlanCleared,
    PlanDrive {
        #[serde(default)]
        paused: bool,
        #[serde(default)]
        resume_on_user_input: Option<bool>,
        #[serde(default)]
        stall: u32,
        #[serde(default)]
        evidence_reset: bool,
        #[serde(default)]
        evidence_add: Vec<String>,
    },
    PlanApproval {
        #[serde(default)]
        parked: bool,
    },
    GoalDrive {
        #[serde(default)]
        stall: u32,
        #[serde(default)]
        evidence_reset: bool,
        #[serde(default)]
        evidence_add: Vec<String>,
    },
    TurnOutcome {
        status: TurnStatus,
        stop_reason: TurnStopReason,
        #[serde(default)]
        task_recovery: Option<crate::TaskRecoveryState>,
        #[serde(default)]
        settled_goal: Option<Box<Goal>>,
    },
    StateReplacement {
        messages: Vec<Message>,
        #[serde(default)]
        goal: Option<Goal>,
        #[serde(default)]
        decisions: Vec<Decision>,
        #[serde(default)]
        plan: Vec<PlanStep>,
    },
    TranscriptBlockOpened {
        #[serde(alias = "id")]
        block_id: TranscriptBlockId,
        kind: TranscriptBlockKind,
        #[serde(default, alias = "text")]
        content: String,
    },
    TranscriptBlockAppended {
        #[serde(alias = "id")]
        block_id: TranscriptBlockId,
        #[serde(alias = "text")]
        delta: String,
    },
    TranscriptBlockReplaced {
        #[serde(alias = "id")]
        block_id: TranscriptBlockId,
        #[serde(alias = "text")]
        content: String,
    },
    TranscriptBlockSettled {
        #[serde(alias = "id")]
        block_id: TranscriptBlockId,
        terminal: TranscriptBlockTerminal,
    },
    TranscriptBlockRecorded {
        #[serde(alias = "id")]
        block_id: TranscriptBlockId,
        kind: TranscriptBlockKind,
        #[serde(default, alias = "text")]
        content: String,
        terminal: TranscriptBlockTerminal,
    },
}

impl From<LegacySessionMeta> for SessionEventKind {
    fn from(meta: LegacySessionMeta) -> Self {
        match meta {
            LegacySessionMeta::RemoteSessionIdentity { session_id } => {
                Self::RemoteSessionIdentity { session_id }
            }
            LegacySessionMeta::PipeFsMode { enabled } => Self::PipeFsMode { enabled },
            LegacySessionMeta::Name { name } => Self::Name { name },
            LegacySessionMeta::Usage {
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                estimated,
            } => Self::Usage {
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                estimated,
            },
            LegacySessionMeta::WorkspaceExecutionStaged {
                execution,
                visible_on_resume,
            } => Self::WorkspaceExecutionStaged {
                execution,
                visible_on_resume,
            },
            LegacySessionMeta::WorkspaceExecutionSettled { operation_id } => {
                Self::WorkspaceExecutionSettled { operation_id }
            }
            LegacySessionMeta::Checkpoints { refs } => Self::Checkpoints { refs },
            LegacySessionMeta::Compaction { messages } => Self::Compaction { messages },
            LegacySessionMeta::Goal { goal } => Self::Goal { goal },
            LegacySessionMeta::GoalCleared => Self::GoalCleared,
            LegacySessionMeta::TaskRecovery { state } => Self::TaskRecovery { state },
            LegacySessionMeta::Decisions { decisions } => Self::Decisions { decisions },
            LegacySessionMeta::Plan { steps } => Self::Plan { steps },
            LegacySessionMeta::PlanCleared => Self::PlanCleared,
            LegacySessionMeta::PlanDrive {
                paused,
                resume_on_user_input,
                stall,
                evidence_reset,
                evidence_add,
            } => Self::PlanDrive {
                paused,
                resume_on_user_input,
                stall,
                evidence_reset,
                evidence_add,
            },
            LegacySessionMeta::PlanApproval { parked } => Self::PlanApproval { parked },
            LegacySessionMeta::GoalDrive {
                stall,
                evidence_reset,
                evidence_add,
            } => Self::GoalDrive {
                stall,
                evidence_reset,
                evidence_add,
            },
            LegacySessionMeta::TurnOutcome {
                status,
                stop_reason,
                task_recovery,
                settled_goal,
            } => Self::TurnOutcome {
                status,
                stop_reason,
                task_recovery,
                settled_goal,
            },
            LegacySessionMeta::StateReplacement {
                messages,
                goal,
                decisions,
                plan,
            } => Self::StateReplacement {
                messages,
                goal,
                decisions,
                plan,
            },
            LegacySessionMeta::TranscriptBlockOpened {
                block_id,
                kind,
                content,
            } => Self::TranscriptBlockOpened {
                block_id,
                kind,
                content,
            },
            LegacySessionMeta::TranscriptBlockAppended { block_id, delta } => {
                Self::TranscriptBlockAppended { block_id, delta }
            }
            LegacySessionMeta::TranscriptBlockReplaced { block_id, content } => {
                Self::TranscriptBlockReplaced { block_id, content }
            }
            LegacySessionMeta::TranscriptBlockSettled { block_id, terminal } => {
                Self::TranscriptBlockSettled { block_id, terminal }
            }
            LegacySessionMeta::TranscriptBlockRecorded {
                block_id,
                kind,
                content,
                terminal,
            } => Self::TranscriptBlockRecorded {
                block_id,
                kind,
                content,
                terminal,
            },
        }
    }
}
