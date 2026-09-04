//! Pure, versioned reduction of durable session events.
//!
//! The reducer deliberately performs no I/O. Local JSONL, remote records, and
//! SQLite snapshot/tail restoration can all decode into [`SessionEvent`] and
//! use the same state transition rules. Legacy JSONL remains a wire projection;
//! the helpers here only read it and do not change its format.

use std::collections::BTreeSet;
use std::fmt;

use hi_ai::{Message, Usage};
use serde::{Deserialize, Serialize};

use crate::session_reducer_compat::{LegacyPlanPauseMigration, LegacySessionMeta};
use crate::session_transcript::{
    TranscriptBlockMutation, apply_transcript_mutation, ensure_all_transcript_blocks_settled,
    validate_transcript_snapshot,
};
use crate::{Decision, Goal, PlanStatus, PlanStep, TurnStatus, TurnStopReason};
use crate::{
    TranscriptBlock, TranscriptBlockId, TranscriptBlockKind, TranscriptBlockTerminal,
    TranscriptBlockTransitionError,
};

/// Current event envelope understood by [`SessionReducer`].
pub const SESSION_EVENT_SCHEMA_VERSION: u16 = 1;
/// Version of the deterministic state transition rules.
pub const SESSION_REDUCER_VERSION: u32 = 2;

/// One ordered input to the session reducer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionEvent {
    pub schema_version: u16,
    /// `None` is used by legacy JSONL, whose order is supplied by the file.
    /// Snapshotted/remote streams should provide contiguous one-based values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    pub kind: SessionEventKind,
}

impl SessionEvent {
    pub fn new(kind: SessionEventKind) -> Self {
        Self {
            schema_version: SESSION_EVENT_SCHEMA_VERSION,
            sequence: None,
            kind,
        }
    }

    pub fn at_sequence(mut self, sequence: u64) -> Self {
        self.sequence = Some(sequence);
        self
    }

    /// Decode one existing session JSONL record without changing its wire
    /// representation. Empty records are ignored; malformed records become an
    /// opaque boundary so legacy migration state cannot bridge corruption.
    pub fn from_legacy_json(line: &str) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        if let Ok(meta) = serde_json::from_str::<LegacySessionMeta>(line) {
            return Some(Self::new(meta.into()));
        }
        Some(Self::new(
            serde_json::from_str::<Message>(line)
                .map(|message| SessionEventKind::Message { message })
                .unwrap_or(SessionEventKind::OpaqueBoundary),
        ))
    }

    /// Decode an existing remote record. Remote `message` payloads are bare
    /// messages; all other record types retain the tagged JSONL metadata body.
    pub fn from_remote_record(record_type: &str, payload_json: &str) -> Self {
        let kind = if record_type == "message" {
            serde_json::from_str::<Message>(payload_json)
                .map(|message| SessionEventKind::Message { message })
                .unwrap_or(SessionEventKind::OpaqueBoundary)
        } else {
            serde_json::from_str::<LegacySessionMeta>(payload_json)
                .map(Into::into)
                .unwrap_or(SessionEventKind::OpaqueBoundary)
        };
        Self::new(kind)
    }

    pub fn opaque_boundary() -> Self {
        Self::new(SessionEventKind::OpaqueBoundary)
    }
}

/// Durable logical state shared by live, local-replay, and remote-replay paths.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionState {
    pub reducer_version: u32,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub checkpoint_refs: Vec<String>,
    #[serde(default)]
    pub remote_session_id: Option<String>,
    #[serde(default)]
    pub pipefs_enabled: Option<bool>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub goal: Option<Goal>,
    #[serde(default)]
    pub decisions: Vec<Decision>,
    #[serde(default)]
    pub plan: Vec<PlanStep>,
    #[serde(default)]
    pub plan_drive_paused: bool,
    #[serde(default)]
    pub plan_drive_resume_on_user_input: bool,
    #[serde(default)]
    pub plan_approval_parked: bool,
    #[serde(default)]
    pub plan_drive_stall: u32,
    #[serde(default)]
    pub goal_drive_stall: u32,
    #[serde(default)]
    pub plan_drive_evidence: BTreeSet<String>,
    #[serde(default)]
    pub goal_drive_evidence: BTreeSet<String>,
    /// Stable presentation identities. Empty for legacy message-only sessions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transcript_blocks: Vec<TranscriptBlock>,
}

impl Default for SessionState {
    fn default() -> Self {
        Self {
            reducer_version: SESSION_REDUCER_VERSION,
            messages: Vec::new(),
            usage: Usage::default(),
            checkpoint_refs: Vec::new(),
            remote_session_id: None,
            pipefs_enabled: None,
            name: None,
            goal: None,
            decisions: Vec::new(),
            plan: Vec::new(),
            plan_drive_paused: false,
            plan_drive_resume_on_user_input: false,
            plan_approval_parked: false,
            plan_drive_stall: 0,
            goal_drive_stall: 0,
            plan_drive_evidence: BTreeSet::new(),
            goal_drive_evidence: BTreeSet::new(),
            transcript_blocks: Vec::new(),
        }
    }
}

impl SessionState {
    /// Structural comparison that includes provider message payloads, whose
    /// public type intentionally does not implement `PartialEq`.
    pub fn semantically_eq(&self, other: &Self) -> bool {
        serde_json::to_value(self).ok() == serde_json::to_value(other).ok()
    }
}

/// New canonical session events. This envelope is independent of the legacy
/// JSONL representation, which is decoded by [`SessionEvent`] helpers above.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEventKind {
    Message {
        message: Message,
    },
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
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
    /// Compatibility form for a block that was persisted only after it ended.
    TranscriptBlockRecorded {
        #[serde(alias = "id")]
        block_id: TranscriptBlockId,
        kind: TranscriptBlockKind,
        #[serde(default, alias = "text")]
        content: String,
        terminal: TranscriptBlockTerminal,
    },
    /// A malformed or unknown legacy record. It has no logical projection but
    /// deliberately breaks adjacency-sensitive compatibility inference.
    OpaqueBoundary,
}

/// Serializable reducer checkpoint for snapshot-plus-tail restoration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionReducerSnapshot {
    pub reducer_version: u32,
    pub through_sequence: u64,
    pub state: SessionState,
    compatibility: LegacyPlanPauseMigration,
}

/// Deterministic session projection. Applying an event either updates the
/// complete state atomically or returns an error without changing the reducer.
#[derive(Clone, Debug, Default)]
pub struct SessionReducer {
    state: SessionState,
    through_sequence: u64,
    compatibility: LegacyPlanPauseMigration,
}

impl SessionReducer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_snapshot(snapshot: SessionReducerSnapshot) -> Result<Self, SessionReduceError> {
        if snapshot.reducer_version != SESSION_REDUCER_VERSION
            || snapshot.state.reducer_version != SESSION_REDUCER_VERSION
        {
            return Err(SessionReduceError::UnsupportedReducerVersion {
                found: snapshot.reducer_version,
                supported: SESSION_REDUCER_VERSION,
            });
        }
        validate_transcript_snapshot(&snapshot.state.transcript_blocks, snapshot.through_sequence)?;
        Ok(Self {
            state: snapshot.state,
            through_sequence: snapshot.through_sequence,
            compatibility: snapshot.compatibility,
        })
    }

    pub fn state(&self) -> &SessionState {
        &self.state
    }

    pub fn through_sequence(&self) -> u64 {
        self.through_sequence
    }

    pub fn snapshot(&self) -> SessionReducerSnapshot {
        SessionReducerSnapshot {
            reducer_version: SESSION_REDUCER_VERSION,
            through_sequence: self.through_sequence,
            state: self.state.clone(),
            compatibility: self.compatibility.clone(),
        }
    }

    /// Apply exactly one event and return the complete new projection.
    pub fn apply(&mut self, event: SessionEvent) -> Result<SessionState, SessionReduceError> {
        if event.schema_version != SESSION_EVENT_SCHEMA_VERSION {
            return Err(SessionReduceError::UnsupportedEventVersion {
                found: event.schema_version,
                supported: SESSION_EVENT_SCHEMA_VERSION,
            });
        }
        let expected = self.through_sequence.saturating_add(1);
        let sequence = event.sequence.unwrap_or(expected);
        if sequence != expected {
            return Err(SessionReduceError::NonContiguousSequence {
                expected,
                found: sequence,
            });
        }
        if let SessionEventKind::RemoteSessionIdentity { session_id } = &event.kind {
            validate_session_id(session_id)?;
        }
        if matches!(&event.kind, SessionEventKind::TurnOutcome { .. }) {
            ensure_all_transcript_blocks_settled(&self.state.transcript_blocks)?;
        }
        let transcript_applied = self.apply_transcript_event(&event.kind, sequence)?;
        if transcript_applied {
            self.compatibility.clear_boundary();
        } else {
            self.apply_kind(event.kind);
        }
        self.through_sequence = sequence;
        self.state.reducer_version = SESSION_REDUCER_VERSION;
        if self
            .state
            .plan
            .iter()
            .all(|step| step.status == PlanStatus::Done)
        {
            self.state.plan.clear();
        }
        Ok(self.state.clone())
    }

    pub fn apply_all(
        &mut self,
        events: impl IntoIterator<Item = SessionEvent>,
    ) -> Result<SessionState, SessionReduceError> {
        for event in events {
            self.apply(event)?;
        }
        Ok(self.state.clone())
    }

    fn apply_kind(&mut self, kind: SessionEventKind) {
        match kind {
            SessionEventKind::Message { message } => {
                self.compatibility.note_message(&message);
                self.state.messages.push(message);
            }
            SessionEventKind::RemoteSessionIdentity { session_id } => {
                self.state.remote_session_id = Some(session_id);
            }
            SessionEventKind::PipeFsMode { enabled } => {
                self.state.pipefs_enabled = Some(enabled);
            }
            SessionEventKind::Name { name } => {
                self.compatibility.clear_boundary();
                self.state.name = (!name.trim().is_empty()).then(|| name.trim().to_owned());
            }
            SessionEventKind::Usage {
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                estimated,
            } => {
                self.state.usage = Usage {
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                    input_includes_cache: false,
                    context_occupancy: input_tokens,
                    rate_limits: None,
                    estimated,
                };
            }
            SessionEventKind::Checkpoints { refs } => self.state.checkpoint_refs = refs,
            SessionEventKind::Compaction { messages } => {
                self.compatibility.clear_boundary();
                self.state.messages = messages;
            }
            SessionEventKind::Goal { goal } => {
                self.compatibility.clear_boundary();
                self.state.goal = Some(goal);
            }
            SessionEventKind::GoalCleared => {
                self.compatibility.clear_boundary();
                self.state.goal = None;
                self.state.goal_drive_evidence.clear();
            }
            SessionEventKind::Decisions { decisions } => {
                self.compatibility.clear_boundary();
                self.state.decisions = normalize_decisions(decisions);
            }
            SessionEventKind::Plan { steps } => {
                self.compatibility.clear_boundary();
                self.state.plan = steps;
            }
            SessionEventKind::PlanCleared => {
                self.compatibility.clear_boundary();
                self.state.plan.clear();
                self.state.plan_drive_evidence.clear();
            }
            SessionEventKind::PlanDrive {
                paused,
                resume_on_user_input,
                stall,
                evidence_reset,
                evidence_add,
            } => {
                self.state.plan_drive_paused = paused;
                self.state.plan_drive_resume_on_user_input =
                    self.compatibility.resolve(paused, resume_on_user_input);
                self.state.plan_drive_stall = stall;
                apply_evidence_delta(
                    &mut self.state.plan_drive_evidence,
                    evidence_reset,
                    evidence_add,
                );
            }
            SessionEventKind::PlanApproval { parked } => {
                self.compatibility.clear_boundary();
                self.state.plan_approval_parked = parked;
            }
            SessionEventKind::GoalDrive {
                stall,
                evidence_reset,
                evidence_add,
            } => {
                self.compatibility.clear_boundary();
                self.state.goal_drive_stall = stall;
                apply_evidence_delta(
                    &mut self.state.goal_drive_evidence,
                    evidence_reset,
                    evidence_add,
                );
            }
            SessionEventKind::TurnOutcome {
                status,
                stop_reason,
            } => {
                if status != TurnStatus::Cancelled
                    && !matches!(
                        stop_reason,
                        TurnStopReason::Cancelled | TurnStopReason::TurnLimit
                    )
                {
                    self.compatibility.clear_boundary();
                }
                if self
                    .compatibility
                    .completed_user_turn_consumes_pause(status, stop_reason)
                {
                    self.state.plan_drive_paused = false;
                    self.state.plan_drive_resume_on_user_input = false;
                    self.state.plan_drive_stall = 0;
                    self.state.plan_drive_evidence.clear();
                }
            }
            SessionEventKind::StateReplacement {
                messages,
                goal,
                decisions,
                plan,
            } => {
                self.compatibility
                    .note_state_replacement(&self.state.messages, &messages);
                self.state.messages = messages;
                self.state.goal = goal;
                self.state.decisions = normalize_decisions(decisions);
                self.state.plan = plan;
            }
            SessionEventKind::TranscriptBlockOpened { .. }
            | SessionEventKind::TranscriptBlockAppended { .. }
            | SessionEventKind::TranscriptBlockReplaced { .. }
            | SessionEventKind::TranscriptBlockSettled { .. }
            | SessionEventKind::TranscriptBlockRecorded { .. } => {
                unreachable!("transcript events are reduced before compatibility events")
            }
            SessionEventKind::OpaqueBoundary => self.compatibility.invalidate(),
        }
    }

    fn apply_transcript_event(
        &mut self,
        kind: &SessionEventKind,
        sequence: u64,
    ) -> Result<bool, SessionReduceError> {
        let mutation = match kind {
            SessionEventKind::TranscriptBlockOpened {
                block_id,
                kind,
                content,
            } => TranscriptBlockMutation::Open {
                id: block_id,
                kind: *kind,
                content,
            },
            SessionEventKind::TranscriptBlockAppended { block_id, delta } => {
                TranscriptBlockMutation::Append {
                    id: block_id,
                    delta,
                }
            }
            SessionEventKind::TranscriptBlockReplaced { block_id, content } => {
                TranscriptBlockMutation::Replace {
                    id: block_id,
                    content,
                }
            }
            SessionEventKind::TranscriptBlockSettled { block_id, terminal } => {
                TranscriptBlockMutation::Settle {
                    id: block_id,
                    terminal: *terminal,
                }
            }
            SessionEventKind::TranscriptBlockRecorded {
                block_id,
                kind,
                content,
                terminal,
            } => TranscriptBlockMutation::Record {
                id: block_id,
                kind: *kind,
                content,
                terminal: *terminal,
            },
            _ => return Ok(false),
        };
        apply_transcript_mutation(&mut self.state.transcript_blocks, mutation, sequence)?;
        Ok(true)
    }
}

/// Reducer failures are compatibility or ordering faults, never partial state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionReduceError {
    UnsupportedEventVersion { found: u16, supported: u16 },
    UnsupportedReducerVersion { found: u32, supported: u32 },
    NonContiguousSequence { expected: u64, found: u64 },
    InvalidRemoteSessionIdentity,
    TranscriptBlock(TranscriptBlockTransitionError),
}

impl fmt::Display for SessionReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedEventVersion { found, supported } => write!(
                f,
                "unsupported session event schema {found}; this binary supports {supported}"
            ),
            Self::UnsupportedReducerVersion { found, supported } => write!(
                f,
                "unsupported session reducer snapshot {found}; this binary supports {supported}"
            ),
            Self::NonContiguousSequence { expected, found } => write!(
                f,
                "non-contiguous session event sequence: expected {expected}, found {found}"
            ),
            Self::InvalidRemoteSessionIdentity => write!(f, "invalid remote session identity"),
            Self::TranscriptBlock(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SessionReduceError {}

impl From<TranscriptBlockTransitionError> for SessionReduceError {
    fn from(value: TranscriptBlockTransitionError) -> Self {
        Self::TranscriptBlock(value)
    }
}

fn validate_session_id(session_id: &str) -> Result<(), SessionReduceError> {
    if session_id.is_empty()
        || session_id.len() > 128
        || !session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(SessionReduceError::InvalidRemoteSessionIdentity);
    }
    Ok(())
}

fn normalize_decisions(decisions: Vec<Decision>) -> Vec<Decision> {
    let mut normalized: Vec<Decision> = Vec::new();
    for decision in decisions {
        if let Some(existing) = normalized
            .iter_mut()
            .find(|existing| existing.summary == decision.summary)
        {
            *existing = decision;
        } else {
            normalized.push(decision);
        }
    }
    normalized
}

fn apply_evidence_delta(evidence: &mut BTreeSet<String>, reset: bool, added: Vec<String>) {
    if reset {
        evidence.clear();
    }
    evidence.extend(added.into_iter().filter(|hash| {
        hash.len() == 64
            && hash
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }));
}

#[cfg(test)]
#[path = "session_reducer_tests.rs"]
mod tests;
