//! Stable, redaction-friendly lifecycle events shared by the coding harness.
//!
//! This crate intentionally contains contracts only. Persistence, TUI
//! projections, and workflow dispatchers live in their owning crates so the
//! agent loop does not depend on a frontend or on a particular database.

use std::fmt;

use serde::{Deserialize, Serialize};

pub const EVENT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventDurability {
    BestEffort,
    Required,
}

impl Default for EventDurability {
    fn default() -> Self {
        Self::BestEffort
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityVerb {
    Start,
    Request,
    Wait,
    Resume,
    Read,
    Write,
    Execute,
    Verify,
    Approve,
    Deny,
    Trigger,
    Complete,
    Fail,
    Cancel,
    Change,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityObject {
    Run,
    Tool,
    Capability,
    Approval,
    Workflow,
    Phase,
    Verification,
    Workspace,
    Git,
    Loop,
    Trigger,
    Race,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivityState {
    Pending,
    Running,
    Waiting,
    Succeeded,
    Failed,
    Denied,
    TimedOut,
    Cancelled,
    Abandoned,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivityRef {
    pub kind: String,
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticActivity {
    pub verb: ActivityVerb,
    pub object: ActivityObject,
    pub state: ActivityState,
    pub group_key: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default)]
    pub refs: Vec<ActivityRef>,
    #[serde(default)]
    pub progress: Option<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum EventKind {
    RunStarted,
    AttemptClaimed,
    AttemptRenewed,
    AttemptLeaseLost,
    AttemptCompleted,
    AttemptFailed,
    RunWaiting,
    RunResumed,
    RunCompleted,
    RunFailed,
    RunCancelled,
    ToolRequested,
    ToolStarted,
    ToolCompleted,
    ToolDenied,
    ToolTimedOut,
    CapabilityRequested,
    PolicyEvaluated,
    RouteSelected,
    EffectPlanned,
    EffectStarted,
    EffectCompleted,
    EffectFailed,
    EffectDenied,
    EffectUnknown,
    EffectReconciled,
    AuditRecorded,
    ApprovalDecided,
    ApprovalConsumed,
    WorkflowStarted,
    WorkflowPaused,
    WorkflowResumed,
    WorkflowCompleted,
    WorkflowFailed,
    PhaseStarted,
    PhaseCompleted,
    VerificationStarted,
    VerificationCompleted,
    GitChanged,
    LoopFired,
    TriggerAccepted,
    TriggerSkipped,
    TriggerStarted,
    TriggerCompleted,
    TriggerFailed,
    RaceStarted,
    RaceCandidateStarted,
    RaceCandidateCompleted,
    RaceCandidateScored,
    RaceWinnerReady,
    RaceApplied,
    RaceCancelled,
    RaceWorkspaceConflict,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EventPayload {
    /// Only identifiers and safe, bounded fields belong here. Full prompts,
    /// diffs, tool arguments, and outputs must remain in raw artifacts.
    #[serde(default)]
    pub fields: serde_json::Map<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_ref: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunEvent {
    pub schema_version: u16,
    pub event_id: String,
    /// Assigned by the durable store. Zero means not persisted yet.
    #[serde(default)]
    pub sequence: u64,
    pub occurred_at_ms: u64,
    #[serde(default)]
    pub context: EventContext,
    pub kind: EventKind,
    pub activity: SemanticActivity,
    #[serde(default)]
    pub payload: EventPayload,
    #[serde(default)]
    pub durability: EventDurability,
}

impl RunEvent {
    pub fn new(kind: EventKind, context: EventContext, activity: SemanticActivity) -> Self {
        Self {
            schema_version: EVENT_SCHEMA_VERSION,
            event_id: uuid::Uuid::new_v4().to_string(),
            sequence: 0,
            occurred_at_ms: now_ms(),
            context,
            kind,
            activity,
            payload: EventPayload::default(),
            durability: EventDurability::BestEffort,
        }
    }

    pub fn required(mut self) -> Self {
        self.durability = EventDurability::Required;
        self
    }

    pub fn with_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.payload.fields.insert(key.into(), value);
        self
    }

    pub fn with_raw_ref(mut self, raw_ref: impl Into<String>) -> Self {
        self.payload.raw_ref = Some(raw_ref.into());
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventReceipt {
    pub event_id: String,
    pub sequence: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventError {
    Invalid(String),
    Persistence(String),
}

impl fmt::Display for EventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid event: {message}"),
            Self::Persistence(message) => write!(f, "event persistence failed: {message}"),
        }
    }
}

impl std::error::Error for EventError {}

pub trait EventSink: Send + Sync {
    fn publish(&self, event: RunEvent) -> Result<EventReceipt, EventError>;
}

/// A durable event sink that can replay canonical events in stream order.
/// Live subscription remains an implementation concern (the CLI uses a
/// broadcast channel), while this contract is sufficient for restart-safe
/// workflow dispatchers and remote observation adapters.
pub trait EventBus: EventSink {
    fn replay_since(&self, sequence: u64) -> Result<Vec<RunEvent>, EventError>;
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activity() -> SemanticActivity {
        SemanticActivity {
            verb: ActivityVerb::Start,
            object: ActivityObject::Run,
            state: ActivityState::Running,
            group_key: "run:r1".into(),
            title: "Run started".into(),
            detail: None,
            refs: vec![],
            progress: None,
        }
    }

    #[test]
    fn serializes_tagged_event_kind_and_redacted_payload() {
        let event = RunEvent::new(EventKind::RunStarted, EventContext::default(), activity())
            .with_field("status", serde_json::json!("running"));
        let value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["schema_version"], EVENT_SCHEMA_VERSION);
        assert_eq!(value["kind"]["type"], "run_started");
        assert_eq!(value["payload"]["fields"]["status"], "running");
        assert!(!value.to_string().contains("prompt"));
    }
}
