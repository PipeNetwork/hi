use serde::{Deserialize, Serialize};

pub const WORKFLOW_HISTORY_MAX: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowRunStatus {
    Active,
    UserPaused,
    BackOffPaused,
    NoProgressPaused,
    InfraPaused,
    Blocked,
    BudgetLimited,
    Interrupted,
    Complete,
    Failed,
    Cancelled,
}

impl WorkflowRunStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Interrupted | Self::Complete | Self::Failed | Self::Cancelled
        )
    }

    pub fn is_paused(self) -> bool {
        matches!(
            self,
            Self::UserPaused
                | Self::BackOffPaused
                | Self::NoProgressPaused
                | Self::InfraPaused
                | Self::Blocked
                | Self::BudgetLimited
        )
    }

    pub fn is_resumable(self) -> bool {
        self.is_paused() || matches!(self, Self::Interrupted | Self::Failed)
    }

    pub fn is_completion_reportable(self) -> bool {
        self.is_terminal() || self == Self::BudgetLimited
    }
}

impl From<crate::PauseKind> for WorkflowRunStatus {
    fn from(kind: crate::PauseKind) -> Self {
        match kind {
            crate::PauseKind::User => Self::UserPaused,
            crate::PauseKind::BackOff => Self::BackOffPaused,
            crate::PauseKind::NoProgress => Self::NoProgressPaused,
            crate::PauseKind::Verification => Self::Blocked,
            crate::PauseKind::Infra => Self::InfraPaused,
            crate::PauseKind::Approval => Self::Blocked,
        }
    }
}

impl From<&crate::WorkflowOutcome> for WorkflowRunStatus {
    fn from(outcome: &crate::WorkflowOutcome) -> Self {
        match outcome {
            crate::WorkflowOutcome::Completed { .. } => Self::Complete,
            crate::WorkflowOutcome::Paused { kind, .. } => (*kind).into(),
            crate::WorkflowOutcome::BudgetExceeded { .. } => Self::BudgetLimited,
            crate::WorkflowOutcome::Cancelled => Self::Cancelled,
            crate::WorkflowOutcome::Failed { .. } => Self::Failed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowHistoryEntry {
    pub event: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowPhaseSnapshot {
    pub title: String,
    pub state: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkflowAgentSnapshot {
    pub agent_id: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub state: String,
    pub tokens_used: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowRunSnapshot {
    pub run_id: String,
    pub revision: u64,
    pub workflow_name: String,
    #[serde(default)]
    pub objective: String,
    pub status: WorkflowRunStatus,
    #[serde(default)]
    pub phases: Vec<WorkflowPhaseSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_phase: Option<String>,
    #[serde(default)]
    pub agents: Vec<WorkflowAgentSnapshot>,
    pub agent_budget: u64,
    pub agents_used: u64,
    pub agents_reserved: u64,
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pause_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_summary: Option<String>,
    #[serde(default)]
    pub history: Vec<WorkflowHistoryEntry>,
}

impl WorkflowRunSnapshot {
    pub fn record_event(&mut self, event: impl Into<String>, detail: Option<String>, at_ms: u64) {
        self.revision = self.revision.saturating_add(1);
        self.history.push(WorkflowHistoryEntry {
            event: event.into(),
            detail,
            at_ms,
        });
        if self.history.len() > WORKFLOW_HISTORY_MAX {
            self.history
                .drain(..self.history.len() - WORKFLOW_HISTORY_MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_predicates_preserve_actionable_pause_reasons() {
        assert!(WorkflowRunStatus::UserPaused.is_paused());
        assert!(WorkflowRunStatus::BudgetLimited.is_resumable());
        assert!(WorkflowRunStatus::BudgetLimited.is_completion_reportable());
        assert!(!WorkflowRunStatus::BudgetLimited.is_terminal());
        assert!(WorkflowRunStatus::Complete.is_terminal());
    }

    #[test]
    fn history_is_bounded_and_revision_is_monotonic() {
        let mut snapshot = WorkflowRunSnapshot {
            run_id: "run-1".into(),
            revision: 0,
            workflow_name: "test".into(),
            objective: String::new(),
            status: WorkflowRunStatus::Active,
            phases: Vec::new(),
            current_phase: None,
            agents: Vec::new(),
            agent_budget: 8,
            agents_used: 0,
            agents_reserved: 0,
            elapsed_ms: 0,
            pause_message: None,
            result_summary: None,
            history: Vec::new(),
        };
        for index in 0..WORKFLOW_HISTORY_MAX + 3 {
            snapshot.record_event(format!("event-{index}"), None, index as u64);
        }
        assert_eq!(snapshot.revision, (WORKFLOW_HISTORY_MAX + 3) as u64);
        assert_eq!(snapshot.history.len(), WORKFLOW_HISTORY_MAX);
        assert_eq!(snapshot.history[0].event, "event-3");
    }
}
