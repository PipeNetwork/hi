//! Lifecycle translation shared by the compatibility background-task registry.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio::task::AbortHandle;

use crate::background_tasks::{BackgroundTaskOutcome, BackgroundTaskState};

pub(crate) fn contract(
    subagent_type: &str,
) -> (crate::BackgroundJobKind, crate::BackgroundJobEffect) {
    match subagent_type.trim().to_ascii_lowercase().as_str() {
        "general-purpose" | "general_purpose" | "generalpurpose" | "delegate" | "code" => (
            crate::BackgroundJobKind::WriteCandidate,
            crate::BackgroundJobEffect::CandidateOnly,
        ),
        _ => (
            crate::BackgroundJobKind::ReadAgent,
            crate::BackgroundJobEffect::ReadOnly,
        ),
    }
}

pub(crate) fn terminal(
    outcome: &BackgroundTaskOutcome,
) -> (crate::BackgroundJobTerminal, Option<String>) {
    match outcome.state {
        BackgroundTaskState::Completed => (crate::BackgroundJobTerminal::Succeeded, None),
        BackgroundTaskState::Cancelled => (
            crate::BackgroundJobTerminal::Cancelled,
            (!outcome.output.is_empty()).then(|| outcome.output.clone()),
        ),
        BackgroundTaskState::Failed => (
            crate::BackgroundJobTerminal::Failed,
            (!outcome.output.is_empty()).then(|| outcome.output.clone()),
        ),
        BackgroundTaskState::Running => (
            crate::BackgroundJobTerminal::Failed,
            Some("background worker returned a non-terminal outcome".into()),
        ),
    }
}

pub(crate) async fn observe_natural_exit(
    managed_job: Option<crate::job_lifecycle::ManagedBackgroundJob>,
    outcome: &mut BackgroundTaskOutcome,
    artifacts: Vec<hi_workspace::ArtifactRef>,
) -> Option<crate::BackgroundJobPublication> {
    if let Err(error) =
        hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::JobAfterNaturalExit)
    {
        outcome.state = BackgroundTaskState::Failed;
        outcome.output = error.to_string();
        outcome.applied = false;
        outcome.changed_files.clear();
    }
    let job = managed_job?;
    let (terminal, detail) = terminal(outcome);
    match job
        .observe_with_artifacts(terminal, detail, artifacts)
        .await
    {
        Ok(publication) => Some(publication),
        Err(error) => {
            outcome.state = BackgroundTaskState::Failed;
            outcome.output = format!("workspace job settlement failed: {error}");
            outcome.applied = false;
            outcome.changed_files.clear();
            None
        }
    }
}

pub(crate) struct CancelSettlement {
    pub abort_handle: AbortHandle,
    pub managed_job: Option<crate::job_lifecycle::ManagedBackgroundJob>,
    pub lifecycle_gate: Arc<tokio::sync::Mutex<()>>,
    pub outcome: BackgroundTaskOutcome,
    pub terminal_outcome: Arc<Mutex<Option<BackgroundTaskOutcome>>>,
    pub outcomes: Arc<Mutex<HashMap<String, BackgroundTaskOutcome>>>,
    pub notify: Arc<Notify>,
    pub completed_notify: Arc<Notify>,
    pub candidates: super::candidates::CandidateQueue,
    pub teardown: crate::BackgroundTaskTeardown,
}

impl CancelSettlement {
    /// Abort is a request, not terminal proof. Publish cancellation only after
    /// Tokio has dropped the worker future and the workspace callback settles.
    pub(crate) async fn run(mut self) -> BackgroundTaskOutcome {
        let _settlement = self.lifecycle_gate.lock().await;
        if let Some(mut existing) = self
            .terminal_outcome
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            if let Err(error) = self.candidates.discard_after_terminal(&self.outcome.id) {
                existing
                    .output
                    .push_str(&format!("\nCandidate artifact cleanup failed: {error:#}"));
            }
            return existing;
        }
        self.abort_handle.abort();
        let failpoint = hi_workspace::hit_harness_failpoint(
            hi_workspace::HarnessFailpoint::JobAfterCancelRequest,
        )
        .err();
        while !self.abort_handle.is_finished() {
            tokio::task::yield_now().await;
        }
        if let Err(error) = self.teardown.wait().await {
            self.outcome.state = BackgroundTaskState::Failed;
            self.outcome.output =
                format!("Task cancellation could not prove child process teardown: {error}");
        }
        if let Some(error) = failpoint {
            self.outcome.output.push_str(&format!("\n{error}"));
        }
        let terminal = if self.outcome.state == BackgroundTaskState::Cancelled {
            crate::BackgroundJobTerminal::Cancelled
        } else {
            crate::BackgroundJobTerminal::Failed
        };
        let acknowledged = if let Some(job) = &self.managed_job {
            match job
                .observe(terminal, Some(self.outcome.output.clone()))
                .await
            {
                Ok(crate::BackgroundJobPublication::Published) => true,
                Ok(crate::BackgroundJobPublication::DurabilityPending) => {
                    self.outcome.state = BackgroundTaskState::Failed;
                    self.outcome.output =
                        "workspace job cancellation remains durability-pending".into();
                    false
                }
                Err(error) => {
                    self.outcome.state = BackgroundTaskState::Failed;
                    self.outcome.output =
                        format!("workspace job cancellation settlement failed: {error}");
                    false
                }
            }
        } else {
            true
        };
        if acknowledged && let Err(error) = self.candidates.discard_after_terminal(&self.outcome.id)
        {
            self.outcome
                .output
                .push_str(&format!("\nCandidate artifact cleanup failed: {error:#}"));
        }
        *self
            .terminal_outcome
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(self.outcome.clone());
        self.outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(self.outcome.id.clone(), self.outcome.clone());
        self.notify.notify_waiters();
        self.completed_notify.notify_waiters();
        self.outcome
    }
}
