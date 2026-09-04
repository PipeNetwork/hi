use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::{Result, bail};
use tokio::sync::Notify;

use crate::candidate_workspace::PersistedDetachedCandidate;
use crate::{
    BackgroundCandidateTransition, BackgroundJobPublication, BackgroundTaskOutcome,
    BackgroundTaskRegistry, BackgroundTaskState,
};

#[derive(Clone)]
pub(super) struct CandidateQueue {
    inner: Arc<Mutex<HashMap<String, CandidateEntry>>>,
    activity: Arc<Notify>,
    active_publications: Arc<AtomicUsize>,
}

/// Keeps abnormal-turn cleanup from restoring workspace bytes while a
/// parent-owned candidate apply/verifier/rollback worker can still write.
pub(super) struct CandidatePublicationLease {
    active: Arc<AtomicUsize>,
    activity: Arc<Notify>,
}

impl Drop for CandidatePublicationLease {
    fn drop(&mut self) {
        let previous = self.active.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "candidate publication lease underflow");
        self.activity.notify_waiters();
    }
}

struct CandidateEntry {
    candidate: Option<PersistedDetachedCandidate>,
    phase: CandidatePhase,
    resolution: Option<CandidateResolution>,
    notify: Arc<Notify>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidatePhase {
    Preparing,
    Ready,
    MergeClaimed,
    CancelRequested,
    Resolved,
}

#[derive(Clone, Debug)]
pub(super) enum CandidateResolution {
    Applied(Vec<String>),
    Rejected {
        detail: String,
        retain_artifact: bool,
    },
}

impl CandidateQueue {
    pub(super) fn new(activity: Arc<Notify>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            activity,
            active_publications: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(super) fn track_publication(&self) -> CandidatePublicationLease {
        self.active_publications.fetch_add(1, Ordering::AcqRel);
        CandidatePublicationLease {
            active: Arc::clone(&self.active_publications),
            activity: Arc::clone(&self.activity),
        }
    }

    pub(super) async fn wait_for_publications(&self, timeout: std::time::Duration) -> bool {
        let settled = async {
            loop {
                let notified = self.activity.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.active_publications.load(Ordering::Acquire) == 0 {
                    return;
                }
                notified.await;
            }
        };
        tokio::time::timeout(timeout, settled).await.is_ok()
    }

    pub(super) fn publish(&self, id: &str, candidate: PersistedDetachedCandidate) -> Result<()> {
        let mut entries = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(entry) = entries.get_mut(id) {
            if entry.phase == CandidatePhase::CancelRequested && entry.candidate.is_none() {
                entry.candidate = Some(candidate);
                bail!("background task {id} was cancelled before candidate publication");
            }
            bail!("background task {id} published more than one candidate");
        }
        entries.insert(
            id.to_owned(),
            CandidateEntry {
                candidate: Some(candidate),
                phase: CandidatePhase::Preparing,
                resolution: None,
                notify: Arc::new(Notify::new()),
            },
        );
        Ok(())
    }

    pub(super) fn finish_preparation(&self, id: &str, ready: bool) -> bool {
        let mut entries = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if ready {
            if let Some(entry) = entries.get_mut(id) {
                if entry.phase != CandidatePhase::Preparing {
                    return false;
                }
                entry.phase = CandidatePhase::Ready;
                entry.notify.notify_waiters();
                self.activity.notify_waiters();
                return true;
            }
        } else if let Some(entry) = entries.remove(id) {
            entry.notify.notify_waiters();
        }
        false
    }

    pub(super) fn claim_ready(&self) -> Vec<(String, PersistedDetachedCandidate)> {
        let mut entries = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut ids = entries
            .iter()
            .filter(|(_, entry)| entry.phase == CandidatePhase::Ready && entry.candidate.is_some())
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        ids.sort();
        ids.into_iter()
            .filter_map(|id| {
                let entry = entries.get_mut(&id)?;
                let candidate = entry.candidate.take()?;
                entry.phase = CandidatePhase::MergeClaimed;
                Some((id, candidate))
            })
            .collect()
    }

    pub(super) fn is_ready(&self, id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .is_some_and(|entry| entry.phase == CandidatePhase::Ready && entry.candidate.is_some())
    }

    pub(super) fn restore(&self, id: &str, candidate: PersistedDetachedCandidate) {
        if let Some(entry) = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_mut(id)
            && entry.phase == CandidatePhase::MergeClaimed
        {
            entry.candidate = Some(candidate);
            entry.phase = CandidatePhase::Ready;
            self.activity.notify_waiters();
        }
    }

    pub(super) fn resolve(&self, id: &str, resolution: CandidateResolution) {
        if let Some(entry) = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_mut(id)
        {
            entry.resolution = Some(resolution);
            entry.phase = CandidatePhase::Resolved;
            entry.notify.notify_waiters();
        }
    }

    /// Reserve cancellation against candidate merge ownership. Once a parent
    /// has claimed a candidate, cancellation must not publish `Cancelled`
    /// while those bytes can still be applied.
    pub(super) fn reserve_cancel(&self, id: &str) -> bool {
        let mut entries = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match entries.get_mut(id) {
            Some(entry) if entry.phase == CandidatePhase::MergeClaimed => false,
            Some(entry) if entry.phase == CandidatePhase::Resolved => false,
            Some(entry) => {
                entry.phase = CandidatePhase::CancelRequested;
                true
            }
            None => {
                entries.insert(
                    id.to_owned(),
                    CandidateEntry {
                        candidate: None,
                        phase: CandidatePhase::CancelRequested,
                        resolution: None,
                        notify: Arc::new(Notify::new()),
                    },
                );
                true
            }
        }
    }

    fn cancel_requested(&self, id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .is_some_and(|entry| entry.phase == CandidatePhase::CancelRequested)
    }

    pub(super) async fn wait(&self, id: &str) -> Option<CandidateResolution> {
        loop {
            let notify = {
                let entries = self
                    .inner
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let entry = entries.get(id)?;
                entry.notify.clone()
            };
            let notified = notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let resolution = self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(id)
                .and_then(|entry| entry.resolution.clone());
            if resolution.is_some() {
                return resolution;
            }
            notified.await;
        }
    }

    pub(super) fn discard(&self, id: &str) {
        if let Some(entry) = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id)
        {
            entry.notify.notify_waiters();
        }
    }

    pub(super) fn discard_after_terminal(&self, id: &str) -> Result<()> {
        let candidate = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id)
            .and_then(|entry| {
                entry.notify.notify_waiters();
                entry.candidate
            });
        match candidate {
            Some(candidate) => candidate.remove_after_terminal(),
            None => Ok(()),
        }
    }

    pub(super) fn artifacts(&self, id: &str) -> Vec<hi_workspace::ArtifactRef> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .and_then(|entry| entry.candidate.as_ref())
            .map(|candidate| vec![candidate.artifact.clone()])
            .unwrap_or_default()
    }

    pub(super) fn clear(&self) {
        let entries = std::mem::take(
            &mut *self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        for entry in entries.into_values() {
            entry.notify.notify_waiters();
        }
    }

    pub(super) async fn settle_worker(
        &self,
        id: &str,
        outcome: &mut BackgroundTaskOutcome,
        publication: Option<BackgroundJobPublication>,
    ) {
        let waits_for_apply = outcome.state == BackgroundTaskState::Completed
            && publication == Some(BackgroundJobPublication::DurabilityPending)
            && self.finish_preparation(id, true);
        if !waits_for_apply {
            if self.cancel_requested(id) {
                std::future::pending::<()>().await;
            }
            self.finish_preparation(id, false);
            return;
        }
        let cleanup = match self.wait(id).await {
            Some(CandidateResolution::Applied(paths)) => {
                outcome.applied = true;
                outcome.changed_files = paths;
                outcome
                    .output
                    .push_str("\nCandidate applied and durably settled by the parent.");
                true
            }
            Some(CandidateResolution::Rejected {
                detail,
                retain_artifact,
            }) => {
                outcome.state = BackgroundTaskState::Failed;
                outcome.applied = false;
                outcome.changed_files.clear();
                outcome
                    .output
                    .push_str(&format!("\nCandidate was not applied: {detail}"));
                !retain_artifact
            }
            None => {
                outcome.state = BackgroundTaskState::Failed;
                outcome.applied = false;
                outcome.changed_files.clear();
                outcome.output = "background task registry shut down before candidate apply".into();
                false
            }
        };
        if !cleanup {
            self.discard(id);
            return;
        }
        if let Err(error) = self.discard_after_terminal(id) {
            outcome
                .output
                .push_str(&format!("\nCandidate artifact cleanup failed: {error:#}"));
        }
    }

    pub(super) fn all_actionable(
        &self,
        ids: &[String],
        outcomes: &[BackgroundTaskOutcome],
    ) -> bool {
        ids.iter()
            .zip(outcomes)
            .all(|(id, outcome)| outcome.state.is_terminal() || self.is_ready(id))
    }

    pub(super) fn any_actionable(
        &self,
        ids: &[String],
        outcomes: &[BackgroundTaskOutcome],
    ) -> bool {
        ids.iter()
            .zip(outcomes)
            .any(|(id, outcome)| outcome.state.is_terminal() || self.is_ready(id))
    }
}

impl BackgroundTaskRegistry {
    pub async fn kill(&self, id: &str) -> Option<BackgroundTaskOutcome> {
        let (description, subagent_type, settlement) = {
            let mut tasks = self.tasks.lock().await;
            let entry = tasks.get_mut(id)?;
            if let Some(outcome) = entry.final_outcome.clone() {
                entry.observed = true;
                return Some(outcome);
            }
            if entry.cancel_requested {
                let mut outcome =
                    BackgroundTaskOutcome::running(id, &entry.description, &entry.subagent_type);
                outcome.output = "Task cancellation is still settling.".into();
                return Some(outcome);
            }
            if !self.candidates.reserve_cancel(id) {
                let mut outcome =
                    BackgroundTaskOutcome::running(id, &entry.description, &entry.subagent_type);
                outcome.output = "Task merge has started and can no longer be cancelled.".into();
                return Some(outcome);
            }
            let indexed_handle = self
                .abort_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(id);
            let abort_handle = entry.abort_handle.take().or(indexed_handle)?;
            entry.cancel_requested = true;
            let outcome = BackgroundTaskOutcome {
                id: id.to_string(),
                description: entry.description.clone(),
                subagent_type: entry.subagent_type.clone(),
                state: BackgroundTaskState::Cancelled,
                output: "Task cancelled by kill_task.".to_string(),
                applied: false,
                changed_files: Vec::new(),
            };
            (
                entry.description.clone(),
                entry.subagent_type.clone(),
                super::lifecycle::CancelSettlement {
                    abort_handle,
                    managed_job: entry.managed_job.clone(),
                    lifecycle_gate: entry.lifecycle_gate.clone(),
                    outcome,
                    terminal_outcome: entry.terminal_outcome.clone(),
                    outcomes: self.outcomes.clone(),
                    notify: entry.notify.clone(),
                    completed_notify: self.completed_notify.clone(),
                    candidates: self.candidates.clone(),
                    teardown: entry.teardown.clone(),
                },
            )
        };

        let mut settlement = tokio::spawn(settlement.run());
        match tokio::time::timeout(super::WORKER_HANDLE_ACK_TIMEOUT, &mut settlement).await {
            Ok(Ok(outcome)) => Some(
                self.commit_worker_terminal(id, &description, &subagent_type, outcome, true)
                    .await,
            ),
            Ok(Err(error)) => Some(BackgroundTaskOutcome {
                id: id.to_string(),
                description,
                subagent_type,
                state: BackgroundTaskState::Failed,
                output: format!("Task cancellation monitor failed: {error}"),
                applied: false,
                changed_files: Vec::new(),
            }),
            Err(_) => {
                let mut outcome = BackgroundTaskOutcome::running(id, &description, &subagent_type);
                outcome.output = "Task cancellation was requested and is still settling.".into();
                Some(outcome)
            }
        }
    }

    pub async fn candidate_workspace_job_id(&self, id: &str) -> Option<String> {
        let job = self.tasks.lock().await.get(id)?.managed_job.clone()?;
        job.workspace_job_id().await
    }

    pub async fn candidate_workspace_verification_ms(&self, id: &str) -> Option<u64> {
        let job = self.tasks.lock().await.get(id)?.managed_job.clone()?;
        job.workspace_job_verification_ms().await
    }

    pub fn publish_candidate(&self, id: &str, candidate: PersistedDetachedCandidate) -> Result<()> {
        self.candidates.publish(id, candidate)
    }

    pub fn claim_ready_candidates(&self) -> Vec<(String, PersistedDetachedCandidate)> {
        self.candidates.claim_ready()
    }

    pub fn track_candidate_publication(&self) -> impl Drop + Send + 'static {
        self.candidates.track_publication()
    }

    pub async fn wait_for_candidate_publications(&self, timeout: std::time::Duration) -> bool {
        self.candidates.wait_for_publications(timeout).await
    }

    pub fn candidate_is_ready(&self, id: &str) -> bool {
        self.candidates.is_ready(id)
    }

    pub fn restore_ready_candidate(&self, id: &str, candidate: PersistedDetachedCandidate) {
        self.candidates.restore(id, candidate);
    }

    pub fn resolve_candidate_applied(&self, id: &str, changed_files: Vec<String>) {
        self.candidates
            .resolve(id, CandidateResolution::Applied(changed_files));
    }

    pub fn resolve_candidate_rejected(&self, id: &str, detail: impl Into<String>) {
        self.candidates.resolve(
            id,
            CandidateResolution::Rejected {
                detail: detail.into(),
                retain_artifact: false,
            },
        );
    }

    pub fn resolve_candidate_retained(&self, id: &str, detail: impl Into<String>) {
        self.candidates.resolve(
            id,
            CandidateResolution::Rejected {
                detail: detail.into(),
                retain_artifact: true,
            },
        );
    }

    pub async fn transition_candidate(
        &self,
        id: &str,
        transition: BackgroundCandidateTransition,
        detail: Option<String>,
    ) -> Result<(), String> {
        let job = self
            .tasks
            .lock()
            .await
            .get(id)
            .and_then(|entry| entry.managed_job.clone())
            .ok_or_else(|| format!("background candidate {id} has no workspace job"))?;
        job.transition_candidate(transition, detail).await
    }
}
