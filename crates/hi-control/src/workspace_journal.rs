use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use async_trait::async_trait;
use hi_workspace::{
    AdmissionDenied, AdmissionDeniedReason, BarrierKind, BarrierReceipt, BarrierStatus,
    CrashRecoveryCapability, ExecutionReport, JobCompletion, JobId, JobPermit, JobSealOutcome,
    JobSealStatus, JobSpec, JobState, JobTerminal, MutationIntent, MutationPermit, RecoveryId,
    RecoveryOutcome, RecoveryStatus, SettlementOutcome, SettlementStatus, WorkspaceAuthority,
    WorkspaceCapabilities, WorkspaceController, WorkspaceState, WorkspaceStatus,
};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::{
    ControlError, Result, WorkspaceProjectionJournal, WorkspaceProjectionStore,
    WorkspaceRecoveryStatus,
};

#[path = "workspace_journal_operation.rs"]
mod operation;
use operation::{OperationJournalFence, apply_overlays, overlay_barrier};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalFailurePolicy {
    /// Foreground local mutations continue, but their outcome is explicitly
    /// audit-degraded and no new resumable writer job is admitted.
    LocalContinueForeground,
    /// PipeFS cannot publish success without its control record, so all new
    /// mutation admission closes after any journal ambiguity.
    PipeFsFailClosed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JournalHealthState {
    Healthy,
    LocalAuditDegraded,
    PipeFsFailClosed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalHealth {
    pub policy: JournalFailurePolicy,
    pub state: JournalHealthState,
    pub detail: Option<String>,
}

#[derive(Clone)]
pub struct JournaledWorkspaceController {
    inner: Arc<dyn WorkspaceController>,
    journal: WorkspaceProjectionJournal,
    health: Arc<Mutex<JournalHealth>>,
    permits: Arc<Mutex<BTreeMap<JobId, JobPermit>>>,
    job_journal_fences: Arc<Mutex<BTreeMap<JobId, JobJournalFence>>>,
    operation_journal_fence: Arc<Mutex<Option<OperationJournalFence>>>,
    status_tx: watch::Sender<WorkspaceStatus>,
}

#[derive(Clone, Debug)]
pub(super) enum JobJournalFence {
    /// The inner controller is being advanced, but its corresponding journal
    /// transition has not been acknowledged yet. Keep the job visible to
    /// status subscribers and barriers throughout that window.
    Pending,
    /// The inner controller advanced farther than the durable projection. The
    /// decorator owns this recovery fence because the inner controller cannot
    /// roll a terminal transition back to RecoveryRequired.
    RecoveryRequired {
        recovery_id: RecoveryId,
        detail: String,
    },
}

impl std::fmt::Debug for JournaledWorkspaceController {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JournaledWorkspaceController")
            .field("binding", &self.inner.binding())
            .field("health", &self.journal_health())
            .finish_non_exhaustive()
    }
}

impl JournaledWorkspaceController {
    pub fn attach(
        inner: Arc<dyn WorkspaceController>,
        store: Arc<dyn WorkspaceProjectionStore>,
    ) -> Result<Self> {
        let binding = inner.binding();
        let policy = match binding.authority {
            WorkspaceAuthority::Local => JournalFailurePolicy::LocalContinueForeground,
            WorkspaceAuthority::PipeFs { .. } => JournalFailurePolicy::PipeFsFailClosed,
        };
        let health = JournalHealth {
            policy,
            state: JournalHealthState::Healthy,
            detail: None,
        };
        let initial_status = inner.status();
        let (status_tx, _) = watch::channel(initial_status);
        let controller = Self {
            inner,
            journal: WorkspaceProjectionJournal::new(store),
            health: Arc::new(Mutex::new(health)),
            permits: Arc::new(Mutex::new(BTreeMap::new())),
            job_journal_fences: Arc::new(Mutex::new(BTreeMap::new())),
            operation_journal_fence: Arc::new(Mutex::new(None)),
            status_tx,
        };

        if let Err(error) = controller.journal.record_binding(
            &controller.inner.binding(),
            &controller.inner.status(),
            &controller.inner.capabilities(),
        ) {
            if policy == JournalFailurePolicy::PipeFsFailClosed {
                return Err(error);
            }
            controller.note_journal_failure(&error);
        }
        controller.start_status_forwarder();
        controller.publish_status();
        Ok(controller)
    }

    pub fn attach_store(
        inner: Arc<dyn WorkspaceController>,
        store: crate::ControlStore,
    ) -> Result<Self> {
        Self::attach(inner, Arc::new(store))
    }

    pub fn local_without_store(
        inner: Arc<dyn WorkspaceController>,
        detail: impl Into<String>,
    ) -> Result<Self> {
        if !matches!(inner.binding().authority, WorkspaceAuthority::Local) {
            return Err(ControlError::Invalid(
                "only a local controller may continue without a control journal".to_owned(),
            ));
        }
        Self::attach(
            inner,
            Arc::new(UnavailableProjectionStore {
                detail: detail.into(),
            }),
        )
    }

    pub fn journal_health(&self) -> JournalHealth {
        lock(&self.health).clone()
    }

    pub fn projection_journal(&self) -> &WorkspaceProjectionJournal {
        &self.journal
    }

    fn journal_is_healthy(&self) -> bool {
        self.journal_health().state == JournalHealthState::Healthy
    }

    fn writer_jobs_allowed(&self) -> bool {
        self.journal_is_healthy()
            && lock(&self.job_journal_fences).is_empty()
            && lock(&self.operation_journal_fence).is_none()
    }

    fn note_journal_failure(&self, error: &ControlError) {
        let mut health = lock(&self.health);
        health.state = match health.policy {
            JournalFailurePolicy::LocalContinueForeground => JournalHealthState::LocalAuditDegraded,
            JournalFailurePolicy::PipeFsFailClosed => JournalHealthState::PipeFsFailClosed,
        };
        health.detail = Some(error.to_string());
        drop(health);
        self.publish_status();
    }

    fn effective_status(&self, status: WorkspaceStatus) -> WorkspaceStatus {
        let health = self.journal_health();
        let job_fences = lock(&self.job_journal_fences).clone();
        let operation_fence = lock(&self.operation_journal_fence).clone();
        apply_overlays(status, &health, &job_fences, operation_fence.as_ref())
    }

    fn publish_status(&self) {
        let status = self.effective_status(self.inner.status());
        self.status_tx.send_replace(status);
    }

    fn start_status_forwarder(&self) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let mut inner_status = self.inner.subscribe();
        let status_tx = self.status_tx.clone();
        let health = self.health.clone();
        let job_journal_fences = self.job_journal_fences.clone();
        let operation_journal_fence = self.operation_journal_fence.clone();
        runtime.spawn(async move {
            while inner_status.changed().await.is_ok() {
                let status = inner_status.borrow_and_update().clone();
                status_tx.send_replace(apply_overlays(
                    status,
                    &lock(&health),
                    &lock(&job_journal_fences),
                    lock(&operation_journal_fence).as_ref(),
                ));
            }
        });
    }

    fn deny(&self, detail: impl Into<String>) -> AdmissionDenied {
        AdmissionDenied {
            reason: AdmissionDeniedReason::CapabilityUnavailable,
            state: self.status().state,
            detail: detail.into(),
        }
    }

    fn project_binding(&self) {
        let status = self.effective_status(self.inner.status());
        if let Err(error) =
            self.journal
                .record_binding(&self.inner.binding(), &status, &self.inner.capabilities())
        {
            self.note_journal_failure(&error);
        }
        self.publish_status();
    }

    fn journal_fence_denial(&self) -> Option<&'static str> {
        if lock(&self.operation_journal_fence).is_some() {
            return Some("workspace operation publication requires journal settlement");
        }
        let fences = lock(&self.job_journal_fences);
        if fences
            .values()
            .any(|fence| matches!(fence, JobJournalFence::RecoveryRequired { .. }))
        {
            Some("workspace job publication requires journal recovery")
        } else if !fences.is_empty() {
            Some("workspace job publication is still settling")
        } else {
            None
        }
    }

    fn forced_job_recovery(&self, recovery: &RecoveryId) -> Option<(JobId, String)> {
        lock(&self.job_journal_fences)
            .iter()
            .find_map(|(job, fence)| match fence {
                JobJournalFence::RecoveryRequired {
                    recovery_id,
                    detail,
                } if recovery_id == recovery => Some((job.clone(), detail.clone())),
                JobJournalFence::Pending | JobJournalFence::RecoveryRequired { .. } => None,
            })
    }
}

#[async_trait]
impl WorkspaceController for JournaledWorkspaceController {
    fn binding(&self) -> hi_workspace::WorkspaceBinding {
        self.inner.binding()
    }

    fn capabilities(&self) -> WorkspaceCapabilities {
        let mut capabilities = self.inner.capabilities();
        if self.journal_health().policy == JournalFailurePolicy::LocalContinueForeground {
            capabilities.crash_recovery = if self.journal_is_healthy() {
                CrashRecoveryCapability::LocalJournal
            } else {
                CrashRecoveryCapability::None
            };
        }
        if !self.writer_jobs_allowed() {
            capabilities.background_writers = false;
            capabilities.candidate_apply = false;
        }
        capabilities
    }

    fn status(&self) -> WorkspaceStatus {
        self.effective_status(self.inner.status())
    }

    fn subscribe(&self) -> watch::Receiver<WorkspaceStatus> {
        self.status_tx.subscribe()
    }

    async fn begin(
        &self,
        intent: MutationIntent,
    ) -> std::result::Result<MutationPermit, AdmissionDenied> {
        if let Some(detail) = self.journal_fence_denial() {
            return Err(self.deny(detail));
        }
        if self.journal_health().state == JournalHealthState::PipeFsFailClosed {
            return Err(self.deny("PipeFS mutation admission is closed until journal recovery"));
        }
        let permit = self.inner.begin(intent).await?;
        if let Err(error) = hi_workspace::hit_harness_failpoint(
            hi_workspace::HarnessFailpoint::AdmissionBeforeJournal,
        ) {
            drop(permit);
            self.publish_status();
            return Err(self.deny(error.to_string()));
        }
        if self.journal_is_healthy() {
            let binding = self.inner.binding();
            if let Err(error) = self
                .journal
                .record_operation_admitted(&binding, &permit.snapshot())
            {
                self.note_journal_failure(&error);
                if self.journal_health().policy == JournalFailurePolicy::PipeFsFailClosed {
                    drop(permit);
                    self.publish_status();
                    return Err(
                        self.deny("PipeFS operation admission could not be durably journaled")
                    );
                }
            }
        }
        if let Err(error) = hi_workspace::hit_harness_failpoint(
            hi_workspace::HarnessFailpoint::AdmissionAfterJournal,
        ) {
            drop(permit);
            self.publish_status();
            return Err(self.deny(error.to_string()));
        }
        self.project_binding();
        Ok(permit)
    }

    async fn settle(
        &self,
        permit: MutationPermit,
        mut execution: ExecutionReport,
    ) -> SettlementOutcome {
        let permit_record = permit.snapshot();
        let pipefs_publication = self.begin_operation_publication(&permit_record);
        let binding_before = self.inner.binding();
        if let Err(error) = hi_workspace::hit_harness_failpoint(
            hi_workspace::HarnessFailpoint::ExecutionAfterEffect,
        ) {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.workspace_may_have_changed = true;
            execution.external_effect_may_have_occurred = true;
            execution.detail = Some(error.to_string());
        }
        let execution_record = execution.clone();
        if self.journal_is_healthy()
            && let Err(error) =
                self.journal
                    .record_operation_execution(&binding_before, &permit_record, &execution)
        {
            self.note_journal_failure(&error);
        }

        // Once execution has been accepted, settlement is always attempted,
        // even if its audit record failed. Dropping the permit here would turn
        // known execution into an abandoned-operation falsehood.
        let mut outcome = self.inner.settle(permit, execution).await;
        if pipefs_publication
            && matches!(
                outcome.status,
                SettlementStatus::Durable | SettlementStatus::NoChange
            )
        {
            return self.finish_operation_publication(permit_record, execution_record, outcome);
        }
        if pipefs_publication {
            self.release_pending_operation(&permit_record.operation_id);
        }
        let binding_after = self.inner.binding();
        if let Err(error) =
            self.journal
                .record_operation_settled(&binding_after, &permit_record, &outcome)
        {
            self.note_journal_failure(&error);
        }
        if let Some(recovery_id) = &outcome.recovery_id
            && let Err(error) = self.journal.record_recovery(
                &binding_after,
                recovery_id,
                Some(permit_record.operation_id.to_string()),
                None,
                WorkspaceRecoveryStatus::Required,
                outcome.detail.clone(),
            )
        {
            self.note_journal_failure(&error);
        }
        self.project_binding();

        let health = self.journal_health();
        match health.state {
            JournalHealthState::Healthy => {}
            JournalHealthState::LocalAuditDegraded => {
                if matches!(
                    outcome.status,
                    SettlementStatus::Durable | SettlementStatus::NoChange
                ) {
                    outcome.status = SettlementStatus::LocalAuditDegraded;
                    outcome.detail = Some(
                        "workspace settled, but the local audit journal is degraded".to_owned(),
                    );
                }
            }
            JournalHealthState::PipeFsFailClosed => {
                outcome.status = SettlementStatus::RecoveryRequired;
                outcome.detail = Some(
                    "PipeFS settlement is not publishable until the control journal is reconciled"
                        .to_owned(),
                );
            }
        }
        outcome
    }

    async fn register_job(&self, spec: JobSpec) -> std::result::Result<JobPermit, AdmissionDenied> {
        if let Some(detail) = self.journal_fence_denial() {
            return Err(self.deny(detail));
        }
        let writer = !matches!(spec.effect_scope, hi_workspace::EffectScope::ReadOnly);
        if writer && !self.writer_jobs_allowed() {
            return Err(self
                .deny("resumable and background writer jobs require a healthy workspace journal"));
        }
        if self.journal_health().state == JournalHealthState::PipeFsFailClosed {
            return Err(self.deny("PipeFS job admission is closed until journal recovery"));
        }

        let permit = self.inner.register_job(spec).await?;
        lock(&self.permits).insert(permit.job_id.clone(), permit.clone());
        if let Err(error) = self
            .journal
            .record_job_registered(&self.inner.binding(), &permit)
        {
            self.note_journal_failure(&error);
            if writer || self.journal_health().policy == JournalFailurePolicy::PipeFsFailClosed {
                let _ = self
                    .inner
                    .seal_job(
                        permit.job_id.clone(),
                        JobTerminal {
                            completion: JobCompletion::Failed,
                            detail: Some(
                                "job admission failed before execution because it was not durably journaled"
                                    .to_owned(),
                            ),
                            artifacts: Vec::new(),
                        },
                    )
                    .await;
                self.publish_status();
                return Err(self.deny("job admission could not be durably journaled"));
            }
        }
        self.project_binding();
        Ok(permit)
    }

    async fn seal_job(&self, job: JobId, terminal: JobTerminal) -> JobSealOutcome {
        let permit = lock(&self.permits).get(&job).cloned();
        let must_fence = permit.as_ref().is_some_and(|permit| {
            !matches!(
                permit.spec.effect_scope,
                hi_workspace::EffectScope::ReadOnly
            )
        }) || self.journal_health().policy
            == JournalFailurePolicy::PipeFsFailClosed;
        if must_fence {
            let mut fences = lock(&self.job_journal_fences);
            match fences.get(&job) {
                Some(JobJournalFence::RecoveryRequired {
                    recovery_id,
                    detail,
                }) => {
                    return JobSealOutcome {
                        job_id: job,
                        status: JobSealStatus::Rejected,
                        state: Some(JobState::RecoveryRequired),
                        recovery_id: Some(recovery_id.clone()),
                        detail: Some(detail.clone()),
                    };
                }
                Some(JobJournalFence::Pending) => {
                    return JobSealOutcome {
                        job_id: job,
                        status: JobSealStatus::Rejected,
                        state: Some(JobState::Settling),
                        recovery_id: None,
                        detail: Some("job publication is already settling".to_owned()),
                    };
                }
                None => {
                    fences.insert(job.clone(), JobJournalFence::Pending);
                }
            }
            drop(fences);
            self.publish_status();
        }

        let outcome = self.inner.seal_job(job.clone(), terminal.clone()).await;
        if outcome.status == JobSealStatus::Sealed {
            if let Some(permit) = permit {
                if let Err(error) = self.journal.record_job_sealed(
                    &self.inner.binding(),
                    &permit,
                    &outcome,
                    &terminal,
                ) {
                    self.note_journal_failure(&error);
                    let writer = !matches!(
                        permit.spec.effect_scope,
                        hi_workspace::EffectScope::ReadOnly
                    );
                    if writer
                        || self.journal_health().policy == JournalFailurePolicy::PipeFsFailClosed
                    {
                        let binding = self.inner.binding();
                        let recovery_id = journal_job_recovery_id(&binding, &job);
                        let detail = format!(
                            "job reached inner state {:?}, but its lifecycle transition was not durably journaled: {error}",
                            outcome.state.unwrap_or(JobState::RecoveryRequired)
                        );
                        lock(&self.job_journal_fences).insert(
                            job.clone(),
                            JobJournalFence::RecoveryRequired {
                                recovery_id: recovery_id.clone(),
                                detail: detail.clone(),
                            },
                        );
                        if let Err(recovery_error) = self.journal.record_recovery(
                            &binding,
                            &recovery_id,
                            None,
                            Some(job.to_string()),
                            WorkspaceRecoveryStatus::Required,
                            Some(detail.clone()),
                        ) {
                            self.note_journal_failure(&recovery_error);
                        }
                        self.project_binding();
                        return JobSealOutcome {
                            job_id: job,
                            status: JobSealStatus::Rejected,
                            state: Some(JobState::RecoveryRequired),
                            recovery_id: Some(recovery_id),
                            detail: Some(detail),
                        };
                    }
                }
                lock(&self.job_journal_fences).remove(&job);
                if outcome.state.is_some_and(JobState::is_terminal) {
                    lock(&self.permits).remove(&job);
                }
            } else {
                let error = ControlError::Invalid(format!(
                    "missing job permit for lifecycle callback {job}"
                ));
                self.note_journal_failure(&error);
                let binding = self.inner.binding();
                let recovery_id = journal_job_recovery_id(&binding, &job);
                let detail = error.to_string();
                lock(&self.job_journal_fences).insert(
                    job.clone(),
                    JobJournalFence::RecoveryRequired {
                        recovery_id: recovery_id.clone(),
                        detail: detail.clone(),
                    },
                );
                self.project_binding();
                return JobSealOutcome {
                    job_id: job,
                    status: JobSealStatus::Rejected,
                    state: Some(JobState::RecoveryRequired),
                    recovery_id: Some(recovery_id),
                    detail: Some(detail),
                };
            }
        } else if must_fence {
            lock(&self.job_journal_fences).remove(&job);
        }
        if let Some(recovery_id) = &outcome.recovery_id
            && let Err(error) = self.journal.record_recovery(
                &self.inner.binding(),
                recovery_id,
                None,
                Some(job.to_string()),
                WorkspaceRecoveryStatus::Required,
                outcome.detail.clone(),
            )
        {
            self.note_journal_failure(&error);
        }
        self.project_binding();
        outcome
    }

    async fn barrier(&self, reason: BarrierKind, deadline: Instant) -> BarrierReceipt {
        let receipt = self.inner.barrier(reason, deadline).await;
        overlay_barrier(
            receipt,
            &self.journal_health(),
            &lock(&self.job_journal_fences),
            lock(&self.operation_journal_fence).as_ref(),
            deadline,
        )
    }

    async fn reconcile(&self, recovery: RecoveryId) -> RecoveryOutcome {
        if let Some(outcome) = self.reconcile_operation_publication(&recovery) {
            return outcome;
        }
        if let Some((_job, detail)) = self.forced_job_recovery(&recovery) {
            return RecoveryOutcome {
                recovery_id: recovery,
                status: RecoveryStatus::Pending,
                binding: self.inner.binding(),
                detail: Some(format!(
                    "{detail}; restart or restore the workspace journal before recovery can complete"
                )),
            };
        }
        let mut outcome = self.inner.reconcile(recovery).await;
        if let Err(error) = self
            .journal
            .record_recovery_outcome(&self.inner.binding(), &outcome)
        {
            self.note_journal_failure(&error);
        }
        self.project_binding();
        if self.journal_health().state == JournalHealthState::PipeFsFailClosed {
            outcome.status = RecoveryStatus::Pending;
            outcome.detail = Some(
                "workspace recovery cannot complete until the PipeFS journal is reconciled"
                    .to_owned(),
            );
        }
        outcome
    }
}

struct UnavailableProjectionStore {
    detail: String,
}

impl UnavailableProjectionStore {
    fn unavailable<T>(&self) -> Result<T> {
        Err(ControlError::Invalid(self.detail.clone()))
    }
}

impl WorkspaceProjectionStore for UnavailableProjectionStore {
    fn commit(
        &self,
        _transition: crate::ProjectionTransition,
        _event: hi_events::RunEvent,
    ) -> Result<crate::ProjectionEventReceipt> {
        self.unavailable()
    }

    fn binding(&self, _id: &str) -> Result<Option<crate::WorkspaceBindingRecord>> {
        self.unavailable()
    }

    fn operation(&self, _id: &str) -> Result<Option<crate::WorkspaceOperationRecord>> {
        self.unavailable()
    }

    fn operations_for_binding(
        &self,
        _binding_id: &str,
    ) -> Result<Vec<crate::WorkspaceOperationRecord>> {
        self.unavailable()
    }

    fn job(&self, _id: &str) -> Result<Option<crate::ControlJobRecord>> {
        self.unavailable()
    }

    fn recovery(&self, _id: &str) -> Result<Option<crate::WorkspaceRecoveryRecord>> {
        self.unavailable()
    }

    fn recoveries_for_operation(
        &self,
        _operation_id: &str,
    ) -> Result<Vec<crate::WorkspaceRecoveryRecord>> {
        self.unavailable()
    }

    fn recoveries_for_job(&self, _job_id: &str) -> Result<Vec<crate::WorkspaceRecoveryRecord>> {
        self.unavailable()
    }

    fn jobs_for_binding(&self, _binding_id: &str) -> Result<Vec<crate::ControlJobRecord>> {
        self.unavailable()
    }
}

fn journal_job_recovery_id(binding: &hi_workspace::WorkspaceBinding, job: &JobId) -> RecoveryId {
    let identity = format!(
        "hi-control-v2:journal-job-recovery:{}:{}:{job}",
        binding.binding_id, binding.epoch
    );
    RecoveryId::new(uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, identity.as_bytes()).to_string())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
#[path = "workspace_journal_tests.rs"]
mod tests;
