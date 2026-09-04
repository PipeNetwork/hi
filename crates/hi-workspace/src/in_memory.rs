use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::watch;

use crate::{
    AdmissionDenied, AdmissionDeniedReason, BarrierKind, BarrierReceipt, BarrierStatus, BindingId,
    ExecutionDisposition, ExecutionReport, JobCompletion, JobId, JobPermit, JobRegistryLimits,
    JobSealOutcome, JobSealStatus, JobSpec, JobState, JobTerminal, MutationIntent, MutationPermit,
    MutationPermitRecord, OperationId, PermitAbandonment, PermitIssuer, RecoveryId, RecoveryKind,
    RecoveryOutcome, RecoveryRecord, RecoveryStatus, SettlementOutcome, SettlementReceipt,
    SettlementStatus, WORKSPACE_CONTRACT_SCHEMA_VERSION, WorkspaceBinding, WorkspaceCapabilities,
    WorkspaceController, WorkspaceId, WorkspaceState, WorkspaceStatus,
};

#[path = "in_memory_limits.rs"]
mod limits;

#[derive(Clone)]
pub struct InMemoryWorkspaceController {
    inner: Arc<Inner>,
}

struct Inner {
    state: Mutex<State>,
    status_tx: watch::Sender<WorkspaceStatus>,
    issuer: PermitIssuer,
}

struct State {
    binding: WorkspaceBinding,
    capabilities: WorkspaceCapabilities,
    status: WorkspaceStatus,
    active_operation: Option<MutationPermitRecord>,
    job_limits: JobRegistryLimits,
    jobs: BTreeMap<JobId, JobRecord>,
    recoveries: BTreeMap<RecoveryId, RecoveryRecord>,
}

struct JobRecord {
    permit: JobPermit,
    state: JobState,
    recovery_id: Option<RecoveryId>,
}

struct AbandonmentHandler {
    inner: Weak<Inner>,
}

impl PermitAbandonment for AbandonmentHandler {
    fn mutation_abandoned(&self, permit: &MutationPermitRecord) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let mut state = lock(&inner.state);
        if state
            .active_operation
            .as_ref()
            .is_none_or(|active| active.operation_id != permit.operation_id)
        {
            return;
        }
        let recovery = make_recovery(
            &state.binding,
            RecoveryKind::AbandonedMutation,
            Some(permit.operation_id.clone()),
            None,
            "admitted mutation permit was dropped before settlement",
        );
        state.active_operation = None;
        state.status.active_operation = None;
        state.status.state = WorkspaceState::RecoveryRequired;
        state.status.recovery_id = Some(recovery.recovery_id.clone());
        state.status.detail = Some(recovery.detail.clone());
        state
            .recoveries
            .insert(recovery.recovery_id.clone(), recovery);
        publish(&inner, &mut state);
    }
}

impl InMemoryWorkspaceController {
    pub fn new_local(
        workspace_id: impl Into<WorkspaceId>,
        workspace_root: impl Into<PathBuf>,
        state_root: impl Into<PathBuf>,
    ) -> Self {
        Self::new_local_at_epoch(workspace_id, workspace_root, state_root, 0)
    }

    pub fn new_local_at_epoch(
        workspace_id: impl Into<WorkspaceId>,
        workspace_root: impl Into<PathBuf>,
        state_root: impl Into<PathBuf>,
        epoch: u64,
    ) -> Self {
        Self::new_local_at_epoch_with_job_limits(
            workspace_id,
            workspace_root,
            state_root,
            epoch,
            JobRegistryLimits::default(),
        )
    }

    /// Compatibility state machine around a legacy PipeFS durability backend.
    /// The host remains responsible for byte and transcript settlement.
    pub fn new_pipefs(
        workspace_id: impl Into<WorkspaceId>,
        session_id: impl Into<String>,
        writer_protocol: u16,
        causal_commit: bool,
        workspace_root: impl Into<PathBuf>,
        state_root: impl Into<PathBuf>,
    ) -> Self {
        Self::new_pipefs_at_epoch(
            workspace_id,
            session_id,
            writer_protocol,
            causal_commit,
            workspace_root,
            state_root,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_pipefs_at_epoch(
        workspace_id: impl Into<WorkspaceId>,
        session_id: impl Into<String>,
        writer_protocol: u16,
        causal_commit: bool,
        workspace_root: impl Into<PathBuf>,
        state_root: impl Into<PathBuf>,
        epoch: u64,
    ) -> Self {
        Self::new_pipefs_at_epoch_with_job_limits(
            workspace_id,
            session_id,
            writer_protocol,
            causal_commit,
            workspace_root,
            state_root,
            epoch,
            JobRegistryLimits::default(),
        )
    }

    /// Rebind a quiescent local controller and fence every record from the old
    /// binding by issuing a new binding id and epoch.
    pub fn rebind(
        &self,
        workspace_root: impl Into<PathBuf>,
        state_root: impl Into<PathBuf>,
    ) -> Result<WorkspaceBinding, AdmissionDenied> {
        let mut state = lock(&self.inner.state);
        let pending_jobs = nonterminal_job_ids(&state.jobs);
        if state.active_operation.is_some()
            || !pending_jobs.is_empty()
            || !state.status.state.admits_mutation()
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::NotReady,
                "workspace must be ready and have no active jobs before rebind",
            ));
        }
        if let Err(error) = crate::hit_harness_failpoint(crate::HarnessFailpoint::RebindAfterDrain)
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::Incompatible,
                error.to_string(),
            ));
        }
        state.binding.binding_id = BindingId::new(uuid::Uuid::new_v4().to_string());
        state.binding.epoch = state.binding.epoch.saturating_add(1);
        state.binding.workspace_root = workspace_root.into();
        state.binding.state_root = state_root.into();
        state.binding.version = crate::WorkspaceVersion::Local {
            generation: 0,
            content_digest: None,
        };
        state.status.binding_id = state.binding.binding_id.clone();
        state.status.epoch = state.binding.epoch;
        state.status.state = WorkspaceState::Ready;
        state.status.detail = None;
        publish(&self.inner, &mut state);
        Ok(state.binding.clone())
    }

    pub fn recovery(&self, recovery_id: &RecoveryId) -> Option<RecoveryRecord> {
        lock(&self.inner.state).recoveries.get(recovery_id).cloned()
    }

    /// Seed a durable recovery discovered before this controller begins
    /// admitting work. Multiple records remain fenced one at a time; resolving
    /// the current record promotes the next unresolved record instead of
    /// prematurely returning to `Ready`.
    pub fn require_recovery(&self, record: RecoveryRecord) -> Result<(), AdmissionDenied> {
        let mut state = lock(&self.inner.state);
        if state.active_operation.is_some() || !nonterminal_job_ids(&state.jobs).is_empty() {
            return Err(denied(
                &state,
                AdmissionDeniedReason::NotReady,
                "cannot seed recovery while workspace work is active",
            ));
        }
        if record.binding_id != state.binding.binding_id
            || record.epoch != state.binding.epoch
            || record.resolved
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::StaleBinding,
                "recovery record does not belong to the current binding and epoch",
            ));
        }
        state.status.state = WorkspaceState::RecoveryRequired;
        state.status.recovery_id = Some(record.recovery_id.clone());
        state.status.detail = Some(record.detail.clone());
        state.recoveries.insert(record.recovery_id.clone(), record);
        publish(&self.inner, &mut state);
        Ok(())
    }

    pub fn job_state(&self, job_id: &JobId) -> Option<JobState> {
        lock(&self.inner.state)
            .jobs
            .get(job_id)
            .map(|record| record.state)
    }
}

#[async_trait]
impl WorkspaceController for InMemoryWorkspaceController {
    fn binding(&self) -> WorkspaceBinding {
        lock(&self.inner.state).binding.clone()
    }

    fn capabilities(&self) -> WorkspaceCapabilities {
        lock(&self.inner.state).capabilities.clone()
    }

    fn status(&self) -> WorkspaceStatus {
        lock(&self.inner.state).status.clone()
    }

    fn subscribe(&self) -> watch::Receiver<WorkspaceStatus> {
        self.inner.status_tx.subscribe()
    }

    async fn begin(&self, intent: MutationIntent) -> Result<MutationPermit, AdmissionDenied> {
        let mut state = lock(&self.inner.state);
        if !state.status.state.admits_mutation() {
            return Err(denied(
                &state,
                AdmissionDeniedReason::NotReady,
                "workspace has an unsettled operation",
            ));
        }
        if state.active_operation.is_some() {
            return Err(denied(
                &state,
                AdmissionDeniedReason::ActiveMutation,
                "another mutation is already active",
            ));
        }
        let live_writer_states = state
            .jobs
            .values()
            .filter(|job| {
                !job.state.is_terminal()
                    && matches!(job.permit.spec.effect_scope, crate::EffectScope::LiveWriter)
            })
            .map(|job| job.state)
            .collect::<Vec<_>>();
        let writers_ready_to_reconcile = !live_writer_states.is_empty()
            && live_writer_states
                .iter()
                .all(|state| *state == JobState::DurabilityPending);
        if !(live_writer_states.is_empty()
            || intent.is_reconciliation() && writers_ready_to_reconcile)
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::ActiveWriter,
                "a live writer job is active",
            ));
        }

        let record = MutationPermitRecord {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            controller_id: self.inner.issuer.controller_id().clone(),
            operation_id: OperationId::new(uuid::Uuid::new_v4().to_string()),
            idempotency_key: intent.replay_class.operation_idempotency_key(),
            binding_id: state.binding.binding_id.clone(),
            epoch: state.binding.epoch,
            base_version: state.binding.version.clone(),
            intent,
            issued_at_ms: now_ms(),
        };
        state.active_operation = Some(record.clone());
        state.status.state = WorkspaceState::Mutating;
        state.status.active_operation = Some(record.operation_id.clone());
        state.status.detail = None;
        publish(&self.inner, &mut state);
        Ok(self.inner.issuer.issue_mutation(record))
    }

    async fn settle(
        &self,
        mut permit: MutationPermit,
        execution: ExecutionReport,
    ) -> SettlementOutcome {
        let fallback_operation = permit.record().operation_id.clone();
        let record = match self.inner.issuer.claim_mutation(&mut permit) {
            Ok(record) => record,
            Err(error) => {
                return SettlementOutcome {
                    status: SettlementStatus::Incompatible,
                    operation_id: fallback_operation,
                    receipt: None,
                    recovery_id: None,
                    detail: Some(error.to_string()),
                };
            }
        };

        let mut state = lock(&self.inner.state);
        let is_current = state.active_operation.as_ref().is_some_and(|active| {
            active.operation_id == record.operation_id
                && active.binding_id == record.binding_id
                && active.epoch == record.epoch
        });
        if !is_current {
            return SettlementOutcome {
                status: SettlementStatus::Incompatible,
                operation_id: record.operation_id,
                receipt: None,
                recovery_id: state.status.recovery_id.clone(),
                detail: Some("mutation permit is stale or is not the active operation".to_owned()),
            };
        }

        state.status.state = WorkspaceState::Settling;
        publish(&self.inner, &mut state);

        // Execution failure is not settlement ambiguity. Once the caller has
        // reaped the process and supplied a definite Failed/Cancelled report,
        // the controller can journal the observed workspace version and the
        // failure together. Only an indeterminate execution lacks enough
        // evidence to publish a terminal receipt.
        let uncertain = matches!(execution.disposition, ExecutionDisposition::Indeterminate);

        if uncertain {
            let recovery = make_recovery(
                &state.binding,
                RecoveryKind::UnsettledMutation,
                Some(record.operation_id.clone()),
                None,
                execution
                    .detail
                    .as_deref()
                    .unwrap_or("mutation effects could not be proven settled"),
            );
            let status = if matches!(execution.disposition, ExecutionDisposition::Indeterminate) {
                SettlementStatus::Indeterminate
            } else {
                SettlementStatus::RecoveryRequired
            };
            state.active_operation = None;
            state.status.active_operation = None;
            state.status.state = WorkspaceState::RecoveryRequired;
            state.status.recovery_id = Some(recovery.recovery_id.clone());
            state.status.detail = Some(recovery.detail.clone());
            state
                .recoveries
                .insert(recovery.recovery_id.clone(), recovery.clone());
            publish(&self.inner, &mut state);
            return SettlementOutcome {
                status,
                operation_id: record.operation_id,
                receipt: None,
                recovery_id: Some(recovery.recovery_id),
                detail: execution.detail,
            };
        }

        let changed = execution.workspace_may_have_changed;
        if changed {
            state.binding.version = state
                .binding
                .version
                .advance_after_settlement(execution.content_digest.clone());
        }
        let receipt = SettlementReceipt {
            receipt_id: uuid::Uuid::new_v4().to_string(),
            operation_id: record.operation_id.clone(),
            binding_id: state.binding.binding_id.clone(),
            epoch: state.binding.epoch,
            version: state.binding.version.clone(),
            transcript_cursor: None,
        };
        state.active_operation = None;
        state.status.active_operation = None;
        state.status.state = WorkspaceState::Ready;
        state.status.recovery_id = None;
        state.status.detail = None;
        publish(&self.inner, &mut state);
        SettlementOutcome {
            status: if changed || execution.external_effect_may_have_occurred {
                SettlementStatus::Durable
            } else {
                SettlementStatus::NoChange
            },
            operation_id: record.operation_id,
            receipt: Some(receipt),
            recovery_id: None,
            detail: execution.detail,
        }
    }

    async fn register_job(&self, spec: JobSpec) -> Result<JobPermit, AdmissionDenied> {
        let mut state = lock(&self.inner.state);
        let belongs_to_active = state.active_operation.as_ref().is_some_and(|operation| {
            spec.parent_operation.as_ref() == Some(&operation.operation_id)
        });
        if !state.status.state.admits_mutation() && !belongs_to_active {
            return Err(denied(
                &state,
                AdmissionDeniedReason::NotReady,
                "workspace has an unsettled operation",
            ));
        }
        if matches!(spec.effect_scope, crate::EffectScope::LiveWriter)
            && ((state.active_operation.is_some() && !belongs_to_active)
                || state.jobs.values().any(|job| {
                    !job.state.is_terminal()
                        && matches!(job.permit.spec.effect_scope, crate::EffectScope::LiveWriter)
                }))
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::ActiveWriter,
                "another live writer is active",
            ));
        }
        let active_jobs = state
            .jobs
            .values()
            .filter(|job| !job.state.is_terminal())
            .count();
        if active_jobs >= state.job_limits.max_active_jobs {
            return Err(denied(
                &state,
                AdmissionDeniedReason::ActiveWriter,
                format!(
                    "active job limit reached ({})",
                    state.job_limits.max_active_jobs
                ),
            ));
        }
        if is_candidate_spec(&spec)
            && state
                .jobs
                .values()
                .filter(|job| is_candidate_spec(&job.permit.spec))
                .filter(|job| matches!(job.state, JobState::Starting | JobState::Running))
                .count()
                >= state.job_limits.max_preparations
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::ActiveWriter,
                format!(
                    "candidate preparation limit reached ({})",
                    state.job_limits.max_preparations
                ),
            ));
        }

        let permit = JobPermit {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            controller_id: state.binding.controller_id.clone(),
            job_id: JobId::new(uuid::Uuid::new_v4().to_string()),
            binding_id: state.binding.binding_id.clone(),
            epoch: state.binding.epoch,
            spec,
            issued_at_ms: now_ms(),
        };
        state.jobs.insert(
            permit.job_id.clone(),
            JobRecord {
                permit: permit.clone(),
                state: JobState::Running,
                recovery_id: None,
            },
        );
        state.status.active_jobs = nonterminal_job_ids(&state.jobs);
        publish(&self.inner, &mut state);
        Ok(permit)
    }

    async fn seal_job(&self, job: JobId, terminal: JobTerminal) -> JobSealOutcome {
        let mut state = lock(&self.inner.state);
        let Some((current, job_recovery_id)) = state
            .jobs
            .get(&job)
            .map(|record| (record.state, record.recovery_id.clone()))
        else {
            return JobSealOutcome {
                job_id: job,
                status: JobSealStatus::NotFound,
                state: None,
                recovery_id: None,
                detail: Some("job was not registered".to_owned()),
            };
        };
        if current.is_terminal() {
            return JobSealOutcome {
                job_id: job,
                status: JobSealStatus::AlreadySealed,
                state: Some(current),
                recovery_id: job_recovery_id,
                detail: terminal.detail,
            };
        }
        let next = completion_state(terminal.completion);
        let transition_allowed = state
            .jobs
            .get(&job)
            .is_some_and(|record| job_transition_allowed(record, next));
        if !transition_allowed {
            return JobSealOutcome {
                job_id: job,
                status: JobSealStatus::Rejected,
                state: Some(current),
                recovery_id: state.status.recovery_id.clone(),
                detail: Some(format!("illegal job transition {current:?} -> {next:?}")),
            };
        }

        state.jobs.get_mut(&job).expect("job was just found").state = next;
        let recovery = if matches!(next, JobState::RecoveryRequired) {
            let recovery = make_recovery(
                &state.binding,
                RecoveryKind::CrashedWriterJob,
                None,
                Some(job.clone()),
                terminal
                    .detail
                    .as_deref()
                    .unwrap_or("job requires workspace recovery"),
            );
            state.status.state = WorkspaceState::RecoveryRequired;
            state.status.recovery_id = Some(recovery.recovery_id.clone());
            state.status.detail = Some(recovery.detail.clone());
            state
                .recoveries
                .insert(recovery.recovery_id.clone(), recovery.clone());
            state
                .jobs
                .get_mut(&job)
                .expect("job was just found")
                .recovery_id = Some(recovery.recovery_id.clone());
            Some(recovery.recovery_id)
        } else {
            None
        };
        state.status.active_jobs = nonterminal_job_ids(&state.jobs);
        publish(&self.inner, &mut state);
        JobSealOutcome {
            job_id: job,
            status: JobSealStatus::Sealed,
            state: Some(next),
            recovery_id: recovery,
            detail: terminal.detail,
        }
    }

    async fn barrier(&self, reason: BarrierKind, deadline: Instant) -> BarrierReceipt {
        let state = lock(&self.inner.state);
        let pending_jobs = nonterminal_job_ids(&state.jobs);
        let status = if state.status.state == WorkspaceState::RecoveryRequired {
            BarrierStatus::RecoveryRequired
        } else if state.active_operation.is_none() && pending_jobs.is_empty() {
            BarrierStatus::Passed
        } else if Instant::now() >= deadline {
            BarrierStatus::TimedOut
        } else {
            BarrierStatus::Blocked
        };
        BarrierReceipt {
            kind: reason,
            status,
            binding_id: state.binding.binding_id.clone(),
            epoch: state.binding.epoch,
            active_operation: state
                .active_operation
                .as_ref()
                .map(|operation| operation.operation_id.clone()),
            pending_jobs,
            recovery_id: state.status.recovery_id.clone(),
            detail: state.status.detail.clone(),
        }
    }

    async fn reconcile(&self, recovery: RecoveryId) -> RecoveryOutcome {
        let mut state = lock(&self.inner.state);
        let Some(record) = state.recoveries.get_mut(&recovery) else {
            return RecoveryOutcome {
                recovery_id: recovery,
                status: RecoveryStatus::NotFound,
                binding: state.binding.clone(),
                detail: Some("recovery record was not found".to_owned()),
            };
        };
        if record.resolved {
            return RecoveryOutcome {
                recovery_id: recovery,
                status: RecoveryStatus::Recovered,
                binding: state.binding.clone(),
                detail: Some("recovery was already resolved".to_owned()),
            };
        }
        record.resolved = true;
        let recovered_job = record.job_id.clone();
        if let Some(job_id) = recovered_job
            && let Some(job) = state.jobs.get_mut(&job_id)
            && job.state == JobState::RecoveryRequired
        {
            job.state = JobState::Failed;
        }
        state.status.active_jobs = nonterminal_job_ids(&state.jobs);
        if state.status.recovery_id.as_ref() == Some(&recovery) {
            if let Some(next) = state
                .recoveries
                .values()
                .find(|candidate| !candidate.resolved)
                .cloned()
            {
                state.status.state = WorkspaceState::RecoveryRequired;
                state.status.recovery_id = Some(next.recovery_id);
                state.status.detail = Some(next.detail);
            } else {
                state.status.state = if state.active_operation.is_some() {
                    WorkspaceState::Mutating
                } else {
                    WorkspaceState::Ready
                };
                state.status.recovery_id = None;
                state.status.detail = None;
            }
        }
        publish(&self.inner, &mut state);
        RecoveryOutcome {
            recovery_id: recovery,
            status: RecoveryStatus::Recovered,
            binding: state.binding.clone(),
            detail: None,
        }
    }
}

fn completion_state(completion: JobCompletion) -> JobState {
    match completion {
        JobCompletion::Succeeded => JobState::Succeeded,
        JobCompletion::ReadyToMerge => JobState::ReadyToMerge,
        JobCompletion::Merging => JobState::Merging,
        JobCompletion::Settling => JobState::Settling,
        JobCompletion::Failed => JobState::Failed,
        JobCompletion::Cancelled => JobState::Cancelled,
        JobCompletion::DurabilityPending => JobState::DurabilityPending,
        JobCompletion::RecoveryRequired => JobState::RecoveryRequired,
        JobCompletion::Stale => JobState::Stale,
    }
}

/// Keep the always-present local controller on the same publication fence as
/// [`WorkspaceJobRegistry`]. Read-only work may finish directly from Running,
/// but a candidate or live writer must first enter Settling so a lifecycle
/// adapter cannot publish success before the workspace receipt exists.
fn job_transition_allowed(record: &JobRecord, next: JobState) -> bool {
    if !record.state.can_transition_to(next) {
        return false;
    }
    let write_job = record.permit.spec.kind == crate::JobKind::WriteCandidate
        || !matches!(
            record.permit.spec.effect_scope,
            crate::EffectScope::ReadOnly
        );
    !write_job || next != JobState::Succeeded || record.state == JobState::Settling
}

fn make_recovery(
    binding: &WorkspaceBinding,
    kind: RecoveryKind,
    operation_id: Option<OperationId>,
    job_id: Option<JobId>,
    detail: impl Into<String>,
) -> RecoveryRecord {
    RecoveryRecord {
        schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
        recovery_id: RecoveryId::new(uuid::Uuid::new_v4().to_string()),
        kind,
        binding_id: binding.binding_id.clone(),
        epoch: binding.epoch,
        operation_id,
        job_id,
        detail: detail.into(),
        created_at_ms: now_ms(),
        resolved: false,
    }
}

fn nonterminal_job_ids(jobs: &BTreeMap<JobId, JobRecord>) -> Vec<JobId> {
    jobs.iter()
        .filter(|(_, record)| !record.state.is_terminal())
        .map(|(id, _)| id.clone())
        .collect()
}

fn is_candidate_spec(spec: &JobSpec) -> bool {
    spec.kind == crate::JobKind::WriteCandidate
        || matches!(spec.effect_scope, crate::EffectScope::CandidateOnly)
}

fn denied(
    state: &State,
    reason: AdmissionDeniedReason,
    detail: impl Into<String>,
) -> AdmissionDenied {
    AdmissionDenied {
        reason,
        state: state.status.state,
        detail: detail.into(),
    }
}

fn publish(inner: &Inner, state: &mut State) {
    state.status.sequence = state.status.sequence.saturating_add(1);
    inner.status_tx.send_replace(state.status.clone());
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
#[path = "in_memory_tests.rs"]
mod tests;
