use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use tokio::sync::watch;

use crate::{
    AdmissionDenied, AdmissionDeniedReason, BarrierKind, BarrierReceipt, BarrierStatus, BindingId,
    ExecutionDisposition, ExecutionReport, JobId, JobPermit, JobRegistryLimits, JobSealOutcome,
    JobSealStatus, JobSpec, JobState, JobTerminal, MutationIntent, MutationPermit,
    MutationPermitRecord, OperationId, PermitAbandonment, PermitIssuer, RecoveryId, RecoveryKind,
    RecoveryOutcome, RecoveryRecord, RecoveryStatus, SettlementOutcome, SettlementReceipt,
    SettlementStatus, WORKSPACE_CONTRACT_SCHEMA_VERSION, WorkspaceAuthority, WorkspaceBinding,
    WorkspaceCapabilities, WorkspaceController, WorkspaceId, WorkspaceState, WorkspaceStatus,
};

#[cfg(test)]
use crate::JobCompletion;

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
    jobs: crate::WorkspaceJobRegistry,
    recoveries: BTreeMap<RecoveryId, RecoveryRecord>,
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
        state.jobs = crate::WorkspaceJobRegistry::with_limits(
            state.binding.clone(),
            state.jobs.snapshot().limits,
        )
        .expect("validated job limits");
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
        let state = lock(&self.inner.state);
        state
            .jobs
            .status(&state.jobs.fence(), job_id)
            .ok()
            .map(|job| job.state)
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

    fn job_state(&self, job: &JobId) -> Option<JobState> {
        InMemoryWorkspaceController::job_state(self, job)
    }

    async fn begin(&self, intent: MutationIntent) -> Result<MutationPermit, AdmissionDenied> {
        let mut state = lock(&self.inner.state);
        if !state.status.state.admits_mutation() {
            let detail = state
                .status
                .admission_block_detail("workspace is not ready for mutation admission");
            return Err(denied(&state, AdmissionDeniedReason::NotReady, detail));
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
            .snapshot()
            .jobs
            .into_iter()
            .filter(|job| {
                !job.state.is_terminal()
                    && matches!(job.permit.spec.effect_scope, crate::EffectScope::LiveWriter)
                    && !is_local_live_process(&state.binding, &job.permit.spec)
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
            let detail = state
                .status
                .admission_block_detail("a live writer job blocks mutation admission");
            return Err(denied(&state, AdmissionDeniedReason::ActiveWriter, detail));
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
                state
                    .status
                    .admission_block_detail("workspace is not ready for job admission"),
            ));
        }
        if matches!(spec.effect_scope, crate::EffectScope::LiveWriter)
            && state.active_operation.is_some()
            && !belongs_to_active
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::ActiveWriter,
                "another live writer blocks job admission",
            ));
        }
        let fence = state.jobs.fence();
        let permit = state.jobs.register_running(&fence, spec).map_err(|error| {
            denied(
                &state,
                AdmissionDeniedReason::ActiveWriter,
                error.to_string(),
            )
        })?;
        state.status.active_jobs = nonterminal_job_ids(&state.jobs);
        publish(&self.inner, &mut state);
        Ok(permit)
    }

    async fn seal_job(&self, job: JobId, terminal: JobTerminal) -> JobSealOutcome {
        let mut state = lock(&self.inner.state);
        let outcome = state.jobs.seal(&state.jobs.fence(), &job, terminal);
        if outcome.status == JobSealStatus::Sealed
            && let Some(recovery_id) = &outcome.recovery_id
            && outcome.state == Some(JobState::RecoveryRequired)
        {
            let record = RecoveryRecord {
                schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
                recovery_id: recovery_id.clone(),
                kind: RecoveryKind::CrashedWriterJob,
                binding_id: state.binding.binding_id.clone(),
                epoch: state.binding.epoch,
                operation_id: None,
                job_id: Some(job),
                detail: outcome
                    .detail
                    .clone()
                    .unwrap_or_else(|| "job requires workspace recovery".into()),
                created_at_ms: now_ms(),
                resolved: false,
            };
            state.status.state = WorkspaceState::RecoveryRequired;
            state.status.recovery_id = Some(recovery_id.clone());
            state.status.detail = Some(record.detail.clone());
            state.recoveries.insert(recovery_id.clone(), record);
        }
        state.status.active_jobs = nonterminal_job_ids(&state.jobs);
        publish(&self.inner, &mut state);
        outcome
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
        if recovered_job.is_some() {
            let _ = state
                .jobs
                .reconcile_recovery(&state.jobs.fence(), &recovery, None);
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

fn is_local_live_process(binding: &WorkspaceBinding, spec: &JobSpec) -> bool {
    matches!(&binding.authority, WorkspaceAuthority::Local)
        && spec.kind == crate::JobKind::Process
        && spec.effect_scope == crate::EffectScope::LiveWriter
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

fn nonterminal_job_ids(jobs: &crate::WorkspaceJobRegistry) -> Vec<JobId> {
    jobs.snapshot()
        .jobs
        .into_iter()
        .filter(|record| !record.state.is_terminal())
        .map(|record| record.permit.job_id)
        .collect()
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
