use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use hi_workspace::{
    AdmissionDenied, AdmissionDeniedReason, ExecutionReport, JobId, JobSealOutcome, JobState,
    MutationPermitRecord, PermitAbandonment, PermitIssuer, RecoveryId, RecoveryKind,
    RecoveryOutcome, RecoveryRecord, RecoveryStatus, SettlementOutcome, SettlementStatus,
    WORKSPACE_CONTRACT_SCHEMA_VERSION, WorkspaceBinding, WorkspaceJobRegistry, WorkspaceState,
    WorkspaceStatus, WorkspaceVersion,
};
use tokio::sync::watch;

use super::{CausalTranscriptBatch, PipeFsSessionBridge, PipeFsWriterMode};
use crate::{PipeFsError, PipeFsWorkspace, WorkspacePhase};

pub(super) struct Inner {
    pub(super) workspace: PipeFsWorkspace,
    pub(super) session: Arc<dyn PipeFsSessionBridge>,
    pub(super) mode: PipeFsWriterMode,
    pub(super) issuer: PermitIssuer,
    pub(super) state: Mutex<State>,
    pub(super) jobs: WorkspaceJobRegistry,
    pub(super) status_tx: watch::Sender<WorkspaceStatus>,
}

pub(super) struct State {
    pub(super) binding: WorkspaceBinding,
    pub(super) status: WorkspaceStatus,
    pub(super) active: Option<MutationPermitRecord>,
    pub(super) recoveries: BTreeMap<RecoveryId, RecoveryEntry>,
}

pub(super) struct RecoveryEntry {
    pub(super) record: RecoveryRecord,
    pub(super) operation: Option<MutationPermitRecord>,
    pub(super) execution: Option<ExecutionReport>,
    pub(super) batch: Option<CausalTranscriptBatch>,
}

pub(super) struct BackendFailure {
    pub(super) error: anyhow::Error,
    pub(super) batch: Option<CausalTranscriptBatch>,
}

impl BackendFailure {
    pub(super) fn without_batch(error: anyhow::Error) -> Self {
        Self { error, batch: None }
    }

    pub(super) fn with_batch(error: anyhow::Error, batch: &CausalTranscriptBatch) -> Self {
        Self {
            error,
            batch: Some(batch.clone()),
        }
    }
}

pub(super) struct AbandonmentHandler {
    pub(super) inner: Weak<Inner>,
}

impl PermitAbandonment for AbandonmentHandler {
    fn mutation_abandoned(&self, permit: &MutationPermitRecord) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let mut state = lock(&inner.state);
        if state
            .active
            .as_ref()
            .is_none_or(|active| active.operation_id != permit.operation_id)
        {
            return;
        }
        require_recovery(
            &inner,
            &mut state,
            RecoveryKind::AbandonedMutation,
            Some(permit.clone()),
            None,
            None,
            "admitted PipeFS mutation was abandoned before settlement".into(),
        );
    }
}

pub(super) fn pipefs_version(
    status: &crate::PipeFsStatus,
    cursor: Option<u64>,
) -> WorkspaceVersion {
    WorkspaceVersion::PipeFs {
        lease_generation: status.lease_generation,
        head: status.last_committed_revision.map(|head| head.to_string()),
        manifest_digest: status.manifest_digest.clone(),
        transcript_cursor: cursor,
    }
}

pub(super) fn classify_admission_failure(error: &anyhow::Error) -> WorkspaceState {
    if error_chain::<PipeFsError>(error)
        .is_some_and(|error| matches!(error, PipeFsError::LeaseLost(_)))
    {
        WorkspaceState::LeaseLost
    } else {
        WorkspaceState::LeaseUncertain
    }
}

pub(super) fn classify_failure(
    error: &anyhow::Error,
    remote: &crate::PipeFsStatus,
) -> (WorkspaceState, SettlementStatus, RecoveryKind) {
    if let Some(error) = error_chain::<PipeFsError>(error) {
        return match error {
            PipeFsError::LeaseLost(_) => (
                WorkspaceState::LeaseLost,
                SettlementStatus::LeaseLost,
                RecoveryKind::LeaseLost,
            ),
            PipeFsError::Conflict(_) => (
                WorkspaceState::Conflict,
                SettlementStatus::Conflict,
                RecoveryKind::Conflict,
            ),
            _ => (
                WorkspaceState::PendingRemote,
                SettlementStatus::Pending,
                RecoveryKind::UnsettledMutation,
            ),
        };
    }
    if remote.transcript_pending || format!("{error:#}").contains("transcript") {
        (
            WorkspaceState::TranscriptPending,
            SettlementStatus::TranscriptPending,
            RecoveryKind::TranscriptPending,
        )
    } else if remote.phase == WorkspacePhase::LeaseLost {
        (
            WorkspaceState::LeaseLost,
            SettlementStatus::LeaseLost,
            RecoveryKind::LeaseLost,
        )
    } else {
        (
            WorkspaceState::PendingRemote,
            SettlementStatus::Pending,
            RecoveryKind::UnsettledMutation,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn require_recovery(
    inner: &Inner,
    state: &mut State,
    kind: RecoveryKind,
    operation: Option<MutationPermitRecord>,
    execution: Option<ExecutionReport>,
    batch: Option<CausalTranscriptBatch>,
    detail: String,
) -> RecoveryId {
    let recovery_id = RecoveryId::new(uuid::Uuid::new_v4().to_string());
    let operation_id = operation.as_ref().map(|op| op.operation_id.clone());
    let record = RecoveryRecord {
        schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
        recovery_id: recovery_id.clone(),
        kind,
        binding_id: state.binding.binding_id.clone(),
        epoch: state.binding.epoch,
        operation_id,
        job_id: None,
        detail: detail.clone(),
        created_at_ms: now_ms(),
        resolved: false,
    };
    state.recoveries.insert(
        recovery_id.clone(),
        RecoveryEntry {
            record,
            operation,
            execution,
            batch,
        },
    );
    state.active = None;
    state.status.active_operation = None;
    state.status.state = WorkspaceState::RecoveryRequired;
    state.status.recovery_id = Some(recovery_id.clone());
    state.status.detail = Some(detail);
    publish(inner, state);
    recovery_id
}

pub(super) fn sync_jobs(inner: &Inner) {
    let jobs = inner.jobs.snapshot();
    let mut state = lock(&inner.state);
    state.status.active_jobs = jobs
        .jobs
        .into_iter()
        .filter(|job| !job.state.is_terminal())
        .map(|job| job.permit.job_id)
        .collect();
    publish(inner, &mut state);
}

pub(super) fn register_job_recovery(inner: &Inner, job_id: &JobId, outcome: &JobSealOutcome) {
    if outcome.state != Some(JobState::RecoveryRequired) {
        return;
    }
    let Some(recovery_id) = outcome.recovery_id.clone() else {
        return;
    };
    let mut state = lock(&inner.state);
    let detail = outcome
        .detail
        .clone()
        .unwrap_or_else(|| "PipeFS job requires explicit workspace recovery".into());
    let binding_id = state.binding.binding_id.clone();
    let epoch = state.binding.epoch;
    state
        .recoveries
        .entry(recovery_id.clone())
        .or_insert_with(|| RecoveryEntry {
            record: RecoveryRecord {
                schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
                recovery_id: recovery_id.clone(),
                kind: RecoveryKind::CrashedWriterJob,
                binding_id,
                epoch,
                operation_id: None,
                job_id: Some(job_id.clone()),
                detail: detail.clone(),
                created_at_ms: now_ms(),
                resolved: false,
            },
            operation: None,
            execution: None,
            batch: None,
        });
    state.status.state = WorkspaceState::RecoveryRequired;
    state.status.recovery_id = Some(recovery_id);
    state.status.detail = Some(detail);
    publish(inner, &mut state);
}

pub(super) fn reconcile_job_recovery(
    inner: &Inner,
    recovery_id: &RecoveryId,
) -> Option<RecoveryOutcome> {
    let job_id = {
        let state = lock(&inner.state);
        state.recoveries.get(recovery_id)?.record.job_id.clone()?
    };
    let result = inner.jobs.reconcile_recovery(
        &inner.jobs.fence(),
        recovery_id,
        Some("PipeFS job recovery was explicitly acknowledged".into()),
    );
    let mut state = lock(&inner.state);
    match result {
        Ok(_) => {
            if let Some(entry) = state.recoveries.get_mut(recovery_id) {
                entry.record.resolved = true;
            }
            state.status.active_jobs = inner
                .jobs
                .snapshot()
                .jobs
                .into_iter()
                .filter(|job| !job.state.is_terminal())
                .map(|job| job.permit.job_id)
                .collect();
            promote_next_recovery(inner, &mut state, recovery_id);
            publish(inner, &mut state);
            Some(recovery_outcome(
                recovery_id.clone(),
                RecoveryStatus::Recovered,
                &state.binding,
                None,
            ))
        }
        Err(error) => Some(recovery_outcome(
            recovery_id.clone(),
            RecoveryStatus::Rejected,
            &state.binding,
            Some(format!("job {job_id} recovery failed: {error}")),
        )),
    }
}

pub(super) fn promote_next_recovery(inner: &Inner, state: &mut State, resolved: &RecoveryId) {
    if state.status.recovery_id.as_ref() != Some(resolved) {
        return;
    }
    if let Some(next) = state
        .recoveries
        .values()
        .find(|entry| !entry.record.resolved)
    {
        state.status.state = recovery_state(next);
        state.status.recovery_id = Some(next.record.recovery_id.clone());
        state.status.detail = Some(next.record.detail.clone());
        if let Err(error) = inner.workspace.retain_for_journal_recovery() {
            state.status.detail = Some(format!(
                "{}; recovery cache marker could not be persisted: {error:#}",
                next.record.detail
            ));
        }
    } else {
        state.status.state = if state.active.is_some() {
            WorkspaceState::Mutating
        } else {
            WorkspaceState::Ready
        };
        state.status.recovery_id = None;
        state.status.detail = None;
    }
}

fn recovery_state(entry: &RecoveryEntry) -> WorkspaceState {
    match entry.record.kind {
        RecoveryKind::LeaseLost => WorkspaceState::LeaseLost,
        RecoveryKind::Conflict => WorkspaceState::Conflict,
        RecoveryKind::TranscriptPending => WorkspaceState::TranscriptPending,
        RecoveryKind::CleanupPending => WorkspaceState::CleanupPending,
        RecoveryKind::IncompatibleState => WorkspaceState::Incompatible,
        RecoveryKind::UnsettledMutation
            if entry.operation.is_some() && entry.execution.is_some() =>
        {
            WorkspaceState::PendingRemote
        }
        _ => WorkspaceState::RecoveryRequired,
    }
}

pub(super) fn denied(
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

pub(super) fn incompatible(
    operation: MutationPermitRecord,
    detail: impl Into<String>,
) -> SettlementOutcome {
    SettlementOutcome {
        status: SettlementStatus::Incompatible,
        operation_id: operation.operation_id,
        receipt: None,
        recovery_id: None,
        detail: Some(detail.into()),
    }
}

pub(super) fn recovery_outcome(
    recovery_id: RecoveryId,
    status: RecoveryStatus,
    binding: &WorkspaceBinding,
    detail: Option<String>,
) -> RecoveryOutcome {
    RecoveryOutcome {
        recovery_id,
        status,
        binding: binding.clone(),
        detail,
    }
}

pub(super) fn error_chain<T: std::error::Error + Send + Sync + 'static>(
    error: &anyhow::Error,
) -> Option<&T> {
    error.chain().find_map(|cause| cause.downcast_ref::<T>())
}

pub(super) fn publish(inner: &Inner, state: &mut State) {
    state.status.sequence = state.status.sequence.saturating_add(1);
    inner.status_tx.send_replace(state.status.clone());
}

pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(super) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
