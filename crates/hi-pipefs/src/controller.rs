use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use hi_workspace::{
    AdmissionDenied, AdmissionDeniedReason, BarrierKind, BarrierReceipt, BarrierStatus,
    ExecutionReport, JobId, JobPermit, JobRegistryLimits, JobSpec, JobState, JobTerminal,
    MutationIntent, MutationPermit, MutationPermitRecord, OperationId, PermitIssuer, RecoveryId,
    RecoveryKind, RecoveryOutcome, RecoveryRecord, RecoveryStatus, SettlementOutcome,
    SettlementReceipt, SettlementStatus, WORKSPACE_CONTRACT_SCHEMA_VERSION, WorkspaceBinding,
    WorkspaceCapabilities, WorkspaceController, WorkspaceId, WorkspaceJobRegistry, WorkspaceState,
    WorkspaceStatus, WorkspaceVersion,
};
use tokio::sync::watch;

use crate::{
    CausalOperationReceipt, CausalTranscriptRecord, PipeFsError, PipeFsLease, PipeFsWorkspace,
    WorkspacePhase,
};

#[path = "controller_state.rs"]
mod state;
use state::*;
#[path = "controller_init.rs"]
mod init;
#[path = "controller_lease_monitor.rs"]
mod lease_monitor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipeFsWriterMode {
    ReadOnly,
    Compatibility,
    Causal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PipeFsLeaseStatus {
    #[default]
    Valid,
    Uncertain,
    Lost,
}

#[derive(Clone, Debug)]
pub struct PipeFsControllerConfig {
    pub workspace_id: WorkspaceId,
    pub session_id: String,
    pub writer_protocol: u16,
    pub causal_commit_available: bool,
    pub writes_available: bool,
    pub workspace_root: PathBuf,
    pub state_root: PathBuf,
    pub epoch: u64,
    pub allow_protocol_one_writes: bool,
}

impl PipeFsControllerConfig {
    pub fn writer_mode(&self) -> PipeFsWriterMode {
        if !self.writes_available {
            PipeFsWriterMode::ReadOnly
        } else if self.writer_protocol >= crate::CAUSAL_WRITER_PROTOCOL
            && self.causal_commit_available
        {
            PipeFsWriterMode::Causal
        } else if self.allow_protocol_one_writes {
            PipeFsWriterMode::Compatibility
        } else {
            PipeFsWriterMode::ReadOnly
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CausalTranscriptBatch {
    pub records: Vec<CausalTranscriptRecord>,
}

#[async_trait]
pub trait PipeFsSessionBridge: Send + Sync {
    fn subscribe_lease_status(&self) -> watch::Receiver<PipeFsLeaseStatus>;
    async fn refresh_lease(&self) -> Result<PipeFsLease>;
    async fn prepare_causal_mutation(&self) -> Result<()>;
    async fn causal_transcript_batch(&self) -> Result<CausalTranscriptBatch>;
    async fn acknowledge_causal_transcript(
        &self,
        batch: &CausalTranscriptBatch,
        cursor: u64,
    ) -> Result<()>;
    async fn flush_compatibility_transcript(
        &self,
        operation: &CausalOperationReceipt,
    ) -> Result<Option<u64>>;
}

#[derive(Clone)]
pub struct PipeFsWorkspaceController {
    inner: Arc<Inner>,
}

impl PipeFsWorkspaceController {
    /// Install a journal-discovered restart fence without granting it a
    /// remote settlement path. Only recoveries reconstructed from PipeFS's
    /// own pending operation evidence may publish remotely.
    pub fn require_restart_recovery(&self, record: RecoveryRecord) -> Result<()> {
        if record.resolved || (record.operation_id.is_none() && record.job_id.is_none()) {
            return Err(anyhow!(
                "restart recovery must be unresolved and identify an operation or job"
            ));
        }
        self.inner.workspace.retain_for_journal_recovery()?;
        let mut state = lock(&self.inner.state);
        if let Some(existing) = state.recoveries.get(&record.recovery_id) {
            if existing.record.operation_id == record.operation_id
                && existing.record.job_id == record.job_id
                && existing.record.binding_id == record.binding_id
                && existing.record.epoch == record.epoch
            {
                return Ok(());
            }
            return Err(anyhow!(
                "restart recovery identity collides with different persisted evidence"
            ));
        }
        if let Some(job_id) = &record.job_id
            && !state.status.active_jobs.contains(job_id)
        {
            state.status.active_jobs.push(job_id.clone());
            state.status.active_jobs.sort();
        }
        let recovery_id = record.recovery_id.clone();
        let detail = record.detail.clone();
        state.recoveries.insert(
            recovery_id.clone(),
            RecoveryEntry {
                record,
                operation: None,
                execution: None,
                batch: None,
            },
        );
        if state.status.recovery_id.is_none() {
            state.status.state = WorkspaceState::RecoveryRequired;
            state.status.recovery_id = Some(recovery_id);
            state.status.detail = Some(detail);
        }
        publish(&self.inner, &mut state);
        Ok(())
    }

    fn start_lease_monitor(&self) {
        lease_monitor::start(&self.inner);
    }

    async fn refresh_version(&self) -> Result<WorkspaceVersion> {
        let lease = self.inner.session.refresh_lease().await?;
        self.inner.workspace.update_lease(lease).await?;
        let status = self.inner.workspace.status().await;
        Ok(pipefs_version(&status, status.transcript_cursor))
    }

    async fn settle_backend(
        &self,
        operation: &MutationPermitRecord,
        execution: &ExecutionReport,
        batch: Option<CausalTranscriptBatch>,
    ) -> std::result::Result<(WorkspaceVersion, Option<CausalTranscriptBatch>), BackendFailure>
    {
        let _ = self
            .refresh_version()
            .await
            .map_err(BackendFailure::without_batch)?;
        let operation_receipt = CausalOperationReceipt {
            operation_id: operation.operation_id.to_string(),
            idempotency_key: operation.idempotency_key.to_string(),
            binding_id: operation.binding_id.to_string(),
            binding_epoch: operation.epoch,
            replay_class: operation.intent.replay_class.clone(),
            execution: execution.clone(),
        };
        match self.inner.mode {
            PipeFsWriterMode::Causal => {
                let batch = match batch {
                    Some(batch) => batch,
                    None => self
                        .inner
                        .session
                        .causal_transcript_batch()
                        .await
                        .map_err(BackendFailure::without_batch)?,
                };
                operation_receipt
                    .validate_binding(operation.binding_id.as_str(), operation.epoch)
                    .map_err(|error| BackendFailure::with_batch(error.into(), &batch))?;
                let receipt = self
                    .inner
                    .workspace
                    .causal_checkpoint(operation_receipt, batch.records.clone())
                    .await
                    .map_err(|error| BackendFailure::with_batch(error, &batch))?;
                hi_workspace::hit_harness_failpoint(
                    hi_workspace::HarnessFailpoint::TranscriptBeforeFlush,
                )
                .map_err(|error| BackendFailure::with_batch(error.into(), &batch))?;
                self.inner
                    .session
                    .acknowledge_causal_transcript(&batch, receipt.transcript_cursor)
                    .await
                    .map_err(|error| BackendFailure::with_batch(error, &batch))?;
                self.inner
                    .workspace
                    .finish_causal_checkpoint(
                        operation.operation_id.as_str(),
                        receipt.transcript_cursor,
                    )
                    .await
                    .map_err(|error| BackendFailure::with_batch(error, &batch))?;
                let status = self.inner.workspace.status().await;
                Ok((
                    pipefs_version(&status, Some(receipt.transcript_cursor)),
                    Some(batch),
                ))
            }
            PipeFsWriterMode::Compatibility => {
                self.inner
                    .workspace
                    .checkpoint_for_compatibility_transcript(operation_receipt.clone())
                    .await
                    .map_err(BackendFailure::without_batch)?;
                hi_workspace::hit_harness_failpoint(
                    hi_workspace::HarnessFailpoint::TranscriptBeforeFlush,
                )
                .map_err(|error| BackendFailure::without_batch(error.into()))?;
                let cursor = self
                    .inner
                    .session
                    .flush_compatibility_transcript(&operation_receipt)
                    .await
                    .map_err(BackendFailure::without_batch)?;
                let cursor = cursor.ok_or_else(|| {
                    BackendFailure::without_batch(anyhow!(
                        "compatibility transcript flush returned no acknowledgement cursor"
                    ))
                })?;
                self.inner
                    .workspace
                    .finish_compatibility_checkpoint(operation.operation_id.as_str(), cursor)
                    .await
                    .map_err(BackendFailure::without_batch)?;
                let status = self.inner.workspace.status().await;
                Ok((pipefs_version(&status, Some(cursor)), None))
            }
            PipeFsWriterMode::ReadOnly => Err(BackendFailure::without_batch(anyhow!(
                "PipeFS writer protocol 2 with causal_commit_v1 is required for mutation"
            ))),
        }
    }

    async fn failed_settlement(
        &self,
        operation: MutationPermitRecord,
        execution: ExecutionReport,
        failure: BackendFailure,
    ) -> SettlementOutcome {
        let BackendFailure { error, batch } = failure;
        let remote = self.inner.workspace.status().await;
        let (state_kind, settlement, recovery_kind) = classify_failure(&error, &remote);
        let detail = format!("{error:#}");
        let mut state = lock(&self.inner.state);
        let recovery = require_recovery(
            &self.inner,
            &mut state,
            recovery_kind,
            Some(operation.clone()),
            Some(execution),
            batch,
            detail.clone(),
        );
        state.status.state = state_kind;
        state.status.recovery_id = Some(recovery.clone());
        state.status.detail = Some(detail.clone());
        publish(&self.inner, &mut state);
        SettlementOutcome {
            status: settlement,
            operation_id: operation.operation_id,
            receipt: None,
            recovery_id: Some(recovery),
            detail: Some(detail),
        }
    }
}

#[async_trait]
impl WorkspaceController for PipeFsWorkspaceController {
    fn binding(&self) -> WorkspaceBinding {
        lock(&self.inner.state).binding.clone()
    }

    fn capabilities(&self) -> WorkspaceCapabilities {
        let causal = self.inner.mode == PipeFsWriterMode::Causal;
        let mut capabilities = WorkspaceCapabilities::pipefs(causal);
        capabilities.candidate_apply = self.inner.mode != PipeFsWriterMode::ReadOnly;
        // Live PipeFS writers require a process-group pause/checkpoint/resume
        // bridge. Candidate writers remain isolated and parent-published.
        capabilities.background_writers = false;
        capabilities
    }

    fn status(&self) -> WorkspaceStatus {
        lock(&self.inner.state).status.clone()
    }

    fn subscribe(&self) -> watch::Receiver<WorkspaceStatus> {
        self.inner.status_tx.subscribe()
    }

    async fn begin(&self, intent: MutationIntent) -> Result<MutationPermit, AdmissionDenied> {
        if self.inner.mode != PipeFsWriterMode::Causal
            && matches!(
                intent.replay_class,
                hi_workspace::ReplayClass::NonReplayableExternal
            )
        {
            return Err(denied(
                &lock(&self.inner.state),
                AdmissionDeniedReason::CapabilityUnavailable,
                "non-replayable PipeFS effects require writer protocol 2 intent acknowledgement",
            ));
        }
        if self.inner.mode == PipeFsWriterMode::ReadOnly {
            return Err(denied(
                &lock(&self.inner.state),
                AdmissionDeniedReason::CapabilityUnavailable,
                "PipeFS is read-only without writer protocol 2 and causal_commit_v1",
            ));
        }
        {
            let state = lock(&self.inner.state);
            if state.active.is_some()
                || state.status.recovery_id.is_some()
                || !matches!(
                    state.status.state,
                    WorkspaceState::Ready
                        | WorkspaceState::LeaseUncertain
                        | WorkspaceState::TranscriptPending
                )
            {
                return Err(denied(
                    &state,
                    AdmissionDeniedReason::NotReady,
                    "PipeFS has unsettled work",
                ));
            }
        }
        if self.inner.jobs.snapshot().jobs.iter().any(|job| {
            !job.state.is_terminal()
                && matches!(
                    job.permit.spec.effect_scope,
                    hi_workspace::EffectScope::LiveWriter
                )
        }) {
            return Err(denied(
                &lock(&self.inner.state),
                AdmissionDeniedReason::ActiveWriter,
                "a live PipeFS writer must stop before foreground mutation admission",
            ));
        }
        let version = match self.refresh_version().await {
            Ok(version) => version,
            Err(error) => {
                let remote = self.inner.workspace.status().await;
                let mut state = lock(&self.inner.state);
                state.status.state = if remote.phase == WorkspacePhase::LeaseLost {
                    WorkspaceState::LeaseLost
                } else {
                    classify_admission_failure(&error)
                };
                state.status.detail = Some(format!("{error:#}"));
                publish(&self.inner, &mut state);
                return Err(denied(
                    &state,
                    AdmissionDeniedReason::NotReady,
                    format!("could not prove the PipeFS lease and head: {error:#}"),
                ));
            }
        };
        if self.inner.mode == PipeFsWriterMode::Causal
            && let Err(error) = self.inner.session.prepare_causal_mutation().await
        {
            let mut state = lock(&self.inner.state);
            state.status.state = WorkspaceState::TranscriptPending;
            state.status.detail = Some(format!(
                "could not drain the stable transcript prefix before PipeFS mutation admission: {error:#}"
            ));
            publish(&self.inner, &mut state);
            return Err(denied(
                &state,
                AdmissionDeniedReason::NotReady,
                "PipeFS transcript prefix is not durably acknowledged",
            ));
        }
        let record = {
            let mut state = lock(&self.inner.state);
            if matches!(
                state.status.state,
                WorkspaceState::LeaseUncertain | WorkspaceState::TranscriptPending
            ) && state.status.recovery_id.is_none()
            {
                state.status.state = WorkspaceState::Ready;
                state.status.detail = None;
            }
            if !state.status.state.admits_mutation() || state.active.is_some() {
                return Err(denied(
                    &state,
                    AdmissionDeniedReason::NotReady,
                    "PipeFS has an unsettled operation",
                ));
            }
            state.binding.version = version;
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
            state.active = Some(record.clone());
            state.status.state = WorkspaceState::Mutating;
            state.status.active_operation = Some(record.operation_id.clone());
            state.status.detail = None;
            publish(&self.inner, &mut state);
            record
        };

        let admission = async {
            if matches!(
                record.intent.replay_class,
                hi_workspace::ReplayClass::NonReplayableExternal
            ) {
                self.inner
                    .workspace
                    .acknowledge_operation_intent(&record)
                    .await?;
            }
            self.inner
                .workspace
                .mutation_started(record.intent.dirty_paths.as_ref().map(|paths| {
                    paths
                        .iter()
                        .map(|path| path.to_string_lossy().into_owned())
                        .collect()
                }))
                .await
        }
        .await;
        if let Err(error) = admission {
            let remote = self.inner.workspace.status().await;
            let mut state = lock(&self.inner.state);
            let (workspace_state, _, recovery_kind) = classify_failure(&error, &remote);
            let detail = format!("PipeFS mutation admission failed: {error:#}");
            let recovery = require_recovery(
                &self.inner,
                &mut state,
                recovery_kind,
                Some(record.clone()),
                Some(ExecutionReport {
                    disposition: hi_workspace::ExecutionDisposition::Failed,
                    workspace_may_have_changed: false,
                    external_effect_may_have_occurred: false,
                    content_digest: None,
                    changed_paths: Vec::new(),
                    artifacts: Vec::new(),
                    detail: Some(detail.clone()),
                }),
                None,
                detail.clone(),
            );
            state.status.state = workspace_state;
            state.status.recovery_id = Some(recovery);
            publish(&self.inner, &mut state);
            return Err(denied(&state, AdmissionDeniedReason::NotReady, detail));
        }
        Ok(self.inner.issuer.issue_mutation(record))
    }

    async fn settle(
        &self,
        mut permit: MutationPermit,
        execution: ExecutionReport,
    ) -> SettlementOutcome {
        let fallback = permit.record().operation_id.clone();
        let operation = match self.inner.issuer.claim_mutation(&mut permit) {
            Ok(operation) => operation,
            Err(error) => {
                return SettlementOutcome {
                    status: SettlementStatus::Incompatible,
                    operation_id: fallback,
                    receipt: None,
                    recovery_id: None,
                    detail: Some(error.to_string()),
                };
            }
        };
        {
            let mut state = lock(&self.inner.state);
            let current = state.active.as_ref().is_some_and(|active| {
                active.operation_id == operation.operation_id
                    && active.binding_id == operation.binding_id
                    && active.epoch == operation.epoch
            });
            if !current {
                return incompatible(operation, "mutation permit is stale");
            }
            state.status.state = WorkspaceState::Settling;
            publish(&self.inner, &mut state);
        }
        match self.settle_backend(&operation, &execution, None).await {
            Ok((version, batch)) => {
                let mut state = lock(&self.inner.state);
                state.binding.version = version.clone();
                let receipt = SettlementReceipt {
                    receipt_id: uuid::Uuid::new_v4().to_string(),
                    operation_id: operation.operation_id.clone(),
                    binding_id: state.binding.binding_id.clone(),
                    epoch: state.binding.epoch,
                    transcript_cursor: match &version {
                        WorkspaceVersion::PipeFs {
                            transcript_cursor, ..
                        } => *transcript_cursor,
                        _ => None,
                    },
                    version,
                };
                if execution.disposition == hi_workspace::ExecutionDisposition::Indeterminate {
                    let detail = execution.detail.clone().unwrap_or_else(|| {
                        "operation publication is durable, but execution effects remain indeterminate"
                            .into()
                    });
                    let recovery = require_recovery(
                        &self.inner,
                        &mut state,
                        RecoveryKind::UnsettledMutation,
                        Some(operation.clone()),
                        Some(execution),
                        batch,
                        detail.clone(),
                    );
                    return SettlementOutcome {
                        status: SettlementStatus::Indeterminate,
                        operation_id: operation.operation_id,
                        receipt: Some(receipt),
                        recovery_id: Some(recovery),
                        detail: Some(detail),
                    };
                }
                state.active = None;
                state.status.active_operation = None;
                state.status.state = WorkspaceState::Ready;
                state.status.recovery_id = None;
                state.status.detail = None;
                publish(&self.inner, &mut state);
                SettlementOutcome {
                    status: if execution.workspace_may_have_changed
                        || execution.external_effect_may_have_occurred
                    {
                        SettlementStatus::Durable
                    } else {
                        SettlementStatus::NoChange
                    },
                    operation_id: operation.operation_id.clone(),
                    receipt: Some(receipt),
                    recovery_id: None,
                    detail: execution.detail,
                }
            }
            Err(failure) => self.failed_settlement(operation, execution, failure).await,
        }
    }

    async fn register_job(&self, spec: JobSpec) -> Result<JobPermit, AdmissionDenied> {
        let state = lock(&self.inner.state);
        if spec.effect_scope == hi_workspace::EffectScope::LiveWriter {
            return Err(denied(
                &state,
                AdmissionDeniedReason::CapabilityUnavailable,
                "live PipeFS writer jobs are disabled until process pause/checkpoint/resume is available",
            ));
        }
        if self.inner.mode == PipeFsWriterMode::ReadOnly
            && spec.effect_scope != hi_workspace::EffectScope::ReadOnly
        {
            return Err(denied(
                &state,
                AdmissionDeniedReason::CapabilityUnavailable,
                "write jobs are unavailable in read-only PipeFS mode",
            ));
        }
        let belongs_to_active = state.active.as_ref().is_some_and(|active| {
            spec.parent_operation.as_ref() == Some(&active.operation_id)
                && active.binding_id == state.binding.binding_id
                && active.epoch == state.binding.epoch
        });
        if !state.status.state.admits_mutation() && !belongs_to_active {
            return Err(denied(
                &state,
                AdmissionDeniedReason::NotReady,
                "PipeFS has unsettled work unrelated to this job",
            ));
        }
        drop(state);
        let fence = self.inner.jobs.fence();
        let permit = self.inner.jobs.register(&fence, spec).map_err(|error| {
            denied(
                &lock(&self.inner.state),
                AdmissionDeniedReason::ActiveWriter,
                error.to_string(),
            )
        })?;
        for (expected, next) in [
            (JobState::Queued, JobState::Starting),
            (JobState::Starting, JobState::Running),
        ] {
            if let Err(error) =
                self.inner
                    .jobs
                    .transition(&fence, &permit.job_id, expected, next, None, Vec::new())
            {
                let _ = self.inner.jobs.seal(
                    &fence,
                    &permit.job_id,
                    JobTerminal {
                        completion: hi_workspace::JobCompletion::Failed,
                        detail: Some(format!("job admission failed before execution: {error}")),
                        artifacts: Vec::new(),
                    },
                );
                sync_jobs(&self.inner);
                return Err(denied(
                    &lock(&self.inner.state),
                    AdmissionDeniedReason::ActiveWriter,
                    error.to_string(),
                ));
            }
        }
        sync_jobs(&self.inner);
        Ok(permit)
    }

    async fn seal_job(&self, job: JobId, terminal: JobTerminal) -> hi_workspace::JobSealOutcome {
        let outcome = self
            .inner
            .jobs
            .seal(&self.inner.jobs.fence(), &job, terminal);
        register_job_recovery(&self.inner, &job, &outcome);
        sync_jobs(&self.inner);
        outcome
    }

    async fn barrier(&self, reason: BarrierKind, deadline: Instant) -> BarrierReceipt {
        let mut receipt = self
            .inner
            .jobs
            .barrier(&self.inner.jobs.fence(), reason, deadline)
            .unwrap_or_else(|error| BarrierReceipt {
                kind: reason,
                status: BarrierStatus::RecoveryRequired,
                binding_id: self.binding().binding_id,
                epoch: self.binding().epoch,
                active_operation: None,
                pending_jobs: Vec::new(),
                recovery_id: None,
                detail: Some(error.to_string()),
            });
        let state = lock(&self.inner.state);
        receipt.active_operation = state.active.as_ref().map(|op| op.operation_id.clone());
        receipt.recovery_id = state.status.recovery_id.clone().or(receipt.recovery_id);
        if receipt.active_operation.is_some() && receipt.status == BarrierStatus::Passed {
            receipt.status = if Instant::now() >= deadline {
                BarrierStatus::TimedOut
            } else {
                BarrierStatus::Blocked
            };
        }
        if state.status.recovery_id.is_some() {
            receipt.status = BarrierStatus::RecoveryRequired;
        }
        receipt
    }

    async fn reconcile(&self, recovery: RecoveryId) -> RecoveryOutcome {
        if let Some(outcome) = reconcile_job_recovery(&self.inner, &recovery) {
            return outcome;
        }
        let (operation, execution, batch) = {
            let state = lock(&self.inner.state);
            let Some(entry) = state.recoveries.get(&recovery) else {
                return recovery_outcome(recovery, RecoveryStatus::NotFound, &state.binding, None);
            };
            if entry.record.resolved {
                return recovery_outcome(
                    recovery,
                    RecoveryStatus::Recovered,
                    &state.binding,
                    Some("recovery was already resolved".into()),
                );
            }
            if entry.record.kind == RecoveryKind::IncompatibleState {
                return recovery_outcome(
                    recovery,
                    RecoveryStatus::Rejected,
                    &state.binding,
                    Some("incompatible PipeFS evidence cannot be replayed remotely".into()),
                );
            }
            (
                entry.operation.clone(),
                entry.execution.clone(),
                entry.batch.clone(),
            )
        };
        let (Some(operation), Some(execution)) = (operation, execution) else {
            return recovery_outcome(
                recovery,
                RecoveryStatus::Rejected,
                &self.binding(),
                Some("abandoned operations require explicit external proof".into()),
            );
        };
        match self.settle_backend(&operation, &execution, batch).await {
            Ok((version, _)) => {
                let mut state = lock(&self.inner.state);
                state.binding.version = version;
                if let Some(entry) = state.recoveries.get_mut(&recovery) {
                    entry.record.resolved = true;
                }
                state.active = None;
                state.status.active_operation = None;
                promote_next_recovery(&self.inner, &mut state, &recovery);
                publish(&self.inner, &mut state);
                recovery_outcome(recovery, RecoveryStatus::Recovered, &state.binding, None)
            }
            Err(failure) => {
                let BackendFailure { error, batch } = failure;
                if let Some(batch) = batch {
                    let mut state = lock(&self.inner.state);
                    if let Some(entry) = state.recoveries.get_mut(&recovery) {
                        entry.batch = Some(batch);
                    }
                }
                recovery_outcome(
                    recovery,
                    if error_chain::<PipeFsError>(&error)
                        .is_some_and(|error| matches!(error, PipeFsError::Conflict(_)))
                    {
                        RecoveryStatus::Conflict
                    } else {
                        RecoveryStatus::Pending
                    },
                    &self.binding(),
                    Some(format!("{error:#}")),
                )
            }
        }
    }
}

#[cfg(test)]
#[path = "controller_tests.rs"]
mod tests;
