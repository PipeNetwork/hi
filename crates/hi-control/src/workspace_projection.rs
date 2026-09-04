use std::sync::{Arc, Mutex, MutexGuard};

use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, RunEvent,
    SemanticActivity,
};
use hi_workspace::{
    ExecutionReport, JobPermit, JobSealOutcome, JobState, JobTerminal, MutationPermitRecord,
    RecoveryId, RecoveryOutcome, RecoveryStatus, ReplayClass, SettlementOutcome, SettlementStatus,
    WorkspaceBinding, WorkspaceCapabilities, WorkspaceJobSnapshot, WorkspaceState, WorkspaceStatus,
};

use crate::{
    ControlEffectScope, ControlError, ControlJobKind, ControlJobRecord, ControlJobState,
    ControlStore, OperationReplayClass, ProjectionEventReceipt, ProjectionTransition, Result,
    WorkspaceAuthority, WorkspaceBindingRecord, WorkspaceOperationRecord, WorkspaceOperationStatus,
    WorkspaceProjectionState, WorkspaceRecoveryRecord, WorkspaceRecoveryStatus,
};

pub trait WorkspaceProjectionStore: Send + Sync {
    fn commit(
        &self,
        transition: ProjectionTransition,
        event: RunEvent,
    ) -> Result<ProjectionEventReceipt>;
    fn binding(&self, id: &str) -> Result<Option<WorkspaceBindingRecord>>;
    fn operation(&self, id: &str) -> Result<Option<WorkspaceOperationRecord>>;
    fn operations_for_binding(&self, binding_id: &str) -> Result<Vec<WorkspaceOperationRecord>>;
    fn job(&self, id: &str) -> Result<Option<ControlJobRecord>>;
    fn recovery(&self, id: &str) -> Result<Option<WorkspaceRecoveryRecord>>;
    fn recoveries_for_binding(&self, _binding_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        Ok(Vec::new())
    }
    fn recoveries_for_operation(&self, operation_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>>;
    fn recoveries_for_job(&self, job_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>>;
    fn jobs_for_binding(&self, binding_id: &str) -> Result<Vec<ControlJobRecord>>;
}

impl WorkspaceProjectionStore for ControlStore {
    fn commit(
        &self,
        transition: ProjectionTransition,
        event: RunEvent,
    ) -> Result<ProjectionEventReceipt> {
        self.commit_projection_event(transition, event)
    }

    fn binding(&self, id: &str) -> Result<Option<WorkspaceBindingRecord>> {
        self.get_workspace_binding(id)
    }

    fn operation(&self, id: &str) -> Result<Option<WorkspaceOperationRecord>> {
        self.get_workspace_operation(id)
    }

    fn operations_for_binding(&self, binding_id: &str) -> Result<Vec<WorkspaceOperationRecord>> {
        self.operations_for_binding(binding_id)
    }

    fn job(&self, id: &str) -> Result<Option<ControlJobRecord>> {
        self.get_job(id)
    }

    fn recovery(&self, id: &str) -> Result<Option<WorkspaceRecoveryRecord>> {
        self.get_workspace_recovery(id)
    }

    fn recoveries_for_binding(&self, binding_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.recoveries_for_binding(binding_id)
    }

    fn recoveries_for_operation(&self, operation_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.recoveries_for_operation(operation_id)
    }

    fn recoveries_for_job(&self, job_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.recoveries_for_job(job_id)
    }

    fn jobs_for_binding(&self, binding_id: &str) -> Result<Vec<ControlJobRecord>> {
        self.jobs_for_binding(binding_id)
    }
}

#[derive(Clone)]
pub struct WorkspaceProjectionJournal {
    pub(crate) store: Arc<dyn WorkspaceProjectionStore>,
    pub(crate) gate: Arc<Mutex<()>>,
}

struct OperationProjectionUpdate {
    status: WorkspaceOperationStatus,
    execution_ref: Option<String>,
    settlement_ref: Option<String>,
    result_version: Option<String>,
    error: Option<String>,
}

impl WorkspaceProjectionJournal {
    pub fn new(store: Arc<dyn WorkspaceProjectionStore>) -> Self {
        Self {
            store,
            gate: Arc::new(Mutex::new(())),
        }
    }

    pub fn from_control_store(store: ControlStore) -> Self {
        Self::new(Arc::new(store))
    }

    pub fn record_binding(
        &self,
        binding: &WorkspaceBinding,
        status: &WorkspaceStatus,
        capabilities: &WorkspaceCapabilities,
    ) -> Result<()> {
        let _gate = lock(&self.gate);
        let existing = self.store.binding(binding.binding_id.as_str())?;
        let now = hi_events::now_ms();
        let record = WorkspaceBindingRecord {
            binding_id: binding.binding_id.to_string(),
            workspace_id: binding.workspace_id.to_string(),
            session_id: session_id(binding),
            epoch: binding.epoch,
            authority: authority(binding),
            state: workspace_state(status.state),
            workspace_version: Some(json_string(&binding.version)?),
            capabilities: Some(serde_json::to_value(capabilities)?),
            revision: existing.as_ref().map_or(1, |record| record.revision + 1),
            opened_at_ms: existing.as_ref().map_or(now, |record| record.opened_at_ms),
            updated_at_ms: now,
            closed_at_ms: None,
        };
        self.commit_binding(record)
    }

    pub fn record_operation_admitted(
        &self,
        binding: &WorkspaceBinding,
        permit: &MutationPermitRecord,
    ) -> Result<()> {
        self.record_operation(
            binding,
            permit,
            OperationProjectionUpdate {
                status: WorkspaceOperationStatus::Admitted,
                execution_ref: None,
                settlement_ref: None,
                result_version: None,
                error: None,
            },
        )
    }

    pub fn record_operation_execution(
        &self,
        binding: &WorkspaceBinding,
        permit: &MutationPermitRecord,
        execution: &ExecutionReport,
    ) -> Result<()> {
        self.record_operation(
            binding,
            permit,
            OperationProjectionUpdate {
                status: WorkspaceOperationStatus::ExecutionRecorded,
                execution_ref: Some(digest_ref(execution)?),
                settlement_ref: None,
                result_version: None,
                error: execution.detail.clone(),
            },
        )
    }

    pub fn record_operation_settled(
        &self,
        binding: &WorkspaceBinding,
        permit: &MutationPermitRecord,
        outcome: &SettlementOutcome,
    ) -> Result<()> {
        self.record_operation(
            binding,
            permit,
            OperationProjectionUpdate {
                status: operation_status(outcome.status),
                execution_ref: None,
                settlement_ref: Some(digest_ref(outcome)?),
                result_version: outcome
                    .receipt
                    .as_ref()
                    .map(|receipt| json_string(&receipt.version))
                    .transpose()?,
                error: outcome.detail.clone(),
            },
        )
    }

    fn record_operation(
        &self,
        binding: &WorkspaceBinding,
        permit: &MutationPermitRecord,
        update: OperationProjectionUpdate,
    ) -> Result<()> {
        let _gate = lock(&self.gate);
        let existing = self.store.operation(permit.operation_id.as_str())?;
        let now = hi_events::now_ms();
        let settled = operation_is_settled(update.status);
        let record = WorkspaceOperationRecord {
            operation_id: permit.operation_id.to_string(),
            binding_id: permit.binding_id.to_string(),
            epoch: permit.epoch,
            session_id: session_id(binding),
            run_id: None,
            attempt_id: None,
            job_id: None,
            kind: effect_scope_name(permit.intent.effect_scope).to_owned(),
            replay_class: replay_class(&permit.intent.replay_class),
            status: update.status,
            operation_digest: stable_digest(permit)?,
            idempotency_key: permit.idempotency_key.to_string(),
            base_version: Some(json_string(&permit.base_version)?),
            result_version: update.result_version.or_else(|| {
                existing
                    .as_ref()
                    .and_then(|record| record.result_version.clone())
            }),
            execution_ref: update.execution_ref.or_else(|| {
                existing
                    .as_ref()
                    .and_then(|record| record.execution_ref.clone())
            }),
            settlement_ref: update.settlement_ref.or_else(|| {
                existing
                    .as_ref()
                    .and_then(|record| record.settlement_ref.clone())
            }),
            error: update.error,
            revision: existing.as_ref().map_or(1, |record| record.revision + 1),
            created_at_ms: existing
                .as_ref()
                .map_or(permit.issued_at_ms, |record| record.created_at_ms),
            updated_at_ms: now.max(permit.issued_at_ms),
            settled_at_ms: settled.then_some(now.max(permit.issued_at_ms)),
        };
        self.commit_operation(record, binding.workspace_id.as_str())
    }

    pub fn record_job_registered(
        &self,
        binding: &WorkspaceBinding,
        permit: &JobPermit,
    ) -> Result<()> {
        self.record_job(binding, permit, JobState::Running, None, None)
    }

    /// Hook for every transition emitted by `WorkspaceJobRegistry`.
    pub fn record_job_snapshot(
        &self,
        binding: &WorkspaceBinding,
        snapshot: &WorkspaceJobSnapshot,
    ) -> Result<()> {
        self.record_job(
            binding,
            &snapshot.permit,
            snapshot.state,
            snapshot.detail.clone(),
            None,
        )
    }

    pub fn record_job_sealed(
        &self,
        binding: &WorkspaceBinding,
        permit: &JobPermit,
        outcome: &JobSealOutcome,
        terminal: &JobTerminal,
    ) -> Result<()> {
        let state = outcome.state.ok_or_else(|| {
            ControlError::Invalid(format!(
                "job {} settlement did not include a state",
                permit.job_id
            ))
        })?;
        self.record_job(
            binding,
            permit,
            state,
            outcome.detail.clone(),
            terminal
                .artifacts
                .first()
                .map(|artifact| artifact.uri.clone()),
        )
    }

    /// Attach crash-recovery evidence fsynced before a terminal lifecycle
    /// callback. This covers a process death between artifact publication and
    /// the subsequent ReadyToMerge transition.
    pub fn record_job_artifact(
        &self,
        job_id: &str,
        artifact: &hi_workspace::ArtifactRef,
        workspace_id: &str,
    ) -> Result<bool> {
        let _gate = lock(&self.gate);
        let Some(mut record) = self.store.job(job_id)? else {
            return Ok(false);
        };
        if record.candidate_ref.as_deref() == Some(&artifact.uri) {
            return Ok(true);
        }
        record.candidate_ref = Some(artifact.uri.clone());
        record.revision = record.revision.saturating_add(1);
        record.updated_at_ms = hi_events::now_ms().max(record.created_at_ms);
        self.commit_job(record, workspace_id)?;
        Ok(true)
    }

    fn record_job(
        &self,
        binding: &WorkspaceBinding,
        permit: &JobPermit,
        state: JobState,
        detail: Option<String>,
        candidate_ref: Option<String>,
    ) -> Result<()> {
        let _gate = lock(&self.gate);
        let existing = self.store.job(permit.job_id.as_str())?;
        let now = hi_events::now_ms();
        let control_state = job_state(state);
        let record = ControlJobRecord {
            job_id: permit.job_id.to_string(),
            session_id: session_id(binding),
            run_id: None,
            attempt_id: None,
            binding_id: Some(permit.binding_id.to_string()),
            epoch: Some(permit.epoch),
            kind: job_kind(permit.spec.kind),
            effect_scope: control_effect_scope(permit.spec.effect_scope),
            state: control_state,
            application_state: None,
            operation_digest: Some(stable_digest(&permit.spec)?),
            idempotency_key: Some(format!("job:{}", permit.job_id)),
            candidate_ref: candidate_ref.or_else(|| {
                existing
                    .as_ref()
                    .and_then(|record| record.candidate_ref.clone())
            }),
            result_ref: existing
                .as_ref()
                .and_then(|record| record.result_ref.clone()),
            workspace_version: Some(json_string(&binding.version)?),
            error: detail,
            revision: existing.as_ref().map_or(1, |record| record.revision + 1),
            created_at_ms: existing
                .as_ref()
                .map_or(permit.issued_at_ms, |record| record.created_at_ms),
            updated_at_ms: now.max(permit.issued_at_ms),
            cancel_requested_at_ms: (control_state == ControlJobState::CancelRequested)
                .then_some(now),
            finished_at_ms: control_state.is_terminal().then_some(now),
        };
        self.commit_job(record, binding.workspace_id.as_str())
    }

    pub fn record_recovery(
        &self,
        binding: &WorkspaceBinding,
        recovery_id: &RecoveryId,
        operation_id: Option<String>,
        job_id: Option<String>,
        status: WorkspaceRecoveryStatus,
        detail: Option<String>,
    ) -> Result<()> {
        let _gate = lock(&self.gate);
        let existing = self.store.recovery(recovery_id.as_str())?;
        let now = hi_events::now_ms();
        let resolved = matches!(
            status,
            WorkspaceRecoveryStatus::Resolved | WorkspaceRecoveryStatus::Discarded
        );
        let record = WorkspaceRecoveryRecord {
            recovery_id: recovery_id.to_string(),
            binding_id: existing
                .as_ref()
                .and_then(|record| record.binding_id.clone())
                .or_else(|| Some(binding.binding_id.to_string())),
            workspace_id: existing.as_ref().map_or_else(
                || binding.workspace_id.to_string(),
                |record| record.workspace_id.clone(),
            ),
            session_id: existing
                .as_ref()
                .and_then(|record| record.session_id.clone())
                .or_else(|| session_id(binding)),
            operation_id: operation_id.or_else(|| {
                existing
                    .as_ref()
                    .and_then(|record| record.operation_id.clone())
            }),
            job_id: job_id.or_else(|| existing.as_ref().and_then(|record| record.job_id.clone())),
            kind: existing.as_ref().map_or_else(
                || "workspace_reconciliation".to_owned(),
                |record| record.kind.clone(),
            ),
            status,
            digest: existing.as_ref().and_then(|record| record.digest.clone()),
            artifact_ref: existing
                .as_ref()
                .and_then(|record| record.artifact_ref.clone()),
            detail: detail.or_else(|| existing.as_ref().and_then(|record| record.detail.clone())),
            error: existing.as_ref().and_then(|record| record.error.clone()),
            revision: existing.as_ref().map_or(1, |record| record.revision + 1),
            created_at_ms: existing.as_ref().map_or(now, |record| record.created_at_ms),
            updated_at_ms: now,
            resolved_at_ms: resolved.then_some(now),
        };
        self.commit_recovery(record)
    }

    pub fn record_recovery_outcome(
        &self,
        binding: &WorkspaceBinding,
        outcome: &RecoveryOutcome,
    ) -> Result<()> {
        let status = match outcome.status {
            RecoveryStatus::Recovered => WorkspaceRecoveryStatus::Resolved,
            RecoveryStatus::Pending | RecoveryStatus::Conflict => WorkspaceRecoveryStatus::Required,
            RecoveryStatus::NotFound | RecoveryStatus::Rejected => WorkspaceRecoveryStatus::Failed,
        };
        self.record_recovery(
            binding,
            &outcome.recovery_id,
            None,
            None,
            status,
            outcome.detail.clone(),
        )?;
        if outcome.status == RecoveryStatus::Recovered {
            self.resolve_linked_recoveries(&outcome.recovery_id)?;
            self.settle_recovered_job(&outcome.recovery_id)?;
            self.settle_recovered_operation(&outcome.recovery_id)?;
        }
        Ok(())
    }

    pub(crate) fn commit_binding(&self, record: WorkspaceBindingRecord) -> Result<()> {
        let event = projection_event(
            "workspace_binding",
            &record.binding_id,
            record.revision,
            record.updated_at_ms,
            &record.workspace_id,
            record.session_id.as_deref(),
            workspace_activity_state(record.state),
        );
        self.store
            .commit(ProjectionTransition::WorkspaceBinding(record), event)?;
        Ok(())
    }

    pub(crate) fn commit_operation(
        &self,
        record: WorkspaceOperationRecord,
        workspace_id: &str,
    ) -> Result<()> {
        let event = projection_event(
            "workspace_operation",
            &record.operation_id,
            record.revision,
            record.updated_at_ms,
            workspace_id,
            record.session_id.as_deref(),
            operation_activity_state(record.status),
        );
        self.store
            .commit(ProjectionTransition::WorkspaceOperation(record), event)?;
        Ok(())
    }

    pub(crate) fn commit_job(&self, record: ControlJobRecord, workspace_id: &str) -> Result<()> {
        let event = projection_event(
            "workspace_job",
            &record.job_id,
            record.revision,
            record.updated_at_ms,
            workspace_id,
            record.session_id.as_deref(),
            job_activity_state(record.state),
        );
        self.store
            .commit(ProjectionTransition::Job(record), event)?;
        Ok(())
    }

    pub(crate) fn commit_recovery(&self, record: WorkspaceRecoveryRecord) -> Result<()> {
        let event = projection_event(
            "workspace_recovery",
            &record.recovery_id,
            record.revision,
            record.updated_at_ms,
            &record.workspace_id,
            record.session_id.as_deref(),
            recovery_activity_state(record.status),
        );
        self.store
            .commit(ProjectionTransition::WorkspaceRecovery(record), event)?;
        Ok(())
    }
}

fn projection_event(
    kind: &str,
    id: &str,
    revision: u64,
    occurred_at_ms: u64,
    workspace_id: &str,
    session_id: Option<&str>,
    state: ActivityState,
) -> RunEvent {
    let identity = format!("hi-control-v2:{kind}:{id}:{revision}");
    let mut event = RunEvent::new(
        EventKind::GitChanged,
        EventContext {
            workspace_id: Some(workspace_id.to_owned()),
            session_id: session_id.map(ToOwned::to_owned),
            correlation_id: Some(id.to_owned()),
            ..EventContext::default()
        },
        SemanticActivity {
            verb: activity_verb(&state),
            object: ActivityObject::Workspace,
            state,
            group_key: kind.to_owned(),
            title: format!("{kind} transition"),
            detail: None,
            refs: Vec::new(),
            progress: None,
        },
    )
    .required()
    .with_field("projection", serde_json::Value::String(kind.to_owned()))
    .with_field("projection_id", serde_json::Value::String(id.to_owned()))
    .with_field("revision", serde_json::Value::from(revision));
    event.event_id =
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, identity.as_bytes()).to_string();
    event.occurred_at_ms = occurred_at_ms;
    event
}

fn activity_verb(state: &ActivityState) -> ActivityVerb {
    match state {
        ActivityState::Succeeded => ActivityVerb::Complete,
        ActivityState::Failed
        | ActivityState::Denied
        | ActivityState::TimedOut
        | ActivityState::Abandoned => ActivityVerb::Fail,
        ActivityState::Cancelled => ActivityVerb::Cancel,
        ActivityState::Waiting => ActivityVerb::Wait,
        ActivityState::Pending => ActivityVerb::Request,
        ActivityState::Running => ActivityVerb::Change,
    }
}

fn authority(binding: &WorkspaceBinding) -> WorkspaceAuthority {
    match binding.authority {
        hi_workspace::WorkspaceAuthority::Local => WorkspaceAuthority::Local,
        hi_workspace::WorkspaceAuthority::PipeFs { .. } => WorkspaceAuthority::PipeFs,
    }
}

fn session_id(binding: &WorkspaceBinding) -> Option<String> {
    match &binding.authority {
        hi_workspace::WorkspaceAuthority::Local => None,
        hi_workspace::WorkspaceAuthority::PipeFs { session_id, .. } => Some(session_id.clone()),
    }
}

fn workspace_state(state: WorkspaceState) -> WorkspaceProjectionState {
    match state {
        WorkspaceState::Ready => WorkspaceProjectionState::Ready,
        WorkspaceState::Mutating => WorkspaceProjectionState::Mutating,
        WorkspaceState::Settling => WorkspaceProjectionState::Settling,
        WorkspaceState::PendingRemote => WorkspaceProjectionState::PendingRemote,
        WorkspaceState::LeaseUncertain => WorkspaceProjectionState::LeaseUncertain,
        WorkspaceState::LeaseLost => WorkspaceProjectionState::LeaseLost,
        WorkspaceState::Conflict => WorkspaceProjectionState::Conflict,
        WorkspaceState::TranscriptPending => WorkspaceProjectionState::TranscriptPending,
        WorkspaceState::CleanupPending => WorkspaceProjectionState::CleanupPending,
        WorkspaceState::RecoveryRequired => WorkspaceProjectionState::RecoveryRequired,
        WorkspaceState::JournalCorrupt => WorkspaceProjectionState::JournalCorrupt,
        WorkspaceState::Incompatible => WorkspaceProjectionState::Incompatible,
        WorkspaceState::LocalAuditDegraded => WorkspaceProjectionState::LocalAuditDegraded,
    }
}

fn operation_status(status: SettlementStatus) -> WorkspaceOperationStatus {
    match status {
        SettlementStatus::Durable => WorkspaceOperationStatus::Durable,
        SettlementStatus::NoChange => WorkspaceOperationStatus::NoChange,
        SettlementStatus::Pending => WorkspaceOperationStatus::Pending,
        SettlementStatus::Indeterminate => WorkspaceOperationStatus::Indeterminate,
        SettlementStatus::LeaseLost => WorkspaceOperationStatus::LeaseLost,
        SettlementStatus::Conflict => WorkspaceOperationStatus::Conflict,
        SettlementStatus::TranscriptPending => WorkspaceOperationStatus::TranscriptPending,
        SettlementStatus::RecoveryRequired => WorkspaceOperationStatus::RecoveryRequired,
        SettlementStatus::LocalAuditDegraded => WorkspaceOperationStatus::LocalAuditDegraded,
        SettlementStatus::Incompatible => WorkspaceOperationStatus::Failed,
    }
}

fn operation_is_settled(status: WorkspaceOperationStatus) -> bool {
    matches!(
        status,
        WorkspaceOperationStatus::Durable
            | WorkspaceOperationStatus::NoChange
            | WorkspaceOperationStatus::LocalAuditDegraded
            | WorkspaceOperationStatus::Failed
    )
}

fn replay_class(replay: &ReplayClass) -> OperationReplayClass {
    match replay {
        ReplayClass::PureWorkspace => OperationReplayClass::PureWorkspace,
        ReplayClass::IdempotentExternal { .. } => OperationReplayClass::IdempotentExternal,
        ReplayClass::NonReplayableExternal => OperationReplayClass::NonReplayableExternal,
    }
}

fn job_kind(kind: hi_workspace::JobKind) -> ControlJobKind {
    match kind {
        hi_workspace::JobKind::Process => ControlJobKind::Process,
        hi_workspace::JobKind::ReadAgent => ControlJobKind::ReadAgent,
        hi_workspace::JobKind::WriteCandidate => ControlJobKind::WriteCandidate,
        hi_workspace::JobKind::Hook => ControlJobKind::Hook,
        hi_workspace::JobKind::Compaction => ControlJobKind::Compaction,
    }
}

fn control_effect_scope(scope: hi_workspace::EffectScope) -> ControlEffectScope {
    match scope {
        hi_workspace::EffectScope::ReadOnly => ControlEffectScope::ReadOnly,
        hi_workspace::EffectScope::CandidateOnly => ControlEffectScope::CandidateOnly,
        hi_workspace::EffectScope::LiveWriter => ControlEffectScope::LiveWriter,
    }
}

fn effect_scope_name(scope: hi_workspace::EffectScope) -> &'static str {
    match scope {
        hi_workspace::EffectScope::ReadOnly => "read_only",
        hi_workspace::EffectScope::CandidateOnly => "candidate_only",
        hi_workspace::EffectScope::LiveWriter => "live_writer",
    }
}

fn job_state(state: JobState) -> ControlJobState {
    match state {
        JobState::Queued => ControlJobState::Queued,
        JobState::Starting => ControlJobState::Starting,
        JobState::Running => ControlJobState::Running,
        JobState::ReadyToMerge => ControlJobState::ReadyToMerge,
        JobState::Merging => ControlJobState::Merging,
        JobState::Settling => ControlJobState::Settling,
        JobState::CancelRequested => ControlJobState::CancelRequested,
        JobState::Succeeded => ControlJobState::Succeeded,
        JobState::Failed => ControlJobState::Failed,
        JobState::Cancelled => ControlJobState::Cancelled,
        JobState::DurabilityPending => ControlJobState::DurabilityPending,
        JobState::RecoveryRequired => ControlJobState::RecoveryRequired,
        JobState::Orphaned => ControlJobState::Orphaned,
        JobState::Stale => ControlJobState::Stale,
    }
}

fn workspace_activity_state(state: WorkspaceProjectionState) -> ActivityState {
    match state {
        WorkspaceProjectionState::Ready | WorkspaceProjectionState::Closed => {
            ActivityState::Succeeded
        }
        WorkspaceProjectionState::Mutating | WorkspaceProjectionState::Settling => {
            ActivityState::Running
        }
        WorkspaceProjectionState::PendingRemote
        | WorkspaceProjectionState::TranscriptPending
        | WorkspaceProjectionState::CleanupPending => ActivityState::Waiting,
        _ => ActivityState::Failed,
    }
}

fn operation_activity_state(state: WorkspaceOperationStatus) -> ActivityState {
    match state {
        WorkspaceOperationStatus::Admitted => ActivityState::Pending,
        WorkspaceOperationStatus::Executing
        | WorkspaceOperationStatus::ExecutionRecorded
        | WorkspaceOperationStatus::Settling => ActivityState::Running,
        WorkspaceOperationStatus::Durable | WorkspaceOperationStatus::NoChange => {
            ActivityState::Succeeded
        }
        WorkspaceOperationStatus::Pending | WorkspaceOperationStatus::TranscriptPending => {
            ActivityState::Waiting
        }
        _ => ActivityState::Failed,
    }
}

fn job_activity_state(state: ControlJobState) -> ActivityState {
    match state {
        ControlJobState::Queued => ActivityState::Pending,
        ControlJobState::Starting
        | ControlJobState::Running
        | ControlJobState::Merging
        | ControlJobState::Settling => ActivityState::Running,
        ControlJobState::ReadyToMerge
        | ControlJobState::DurabilityPending
        | ControlJobState::RecoveryRequired
        | ControlJobState::CancelRequested => ActivityState::Waiting,
        ControlJobState::Succeeded => ActivityState::Succeeded,
        ControlJobState::Cancelled => ActivityState::Cancelled,
        ControlJobState::Orphaned => ActivityState::Abandoned,
        ControlJobState::Failed | ControlJobState::Stale => ActivityState::Failed,
    }
}

fn recovery_activity_state(status: WorkspaceRecoveryStatus) -> ActivityState {
    match status {
        WorkspaceRecoveryStatus::Required | WorkspaceRecoveryStatus::Inspecting => {
            ActivityState::Waiting
        }
        WorkspaceRecoveryStatus::Retrying => ActivityState::Running,
        WorkspaceRecoveryStatus::Resolved | WorkspaceRecoveryStatus::Discarded => {
            ActivityState::Succeeded
        }
        WorkspaceRecoveryStatus::Failed => ActivityState::Failed,
    }
}

fn stable_digest(value: &impl serde::Serialize) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn digest_ref(value: &impl serde::Serialize) -> Result<String> {
    Ok(format!("record://blake3/{}", stable_digest(value)?))
}

fn json_string(value: &impl serde::Serialize) -> Result<String> {
    serde_json::to_string(value).map_err(Into::into)
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
