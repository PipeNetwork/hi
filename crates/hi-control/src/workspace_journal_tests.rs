use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hi_workspace::{
    ArtifactRef, EffectScope, ExecutionReport, InMemoryWorkspaceController, JobCompletion, JobKind,
    JobLimits, JobSealStatus, JobSpec, JobState, JobTerminal, MutationIntent, SettlementStatus,
    WorkspaceController, WorkspaceJobRegistry, WorkspaceState,
};

use crate::{
    ControlJobRecord, ControlJobState, ControlStore, JournalHealthState,
    JournaledWorkspaceController, ProjectionEventReceipt, ProjectionTransition, Result,
    WorkspaceBindingRecord, WorkspaceOperationRecord, WorkspaceProjectionStore,
    WorkspaceRecoveryRecord,
};

#[derive(Clone)]
struct FaultStore {
    inner: ControlStore,
    fail_commits: Arc<AtomicBool>,
    fail_next_succeeded_job_commit: Arc<AtomicBool>,
}

impl FaultStore {
    fn new(inner: ControlStore) -> Self {
        Self {
            inner,
            fail_commits: Arc::new(AtomicBool::new(false)),
            fail_next_succeeded_job_commit: Arc::new(AtomicBool::new(false)),
        }
    }

    fn set_failing(&self, failing: bool) {
        self.fail_commits.store(failing, Ordering::SeqCst);
    }

    fn fail_next_succeeded_job_commit(&self) {
        self.fail_next_succeeded_job_commit
            .store(true, Ordering::SeqCst);
    }
}

impl WorkspaceProjectionStore for FaultStore {
    fn commit(
        &self,
        transition: ProjectionTransition,
        event: hi_events::RunEvent,
    ) -> Result<ProjectionEventReceipt> {
        if matches!(
            &transition,
            ProjectionTransition::Job(record) if record.state == ControlJobState::Succeeded
        ) && self
            .fail_next_succeeded_job_commit
            .swap(false, Ordering::SeqCst)
        {
            return Err(crate::ControlError::Invalid(
                "injected final-Succeeded journal failure".into(),
            ));
        }
        if self.fail_commits.load(Ordering::SeqCst) {
            return Err(crate::ControlError::Invalid(
                "injected journal failure".into(),
            ));
        }
        self.inner.commit_projection_event(transition, event)
    }

    fn binding(&self, id: &str) -> Result<Option<WorkspaceBindingRecord>> {
        self.inner.get_workspace_binding(id)
    }

    fn operation(&self, id: &str) -> Result<Option<WorkspaceOperationRecord>> {
        self.inner.get_workspace_operation(id)
    }

    fn operations_for_binding(&self, binding_id: &str) -> Result<Vec<WorkspaceOperationRecord>> {
        self.inner.operations_for_binding(binding_id)
    }

    fn job(&self, id: &str) -> Result<Option<ControlJobRecord>> {
        self.inner.get_job(id)
    }

    fn recovery(&self, id: &str) -> Result<Option<WorkspaceRecoveryRecord>> {
        self.inner.get_workspace_recovery(id)
    }

    fn recoveries_for_operation(&self, operation_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.inner.recoveries_for_operation(operation_id)
    }

    fn recoveries_for_job(&self, job_id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.inner.recoveries_for_job(job_id)
    }

    fn jobs_for_binding(&self, binding_id: &str) -> Result<Vec<ControlJobRecord>> {
        self.inner.jobs_for_binding(binding_id)
    }
}

fn store() -> (tempfile::TempDir, ControlStore) {
    let directory = tempfile::tempdir().unwrap();
    let store = ControlStore::open(directory.path().join("events.sqlite3")).unwrap();
    (directory, store)
}

fn candidate_spec(name: &str) -> JobSpec {
    JobSpec {
        kind: JobKind::WriteCandidate,
        effect_scope: EffectScope::CandidateOnly,
        name: name.to_owned(),
        limits: JobLimits::default(),
        parent_operation: None,
    }
}

fn reader_spec(name: &str) -> JobSpec {
    JobSpec {
        kind: JobKind::ReadAgent,
        effect_scope: EffectScope::ReadOnly,
        name: name.to_owned(),
        limits: JobLimits::default(),
        parent_operation: None,
    }
}

#[tokio::test]
async fn local_mutation_projects_admission_execution_settlement_and_binding() {
    let (_directory, store) = store();
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach_store(inner, store.clone()).unwrap();

    let permit = controller
        .begin(MutationIntent::workspace("edit"))
        .await
        .unwrap();
    let operation_id = permit.record().operation_id.clone();
    let outcome = controller
        .settle(
            permit,
            ExecutionReport::succeeded(Some("content-digest".into())),
        )
        .await;

    assert_eq!(outcome.status, SettlementStatus::Durable);
    let operation = store
        .get_workspace_operation(operation_id.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(operation.state_name(), "durable");
    assert_eq!(operation.revision, 3);
    assert!(operation.execution_ref.is_some());
    assert!(operation.settlement_ref.is_some());
    assert!(operation.result_version.is_some());
    let binding = controller.binding();
    let persisted_binding = store
        .get_workspace_binding(binding.binding_id.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(persisted_binding.state_name(), "ready");
    assert!(store.max_event_sequence().unwrap() >= 6);
}

#[tokio::test]
async fn local_journal_failure_surfaces_degradation_but_keeps_foreground_work() {
    let (_directory, store) = store();
    let fault = FaultStore::new(store);
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    fault.set_failing(true);

    let permit = controller
        .begin(MutationIntent::workspace("foreground edit"))
        .await
        .unwrap();
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(Some("digest".into())))
        .await;
    assert_eq!(outcome.status, SettlementStatus::LocalAuditDegraded);
    assert_eq!(
        controller.journal_health().state,
        JournalHealthState::LocalAuditDegraded
    );
    assert_eq!(
        controller.status().state,
        WorkspaceState::LocalAuditDegraded
    );
    assert!(!controller.capabilities().candidate_apply);
    assert_eq!(
        controller.capabilities().crash_recovery,
        hi_workspace::CrashRecoveryCapability::None
    );
    assert!(!controller.capabilities().background_writers);

    let denied = controller
        .register_job(candidate_spec("background writer"))
        .await
        .unwrap_err();
    assert!(denied.detail.contains("healthy workspace journal"));

    // A foreground mutation is still admitted by the underlying Ready state,
    // despite the explicit degraded status presented by the decorator.
    let second = controller
        .begin(MutationIntent::workspace("second foreground edit"))
        .await
        .unwrap();
    let second_outcome = controller
        .settle(second, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(second_outcome.status, SettlementStatus::LocalAuditDegraded);
}

#[tokio::test]
async fn pipefs_journal_failure_closes_admission_and_does_not_return_a_permit() {
    let (_directory, store) = store();
    let fault = FaultStore::new(store);
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "workspace",
        "session",
        2,
        true,
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    fault.set_failing(true);

    assert!(
        controller
            .begin(MutationIntent::workspace("remote edit"))
            .await
            .is_err()
    );
    assert_eq!(
        controller.journal_health().state,
        JournalHealthState::PipeFsFailClosed
    );
    assert_eq!(controller.status().state, WorkspaceState::JournalCorrupt);
    assert!(
        controller
            .begin(MutationIntent::workspace("blocked edit"))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn pipefs_settlement_is_not_reported_durable_after_journal_ambiguity() {
    let (_directory, store) = store();
    let fault = FaultStore::new(store);
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "workspace",
        "session",
        2,
        true,
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    let permit = controller
        .begin(MutationIntent::workspace("remote edit"))
        .await
        .unwrap();
    fault.set_failing(true);

    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(Some("digest".into())))
        .await;
    assert_eq!(outcome.status, SettlementStatus::RecoveryRequired);
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    assert!(
        controller
            .begin(MutationIntent::workspace("must remain blocked"))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn failed_local_writer_admission_does_not_poison_foreground_mutations() {
    let (_directory, store) = store();
    let fault = FaultStore::new(store);
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    fault.set_failing(true);

    assert!(
        controller
            .register_job(candidate_spec("candidate"))
            .await
            .is_err()
    );
    let permit = controller
        .begin(MutationIntent::workspace("foreground remains available"))
        .await
        .unwrap();
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::LocalAuditDegraded);
}

#[tokio::test]
async fn nonterminal_local_writer_journal_failure_stays_recovery_required() {
    let (_directory, store) = store();
    let fault = FaultStore::new(store.clone());
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    let job = controller
        .register_job(candidate_spec("candidate"))
        .await
        .unwrap();
    fault.set_failing(true);

    let outcome = controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::ReadyToMerge,
                detail: None,
                artifacts: vec![ArtifactRef {
                    uri: "artifact://candidate/evidence".into(),
                    digest: Some("blake3:evidence".into()),
                    size_bytes: Some(42),
                }],
            },
        )
        .await;
    assert_eq!(outcome.status, JobSealStatus::Rejected);
    assert_eq!(outcome.state, Some(JobState::RecoveryRequired));
    let recovery_id = outcome
        .recovery_id
        .expect("divergent inner and journal states need a recovery identity");
    let status = controller.status();
    assert_eq!(status.state, WorkspaceState::RecoveryRequired);
    assert_eq!(status.recovery_id.as_ref(), Some(&recovery_id));
    assert!(status.active_jobs.contains(&job.job_id));
    assert!(
        controller
            .begin(MutationIntent::workspace("must stay closed"))
            .await
            .is_err()
    );
    let barrier = controller
        .barrier(
            hi_workspace::BarrierKind::Publish,
            std::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .await;
    assert_eq!(
        barrier.status,
        hi_workspace::BarrierStatus::RecoveryRequired
    );
    assert_eq!(barrier.recovery_id.as_ref(), Some(&recovery_id));
    assert!(barrier.pending_jobs.contains(&job.job_id));
    let persisted = store.get_job(job.job_id.as_str()).unwrap().unwrap();
    assert_eq!(persisted.state, ControlJobState::Running);
    assert!(persisted.candidate_ref.is_none());
}

#[tokio::test]
async fn final_succeeded_writer_journal_failure_stays_recovery_required_and_blocks_barriers() {
    let (_directory, store) = store();
    let fault = FaultStore::new(store.clone());
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    let job = controller
        .register_job(candidate_spec("candidate"))
        .await
        .unwrap();

    for completion in [
        JobCompletion::ReadyToMerge,
        JobCompletion::Merging,
        JobCompletion::Settling,
    ] {
        let outcome = controller
            .seal_job(
                job.job_id.clone(),
                JobTerminal {
                    completion,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await;
        assert_eq!(outcome.status, JobSealStatus::Sealed);
    }

    fault.fail_next_succeeded_job_commit();
    let outcome = controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::Succeeded,
                detail: Some("candidate applied and verified".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(outcome.status, JobSealStatus::Rejected);
    assert_eq!(outcome.state, Some(JobState::RecoveryRequired));
    let recovery_id = outcome
        .recovery_id
        .clone()
        .expect("journal ambiguity must have a recovery identity");

    let status = controller.status();
    assert_eq!(status.state, WorkspaceState::RecoveryRequired);
    assert_eq!(status.recovery_id.as_ref(), Some(&recovery_id));
    assert!(status.active_jobs.contains(&job.job_id));
    assert!(
        controller
            .begin(MutationIntent::workspace("must stay closed"))
            .await
            .is_err()
    );

    let barrier = controller
        .barrier(
            hi_workspace::BarrierKind::Publish,
            std::time::Instant::now() + std::time::Duration::from_secs(1),
        )
        .await;
    assert_eq!(
        barrier.status,
        hi_workspace::BarrierStatus::RecoveryRequired
    );
    assert_eq!(barrier.recovery_id.as_ref(), Some(&recovery_id));
    assert!(barrier.pending_jobs.contains(&job.job_id));

    // The inner state is already Succeeded. A retry must not turn that
    // AlreadySealed response into an acknowledgement of RecoveryRequired.
    let retry = controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::RecoveryRequired,
                detail: Some("retry recovery transition".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(retry.status, JobSealStatus::Rejected);
    assert_eq!(retry.state, Some(JobState::RecoveryRequired));
    assert_eq!(retry.recovery_id.as_ref(), Some(&recovery_id));

    assert_eq!(
        store.get_job(job.job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::Settling,
        "the durable projection must never claim Succeeded"
    );
    let recovery = store
        .get_workspace_recovery(recovery_id.as_str())
        .unwrap()
        .expect("the one-shot failure allows the recovery marker to be persisted");
    assert_eq!(recovery.job_id.as_deref(), Some(job.job_id.as_str()));
    assert_eq!(recovery.status, crate::WorkspaceRecoveryStatus::Required);
}

#[tokio::test]
async fn restart_reconciliation_is_idempotent_and_preserves_writer_evidence() {
    let (_directory, store) = store();
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach_store(inner, store.clone()).unwrap();
    let candidate = controller
        .register_job(candidate_spec("candidate"))
        .await
        .unwrap();
    let reader = controller
        .register_job(reader_spec("reader"))
        .await
        .unwrap();
    let binding = controller.binding();
    let before = store.max_event_sequence().unwrap();

    let first = controller
        .projection_journal()
        .reconcile_jobs_after_restart(&binding)
        .unwrap();
    assert!(first.recovery_required.contains(&candidate.job_id));
    assert!(first.orphaned.contains(&reader.job_id));
    assert_eq!(first.recovery_ids.len(), 1);
    assert_eq!(
        store
            .get_job(candidate.job_id.as_str())
            .unwrap()
            .unwrap()
            .state,
        ControlJobState::RecoveryRequired
    );
    assert_eq!(
        store
            .get_job(reader.job_id.as_str())
            .unwrap()
            .unwrap()
            .state,
        ControlJobState::Orphaned
    );
    let recovery = store
        .get_workspace_recovery(first.recovery_ids[0].as_str())
        .unwrap()
        .unwrap();
    assert_eq!(recovery.job_id.as_deref(), Some(candidate.job_id.as_str()));
    assert_eq!(store.unsettled_jobs().unwrap().len(), 1);
    assert_eq!(
        store
            .get_workspace_binding(binding.binding_id.as_str())
            .unwrap()
            .unwrap()
            .state,
        crate::WorkspaceProjectionState::RecoveryRequired
    );

    let after_first = store.max_event_sequence().unwrap();
    assert!(after_first > before);
    let second = controller
        .projection_journal()
        .reconcile_jobs_after_restart(&binding)
        .unwrap();
    assert_eq!(second.recovery_ids, first.recovery_ids);
    assert_eq!(store.max_event_sequence().unwrap(), after_first);
}

#[tokio::test]
async fn restart_reconciliation_fences_an_unsettled_foreground_operation() {
    let (_directory, store) = store();
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        "foreground-workspace",
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach_store(inner, store.clone()).unwrap();
    let permit = controller
        .begin(MutationIntent::workspace("interrupted edit"))
        .await
        .unwrap();
    let operation_id = permit.record().operation_id.clone();
    let binding = controller.binding();

    // Model the crash boundary after admission and before settlement. The
    // admitted operation itself is durable evidence even if no later callback
    // had time to append another record.
    drop(permit);
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    let historical = store
        .unsettled_workspace_bindings("foreground-workspace")
        .unwrap();
    assert_eq!(historical.len(), 1);
    assert_eq!(historical[0].binding_id, binding.binding_id.as_str());

    let first = controller
        .projection_journal()
        .reconcile_jobs_after_restart(&binding)
        .unwrap();
    assert_eq!(
        first.operation_recovery_required.as_slice(),
        std::slice::from_ref(&operation_id)
    );
    assert_eq!(first.recovery_ids.len(), 1);
    let operation = store
        .get_workspace_operation(operation_id.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(
        operation.status,
        crate::WorkspaceOperationStatus::RecoveryRequired
    );
    let recovery = store
        .get_workspace_recovery(first.recovery_ids[0].as_str())
        .unwrap()
        .unwrap();
    assert_eq!(
        recovery.operation_id.as_deref(),
        Some(operation_id.as_str())
    );
    assert_eq!(recovery.job_id, None);

    let after_first = store.max_event_sequence().unwrap();
    let second = controller
        .projection_journal()
        .reconcile_jobs_after_restart(&binding)
        .unwrap();
    assert_eq!(second.recovery_ids, first.recovery_ids);
    assert_eq!(store.max_event_sequence().unwrap(), after_first);
}

#[tokio::test]
async fn deterministic_controller_retry_resolves_the_matching_old_operation() {
    let (_directory, store) = store();
    let old_inner: Arc<dyn WorkspaceController> =
        Arc::new(InMemoryWorkspaceController::new_pipefs_at_epoch(
            "restart-workspace",
            "restart-session",
            2,
            true,
            "/old/work",
            "/old/state",
            4,
        ));
    let old = JournaledWorkspaceController::attach_store(old_inner, store.clone()).unwrap();
    let permit = old
        .begin(MutationIntent::workspace("interrupted publication"))
        .await
        .unwrap();
    let operation = permit.snapshot();
    let old_binding = old.binding();
    drop(permit);
    let legacy_recovery = hi_workspace::RecoveryId::new("legacy-random-recovery");
    old.projection_journal()
        .record_recovery(
            &old_binding,
            &legacy_recovery,
            Some(operation.operation_id.to_string()),
            None,
            crate::WorkspaceRecoveryStatus::Required,
            Some("original ambiguous settlement".into()),
        )
        .unwrap();
    let report = old
        .projection_journal()
        .reconcile_jobs_after_restart(&old_binding)
        .unwrap();
    let recovery_id = hi_workspace::restart_operation_recovery_id(
        &operation.binding_id,
        operation.epoch,
        &operation.operation_id,
    );
    assert_eq!(report.recovery_ids, std::slice::from_ref(&recovery_id));

    let restarted_raw = InMemoryWorkspaceController::new_pipefs_at_epoch(
        "restart-workspace",
        "restart-session",
        2,
        true,
        "/new/work",
        "/new/state",
        5,
    );
    let current = restarted_raw.binding();
    restarted_raw
        .require_recovery(hi_workspace::RecoveryRecord {
            schema_version: hi_workspace::WORKSPACE_CONTRACT_SCHEMA_VERSION,
            recovery_id: recovery_id.clone(),
            kind: hi_workspace::RecoveryKind::UnsettledMutation,
            binding_id: current.binding_id,
            epoch: current.epoch,
            operation_id: Some(operation.operation_id.clone()),
            job_id: None,
            detail: "restored exact PipeFS operation evidence".into(),
            created_at_ms: operation.issued_at_ms,
            resolved: false,
        })
        .unwrap();
    let restarted: Arc<dyn WorkspaceController> = Arc::new(restarted_raw);
    let restarted = JournaledWorkspaceController::attach_store(restarted, store.clone()).unwrap();
    let outcome = restarted.reconcile(recovery_id.clone()).await;

    assert_eq!(outcome.status, hi_workspace::RecoveryStatus::Recovered);
    assert_eq!(
        store
            .get_workspace_recovery(recovery_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        crate::WorkspaceRecoveryStatus::Resolved
    );
    assert_eq!(
        store
            .get_workspace_recovery(legacy_recovery.as_str())
            .unwrap()
            .unwrap()
            .status,
        crate::WorkspaceRecoveryStatus::Resolved
    );
    assert_eq!(
        store
            .get_workspace_operation(operation.operation_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        crate::WorkspaceOperationStatus::Failed
    );
}

#[test]
fn transition_validated_registry_snapshots_advance_the_same_job_projection() {
    let (_directory, store) = store();
    let binding = hi_workspace::WorkspaceBinding::new_local(
        "controller".into(),
        "workspace".into(),
        "/work".into(),
        "/state".into(),
    );
    let journal = crate::WorkspaceProjectionJournal::from_control_store(store.clone());
    journal
        .record_binding(
            &binding,
            &hi_workspace::WorkspaceStatus::ready(&binding),
            &hi_workspace::WorkspaceCapabilities::local(),
        )
        .unwrap();
    let registry = WorkspaceJobRegistry::new(binding.clone());
    let fence = registry.fence();
    let permit = registry
        .register(&fence, candidate_spec("candidate"))
        .unwrap();
    journal
        .record_job_snapshot(&binding, &registry.status(&fence, &permit.job_id).unwrap())
        .unwrap();
    let starting = registry
        .transition(
            &fence,
            &permit.job_id,
            JobState::Queued,
            JobState::Starting,
            None,
            Vec::new(),
        )
        .unwrap();
    journal.record_job_snapshot(&binding, &starting).unwrap();

    let persisted = store.get_job(permit.job_id.as_str()).unwrap().unwrap();
    assert_eq!(persisted.state, ControlJobState::Starting);
    assert_eq!(persisted.revision, 2);
}

trait ProjectionStateName {
    fn state_name(&self) -> &'static str;
}

impl ProjectionStateName for WorkspaceOperationRecord {
    fn state_name(&self) -> &'static str {
        match self.status {
            crate::WorkspaceOperationStatus::Durable => "durable",
            _ => "other",
        }
    }
}

impl ProjectionStateName for WorkspaceBindingRecord {
    fn state_name(&self) -> &'static str {
        match self.state {
            crate::WorkspaceProjectionState::Ready => "ready",
            _ => "other",
        }
    }
}
