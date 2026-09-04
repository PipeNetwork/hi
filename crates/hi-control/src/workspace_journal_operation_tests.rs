use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use hi_workspace::{
    BarrierKind, BarrierStatus, InMemoryWorkspaceController, MutationIntent, OperationId,
    SettlementStatus, WorkspaceController, WorkspaceState, restart_operation_recovery_id,
};

use crate::{
    ControlError, ControlJobRecord, ControlStore, ProjectionEventReceipt, ProjectionTransition,
    WorkspaceBindingRecord, WorkspaceOperationRecord, WorkspaceProjectionJournal,
    WorkspaceProjectionStore, WorkspaceRecoveryRecord, WorkspaceRecoveryStatus,
};

#[derive(Clone)]
struct FaultStore {
    inner: ControlStore,
    fail_writes: Arc<AtomicBool>,
    fail_reads: Arc<AtomicBool>,
    fail_next_settlement: Arc<AtomicBool>,
}

impl FaultStore {
    fn new(inner: ControlStore) -> Self {
        Self {
            inner,
            fail_writes: Arc::new(AtomicBool::new(false)),
            fail_reads: Arc::new(AtomicBool::new(false)),
            fail_next_settlement: Arc::new(AtomicBool::new(false)),
        }
    }

    fn check_read(&self) -> crate::Result<()> {
        if self.fail_reads.load(Ordering::SeqCst) {
            Err(ControlError::Invalid(
                "injected journal reload failure".into(),
            ))
        } else {
            Ok(())
        }
    }
}

impl WorkspaceProjectionStore for FaultStore {
    fn commit(
        &self,
        transition: ProjectionTransition,
        event: hi_events::RunEvent,
    ) -> crate::Result<ProjectionEventReceipt> {
        if self.fail_writes.load(Ordering::SeqCst) {
            return Err(ControlError::Invalid(
                "injected operation journal failure".into(),
            ));
        }
        if matches!(
            &transition,
            ProjectionTransition::WorkspaceOperation(operation)
                if matches!(
                    operation.status,
                    crate::WorkspaceOperationStatus::Durable
                        | crate::WorkspaceOperationStatus::NoChange
                )
        ) && self.fail_next_settlement.swap(false, Ordering::SeqCst)
        {
            return Err(ControlError::Invalid(
                "injected final operation journal failure".into(),
            ));
        }
        self.inner.commit_projection_event(transition, event)
    }

    fn binding(&self, id: &str) -> crate::Result<Option<WorkspaceBindingRecord>> {
        self.check_read()?;
        self.inner.get_workspace_binding(id)
    }

    fn operation(&self, id: &str) -> crate::Result<Option<WorkspaceOperationRecord>> {
        self.check_read()?;
        self.inner.get_workspace_operation(id)
    }

    fn operations_for_binding(
        &self,
        binding_id: &str,
    ) -> crate::Result<Vec<WorkspaceOperationRecord>> {
        self.check_read()?;
        self.inner.operations_for_binding(binding_id)
    }

    fn job(&self, id: &str) -> crate::Result<Option<ControlJobRecord>> {
        self.check_read()?;
        self.inner.get_job(id)
    }

    fn recovery(&self, id: &str) -> crate::Result<Option<WorkspaceRecoveryRecord>> {
        self.check_read()?;
        self.inner.get_workspace_recovery(id)
    }

    fn recoveries_for_operation(
        &self,
        operation_id: &str,
    ) -> crate::Result<Vec<WorkspaceRecoveryRecord>> {
        self.check_read()?;
        self.inner.recoveries_for_operation(operation_id)
    }

    fn recoveries_for_job(&self, job_id: &str) -> crate::Result<Vec<WorkspaceRecoveryRecord>> {
        self.check_read()?;
        self.inner.recoveries_for_job(job_id)
    }

    fn jobs_for_binding(&self, binding_id: &str) -> crate::Result<Vec<ControlJobRecord>> {
        self.check_read()?;
        self.inner.jobs_for_binding(binding_id)
    }
}

fn setup() -> (
    tempfile::TempDir,
    ControlStore,
    FaultStore,
    JournaledWorkspaceController,
) {
    let directory = tempfile::tempdir().unwrap();
    let store = ControlStore::open(directory.path().join("events.sqlite3")).unwrap();
    let fault = FaultStore::new(store.clone());
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "operation-fence-workspace",
        "operation-fence-session",
        2,
        true,
        "/work",
        "/state",
    ));
    let controller = JournaledWorkspaceController::attach(inner, Arc::new(fault.clone())).unwrap();
    (directory, store, fault, controller)
}

#[tokio::test]
async fn durable_remote_effect_stays_fenced_until_store_repair_and_reload() {
    let (_directory, store, fault, controller) = setup();
    let permit = controller
        .begin(MutationIntent::workspace("remote edit"))
        .await
        .unwrap();
    let permit_record = permit.snapshot();
    let recovery_id = restart_operation_recovery_id(
        &permit_record.binding_id,
        permit_record.epoch,
        &permit_record.operation_id,
    );
    fault.fail_writes.store(true, Ordering::SeqCst);

    let outcome = controller
        .settle(
            permit,
            ExecutionReport::succeeded(Some("remote-digest".into())),
        )
        .await;
    assert_eq!(outcome.status, SettlementStatus::RecoveryRequired);
    assert_eq!(outcome.recovery_id.as_ref(), Some(&recovery_id));
    let status = controller.status();
    assert_eq!(status.state, WorkspaceState::RecoveryRequired);
    assert_eq!(status.recovery_id.as_ref(), Some(&recovery_id));
    assert_eq!(
        status.active_operation,
        Some(permit_record.operation_id.clone())
    );
    assert!(
        controller
            .begin(MutationIntent::workspace("must remain closed"))
            .await
            .is_err()
    );
    let barrier = controller
        .barrier(
            BarrierKind::Publish,
            Instant::now() + std::time::Duration::from_secs(1),
        )
        .await;
    assert_eq!(barrier.status, BarrierStatus::RecoveryRequired);
    assert_eq!(barrier.recovery_id.as_ref(), Some(&recovery_id));

    let still_faulted = controller.reconcile(recovery_id.clone()).await;
    assert_eq!(still_faulted.status, RecoveryStatus::Pending);
    assert!(still_faulted.detail.unwrap().contains("still pending"));
    assert_eq!(controller.status().recovery_id.as_ref(), Some(&recovery_id));

    fault.fail_writes.store(false, Ordering::SeqCst);
    fault.fail_reads.store(true, Ordering::SeqCst);
    let cannot_reload = controller.reconcile(recovery_id.clone()).await;
    assert_eq!(cannot_reload.status, RecoveryStatus::Pending);
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);

    fault.fail_reads.store(false, Ordering::SeqCst);
    let repaired = controller.reconcile(recovery_id.clone()).await;
    assert_eq!(repaired.status, RecoveryStatus::Recovered);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert_eq!(
        controller.journal_health().state,
        JournalHealthState::Healthy
    );
    let operation = store
        .get_workspace_operation(permit_record.operation_id.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(operation.status, crate::WorkspaceOperationStatus::Durable);
    assert!(operation.execution_ref.is_some());
    assert!(operation.settlement_ref.is_some());
    assert_eq!(
        store
            .get_workspace_recovery(recovery_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceRecoveryStatus::Resolved
    );

    let next = controller
        .begin(MutationIntent::workspace("admission reopened"))
        .await
        .unwrap();
    let next = controller
        .settle(next, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(next.status, SettlementStatus::NoChange);
}

#[tokio::test]
async fn failed_final_publication_restarts_with_the_same_operation_recovery_id() {
    let (_directory, store, fault, controller) = setup();
    let binding = controller.binding();
    let permit = controller
        .begin(MutationIntent::workspace("remote edit"))
        .await
        .unwrap();
    let operation_id = permit.record().operation_id.clone();
    let expected = restart_operation_recovery_id(&binding.binding_id, binding.epoch, &operation_id);
    fault.fail_next_settlement.store(true, Ordering::SeqCst);
    let outcome = controller
        .settle(
            permit,
            ExecutionReport::succeeded(Some("remote-digest".into())),
        )
        .await;
    assert_eq!(outcome.recovery_id.as_ref(), Some(&expected));
    drop(controller);

    let journal = WorkspaceProjectionJournal::new(Arc::new(fault));
    let report = journal.reconcile_jobs_after_restart(&binding).unwrap();
    assert_eq!(report.recovery_ids, vec![expected.clone()]);
    assert_eq!(report.operation_recovery_required, vec![operation_id]);
    let persisted = store
        .get_workspace_recovery(expected.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(persisted.status, WorkspaceRecoveryStatus::Required);
}

#[tokio::test]
async fn settlement_proof_rejects_a_different_remote_operation() {
    let raw = InMemoryWorkspaceController::new_pipefs(
        "proof-workspace",
        "proof-session",
        2,
        true,
        "/work",
        "/state",
    );
    let permit = raw
        .begin(MutationIntent::workspace("remote edit"))
        .await
        .unwrap();
    let record = permit.snapshot();
    let mut outcome = raw
        .settle(
            permit,
            ExecutionReport::succeeded(Some("remote-digest".into())),
        )
        .await;
    validate_exact_settlement_proof(&record, &outcome, &raw.binding()).unwrap();
    outcome.receipt.as_mut().unwrap().operation_id = OperationId::new("different-operation");
    assert!(validate_exact_settlement_proof(&record, &outcome, &raw.binding()).is_err());
}
