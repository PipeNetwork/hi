use std::sync::{Arc, Mutex};

use hi_workspace::{
    InMemoryWorkspaceController, MutationIntent, RecoveryId, SettlementOutcome, SettlementStatus,
    WorkspaceController, restart_operation_recovery_id,
};

use crate::{
    ControlError, ControlJobRecord, ControlStore, ProjectionEventReceipt, ProjectionTransition,
    Result, WorkspaceBindingRecord, WorkspaceOperationRecord, WorkspaceOperationStatus,
    WorkspaceProjectionJournal, WorkspaceProjectionStore, WorkspaceRecoveryRecord,
    WorkspaceRecoveryStatus,
};

#[derive(Clone, Copy)]
enum Failure {
    Disposition,
    Lifecycle,
}

#[derive(Clone)]
struct FailOnceStore {
    inner: ControlStore,
    failure: Arc<Mutex<Option<Failure>>>,
}

impl FailOnceStore {
    fn arm(&self, failure: Failure) {
        *self.failure.lock().unwrap() = Some(failure);
    }
}

impl WorkspaceProjectionStore for FailOnceStore {
    fn commit(
        &self,
        transition: ProjectionTransition,
        event: hi_events::RunEvent,
    ) -> Result<ProjectionEventReceipt> {
        let fail = match &transition {
            ProjectionTransition::WorkspaceRecovery(record)
                if record.status == WorkspaceRecoveryStatus::Discarded =>
            {
                matches!(*self.failure.lock().unwrap(), Some(Failure::Disposition))
            }
            ProjectionTransition::WorkspaceOperation(record)
                if record.status == WorkspaceOperationStatus::Failed =>
            {
                matches!(*self.failure.lock().unwrap(), Some(Failure::Lifecycle))
            }
            _ => false,
        };
        if fail {
            self.failure.lock().unwrap().take();
            return Err(ControlError::Invalid(
                "injected recovery commit failure".into(),
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

    fn operations_for_binding(&self, id: &str) -> Result<Vec<WorkspaceOperationRecord>> {
        self.inner.operations_for_binding(id)
    }

    fn job(&self, id: &str) -> Result<Option<ControlJobRecord>> {
        self.inner.get_job(id)
    }

    fn recovery(&self, id: &str) -> Result<Option<WorkspaceRecoveryRecord>> {
        self.inner.get_workspace_recovery(id)
    }

    fn recoveries_for_operation(&self, id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.inner.recoveries_for_operation(id)
    }

    fn recoveries_for_job(&self, id: &str) -> Result<Vec<WorkspaceRecoveryRecord>> {
        self.inner.recoveries_for_job(id)
    }

    fn jobs_for_binding(&self, id: &str) -> Result<Vec<ControlJobRecord>> {
        self.inner.jobs_for_binding(id)
    }
}

#[tokio::test]
async fn local_discard_is_fail_closed_and_idempotent_across_partial_commits() {
    let directory = tempfile::tempdir().unwrap();
    let store = ControlStore::open_for_state(directory.path()).unwrap();
    let controller = InMemoryWorkspaceController::new_local(
        "workspace",
        directory.path().join("work"),
        directory.path(),
    );
    let binding = controller.binding();
    let base = WorkspaceProjectionJournal::from_control_store(store.clone());
    base.record_binding(&binding, &controller.status(), &controller.capabilities())
        .unwrap();
    let permit = controller
        .begin(MutationIntent::workspace("interrupted"))
        .await
        .unwrap();
    let operation_id = permit.record().operation_id.clone();
    base.record_operation_admitted(&binding, &permit.snapshot())
        .unwrap();
    drop(permit);
    let recovery_id =
        restart_operation_recovery_id(&binding.binding_id, binding.epoch, &operation_id);
    let fault = FailOnceStore {
        inner: store.clone(),
        failure: Arc::new(Mutex::new(None)),
    };
    let journal = WorkspaceProjectionJournal::new(Arc::new(fault.clone()));

    fault.arm(Failure::Disposition);
    assert!(
        journal
            .discard_local_restart_recovery(
                "workspace",
                binding.binding_id.as_str(),
                &recovery_id,
                Some(operation_id.as_str()),
                None,
                "blake3:confirmed",
            )
            .is_err()
    );
    assert_eq!(
        store
            .get_workspace_operation(operation_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceOperationStatus::Admitted
    );
    assert_eq!(
        store
            .get_workspace_recovery(recovery_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceRecoveryStatus::Required
    );
    assert_eq!(
        store
            .unsettled_workspace_bindings("workspace")
            .unwrap()
            .len(),
        1
    );

    fault.arm(Failure::Lifecycle);
    assert!(
        journal
            .discard_local_restart_recovery(
                "workspace",
                binding.binding_id.as_str(),
                &recovery_id,
                Some(operation_id.as_str()),
                None,
                "blake3:confirmed",
            )
            .is_err()
    );
    assert_eq!(
        store
            .get_workspace_recovery(recovery_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceRecoveryStatus::Discarded
    );
    assert_eq!(
        store
            .unsettled_workspace_bindings("workspace")
            .unwrap()
            .len(),
        1
    );

    for _ in 0..2 {
        journal
            .discard_local_restart_recovery(
                "workspace",
                binding.binding_id.as_str(),
                &recovery_id,
                Some(operation_id.as_str()),
                None,
                "blake3:confirmed",
            )
            .unwrap();
    }
    assert_eq!(
        store
            .get_workspace_operation(operation_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceOperationStatus::Failed
    );
    assert!(
        store
            .unsettled_workspace_bindings("workspace")
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn unresolved_recovery_seeds_restart_without_downgrading_durable_operation() {
    let directory = tempfile::tempdir().unwrap();
    let store = ControlStore::open_for_state(directory.path()).unwrap();
    let controller = InMemoryWorkspaceController::new_local(
        "workspace",
        directory.path().join("work"),
        directory.path(),
    );
    let binding = controller.binding();
    let journal = WorkspaceProjectionJournal::from_control_store(store.clone());
    journal
        .record_binding(&binding, &controller.status(), &controller.capabilities())
        .unwrap();
    let permit = controller
        .begin(MutationIntent::workspace("settled edit"))
        .await
        .unwrap();
    let snapshot = permit.snapshot();
    journal
        .record_operation_admitted(&binding, &snapshot)
        .unwrap();
    journal
        .record_operation_settled(
            &binding,
            &snapshot,
            &SettlementOutcome {
                status: SettlementStatus::Durable,
                operation_id: snapshot.operation_id.clone(),
                receipt: None,
                recovery_id: None,
                detail: None,
            },
        )
        .unwrap();
    drop(permit);
    let legacy = RecoveryId::new("legacy-repair-fence");
    journal
        .record_recovery(
            &binding,
            &legacy,
            Some(snapshot.operation_id.to_string()),
            None,
            WorkspaceRecoveryStatus::Required,
            Some("settlement persisted before recovery repair".into()),
        )
        .unwrap();
    let binding_fence = RecoveryId::new("binding-level-repair-fence");
    journal
        .record_recovery(
            &binding,
            &binding_fence,
            None,
            None,
            WorkspaceRecoveryStatus::Required,
            Some("binding repair was interrupted".into()),
        )
        .unwrap();

    assert_eq!(
        store
            .unsettled_workspace_bindings("workspace")
            .unwrap()
            .len(),
        1
    );
    let report = journal.reconcile_jobs_after_restart(&binding).unwrap();
    let stable =
        restart_operation_recovery_id(&binding.binding_id, binding.epoch, &snapshot.operation_id);
    assert_eq!(report.recovery_ids.len(), 2);
    assert!(report.recovery_ids.contains(&stable));
    assert!(report.recovery_ids.contains(&binding_fence));
    assert_eq!(
        store
            .get_workspace_operation(snapshot.operation_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceOperationStatus::Durable
    );
    let receipt = journal
        .discard_local_restart_recovery(
            "workspace",
            binding.binding_id.as_str(),
            &binding_fence,
            None,
            None,
            "blake3:binding-confirmed",
        )
        .unwrap();
    assert!(!receipt.lifecycle_marked_failed);
    assert_eq!(
        store
            .get_workspace_recovery(binding_fence.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceRecoveryStatus::Discarded
    );
}
