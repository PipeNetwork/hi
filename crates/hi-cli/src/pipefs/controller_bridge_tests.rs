use std::sync::Arc;

use hi_control::{ControlJobState, JournaledWorkspaceController, WorkspaceOperationStatus};
use hi_workspace::{
    EffectScope, ExecutionReport, IdempotencyKey, InMemoryWorkspaceController, JobKind, JobLimits,
    JobSpec, MutationIntent, ReplayClass, WorkspaceController, restart_operation_recovery_id,
};

use super::*;

fn old_controller(
    store: &ControlStore,
    workspace: &str,
    session: &str,
    epoch: u64,
) -> JournaledWorkspaceController {
    let inner: Arc<dyn WorkspaceController> =
        Arc::new(InMemoryWorkspaceController::new_pipefs_at_epoch(
            workspace,
            session,
            2,
            true,
            format!("/work/{workspace}"),
            format!("/state/{workspace}"),
            epoch,
        ));
    JournaledWorkspaceController::attach_store(inner, store.clone()).unwrap()
}

#[tokio::test]
async fn one_pending_archive_cannot_mask_other_operations_or_writer_jobs() {
    let directory = tempfile::tempdir().unwrap();
    let store = ControlStore::open(directory.path().join("events.sqlite3")).unwrap();
    let first = old_controller(&store, "workspace", "session", 3);
    let first_permit = first
        .begin(MutationIntent {
            effect_scope: EffectScope::LiveWriter,
            replay_class: ReplayClass::IdempotentExternal {
                key: IdempotencyKey::new("first-idempotency-key"),
            },
            dirty_paths: None,
            description: Some("first interrupted operation".into()),
        })
        .await
        .unwrap();
    let first_operation = first_permit.snapshot();
    drop(first_permit);

    let second = old_controller(&store, "workspace", "session", 4);
    let second_permit = second
        .begin(MutationIntent::workspace("second interrupted operation"))
        .await
        .unwrap();
    let second_operation = second_permit.snapshot();
    let writer = second
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "interrupted candidate".into(),
            limits: JobLimits::default(),
            parent_operation: Some(second_operation.operation_id.clone()),
        })
        .await
        .unwrap();
    drop(second_permit);

    let evidence = CausalOperationReceipt {
        operation_id: first_operation.operation_id.to_string(),
        idempotency_key: first_operation.idempotency_key.to_string(),
        binding_id: first_operation.binding_id.to_string(),
        binding_epoch: first_operation.epoch,
        replay_class: first_operation.intent.replay_class.clone(),
        execution: ExecutionReport::succeeded(None),
    };
    let historical = store.unsettled_pipefs_bindings("session").unwrap();
    assert_eq!(historical.len(), 2);
    let current = hi_workspace::WorkspaceBinding::new_pipefs(
        "current-controller".into(),
        "workspace".into(),
        "session".into(),
        2,
        "/work/current".into(),
        "/state/current".into(),
    );
    let plan = restart_recovery_plan(&store, &historical, &current, Some(&evidence)).unwrap();
    let expected = restart_operation_recovery_id(
        &first_operation.binding_id,
        first_operation.epoch,
        &first_operation.operation_id,
    );

    assert_eq!(plan.matched_real.as_ref(), Some(&expected));
    assert_eq!(plan.unmatched.len(), 2);
    assert!(
        plan.unmatched
            .iter()
            .any(|record| { record.operation_id.as_ref() == Some(&second_operation.operation_id) })
    );
    assert!(
        plan.unmatched
            .iter()
            .any(|record| record.job_id.as_ref() == Some(&writer.job_id))
    );
    assert_eq!(
        store
            .get_workspace_operation(first_operation.operation_id.as_str())
            .unwrap()
            .unwrap()
            .status,
        WorkspaceOperationStatus::RecoveryRequired
    );
    assert_eq!(
        store
            .get_job(writer.job_id.as_str())
            .unwrap()
            .unwrap()
            .state,
        ControlJobState::RecoveryRequired
    );

    let mut malformed_evidence = evidence.clone();
    malformed_evidence.replay_class = ReplayClass::IdempotentExternal {
        key: IdempotencyKey::new("different-embedded-key"),
    };
    let malformed_error =
        restart_recovery_plan(&store, &historical, &current, Some(&malformed_evidence))
            .unwrap_err();
    assert!(
        malformed_error
            .to_string()
            .contains("exact identity fences")
    );

    let mut wrong_evidence = evidence;
    wrong_evidence.idempotency_key = "different-idempotency-key".into();
    let retry_error =
        restart_recovery_plan(&store, &historical, &current, Some(&wrong_evidence)).unwrap_err();
    assert!(retry_error.to_string().contains("exact identity fences"));
}
