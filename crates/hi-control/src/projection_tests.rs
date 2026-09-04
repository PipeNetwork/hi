use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, RunEvent,
    SemanticActivity,
};

use super::*;
use crate::CONTROL_SCHEMA_VERSION;

fn store() -> ControlStore {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.sqlite3");
    let store = ControlStore::open(path).unwrap();
    std::mem::forget(dir);
    store
}

fn required_event(title: &str) -> RunEvent {
    RunEvent::new(
        EventKind::GitChanged,
        EventContext {
            workspace_id: Some("workspace".into()),
            session_id: Some("session".into()),
            ..EventContext::default()
        },
        SemanticActivity {
            verb: ActivityVerb::Change,
            object: ActivityObject::Workspace,
            state: ActivityState::Running,
            group_key: "workspace".into(),
            title: title.into(),
            detail: None,
            refs: vec![],
            progress: None,
        },
    )
    .required()
}

fn binding(revision: u64, state: WorkspaceProjectionState) -> WorkspaceBindingRecord {
    WorkspaceBindingRecord {
        binding_id: "binding-1".into(),
        workspace_id: "workspace".into(),
        session_id: Some("session".into()),
        epoch: 7,
        authority: WorkspaceAuthority::PipeFs,
        state,
        workspace_version: Some(format!("version-{revision}")),
        capabilities: Some(serde_json::json!({"causal_commit": true})),
        revision,
        opened_at_ms: 100,
        updated_at_ms: 100 + revision,
        closed_at_ms: None,
    }
}

fn job(revision: u64, state: ControlJobState) -> ControlJobRecord {
    ControlJobRecord {
        job_id: "job-1".into(),
        session_id: Some("session".into()),
        run_id: None,
        attempt_id: None,
        binding_id: Some("binding-1".into()),
        epoch: Some(7),
        kind: ControlJobKind::WriteCandidate,
        effect_scope: ControlEffectScope::CandidateOnly,
        state,
        application_state: None,
        operation_digest: Some("job-digest".into()),
        idempotency_key: Some("job-key".into()),
        candidate_ref: Some("artifact://candidate".into()),
        result_ref: None,
        workspace_version: Some("version-1".into()),
        error: None,
        revision,
        created_at_ms: 110,
        updated_at_ms: 110 + revision,
        cancel_requested_at_ms: None,
        finished_at_ms: state.is_terminal().then_some(110 + revision),
    }
}

fn operation(binding_id: &str) -> WorkspaceOperationRecord {
    WorkspaceOperationRecord {
        operation_id: "operation-1".into(),
        binding_id: binding_id.into(),
        epoch: 7,
        session_id: Some("session".into()),
        run_id: None,
        attempt_id: None,
        job_id: Some("job-1".into()),
        kind: "candidate_apply".into(),
        replay_class: OperationReplayClass::PureWorkspace,
        status: WorkspaceOperationStatus::Admitted,
        operation_digest: "operation-digest".into(),
        idempotency_key: "operation-key".into(),
        base_version: Some("version-1".into()),
        result_version: None,
        execution_ref: None,
        settlement_ref: None,
        error: None,
        revision: 1,
        created_at_ms: 120,
        updated_at_ms: 120,
        settled_at_ms: None,
    }
}

#[test]
fn projection_and_required_event_commit_atomically_and_retry_exactly() {
    let store = store();
    let transition =
        ProjectionTransition::WorkspaceBinding(binding(1, WorkspaceProjectionState::Ready));
    let event = required_event("binding opened");

    let first = store
        .commit_projection_event(transition.clone(), event.clone())
        .unwrap();
    let second = store.commit_projection_event(transition, event).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.event.sequence, 1);
    assert_eq!(store.max_event_sequence().unwrap(), 1);
    assert_eq!(
        store.get_workspace_binding("binding-1").unwrap(),
        Some(binding(1, WorkspaceProjectionState::Ready))
    );
}

#[test]
fn schema_and_projection_reopen_idempotently() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.sqlite3");
    {
        let store = ControlStore::open(&path).unwrap();
        store
            .commit_projection_event(
                ProjectionTransition::WorkspaceBinding(binding(1, WorkspaceProjectionState::Ready)),
                required_event("binding opened"),
            )
            .unwrap();
    }

    let reopened = ControlStore::open(&path).unwrap();
    assert_eq!(reopened.schema_version().unwrap(), CONTROL_SCHEMA_VERSION);
    assert_eq!(
        reopened.get_workspace_binding("binding-1").unwrap(),
        Some(binding(1, WorkspaceProjectionState::Ready))
    );
    assert_eq!(reopened.replay_events(0).unwrap().len(), 1);
}

#[test]
fn non_required_event_cannot_advance_a_projection() {
    let store = store();
    let event = RunEvent::new(
        EventKind::GitChanged,
        EventContext::default(),
        SemanticActivity {
            verb: ActivityVerb::Change,
            object: ActivityObject::Workspace,
            state: ActivityState::Running,
            group_key: "workspace".into(),
            title: "best effort".into(),
            detail: None,
            refs: vec![],
            progress: None,
        },
    );

    assert!(matches!(
        store.commit_projection_event(
            ProjectionTransition::WorkspaceBinding(binding(1, WorkspaceProjectionState::Ready)),
            event
        ),
        Err(ControlError::Invalid(_))
    ));
    assert_eq!(store.max_event_sequence().unwrap(), 0);
    assert!(store.get_workspace_binding("binding-1").unwrap().is_none());
}

#[test]
fn projection_failure_rolls_back_the_event() {
    let store = store();
    let event = required_event("orphan operation");
    let event_id = event.event_id.clone();

    assert!(matches!(
        store.commit_projection_event(
            ProjectionTransition::WorkspaceOperation(operation("missing-binding")),
            event
        ),
        Err(ControlError::Database(_))
    ));
    assert_eq!(store.max_event_sequence().unwrap(), 0);
    assert!(
        store
            .replay_events(0)
            .unwrap()
            .iter()
            .all(|event| event.event_id != event_id)
    );
    assert!(
        store
            .get_workspace_operation("operation-1")
            .unwrap()
            .is_none()
    );
}

#[test]
fn stale_revision_rolls_back_its_event() {
    let store = store();
    store
        .commit_projection_event(
            ProjectionTransition::WorkspaceBinding(binding(1, WorkspaceProjectionState::Ready)),
            required_event("binding opened"),
        )
        .unwrap();
    let stale_event = required_event("skipped revision");

    assert!(matches!(
        store.commit_projection_event(
            ProjectionTransition::WorkspaceBinding(binding(3, WorkspaceProjectionState::Settling)),
            stale_event
        ),
        Err(ControlError::Invalid(_))
    ));
    assert_eq!(store.max_event_sequence().unwrap(), 1);
    assert_eq!(
        store.get_workspace_binding("binding-1").unwrap(),
        Some(binding(1, WorkspaceProjectionState::Ready))
    );
}

#[test]
fn an_event_id_cannot_be_reused_for_another_transition() {
    let store = store();
    let event = required_event("binding opened");
    store
        .commit_projection_event(
            ProjectionTransition::WorkspaceBinding(binding(1, WorkspaceProjectionState::Ready)),
            event.clone(),
        )
        .unwrap();
    let mut changed = binding(1, WorkspaceProjectionState::Mutating);
    changed.workspace_version = Some("different".into());

    assert!(matches!(
        store.commit_projection_event(ProjectionTransition::WorkspaceBinding(changed), event),
        Err(ControlError::Invalid(_))
    ));
    assert_eq!(store.max_event_sequence().unwrap(), 1);
}

#[test]
fn all_v2_projections_round_trip_and_are_queryable() {
    let store = store();
    store
        .commit_projection_event(
            ProjectionTransition::WorkspaceBinding(binding(1, WorkspaceProjectionState::Ready)),
            required_event("binding"),
        )
        .unwrap();
    let queued_job = job(1, ControlJobState::Queued);
    store
        .commit_projection_event(
            ProjectionTransition::Job(queued_job.clone()),
            required_event("job"),
        )
        .unwrap();
    let operation = operation("binding-1");
    store
        .commit_projection_event(
            ProjectionTransition::WorkspaceOperation(operation.clone()),
            required_event("operation"),
        )
        .unwrap();
    let recovery = WorkspaceRecoveryRecord {
        recovery_id: "recovery-1".into(),
        binding_id: Some("binding-1".into()),
        workspace_id: "workspace".into(),
        session_id: Some("session".into()),
        operation_id: Some("operation-1".into()),
        job_id: Some("job-1".into()),
        kind: "ambiguous_commit".into(),
        status: WorkspaceRecoveryStatus::Required,
        digest: Some("pending-digest".into()),
        artifact_ref: Some("artifact://pending".into()),
        detail: None,
        error: None,
        revision: 1,
        created_at_ms: 130,
        updated_at_ms: 130,
        resolved_at_ms: None,
    };
    store
        .commit_projection_event(
            ProjectionTransition::WorkspaceRecovery(recovery.clone()),
            required_event("recovery"),
        )
        .unwrap();
    let snapshot = SessionSnapshotRecord {
        snapshot_id: "snapshot-1".into(),
        session_id: "session".into(),
        reducer_version: 2,
        through_sequence: 4,
        state_ref: "artifact://state".into(),
        state_digest: "state-digest".into(),
        state_bytes: 512,
        revision: 1,
        created_at_ms: 140,
    };
    store
        .commit_projection_event(
            ProjectionTransition::SessionSnapshot(snapshot.clone()),
            required_event("snapshot"),
        )
        .unwrap();

    assert_eq!(store.get_job("job-1").unwrap(), Some(queued_job.clone()));
    assert_eq!(
        store.get_workspace_operation("operation-1").unwrap(),
        Some(operation)
    );
    assert_eq!(
        store.get_workspace_recovery("recovery-1").unwrap(),
        Some(recovery.clone())
    );
    assert_eq!(
        store.jobs_for_binding("binding-1").unwrap(),
        vec![queued_job]
    );
    assert_eq!(store.unsettled_jobs().unwrap().len(), 1);
    assert_eq!(
        store.recoveries_for_workspace("workspace").unwrap(),
        vec![recovery]
    );
    assert_eq!(
        store.latest_session_snapshot("session").unwrap(),
        Some(snapshot)
    );
}

#[test]
fn terminal_job_records_are_immutable() {
    let store = store();
    store
        .commit_projection_event(
            ProjectionTransition::WorkspaceBinding(binding(1, WorkspaceProjectionState::Ready)),
            required_event("binding"),
        )
        .unwrap();
    store
        .commit_projection_event(
            ProjectionTransition::Job(job(1, ControlJobState::Running)),
            required_event("running"),
        )
        .unwrap();
    store
        .commit_projection_event(
            ProjectionTransition::Job(job(2, ControlJobState::Succeeded)),
            required_event("succeeded"),
        )
        .unwrap();

    assert!(store.unsettled_jobs().unwrap().is_empty());
    assert!(matches!(
        store.commit_projection_event(
            ProjectionTransition::Job(job(3, ControlJobState::Failed)),
            required_event("late failure")
        ),
        Err(ControlError::Invalid(_))
    ));
    assert_eq!(store.max_event_sequence().unwrap(), 3);
    assert_eq!(
        store.get_job("job-1").unwrap().unwrap().state,
        ControlJobState::Succeeded
    );
}

#[test]
fn store_uses_schema_v2_with_foreign_keys_enabled() {
    let store = store();
    assert_eq!(store.schema_version().unwrap(), CONTROL_SCHEMA_VERSION);
    let connection = store.lock().unwrap();
    let enabled: i64 = connection
        .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
        .unwrap();
    assert_eq!(enabled, 1);
}
