use std::fs;
use std::sync::Arc;

use hi_control::{
    ControlStore, JournaledWorkspaceController, WorkspaceOperationStatus, WorkspaceRecoveryStatus,
};
use hi_workspace::{InMemoryWorkspaceController, MutationIntent, WorkspaceController};

use super::*;

#[test]
fn whole_workspace_scan_tracks_content_including_vcs_and_excludes_only_runtime() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().canonicalize().unwrap();
    let state = workspace.join(".hi/state");
    fs::create_dir_all(&state).unwrap();
    fs::create_dir_all(workspace.join(".git")).unwrap();
    fs::write(workspace.join("source.txt"), b"one").unwrap();
    fs::write(state.join("journal-noise"), b"first").unwrap();
    fs::write(workspace.join(".git/HEAD"), b"first").unwrap();

    let first = scan_workspace(&workspace, &state).unwrap();
    fs::write(state.join("journal-noise"), b"second").unwrap();
    let excluded_changes = scan_workspace(&workspace, &state).unwrap();
    assert_eq!(first.digest, excluded_changes.digest);

    fs::write(workspace.join(".git/HEAD"), b"second").unwrap();
    let vcs_change = scan_workspace(&workspace, &state).unwrap();
    assert_ne!(first.digest, vcs_change.digest);

    fs::write(workspace.join("source.txt"), b"two").unwrap();
    let source_change = scan_workspace(&workspace, &state).unwrap();
    assert_ne!(first.digest, source_change.digest);
    assert!(
        first
            .exclusions
            .iter()
            .any(|value| value.contains("runtime state"))
    );
}

#[tokio::test]
async fn restart_inventory_uses_stable_id_and_discard_preserves_current_bytes() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().canonicalize().unwrap();
    let state = workspace.join(".hi/state");
    fs::create_dir_all(&state).unwrap();
    fs::write(workspace.join("source.txt"), b"accepted current bytes").unwrap();
    let workspace_id = workspace_id(&workspace);
    let store = ControlStore::open_for_state(&state).unwrap();
    let inner: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        workspace_id.clone(),
        &workspace,
        &state,
    ));
    let controller = JournaledWorkspaceController::attach_store(inner, store.clone()).unwrap();
    let permit = controller
        .begin(MutationIntent::workspace("interrupted edit"))
        .await
        .unwrap();
    let operation_id = permit.record().operation_id.to_string();
    let binding = controller.binding();
    drop(permit);

    let service = LocalRecoveryService::new(workspace.clone(), state.clone());
    let inventory = service.inventory().unwrap();
    assert_eq!(inventory.len(), 1);
    let recovery = &inventory[0];
    let expected = restart_operation_recovery_id(
        &binding.binding_id,
        binding.epoch,
        &OperationId::new(operation_id.clone()),
    );
    assert_eq!(recovery.recovery_id, expected.to_string());
    assert!(!recovery.retry_safe);
    assert!(!recovery.process_reaping_proven);
    assert_eq!(
        recovery.operation.as_ref().unwrap().operation_id,
        operation_id
    );
    let confirmation = recovery
        .proof
        .confirmation_digest
        .clone()
        .expect("workspace is scannable");
    assert!(
        service
            .discard(&recovery.recovery_id, "blake3:wrong")
            .is_err()
    );

    let bytes_before = fs::read(workspace.join("source.txt")).unwrap();
    let receipt = service
        .discard(&recovery.recovery_id, &confirmation)
        .unwrap();
    assert_eq!(receipt.recovery_id, expected);
    assert_eq!(
        fs::read(workspace.join("source.txt")).unwrap(),
        bytes_before
    );
    let operation = store
        .get_workspace_operation(&operation_id)
        .unwrap()
        .unwrap();
    assert_eq!(operation.status, WorkspaceOperationStatus::Failed);
    let persisted = store
        .get_workspace_recovery(expected.as_str())
        .unwrap()
        .unwrap();
    assert_eq!(persisted.status, WorkspaceRecoveryStatus::Discarded);
    let detail = persisted.detail.unwrap();
    assert!(detail.contains("process reaping"));
    assert!(detail.contains("not inferred"));
    assert!(service.inventory().unwrap().is_empty());
    assert!(
        store
            .unsettled_workspace_bindings(&workspace_id)
            .unwrap()
            .is_empty(),
        "the next startup must be able to create a fresh binding"
    );
}

#[test]
fn confirmation_is_bound_to_recovery_identity() {
    let binding = WorkspaceBindingRecord {
        binding_id: "binding".into(),
        workspace_id: "workspace".into(),
        session_id: None,
        epoch: 4,
        authority: WorkspaceAuthority::Local,
        state: hi_control::WorkspaceProjectionState::RecoveryRequired,
        workspace_version: None,
        capabilities: None,
        revision: 1,
        opened_at_ms: 1,
        updated_at_ms: 1,
        closed_at_ms: None,
    };
    let target = Target {
        recovery_id: RecoveryId::new("one"),
        binding: binding.clone(),
        operation: None,
        job: Some(ControlJobRecord {
            job_id: "job".into(),
            session_id: None,
            run_id: None,
            attempt_id: None,
            binding_id: Some(binding.binding_id.clone()),
            epoch: Some(binding.epoch),
            kind: ControlJobKind::WriteCandidate,
            effect_scope: ControlEffectScope::CandidateOnly,
            state: hi_control::ControlJobState::RecoveryRequired,
            application_state: None,
            operation_digest: None,
            idempotency_key: None,
            candidate_ref: Some("artifact://candidate".into()),
            result_ref: None,
            workspace_version: None,
            error: None,
            revision: 1,
            created_at_ms: 1,
            updated_at_ms: 1,
            cancel_requested_at_ms: None,
            finished_at_ms: None,
        }),
        evidence: Vec::new(),
    };
    let mut other = target.clone();
    other.recovery_id = RecoveryId::new("two");
    assert_ne!(
        confirmation_digest("workspace", &target, "blake3:bytes"),
        confirmation_digest("workspace", &other, "blake3:bytes")
    );
}

#[test]
fn local_retry_is_rejected_and_discard_requires_quiescence_acknowledgement() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().canonicalize().unwrap();
    let state = workspace.join(".hi/state");
    fs::create_dir_all(&state).unwrap();
    let retry = WorkspaceCommand::Recover {
        command: RecoveryCommand::Retry(super::super::InspectArgs {
            recovery_id: "stable-id".into(),
            session: None,
            json: false,
        }),
    };
    assert!(
        run_at(retry, workspace.clone(), state.clone())
            .unwrap_err()
            .to_string()
            .contains("cannot prove the old writer was reaped")
    );
    let discard = WorkspaceCommand::Recover {
        command: RecoveryCommand::Discard(super::super::DiscardArgs {
            recovery_id: "stable-id".into(),
            session: None,
            confirm: "blake3:proof".into(),
            accept_current_bytes: false,
        }),
    };
    assert!(
        run_at(discard, workspace, state)
            .unwrap_err()
            .to_string()
            .contains("external writers are stopped")
    );
}
