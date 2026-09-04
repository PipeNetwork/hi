use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use hi_workspace::{
    ExecutionReport, InMemoryWorkspaceController, MutationIntent, WorkspaceController,
    WorkspaceState,
};

use super::WorkspaceCoordination;
use crate::WorkspaceDurability;

#[derive(Default)]
struct CountingDurability {
    admissions: AtomicUsize,
    checkpoints: AtomicUsize,
}

#[async_trait]
impl WorkspaceDurability for CountingDurability {
    async fn mutation_started(&self, _dirty_paths: Option<Vec<String>>) -> Result<()> {
        self.admissions.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn checkpoint(&self) -> Result<()> {
        self.checkpoints.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn disabled_controller_gate_uses_legacy_admission_but_keeps_status() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let mut harness = hi_workspace::ResolvedHarnessSettings::default();
    harness.features.workspace_controller_v2 = false;
    let subject = WorkspaceCoordination::new_local_with_settings(root.path(), &state, harness);
    let durability = Arc::new(CountingDurability::default());

    subject
        .begin(Some(durability.clone()), Some(vec!["changed.txt".into()]))
        .await
        .unwrap();
    assert_eq!(subject.status().state, WorkspaceState::Ready);
    subject
        .checkpoint(
            Some(durability.clone()),
            ExecutionReport::succeeded(Some("digest".into())),
        )
        .await
        .unwrap();
    assert_eq!(durability.admissions.load(Ordering::SeqCst), 1);
    assert_eq!(durability.checkpoints.load(Ordering::SeqCst), 1);
    assert_eq!(subject.status().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn disabled_controller_gate_never_bypasses_existing_recovery() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let mut harness = hi_workspace::ResolvedHarnessSettings::default();
    harness.features.workspace_controller_v2 = false;
    let subject = WorkspaceCoordination::new_local_with_settings(root.path(), &state, harness);
    let controller = Arc::new(InMemoryWorkspaceController::new_local(
        "recovery-test",
        root.path(),
        &state,
    ));
    let permit = controller
        .begin(MutationIntent::workspace("abandoned setup operation"))
        .await
        .unwrap();
    drop(permit);
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    subject.install_controller(controller).unwrap();

    let error = subject.begin(None, None).await.unwrap_err();
    assert!(error.to_string().contains("recovery remains required"));
    assert_eq!(subject.status().state, WorkspaceState::RecoveryRequired);
    assert!(subject.status().recovery_id.is_some());
}

#[tokio::test]
async fn disabled_controller_gate_denies_unfenced_pipefs_admission() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let mut harness = hi_workspace::ResolvedHarnessSettings::default();
    harness.features.workspace_controller_v2 = false;
    let subject = WorkspaceCoordination::new_local_with_settings(root.path(), &state, harness);
    let controller = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "rollback-test",
        "session",
        2,
        true,
        root.path(),
        &state,
    ));
    subject.install_controller(controller).unwrap();

    let error = subject.begin(None, None).await.unwrap_err();
    assert!(error.to_string().contains("legacy durability fence"));
    assert_eq!(subject.status().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn rollback_gate_never_bypasses_an_already_admitted_settlement() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let mut subject = WorkspaceCoordination::new_local(root.path(), &state);
    subject.begin(None, None).await.unwrap();
    assert_eq!(subject.status().state, WorkspaceState::Mutating);

    subject.harness.features.workspace_controller_v2 = false;
    subject
        .checkpoint(None, ExecutionReport::succeeded(Some("settled".into())))
        .await
        .unwrap();
    assert_eq!(subject.status().state, WorkspaceState::Ready);
    assert!(subject.status().active_operation.is_none());
}
