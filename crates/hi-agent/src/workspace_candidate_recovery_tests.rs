use std::sync::Arc;

use hi_workspace::{
    CandidateChange, CandidateFileKind, CandidateId, CandidatePostimage, CandidateRoute,
    CandidateVerification, EffectScope, InMemoryWorkspaceController, JobCompletion, JobId, JobKind,
    JobLimits, JobSealStatus, JobSpec, JobTerminal, VerifiedCandidate, VerifiedCandidateDraft,
    WorkspaceBinding, WorkspaceController,
};

use super::super::{WorkspaceCoordination, workspace_id};

fn persist_candidate(
    binding: &WorkspaceBinding,
    job_id: &JobId,
    state_root: &std::path::Path,
    label: &str,
) -> hi_tools::candidate_workspace::PersistedDetachedCandidate {
    let candidate = VerifiedCandidate::create(VerifiedCandidateDraft {
        candidate_id: CandidateId::new(format!("{label}-candidate")),
        job_id: job_id.clone(),
        source_binding_id: binding.binding_id.clone(),
        source_epoch: binding.epoch,
        base_version: binding.version.clone(),
        before_digest: "git:before".into(),
        after_digest: "git:after".into(),
        changes: vec![CandidateChange {
            path: format!("{label}.txt").into(),
            before: None,
            after: Some(CandidatePostimage::new(
                CandidateFileKind::Regular,
                0o644,
                b"candidate\n".to_vec(),
            )),
        }],
        verification: vec![CandidateVerification {
            name: "test".into(),
            passed: true,
            verifier_digest: "blake3:verification".into(),
            detail: None,
            artifacts: Vec::new(),
        }],
        destination_verification: vec![hi_workspace::CandidateDestinationVerifier {
            name: "test".into(),
            command: "true".into(),
            timeout_ms: 5_000,
        }],
        destination_verification_budget_ms: 5_000,
        artifacts: Vec::new(),
        effective_route: CandidateRoute {
            provider: "test".into(),
            model: "test".into(),
            actual_model_revision: None,
            capability_digest: "blake3:capabilities".into(),
        },
    })
    .unwrap();
    hi_tools::candidate_workspace::PersistedDetachedCandidate::persist(
        hi_tools::candidate_workspace::DetachedVerifiedCandidate {
            candidate,
            source_snapshot_id: format!("snapshot-{label}"),
        },
        state_root,
    )
    .unwrap()
}

#[tokio::test]
async fn restart_surfaces_fsynced_candidate_artifact_and_cleans_only_after_ack() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let store = hi_control::ControlStore::open_for_state(&state).unwrap();
    let raw: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        workspace_id(root.path()),
        root.path(),
        &state,
    ));
    let original =
        hi_control::JournaledWorkspaceController::attach_store(raw, store.clone()).unwrap();
    let binding = original.binding();
    let job = original
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "crash candidate".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();
    let persisted = persist_candidate(&binding, &job.job_id, &state, "crash");
    let artifact_path = persisted.path().to_path_buf();
    let artifact_uri = persisted.artifact.uri.clone();
    drop(persisted);
    drop(original);

    let restarted = WorkspaceCoordination::new_local(root.path(), &state);
    assert_eq!(
        restarted.status().state,
        hi_workspace::WorkspaceState::RecoveryRequired
    );
    assert_eq!(
        store
            .get_job(job.job_id.as_str())
            .unwrap()
            .unwrap()
            .candidate_ref,
        Some(artifact_uri.clone())
    );
    let recoveries = store
        .recoveries_for_workspace(&workspace_id(root.path()))
        .unwrap();
    assert!(
        recoveries
            .iter()
            .any(|recovery| recovery.artifact_ref.as_deref() == Some(&artifact_uri))
    );
    assert!(artifact_path.exists());

    restarted.reconcile_after_external_proof().await.unwrap();
    drop(restarted);
    let clean_restart = WorkspaceCoordination::new_local(root.path(), &state);
    assert_eq!(
        clean_restart.status().state,
        hi_workspace::WorkspaceState::Ready
    );
    assert!(!artifact_path.exists());
}

#[tokio::test]
async fn restart_retains_stale_candidate_for_review_or_rerun() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    std::fs::create_dir_all(&state).unwrap();
    let store = hi_control::ControlStore::open_for_state(&state).unwrap();
    let raw: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
        workspace_id(root.path()),
        root.path(),
        &state,
    ));
    let original =
        hi_control::JournaledWorkspaceController::attach_store(raw, store.clone()).unwrap();
    let binding = original.binding();
    let job = original
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "stale candidate".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();
    let persisted = persist_candidate(&binding, &job.job_id, &state, "stale");
    let artifact_path = persisted.path().to_path_buf();
    let artifact_uri = persisted.artifact.uri.clone();

    let ready = original
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::ReadyToMerge,
                detail: Some("candidate prepared".into()),
                artifacts: vec![persisted.artifact.clone()],
            },
        )
        .await;
    assert_eq!(ready.status, JobSealStatus::Sealed);
    let stale = original
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::Stale,
                detail: Some("base version changed".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(stale.status, JobSealStatus::Sealed);
    drop(persisted);
    drop(original);

    let restarted = WorkspaceCoordination::new_local(root.path(), &state);
    assert_eq!(
        restarted.status().state,
        hi_workspace::WorkspaceState::Ready
    );
    let persisted_job = store.get_job(job.job_id.as_str()).unwrap().unwrap();
    assert_eq!(persisted_job.state, hi_control::ControlJobState::Stale);
    assert_eq!(persisted_job.candidate_ref.as_deref(), Some(&*artifact_uri));
    assert!(
        artifact_path.exists(),
        "stale candidate evidence must survive restart for review or rerun"
    );
}

#[test]
fn restart_cleanup_requires_an_explicit_disposable_candidate_outcome() {
    use hi_control::ControlJobState;

    for disposable in [
        ControlJobState::Succeeded,
        ControlJobState::Failed,
        ControlJobState::Cancelled,
    ] {
        assert!(super::candidate_artifact_is_disposable(disposable));
    }
    for retained in [
        ControlJobState::Queued,
        ControlJobState::Starting,
        ControlJobState::Running,
        ControlJobState::ReadyToMerge,
        ControlJobState::Merging,
        ControlJobState::Settling,
        ControlJobState::CancelRequested,
        ControlJobState::DurabilityPending,
        ControlJobState::RecoveryRequired,
        ControlJobState::Orphaned,
        ControlJobState::Stale,
    ] {
        assert!(!super::candidate_artifact_is_disposable(retained));
    }
}
