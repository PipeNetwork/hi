use std::fs;

use super::*;

fn test_cache_scope() -> PipeFsCacheScope {
    PipeFsClient::new(crate::PipeFsClientConfig::new(
        "http://127.0.0.1:1",
        "test-key",
    ))
    .unwrap()
    .cache_scope()
}

fn test_workspace(temporary: &tempfile::TempDir, session_id: &str) -> PipeFsWorkspace {
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    PipeFsWorkspace::new(
        PipeFsClient::new(crate::PipeFsClientConfig::new(
            "http://127.0.0.1:1",
            "test-key",
        ))
        .unwrap(),
        PipeFsLease {
            token: "token".to_string(),
            generation: 1,
        },
        PipeFsWorkspaceConfig {
            session_id: session_id.to_string(),
            cache_scope: test_cache_scope(),
            original_workspace_root: workspace_root,
            original_state_root: state_root,
            cache_base: Some(temporary.path().join("cache")),
        },
    )
    .unwrap()
}

fn operation(operation_id: &str) -> CausalOperationReceipt {
    CausalOperationReceipt {
        operation_id: operation_id.to_string(),
        idempotency_key: format!("idempotency-{operation_id}"),
        binding_id: "binding-1".to_string(),
        binding_epoch: 1,
        replay_class: hi_workspace::ReplayClass::PureWorkspace,
        execution: hi_workspace::ExecutionReport::succeeded(None),
    }
}

#[tokio::test]
async fn terminal_process_keeps_recovery_marker_until_final_checkpoint() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "background-marker-test");
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Clean;
        workspace.persist_locked(&state).unwrap();
    }

    workspace
        .background_process_state("process-1", true)
        .await
        .unwrap();
    workspace
        .background_process_state("process-1", false)
        .await
        .unwrap();
    assert!(
        workspace.inner.recovery_marker.is_file(),
        "process exit must retain byte-recovery evidence before settlement"
    );
    {
        let mut state = workspace.inner.state.lock().await;
        assert!(state.active_background_processes.is_empty());
        assert!(state.background_reconciliation_pending);
        workspace.clear_recovery_marker_if_safe(&state);
        assert!(workspace.inner.recovery_marker.is_file());
        workspace
            .acknowledge_background_reconciliation_locked(&mut state)
            .unwrap();
    }
    assert!(!workspace.inner.recovery_marker.exists());
}

#[tokio::test]
async fn later_terminal_generation_survives_older_transcript_acknowledgement() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "terminal-generation-race-test");
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.transcript_cursor = Some(3);
        state.pending_compatibility = Some(PendingCompatibilityOperation {
            operation: operation("older-checkpoint"),
            minimum_transcript_cursor: Some(4),
            background_terminal_generation: Some(0),
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(
        &workspace.inner.recovery_marker,
        b"checkpoint awaiting transcript acknowledgement\n",
    )
    .unwrap();

    // This terminal observation lands after the staged checkpoint returned to
    // the transcript flusher but before that flusher acknowledges it.
    workspace
        .background_process_state("late-writer", true)
        .await
        .unwrap();
    workspace
        .background_process_state("late-writer", false)
        .await
        .unwrap();
    workspace
        .finish_compatibility_checkpoint("older-checkpoint", 4)
        .await
        .unwrap();

    let state = workspace.inner.state.lock().await;
    assert_eq!(state.background_terminal_generation, 1);
    assert!(state.background_reconciliation_pending);
    assert!(workspace.inner.recovery_marker.is_file());
    drop(state);
    assert!(
        workspace
            .controller_mutation_started(Some(vec!["new-write.txt".into()]), false)
            .await
            .is_err(),
        "a newer terminal writer must close ordinary controller-v2 admission"
    );
}

#[tokio::test]
async fn later_terminal_generation_survives_older_causal_acknowledgement() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "causal-terminal-generation-race-test");
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.pending_causal = Some(PendingCausalOperation {
            operation: operation("older-causal-checkpoint"),
            transcript_records: Vec::new(),
            receipt: Some(CausalCommitReceipt {
                head: None,
                manifest_digest: None,
                transcript_cursor: 6,
                operation_id: "older-causal-checkpoint".into(),
                replayed: false,
            }),
            background_terminal_generation: Some(0),
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(
        &workspace.inner.recovery_marker,
        b"causal checkpoint awaiting transcript acknowledgement\n",
    )
    .unwrap();

    workspace
        .background_process_state("late-causal-writer", true)
        .await
        .unwrap();
    workspace
        .background_process_state("late-causal-writer", false)
        .await
        .unwrap();
    workspace
        .finish_causal_checkpoint("older-causal-checkpoint", 6)
        .await
        .unwrap();

    let state = workspace.inner.state.lock().await;
    assert_eq!(state.background_terminal_generation, 1);
    assert!(state.background_reconciliation_pending);
    assert!(workspace.inner.recovery_marker.is_file());
}

#[tokio::test]
async fn causal_recovery_acknowledgement_never_regresses_a_newer_transcript_cursor() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "causal-cursor-race-test");
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.transcript_cursor = Some(11);
        state.pending_causal = Some(PendingCausalOperation {
            operation: operation("older-causal-cursor"),
            transcript_records: Vec::new(),
            receipt: Some(CausalCommitReceipt {
                head: None,
                manifest_digest: None,
                transcript_cursor: 6,
                operation_id: "older-causal-cursor".into(),
                replayed: true,
            }),
            background_terminal_generation: Some(0),
        });
        workspace.persist_locked(&state).unwrap();
    }

    assert_eq!(workspace.status().await.transcript_cursor, Some(11));
    workspace
        .finish_causal_checkpoint("older-causal-cursor", 6)
        .await
        .unwrap();
    assert_eq!(workspace.status().await.transcript_cursor, Some(11));
}

#[tokio::test]
async fn legacy_pending_operation_cannot_clear_terminal_recovery_evidence() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "legacy-terminal-generation-test");
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.background_reconciliation_pending = true;
        state.pending_compatibility = Some(PendingCompatibilityOperation {
            operation: operation("legacy-checkpoint"),
            minimum_transcript_cursor: Some(1),
            background_terminal_generation: None,
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(
        &workspace.inner.recovery_marker,
        b"legacy checkpoint has unknown terminal coverage\n",
    )
    .unwrap();

    workspace
        .finish_compatibility_checkpoint("legacy-checkpoint", 7)
        .await
        .unwrap();

    let state = workspace.inner.state.lock().await;
    assert!(state.background_reconciliation_pending);
    assert!(workspace.inner.recovery_marker.is_file());
}

#[tokio::test]
async fn compatibility_boundary_retains_archive_until_acknowledgement() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "compatibility-ack-test");
    let head = Uuid::new_v4();
    let manifest = "b".repeat(64);
    let operation = CausalOperationReceipt {
        execution: hi_workspace::ExecutionReport::succeeded(Some(manifest.clone())),
        ..operation("operation-1")
    };
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.remote = Some(PipeFsRemoteStateDisk {
            enabled: true,
            current_head: Some(head),
            sequence: 1,
            manifest_digest: Some(manifest.clone()),
            logical_size_bytes: 7,
            restore_chain: Vec::new(),
        });
        state.pending = Some(PendingRevision {
            expected_base_revision_id: None,
            revision_type: RevisionKind::Full,
            archive_blake3: "a".repeat(64),
            archive_size_bytes: 6,
            manifest_digest: manifest,
            logical_size_bytes: 7,
            idempotency_key: "revision-key".into(),
            snapshot: Snapshot::default(),
            background_terminal_generation: Some(0),
        });
        state.pending_compatibility = Some(PendingCompatibilityOperation {
            operation: operation.clone(),
            minimum_transcript_cursor: Some(1),
            background_terminal_generation: Some(0),
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(&workspace.inner.pending_archive, b"staged").unwrap();
    write_private(&workspace.inner.recovery_marker, b"pending transcript\n").unwrap();

    assert_eq!(
        workspace
            .checkpoint_for_compatibility_transcript(operation)
            .await
            .unwrap(),
        Some(head)
    );
    assert!(workspace.inner.pending_archive.is_file());
    assert!(workspace.inner.recovery_marker.is_file());

    workspace
        .finish_compatibility_checkpoint("operation-1", 9)
        .await
        .unwrap();
    let state = workspace.inner.state.lock().await;
    assert_eq!(state.phase, WorkspacePhase::Clean);
    assert!(state.pending.is_none());
    assert!(state.pending_compatibility.is_none());
    assert!(!workspace.inner.pending_archive.exists());
    assert!(!workspace.inner.recovery_marker.exists());
}

#[tokio::test]
async fn compatibility_acknowledgement_must_advance_its_persisted_cursor_fence() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "compatibility-cursor-fence-test");
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.transcript_cursor = Some(9);
        state.pending_compatibility = Some(PendingCompatibilityOperation {
            operation: operation("cursor-fenced-operation"),
            minimum_transcript_cursor: Some(10),
            background_terminal_generation: Some(0),
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(&workspace.inner.pending_archive, b"staged").unwrap();
    write_private(
        &workspace.inner.recovery_marker,
        b"pending compatibility transcript acknowledgement\n",
    )
    .unwrap();

    for cursor in [8, 9] {
        let error = workspace
            .finish_compatibility_checkpoint("cursor-fenced-operation", cursor)
            .await
            .expect_err("a regressive or non-advancing cursor must be rejected");
        assert!(error.to_string().contains("does not advance"));
        let state = workspace.inner.state.lock().await;
        assert_eq!(state.phase, WorkspacePhase::Pending);
        assert_eq!(state.transcript_cursor, Some(9));
        assert!(state.pending_compatibility.is_some());
        assert!(workspace.inner.pending_archive.is_file());
        assert!(workspace.inner.recovery_marker.is_file());
    }

    workspace
        .finish_compatibility_checkpoint("cursor-fenced-operation", 10)
        .await
        .unwrap();
    let state = workspace.inner.state.lock().await;
    assert_eq!(state.phase, WorkspacePhase::Clean);
    assert_eq!(state.transcript_cursor, Some(10));
    assert!(state.pending_compatibility.is_none());
    assert!(!workspace.inner.pending_archive.exists());
    assert!(!workspace.inner.recovery_marker.exists());
}

#[tokio::test]
async fn legacy_compatibility_acknowledgement_without_cursor_fence_fails_closed() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = test_workspace(&temporary, "legacy-compatibility-cursor-test");
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.pending_compatibility = Some(PendingCompatibilityOperation {
            operation: operation("legacy-cursor-operation"),
            minimum_transcript_cursor: None,
            background_terminal_generation: None,
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(&workspace.inner.pending_archive, b"legacy staged").unwrap();
    write_private(
        &workspace.inner.recovery_marker,
        b"legacy compatibility recovery\n",
    )
    .unwrap();

    let error = workspace
        .finish_compatibility_checkpoint("legacy-cursor-operation", 100)
        .await
        .expect_err("legacy state without a cursor fence must fail closed");
    assert!(error.to_string().contains("no persisted cursor fence"));
    let state = workspace.inner.state.lock().await;
    assert_eq!(state.phase, WorkspacePhase::Pending);
    assert!(state.pending_compatibility.is_some());
    assert!(workspace.inner.pending_archive.is_file());
    assert!(workspace.inner.recovery_marker.is_file());
}
