use super::*;

#[tokio::test]
async fn advancing_lease_generation_acknowledges_a_lost_commit_response() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    let committed_head = Uuid::new_v4();
    let manifest = "b".repeat(64);
    let client = client_serving_remote_state(
        "lease-lost-ack-test",
        Some(committed_head),
        Some(&manifest),
        7,
    )
    .await;
    let cache_scope = client.cache_scope();
    let workspace = PipeFsWorkspace::new(
        client,
        PipeFsLease {
            token: "old-token".to_string(),
            generation: 4,
        },
        PipeFsWorkspaceConfig {
            session_id: "lease-lost-ack-test".to_string(),
            cache_scope,
            original_workspace_root: workspace_root,
            original_state_root: state_root,
            cache_base: Some(temporary.path().join("cache")),
        },
    )
    .unwrap();
    let committed_snapshot = Snapshot {
        logical_size_bytes: 7,
        manifest_digest: Some(manifest.clone()),
        ..Snapshot::default()
    };
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.pending = Some(PendingRevision {
            expected_base_revision_id: None,
            revision_type: RevisionKind::Full,
            archive_blake3: "a".repeat(64),
            archive_size_bytes: 6,
            manifest_digest: manifest.clone(),
            logical_size_bytes: 7,
            idempotency_key: "old-generation".to_string(),
            snapshot: committed_snapshot.clone(),
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(&workspace.inner.pending_archive, b"staged").unwrap();

    workspace
        .update_lease(PipeFsLease {
            token: "new-token".to_string(),
            generation: 5,
        })
        .await
        .unwrap();

    let state = workspace.inner.state.lock().await;
    assert_eq!(state.phase, WorkspacePhase::Dirty);
    assert!(state.pending.is_none());
    assert_eq!(state.snapshot.as_ref(), Some(&committed_snapshot));
    assert_eq!(
        state.remote.as_ref().and_then(|remote| remote.current_head),
        Some(committed_head)
    );
    assert!(workspace.inner.recovery_marker.is_file());
    assert!(!workspace.inner.pending_archive.exists());
}

#[tokio::test]
async fn advancing_lease_generation_preserves_pending_evidence_without_remote_proof() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    let client = PipeFsClient::new(crate::PipeFsClientConfig::new(
        "http://127.0.0.1:1",
        "test-key",
    ))
    .unwrap();
    let workspace = PipeFsWorkspace::new(
        client.clone(),
        PipeFsLease {
            token: "old-token".to_string(),
            generation: 4,
        },
        PipeFsWorkspaceConfig {
            session_id: "lease-no-proof-test".to_string(),
            cache_scope: client.cache_scope(),
            original_workspace_root: workspace_root,
            original_state_root: state_root,
            cache_base: Some(temporary.path().join("cache")),
        },
    )
    .unwrap();
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.pending = Some(PendingRevision {
            expected_base_revision_id: None,
            revision_type: RevisionKind::Full,
            archive_blake3: "a".repeat(64),
            archive_size_bytes: 6,
            manifest_digest: "b".repeat(64),
            logical_size_bytes: 0,
            idempotency_key: "old-generation".to_string(),
            snapshot: Snapshot::default(),
        });
        workspace.persist_locked(&state).unwrap();
    }
    write_private(&workspace.inner.pending_archive, b"staged").unwrap();

    workspace
        .update_lease(PipeFsLease {
            token: "new-token".to_string(),
            generation: 5,
        })
        .await
        .expect_err("remote proof is required before discarding Pending");

    let state = workspace.inner.state.lock().await;
    assert_eq!(state.phase, WorkspacePhase::Pending);
    assert!(state.pending.is_some());
    assert!(workspace.inner.pending_archive.is_file());
    assert!(workspace.inner.recovery_marker.is_file());
}

#[tokio::test]
async fn dirty_lease_loss_always_writes_discoverable_recovery_marker() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    let workspace = PipeFsWorkspace::new(
        PipeFsClient::new(crate::PipeFsClientConfig::new(
            "http://127.0.0.1:1",
            "test-key",
        ))
        .unwrap(),
        PipeFsLease {
            token: "token".to_string(),
            generation: 4,
        },
        PipeFsWorkspaceConfig {
            session_id: "dirty-lease-loss-test".to_string(),
            cache_scope: test_cache_scope(),
            original_workspace_root: workspace_root,
            original_state_root: state_root,
            cache_base: Some(temporary.path().join("cache")),
        },
    )
    .unwrap();
    {
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Dirty;
        state.dirty_paths.insert("changed.txt".to_string());
        workspace.persist_locked(&state).unwrap();
    }
    assert!(!workspace.inner.recovery_marker.exists());

    workspace.mark_lease_lost("taken over").await.unwrap();

    assert!(workspace.inner.recovery_marker.is_file());
    assert_eq!(
        workspace.inner.state.lock().await.phase,
        WorkspacePhase::LeaseLost
    );
}

#[tokio::test]
async fn removes_only_stale_clean_cache_represented_by_remote_head() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    let make_workspace = |generation| {
        PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "http://127.0.0.1:1",
                "test-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: format!("token-{generation}"),
                generation,
            },
            PipeFsWorkspaceConfig {
                session_id: "clean-cache-test".to_string(),
                cache_scope: test_cache_scope(),
                original_workspace_root: workspace_root.clone(),
                original_state_root: state_root.clone(),
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap()
    };
    let old = make_workspace(1);
    let head = Uuid::new_v4();
    let remote = PipeFsRemoteState {
        session_id: "clean-cache-test".to_string(),
        enabled: true,
        current_head: Some(head),
        sequence: 1,
        manifest_digest: Some("a".repeat(64)),
        logical_size_bytes: 0,
        restore_chain: Vec::new(),
    };
    {
        let mut state = old.inner.state.lock().await;
        let materialized = old.inner.cache_root.join("workspace-clean");
        create_private_dir(&materialized).unwrap();
        state.phase = WorkspacePhase::Clean;
        state.remote = Some((&remote).into());
        state.snapshot = Some(Snapshot {
            manifest_digest: remote.manifest_digest.clone(),
            ..Snapshot::default()
        });
        state.materialized_root = Some(materialized);
        old.persist_locked(&state).unwrap();
    }
    let old_root = old.inner.cache_root.clone();
    let current = make_workspace(2);

    current.cleanup_stale_clean_caches(&remote);

    assert!(!old_root.exists());
    assert!(current.inner.cache_root.exists());
}

#[tokio::test]
async fn same_generation_restart_preserves_clean_controller_with_unscanned_drift() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    let config = PipeFsWorkspaceConfig {
        session_id: "same-generation-drift-test".to_string(),
        cache_scope: test_cache_scope(),
        original_workspace_root: workspace_root,
        original_state_root: state_root,
        cache_base: Some(temporary.path().join("cache")),
    };
    let make_workspace = || {
        PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "http://127.0.0.1:1",
                "test-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: "same-token".to_string(),
                generation: 7,
            },
            config.clone(),
        )
        .unwrap()
    };
    let old = make_workspace();
    let old_materialized = old.inner.cache_root.join("workspace-old");
    create_private_dir(&old_materialized).unwrap();
    let committed_snapshot = scan_workspace(&old_materialized).unwrap();
    {
        let mut state = old.inner.state.lock().await;
        state.phase = WorkspacePhase::Clean;
        state.snapshot = Some(committed_snapshot);
        state.materialized_root = Some(old_materialized.clone());
        old.persist_locked(&state).unwrap();
    }

    // Simulate a direct editor write followed by SIGKILL, after the
    // controller had most recently persisted `Clean`.
    fs::write(old_materialized.join("unfenced-change"), "keep me").unwrap();
    let old_cache = old.inner.cache_root.clone();
    drop(old);

    let resumed = make_workspace();
    assert_ne!(resumed.inner.cache_root, old_cache);
    assert!(old_cache.join("recovery-required").is_file());
    assert_eq!(
        fs::read_to_string(old_materialized.join("unfenced-change")).unwrap(),
        "keep me"
    );
    assert!(
        recovery_caches(
            &resumed.inner.session_cache_root,
            Some(&resumed.inner.cache_root)
        )
        .contains(&old_cache)
    );
}

#[tokio::test]
async fn stale_clean_cache_with_drift_is_marked_instead_of_deleted() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    let make_workspace = |generation| {
        PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "http://127.0.0.1:1",
                "test-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: format!("token-{generation}"),
                generation,
            },
            PipeFsWorkspaceConfig {
                session_id: "stale-drift-test".to_string(),
                cache_scope: test_cache_scope(),
                original_workspace_root: workspace_root.clone(),
                original_state_root: state_root.clone(),
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap()
    };
    let old = make_workspace(1);
    let materialized = old.inner.cache_root.join("workspace-old");
    create_private_dir(&materialized).unwrap();
    let snapshot = scan_workspace(&materialized).unwrap();
    let remote = PipeFsRemoteState {
        session_id: "stale-drift-test".to_string(),
        enabled: true,
        current_head: Some(Uuid::new_v4()),
        sequence: 1,
        manifest_digest: Some("a".repeat(64)),
        logical_size_bytes: 0,
        restore_chain: Vec::new(),
    };
    {
        let mut state = old.inner.state.lock().await;
        state.phase = WorkspacePhase::Clean;
        state.remote = Some((&remote).into());
        state.snapshot = Some(snapshot);
        state.materialized_root = Some(materialized.clone());
        old.persist_locked(&state).unwrap();
    }
    fs::write(materialized.join("late-write"), "preserve").unwrap();
    let old_cache = old.inner.cache_root.clone();
    drop(old);
    let current = make_workspace(2);

    current.cleanup_stale_clean_caches(&remote);

    assert!(old_cache.exists());
    assert!(old_cache.join("recovery-required").is_file());
    assert_eq!(
        fs::read_to_string(materialized.join("late-write")).unwrap(),
        "preserve"
    );
}

#[cfg(unix)]
#[test]
fn recovery_discovery_never_follows_generation_or_marker_symlinks() {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir().unwrap();
    let session_root = temporary.path().join("session");
    let outside = temporary.path().join("outside");
    fs::create_dir_all(&session_root).unwrap();
    fs::create_dir_all(&outside).unwrap();
    fs::write(outside.join("recovery-required"), b"outside\n").unwrap();

    symlink(&outside, session_root.join("linked-generation")).unwrap();

    let linked_marker = session_root.join("linked-marker");
    fs::create_dir_all(&linked_marker).unwrap();
    symlink(
        outside.join("recovery-required"),
        linked_marker.join("recovery-required"),
    )
    .unwrap();

    let genuine = session_root.join("genuine");
    fs::create_dir_all(&genuine).unwrap();
    fs::write(genuine.join("recovery-required"), b"recover\n").unwrap();

    assert_eq!(recovery_caches(&session_root, None), vec![genuine]);
}

#[tokio::test]
async fn rejects_workspace_larger_than_server_capability_before_staging() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace_root = temporary.path().join("workspace");
    let state_root = temporary.path().join("state");
    fs::create_dir_all(&workspace_root).unwrap();
    fs::create_dir_all(&state_root).unwrap();
    fs::write(workspace_root.join("large"), b"0123456789").unwrap();
    let workspace = PipeFsWorkspace::new(
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
            session_id: "workspace-limit-test".to_string(),
            cache_scope: test_cache_scope(),
            original_workspace_root: workspace_root.clone(),
            original_state_root: state_root,
            cache_base: Some(temporary.path().join("cache")),
        },
    )
    .unwrap();
    let mut state = workspace.inner.state.lock().await;
    state.phase = WorkspacePhase::Dirty;
    state.materialized_root = Some(workspace_root);
    state.snapshot = Some(Snapshot::default());
    state.capabilities = Some(CapabilitiesDisk {
        maximum_revision_bytes: 1_000_000,
        maximum_workspace_bytes: 5,
        maximum_delta_chain: 20,
        writes_available: true,
        restore_available: true,
        causal_commit_available: false,
    });
    workspace.persist_locked(&state).unwrap();
    drop(state);

    let error = workspace.checkpoint().await.unwrap_err();

    assert!(error.to_string().contains("exceeding the server limit"));
    let state = workspace.inner.state.lock().await;
    assert!(state.pending.is_none());
    assert!(!workspace.inner.pending_archive.exists());
    assert!(
        fs::read_dir(&workspace.inner.cache_root)
            .unwrap()
            .filter_map(Result::ok)
            .all(|entry| !entry.file_name().to_string_lossy().ends_with(".tar.zst"))
    );
}
