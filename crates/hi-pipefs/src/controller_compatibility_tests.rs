use super::*;

#[test]
fn protocol_one_requires_the_explicit_current_client_compatibility_adapter() {
    let config = PipeFsControllerConfig {
        workspace_id: "workspace".into(),
        session_id: "session".into(),
        writer_protocol: 1,
        causal_commit_available: false,
        writes_available: true,
        workspace_root: "/work".into(),
        state_root: "/state".into(),
        epoch: 0,
        allow_protocol_one_writes: false,
    };
    assert_eq!(config.writer_mode(), PipeFsWriterMode::ReadOnly);
    let mut server_read_only = config.clone();
    server_read_only.writer_protocol = 2;
    server_read_only.causal_commit_available = true;
    server_read_only.writes_available = false;
    assert_eq!(server_read_only.writer_mode(), PipeFsWriterMode::ReadOnly);
    let mut legacy = config;
    legacy.allow_protocol_one_writes = true;
    assert_eq!(legacy.writer_mode(), PipeFsWriterMode::Compatibility);

    legacy.writer_protocol = 2;
    assert_eq!(legacy.writer_mode(), PipeFsWriterMode::ReadOnly);
}

#[tokio::test]
async fn compatibility_flush_ambiguity_blocks_until_typed_recovery() {
    let (temporary, source, session, server) = subject(false).await;
    let controller = compatibility_controller(&source, session.clone()).await;
    let capabilities = controller.capabilities();
    assert!(!capabilities.causal_commit);
    assert!(capabilities.candidate_apply);
    assert!(!capabilities.background_writers);

    session
        .fail_compatibility_flush
        .store(true, Ordering::SeqCst);
    let permit = controller
        .begin(MutationIntent::workspace("legacy foreground"))
        .await
        .unwrap();
    let candidate = controller
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "isolated compatibility candidate".into(),
            limits: JobLimits::default(),
            parent_operation: Some(permit.record().operation_id.clone()),
        })
        .await
        .unwrap();
    assert_eq!(
        controller
            .seal_job(
                candidate.job_id,
                JobTerminal {
                    completion: JobCompletion::Failed,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await
            .status,
        JobSealStatus::Sealed
    );
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::TranscriptPending);
    assert_eq!(controller.status().state, WorkspaceState::TranscriptPending);
    let pending_status = source.inner.workspace.status().await;
    assert!(pending_status.transcript_pending);
    let controller_state = find_file(temporary.path(), "controller.json");
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&controller_state).unwrap()).unwrap();
    assert!(persisted["pending_compatibility"].is_object());
    assert!(
        controller
            .begin(MutationIntent::workspace("blocked"))
            .await
            .is_err()
    );
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 0);

    // Simulate a process crash after workspace CAS but before transcript
    // acknowledgement. A fresh typed controller must reconstruct recovery
    // from the PipeFS cache instead of dead-ending on the old journal state.
    drop(controller);
    let restarted = compatibility_controller(&source, session.clone()).await;
    assert_eq!(restarted.status().state, WorkspaceState::TranscriptPending);

    session
        .fail_compatibility_flush
        .store(false, Ordering::SeqCst);
    let recovered = restarted
        .reconcile(restarted.status().recovery_id.unwrap())
        .await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(session.compatibility_flushes.load(Ordering::SeqCst), 1);
    assert_eq!(restarted.status().state, WorkspaceState::Ready);
    let recovered_status = source.inner.workspace.status().await;
    assert!(!recovered_status.transcript_pending);
    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(controller_state).unwrap()).unwrap();
    assert!(persisted["pending_compatibility"].is_null());
}
