use super::*;

#[tokio::test]
async fn matched_restart_retry_promotes_unmatched_fence_without_remote_replay() {
    let (_temporary, source, session, _server) = subject(false).await;
    let controller = compatibility_controller(&source, session.clone()).await;
    session
        .fail_compatibility_flush
        .store(true, Ordering::SeqCst);
    let permit = controller
        .begin(MutationIntent::workspace("interrupted publication"))
        .await
        .unwrap();
    let expected = hi_workspace::restart_operation_recovery_id(
        &permit.record().binding_id,
        permit.record().epoch,
        &permit.record().operation_id,
    );
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::TranscriptPending);
    drop(controller);

    let restarted = compatibility_controller(&source, session.clone()).await;
    assert_eq!(restarted.status().recovery_id.as_ref(), Some(&expected));
    let binding = restarted.binding();
    let unmatched = RecoveryId::new("unmatched-journal-operation");
    restarted
        .require_restart_recovery(RecoveryRecord {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            recovery_id: unmatched.clone(),
            kind: RecoveryKind::AbandonedMutation,
            binding_id: binding.binding_id,
            epoch: binding.epoch,
            operation_id: Some(OperationId::new("unmatched-operation")),
            job_id: None,
            detail: "unmatched journal operation remains blocked".into(),
            created_at_ms: 1,
            resolved: false,
        })
        .unwrap();
    assert_eq!(restarted.status().recovery_id.as_ref(), Some(&expected));

    session
        .fail_compatibility_flush
        .store(false, Ordering::SeqCst);
    let recovered = restarted.reconcile(expected).await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(session.compatibility_flushes.load(Ordering::SeqCst), 1);
    assert_eq!(restarted.status().state, WorkspaceState::RecoveryRequired);
    assert_eq!(restarted.status().recovery_id.as_ref(), Some(&unmatched));

    let rejected = restarted.reconcile(unmatched).await;
    assert_eq!(rejected.status, RecoveryStatus::Rejected);
    assert_eq!(session.compatibility_flushes.load(Ordering::SeqCst), 1);
    assert_eq!(restarted.status().state, WorkspaceState::RecoveryRequired);
}

#[tokio::test]
async fn journal_seeded_writer_recovery_terminalizes_without_replay_and_is_idempotent() {
    let (temporary, controller, _session, server) = subject(false).await;
    let binding = controller.binding();
    let job_id = JobId::new("journal-restored-writer");
    let recovery_id =
        hi_workspace::restart_job_recovery_id(&binding.binding_id, binding.epoch, &job_id);

    controller
        .require_restart_recovery(RecoveryRecord {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            recovery_id: recovery_id.clone(),
            kind: RecoveryKind::CrashedWriterJob,
            binding_id: binding.binding_id,
            epoch: binding.epoch,
            operation_id: None,
            job_id: Some(job_id.clone()),
            detail: "writer was active when the prior harness process stopped".into(),
            created_at_ms: 1,
            resolved: false,
        })
        .unwrap();

    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    assert_eq!(controller.status().active_jobs, vec![job_id]);
    assert!(find_file(temporary.path(), "recovery-required").is_file());

    let denied = controller
        .begin(MutationIntent::workspace("blocked during writer recovery"))
        .await
        .unwrap_err();
    assert_eq!(denied.reason, AdmissionDeniedReason::NotReady);
    assert!(denied.detail.contains("state=RecoveryRequired"));
    assert!(
        denied
            .detail
            .contains(&format!("recovery_id={recovery_id}"))
    );
    assert!(
        denied
            .detail
            .contains("writer was active when the prior harness process stopped")
    );

    let recovered = controller.reconcile(recovery_id.clone()).await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert!(
        recovered
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("finalized as failed"))
    );
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert!(controller.status().active_jobs.is_empty());
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 0);

    // The controller receipt is safe to replay after a lost response. It does
    // not execute or publish the interrupted job a second time, and the cache
    // marker remains until the outer durable journal has consumed the receipt.
    let replayed = controller.reconcile(recovery_id).await;
    assert_eq!(replayed.status, RecoveryStatus::Recovered);
    assert!(
        replayed
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("already resolved"))
    );
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 0);
    assert!(find_file(temporary.path(), "recovery-required").is_file());
}

#[tokio::test]
async fn restarted_journal_terminalizes_writer_exactly_once_after_pipefs_acknowledgement() {
    let (temporary, controller, session, server) = subject(false).await;
    let store = hi_control::ControlStore::open(temporary.path().join("events.sqlite3")).unwrap();
    let first: Arc<dyn WorkspaceController> = Arc::new(controller.clone());
    let first =
        hi_control::JournaledWorkspaceController::attach_store(first, store.clone()).unwrap();
    let job = first
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "candidate interrupted by process restart".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();
    let historical_binding = first.binding();
    drop(first);

    let journal = hi_control::WorkspaceProjectionJournal::from_control_store(store.clone());
    let report = journal
        .reconcile_jobs_after_restart(&historical_binding)
        .unwrap();
    assert_eq!(report.recovery_required, vec![job.job_id.clone()]);
    assert_eq!(report.recovery_ids.len(), 1);
    let recovery_id = report.recovery_ids[0].clone();
    let recovery = store
        .get_workspace_recovery(recovery_id.as_str())
        .unwrap()
        .unwrap();

    // A restarted controller has a fresh, empty in-memory registry. The
    // durable control record is its only lifecycle authority.
    let restarted = compatibility_controller(&controller, session).await;
    restarted
        .require_restart_recovery(RecoveryRecord {
            schema_version: WORKSPACE_CONTRACT_SCHEMA_VERSION,
            recovery_id: recovery_id.clone(),
            kind: RecoveryKind::CrashedWriterJob,
            binding_id: historical_binding.binding_id,
            epoch: historical_binding.epoch,
            operation_id: None,
            job_id: Some(job.job_id.clone()),
            detail: recovery.detail.unwrap(),
            created_at_ms: recovery.created_at_ms,
            resolved: false,
        })
        .unwrap();
    let restarted: Arc<dyn WorkspaceController> = Arc::new(restarted);
    let restarted =
        hi_control::JournaledWorkspaceController::attach_store(restarted, store.clone()).unwrap();

    let first_receipt = restarted.reconcile(recovery_id.clone()).await;
    assert_eq!(first_receipt.status, RecoveryStatus::Recovered);
    let terminal = store.get_job(job.job_id.as_str()).unwrap().unwrap();
    assert_eq!(terminal.state, hi_control::ControlJobState::Failed);
    let terminal_revision = terminal.revision;
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 0);

    // A lost response may make the host repeat recovery. The second receipt
    // is idempotent and cannot append another terminal job transition.
    let replayed = restarted.reconcile(recovery_id).await;
    assert_eq!(replayed.status, RecoveryStatus::Recovered);
    let still_terminal = store.get_job(job.job_id.as_str()).unwrap().unwrap();
    assert_eq!(still_terminal.state, hi_control::ControlJobState::Failed);
    assert_eq!(still_terminal.revision, terminal_revision);
    assert_eq!(server.causal_calls.load(Ordering::SeqCst), 0);
}
