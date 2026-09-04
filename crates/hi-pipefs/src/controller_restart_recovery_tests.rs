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
