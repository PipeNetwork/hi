use hi_workspace::ReplayClass;

use super::*;

#[tokio::test]
async fn causal_admission_drains_the_stable_transcript_prefix_before_execution() {
    let (_temporary, controller, session, _server) = subject(false).await;
    session.fail_preflight.store(true, Ordering::SeqCst);

    let denied = controller
        .begin(MutationIntent::workspace("must not execute yet"))
        .await
        .unwrap_err();
    assert_eq!(denied.reason, AdmissionDeniedReason::NotReady);
    assert_eq!(controller.status().state, WorkspaceState::TranscriptPending);
    assert!(controller.status().active_operation.is_none());
    assert!(controller.status().recovery_id.is_none());

    session.fail_preflight.store(false, Ordering::SeqCst);
    let permit = controller
        .begin(MutationIntent::workspace("safe after prefix flush"))
        .await
        .unwrap();
    assert_eq!(session.preflights.load(Ordering::SeqCst), 2);
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::NoChange);
}

#[tokio::test]
async fn protocol_one_nonreplayable_effect_is_cleanly_denied_before_admission() {
    let (_temporary, source, session, _server) = subject(false).await;
    let controller = compatibility_controller(&source, session).await;
    let denied = controller
        .begin(MutationIntent {
            effect_scope: EffectScope::LiveWriter,
            replay_class: ReplayClass::NonReplayableExternal,
            dirty_paths: None,
            description: Some("external publish".into()),
        })
        .await
        .unwrap_err();

    assert_eq!(denied.reason, AdmissionDeniedReason::CapabilityUnavailable);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert!(controller.status().active_operation.is_none());
    assert!(controller.status().recovery_id.is_none());
}
