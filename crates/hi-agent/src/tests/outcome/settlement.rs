use super::*;

#[tokio::test]
async fn provider_failure_returns_agent_owned_cleanup_receipt() {
    let workspace = IsolatedWorkspace::new("typed-failure-cleanup");
    let mut cfg = workspace.config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.suggest_next_prompt = false;
    let (mut agent, _) = scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], cfg);
    let error = agent.run_turn("fail once", &mut NullUi).await.unwrap_err();
    let failure =
        crate::TurnFailure::from_error(&error).expect("Agent must return the cleanup receipt");
    assert_eq!(failure.outcome.status, TurnStatus::Failed);
    assert!(!failure.settlement_pending, "{failure:#}");
    assert!(failure.cleanup_diagnostics.is_empty(), "{failure:#}");
    assert!(
        failure
            .original
            .downcast_ref::<hi_ai::ProviderError>()
            .is_some()
    );
    assert!(agent.workspace.active_turn_background_baseline.is_none());
    assert!(agent.workspace.active_turn_task_baseline.is_none());
    assert!(agent.workspace.active_turn_ledger_revision.is_none());
    assert_eq!(agent.turn_phase(), TurnPhase::Done);
}

#[tokio::test(start_paused = true)]
async fn cancellation_deadline_is_shared_and_never_restarts() {
    let cancellation = TurnCancellation::new();
    let deadline = cancellation.settlement_deadline();
    let clone = cancellation.clone();
    tokio::time::advance(std::time::Duration::from_secs(59)).await;
    clone.cancel();
    assert_eq!(clone.settlement_deadline(), deadline);
    assert_eq!(cancellation.settlement_deadline(), deadline);
    assert_eq!(
        deadline.saturating_duration_since(tokio::time::Instant::now()),
        std::time::Duration::from_secs(1)
    );
}
