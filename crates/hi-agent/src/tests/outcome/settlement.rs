use super::*;

struct PreservedMtimeMutationUi {
    path: std::path::PathBuf,
}

impl Ui for PreservedMtimeMutationUi {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {}
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {
        let modified = std::fs::metadata(&self.path).unwrap().modified().unwrap();
        std::fs::write(&self.path, "changed\n").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&self.path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
    }
}

#[tokio::test]
async fn final_ui_edit_with_preserved_size_and_mtime_invalidates_verification() {
    let workspace = IsolatedWorkspace::new("same-stamp-final-ui");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let write = completion(
        vec![Content::ToolCall {
            id: "write".into(),
            name: "write".into(),
            arguments: serde_json::json!({"path":"work.rs","content":"checked\n"}).to_string(),
        }],
        1,
        1,
    );
    let mut agent = agent(
        vec![write, completion(vec![Content::Text("done".into())], 1, 1)],
        cfg,
    );
    let mut ui = PreservedMtimeMutationUi {
        path: workspace.path("work.rs"),
    };
    let outcome = agent.run_turn("implement work.rs", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert!(outcome.verified_workspace_revision.is_none());
    assert!(outcome.changed_files.contains(&"work.rs".to_string()));
    assert_eq!(std::fs::read_to_string(&ui.path).unwrap(), "changed\n");
}

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
