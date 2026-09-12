use super::common::*;
use super::*;

#[test]
fn natural_build_continuation_uses_implementation_guards() {
    for prompt in [
        "review plan.md and lets keep building this",
        "continue building this",
        "keep implementing the active plan",
    ] {
        assert!(
            classify_implementation_intent(prompt).is_some(),
            "explicit continuation should keep implementing: {prompt:?}"
        );
    }
    assert_eq!(
        classify_implementation_intent("consider what we should keep building, but do not edit"),
        None
    );
}

#[tokio::test]
async fn early_unique_discovery_keeps_inspection_tools() {
    let workspace = IsolatedWorkspace::new("fresh-discovery-retains-tools");
    let mut responses = Vec::new();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    // Two unique files stay under the unique-inspection edit challenge.
    for file in 0..2 {
        let relative = format!("src/context-{file}.rs");
        std::fs::write(
            workspace.path(&relative),
            format!("pub const CONTEXT_{file}: usize = {file};\n"),
        )
        .unwrap();
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("read-{file}"),
                name: "read".into(),
                arguments: serde_json::json!({"path": relative}).to_string(),
            }],
            1,
            1,
        ));
    }
    let changed = workspace.path("src/implemented.rs");
    responses.push(write_completion(&changed.to_string_lossy()));
    responses.push(completion(vec![Content::Text("implemented".into())], 1, 1));

    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(responses),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("review the implementation and fix the bug", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert!(changed.exists());
    let read_entries = agent
        .last_turn_telemetry()
        .tool_timeline
        .iter()
        .filter(|entry| entry.tool == "read")
        .collect::<Vec<_>>();
    assert_eq!(read_entries.len(), 2);
    assert!(
        read_entries
            .iter()
            .all(|entry| entry.status == hi_tools::ToolStatus::Succeeded)
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("without editing"))
    );
    let recorded_tools = tool_names.lock().unwrap();
    for tools in recorded_tools.iter().take(3) {
        assert!(tools.iter().any(|name| name == "read"));
        assert!(tools.iter().any(|name| name == "bash"));
    }
    assert!(
        modes
            .lock()
            .unwrap()
            .iter()
            .take(3)
            .all(|mode| *mode == ToolMode::Auto)
    );
}
