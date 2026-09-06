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
async fn bounded_discovery_seals_a_mutation_only_recovery_round() {
    let workspace = IsolatedWorkspace::new("bounded-discovery-forces-mutation");
    let mut responses = Vec::new();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    // Twelve distinct reads reproduce the weak-model behavior from the live
    // session without relying on exact-call repeat detection. The first ten
    // spend the ordinary discovery budget; two bounded advisory rounds remain.
    for file in 0..12 {
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
    assert_eq!(read_entries.len(), 12);
    assert!(
        read_entries
            .iter()
            .all(|entry| entry.status == hi_tools::ToolStatus::Succeeded)
    );
    assert!(ui.statuses.iter().any(|status| {
        status.contains("mutation request used 12 model rounds (12 tools) without editing")
    }));
    let recorded_tools = tool_names.lock().unwrap();
    let recovery_tools = &recorded_tools[12];
    assert!(!recovery_tools.is_empty());
    assert!(recovery_tools.iter().all(|name| {
        hi_tools::tool_metadata(name)
            .is_some_and(|metadata| metadata.capability == hi_tools::ToolCapability::Mutation)
    }));
    assert!(
        !recovery_tools
            .iter()
            .any(|name| name == "read" || name == "grep" || name == "bash")
    );
    assert_eq!(modes.lock().unwrap()[12], ToolMode::Required);
}
