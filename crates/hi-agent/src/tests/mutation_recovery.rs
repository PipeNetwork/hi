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
async fn productive_discovery_continues_to_plan_instead_of_stalling() {
    let workspace = IsolatedWorkspace::new("productive-discovery-continues");
    let mut responses = Vec::new();
    let mut file = 0;
    // Exact batch cardinalities from the live failure: fourteen productive
    // rounds and thirty-three calls. The advisory bound must not terminate it.
    for batch_size in [2, 3, 3, 3, 3, 3, 4, 2, 1, 1, 2, 2, 2, 2] {
        let mut calls = Vec::new();
        for _ in 0..batch_size {
            let relative = format!("src/context-{file}.rs");
            let path = workspace.path(&relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(
                &path,
                format!("pub const CONTEXT_{file}: usize = {file};\n"),
            )
            .unwrap();
            calls.push(Content::ToolCall {
                id: format!("read-{file}"),
                name: "read".into(),
                arguments: serde_json::json!({"path": relative}).to_string(),
            });
            file += 1;
        }
        responses.push(completion(calls, 1, 1));
    }
    responses.push(completion(
        vec![Content::ToolCall {
            id: "plan-active".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Implement the selected component", "status": "active"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    let post_plan = workspace.path("src/post-plan.rs");
    std::fs::write(&post_plan, "final targeted context\n").unwrap();
    responses.push(completion(
        vec![Content::ToolCall {
            id: "post-plan-read".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "src/post-plan.rs"}).to_string(),
        }],
        1,
        1,
    ));
    let changed = workspace.path("src/implemented.rs");
    responses.push(write_completion(&changed.to_string_lossy()));
    responses.push(completion(
        vec![Content::ToolCall {
            id: "plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Implement the selected component", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    responses.push(bash_completion("cargo test --help"));
    responses.push(completion(vec![Content::Text("implemented".into())], 1, 1));

    let mut cfg = workspace.config();
    cfg.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
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
        .run_turn("review plan.md and lets keep building this", &mut ui)
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
    assert_eq!(read_entries.len(), 34);
    assert!(
        read_entries
            .iter()
            .all(|entry| entry.status == hi_tools::ToolStatus::Succeeded)
    );
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|status| status.contains("requesting an implementation step"))
            .count(),
        2,
        "discovery nudges are bounded and advisory: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("turn stopped incomplete"))
    );
    assert!(!agent.last_turn_telemetry().stalled_unfinished);
    let continued_tools = tool_names.lock().unwrap()[14]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(continued_tools.contains("read"));
    assert!(continued_tools.contains("update_plan"));
    assert!(continued_tools.contains("write"));
    assert_eq!(modes.lock().unwrap()[14], ToolMode::Auto);
    assert!(
        tool_names.lock().unwrap()[15]
            .iter()
            .any(|name| name == "read")
    );
    assert_eq!(modes.lock().unwrap()[15], ToolMode::Required);
    assert_eq!(modes.lock().unwrap()[16], ToolMode::Required);
}
