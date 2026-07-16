use super::common::*;
use super::*;

#[tokio::test]
async fn program_question_gets_repository_tools_in_dynamic_mode() {
    let workspace = IsolatedWorkspace::new("dynamic-program-question");
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"sample\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "inspect".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "Cargo.toml"}).to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "This is a Rust package named sample, currently at version 0.1.0.".into(),
                )],
                1,
                1,
            ),
        ]),
        tool_names: tool_names.clone(),
        modes,
    };
    let mut config = workspace.config();
    config.tool_set = ToolSet::Dynamic;
    config.long_horizon = true;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();

    let outcome = agent
        .run_turn("what does this program do", &mut RecUi::default())
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    let first_request = &tool_names.lock().unwrap()[0];
    for expected in ["read", "list", "grep", "glob"] {
        assert!(
            first_request.iter().any(|name| name == expected),
            "missing {expected} from dynamic tools: {first_request:?}"
        );
    }
    assert!(!first_request.iter().any(|name| name == "write"));
}
