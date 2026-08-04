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
    config.memory.tool_set = ToolSet::Dynamic;
    config.subagents.long_horizon = true;
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
    assert!(
        !first_request.iter().any(|name| name == "bash_output"),
        "fresh read-only turns must not advertise background polling: {first_request:?}"
    );
    assert!(!first_request.iter().any(|name| name == "bash_kill"));
    assert!(!first_request.iter().any(|name| name == "write"));
}

#[tokio::test]
async fn fresh_status_does_not_advertise_background_polling() {
    let workspace = IsolatedWorkspace::new("dynamic-status-background-poll");
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![completion(
            vec![Content::Text("No background work is running.".into())],
            1,
            1,
        )]),
        tool_names: tool_names.clone(),
        modes: std::sync::Arc::new(Mutex::new(Vec::new())),
    };
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Dynamic;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();

    agent
        .run_turn("status", &mut RecUi::default())
        .await
        .unwrap();

    let requests = tool_names.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].iter().any(|name| name == "bash_output"),
        "fresh status request tools: {:?}",
        requests[0]
    );
    assert!(!requests[0].iter().any(|name| name == "bash_kill"));
}

#[tokio::test]
async fn explicit_no_edit_batched_review_uses_the_lean_catalog_on_first_request() {
    let workspace = IsolatedWorkspace::new("dynamic-explicit-no-edit-review");
    for path in [
        "crates/hi-ai/src/openai/request.rs",
        "crates/hi-ai/src/openai/stream.rs",
    ] {
        let file = workspace.path(path);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, "DeepSeek tool-calling context\n").unwrap();
    }
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "batch-read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({
                        "paths": [
                            "crates/hi-ai/src/openai/request.rs",
                            "crates/hi-ai/src/openai/stream.rs"
                        ]
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "- request.rs builds DeepSeek requests.\n- stream.rs parses DeepSeek responses.".into(),
                )],
                1,
                1,
            ),
        ]),
        tool_names: tool_names.clone(),
        modes: std::sync::Arc::new(Mutex::new(Vec::new())),
    };
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Dynamic;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();

    let outcome = agent
        .run_turn(
            "Read crates/hi-ai/src/openai/request.rs and crates/hi-ai/src/openai/stream.rs in one batched read call using the paths array. Return exactly two concise bullets about DeepSeek tool calling. Do not edit files.",
            &mut RecUi::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    let requests = tool_names.lock().unwrap();
    assert_eq!(requests[0], vec!["read"]);
}

#[tokio::test]
async fn bounded_exact_review_switches_to_text_after_first_evidence_pass() {
    let workspace = IsolatedWorkspace::new("bounded-exact-review-text-follow-up");
    for path in [
        "crates/hi-ai/src/openai/request.rs",
        "crates/hi-ai/src/openai/stream.rs",
    ] {
        let file = workspace.path(path);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, "DeepSeek tool-calling context\n").unwrap();
    }
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "bounded-read".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({
                        "paths": [
                            "crates/hi-ai/src/openai/request.rs",
                            "crates/hi-ai/src/openai/stream.rs"
                        ]
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "No concrete bug found in the bounded first pass.".into(),
                )],
                1,
                1,
            ),
        ]),
        tool_names: tool_names.clone(),
        modes: std::sync::Arc::new(Mutex::new(Vec::new())),
    };
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Dynamic;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();

    let outcome = agent
        .run_turn(
            "Review only crates/hi-ai/src/openai/request.rs and crates/hi-ai/src/openai/stream.rs for one concrete bug. Start with one batched read call using the paths array. Do not edit files.",
            &mut RecUi::default(),
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    let requests = tool_names.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0], vec!["read", "grep"]);
    assert!(
        requests[1].is_empty(),
        "bounded exact-file review should request a text-only follow-up: {:?}",
        requests[1]
    );
}
