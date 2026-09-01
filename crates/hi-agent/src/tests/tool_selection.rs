use super::common::*;
use super::*;

#[tokio::test]
async fn exact_text_response_contract_advertises_no_tools() {
    let workspace = IsolatedWorkspace::new("exact-text-no-tools");
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![completion(vec![Content::Text("ok".into())], 1, 1)]),
        tool_names: tool_names.clone(),
        modes,
    };
    let mut config = workspace.config();
    // The literal-response fast path must win even over an explicitly broad
    // catalog; this is an output contract, not a repository task.
    config.memory.tool_set = ToolSet::Full;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("Reply with exactly: ok", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(ui.assistant, "ok");
    assert_eq!(
        tool_names.lock().unwrap().as_slice(),
        &[Vec::<String>::new()]
    );
}

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
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
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
        modes: modes.clone(),
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
    let modes = modes.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0], vec!["read", "grep"]);
    assert_eq!(
        requests[1], requests[0],
        "wrap-up keeps the same catalog for prefix cache: {:?}",
        requests[1]
    );
    assert_eq!(
        modes[1],
        ToolMode::ChatOnly,
        "bounded exact-file review follow-up is chat-only: {modes:?}"
    );
}

#[tokio::test]
async fn bare_review_codebase_first_request_advertises_inspection_tools() {
    let workspace = IsolatedWorkspace::new("bare-review-codebase-first-request");
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"sample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(workspace.path("README.md"), "# sample\n").unwrap();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn ready() {}\n").unwrap();

    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let thinking = std::sync::Arc::new(Mutex::new(Vec::new()));
    let finished = Content::Text(
        "Findings:\n- Workspace is a small Rust crate.\n\nLimits:\n- Limited to inspected evidence."
            .into(),
    );
    struct Rec {
        responses: Mutex<Vec<Completion>>,
        tool_names: std::sync::Arc<Mutex<Vec<Vec<String>>>>,
        modes: std::sync::Arc<Mutex<Vec<ToolMode>>>,
        thinking: std::sync::Arc<Mutex<Vec<Option<bool>>>>,
    }
    #[async_trait::async_trait]
    impl hi_ai::Provider for Rec {
        async fn stream(
            &self,
            request: hi_ai::ChatRequest,
            _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
        ) -> anyhow::Result<Completion> {
            self.tool_names
                .lock()
                .unwrap()
                .push(request.tools.iter().map(|tool| tool.name.clone()).collect());
            self.modes.lock().unwrap().push(request.profile.tool_mode);
            self.thinking
                .lock()
                .unwrap()
                .push(request.profile.deepseek_thinking);
            pop_canned_completion(&self.responses, "Rec")
        }
    }
    let provider = Rec {
        responses: Mutex::new(
            (0..12)
                .map(|_| completion(vec![finished.clone()], 1, 1))
                .collect(),
        ),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
        thinking: thinking.clone(),
    };
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Dynamic;
    config.gates.read_only_preflight = true;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("review codebase", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    let requests = tool_names.lock().unwrap();
    let modes = modes.lock().unwrap();
    assert!(
        !requests.is_empty(),
        "review codebase should send at least one model request; statuses={:?}",
        ui.statuses
    );
    let first = &requests[0];
    assert!(
        !first.is_empty(),
        "first review-codebase request must advertise inspection tools, got {first:?} modes={modes:?} statuses={:?}",
        ui.statuses
    );
    for expected in ["read", "grep", "repo_map"] {
        assert!(
            first.iter().any(|name| name == expected),
            "missing {expected} from first review-codebase request: {first:?}"
        );
    }
    assert_ne!(
        modes[0],
        ToolMode::ChatOnly,
        "first review-codebase request must not be chat-only; modes={modes:?} requests={requests:?} statuses={:?}",
        ui.statuses
    );
    assert_eq!(
        thinking.lock().unwrap()[0],
        Some(false),
        "DeepSeek inspection pass should disable thinking; statuses={:?}",
        ui.statuses
    );
    if requests.len() > 1 {
        assert!(
            !requests[1].is_empty(),
            "citation-repair follow-up must keep inspection tools so the model can cite preflight evidence: {:?}",
            requests[1]
        );
        assert_ne!(
            modes[1],
            ToolMode::ChatOnly,
            "citation repair must not strip tools; modes={modes:?} statuses={:?}",
            ui.statuses
        );
    }
}

#[tokio::test]
async fn bare_review_codebase_wraps_up_chat_only_after_two_inspection_rounds() {
    let workspace = IsolatedWorkspace::new("bare-review-codebase-wrap-up");
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"sample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(workspace.path("README.md"), "# sample\n").unwrap();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn ready() {}\n").unwrap();

    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "read-1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"Cargo.toml"}"#.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read-2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"src/lib.rs"}"#.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "Findings:\n- Cargo.toml and src/lib.rs were inspected.\n\nLimits:\n- Limited to inspected evidence."
                        .into(),
                )],
                1,
                1,
            ),
        ]),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Dynamic;
    config.gates.read_only_preflight = true;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();

    let outcome = agent
        .run_turn("review codebase", &mut RecUi::default())
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    let requests = tool_names.lock().unwrap();
    let modes = modes.lock().unwrap();
    assert!(
        requests.len() >= 3,
        "expected preflight-backed inspection plus wrap-up, got {requests:?}"
    );
    assert!(
        !requests[0].is_empty(),
        "first inspection request should advertise tools: {:?}",
        requests[0]
    );
    assert!(
        !requests[1].is_empty(),
        "second inspection request should advertise tools: {:?}",
        requests[1]
    );
    let last = requests.last().expect("wrap-up request");
    assert!(
        !last.is_empty(),
        "wrap-up keeps the inspection catalog for prefix cache: {last:?}"
    );
    assert_eq!(*modes.last().expect("wrap-up mode"), ToolMode::ChatOnly);
}

#[tokio::test]
async fn bare_review_citation_repair_after_wrap_up_keeps_inspection_tools() {
    let workspace = IsolatedWorkspace::new("bare-review-citation-repair-keeps-tools");
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"sample\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(workspace.path("README.md"), "# sample\n").unwrap();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn ready() {}\n").unwrap();

    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "read-1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"Cargo.toml"}"#.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::ToolCall {
                    id: "read-2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"src/lib.rs"}"#.into(),
                }],
                1,
                1,
            ),
            // Wrap-up with no citation of inspected paths → ConcreteAnswer repair.
            completion(
                vec![Content::Text(
                    "The codebase looks generally healthy with no obvious issues.".into(),
                )],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "Findings:\n- Cargo.toml and src/lib.rs were inspected.\n\nLimits:\n- Limited to inspected evidence."
                        .into(),
                )],
                1,
                1,
            ),
        ]),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Dynamic;
    config.gates.read_only_preflight = true;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();

    let outcome = agent
        .run_turn("review codebase", &mut RecUi::default())
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    let requests = tool_names.lock().unwrap();
    let modes = modes.lock().unwrap();
    assert!(
        requests.len() >= 4,
        "inspection, wrap-up, and citation-repair: {requests:?}"
    );
    assert_eq!(
        modes.get(2).copied(),
        Some(ToolMode::ChatOnly),
        "first wrap-up is still chat-only: {modes:?}"
    );
    let repair = requests.get(3).expect("citation-repair request");
    assert!(
        !repair.is_empty(),
        "citation-repair after wrap-up must keep inspection tools: requests={requests:?} modes={modes:?}"
    );
    assert_ne!(
        modes.get(3).copied(),
        Some(ToolMode::ChatOnly),
        "citation-repair must not be pinned chat-only: {modes:?}"
    );
}
