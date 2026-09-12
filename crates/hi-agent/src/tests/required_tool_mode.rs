use super::common::*;
use super::*;

struct AutoOnlyRequiredProbe {
    requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for AutoOnlyRequiredProbe {
    async fn stream(
        &self,
        _: hi_ai::ChatRequest,
        _: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<hi_ai::Completion> {
        self.requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(completion(vec![Content::Text("narrative".into())], 1, 1))
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        let mut capabilities = hi_ai::ProviderCapabilities::native_tools(false);
        capabilities.tool_choice.required = false;
        capabilities
    }
}

struct SteeringRequiredProbe {
    requests: std::sync::atomic::AtomicUsize,
    modes: std::sync::Arc<Mutex<Vec<ToolMode>>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for SteeringRequiredProbe {
    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        _: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<hi_ai::Completion> {
        self.modes.lock().unwrap().push(request.profile.tool_mode);
        let round = self
            .requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match round {
            0..=1 => Ok(completion(
                vec![Content::ToolCall {
                    id: format!("read-{round}"),
                    name: "read".into(),
                    arguments: serde_json::json!({
                        "path": format!("src/context-{round}.rs")
                    })
                    .to_string(),
                }],
                1,
                1,
            )),
            2 => Ok(completion(
                vec![Content::Text("Let me edit the file now.".into())],
                1,
                1,
            )),
            _ => Err(hi_ai::ProviderError::new(
                ProviderErrorKind::PolicyBlocked,
                "end steering probe",
            )
            .into()),
        }
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        test_provider_capabilities()
    }
}

#[tokio::test]
async fn unsupported_explicit_required_mode_fails_before_provider_send() {
    let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let provider = AutoOnlyRequiredProbe {
        requests: requests.clone(),
    };
    let mut cfg = config();
    cfg.routing.tool_mode = hi_ai::ToolMode::Required;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();

    let error = agent
        .run_turn("Implement the requested change.", &mut NullUi)
        .await
        .expect_err("unsupported Required policy must fail closed");

    assert_eq!(
        hi_ai::provider_error_kind(&error),
        Some(ProviderErrorKind::UnsupportedTools)
    );
    assert_eq!(requests.load(std::sync::atomic::Ordering::Relaxed), 0);
    let failure = crate::TurnFailure::from_error(&error).expect("settled failure receipt");
    assert_eq!(failure.outcome.status, TurnStatus::Failed);
    assert!(
        agent.last_assistant_text().is_some(),
        "failure has a durable closeout"
    );
    assert_ne!(agent.last_assistant_text().as_deref(), Some("narrative"));
}

#[tokio::test]
async fn required_mode_rejects_narration_without_weakening_the_request() {
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let responses = (0..MAX_TOOL_PROTOCOL_RETRIES)
        .map(|_| {
            completion(
                vec![Content::Text("Let me call the tool next.".into())],
                1,
                1,
            )
        })
        .collect();
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut cfg = config();
    cfg.routing.tool_mode = hi_ai::ToolMode::Required;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let error = agent
        .run_turn("Implement the requested change.", &mut ui)
        .await
        .expect_err("a required-tool narrative must not publish as success");

    assert_eq!(
        hi_ai::provider_error_kind(&error),
        Some(ProviderErrorKind::ToolProtocol)
    );
    let modes = modes.lock().unwrap();
    assert_eq!(
        modes.len(),
        MAX_TOOL_PROTOCOL_RETRIES as usize,
        "the consecutive protocol budget is {MAX_TOOL_PROTOCOL_RETRIES} sends, not one extra after exhaustion"
    );
    assert_eq!(
        agent.last_turn_usage().input_tokens,
        u64::from(MAX_TOOL_PROTOCOL_RETRIES)
    );
    assert_eq!(
        agent.last_turn_usage().output_tokens,
        u64::from(MAX_TOOL_PROTOCOL_RETRIES)
    );
    assert_eq!(
        agent.last_turn_telemetry().accepted_completions,
        0,
        "rejected narration must not establish accepted occupancy"
    );
    assert!(modes.iter().all(|mode| *mode == hi_ai::ToolMode::Required));
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("plain-text tool-call parsing")),
        "explicit Required mode must never fall back to Auto: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn fresh_discovery_does_not_turn_narration_into_a_protocol_error() {
    let workspace = IsolatedWorkspace::new("required-mutation-repair");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    for round in 0..2 {
        std::fs::write(
            workspace.path(format!("src/context-{round}.rs")),
            format!("pub const CONTEXT_{round}: usize = {round};\n"),
        )
        .unwrap();
    }
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = SteeringRequiredProbe {
        requests: std::sync::atomic::AtomicUsize::new(0),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), workspace.config()).unwrap();
    let mut ui = RecUi::default();

    let error = agent
        .run_turn(
            "Fix the parser bug after inspecting the implementation.",
            &mut ui,
        )
        .await
        .expect_err("the probe's terminal provider error should end the turn");

    assert_eq!(
        hi_ai::provider_error_kind(&error),
        Some(ProviderErrorKind::PolicyBlocked)
    );
    let modes = modes.lock().unwrap();
    assert_eq!(modes.len(), 4);
    assert_eq!(
        &modes[2..],
        &[ToolMode::Auto, ToolMode::Auto],
        "investigation must not force a Required contract"
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("invalid tool turn")),
        "valid narration must not trigger protocol recovery: {:?}",
        ui.statuses
    );
}
