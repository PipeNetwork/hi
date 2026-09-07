use super::*;
use crate::provider::{ProviderError, ProviderErrorKind};
use crate::types::{RateLimitBucket, RateLimitState, RequestProfile, ToolMode, ToolSpec};
use serde_json::json;
use std::sync::Mutex;

#[derive(Clone)]
struct RecordingProvider {
    requests: Arc<Mutex<Vec<ChatRequest>>>,
    results: Arc<Mutex<Vec<Result<Completion, String>>>>,
    models: Vec<ServedModel>,
}

impl RecordingProvider {
    fn new(results: Vec<Result<Completion, String>>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(Mutex::new(results)),
            models: Vec::new(),
        }
    }

    fn with_models(mut self, models: Vec<ServedModel>) -> Self {
        self.models = models;
        self
    }

    fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Provider for RecordingProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        self.requests.lock().unwrap().push(request);
        sink(StreamEvent::Text("visible".to_string()));
        match self.results.lock().unwrap().remove(0) {
            Ok(completion) => Ok(completion),
            Err(message) => Err(ProviderError::new(ProviderErrorKind::Other, message).into()),
        }
    }

    async fn list_models(&self) -> Result<Vec<ServedModel>> {
        Ok(self.models.clone())
    }
}

fn request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        request_id: None,
        retry_attempt: 0,
        execution: Default::default(),
        user_turn: true,
        canonical_objective: Some("fix this".into()),
        messages: Arc::new(vec![
            Message::system("secret system"),
            Message::user("fix this"),
            Message::assistant(vec![Content::ToolCall {
                id: "call-1".into(),
                name: "read".into(),
                arguments: "{\"path\":\"src/main.rs\"}".into(),
            }]),
            Message::tool_result("call-1", "x".repeat(20)),
        ]),
        tools: Arc::from([ToolSpec {
            name: "read".into(),
            description: "read a file".into(),
            parameters: json!({"type":"object"}),
        }]),
        tool_envelope: None,
        max_tokens: 8192,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        thinking_budget: Some(1024),
        reasoning_effort: None,
        profile: RequestProfile::default(),
    }
}

fn completion(text: &str, input: u64, output: u64) -> Completion {
    Completion {
        content: vec![Content::Text(text.to_string())],
        usage: Usage {
            input_tokens: input,
            output_tokens: output,
            context_occupancy: input,
            ..Usage::default()
        },
        stop_reason: Some("stop".into()),
        refusal: None,
        tool_call_channel: crate::ToolCallChannel::None,
    }
}

#[test]
fn reference_view_drops_system_and_flattens_tools() {
    let mut messages = request(MOA_MODEL_CONSERVATIVE).messages.as_ref().clone();
    messages.push(Message::user_with_image("see image", "abc123", "image/png"));
    let out = reference_messages(&messages, 5);
    assert!(out.iter().all(|m| m.text() != "secret system"));
    assert!(out.iter().all(|m| m.role != Role::Tool));
    let text = out.iter().map(Message::text).collect::<Vec<_>>().join("\n");
    assert!(text.contains("[assistant requested tool `read`"));
    assert!(text.contains("[tool result for `call-1`]"));
    assert!(text.contains("[truncated"));
    assert!(text.contains("[image omitted]"));
}

#[tokio::test]
async fn normal_model_requests_bypass_moa() {
    let passthrough = RecordingProvider::new(vec![Ok(completion("direct", 1, 2))]);
    let passthrough_handle = passthrough.clone();
    let routes = RecordingProvider::new(vec![]);
    let routes_handle = routes.clone();
    let provider = MoaProvider::new(
        Box::new(passthrough),
        Box::new(routes),
        MoaConfig::default(),
    )
    .unwrap();
    let mut events = Vec::new();
    let mut sink = |event| events.push(event);
    let out = provider
        .stream(request("ipop/coder-balanced"), &mut sink)
        .await
        .unwrap();
    assert_eq!(completion_text(&out.content), "direct");
    assert_eq!(passthrough_handle.requests().len(), 1);
    assert!(routes_handle.requests().is_empty());
}

#[tokio::test]
async fn aggregator_receives_original_tools_and_guidance() {
    let passthrough = RecordingProvider::new(vec![]);
    let routes = RecordingProvider::new(vec![
        Ok(completion("check the parser edge case", 10, 3)),
        Ok(completion("done", 4, 2)),
    ]);
    let routes_handle = routes.clone();
    let provider = MoaProvider::new(
        Box::new(passthrough),
        Box::new(routes),
        MoaConfig::default(),
    )
    .unwrap();
    let mut events = Vec::new();
    let mut sink = |event| match event {
        StreamEvent::Status(status) => events.push(format!("status:{status}")),
        StreamEvent::Text(text) => events.push(format!("text:{text}")),
        StreamEvent::Reasoning(text) => events.push(format!("reasoning:{text}")),
        StreamEvent::WireAudit(_) => events.push("wire_audit".into()),
        StreamEvent::Warning(warning) => events.push(format!("warning:{warning}")),
        StreamEvent::ToolCallDelta { .. } | StreamEvent::ProviderAttempt(_) => {}
    };
    let out = provider
        .stream(request(MOA_MODEL_CONSERVATIVE), &mut sink)
        .await
        .unwrap();

    assert_eq!(out.usage.input_tokens, 14);
    assert_eq!(out.usage.output_tokens, 5);
    assert_eq!(
        events,
        vec![
            "status:MoA reference: pipe/auto-coder".to_string(),
            "status:MoA aggregating: ipop/coder-balanced".to_string(),
            "text:visible".to_string()
        ]
    );

    let requests = routes_handle.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].model, MOA_REFERENCE_CONSERVATIVE);
    assert_eq!(requests[0].profile.tool_mode, ToolMode::ChatOnly);
    assert!(requests[0].tools.is_empty());
    assert!(requests[0].tool_envelope.is_some());
    assert_eq!(requests[0].max_tokens, 2048);
    assert_eq!(requests[0].thinking_budget, None);

    assert_eq!(requests[1].model, MOA_AGGREGATOR_CONSERVATIVE);
    assert_eq!(requests[1].tools.len(), 1);
    assert_eq!(requests[1].profile.tool_mode, ToolMode::Auto);
    let guidance = requests[1]
        .messages
        .iter()
        .find(|m| m.role == Role::User)
        .unwrap()
        .text();
    assert!(guidance.contains("check the parser edge case"));
    assert!(guidance.contains("[Private MoA guidance]"));
}

#[tokio::test]
async fn reference_failure_still_runs_aggregator() {
    let passthrough = RecordingProvider::new(vec![]);
    let routes = RecordingProvider::new(vec![
        Err("reference down".into()),
        Ok(completion("aggregate", 4, 2)),
    ]);
    let routes_handle = routes.clone();
    let provider = MoaProvider::new(
        Box::new(passthrough),
        Box::new(routes),
        MoaConfig::default(),
    )
    .unwrap();
    let mut sink = |_event| {};
    let out = provider
        .stream(request(MOA_MODEL_CONSERVATIVE), &mut sink)
        .await
        .unwrap();
    assert_eq!(completion_text(&out.content), "aggregate");
    let requests = routes_handle.requests();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].messages.iter().any(|m| {
        m.role == Role::User
            && m.text()
                .contains("Reference `pipe/auto-coder` was unavailable")
    }));
}

#[tokio::test]
async fn reference_recovery_cannot_spend_aggregator_reservation_or_refill_on_reentry() {
    use crate::test_support::{ChatStep, ScriptedOpenAiServer, ScriptedResponse};
    use crate::{OpenAiProvider, RequestExecution, RequestExecutionPolicy};

    struct RecoveringReference(OpenAiProvider);
    #[async_trait]
    impl Provider for RecoveringReference {
        async fn stream(
            &self,
            request: ChatRequest,
            sink: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            loop {
                request.execution.ensure_available()?;
                match self.0.stream(request.clone(), sink).await {
                    Err(err) if request.model == MOA_REFERENCE_CONSERVATIVE => {
                        if crate::provider_error_retryable(&err) == Some(false) {
                            return Err(err);
                        }
                    }
                    result => return result,
                }
            }
        }

        async fn list_models(&self) -> Result<Vec<ServedModel>> {
            Ok(vec![])
        }
    }

    for max_attempts in [1, 4] {
        let mut steps: Vec<_> = (0..max_attempts)
            .map(|_| {
                ChatStep::new(ScriptedResponse::http_error(
                    503,
                    r#"{"error":{"message":"reference unavailable","retryable":true}}"#,
                ))
            })
            .collect();
        steps.push(ChatStep::new(ScriptedResponse::text("aggregate")));
        let Some(server) = ScriptedOpenAiServer::new(steps) else {
            return;
        };
        let provider = MoaProvider::new(
            Box::new(RecordingProvider::new(vec![])),
            Box::new(RecoveringReference(OpenAiProvider::new(
                server.url().into(),
                "test".into(),
            ))),
            MoaConfig::default(),
        )
        .unwrap();
        let mut request = request(MOA_MODEL_CONSERVATIVE);
        request.execution = Arc::new(RequestExecution::new(RequestExecutionPolicy {
            max_attempts,
            ..Default::default()
        }));
        let execution = request.execution.clone();
        let out = provider.stream(request.clone(), &mut |_| {}).await.unwrap();
        assert_eq!(completion_text(&out.content), "aggregate");
        assert_eq!(execution.attempt_limit(), max_attempts + 1);
        assert_eq!(execution.attempts(), max_attempts + 1);
        let requests = server.inspection().requests;
        assert_eq!(requests.len(), (max_attempts + 1) as usize);
        assert!(
            requests[..max_attempts as usize]
                .iter()
                .all(|r| r.body.contains(MOA_REFERENCE_CONSERVATIVE))
        );
        assert!(
            requests
                .last()
                .unwrap()
                .body
                .contains(MOA_AGGREGATOR_CONSERVATIVE)
        );
        assert!(provider.stream(request, &mut |_| {}).await.is_err());
        assert_eq!(server.inspection().requests.len(), requests.len());
        server.assert_clean().unwrap();
    }
}

#[test]
fn recursive_moa_routes_are_rejected() {
    let config = MoaConfig {
        presets: BTreeMap::from([(
            MOA_PRESET_CONSERVATIVE.to_string(),
            MoaPreset {
                reference_models: vec![MOA_MODEL_CONSERVATIVE.to_string()],
                ..MoaPreset::default()
            },
        )]),
        ..MoaConfig::default()
    };
    let err = config.validate().unwrap_err().to_string();
    assert!(err.contains("reference_models"));
}

#[test]
fn preserves_aggregator_rate_limits_when_summing_usage() {
    let mut aggregate = Usage {
        input_tokens: 4,
        output_tokens: 2,
        context_occupancy: 4,
        rate_limits: Some(RateLimitState {
            requests_min: RateLimitBucket {
                limit: 10,
                remaining: 8,
                reset_seconds: 1,
            },
            ..RateLimitState::default()
        }),
        ..Usage::default()
    };
    let reference = Usage {
        input_tokens: 10,
        output_tokens: 3,
        context_occupancy: 10,
        rate_limits: Some(RateLimitState {
            requests_min: RateLimitBucket {
                limit: 100,
                remaining: 90,
                reset_seconds: 2,
            },
            ..RateLimitState::default()
        }),
        ..Usage::default()
    };
    add_reference_usage(&mut aggregate, reference);
    assert_eq!(aggregate.input_tokens, 14);
    assert_eq!(aggregate.output_tokens, 5);
    assert_eq!(aggregate.context_occupancy, 4);
    assert_eq!(aggregate.rate_limits.unwrap().requests_min.limit, 10);
}

#[tokio::test]
async fn list_models_adds_virtual_model() {
    let passthrough = RecordingProvider::new(vec![]).with_models(vec![ServedModel {
        id: "real".into(),
        context_window: None,
        max_output_tokens: None,
        price: None,
        provider_label: None,
        status: None,
        available: true,
        availability_reason: None,
        capabilities: Vec::new(),
    }]);
    let provider = MoaProvider::new(
        Box::new(passthrough),
        Box::new(RecordingProvider::new(vec![])),
        MoaConfig::default(),
    )
    .unwrap();
    let models = provider.list_models().await.unwrap();
    assert!(models.iter().any(|m| m.id == "real"));
    assert!(models.iter().any(|m| {
        m.id == MOA_MODEL_CONSERVATIVE && m.provider_label.as_deref() == Some("virtual MoA route")
    }));
}
