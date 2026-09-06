use super::common::*;
use super::*;
use async_trait::async_trait;
use hi_ai::test_support::{FakeOpenAiServer, Response, sse_text};
use hi_ai::{ChatRequest, Completion, OpenAiProvider, Provider, ProviderError, StreamEvent};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct ProtocolThenOpenAi {
    failures: AtomicUsize,
    openai: OpenAiProvider,
    tail: Mutex<Vec<Completion>>,
}

#[async_trait]
impl Provider for ProtocolThenOpenAi {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        let attempt = self.failures.fetch_add(1, Ordering::SeqCst);
        if attempt < 5 {
            return Err(ProviderError::new(
                ProviderErrorKind::ToolProtocol,
                "scripted structured-call failure",
            )
            .into());
        }
        if attempt == 5 {
            return self.openai.stream(request, sink).await;
        }
        pop_canned_completion(&self.tail, "ProtocolThenOpenAi")
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        self.openai.capabilities()
    }
}

struct CapFallbackProbe {
    responses: Mutex<Vec<Completion>>,
    requests: Arc<Mutex<Vec<(hi_ai::ToolMode, bool)>>>,
}

#[async_trait]
impl Provider for CapFallbackProbe {
    async fn stream(
        &self,
        request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        self.requests.lock().unwrap().push((
            request.profile.tool_mode,
            request
                .tool_envelope
                .as_deref()
                .is_some_and(hi_ai::RequestToolEnvelope::requests_text_tool_fallback),
        ));
        pop_canned_completion(&self.responses, "CapFallbackProbe")
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        test_provider_capabilities()
    }
}

#[tokio::test]
async fn pending_text_fallback_cannot_escape_the_step_cap_wrap_up() {
    let workspace = IsolatedWorkspace::new("text-fallback-step-cap");
    let destination = workspace.path("must-not-exist.txt");
    let xml_call = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{}</arg_value><arg_key>content</arg_key><arg_value>escaped</arg_value></tool_call>",
        destination.display()
    );
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = CapFallbackProbe {
        responses: Mutex::new(vec![
            completion(
                vec![Content::ToolCall {
                    id: "invalid-write".into(),
                    name: "write".into(),
                    arguments: "{}".into(),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text(xml_call)], 1, 1),
        ]),
        requests: requests.clone(),
    };
    let mut config = workspace.config();
    config.memory.tool_set = ToolSet::Full;
    config.loop_limits.max_steps = 1;
    config.loop_limits.max_repeat_nudges = 0;
    config.gates.allow_unverified = true;
    config.gates.proactive_verify = false;
    config.memory.finalize = false;
    let mut agent = Agent::new(Arc::new(provider), config).unwrap();
    let mut ui = RecUi::default();

    agent
        .run_turn(
            &format!("Create {} containing escaped.", destination.display()),
            &mut ui,
        )
        .await
        .unwrap();

    assert!(
        !destination.exists(),
        "the tool-free cap wrap-up executed a pending text fallback"
    );
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2, "expected one work round and one wrap-up");
    assert_eq!(requests[0].0, hi_ai::ToolMode::Auto);
    assert_eq!(requests[1], (hi_ai::ToolMode::ChatOnly, false));
    assert_eq!(
        ui.tool_results
            .iter()
            .filter(|(name, _)| name == "write")
            .count(),
        1,
        "only the initial invalid structured call should reach execution validation"
    );
}

#[tokio::test]
async fn fallback_rejects_multiple_calls_before_executing_any_prefix() {
    let workspace = IsolatedWorkspace::new("text-fallback-exactly-one");
    let rejected = workspace.path("rejected.txt");
    let accepted = workspace.path("accepted.txt");
    let multi = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{}</arg_value><arg_key>content</arg_key><arg_value>bad</arg_value></tool_call>\
         <tool_call>write<arg_key>path</arg_key><arg_value>{}</arg_value><arg_key>content</arg_key><arg_value>also bad</arg_value></tool_call>",
        rejected.display(),
        accepted.display(),
    );
    let single = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{}</arg_value><arg_key>content</arg_key><arg_value>good</arg_value></tool_call>",
        accepted.display(),
    );
    let mut steps = (0..5)
        .map(|_| ProviderStep::Error(ProviderErrorKind::ToolProtocol))
        .collect::<Vec<_>>();
    steps.extend([
        ProviderStep::Completion(completion(vec![Content::Text(multi)], 1, 1)),
        ProviderStep::Completion(completion(vec![Content::Text(single)], 1, 1)),
        ProviderStep::Completion(bash_completion("true # validate exact fallback")),
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "Created accepted.txt and validated it.".into(),
            )],
            1,
            1,
        )),
    ]);
    let mut config = workspace.config();
    config.gates.allow_unverified = true;
    config.loop_limits.max_repeat_nudges = 1;
    let (mut agent, _) = scripted_agent(steps, config);
    let mut ui = RecUi::default();

    agent
        .run_turn("/build the requested file", &mut ui)
        .await
        .unwrap();

    assert!(!rejected.exists(), "the first call in a rejected batch ran");
    assert_eq!(std::fs::read_to_string(accepted).unwrap(), "good");
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("multiple calls"))
    );
}

#[tokio::test]
async fn openai_sse_fallback_is_promoted_and_executed_by_the_agent() {
    let workspace = IsolatedWorkspace::new("openai-stream-text-fallback");
    let destination = workspace.path("streamed.txt");
    let xml = format!(
        "I will create it.\n<tool_call>write<arg_key>path</arg_key><arg_value>{}</arg_value><arg_key>content</arg_key><arg_value>from stream</arg_value></tool_call>",
        destination.display(),
    );
    let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_text(&xml))]) else {
        return;
    };
    let provider = ProtocolThenOpenAi {
        failures: AtomicUsize::new(0),
        openai: OpenAiProvider::new(server.url().to_string(), "test".into()),
        tail: Mutex::new(vec![
            bash_completion("true # validate streamed fallback"),
            completion(
                vec![Content::Text(
                    "Created streamed.txt and validated it.".into(),
                )],
                1,
                1,
            ),
        ]),
    };
    let mut config = workspace.config();
    config.gates.allow_unverified = true;
    config.loop_limits.max_repeat_nudges = 1;
    let mut agent = Agent::new(std::sync::Arc::new(provider), config).unwrap();
    let mut ui = RecUi::default();

    agent
        .run_turn("/build the requested streamed file", &mut ui)
        .await
        .unwrap();

    assert_eq!(std::fs::read_to_string(destination).unwrap(), "from stream");
    assert!(!ui.assistant.contains("<tool_call>"));
    assert_eq!(
        agent.last_turn_telemetry().tool_call_channel,
        "text_fallback"
    );
}
