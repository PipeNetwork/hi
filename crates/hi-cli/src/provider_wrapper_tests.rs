use std::fs;
use std::sync::{Arc, Mutex, atomic::AtomicBool};

use anyhow::Result;
use async_trait::async_trait;
use hi_agent::{Observation, ObservationReceipt, ObservationSink};
use hi_ai::{
    CapabilityRoute, ChatRequest, Completion, Content, Provider, ProviderCapabilities,
    ProviderCapabilityCandidate, ProviderRequestContext, StreamEvent, ToolMode,
};

use crate::rsi_observation::ObservedProvider;
use crate::rsi_remote::{RsiRemoteProvider, RsiSettings};

struct AcceptingSink;

impl ObservationSink for AcceptingSink {
    fn observe(&self, _: Observation) -> Result<ObservationReceipt> {
        Ok(ObservationReceipt {
            event_hash: "a".repeat(64),
            sequence: 1,
        })
    }
}

struct CapabilityProvider {
    exact: ProviderCapabilities,
    candidates: Vec<ProviderCapabilityCandidate>,
}

struct RequestRecordingProvider {
    requests: Arc<Mutex<Vec<ChatRequest>>>,
}

#[async_trait]
impl Provider for RequestRecordingProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::native_tools(true)
    }

    async fn stream(
        &self,
        request: ChatRequest,
        _: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        self.requests.lock().unwrap().push(request);
        Ok(Completion {
            content: vec![Content::Text("done".into())],
            stop_reason: Some("stop".into()),
            ..Completion::default()
        })
    }
}

#[async_trait]
impl Provider for CapabilityProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.exact.clone()
    }

    fn capability_candidates(&self, _: &str, _: &str) -> Vec<ProviderCapabilityCandidate> {
        self.candidates.clone()
    }

    fn capability_candidates_for_request(
        &self,
        _: &str,
        model: &str,
        context: ProviderRequestContext<'_>,
    ) -> Vec<ProviderCapabilityCandidate> {
        if context.user_turn {
            vec![ProviderCapabilityCandidate::new(
                CapabilityRoute::new("request-aware", model),
                ProviderCapabilities::default(),
            )]
        } else {
            self.candidates.clone()
        }
    }

    async fn stream(
        &self,
        _: ChatRequest,
        _: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        Ok(Completion::default())
    }
}

#[test]
fn observation_wrapper_preserves_exact_and_multi_route_capabilities() {
    let mut exact = ProviderCapabilities::native_tools(true);
    exact.parallel_tool_calls = true;
    exact.request_limits.max_input_tokens = Some(32_768);
    exact.request_limits.max_tools = Some(17);
    exact.actual_model_revision = Some("primary-revision".into());

    let mut fallback = ProviderCapabilities::native_tools(false);
    fallback.request_limits.max_output_tokens = Some(4_096);
    fallback.actual_model_revision = Some("fallback-revision".into());
    let candidates = vec![
        ProviderCapabilityCandidate::new(CapabilityRoute::new("primary", "model-a"), exact.clone()),
        ProviderCapabilityCandidate::new(CapabilityRoute::new("fallback", "model-b"), fallback),
    ];
    let provider = ObservedProvider::new(
        Arc::new(CapabilityProvider {
            exact: exact.clone(),
            candidates: candidates.clone(),
        }),
        Arc::new(AcceptingSink),
        None,
        false,
    );

    assert_eq!(Provider::capabilities(&provider), exact);
    assert_eq!(
        Provider::capability_candidates(&provider, "effective", "model"),
        candidates
    );
    let primary = Provider::capability_candidates_for_request(
        &provider,
        "effective",
        "model",
        ProviderRequestContext::user_turn("fix it"),
    );
    assert_eq!(primary.len(), 1);
    assert_eq!(primary[0].target.route, "request-aware");
    assert!(!primary[0].declared.native_tool_calls);
}

#[tokio::test]
async fn disabled_rsi_and_observation_stack_keeps_agent_tools_executable() {
    let root = tempfile::tempdir().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let rsi = RsiRemoteProvider::new(
        Arc::new(RequestRecordingProvider {
            requests: requests.clone(),
        }),
        Arc::new(AtomicBool::new(false)),
        root.path().to_path_buf(),
        root.path().join("rsi-state"),
        RsiSettings::resolve(
            Some("https://api.pipenetwork.ai"),
            Some("test-key"),
            None,
            None,
            None,
            "",
            "",
        )
        .unwrap(),
        Arc::new(|_, _, _| Ok(())),
    )
    .unwrap();
    let provider = ObservedProvider::new(Arc::new(rsi), Arc::new(AcceptingSink), None, false);
    let mut config = hi_agent::AgentConfig::default();
    config.paths.workspace_root = root.path().to_path_buf();
    config.paths.state_root = root.path().join(".hi");
    config.routing.model = "tool-capable-model".into();
    config.memory.tool_set = hi_agent::ToolSet::Full;
    config.loop_limits.max_steps = 1;
    let mut agent = hi_agent::Agent::new(Arc::new(provider), config).unwrap();

    agent
        .run_turn("Inspect the workspace.", &mut hi_agent::ui::NullUi)
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    let request = requests.first().expect("one observed model request");
    assert!(
        !request.tools.is_empty(),
        "the agent must advertise its tools"
    );
    assert_ne!(request.profile.tool_mode, ToolMode::ChatOnly);
    let envelope = request
        .tool_envelope
        .as_deref()
        .expect("the request must carry its sealed tool envelope");
    assert_eq!(
        envelope.payload["tool_mode"],
        serde_json::to_value(request.profile.tool_mode).unwrap()
    );
    let requested_names = request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<Vec<_>>();
    let enveloped_names = envelope.payload["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(enveloped_names, requested_names);
    for expected in ["read", "grep", "bash"] {
        assert!(
            requested_names.contains(&expected),
            "expected `{expected}` in the executable envelope: {requested_names:?}"
        );
    }
    assert!(envelope.digest.starts_with("blake3:"));
}

#[test]
fn disabled_rsi_wrapper_preserves_inner_tool_capabilities() {
    let root = std::env::temp_dir().join(format!(
        "hi-rsi-provider-capabilities-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let settings = RsiSettings::resolve(
        Some("https://api.pipenetwork.ai"),
        Some("test-key"),
        None,
        None,
        None,
        "",
        "",
    )
    .unwrap();
    let provider = RsiRemoteProvider::new(
        Arc::new(hi_ai::OpenAiProvider::new(
            "https://api.pipenetwork.ai/v1".into(),
            "test-key".into(),
        )),
        Arc::new(AtomicBool::new(false)),
        root.clone(),
        root.join("state"),
        settings,
        Arc::new(|_, _, _| Ok(())),
    )
    .unwrap();

    let capabilities = Provider::capabilities(&provider);
    assert!(capabilities.native_tool_calls);
    assert!(capabilities.tool_choice.automatic);
    let candidates = Provider::capability_candidates(&provider, "pipenetwork", "pipe/test-model");
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].declared.native_tool_calls);

    fs::remove_dir_all(root).unwrap();
}
