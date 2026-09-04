use std::fs;
use std::sync::{Arc, atomic::AtomicBool};

use anyhow::Result;
use async_trait::async_trait;
use hi_agent::{Observation, ObservationReceipt, ObservationSink};
use hi_ai::{
    CapabilityRoute, ChatRequest, Completion, Provider, ProviderCapabilities,
    ProviderCapabilityCandidate, StreamEvent,
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

#[async_trait]
impl Provider for CapabilityProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.exact.clone()
    }

    fn capability_candidates(&self, _: &str, _: &str) -> Vec<ProviderCapabilityCandidate> {
        self.candidates.clone()
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
