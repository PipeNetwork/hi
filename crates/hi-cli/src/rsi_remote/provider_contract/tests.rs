use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::Result;
use async_trait::async_trait;
use hi_ai::{
    ChatRequest, Completion, Provider, ProviderCapabilities, ProviderRequestContext,
    RequestProfile, RequestToolEnvelope, StreamEvent,
};

use super::super::{RsiRemoteProvider, RsiSettings};
use super::SealedRsiRoute;

struct ToolCapableProvider;

#[async_trait]
impl Provider for ToolCapableProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::native_tools(true)
    }

    async fn stream(
        &self,
        _: ChatRequest,
        _: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        Ok(Completion::default())
    }
}

fn sealed_request(route: &str) -> ChatRequest {
    ChatRequest {
        execution: Default::default(),
        model: "configured-model".into(),
        request_id: None,
        retry_attempt: 0,
        user_turn: true,
        canonical_objective: Some("fix the parser".into()),
        messages: Arc::new(Vec::new()),
        tools: Arc::from([]),
        tool_envelope: Some(Arc::new(RequestToolEnvelope {
            digest: "test-envelope".into(),
            payload: serde_json::json!({
                "provider": {
                    "capability_record": {
                        "members": [{"target": {
                            "route": route,
                            "model": "server-selected/stable"
                        }}]
                    }
                }
            }),
        })),
        max_tokens: 100,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: RequestProfile::default(),
    }
}

fn subject(enabled: Arc<AtomicBool>, root: &std::path::Path) -> RsiRemoteProvider {
    RsiRemoteProvider::new(
        Arc::new(ToolCapableProvider),
        enabled,
        root.to_path_buf(),
        root.join("state"),
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
    .unwrap()
}

#[test]
fn enabled_primary_is_remote_while_auxiliary_and_disabled_requests_are_inner() {
    let root = tempfile::tempdir().unwrap();
    let enabled = Arc::new(AtomicBool::new(true));
    let provider = subject(enabled.clone(), root.path());

    let primary = provider.capability_candidates_for_request(
        "configured-route",
        "configured-model",
        ProviderRequestContext::user_turn("fix the parser"),
    );
    assert_eq!(primary.len(), 1);
    assert!(!primary[0].declared.native_tool_calls);
    assert!(
        primary[0]
            .target
            .route
            .starts_with("rsi_remote@endpoint:blake3:")
    );
    assert!(!primary[0].target.route.contains("pipenetwork.ai"));
    assert_eq!(primary[0].target.model, "server-selected/stable");

    let auxiliary = provider.capability_candidates_for_request(
        "configured-route",
        "configured-model",
        ProviderRequestContext::auxiliary(),
    );
    assert_eq!(auxiliary.len(), 1);
    assert!(auxiliary[0].declared.native_tool_calls);
    assert_eq!(auxiliary[0].target.route, "configured-route");

    enabled.store(false, Ordering::SeqCst);
    assert!(Provider::capabilities(&provider).native_tool_calls);
    let disabled_primary = provider.capability_candidates_for_request(
        "configured-route",
        "configured-model",
        ProviderRequestContext::user_turn("fix the parser"),
    );
    assert_eq!(disabled_primary, auxiliary);
}

#[test]
fn generic_enabled_declaration_includes_every_possible_route() {
    let root = tempfile::tempdir().unwrap();
    let provider = subject(Arc::new(AtomicBool::new(true)), root.path());

    assert!(!Provider::capabilities(&provider).native_tool_calls);
    let candidates = provider.capability_candidates("configured-route", "configured-model");
    assert_eq!(candidates.len(), 2);
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.declared.native_tool_calls)
    );
    assert!(candidates.iter().any(|candidate| {
        candidate
            .target
            .route
            .starts_with("rsi_remote@endpoint:blake3:")
            && !candidate.declared.native_tool_calls
    }));
}

#[tokio::test]
async fn sealed_route_intent_wins_over_late_enablement_changes() {
    let root = tempfile::tempdir().unwrap();
    let enabled = Arc::new(AtomicBool::new(false));
    let provider = subject(enabled.clone(), root.path());

    let inner_route = provider
        .capability_candidates_for_request(
            "configured-route",
            "configured-model",
            ProviderRequestContext::user_turn("fix the parser"),
        )
        .remove(0)
        .target
        .route;
    enabled.store(true, Ordering::SeqCst);
    let mut sink = |_: StreamEvent| {};
    provider
        .stream(sealed_request(&inner_route), &mut sink)
        .await
        .expect("an inner-sealed request must not jump to RSI after enablement changes");

    let remote_route = provider
        .capability_candidates_for_request(
            "configured-route",
            "configured-model",
            ProviderRequestContext::user_turn("fix the parser"),
        )
        .remove(0)
        .target
        .route;
    provider.settings.channel.store(1, Ordering::SeqCst);
    enabled.store(false, Ordering::SeqCst);
    assert_eq!(
        provider
            .sealed_route(&sealed_request(&remote_route))
            .unwrap(),
        Some(SealedRsiRoute::Remote("stable".into())),
        "a remote-sealed request retains its route after a late disable"
    );
}
