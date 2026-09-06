use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use hi_agent::{LspMode, ReviewPolicy, ToolSet, VerificationMode};
use hi_ai::{
    ChatRequest, Completion, Provider, ProviderCapabilities, ProviderRequestContext,
    RequestProfile, RequestToolEnvelope, StreamEvent,
};
use hi_outcome::{OutcomeClient, OutcomeClientConfig, OutcomeMode, OutcomeOffer};

use super::super::OutcomeRouteProvider;
use crate::config::{QualitySettings, RaceSettings};

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
                        "members": [{"target": {"route": route}}]
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

fn subject(mode: OutcomeMode, root: &std::path::Path) -> OutcomeRouteProvider {
    OutcomeRouteProvider {
        inner: Arc::new(ToolCapableProvider),
        inner_rsi: None,
        client: OutcomeClient::new(OutcomeClientConfig {
            origin: "https://api.pipenetwork.ai".into(),
            api_key: "test-key".into(),
        })
        .unwrap(),
        workspace_root: root.to_path_buf(),
        state_root: root.join("state"),
        mode,
        offer: OutcomeOffer::Quality,
        quality: QualitySettings {
            verification: VerificationMode::Auto,
            max_verify_repairs: 0,
            max_verify_repairs_explicit: false,
            review: ReviewPolicy::Off,
            clippy: false,
            lsp_mode: LspMode::Off,
            tool_set: ToolSet::Full,
            context_exclusions: Vec::new(),
            race: RaceSettings::default(),
        },
        turn_deadline_secs: None,
        maximum_cost_microusd: 1,
    }
}

#[test]
fn auxiliary_and_non_outcome_primary_requests_keep_exact_inner_capabilities() {
    let root = tempfile::tempdir().unwrap();
    let provider = subject(OutcomeMode::Auto, root.path());

    let auxiliary = provider.capability_candidates_for_request(
        "configured-route",
        "configured-model",
        ProviderRequestContext::auxiliary(),
    );
    assert_eq!(auxiliary.len(), 1);
    assert!(auxiliary[0].declared.native_tool_calls);

    let qa = provider.capability_candidates_for_request(
        "configured-route",
        "configured-model",
        ProviderRequestContext::user_turn("what does this parser do?"),
    );
    assert_eq!(qa, auxiliary);
}

#[test]
fn routed_primary_includes_text_only_remote_and_inner_fail_open() {
    let root = tempfile::tempdir().unwrap();
    let provider = subject(OutcomeMode::Tasks, root.path());

    assert!(!provider.capabilities().native_tool_calls);
    let primary = provider.capability_candidates_for_request(
        "configured-route",
        "configured-model",
        ProviderRequestContext::user_turn("fix the parser"),
    );
    assert_eq!(primary.len(), 2);
    assert!(
        primary
            .iter()
            .any(|candidate| candidate.declared.native_tool_calls)
    );
    let remote = primary
        .iter()
        .find(|candidate| {
            candidate
                .target
                .route
                .starts_with("outcome_task@endpoint:blake3:")
        })
        .expect("remote task capability member");
    assert!(!remote.declared.native_tool_calls);
    assert!(!remote.target.route.contains("pipenetwork.ai"));
    assert_eq!(remote.target.model, "route_quality");

    let unknown_primary = provider.capability_candidates_for_request(
        "configured-route",
        "configured-model",
        ProviderRequestContext {
            user_turn: true,
            canonical_objective: None,
        },
    );
    assert_eq!(unknown_primary.len(), 2);
}

#[tokio::test]
async fn sealed_route_intent_wins_over_cargo_shape_changes_in_both_directions() {
    let root = tempfile::tempdir().unwrap();
    let provider = subject(OutcomeMode::Auto, root.path());
    let objective = "fix the failing tests in the parser";
    let inner_route = provider
        .capability_candidates_for_request(
            "configured-route",
            "configured-model",
            ProviderRequestContext::user_turn(objective),
        )
        .remove(0)
        .target
        .route;
    std::fs::write(
        root.path().join("Cargo.toml"),
        "[package]\nname='fixture'\n",
    )
    .unwrap();
    let mut sink = |_: StreamEvent| {};

    provider
        .stream(sealed_request(&inner_route), &mut sink)
        .await
        .expect("an inner-sealed request must not jump to Outcome after Cargo appears");

    let remote_route = provider
        .capability_candidates_for_request(
            "configured-route",
            "configured-model",
            ProviderRequestContext::user_turn(objective),
        )
        .into_iter()
        .find(|candidate| {
            candidate
                .target
                .route
                .starts_with("outcome_task@endpoint:blake3:")
        })
        .unwrap()
        .target
        .route;
    std::fs::remove_file(root.path().join("Cargo.toml")).unwrap();
    assert_eq!(
        provider
            .sealed_outcome_intent(&sealed_request(&remote_route))
            .unwrap(),
        Some(true),
        "a remote-sealed request retains its route after Cargo disappears"
    );
}
