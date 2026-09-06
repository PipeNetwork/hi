use std::sync::Arc;

use super::common::*;
use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SeenContext {
    user_turn: bool,
    objective: Option<String>,
}

struct RequestAwareProvider {
    contexts: Arc<Mutex<Vec<SeenContext>>>,
    requests: Arc<Mutex<Vec<hi_ai::ChatRequest>>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for RequestAwareProvider {
    fn capability_candidates_for_request(
        &self,
        route: &str,
        model: &str,
        context: hi_ai::ProviderRequestContext<'_>,
    ) -> Vec<hi_ai::ProviderCapabilityCandidate> {
        self.contexts.lock().unwrap().push(SeenContext {
            user_turn: context.user_turn,
            objective: context.canonical_objective.map(str::to_owned),
        });
        let capabilities = if context.user_turn {
            hi_ai::ProviderCapabilities::default()
        } else {
            hi_ai::ProviderCapabilities::native_tools(true)
        };
        vec![hi_ai::ProviderCapabilityCandidate::new(
            hi_ai::CapabilityRoute::new(route, model),
            capabilities,
        )]
    }

    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        _: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        self.requests.lock().unwrap().push(request);
        Ok(completion(vec![Content::Text("done".into())], 1, 1))
    }
}

#[tokio::test]
async fn sealing_passes_primary_objective_and_keeps_auxiliary_route_tool_capable() {
    let workspace = IsolatedWorkspace::new("provider-request-context");
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut config = workspace.config();
    config.loop_limits.max_steps = 1;
    let mut agent = Agent::new(
        Arc::new(RequestAwareProvider {
            contexts: contexts.clone(),
            requests: requests.clone(),
        }),
        config,
    )
    .unwrap();

    let auxiliary = agent
        .seal_auxiliary_request(
            "m",
            Arc::from([hi_ai::ToolSpec {
                name: "read".into(),
                description: "Read a file".into(),
                parameters: serde_json::json!({"type": "object"}),
            }]),
            hi_ai::ToolMode::Auto,
            100,
            1,
        )
        .await;
    assert_eq!(auxiliary.tool_mode, hi_ai::ToolMode::Auto);
    assert_eq!(auxiliary.tools.len(), 1);

    agent
        .run_turn("inspect the exact route", &mut RecUi::default())
        .await
        .unwrap();
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].profile.tool_mode, hi_ai::ToolMode::ChatOnly);
    let envelope = requests[0].tool_envelope.as_deref().unwrap();
    assert_eq!(envelope.payload["tool_mode"], "chat-only");
    assert_eq!(
        envelope.payload["provider"]["capability_record"]["capabilities"]["native_tool_calls"],
        false
    );
    let contexts = contexts.lock().unwrap();
    let (primary, auxiliary) = contexts.split_last().expect("capability lookups");
    assert_eq!(
        primary,
        &SeenContext {
            user_turn: true,
            objective: Some("inspect the exact route".into()),
        }
    );
    assert!(!auxiliary.is_empty());
    assert!(auxiliary.iter().all(|context| {
        context
            == &SeenContext {
                user_turn: false,
                objective: None,
            }
    }));
}
