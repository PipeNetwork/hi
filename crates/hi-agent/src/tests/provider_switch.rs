use super::common::*;
use super::*;

#[derive(Clone, Debug)]
struct SeenRequest {
    mode: hi_ai::ToolMode,
    tools: Vec<String>,
    capability_route: String,
}

struct RouteRecordingProvider {
    seen: std::sync::Arc<Mutex<Vec<SeenRequest>>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for RouteRecordingProvider {
    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        _: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        let capability_route = request
            .tool_envelope
            .as_deref()
            .and_then(|envelope| envelope.payload["provider"]["route"].as_str())
            .unwrap_or("missing")
            .to_string();
        self.seen.lock().unwrap().push(SeenRequest {
            mode: request.profile.tool_mode,
            tools: request.tools.iter().map(|tool| tool.name.clone()).collect(),
            capability_route,
        });
        Ok(completion(
            vec![Content::Text("The file contains a test function.".into())],
            1,
            1,
        ))
    }

    fn capabilities(&self) -> hi_ai::ProviderCapabilities {
        hi_ai::ProviderCapabilities::native_tools(true)
    }
}

#[tokio::test]
async fn provider_switch_rotates_stale_no_tools_observation_and_updates_route_identity() {
    let workspace = IsolatedWorkspace::new("provider-switch-capability-route");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "pub fn test_value() -> u8 { 1 }\n",
    )
    .unwrap();

    let old_route = hi_ai::endpoint_capability_route(
        "openai",
        "https://old-user:old-secret@old.invalid/v1?key=old-secret",
    );
    let new_route = hi_ai::endpoint_capability_route(
        "openai",
        "https://new-user:new-secret@new.invalid/v1?key=new-secret",
    );
    let mut config = workspace.config();
    config.routing.provider_route = Some("openai".into());
    config.routing.capability_route = Some(old_route.clone());
    config.loop_limits.max_steps = 1;

    let old_seen = std::sync::Arc::new(Mutex::new(Vec::new()));
    let declared = hi_ai::ProviderCapabilities::native_tools(true);
    let registry = hi_ai::ProviderCapabilityRegistry::default();
    registry.seed_observation(
        hi_ai::CapabilityRoute::new(&old_route, "m"),
        declared,
        hi_ai::CapabilityProbeObservation {
            capabilities: hi_ai::ProviderCapabilities::default(),
            actual_model_revision: Some("old-no-tools".into()),
        },
    );
    let mut agent = Agent::new(
        std::sync::Arc::new(RouteRecordingProvider {
            seen: old_seen.clone(),
        }),
        config,
    )
    .unwrap();
    agent.set_provider_capability_registry(registry);

    let mut ui = RecUi::default();
    let _ = agent
        .run_turn("Read src/lib.rs and summarize it.", &mut ui)
        .await;
    let old_request = old_seen.lock().unwrap().first().cloned().unwrap();
    assert_eq!(old_request.mode, hi_ai::ToolMode::ChatOnly);
    assert_eq!(old_request.capability_route, old_route);

    let new_seen = std::sync::Arc::new(Mutex::new(Vec::new()));
    agent.set_provider_with_route(
        std::sync::Arc::new(RouteRecordingProvider {
            seen: new_seen.clone(),
        }),
        AgentProviderRoute::new("openai", new_route.clone()),
        "m".into(),
        None,
        100,
        true,
        None,
    );
    let _ = agent
        .run_turn("Read src/lib.rs and summarize it again.", &mut ui)
        .await;

    let new_request = new_seen.lock().unwrap().first().cloned().unwrap();
    assert_ne!(new_request.mode, hi_ai::ToolMode::ChatOnly);
    assert!(new_request.tools.iter().any(|tool| tool == "read"));
    assert_eq!(new_request.capability_route, new_route);
    assert_eq!(agent.provider_route(), Some("openai"));
    assert!(!new_request.capability_route.contains("new.invalid"));
    assert!(!new_request.capability_route.contains("new-secret"));
}

#[tokio::test]
async fn legacy_provider_switch_also_rotates_observations_for_the_same_route_key() {
    let workspace = IsolatedWorkspace::new("legacy-provider-switch-capability-route");
    let route = hi_ai::endpoint_capability_route("openai", "https://gateway.invalid/v1");
    let mut config = workspace.config();
    config.routing.provider_route = Some("openai".into());
    config.routing.capability_route = Some(route.clone());
    let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = || {
        std::sync::Arc::new(RouteRecordingProvider { seen: seen.clone() })
            as std::sync::Arc<dyn hi_ai::Provider>
    };
    let declared = hi_ai::ProviderCapabilities::native_tools(true);
    let registry = hi_ai::ProviderCapabilityRegistry::default();
    registry.seed_observation(
        hi_ai::CapabilityRoute::new(&route, "m"),
        declared,
        hi_ai::CapabilityProbeObservation {
            capabilities: hi_ai::ProviderCapabilities::default(),
            actual_model_revision: Some("stale-no-tools".into()),
        },
    );
    let mut agent = Agent::new(provider(), config).unwrap();
    agent.set_provider_capability_registry(registry);
    assert!(
        !agent
            .effective_provider_capabilities()
            .await
            .capabilities
            .native_tool_calls
    );

    agent.set_provider(provider(), "m".into(), None, 100, true, None);

    let refreshed = agent.effective_provider_capabilities().await;
    assert!(refreshed.capabilities.native_tool_calls);
    assert_eq!(refreshed.target.route, route);
}
