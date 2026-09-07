use super::common::*;
use super::*;

type ChatOnlyObservation = (
    Agent,
    std::sync::Arc<Mutex<Vec<Vec<String>>>>,
    std::sync::Arc<Mutex<Vec<ToolMode>>>,
    hi_ai::ProviderCapabilityRegistry,
);

fn route_config(label: &str) -> AgentConfig {
    let mut config = config();
    config.subagents.explore_subagents = true;
    config.routing.provider_route = Some(label.into());
    config.routing.capability_route = Some(format!("{label}-capabilities"));
    config
}

fn agent_with_chat_only_observation(workspace: &IsolatedWorkspace) -> ChatOnlyObservation {
    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let answer = || completion(vec![Content::Text("read-only answer".into())], 1, 1);
    let provider = std::sync::Arc::new(RecordRequests {
        // A ChatOnly explorer exhausts bounded no-evidence repair. Registry
        // inheritance still has to hold for every attempted child request.
        responses: Mutex::new(std::iter::repeat_with(answer).take(12).collect()),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    });
    let mut config = workspace.config();
    config.subagents.explore_subagents = true;
    config.memory.finalize = false;
    config.routing.provider_route = Some("driver".into());
    config.routing.capability_route = Some("driver-capabilities".into());
    let model = config.routing.model.clone();
    let mut agent = Agent::new(provider, config).unwrap();
    let registry = hi_ai::ProviderCapabilityRegistry::default();
    registry.seed_observation(
        hi_ai::CapabilityRoute::new("driver-capabilities", model),
        test_provider_capabilities(),
        hi_ai::CapabilityProbeObservation {
            capabilities: hi_ai::ProviderCapabilities::default(),
            actual_model_revision: Some("observed-chat-only".into()),
        },
    );
    agent.set_provider_capability_registry(registry.clone());
    (agent, tool_names, modes, registry)
}

#[test]
fn explore_endpoint_fences_foreground_and_background_child_routes() {
    let endpoint = "https://user:secret@explore.invalid/v1?token=hidden";
    let mut agent = agent(Vec::new(), route_config("driver"));
    agent.set_explore_route(
        Some("explore-model".into()),
        Some(endpoint.into()),
        Some("separate-key".into()),
    );

    let foreground = agent
        .prepare_explore(r#"{"task":"inspect the provider route"}"#)
        .expect("foreground explore job");
    let expected = hi_ai::endpoint_capability_route("explore", endpoint);
    assert_eq!(
        foreground.child_config.routing.provider_route.as_deref(),
        Some("explore")
    );
    assert_eq!(
        foreground.child_config.routing.capability_route.as_deref(),
        Some(expected.as_str())
    );

    let background = agent.background_explore_route_for_test();
    assert_eq!(background.label, "explore");
    assert_eq!(background.capability_identity, expected);
    assert!(!background.capability_identity.contains("explore.invalid"));
    assert!(!background.capability_identity.contains("secret"));
    assert!(!background.capability_identity.contains("hidden"));
}

#[tokio::test]
async fn foreground_explore_inherits_the_parent_capability_registry() {
    let workspace = IsolatedWorkspace::new("explore-capability-registry");
    let (mut agent, tool_names, modes, registry) = agent_with_chat_only_observation(&workspace);
    let job = agent
        .prepare_explore(r#"{"task":"summarize the workspace"}"#)
        .expect("explore job");
    let result = crate::agent::run_explore_job(job, &mut NullUi).await;

    assert_eq!(result.outcome.status, hi_tools::ToolStatus::Succeeded);
    assert!(
        registry
            .audit_records()
            .iter()
            .any(|record| record.cache_hit),
        "child did not share the seeded registry: {:?}",
        registry.audit_records()
    );
    assert_eq!(modes.lock().unwrap().first(), Some(&ToolMode::ChatOnly));
    assert!(
        tool_names
            .lock()
            .unwrap()
            .first()
            .is_some_and(|tools| tools.iter().any(|tool| tool == "read")),
        "ChatOnly retains the catalog for audit but cannot admit its calls"
    );
}

#[tokio::test]
async fn background_explore_inherits_the_parent_capability_registry() {
    let workspace = IsolatedWorkspace::new("background-explore-capability-registry");
    let (mut agent, tool_names, modes, registry) = agent_with_chat_only_observation(&workspace);
    let spawned = agent
        .handle_task(
            r#"{"description":"inspect","prompt":"summarize the workspace","subagent_type":"explore"}"#,
            &mut NullUi,
        )
        .await;
    assert_eq!(spawned.status, hi_tools::ToolStatus::Succeeded);
    let task_id = agent
        .background_task_registry()
        .list()
        .await
        .into_iter()
        .next()
        .expect("spawned background task");
    let terminal = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let outcome = agent
                .background_task_registry()
                .poll(&task_id, std::time::Duration::ZERO)
                .await
                .expect("background task remains registered");
            if outcome.state != hi_tools::BackgroundTaskState::Running {
                break outcome;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background explore should finish");

    assert_eq!(
        terminal.state,
        hi_tools::BackgroundTaskState::Failed,
        "a catalog-only explorer cannot complete an evidence-required task: {}",
        terminal.output
    );
    assert!(
        registry
            .audit_records()
            .iter()
            .any(|record| record.cache_hit),
        "child did not share the seeded registry: {:?}",
        registry.audit_records()
    );
    assert_eq!(modes.lock().unwrap().first(), Some(&ToolMode::ChatOnly));
    assert!(
        tool_names
            .lock()
            .unwrap()
            .first()
            .is_some_and(|tools| tools.iter().any(|tool| tool == "read")),
        "ChatOnly retains the catalog for audit but cannot admit its calls"
    );
}
