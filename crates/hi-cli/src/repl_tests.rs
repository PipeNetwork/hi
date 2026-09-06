use std::sync::Arc;

use hi_ai::ToolMode;

use super::apply_resolved_provider;

fn switch_settings(tool_mode: ToolMode) -> crate::config::Settings {
    let mut config = crate::config::Config::default();
    config.profiles.insert(
        "switch".into(),
        crate::config::Profile {
            provider: Some(crate::config::ProviderName::Ollama),
            model: Some("local-switch-model".into()),
            tool_mode: Some(tool_mode),
            ..Default::default()
        },
    );
    crate::config::resolve_named_profile(&config, "switch").unwrap()
}

fn provider(settings: &crate::config::Settings) -> Arc<dyn hi_ai::Provider> {
    crate::build_chain(settings, Vec::new()).into()
}

#[test]
fn resolved_provider_switch_replaces_tool_mode_in_both_directions() {
    let automatic = switch_settings(ToolMode::Auto);
    let initial = provider(&automatic);
    let workspace = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let mut agent_config = hi_agent::AgentConfig::default();
    agent_config.paths.workspace_root = workspace.path().to_path_buf();
    agent_config.paths.state_root = state.path().to_path_buf();
    let mut agent = hi_agent::Agent::new(initial, agent_config).unwrap();

    let chat_only = switch_settings(ToolMode::ChatOnly);
    apply_resolved_provider(&mut agent, &chat_only, provider(&chat_only), None);
    assert_eq!(agent.tool_mode(), ToolMode::ChatOnly);

    apply_resolved_provider(&mut agent, &automatic, provider(&automatic), None);
    assert_eq!(agent.tool_mode(), ToolMode::Auto);
}
