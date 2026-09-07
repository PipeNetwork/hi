use super::handle_command;
use hi_agent::{AgentConfig, AgentPaths, Command, Goal};
use std::sync::Arc;

#[test]
#[allow(clippy::field_reassign_with_default)] // test assembles config field-by-field for clarity
fn cli_goal_budget_is_a_control_command_not_a_new_objective() {
    let root = std::env::temp_dir().join(format!(
        "hi-cli-goal-budget-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&root).expect("workspace");
    let mut config = AgentConfig::default();
    config.paths = AgentPaths {
        workspace_root: root.clone(),
        state_root: root.join(".hi-state"),
    };
    config.subagents.long_horizon = true;
    let provider = Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "test".into(),
    ));
    let mut agent = hi_agent::Agent::new(provider, config).expect("agent");
    agent
        .set_structured_goal(Some(Goal::new("ship it", vec!["implement it".into()])))
        .expect("goal accepted");

    handle_command(
        &mut agent,
        Command::Goal("budget 7".into()),
        None,
        None,
        None,
        None,
    );

    let goal = agent.structured_goal().expect("structured goal remains");
    assert_eq!(goal.objective, "ship it");
    assert_eq!(goal.turn_budget, Some(7));
    assert!(!goal.budget_auto);

    let _ = std::fs::remove_dir_all(root);
}
