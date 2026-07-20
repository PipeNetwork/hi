use super::*;

fn goal_test_config(label: &str) -> (std::path::PathBuf, hi_agent::AgentConfig) {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "hi-tui-goal-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let config = hi_agent::AgentConfig {
paths: hi_agent::AgentPaths {
workspace_root: root.clone(),
state_root: root.join(".hi-state"),
..hi_agent::AgentPaths::default()
},
routing: hi_agent::AgentRouting {
model: "test-model".into(),
provider_route: Some("custom-test-provider".into()),
..hi_agent::AgentRouting::default()
},
gates: hi_agent::AgentGates {
lsp_mode: hi_agent::LspMode::Off,
..hi_agent::AgentGates::default()
},
subagents: hi_agent::AgentSubagents {
long_horizon: true,
..hi_agent::AgentSubagents::default()
},
..hi_agent::AgentConfig::default()
};
    (root, config)
}

fn goal_test_provider() -> std::sync::Arc<dyn hi_ai::Provider> {
    std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "unused".into(),
    ))
}

#[tokio::test]
async fn exact_plan_document_goal_becomes_structured_and_starts_driving() {
    const LINE: &str = "/goal review the plan.md document and fully build this";
    const OBJECTIVE: &str = "review the plan.md document and fully build this";

    let parsed = hi_agent::command::parse(LINE).expect("slash command");
    assert_eq!(parsed, hi_agent::Command::Goal(OBJECTIVE.into()));
    assert!(hi_agent::command::goal_arg_is_objective(OBJECTIVE));

    let (root, config) = goal_test_config("exact-command");
    assert!(config.subagents.planner_model.is_none());
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    let mut app = test_app("custom", "test-model");

    app.handle_command(&mut agent, parsed).await;

    let goal = agent.structured_goal().expect("structured goal installed");
    assert_eq!(goal.objective, OBJECTIVE);
    assert_eq!(goal.sub_goals.len(), 1);
    assert_eq!(goal.sub_goals[0].description, OBJECTIVE);
    app.maybe_queue_goal_drive(&agent);
    assert_eq!(
        app.queue.pop_front().as_deref(),
        Some(hi_agent::GOAL_CONTINUE_PROMPT)
    );

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn resumed_active_goal_is_queued_without_displacing_user_input() {
    let (root, config) = goal_test_config("resume-drive");
    let goal = hi_agent::Goal::new(
        "review plan.md and build it",
        vec!["review plan.md".into(), "implement it".into()],
    );
    let mut agent = hi_agent::Agent::resume(
        goal_test_provider(),
        config,
        vec![hi_ai::Message::system("test system")],
        hi_ai::Usage::default(),
        Vec::new(),
        Some(goal),
        hi_agent::DecisionLog::default(),
    )
    .unwrap();
    let mut app = test_app("custom", "test-model");

    app.refresh_goal(&agent);
    app.maybe_queue_goal_drive(&agent);
    assert_eq!(app.queue[0], hi_agent::GOAL_CONTINUE_PROMPT);

    app.queue.clear();
    app.queue.push_back("user guidance takes priority".into());
    app.maybe_queue_goal_drive(&agent);
    assert_eq!(app.queue[0], "user guidance takes priority");

    app.queue.clear();
    assert!(agent.set_goal_paused(true));
    app.maybe_queue_goal_drive(&agent);
    assert!(app.queue.is_empty());

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}
