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

#[tokio::test]
async fn checkbox_plan_md_ingests_as_sub_goals_without_planner() {
    let (root, mut config) = goal_test_config("ingest-checklist");
    config.paths.workspace_root = std::fs::canonicalize(&root).unwrap();
    std::fs::write(
        config.paths.workspace_root.join("plan.md"),
        "- [x] already shipped\n- [ ] wire the CLI\n- [ ] pass tests\n",
    )
    .unwrap();
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    let mut app = test_app("custom", "test-model");

    app.handle_command(
        &mut agent,
        hi_agent::Command::Goal("implement plan.md".into()),
    )
    .await;

    let goal = agent.structured_goal().expect("structured goal installed");
    assert_eq!(goal.objective, "implement plan.md");
    assert_eq!(goal.sub_goals.len(), 3);
    assert_eq!(goal.sub_goals[0].status, hi_agent::GoalStatus::Done);
    assert_eq!(goal.sub_goals[1].description, "wire the CLI");
    assert_eq!(goal.sub_goals[1].status, hi_agent::GoalStatus::Active);
    app.maybe_queue_goal_drive(&agent);
    assert_eq!(
        app.queue.pop_front().as_deref(),
        Some(hi_agent::GOAL_CONTINUE_PROMPT)
    );

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn goal_budget_refreshes_the_pinned_goal_state() {
    let (root, config) = goal_test_config("budget-refresh");
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    agent
        .set_structured_goal(Some(hi_agent::Goal::new(
            "ship it",
            vec!["implement it".into()],
        )))
        .unwrap();
    let mut app = test_app("custom", "test-model");
    app.refresh_goal(&agent);

    app.handle_command(&mut agent, hi_agent::Command::Goal("budget 7".into()))
        .await;

    assert_eq!(agent.structured_goal().and_then(|g| g.turn_budget), Some(7));
    assert_eq!(app.goal.as_ref().and_then(|g| g.turn_budget), Some(7));

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

    assert!(agent.set_goal_paused(false));
    agent.reset_goal_drive_stall();
    app.maybe_queue_goal_drive(&agent);
    assert_eq!(app.queue[0], hi_agent::GOAL_CONTINUE_PROMPT);

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

fn incomplete_plan_outcome() -> hi_agent::TurnOutcome {
    hi_agent::TurnOutcome {
        status: hi_agent::TurnStatus::Incomplete,
        verification: hi_agent::VerificationStatus::Unverified,
        review: hi_agent::ReviewStatus::NotRequired,
        stop_reason: hi_agent::TurnStopReason::Stalled,
        changed_files: Vec::new(),
        verified_workspace_revision: None,
        effective_route: hi_agent::EffectiveModelRoute {
            provider: Some("test".into()),
            model: "model".into(),
        },
        review_same_model: false,
        leftover: Some("1/1 remaining — wire the scheduler".into()),
        plan_leftover: Some("1/1 remaining — wire the scheduler".into()),
    }
}

#[test]
fn incomplete_plan_enqueues_plan_drive() {
    let (root, config) = goal_test_config("plan-drive");
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    agent.restore_plan(vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }]);
    let mut app = test_app("custom", "test-model");
    let outcome = incomplete_plan_outcome();

    app.maybe_queue_drive(&agent, Some(&outcome));
    assert_eq!(
        app.queue.pop_front().as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );

    app.queue.push_back("user guidance takes priority".into());
    app.maybe_queue_drive(&agent, Some(&outcome));
    assert_eq!(app.queue[0], "user guidance takes priority");

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn goal_drive_wins_over_plan_drive() {
    let (root, config) = goal_test_config("goal-wins-plan");
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    agent
        .set_structured_goal(Some(hi_agent::Goal::new(
            "ship it",
            vec!["implement it".into()],
        )))
        .unwrap();
    agent.restore_plan(vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }]);
    let mut app = test_app("custom", "test-model");
    let outcome = incomplete_plan_outcome();
    app.maybe_queue_goal_drive(&agent);
    app.maybe_queue_drive(&agent, Some(&outcome));
    assert_eq!(app.queue.len(), 1);
    assert_eq!(app.queue[0], hi_agent::GOAL_CONTINUE_PROMPT);

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn completed_leftover_still_enqueues_plan_drive() {
    let (root, config) = goal_test_config("completed-leftover-drive");
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    agent.restore_plan(vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }]);
    let mut app = test_app("custom", "test-model");
    let mut outcome = incomplete_plan_outcome();
    outcome.status = hi_agent::TurnStatus::Completed;
    outcome.stop_reason = hi_agent::TurnStopReason::Completed;
    app.maybe_queue_drive(&agent, Some(&outcome));
    assert_eq!(
        app.queue.pop_front().as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );
    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn plan_pause_stops_enqueue_and_resume_restarts() {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    let (root, config) = goal_test_config("plan-pause");
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    agent.restore_plan(vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }]);
    let mut app = test_app("custom", "test-model");
    let outcome = incomplete_plan_outcome();

    let pause =
        hi_agent::handle_session_command(&mut agent, &hi_agent::Command::Plan("pause".into()), &[])
            .expect("pause");
    assert!(pause.follow_up_prompt.is_none());
    app.refresh_goal(&agent);
    app.maybe_queue_drive(&agent, Some(&outcome));
    assert!(app.queue.is_empty(), "pause must stop auto-enqueue");
    assert_eq!(
        app.edit_key(&KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
            .as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );

    let resume = hi_agent::handle_session_command(
        &mut agent,
        &hi_agent::Command::Plan("resume".into()),
        &[],
    )
    .expect("resume");
    assert_eq!(
        resume.follow_up_prompt.as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );
    app.refresh_goal(&agent);
    let _ = app.enqueue_prompt_front(resume.follow_up_prompt.unwrap());
    assert_eq!(
        app.queue.pop_front().as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn goal_unattended_toggles_on_the_active_goal() {
    let (root, config) = goal_test_config("unattended");
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    agent
        .set_structured_goal(Some(hi_agent::Goal::new(
            "ship it",
            vec!["implement it".into()],
        )))
        .unwrap();
    let mut app = test_app("custom", "test-model");

    app.handle_command(&mut agent, hi_agent::Command::Goal("unattended on".into()))
        .await;
    assert!(agent.structured_goal().is_some_and(|g| g.unattended));

    app.handle_command(&mut agent, hi_agent::Command::Goal("unattended off".into()))
        .await;
    assert!(agent.structured_goal().is_some_and(|g| !g.unattended));

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn goal_workflow_without_checklist_errors_and_does_not_install() {
    let (root, config) = goal_test_config("workflow-missing");
    let mut agent = hi_agent::Agent::new(goal_test_provider(), config).unwrap();
    let mut app = test_app("custom", "test-model");

    app.handle_command(
        &mut agent,
        hi_agent::Command::Goal("--workflow missing.md".into()),
    )
    .await;

    assert!(agent.structured_goal().is_none());
    let transcript = app.transcript_text();
    assert!(
        transcript.contains("--workflow needs a checklist"),
        "{transcript}"
    );

    drop(agent);
    let _ = std::fs::remove_dir_all(root);
}
