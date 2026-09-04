use super::{dequeue_ready_prompt, handle_idle_plan_approval_key};
use crate::tests::test_app;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

fn fixture() -> (tempfile::TempDir, hi_agent::Agent, crate::App) {
    let root = tempfile::tempdir().unwrap();
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "unused".into(),
    ));
    let config = hi_agent::AgentConfig {
        paths: hi_agent::AgentPaths {
            workspace_root: root.path().to_path_buf(),
            state_root: root.path().join(".hi-state"),
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
    let mut agent = hi_agent::Agent::new(provider, config).unwrap();
    agent.restore_plan(vec![hi_agent::PlanStep {
        title: "implement the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }]);
    let mut app = test_app("custom", "test-model");
    app.refresh_goal(&agent);
    (root, agent, app)
}

#[test]
fn visible_plan_approval_preserves_all_queued_work() {
    let (_root, agent, mut app) = fixture();
    app.queue.push_back("add regression coverage".into());
    app.queue.push_back(hi_agent::PLAN_DRIVE_PROMPT.into());
    app.open_plan_approval();

    assert_eq!(dequeue_ready_prompt(&mut app, &agent), None);
    assert_eq!(app.queue.len(), 2);
    assert_eq!(app.queue[0], "add regression coverage");
    assert_eq!(app.queue[1], hi_agent::PLAN_DRIVE_PROMPT);
}

#[test]
fn parked_plan_approval_discards_synthetic_drive_and_preserves_user_work() {
    let (_root, mut agent, mut app) = fixture();
    app.open_plan_approval();
    app.park_plan_approval(&mut agent);
    app.queue.push_back(hi_agent::PLAN_DRIVE_PROMPT.into());
    app.queue.push_back(hi_agent::GOAL_CONTINUE_PROMPT.into());
    app.queue.push_back("revise the scheduler design".into());

    assert_eq!(
        dequeue_ready_prompt(&mut app, &agent).as_deref(),
        Some("revise the scheduler design")
    );
    assert!(app.queue.is_empty());
    assert!(agent.plan_approval_parked());
}

#[test]
fn queued_plan_drive_cannot_restart_paused_or_planning_work() {
    let (_root, mut agent, mut app) = fixture();
    agent.set_plan_drive_paused(true);
    app.queue.push_back(hi_agent::PLAN_DRIVE_PROMPT.into());
    assert_eq!(dequeue_ready_prompt(&mut app, &agent), None);

    agent.set_plan_drive_paused(false);
    agent.set_plan_mode(true);
    app.queue.push_back(hi_agent::PLAN_DRIVE_PROMPT.into());
    assert_eq!(dequeue_ready_prompt(&mut app, &agent), None);

    agent.set_plan_mode(false);
    agent.clear_pinned_plan();
    app.queue.push_back(hi_agent::PLAN_DRIVE_PROMPT.into());
    assert_eq!(dequeue_ready_prompt(&mut app, &agent), None);
}

#[test]
fn queued_goal_drive_cannot_restart_paused_goal() {
    let (_root, mut agent, mut app) = fixture();
    agent
        .set_structured_goal(Some(hi_agent::Goal::new(
            "ship the scheduler",
            vec!["implement the scheduler".into()],
        )))
        .unwrap();
    agent.set_goal_pause_reason(hi_agent::GoalPauseReason::User);
    app.queue.push_back(hi_agent::GOAL_CONTINUE_PROMPT.into());
    app.queue.push_back("explain the design".into());

    assert_eq!(
        dequeue_ready_prompt(&mut app, &agent).as_deref(),
        Some("explain the design")
    );
}

#[tokio::test]
async fn explicit_plan_resume_still_runs_queued_execution() {
    let (_root, mut agent, mut app) = fixture();
    agent.set_plan_drive_paused(true);
    app.handle_command(&mut agent, hi_agent::Command::Plan("resume".into()))
        .await;

    assert_eq!(
        dequeue_ready_prompt(&mut app, &agent).as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );
}

#[test]
fn idle_approve_starts_execution_and_retains_composer_and_queue() {
    let (_root, mut agent, mut app) = fixture();
    agent.set_plan_mode(true);
    app.refresh_goal(&agent);
    app.open_plan_approval();
    app.input.set("/status");
    app.sync_completion();
    app.queue.push_back("then run all tests".into());

    let prompt = handle_idle_plan_approval_key(
        &mut app,
        &mut agent,
        &KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE),
    );

    assert_eq!(prompt.as_deref(), Some(hi_agent::PLAN_DRIVE_PROMPT));
    assert!(!agent.plan_mode());
    assert!(!agent.plan_approval_parked());
    assert!(app.plan_approval.is_none());
    assert_eq!(app.input.text(), "/status");
    assert_eq!(
        app.queue.front().map(String::as_str),
        Some("then run all tests")
    );
}

#[test]
fn stale_idle_approve_does_not_start_execution() {
    let (_root, mut agent, mut app) = fixture();
    app.open_plan_approval();
    agent.clear_pinned_plan();

    assert_eq!(
        handle_idle_plan_approval_key(
            &mut app,
            &mut agent,
            &KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE),
        ),
        None
    );
    assert!(app.plan.is_empty());
    assert!(app.plan_approval.is_none());
}

#[test]
fn idle_overlay_key_cannot_approve_a_hidden_plan() {
    let (_root, mut agent, mut app) = fixture();
    agent.set_plan_mode(true);
    app.refresh_goal(&agent);
    app.open_plan_approval();
    app.tutorial = Some(crate::tutorial::TutorialOverlay::fresh());
    let key = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
    assert_eq!(
        handle_idle_plan_approval_key(&mut app, &mut agent, &key),
        None
    );
    assert!(agent.plan_mode());
    assert!(app.plan_approval_capturing());
    app.tutorial = None;
    assert_eq!(
        handle_idle_plan_approval_key(&mut app, &mut agent, &key).as_deref(),
        Some(hi_agent::PLAN_DRIVE_PROMPT)
    );
}
