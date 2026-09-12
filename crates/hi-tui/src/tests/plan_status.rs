use super::*;

#[test]
fn plan_mode_paints_plan_flag_on_the_composer() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan_mode = true;
    app.permission_mode = hi_agent::PermissionMode::Ask;
    let mut term = Terminal::new(TestBackend::new(80, 16)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("plan"), "plan flag shown:\n{screen}");
    assert!(
        !screen.contains("always-approve"),
        "always-approve hidden while planning:\n{screen}"
    );
}

#[test]
fn leaving_plan_with_leftover_opens_approval_card() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan_mode = true;
    app.permission_mode = hi_agent::PermissionMode::Ask;
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.cycle_session_face();
    assert_eq!(app.session_face(), crate::session_face::SessionFace::Always);
    assert!(app.plan_approval.is_some());
}

#[test]
fn plan_approval_card_renders_choices() {
    let mut app = test_app("openai", "gpt-4o");
    app.plan = vec![hi_agent::PlanStep {
        title: "wire the scheduler".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.open_plan_approval();
    let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
    term.draw(|f| app.render(f)).unwrap();
    let screen = dump(&term);
    assert!(screen.contains("Plan approval"), "{screen}");
    assert!(screen.contains("Approve"), "{screen}");
    assert!(screen.contains("Request changes"), "{screen}");
    assert!(screen.contains("wire the scheduler"), "{screen}");
}

#[tokio::test]
async fn plan_status_preserves_paused_drive_without_opening_approval() {
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
        ..hi_agent::AgentConfig::default()
    };
    let mut agent = hi_agent::Agent::new(provider, config).unwrap();
    let plan = vec![hi_agent::PlanStep {
        title: "Durable pending smoke step".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    agent.restore_plan(plan.clone());
    agent.restore_plan_drive(true, 1, Vec::new());
    let mut app = test_app("openai", "test-model");
    app.plan = plan;
    app.refresh_goal(&agent);

    app.handle_command(&mut agent, hi_agent::Command::Plan("status".into()))
        .await;

    assert!(app.plan_approval.is_none());
    assert!(agent.plan_drive_paused());
    assert_eq!(agent.plan_drive_stall(), 1);
    assert!(app.transcript_text().contains("Durable pending smoke step"));
    assert!(app.transcript_text().contains("plan drive: paused"));
    assert!(app.queue.is_empty());

    // Explicitly viewing the checklist keeps its existing approval behavior.
    app.handle_command(&mut agent, hi_agent::Command::Plan("show".into()))
        .await;
    assert!(app.plan_approval.is_some());
}
