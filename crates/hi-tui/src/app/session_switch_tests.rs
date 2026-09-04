use crate::tests::test_app;

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
        gates: hi_agent::AgentGates {
            lsp_mode: hi_agent::LspMode::Off,
            ..hi_agent::AgentGates::default()
        },
        ..hi_agent::AgentConfig::default()
    };
    let agent = hi_agent::Agent::new(provider, config).unwrap();
    let mut app = test_app("custom", "test-model");
    app.auto_approve_session = true;
    app.add_auto_approve_path("src/previous.rs");
    app.add_auto_approve_mcp("source-control".into(), "publish".into());
    app.plan = vec![hi_agent::PlanStep {
        title: "previous session task".into(),
        status: hi_agent::PlanStatus::Pending,
    }];
    app.open_plan_approval();
    app.plan_approval
        .as_mut()
        .unwrap()
        .comments
        .push(crate::plan_approval::PlanComment {
            step: 0,
            text: "previous session comment".into(),
        });
    app.plan_mode = true;
    app.permission_mode = hi_agent::PermissionMode::Always;
    app.plan_drive_paused = true;
    app.plan_drive_pause_dirty = true;
    (root, agent, app)
}

#[tokio::test]
async fn successful_switch_clears_previous_session_grants_and_pending_ui_choices() {
    let (_root, mut agent, mut app) = fixture();
    app.session_switcher = Some(Box::new(|id, agent| {
        Box::pin(async move {
            agent.set_plan_mode(false);
            agent.set_permission_mode(hi_agent::PermissionMode::Ask);
            agent.restore_plan(vec![hi_agent::PlanStep {
                title: "destination session task".into(),
                status: hi_agent::PlanStatus::Pending,
            }]);
            agent.restore_plan_approval_parked(true);
            Ok(crate::SessionSwitchInfo {
                id: id.into(),
                summary: "destination history".into(),
            })
        })
    }));

    app.switch_session(&mut agent, "destination-session").await;

    assert!(!app.auto_approve_session);
    assert!(!app.path_auto_approved("src/new.rs"));
    assert!(!app.mcp_auto_approved("source-control", "publish"));
    assert!(!app.session_face_dirty);
    assert!(!app.plan_drive_pause_dirty);
    assert!(!app.plan_mode);
    assert!(!app.plan_drive_paused);
    assert_eq!(app.permission_mode, hi_agent::PermissionMode::Ask);
    assert_eq!(app.plan[0].title, "destination session task");
    let card = app.plan_approval.as_ref().unwrap();
    assert!(card.parked);
    assert!(card.comments.is_empty());

    // A later turn must not push the old session's deferred state into this one.
    assert!(app.push_session_face(&mut agent));
    assert!(!agent.plan_mode());
    assert!(!agent.plan_drive_paused());
    assert!(agent.plan_approval_parked());
    assert_eq!(agent.permission_mode(), hi_agent::PermissionMode::Ask);
}

#[tokio::test]
async fn failed_switch_preserves_current_session_grants_and_pending_ui_choices() {
    let (_root, mut agent, mut app) = fixture();
    app.session_switcher = Some(Box::new(|_, _| {
        Box::pin(async { anyhow::bail!("destination unavailable") })
    }));

    app.switch_session(&mut agent, "destination-session").await;

    assert!(app.auto_approve_session);
    assert!(app.path_auto_approved("src/new.rs"));
    assert!(app.mcp_auto_approved("source-control", "publish"));
    assert!(app.session_face_dirty);
    assert!(app.plan_drive_pause_dirty);
    assert!(app.plan_mode);
    assert!(app.plan_drive_paused);
    assert_eq!(app.plan[0].title, "previous session task");
    let card = app.plan_approval.as_ref().unwrap();
    assert!(!card.parked);
    assert_eq!(card.comments[0].text, "previous session comment");
    assert!(app.transcript_text().contains("destination unavailable"));
}
