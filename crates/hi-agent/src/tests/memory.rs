use super::common::*;
use super::*;
use hi_workspace::{
    ExecutionReport, InMemoryWorkspaceController, MutationIntent, WorkspaceController,
};

#[tokio::test]
async fn update_memory_writes_file_without_polluting_history() {
    // Use a unique workspace so the project-memory override remains inside
    // the controller's authoritative root.
    let dir = std::env::temp_dir().join(format!(
        "hi-mem-write-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("memory.md");
    let _ = std::fs::remove_file(&path);
    // The model returns a distilled bullet list.
    let mut cfg = config();
    cfg.paths.workspace_root = dir.clone();
    cfg.paths.state_root = dir.join(".state");
    let mut agent = agent(
        vec![completion(
            vec![Content::Text(
                "- always run cargo fmt\n- tests live in tests/".into(),
            )],
            7,
            4,
        )],
        cfg,
    );
    agent
        .messages_mut()
        .push(Message::user("Actually, always run cargo fmt"));
    let before = agent.messages().len();
    agent.update_memory_at(path.clone(), &mut NullUi).await;

    let written = std::fs::read_to_string(&path).expect("memory file written");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        written.contains("always run cargo fmt"),
        "distilled: {written}"
    );
    assert_eq!(
        agent.messages().len(),
        before,
        "session history not polluted"
    );
    assert_eq!(agent.totals().output_tokens, 4, "usage counted");
}

#[tokio::test]
async fn update_memory_persists_usage_without_new_messages() {
    let dir = std::env::temp_dir().join(format!(
        "hi-memory-persist-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("memory.md");
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let mut cfg = config();
    cfg.paths.workspace_root = dir.clone();
    cfg.paths.state_root = dir.join(".state");
    let mut agent = agent(
        vec![completion(vec![Content::Text("- note".into())], 10, 5)],
        cfg,
    );
    agent
        .messages_mut()
        .push(Message::user("I prefer a short durable note"));
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));

    agent.update_memory_at(path.clone(), &mut NullUi).await;
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        *records.lock().unwrap(),
        vec![Usage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        }]
    );
}

#[tokio::test]
async fn project_memory_override_outside_workspace_fails_closed() {
    let cfg = config();
    let external = std::env::temp_dir().join(format!(
        "hi-external-project-memory-{}-{}.md",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut agent = agent(
        vec![completion(
            vec![Content::Text(
                "- Error messages include actionable recovery guidance".into(),
            )],
            1,
            1,
        )],
        cfg,
    );
    agent.messages_mut().push(Message::user(
        "Actually, include actionable recovery guidance",
    ));
    let mut ui = RecordingUi::default();

    agent.update_memory_at(external.clone(), &mut ui).await;

    assert!(!external.exists());
    assert!(
        ui.statuses.iter().any(|status| {
            status.contains("project memory not saved") && status.contains("host-local state")
        }),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn legacy_pipefs_disables_unstaged_project_memory_writes() {
    let mut cfg = config();
    cfg.harness.features.workspace_controller_v2 = false;
    let root = cfg.paths.workspace_root.clone();
    let state = cfg.paths.state_root.clone();
    let path = root.join(".hi/memory.md");
    let mut agent = agent(
        vec![completion(
            vec![Content::Text(
                "- Error messages include actionable recovery guidance".into(),
            )],
            1,
            1,
        )],
        cfg,
    );
    agent.messages_mut().push(Message::user(
        "Actually, include actionable recovery guidance",
    ));
    agent
        .install_workspace_controller(std::sync::Arc::new(
            InMemoryWorkspaceController::new_pipefs(
                "legacy-memory-workspace",
                "legacy-memory-session",
                1,
                false,
                &root,
                &state,
            ),
        ))
        .unwrap();
    let mut ui = RecordingUi::default();

    agent.update_memory_at(path.clone(), &mut ui).await;

    assert!(!path.exists());
    assert!(
        ui.statuses.iter().any(|status| {
            status.contains("project memory not saved")
                && status.contains("requires workspace_controller_v2 under PipeFS")
        }),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn update_memory_is_best_effort_on_error() {
    // A provider error at quit must not panic or leave a file behind.
    let path = std::env::temp_dir().join(format!("hi-mem-{}-err.md", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let (mut agent, _requests) = scripted_agent(
        vec![ProviderStep::Error(ProviderErrorKind::Outage)],
        config(),
    );
    agent
        .messages_mut()
        .push(Message::user("Actually, remember this correction"));
    agent.update_memory_at(path.clone(), &mut NullUi).await;
    assert!(!path.exists(), "nothing written when distillation fails");
}

#[tokio::test]
async fn closed_admission_prevents_project_memory_write_and_success_status() {
    let cfg = config();
    let root = cfg.paths.workspace_root.clone();
    let state = cfg.paths.state_root.clone();
    let path = root.join(".hi/memory.md");
    assert!(!path.exists());
    let mut agent = agent(
        vec![completion(
            vec![Content::Text(
                "- Error messages include actionable recovery guidance".into(),
            )],
            3,
            2,
        )],
        cfg,
    );
    agent.messages_mut().push(Message::user(
        "Actually, include actionable recovery guidance",
    ));
    let controller = std::sync::Arc::new(InMemoryWorkspaceController::new_pipefs(
        "memory-workspace",
        "memory-session",
        2,
        true,
        &root,
        &state,
    ));
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let permit = controller
        .begin(MutationIntent::workspace("existing writer"))
        .await
        .unwrap();
    let mut ui = RecordingUi::default();

    agent.update_memory_at(path.clone(), &mut ui).await;

    assert!(
        !path.exists(),
        "admission must precede the memory effect; statuses: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.starts_with("✓ saved"))
    );
    assert!(
        ui.statuses.iter().any(|status| {
            status.contains("project memory not saved")
                && status.contains("workspace controller refused")
        }),
        "statuses: {:?}",
        ui.statuses
    );
    let settled = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert!(settled.receipt.is_some());
}
