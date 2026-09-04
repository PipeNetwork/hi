use std::sync::Arc;

use agent_client_protocol as acp;
use hi_workspace::{InMemoryWorkspaceController, WorkspaceController, WorkspaceState};

use super::*;

struct NeverProvider;

#[async_trait::async_trait]
impl Provider for NeverProvider {
    async fn stream(
        &self,
        _request: hi_ai::ChatRequest,
        _on_event: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> anyhow::Result<hi_ai::Completion> {
        unreachable!("close-session tests do not make model requests")
    }
}

fn config(workspace: &Path, state: &Path) -> AgentConfig {
    let mut config = AgentConfig::default();
    config.paths.workspace_root = workspace.to_path_buf();
    config.paths.state_root = state.to_path_buf();
    config.gates.lsp_mode = hi_agent::LspMode::Off;
    config.gates.verification = hi_agent::VerificationMode::Disabled;
    config.sandbox_policy = Some(hi_tools::sandbox::SandboxPolicy::Off);
    config.suppress_initial_project_hooks = true;
    config
}

async fn insert_session(
    shell: &HiShell,
    session_id: &acp::SessionId,
    agent: Agent,
) -> Arc<Session> {
    let session = Arc::new(Session {
        agent: Mutex::new(agent),
        active_turn: Mutex::new(None),
        closed: AtomicBool::new(false),
    });
    shell
        .sessions
        .lock()
        .await
        .insert(session_id.clone(), session.clone());
    session
}

#[tokio::test]
async fn close_waits_for_task_settlement_and_exit_barrier_before_acknowledging() {
    let workspace = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let config = config(workspace.path(), state.path());
    let provider: Arc<dyn Provider> = Arc::new(NeverProvider);
    let agent = Agent::new(provider.clone(), config.clone()).unwrap();
    let controller = Arc::new(InMemoryWorkspaceController::new_local(
        "acp-close",
        workspace.path(),
        state.path(),
    ));
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let tasks = agent.background_task_registry();
    let task_id = tasks
        .spawn(
            "pending reader",
            "explore",
            Box::new(|| Box::pin(std::future::pending::<hi_tools::BackgroundTaskOutcome>())),
        )
        .await
        .unwrap();
    let shell = HiShell::new(ShellConfig {
        provider,
        template: config,
        models: Vec::new(),
    });
    let session_id = acp::SessionId::new("close-success");
    insert_session(&shell, &session_id, agent).await;

    acp::Agent::close_session(&shell, acp::CloseSessionRequest::new(session_id.clone()))
        .await
        .unwrap();

    assert!(!shell.sessions.lock().await.contains_key(&session_id));
    assert!(shell.snapshots.lock().await.contains_key(&session_id));
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert!(controller.status().active_jobs.is_empty());
    assert_eq!(
        tasks
            .poll(&task_id, std::time::Duration::ZERO)
            .await
            .unwrap()
            .state,
        hi_tools::BackgroundTaskState::Cancelled
    );
}

#[tokio::test]
async fn close_failure_keeps_the_session_fenced_and_returns_recovery_guidance() {
    let workspace = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let config = config(workspace.path(), state.path());
    let provider: Arc<dyn Provider> = Arc::new(NeverProvider);
    let agent = Agent::new(provider.clone(), config.clone()).unwrap();
    let controller = Arc::new(InMemoryWorkspaceController::new_local(
        "acp-close-recovery",
        workspace.path(),
        state.path(),
    ));
    let permit = controller
        .begin(hi_workspace::MutationIntent::workspace("uncertain write"))
        .await
        .unwrap();
    drop(permit);
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let shell = HiShell::new(ShellConfig {
        provider,
        template: config,
        models: Vec::new(),
    });
    let session_id = acp::SessionId::new("close-recovery");
    let session = insert_session(&shell, &session_id, agent).await;

    let error =
        acp::Agent::close_session(&shell, acp::CloseSessionRequest::new(session_id.clone()))
            .await
            .unwrap_err();

    let rendered = format!("{error:?}");
    assert!(rendered.contains("hi workspace status"), "{rendered}");
    assert!(rendered.contains("retry close"), "{rendered}");
    assert!(shell.sessions.lock().await.contains_key(&session_id));
    assert!(session.closed.load(Ordering::Acquire));
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    assert!(!shell.snapshots.lock().await.contains_key(&session_id));

    let recovery_id = controller.status().recovery_id.unwrap();
    let recovery = controller.reconcile(recovery_id).await;
    assert_eq!(recovery.status, hi_workspace::RecoveryStatus::Recovered);
    acp::Agent::close_session(&shell, acp::CloseSessionRequest::new(session_id.clone()))
        .await
        .unwrap();
    assert!(!shell.sessions.lock().await.contains_key(&session_id));
    assert!(shell.snapshots.lock().await.contains_key(&session_id));
}

#[tokio::test]
async fn transport_close_settles_every_session_before_dropping_the_shell() {
    let workspace_one = tempfile::tempdir().unwrap();
    let state_one = tempfile::tempdir().unwrap();
    let workspace_two = tempfile::tempdir().unwrap();
    let state_two = tempfile::tempdir().unwrap();
    let provider: Arc<dyn Provider> = Arc::new(NeverProvider);
    let agent_one = Agent::new(
        provider.clone(),
        config(workspace_one.path(), state_one.path()),
    )
    .unwrap();
    let controller_one = Arc::new(InMemoryWorkspaceController::new_local(
        "acp-eof-one",
        workspace_one.path(),
        state_one.path(),
    ));
    agent_one
        .install_workspace_controller(controller_one.clone())
        .unwrap();
    let tasks = agent_one.background_task_registry();
    let task_id = tasks
        .spawn(
            "pending reader at EOF",
            "explore",
            Box::new(|| Box::pin(std::future::pending::<hi_tools::BackgroundTaskOutcome>())),
        )
        .await
        .unwrap();
    let agent_two = Agent::new(
        provider.clone(),
        config(workspace_two.path(), state_two.path()),
    )
    .unwrap();
    let controller_two = Arc::new(InMemoryWorkspaceController::new_local(
        "acp-eof-two",
        workspace_two.path(),
        state_two.path(),
    ));
    agent_two
        .install_workspace_controller(controller_two.clone())
        .unwrap();
    let shell = HiShell::new(ShellConfig {
        provider,
        template: config(workspace_one.path(), state_one.path()),
        models: Vec::new(),
    });
    let session_one = acp::SessionId::new("eof-one");
    let session_two = acp::SessionId::new("eof-two");
    insert_session(&shell, &session_one, agent_one).await;
    insert_session(&shell, &session_two, agent_two).await;

    shell
        .settle_all_sessions_for_transport_close()
        .await
        .unwrap();

    assert!(shell.sessions.lock().await.is_empty());
    let snapshots = shell.snapshots.lock().await;
    assert!(snapshots.contains_key(&session_one));
    assert!(snapshots.contains_key(&session_two));
    assert!(controller_one.status().active_jobs.is_empty());
    assert!(controller_two.status().active_jobs.is_empty());
    assert_eq!(
        tasks
            .poll(&task_id, std::time::Duration::ZERO)
            .await
            .unwrap()
            .state,
        hi_tools::BackgroundTaskState::Cancelled
    );
}

#[tokio::test]
async fn transport_close_retains_failed_session_but_finishes_independent_sessions() {
    let failed_workspace = tempfile::tempdir().unwrap();
    let failed_state = tempfile::tempdir().unwrap();
    let clean_workspace = tempfile::tempdir().unwrap();
    let clean_state = tempfile::tempdir().unwrap();
    let provider: Arc<dyn Provider> = Arc::new(NeverProvider);
    let failed_agent = Agent::new(
        provider.clone(),
        config(failed_workspace.path(), failed_state.path()),
    )
    .unwrap();
    let failed_controller = Arc::new(InMemoryWorkspaceController::new_local(
        "acp-eof-failed",
        failed_workspace.path(),
        failed_state.path(),
    ));
    let permit = failed_controller
        .begin(hi_workspace::MutationIntent::workspace(
            "uncertain EOF write",
        ))
        .await
        .unwrap();
    drop(permit);
    failed_agent
        .install_workspace_controller(failed_controller)
        .unwrap();
    let clean_agent = Agent::new(
        provider.clone(),
        config(clean_workspace.path(), clean_state.path()),
    )
    .unwrap();
    let shell = HiShell::new(ShellConfig {
        provider,
        template: config(clean_workspace.path(), clean_state.path()),
        models: Vec::new(),
    });
    let failed_id = acp::SessionId::new("eof-failed");
    let clean_id = acp::SessionId::new("eof-clean");
    let failed_session = insert_session(&shell, &failed_id, failed_agent).await;
    insert_session(&shell, &clean_id, clean_agent).await;

    let error = shell
        .settle_all_sessions_for_transport_close()
        .await
        .unwrap_err();

    let rendered = format!("{error:#}");
    assert!(rendered.contains("eof-failed"), "{rendered}");
    assert!(rendered.contains("hi workspace recover list"), "{rendered}");
    assert!(shell.sessions.lock().await.contains_key(&failed_id));
    assert!(failed_session.closed.load(Ordering::Acquire));
    assert!(!shell.sessions.lock().await.contains_key(&clean_id));
    assert!(shell.snapshots.lock().await.contains_key(&clean_id));
    assert!(!shell.snapshots.lock().await.contains_key(&failed_id));
}
