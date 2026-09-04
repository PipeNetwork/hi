//! Fail-closed daemon owner shutdown.

use std::future::Future;

use anyhow::{Result, anyhow};

use super::{RemoteSessionSink, RemoteUi};

pub(super) async fn complete_daemon_run(
    agent: &mut hi_agent::Agent,
    result: Result<()>,
    sync_handle: Option<&std::sync::Arc<RemoteSessionSink>>,
    remote_ui: Option<&std::sync::Arc<RemoteUi>>,
    session_id: &str,
) -> Result<()> {
    let graceful = result.is_ok();
    complete_daemon_exit(agent, result, || async move {
        if !graceful {
            return;
        }
        println!("\x1b[2m⟳ daemon workspace settled — ending remote session\x1b[0m");
        if let Some(handle) = sync_handle {
            if let Err(error) = handle.flush().await {
                eprintln!("\x1b[33msync: {error:#}\x1b[0m");
            }
            handle.end_session().await;
        }
        if let Some(remote_ui) = remote_ui
            && let Err(error) = remote_ui.flush().await
        {
            eprintln!("\x1b[33msync events: {error:#}\x1b[0m");
        }
        if let Some(directory) = crate::session::sessions_dir() {
            let _ = std::fs::remove_file(directory.join(format!("{session_id}.token")));
        }
    })
    .await
}

pub(super) async fn complete_daemon_exit<F, Fut>(
    agent: &mut hi_agent::Agent,
    result: Result<()>,
    after_settlement: F,
) -> Result<()>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = ()>,
{
    if let Err(settlement) = agent.settle_workspace_for_exit().await {
        return match result {
            Ok(()) => Err(settlement.context(
                "daemon workspace shutdown did not settle; the remote session was not ended. \
                 Inspect `hi workspace status` and `hi workspace recover list`, recover the \
                 workspace, then stop the daemon again",
            )),
            Err(cause) => Err(anyhow!(
                "daemon stopped after {cause:#}, and workspace shutdown did not settle; the \
                 remote session was not ended. Inspect `hi workspace status` and \
                 `hi workspace recover list`, recover the workspace, then stop the daemon \
                 again: {settlement:#}"
            )),
        };
    }

    after_settlement().await;
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;
    use hi_ai::{Completion, Provider, StreamEvent};
    use hi_workspace::{
        InMemoryWorkspaceController, MutationIntent, WorkspaceController, WorkspaceState,
    };

    use super::*;

    struct NeverProvider;

    #[async_trait]
    impl Provider for NeverProvider {
        async fn stream(
            &self,
            _request: hi_ai::ChatRequest,
            _on_event: &mut (dyn FnMut(StreamEvent) + Send),
        ) -> Result<Completion> {
            unreachable!("daemon shutdown tests do not make model requests")
        }
    }

    fn subject() -> (
        tempfile::TempDir,
        tempfile::TempDir,
        hi_agent::Agent,
        Arc<InMemoryWorkspaceController>,
    ) {
        let workspace = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut config = hi_agent::AgentConfig::default();
        config.paths.workspace_root = workspace.path().to_path_buf();
        config.paths.state_root = state.path().to_path_buf();
        config.gates.lsp_mode = hi_agent::LspMode::Off;
        config.gates.verification = hi_agent::VerificationMode::Disabled;
        config.sandbox_policy = Some(hi_tools::sandbox::SandboxPolicy::Off);
        config.suppress_initial_project_hooks = true;
        let agent = hi_agent::Agent::new(Arc::new(NeverProvider), config).unwrap();
        let controller = Arc::new(InMemoryWorkspaceController::new_local(
            "daemon-exit",
            workspace.path(),
            state.path(),
        ));
        agent
            .install_workspace_controller(controller.clone())
            .unwrap();
        (workspace, state, agent, controller)
    }

    #[tokio::test]
    async fn fatal_exit_settles_jobs_before_post_barrier_cleanup() {
        let (_workspace, _state, mut agent, controller) = subject();
        let tasks = agent.background_task_registry();
        let task_id = tasks
            .spawn(
                "pending daemon reader",
                "explore",
                Box::new(|| Box::pin(std::future::pending())),
            )
            .await
            .unwrap();
        let cleanup_ran = Arc::new(AtomicBool::new(false));
        let cleanup_flag = cleanup_ran.clone();
        let cleanup_controller = controller.clone();

        let error = complete_daemon_exit(
            &mut agent,
            Err(anyhow!("terminal poll")),
            move || async move {
                assert_eq!(cleanup_controller.status().state, WorkspaceState::Ready);
                assert!(cleanup_controller.status().active_jobs.is_empty());
                cleanup_flag.store(true, Ordering::Release);
            },
        )
        .await
        .unwrap_err();

        assert!(format!("{error:#}").contains("terminal poll"));
        assert!(cleanup_ran.load(Ordering::Acquire));
        assert!(
            tasks
                .poll(&task_id, std::time::Duration::ZERO)
                .await
                .unwrap()
                .state
                .is_terminal()
        );
    }

    #[tokio::test]
    async fn failed_barrier_never_runs_remote_end_cleanup() {
        let (_workspace, _state, mut agent, controller) = subject();
        let permit = controller
            .begin(MutationIntent::workspace("ambiguous daemon write"))
            .await
            .unwrap();
        drop(permit);
        let cleanup_ran = Arc::new(AtomicBool::new(false));
        let cleanup_flag = cleanup_ran.clone();

        let error = complete_daemon_exit(&mut agent, Ok(()), move || async move {
            cleanup_flag.store(true, Ordering::Release);
        })
        .await
        .unwrap_err();

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("remote session was not ended"),
            "{rendered}"
        );
        assert!(rendered.contains("hi workspace recover list"), "{rendered}");
        assert!(!cleanup_ran.load(Ordering::Acquire));
        assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    }
}
