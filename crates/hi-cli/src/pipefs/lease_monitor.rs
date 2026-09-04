use std::sync::{Arc, Mutex};

use tokio_util::sync::CancellationToken;

use crate::sync::RemoteSessionSink;

pub(super) fn build_durability(
    workspace: hi_pipefs::PipeFsWorkspace,
    sync: Arc<RemoteSessionSink>,
    background_processes: Arc<hi_tools::BackgroundRegistry>,
    foreground_processes: hi_tools::ForegroundProcessRegistry,
) -> Arc<super::PipeFsDurability> {
    let failure: Arc<Mutex<Option<String>>> = Arc::default();
    let monitor = LeaseLossMonitor::start(
        workspace.clone(),
        sync.clone(),
        background_processes.clone(),
        foreground_processes,
        failure.clone(),
    );
    Arc::new(super::PipeFsDurability {
        workspace,
        sync,
        background_processes,
        background_checkpoints: Arc::default(),
        failure,
        _lease_loss_monitor: monitor,
    })
}

/// Bridges the transcript writer's pushed lease-loss signal into the PipeFS
/// byte controller. It owns no lease and exits with the durability backend.
pub(super) struct LeaseLossMonitor {
    cancel: CancellationToken,
}

impl LeaseLossMonitor {
    pub(super) fn start(
        workspace: hi_pipefs::PipeFsWorkspace,
        sync: Arc<RemoteSessionSink>,
        background: Arc<hi_tools::BackgroundRegistry>,
        foreground: hi_tools::ForegroundProcessRegistry,
        failure: Arc<Mutex<Option<String>>>,
    ) -> Self {
        Self::start_with_status(
            workspace,
            sync.subscribe_writer_lease_status(),
            background,
            foreground,
            failure,
        )
    }

    fn start_with_status(
        workspace: hi_pipefs::PipeFsWorkspace,
        mut lease_status: tokio::sync::watch::Receiver<hi_pipefs::PipeFsLeaseStatus>,
        background: Arc<hi_tools::BackgroundRegistry>,
        foreground: hi_tools::ForegroundProcessRegistry,
        failure: Arc<Mutex<Option<String>>>,
    ) -> Self {
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        tokio::spawn(async move {
            let mut observed = *lease_status.borrow_and_update();
            if observed == hi_pipefs::PipeFsLeaseStatus::Valid {
                loop {
                    tokio::select! {
                        _ = task_cancel.cancelled() => return,
                        changed = lease_status.changed() => {
                            if changed.is_err() {
                                return;
                            }
                            observed = *lease_status.borrow_and_update();
                            if observed != hi_pipefs::PipeFsLeaseStatus::Valid {
                                break;
                            }
                        }
                    }
                }
            }

            let detail = match observed {
                hi_pipefs::PipeFsLeaseStatus::Lost => {
                    let reason =
                        "the shared HI writer lease was taken over by another machine";
                    match workspace.mark_lease_lost(reason).await {
                        Ok(()) => format!("lease_lost: {reason}"),
                        Err(error) => {
                            format!("lease_lost: {reason}; recovery marker failed: {error:#}")
                        }
                    }
                }
                hi_pipefs::PipeFsLeaseStatus::Uncertain => {
                    "lease_uncertain: the shared HI writer lease could not be refreshed; live writers were stopped"
                        .to_string()
                }
                hi_pipefs::PipeFsLeaseStatus::Valid => return,
            };
            *failure
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(detail);
            stop_workspace_processes(&foreground, &background).await;
        });
        Self { cancel }
    }
}

async fn stop_workspace_processes(
    foreground: &hi_tools::ForegroundProcessRegistry,
    background: &hi_tools::BackgroundRegistry,
) {
    foreground.kill_all();
    background.kill_all();
    let _ = tokio::join!(
        foreground.wait_until_empty(std::time::Duration::from_secs(5)),
        background.ensure_quiescent_and_reaped(),
    );
}

impl Drop for LeaseLossMonitor {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn inert_workspace(root: &std::path::Path) -> hi_pipefs::PipeFsWorkspace {
        let original = root.join("original");
        let state = root.join("state");
        std::fs::create_dir_all(&original).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        let client = hi_pipefs::PipeFsClient::new(hi_pipefs::PipeFsClientConfig::new(
            "http://127.0.0.1:1",
            "test-key",
        ))
        .unwrap();
        let cache_scope = client.cache_scope();
        hi_pipefs::PipeFsWorkspace::new(
            client,
            hi_pipefs::PipeFsLease {
                token: "lease-token".into(),
                generation: 3,
            },
            hi_pipefs::PipeFsWorkspaceConfig {
                session_id: "lease-uncertain-test".into(),
                cache_scope,
                original_workspace_root: original,
                original_state_root: state,
                cache_base: Some(root.join("cache")),
            },
        )
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lease_loss_stop_kills_foreground_writer_before_its_effect() {
        let root = tempfile::tempdir().unwrap();
        let runner = hi_tools::ProcessRunner::new_with_policy(
            root.path(),
            hi_tools::sandbox::SandboxPolicy::Off,
        )
        .unwrap();
        let foreground = runner.foreground_registry();
        let background = hi_tools::BackgroundRegistry::default();
        let execution = tokio::spawn(async move {
            runner
                .run_shell_maybe_timeout("sleep 0.2; touch late-write", None)
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while foreground.active_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("foreground writer must register");

        stop_workspace_processes(&foreground, &background).await;
        let outcome = execution.await.unwrap().unwrap();
        assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(!root.path().join("late-write").exists());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lease_uncertainty_signal_fences_and_reaps_foreground_writers() {
        let root = tempfile::tempdir().unwrap();
        let runner = hi_tools::ProcessRunner::new_with_policy(
            root.path(),
            hi_tools::sandbox::SandboxPolicy::Off,
        )
        .unwrap();
        let foreground = runner.foreground_registry();
        let background = Arc::new(hi_tools::BackgroundRegistry::default());
        let failure = Arc::new(Mutex::new(None));
        let (lease_status, receiver) =
            tokio::sync::watch::channel(hi_pipefs::PipeFsLeaseStatus::Valid);
        let _monitor = LeaseLossMonitor::start_with_status(
            inert_workspace(root.path()),
            receiver,
            background,
            foreground.clone(),
            failure.clone(),
        );
        let execution = tokio::spawn(async move {
            runner
                .run_shell_maybe_timeout("sleep 1; touch uncertain-write", None)
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while foreground.active_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("foreground writer must register");

        lease_status.send_replace(hi_pipefs::PipeFsLeaseStatus::Uncertain);
        let outcome = execution.await.unwrap().unwrap();
        assert_eq!(outcome.status, hi_tools::ToolStatus::Failed);
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert!(!root.path().join("uncertain-write").exists());
        assert!(
            failure
                .lock()
                .unwrap()
                .as_deref()
                .is_some_and(|detail| detail.contains("lease_uncertain"))
        );
    }
}
