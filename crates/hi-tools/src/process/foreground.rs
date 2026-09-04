use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::Notify;

/// Workspace-scoped inventory of foreground process groups.
///
/// Durability controllers clone this handle so a pushed lease-loss event can
/// stop commands which are still executing inside the materialized workspace.
#[derive(Clone, Debug, Default)]
pub struct ForegroundProcessRegistry {
    inner: Arc<RegistryInner>,
}

#[derive(Debug, Default)]
struct RegistryInner {
    groups: Mutex<BTreeSet<i32>>,
    kill_latched: AtomicBool,
    changed: Notify,
}

impl ForegroundProcessRegistry {
    pub(crate) fn register(&self, child: &tokio::process::Child) -> ForegroundProcessRegistration {
        #[cfg(unix)]
        let pgid = child.id().map(|pid| pid as i32);
        #[cfg(not(unix))]
        let pgid = None;
        if let Some(pgid) = pgid {
            self.inner
                .groups
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(pgid);
            if self.inner.kill_latched.load(Ordering::Acquire) {
                super::kill_group(pgid);
            }
        }
        ForegroundProcessRegistration {
            registry: self.clone(),
            pgid,
        }
    }

    pub fn active_count(&self) -> usize {
        self.inner
            .groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Signal the process groups which are live right now without fencing
    /// future commands in this workspace runtime.
    ///
    /// This is the turn-cancellation counterpart to [`Self::kill_all`]. Lease
    /// loss is permanent for the runtime and therefore latches; cancelling one
    /// turn must leave the next turn able to spawn processes normally.
    pub fn kill_current(&self) {
        let groups = self
            .inner
            .groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .copied()
            .collect::<Vec<_>>();
        for pgid in groups {
            super::kill_group(pgid);
        }
    }

    /// Signal every foreground process group which was live at this boundary.
    /// The owning capture futures remain responsible for observing and reaping
    /// their direct children.
    pub fn kill_all(&self) {
        self.inner.kill_latched.store(true, Ordering::Release);
        self.kill_current();
    }

    /// Wait until capture owners have reaped and unregistered all children.
    pub async fn wait_until_empty(&self, timeout: Duration) -> bool {
        let wait = async {
            loop {
                let changed = self.inner.changed.notified();
                if self.active_count() == 0 {
                    return;
                }
                changed.await;
            }
        };
        tokio::time::timeout(timeout, wait).await.is_ok()
    }
}

/// RAII membership token transferred through the foreground-to-background
/// adoption boundary. Dropping it only removes registry membership; process
/// ownership and kill-on-drop remain separate.
#[derive(Debug)]
pub struct ForegroundProcessRegistration {
    registry: ForegroundProcessRegistry,
    pgid: Option<i32>,
}

impl Drop for ForegroundProcessRegistration {
    fn drop(&mut self) {
        let Some(pgid) = self.pgid.take() else {
            return;
        };
        self.registry
            .inner
            .groups
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&pgid);
        self.registry.inner.changed.notify_one();
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn external_kill_stops_and_reaps_foreground_group() {
        let root = tempfile::tempdir().unwrap();
        let runner =
            crate::ProcessRunner::new_with_policy(root.path(), crate::sandbox::SandboxPolicy::Off)
                .unwrap();
        let registry = runner.foreground_registry();
        let execution =
            tokio::spawn(async move { runner.run_shell_maybe_timeout("sleep 600", None).await });
        tokio::time::timeout(Duration::from_secs(2), async {
            while registry.active_count() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("foreground process must register");

        registry.kill_all();
        assert!(
            registry.wait_until_empty(Duration::from_secs(2)).await,
            "capture owner must reap and unregister the killed process"
        );
        let outcome = execution.await.unwrap().unwrap();
        assert_eq!(outcome.status, crate::ToolStatus::Failed);

        let late = runner_for_latched_registry(&registry, root.path());
        assert_eq!(late.await.status, crate::ToolStatus::Failed);
    }

    async fn runner_for_latched_registry(
        registry: &ForegroundProcessRegistry,
        root: &std::path::Path,
    ) -> crate::ProcessExecution {
        let mut runner =
            crate::ProcessRunner::new_with_policy(root, crate::sandbox::SandboxPolicy::Off)
                .unwrap();
        runner.foreground = registry.clone();
        tokio::time::timeout(
            Duration::from_secs(2),
            runner.run_shell_maybe_timeout("sleep 600", None),
        )
        .await
        .expect("a process registered after lease loss must be killed")
        .unwrap()
    }
}
