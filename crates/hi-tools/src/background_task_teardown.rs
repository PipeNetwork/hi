//! Cancellation-safe teardown acknowledgement for background task children.

use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

use super::{BackgroundTaskRegistry, BgFuture, WORKER_HANDLE_ACK_TIMEOUT};

#[derive(Clone, Debug, Default)]
pub struct BackgroundTaskTeardown {
    inner: Arc<TeardownInner>,
}

#[derive(Debug, Default)]
struct TeardownInner {
    state: Mutex<TeardownState>,
    changed: Notify,
}

#[derive(Clone, Debug, Default)]
enum TeardownState {
    #[default]
    Unarmed,
    Pending,
    Finished(Result<(), String>),
}

impl BackgroundTaskTeardown {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare that this task now owns child processes which must be reaped.
    pub fn arm(&self) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*state, TeardownState::Unarmed) {
            *state = TeardownState::Pending;
        }
    }

    /// Publish the one authoritative teardown result. Repeated completion is
    /// deliberately idempotent so cancellation and natural-exit races cannot
    /// overwrite one another.
    pub fn finish(&self, result: Result<(), String>) {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if matches!(*state, TeardownState::Finished(_)) {
            return;
        }
        *state = TeardownState::Finished(result);
        drop(state);
        self.inner.changed.notify_waiters();
    }

    pub(crate) async fn wait(&self) -> Result<(), String> {
        loop {
            let changed = self.inner.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let state = self
                .inner
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            match state {
                TeardownState::Unarmed => return Ok(()),
                TeardownState::Finished(result) => return result,
                TeardownState::Pending => changed.await,
            }
        }
    }
}

impl BackgroundTaskRegistry {
    pub async fn spawn_after_with_teardown(
        &self,
        description: &str,
        subagent_type: &str,
        dependencies: &[String],
        teardown: BackgroundTaskTeardown,
        future_factory: Box<dyn FnOnce() -> BgFuture + Send + 'static>,
    ) -> anyhow::Result<String> {
        let worker = self.next_worker_tx().clone();
        self.spawn_after_on_worker(
            description,
            subagent_type,
            dependencies,
            future_factory,
            teardown,
            worker,
            WORKER_HANDLE_ACK_TIMEOUT,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::BackgroundTaskTeardown;

    struct DelayedFinish {
        teardown: BackgroundTaskTeardown,
        finished: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for DelayedFinish {
        fn drop(&mut self) {
            let teardown = self.teardown.clone();
            let finished = self.finished.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(75)).await;
                finished.store(true, std::sync::atomic::Ordering::Release);
                teardown.finish(Ok(()));
            });
        }
    }

    #[tokio::test]
    async fn wait_blocks_only_after_arm_and_first_finish_wins() {
        let unarmed = BackgroundTaskTeardown::new();
        unarmed.wait().await.unwrap();

        let teardown = BackgroundTaskTeardown::new();
        teardown.arm();
        let waiter = tokio::spawn({
            let teardown = teardown.clone();
            async move { teardown.wait().await }
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        teardown.finish(Ok(()));
        teardown.finish(Err("late overwrite".into()));
        waiter.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn task_cancellation_is_not_terminal_before_child_teardown() {
        let registry = std::sync::Arc::new(super::BackgroundTaskRegistry::new());
        let teardown = BackgroundTaskTeardown::new();
        let task_teardown = teardown.clone();
        let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task_finished = finished.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let id = registry
            .spawn_after_with_teardown(
                "reap before cancellation",
                "explore",
                &[],
                teardown,
                Box::new(move || {
                    Box::pin(async move {
                        task_teardown.arm();
                        let _finish = DelayedFinish {
                            teardown: task_teardown,
                            finished: task_finished,
                        };
                        let _ = started_tx.send(());
                        std::future::pending::<()>().await;
                        unreachable!()
                    })
                }),
            )
            .await
            .unwrap();
        started_rx.await.unwrap();

        let cancelling = tokio::spawn({
            let registry = registry.clone();
            let id = id.clone();
            async move { registry.kill(&id).await.unwrap() }
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(!cancelling.is_finished());
        assert!(!finished.load(std::sync::atomic::Ordering::Acquire));

        let outcome = cancelling.await.unwrap();
        assert_eq!(outcome.state, crate::BackgroundTaskState::Cancelled);
        assert!(finished.load(std::sync::atomic::Ordering::Acquire));
    }
}
