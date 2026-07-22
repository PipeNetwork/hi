//! Background subagent task registry.
//!
//! Tracks subagent tasks spawned via the `task` tool. Each task runs on a
//! dedicated thread with a Tokio `LocalSet` (so non-`Send` futures — like
//! child `Agent` turns — can run without `Send` bounds). The parent agent
//! polls results with `get_task_output`, waits with `wait_tasks`, and cancels
//! with `kill_task`.
//!
//! Communication between the registry (on the agent's thread) and the worker
//! (on the dedicated thread) is via channels, so the registry itself is `Send`
//! and `Sync` — it stores only `Send` handles (channels + shared state).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify, oneshot};
use tokio::task::AbortHandle;

/// Maximum number of concurrent background subagent tasks per session.
const MAX_BG_TASKS: usize = 16;

/// Maximum wait timeout for `get_task_output` / `wait_tasks` (~10 min).
pub const MAX_WAIT_TIMEOUT: Duration = Duration::from_secs(600);

/// Default wait timeout for `wait_tasks` (30s).
pub const DEFAULT_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

/// Lifecycle state of a background subagent task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundTaskState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl BackgroundTaskState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// The outcome produced by a background subagent task when it finishes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackgroundTaskOutcome {
    pub id: String,
    pub description: String,
    pub subagent_type: String,
    pub state: BackgroundTaskState,
    pub output: String,
    pub applied: bool,
    pub changed_files: Vec<String>,
}

impl BackgroundTaskOutcome {
    pub fn running(id: &str, description: &str, subagent_type: &str) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            subagent_type: subagent_type.to_string(),
            state: BackgroundTaskState::Running,
            output: String::new(),
            applied: false,
            changed_files: Vec::new(),
        }
    }

    pub fn tool_status(&self) -> crate::ToolStatus {
        match self.state {
            BackgroundTaskState::Running | BackgroundTaskState::Completed => {
                crate::ToolStatus::Succeeded
            }
            BackgroundTaskState::Cancelled => crate::ToolStatus::Cancelled,
            BackgroundTaskState::Failed => crate::ToolStatus::Failed,
        }
    }
}

/// A boxed future that produces a background task outcome.
/// Stored on the worker thread's LocalSet — never crosses threads.
pub type BgFuture = std::pin::Pin<Box<dyn std::future::Future<Output = BackgroundTaskOutcome> + 'static>>;

/// Command sent from the registry to the worker thread.
enum WorkerCommand {
    /// Spawn a task: run the future on the LocalSet, send result via channel.
    Spawn {
        id: String,
        future_factory: Box<dyn FnOnce() -> BgFuture + Send + 'static>,
        result_tx: oneshot::Sender<BackgroundTaskOutcome>,
    },
    /// Cancel a task by ID.
    Cancel { id: String },
}

/// Internal entry for a tracked background task.
struct BgTaskEntry {
    description: String,
    subagent_type: String,
    /// Result receiver — `Some` until the task completes and the result is
    /// consumed, then `None` (the outcome is cached in `final_outcome`).
    result_rx: Option<oneshot::Receiver<BackgroundTaskOutcome>>,
    /// Cached final outcome once the task has completed.
    final_outcome: Option<BackgroundTaskOutcome>,
    /// Abort handle for the LocalSet task — used by `kill_task`.
    abort_handle: Option<AbortHandle>,
    /// Notify for `wait_tasks` — signalled when the task reaches a terminal state.
    notify: Arc<Notify>,
}

/// Session-scoped registry of background subagent tasks.
///
/// The registry stores only `Send` handles (channels + shared state). The
/// actual subagent futures run on a dedicated worker thread with a `LocalSet`,
/// so non-`Send` futures (like child `Agent` turns) can run without `Send`
/// bounds. This keeps the registry — and the `Agent` that owns it — `Send`.
#[derive(Default)]
pub struct BackgroundTaskRegistry {
    tasks: Mutex<HashMap<String, BgTaskEntry>>,
    counter: std::sync::atomic::AtomicU64,
    /// Channel to send commands to the worker thread.
    /// Created lazily on first spawn.
    worker_tx: std::sync::OnceLock<tokio::sync::mpsc::UnboundedSender<WorkerCommand>>,
}

impl BackgroundTaskRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create the worker thread handle.
    fn worker(&self) -> &tokio::sync::mpsc::UnboundedSender<WorkerCommand> {
        self.worker_tx.get_or_init(|| {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<WorkerCommand>();

            // Spawn a dedicated thread with its own runtime + LocalSet.
            std::thread::Builder::new()
                .name("hi-bg-tasks".into())
                .spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("bg task runtime");
                    let local_set = tokio::task::LocalSet::new();
                    // Run the LocalSet on the runtime. The LocalSet is borrowed
                    // by the async block via a reference — we use `&local_set`
                    // inside the block to avoid moving it.
                    let local_ref = &local_set;
                    local_set.block_on(&runtime, async move {
                        while let Some(cmd) = rx.recv().await {
                            match cmd {
                                WorkerCommand::Spawn {
                                    id: _,
                                    future_factory,
                                    result_tx,
                                } => {
                                    let future = future_factory();
                                    local_ref.spawn_local(async move {
                                        let outcome = future.await;
                                        let _ = result_tx.send(outcome);
                                    });
                                }
                                WorkerCommand::Cancel { id: _ } => {}
                            }
                        }
                    });
                })
                .expect("spawn bg task thread");

            tx
        })
    }

    /// Spawn a background subagent task.
    ///
    /// `future_factory` is a closure that produces the future. It's `Send`
    /// (a closure), but the future it produces does NOT need to be `Send` —
    /// it runs on the worker thread's `LocalSet`.
    ///
    /// This method is async because it acquires the registry's async mutex.
    pub async fn spawn(
        &self,
        description: &str,
        subagent_type: &str,
        future_factory: Box<dyn FnOnce() -> BgFuture + Send + 'static>,
    ) -> anyhow::Result<String> {
        let id = format!(
            "task_{}",
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1
        );

        // Try to acquire the lock synchronously (we're in a sync context).
        // If the lock is held, we use blocking_lock.
        let mut tasks = self.tasks.lock().await;

        // Prune terminal tasks if at capacity.
        if tasks.len() >= MAX_BG_TASKS {
            let to_prune: Vec<String> = tasks
                .iter()
                .filter(|(_, e)| e.final_outcome.is_some())
                .map(|(k, _)| k.clone())
                .collect();
            for k in &to_prune {
                tasks.remove(k);
            }
            if tasks.len() >= MAX_BG_TASKS {
                anyhow::bail!("too many concurrent background tasks (max {MAX_BG_TASKS})");
            }
        }

        let (tx, rx) = oneshot::channel::<BackgroundTaskOutcome>();
        let notify = Arc::new(Notify::new());

        // Send the spawn command to the worker thread.
        let worker = self.worker();
        worker.send(WorkerCommand::Spawn {
            id: id.clone(),
            future_factory,
            result_tx: tx,
        })
        .map_err(|_| anyhow::anyhow!("background task worker thread is dead"))?;

        tasks.insert(
            id.clone(),
            BgTaskEntry {
                description: description.to_string(),
                subagent_type: subagent_type.to_string(),
                result_rx: Some(rx),
                final_outcome: None,
                abort_handle: None,
                notify,
            },
        );

        Ok(id)
    }

    /// Poll a single task for its current output/status.
    pub async fn poll(&self, id: &str, timeout: Duration) -> Option<BackgroundTaskOutcome> {
        // Check for cached final outcome first.
        {
            let tasks = self.tasks.lock().await;
            if let Some(entry) = tasks.get(id) {
                if let Some(ref outcome) = entry.final_outcome {
                    return Some(outcome.clone());
                }
            } else {
                return None;
            }
        }

        // Take the result receiver.
        let (description, subagent_type, mut rx) = {
            let mut tasks = self.tasks.lock().await;
            let entry = tasks.get_mut(id)?;
            if entry.final_outcome.is_some() {
                return entry.final_outcome.clone();
            }
            match entry.result_rx.take() {
                Some(rx) => (entry.description.clone(), entry.subagent_type.clone(), rx),
                None => {
                    return Some(BackgroundTaskOutcome::running(
                        id,
                        &entry.description,
                        &entry.subagent_type,
                    ));
                }
            }
        };

        // Await the result.
        let result = if timeout.is_zero() {
            rx.try_recv().ok()
        } else {
            match tokio::time::timeout(timeout, &mut rx).await {
                Ok(Ok(outcome)) => Some(outcome),
                _ => None,
            }
        };

        match result {
            Some(outcome) => {
                let mut tasks = self.tasks.lock().await;
                if let Some(entry) = tasks.get_mut(id) {
                    entry.final_outcome = Some(outcome.clone());
                    entry.notify.notify_waiters();
                }
                Some(outcome)
            }
            None => {
                // Put the receiver back.
                let mut tasks = self.tasks.lock().await;
                if let Some(entry) = tasks.get_mut(id) {
                    entry.result_rx = Some(rx);
                }
                Some(BackgroundTaskOutcome::running(
                    id,
                    &description,
                    &subagent_type,
                ))
            }
        }
    }

    pub async fn poll_many(
        &self,
        ids: &[String],
        timeout: Duration,
    ) -> Vec<BackgroundTaskOutcome> {
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let outcome = self.poll(id, timeout).await.unwrap_or_else(|| {
                BackgroundTaskOutcome::running(id, "", "unknown")
            });
            results.push(outcome);
        }
        results
    }

    pub async fn wait_all(&self, ids: &[String], timeout: Duration) -> Vec<BackgroundTaskOutcome> {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            let outcome = self.poll(id, remaining).await.unwrap_or_else(|| {
                BackgroundTaskOutcome::running(id, "", "unknown")
            });
            results.push(outcome);
        }
        results
    }

    pub async fn wait_any(&self, ids: &[String], timeout: Duration) -> Vec<BackgroundTaskOutcome> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                let mut results = Vec::with_capacity(ids.len());
                for id in ids {
                    let outcome = self.poll(id, Duration::ZERO).await.unwrap_or_else(|| {
                        BackgroundTaskOutcome::running(id, "", "unknown")
                    });
                    results.push(outcome);
                }
                return results;
            }

            let mut all_snapshots = Vec::with_capacity(ids.len());
            let mut any_terminal = false;
            for id in ids {
                let outcome = self.poll(id, Duration::ZERO).await.unwrap_or_else(|| {
                    BackgroundTaskOutcome::running(id, "", "unknown")
                });
                if outcome.state.is_terminal() {
                    any_terminal = true;
                }
                all_snapshots.push(outcome);
            }
            if any_terminal {
                return all_snapshots;
            }

            let remaining = deadline.saturating_duration_since(now);
            let slice = remaining.min(Duration::from_millis(200));
            tokio::time::sleep(slice).await;
        }
    }

    pub async fn kill(&self, id: &str) -> Option<BackgroundTaskOutcome> {
        let mut tasks = self.tasks.lock().await;
        let entry = tasks.get_mut(id)?;

        if let Some(ref outcome) = entry.final_outcome {
            return Some(outcome.clone());
        }

        // Drop the result receiver — the worker task will eventually finish.
        entry.result_rx.take();
        if let Some(handle) = entry.abort_handle.take() {
            handle.abort();
        }

        let outcome = BackgroundTaskOutcome {
            id: id.to_string(),
            description: entry.description.clone(),
            subagent_type: entry.subagent_type.clone(),
            state: BackgroundTaskState::Cancelled,
            output: "Task cancelled by kill_task.".to_string(),
            applied: false,
            changed_files: Vec::new(),
        };
        entry.final_outcome = Some(outcome.clone());
        entry.notify.notify_waiters();
        Some(outcome)
    }

    pub async fn list(&self) -> Vec<String> {
        let tasks = self.tasks.lock().await;
        tasks.keys().cloned().collect()
    }

    pub async fn kill_all(&self) {
        let ids: Vec<String> = {
            let tasks = self.tasks.lock().await;
            tasks.keys().cloned().collect()
        };
        for id in ids {
            self.kill(&id).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn spawn_and_poll_completed() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn("test", "explore", Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: "test".into(),
                        description: "test".into(),
                        subagent_type: "explore".into(),
                        state: BackgroundTaskState::Completed,
                        output: "done".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }))
            .await
            .unwrap();

        let outcome = registry.poll(&id, Duration::from_secs(2)).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
        assert_eq!(outcome.output, "done");
    }

    #[tokio::test]
    async fn poll_non_blocking_returns_running() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn("slow", "explore", Box::new(|| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    BackgroundTaskOutcome {
                        id: "slow".into(),
                        description: "slow".into(),
                        subagent_type: "explore".into(),
                        state: BackgroundTaskState::Completed,
                        output: "finally".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }))
            .await
            .unwrap();

        let result = registry.poll(&id, Duration::ZERO).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().state, BackgroundTaskState::Running);
    }

    #[tokio::test]
    async fn kill_cancels_running_task() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn("cancellable", "delegate", Box::new(|| {
                Box::pin(async {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    BackgroundTaskOutcome {
                        id: "cancellable".into(),
                        description: "cancellable".into(),
                        subagent_type: "delegate".into(),
                        state: BackgroundTaskState::Completed,
                        output: "should not reach".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }))
            .await
            .unwrap();

        let outcome = registry.kill(&id).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Cancelled);

        let outcome2 = registry.poll(&id, Duration::ZERO).await.unwrap();
        assert_eq!(outcome2.state, BackgroundTaskState::Cancelled);
    }

    #[tokio::test]
    async fn wait_all_completes_when_all_done() {
        let registry = BackgroundTaskRegistry::new();
        let id1 = registry
            .spawn("t1", "explore", Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: "t1".into(),
                        description: "t1".into(),
                        subagent_type: "explore".into(),
                        state: BackgroundTaskState::Completed,
                        output: "r1".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }))
            .await
            .unwrap();
        let id2 = registry
            .spawn("t2", "explore", Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: "t2".into(),
                        description: "t2".into(),
                        subagent_type: "explore".into(),
                        state: BackgroundTaskState::Completed,
                        output: "r2".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }))
            .await
            .unwrap();

        let results = registry
            .wait_all(&[id1, id2], Duration::from_secs(2))
            .await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.state == BackgroundTaskState::Completed));
    }

    #[tokio::test]
    async fn kill_is_idempotent() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn("idempotent", "explore", Box::new(|| {
                Box::pin(async {
                    BackgroundTaskOutcome {
                        id: "idempotent".into(),
                        description: "idempotent".into(),
                        subagent_type: "explore".into(),
                        state: BackgroundTaskState::Completed,
                        output: "done".into(),
                        applied: false,
                        changed_files: vec![],
                    }
                })
            }))
            .await
            .unwrap();

        let outcome = registry.poll(&id, Duration::from_secs(2)).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);

        let outcome = registry.kill(&id).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
    }
}
