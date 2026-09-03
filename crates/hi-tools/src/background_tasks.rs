//! Background subagent task registry.
//!
//! Tracks subagent tasks spawned via the `task` tool. Each task runs on one
//! of N dedicated worker threads, each with its own Tokio `LocalSet` (so
//! non-`Send` futures — like child `Agent` turns — can run without `Send`
//! bounds). Using N threads instead of one gives true OS-thread parallelism
//! among background subagents: up to `BG_WORKER_THREADS` tasks run
//! concurrently on separate threads. The parent agent polls results with
//! `get_task_output`, waits with `wait_tasks`, and cancels with `kill_task`.
//!
//! Communication between the registry (on the agent's thread) and the workers
//! is via per-worker channels, so the registry itself is `Send` and `Sync` —
//! it stores only `Send` handles (channels + shared state). A shared `Notify`
//! replaces the old 200ms busy-poll loop in `wait_all`/`wait_any` with
//! event-driven waking.

use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use futures_util::{FutureExt, future::join_all};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, Notify, oneshot, watch};
use tokio::task::AbortHandle;

/// Maximum number of concurrent background subagent tasks per session.
const MAX_BG_TASKS: usize = 16;

/// Admission failure when every registry slot is still live or owns a terminal
/// result that the caller has not observed yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackgroundTaskCapacityError {
    pub maximum: usize,
    pub running: usize,
    pub unobserved_terminal: usize,
}

impl std::fmt::Display for BackgroundTaskCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "background task capacity reached (max {}): {} running and {} completed/failed result(s) not yet observed; call get_task_output or wait_tasks for existing task IDs, then retry (or use kill_task for work no longer needed)",
            self.maximum, self.running, self.unobserved_terminal
        )
    }
}

impl std::error::Error for BackgroundTaskCapacityError {}

/// A worker normally acknowledges a queued command in the same scheduler
/// tick. Bound this wait so a non-yielding LocalSet task cannot make registry
/// operations wait forever behind the worker's command loop.
const WORKER_HANDLE_ACK_TIMEOUT: Duration = Duration::from_secs(2);

fn worker_threads() -> usize {
    std::env::var("HI_BACKGROUND_SUBAGENT_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4)
        .clamp(1, MAX_BG_TASKS - 1)
}

/// Number of worker threads in the background task pool. One slot from the
/// session budget is intentionally left for foreground work so a saturated
/// background fleet cannot starve interactive execution.
/// Legacy maximum used by saturation tests; runtime defaults are configured by
/// [`worker_threads`] and may be lower.
#[cfg(test)]
const BG_WORKER_THREADS: usize = MAX_BG_TASKS - 1;

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

    /// Fill registry identity onto a worker-produced outcome.
    ///
    /// Background workers often leave `id` / `description` empty (they don't know
    /// the registry handle). Call this before caching a terminal result so polls
    /// and status lines keep the human label.
    pub fn with_registry_identity(
        mut self,
        id: &str,
        description: &str,
        subagent_type: &str,
    ) -> Self {
        if self.id.is_empty() {
            self.id = id.to_string();
        }
        if self.description.is_empty() {
            self.description = description.to_string();
        }
        if self.subagent_type.is_empty() {
            self.subagent_type = subagent_type.to_string();
        }
        self
    }

    pub fn tool_status(&self) -> crate::ToolStatus {
        match self.state {
            // Still running is not success — callers that treat Succeeded as
            // "done" must check `state` (or `is_terminal`) separately.
            BackgroundTaskState::Completed => crate::ToolStatus::Succeeded,
            BackgroundTaskState::Running => crate::ToolStatus::Succeeded,
            BackgroundTaskState::Cancelled => crate::ToolStatus::Cancelled,
            BackgroundTaskState::Failed => crate::ToolStatus::Failed,
        }
    }
}

/// Outcome returned when a polled/waited task ID is not in the registry.
///
/// Uses [`BackgroundTaskState::Failed`] (not `Running`) so the caller learns the
/// task doesn't exist rather than waiting forever, and carries an explicit
/// message instead of an empty description + `"unknown"` type that rendered as
/// the confusing `(/unknown)` fragment.
fn not_found_outcome(id: &str) -> BackgroundTaskOutcome {
    BackgroundTaskOutcome {
        id: id.to_string(),
        description: String::new(),
        subagent_type: String::new(),
        state: BackgroundTaskState::Failed,
        output: format!("no task with id \"{id}\""),
        applied: false,
        changed_files: Vec::new(),
    }
}

/// A boxed future that produces a background task outcome.
/// Stored on a worker thread's `LocalSet` — never crosses threads, so it
/// does not need to be `Send`.
pub type BgFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = BackgroundTaskOutcome> + 'static>>;
type BgFutureFactory = Box<dyn FnOnce() -> BgFuture + Send + 'static>;
type SharedBgFutureFactory = Arc<std::sync::Mutex<Option<BgFutureFactory>>>;

/// Command sent from the registry to a worker thread.
enum WorkerCommand {
    /// Spawn a task: run the future on the worker's LocalSet, send result via
    /// channel, and send the `AbortHandle` back so the registry can cancel it.
    /// The `completed_notify` is signalled when the task finishes so
    /// `wait_all`/`wait_any` wake immediately instead of busy-polling.
    Spawn {
        future_factory: SharedBgFutureFactory,
        result_tx: oneshot::Sender<BackgroundTaskOutcome>,
        handle_tx: oneshot::Sender<AbortHandle>,
        activation_rx: oneshot::Receiver<()>,
        dispatch_cancelled: Arc<AtomicBool>,
        task_id: String,
        task_notify: Arc<Notify>,
        terminal_outcome: Arc<std::sync::Mutex<Option<BackgroundTaskOutcome>>>,
        outcomes: Arc<std::sync::Mutex<HashMap<String, BackgroundTaskOutcome>>>,
        completed_notify: Arc<Notify>,
        shutdown: watch::Receiver<bool>,
        abort_handles: Arc<std::sync::Mutex<HashMap<String, AbortHandle>>>,
    },
}

/// Owns the provisional dispatch until the worker is acknowledged and
/// activated. Dropping the spawn future also cancels the queued command and
/// publishes a terminal state, so it cannot leave an unaddressable running
/// entry behind.
struct PendingWorkerDispatch {
    id: String,
    description: String,
    subagent_type: String,
    activation_tx: Option<oneshot::Sender<()>>,
    dispatch_cancelled: Arc<AtomicBool>,
    future_factory: SharedBgFutureFactory,
    terminal_outcome: Arc<std::sync::Mutex<Option<BackgroundTaskOutcome>>>,
    outcomes: Arc<std::sync::Mutex<HashMap<String, BackgroundTaskOutcome>>>,
    notify: Arc<Notify>,
    completed_notify: Arc<Notify>,
    abort_handles: Arc<std::sync::Mutex<HashMap<String, AbortHandle>>>,
    finished: bool,
}

impl PendingWorkerDispatch {
    fn cancel(&mut self) {
        if self.finished {
            return;
        }
        self.dispatch_cancelled.store(true, Ordering::Release);
        self.activation_tx.take();
        self.future_factory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(handle) = self
            .abort_handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.id)
        {
            handle.abort();
        }

        let fallback = BackgroundTaskOutcome {
            id: self.id.clone(),
            description: self.description.clone(),
            subagent_type: self.subagent_type.clone(),
            state: BackgroundTaskState::Cancelled,
            output: "Task cancelled before worker dispatch completed.".into(),
            applied: false,
            changed_files: Vec::new(),
        };
        {
            let mut terminal = self
                .terminal_outcome
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut cached = self
                .outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let outcome = terminal
                .as_ref()
                .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                .or_else(|| {
                    cached
                        .get(&self.id)
                        .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                })
                .cloned()
                .unwrap_or(fallback);
            *terminal = Some(outcome.clone());
            cached.insert(self.id.clone(), outcome);
        }
        self.notify.notify_waiters();
        self.completed_notify.notify_waiters();
        self.finished = true;
    }

    fn activate(&mut self) -> bool {
        let activated = self
            .activation_tx
            .take()
            .is_some_and(|activation_tx| activation_tx.send(()).is_ok());
        if activated {
            self.finished = true;
        } else {
            self.cancel();
        }
        activated
    }
}

impl Drop for PendingWorkerDispatch {
    fn drop(&mut self) {
        self.cancel();
    }
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
    /// True only after a terminal result was explicitly returned by poll/wait
    /// (or by an explicit kill). Capacity pressure may reclaim terminal entries
    /// only after this acknowledgement; worker publication alone is not delivery.
    observed: bool,
    /// Stable terminal state shared with workers and dependency gates. Unlike
    /// the registry cache, this cell survives capacity pruning while an
    /// already-queued dependent still holds a reference to it.
    terminal_outcome: Arc<std::sync::Mutex<Option<BackgroundTaskOutcome>>>,
    /// Abort handle for the LocalSet task — used by `kill_task`.
    abort_handle: Option<AbortHandle>,
    /// Notify for `wait_tasks` — signalled when the task reaches a terminal state.
    notify: Arc<Notify>,
}

#[derive(Clone)]
struct DependencyGate {
    id: String,
    terminal_outcome: Arc<std::sync::Mutex<Option<BackgroundTaskOutcome>>>,
    notify: Arc<Notify>,
}

async fn wait_for_dependencies(
    dependency_gates: Vec<DependencyGate>,
) -> Result<(), (String, BackgroundTaskState)> {
    for gate in dependency_gates {
        let outcome = loop {
            // Arm the notification before checking the stable cell so a
            // dependency completing between those operations cannot strand
            // this gate. The cell itself outlives registry capacity pruning.
            let notified = gate.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(outcome) = gate
                .terminal_outcome
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
            {
                break outcome;
            }
            notified.await;
        };
        if outcome.state != BackgroundTaskState::Completed {
            return Err((gate.id, outcome.state));
        }
    }
    Ok(())
}

/// Session-scoped registry of background subagent tasks.
///
/// The registry stores only `Send` handles (channels + shared state). The
/// actual subagent futures run on a pool of dedicated worker threads, each
/// with its own `LocalSet`, so non-`Send` futures (like child `Agent` turns)
/// can run without `Send` bounds while still getting true parallelism across
/// threads. This keeps the registry — and the `Agent` that owns it — `Send`.
pub struct BackgroundTaskRegistry {
    tasks: Mutex<HashMap<String, BgTaskEntry>>,
    counter: std::sync::atomic::AtomicU64,
    /// Round-robin counter for distributing tasks across workers.
    next_worker: std::sync::atomic::AtomicU64,
    /// Shared notify — signalled whenever any task reaches a terminal state.
    /// Replaces the old 200ms busy-poll loop in `wait_all`/`wait_any`.
    /// `Arc` so the worker threads can signal it when a task completes.
    completed_notify: Arc<Notify>,
    outcomes: Arc<std::sync::Mutex<HashMap<String, BackgroundTaskOutcome>>>,
    /// Synchronous registry-lifetime signal. Worker commands receive a
    /// subscriber so even a command queued before Drop cannot start its future
    /// after the owning session disappears.
    shutdown: watch::Sender<bool>,
    /// Drop cannot await `tasks`, so retain a synchronous abort index as the
    /// hard backstop for every LocalSet task owned by this registry.
    abort_handles: Arc<std::sync::Mutex<HashMap<String, AbortHandle>>>,
}

async fn wait_for_registry_shutdown(shutdown: &mut watch::Receiver<bool>) {
    loop {
        if *shutdown.borrow_and_update() {
            return;
        }
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

fn registry_dropped_outcome(id: &str) -> BackgroundTaskOutcome {
    BackgroundTaskOutcome {
        id: id.to_string(),
        description: String::new(),
        subagent_type: String::new(),
        state: BackgroundTaskState::Cancelled,
        output: "Task cancelled because its registry was dropped.".into(),
        applied: false,
        changed_files: Vec::new(),
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

fn worker_panicked_outcome(
    id: &str,
    phase: &str,
    payload: Box<dyn std::any::Any + Send>,
) -> BackgroundTaskOutcome {
    BackgroundTaskOutcome {
        id: id.to_string(),
        description: String::new(),
        subagent_type: String::new(),
        state: BackgroundTaskState::Failed,
        output: format!(
            "Background task panicked while {phase}: {}",
            panic_payload_message(payload.as_ref())
        ),
        applied: false,
        changed_files: Vec::new(),
    }
}

fn dispatch_worker_command(local_set: &tokio::task::LocalSet, cmd: WorkerCommand) {
    let WorkerCommand::Spawn {
        future_factory,
        result_tx,
        handle_tx,
        activation_rx,
        dispatch_cancelled,
        task_id,
        task_notify,
        terminal_outcome,
        outcomes,
        completed_notify,
        mut shutdown,
        abort_handles,
    } = cmd;

    // A command can remain queued after its bounded acknowledgment wait has
    // expired. Discard it before constructing the future when that happens.
    if dispatch_cancelled.load(Ordering::Acquire) {
        return;
    }

    let worker_abort_handles = abort_handles.clone();
    let worker_task_id = task_id.clone();
    let handle = local_set.spawn_local(async move {
        // The registry activates the task only after it has installed the
        // worker handle into the provisional entry. A timed-out or dropped
        // dispatcher closes this channel, so late commands cannot execute.
        if activation_rx.await.is_err() || dispatch_cancelled.load(Ordering::Acquire) {
            worker_abort_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&worker_task_id);
            return;
        }

        let Some(future_factory) = future_factory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        else {
            worker_abort_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&worker_task_id);
            return;
        };
        let outcome = if *shutdown.borrow() {
            registry_dropped_outcome(&worker_task_id)
        } else {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(future_factory)) {
                Err(payload) => {
                    worker_panicked_outcome(&worker_task_id, "constructing its future", payload)
                }
                Ok(future) => {
                    let guarded = std::panic::AssertUnwindSafe(future).catch_unwind();
                    tokio::pin!(guarded);
                    tokio::select! {
                        biased;
                        _ = wait_for_registry_shutdown(&mut shutdown) => {
                            registry_dropped_outcome(&worker_task_id)
                        }
                        result = &mut guarded => match result {
                            Ok(outcome) => outcome,
                            Err(payload) => worker_panicked_outcome(
                                &worker_task_id,
                                "running its future",
                                payload,
                            ),
                        },
                    }
                }
            }
        };
        worker_abort_handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&worker_task_id);
        let outcome = {
            // Publish atomically with respect to kill/poll. A cancellation
            // already recorded in either store is authoritative over a
            // concurrently produced worker result.
            let mut terminal = terminal_outcome
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut cached = outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let outcome = terminal
                .as_ref()
                .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                .or_else(|| {
                    cached
                        .get(&worker_task_id)
                        .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                })
                .cloned()
                .unwrap_or(outcome);
            *terminal = Some(outcome.clone());
            cached.insert(worker_task_id.clone(), outcome.clone());
            outcome
        };
        let _ = result_tx.send(outcome);
        task_notify.notify_waiters();
        completed_notify.notify_waiters();
    });
    let abort_handle = handle.abort_handle();
    abort_handles
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(task_id.clone(), abort_handle.clone());
    if let Err(abort_handle) = handle_tx.send(abort_handle) {
        // The spawn caller disappeared before it could activate the task. Do
        // not leave an unaddressable worker future waiting on its handshake.
        abort_handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&task_id);
        abort_handle.abort();
    }
}

static GLOBAL_WORKERS: std::sync::OnceLock<Vec<tokio::sync::mpsc::UnboundedSender<WorkerCommand>>> =
    std::sync::OnceLock::new();

fn global_workers() -> &'static [tokio::sync::mpsc::UnboundedSender<WorkerCommand>] {
    GLOBAL_WORKERS.get_or_init(|| {
        let worker_count = worker_threads();
        let mut senders = Vec::with_capacity(worker_count);
        for idx in 0..worker_count {
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<WorkerCommand>();

            // Each worker is a dedicated OS thread with its own
            // current-thread runtime + LocalSet. Non-`Send` futures (like
            // child `Agent` turns) run on the LocalSet. Using N threads
            // gives true OS-thread parallelism among background tasks.
            std::thread::Builder::new()
                .name(format!("hi-bg-tasks-{idx}"))
                .spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .expect("bg task runtime");
                    let local_set = tokio::task::LocalSet::new();
                    let local_ref = &local_set;
                    local_set.block_on(&runtime, async move {
                        while let Some(cmd) = rx.recv().await {
                            dispatch_worker_command(local_ref, cmd);
                        }
                    });
                })
                .expect("spawn bg task worker thread");

            senders.push(tx);
        }
        senders
    })
}

impl BackgroundTaskRegistry {
    pub fn new() -> Self {
        let (shutdown, _) = watch::channel(false);
        Self {
            tasks: Mutex::new(HashMap::new()),
            counter: std::sync::atomic::AtomicU64::new(0),
            next_worker: std::sync::atomic::AtomicU64::new(0),
            completed_notify: Arc::new(Notify::new()),
            outcomes: Arc::new(std::sync::Mutex::new(HashMap::new())),
            shutdown,
            abort_handles: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    /// Permanently stop this registry and synchronously request cancellation
    /// for every task already installed on a worker.
    ///
    /// This is synchronous so owners such as `Agent` can call it from `Drop`.
    /// The worker-side shutdown subscriber also covers commands that were
    /// queued but had not installed their abort handle yet.
    pub fn shutdown(&self) {
        // `send_replace` retains the terminal value even when there are
        // currently no receivers. A surviving `Arc` must not be able to start
        // new work after its owning Agent has gone away.
        self.shutdown.send_replace(true);
        let handles = self
            .abort_handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .drain()
            .collect::<Vec<_>>();
        if !handles.is_empty() {
            let mut outcomes = self
                .outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (id, _) in &handles {
                outcomes.insert(id.clone(), registry_dropped_outcome(id));
            }
            drop(outcomes);
            self.completed_notify.notify_waiters();
        }
        let handles = handles
            .into_iter()
            .map(|(_, handle)| handle)
            .collect::<Vec<_>>();
        for handle in handles {
            handle.abort();
        }
    }
}

impl Drop for BackgroundTaskRegistry {
    fn drop(&mut self) {
        // Drop is synchronous and may run outside a Tokio runtime. Notify
        // queued commands, then abort every task already installed on a worker.
        // Aborting a Tokio task drops its future on that worker, which in turn
        // drops a child Agent and its process/runtime guards.
        self.shutdown();
    }
}

impl Default for BackgroundTaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BackgroundTaskRegistry {
    fn workers(&self) -> &[tokio::sync::mpsc::UnboundedSender<WorkerCommand>] {
        global_workers()
    }

    /// Pick the next worker channel (round-robin).
    fn next_worker_tx(&self) -> &tokio::sync::mpsc::UnboundedSender<WorkerCommand> {
        let workers = self.workers();
        let idx = self
            .next_worker
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % workers.len() as u64;
        &workers[idx as usize]
    }

    /// Commit a worker-produced terminal result without allowing a concurrent
    /// cancellation to be overwritten. The task-map lock is the linearization
    /// point shared with [`Self::kill`].
    async fn commit_worker_terminal(
        &self,
        id: &str,
        description: &str,
        subagent_type: &str,
        proposed: BackgroundTaskOutcome,
        observed: bool,
    ) -> BackgroundTaskOutcome {
        let mut tasks = self.tasks.lock().await;
        if let Some(entry) = tasks.get_mut(id) {
            if let Some(outcome) = entry.final_outcome.clone() {
                entry.observed |= observed;
                return outcome;
            }

            let outcome = {
                let mut terminal = entry
                    .terminal_outcome
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut cached = self
                    .outcomes
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let outcome = terminal
                    .as_ref()
                    .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                    .or_else(|| {
                        cached
                            .get(id)
                            .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                    })
                    .cloned()
                    .unwrap_or(proposed)
                    .with_registry_identity(id, description, subagent_type);
                *terminal = Some(outcome.clone());
                cached.insert(id.to_string(), outcome.clone());
                outcome
            };
            entry.abort_handle.take();
            entry.final_outcome = Some(outcome.clone());
            entry.observed |= observed;
            entry.notify.notify_waiters();
            drop(tasks);
            self.abort_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(id);
            self.completed_notify.notify_waiters();
            return outcome;
        }
        drop(tasks);

        // Capacity pruning can remove an entry after its worker published but
        // before a concurrent poll consumes the oneshot. Preserve any shared
        // cancellation even in that narrow case.
        let mut cached = self
            .outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let outcome = cached
            .get(id)
            .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
            .cloned()
            .unwrap_or(proposed)
            .with_registry_identity(id, description, subagent_type);
        cached.insert(id.to_string(), outcome.clone());
        drop(cached);
        self.completed_notify.notify_waiters();
        outcome
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
        self.spawn_after(description, subagent_type, &[], future_factory)
            .await
    }

    /// Spawn a task after all named dependencies reach terminal success.
    pub async fn spawn_after(
        &self,
        description: &str,
        subagent_type: &str,
        dependencies: &[String],
        future_factory: Box<dyn FnOnce() -> BgFuture + Send + 'static>,
    ) -> anyhow::Result<String> {
        let worker = self.next_worker_tx().clone();
        self.spawn_after_on_worker(
            description,
            subagent_type,
            dependencies,
            future_factory,
            worker,
            WORKER_HANDLE_ACK_TIMEOUT,
        )
        .await
    }

    async fn rollback_provisional_dispatch(&self, id: &str) {
        let removed = self.tasks.lock().await.remove(id);
        self.outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id);
        if let Some(handle) = self
            .abort_handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id)
        {
            handle.abort();
        }
        if let Some(entry) = removed {
            entry.notify.notify_waiters();
            self.completed_notify.notify_waiters();
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn spawn_after_on_worker(
        &self,
        description: &str,
        subagent_type: &str,
        dependencies: &[String],
        future_factory: Box<dyn FnOnce() -> BgFuture + Send + 'static>,
        worker: tokio::sync::mpsc::UnboundedSender<WorkerCommand>,
        worker_ack_timeout: Duration,
    ) -> anyhow::Result<String> {
        if *self.shutdown.borrow() {
            anyhow::bail!("background task registry is shut down");
        }

        let id = format!(
            "task_{}",
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1
        );

        let mut tasks = self.tasks.lock().await;

        for dependency in dependencies {
            if dependency == &id {
                anyhow::bail!("task cannot depend on itself");
            }
            if !tasks.contains_key(dependency) {
                anyhow::bail!("unknown dependency task ID: {dependency}");
            }
        }

        // Capture stable dependency cells before capacity reclamation. An
        // acknowledged terminal dependency may then leave the registry while
        // this newly queued task safely retains the exact terminal outcome.
        let dependency_gates = dependencies
            .iter()
            .map(|dependency| {
                let entry = tasks.get(dependency).expect("dependency validated");
                DependencyGate {
                    id: dependency.clone(),
                    terminal_outcome: entry.terminal_outcome.clone(),
                    notify: entry.notify.clone(),
                }
            })
            .collect::<Vec<_>>();

        // Reclaim only terminal results that were explicitly returned to a
        // caller. Worker publication is not acknowledgement: removing an unread
        // result here would make its task ID resolve to `not found` forever.
        if tasks.len() >= MAX_BG_TASKS {
            let cached_terminal = self
                .outcomes
                .lock()
                .expect("outcome cache poisoned")
                .iter()
                .filter(|(_, outcome)| outcome.state.is_terminal())
                .map(|(key, _)| key.clone())
                .collect::<std::collections::HashSet<_>>();
            let to_prune: Vec<String> = tasks
                .iter()
                .filter(|(key, entry)| {
                    entry.observed
                        && (entry.final_outcome.is_some() || cached_terminal.contains(*key))
                })
                .map(|(k, _)| k.clone())
                .collect();
            for k in &to_prune {
                tasks.remove(k);
            }
            if !to_prune.is_empty() {
                let mut outcomes = self.outcomes.lock().expect("outcome cache poisoned");
                for key in &to_prune {
                    outcomes.remove(key);
                }
            }
            if tasks.len() >= MAX_BG_TASKS {
                let terminal = tasks
                    .iter()
                    .filter(|(key, entry)| {
                        entry.final_outcome.is_some() || cached_terminal.contains(*key)
                    })
                    .count();
                return Err(BackgroundTaskCapacityError {
                    maximum: MAX_BG_TASKS,
                    running: tasks.len().saturating_sub(terminal),
                    unobserved_terminal: terminal,
                }
                .into());
            }
        }

        let (tx, rx) = oneshot::channel::<BackgroundTaskOutcome>();
        let (handle_tx, handle_rx) = oneshot::channel::<AbortHandle>();
        let (activation_tx, activation_rx) = oneshot::channel::<()>();
        let dispatch_cancelled = Arc::new(AtomicBool::new(false));
        let notify = Arc::new(Notify::new());
        let terminal_outcome = Arc::new(std::sync::Mutex::new(None));

        let gated_factory: Box<dyn FnOnce() -> BgFuture + Send + 'static> = if dependency_gates
            .is_empty()
        {
            future_factory
        } else {
            Box::new(move || {
                Box::pin(async move {
                    if let Err((dependency, state)) = wait_for_dependencies(dependency_gates).await
                    {
                        // Identity is stamped by the registry on poll;
                        // leave id/description empty here deliberately.
                        return BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Failed,
                            output: format!("Dependency {dependency} did not succeed ({state:?})."),
                            applied: false,
                            changed_files: Vec::new(),
                        };
                    }
                    future_factory().await
                })
            })
        };
        let gated_factory = Arc::new(std::sync::Mutex::new(Some(gated_factory)));

        // Register provisionally before dispatch, then release the registry
        // mutex. Poll/kill/list remain responsive even if this worker's command
        // loop is stalled by a non-yielding LocalSet task.
        tasks.insert(
            id.clone(),
            BgTaskEntry {
                description: description.to_string(),
                subagent_type: subagent_type.to_string(),
                result_rx: Some(rx),
                final_outcome: None,
                observed: false,
                terminal_outcome: terminal_outcome.clone(),
                abort_handle: None,
                notify: notify.clone(),
            },
        );
        drop(tasks);

        let mut pending_dispatch = PendingWorkerDispatch {
            id: id.clone(),
            description: description.to_string(),
            subagent_type: subagent_type.to_string(),
            activation_tx: Some(activation_tx),
            dispatch_cancelled: dispatch_cancelled.clone(),
            future_factory: gated_factory.clone(),
            terminal_outcome: terminal_outcome.clone(),
            outcomes: self.outcomes.clone(),
            notify: notify.clone(),
            completed_notify: self.completed_notify.clone(),
            abort_handles: self.abort_handles.clone(),
            finished: false,
        };

        if *self.shutdown.borrow() {
            pending_dispatch.cancel();
            self.rollback_provisional_dispatch(&id).await;
            anyhow::bail!("background task registry is shut down");
        }

        if worker
            .send(WorkerCommand::Spawn {
                future_factory: gated_factory,
                result_tx: tx,
                handle_tx,
                activation_rx,
                dispatch_cancelled,
                task_id: id.clone(),
                task_notify: notify,
                terminal_outcome,
                outcomes: self.outcomes.clone(),
                completed_notify: self.completed_notify.clone(),
                shutdown: self.shutdown.subscribe(),
                abort_handles: self.abort_handles.clone(),
            })
            .is_err()
        {
            pending_dispatch.cancel();
            self.rollback_provisional_dispatch(&id).await;
            anyhow::bail!("background task worker thread is dead");
        }

        let abort_handle = match tokio::time::timeout(worker_ack_timeout, handle_rx).await {
            Ok(Ok(handle)) => handle,
            Ok(Err(_)) => {
                pending_dispatch.cancel();
                self.rollback_provisional_dispatch(&id).await;
                anyhow::bail!("background task worker stopped before acknowledging task {id}");
            }
            Err(_) => {
                pending_dispatch.cancel();
                self.rollback_provisional_dispatch(&id).await;
                anyhow::bail!(
                    "background task worker did not acknowledge task {id} within {:.1}s",
                    worker_ack_timeout.as_secs_f64()
                );
            }
        };

        let can_activate = {
            let mut tasks = self.tasks.lock().await;
            tasks.get_mut(&id).is_some_and(|entry| {
                if entry.final_outcome.is_some() || *self.shutdown.borrow() {
                    false
                } else {
                    entry.abort_handle = Some(abort_handle.clone());
                    true
                }
            })
        };
        if !can_activate || !pending_dispatch.activate() {
            abort_handle.abort();
            pending_dispatch.cancel();
            self.rollback_provisional_dispatch(&id).await;
            anyhow::bail!("background task {id} was cancelled before worker activation");
        }

        Ok(id)
    }

    /// Poll a single task for its current output/status.
    pub async fn poll(&self, id: &str, timeout: Duration) -> Option<BackgroundTaskOutcome> {
        self.poll_inner(id, timeout, true).await
    }

    /// Poll without necessarily acknowledging a terminal result. Aggregate
    /// waits use `observed = false` for intermediate snapshots, then acknowledge
    /// only the stable result set they actually return to their caller.
    async fn poll_inner(
        &self,
        id: &str,
        timeout: Duration,
        observed: bool,
    ) -> Option<BackgroundTaskOutcome> {
        // Workers publish into the shared cache before notifying the registry.
        // Shutdown also publishes cancellation there before aborting wrappers,
        // because an aborted wrapper cannot send through its result oneshot.
        let shared_terminal = self
            .outcomes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(id)
            .filter(|outcome| outcome.state.is_terminal())
            .cloned();
        {
            let mut tasks = self.tasks.lock().await;
            let entry = tasks.get_mut(id)?;
            if let Some(outcome) = entry.final_outcome.clone() {
                entry.observed |= observed;
                return Some(outcome);
            }
            let stable_terminal = entry
                .terminal_outcome
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
                .filter(|outcome| outcome.state.is_terminal())
                .cloned();
            let known_terminal = stable_terminal
                .as_ref()
                .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                .cloned()
                .or_else(|| {
                    shared_terminal
                        .as_ref()
                        .filter(|outcome| outcome.state == BackgroundTaskState::Cancelled)
                        .cloned()
                })
                .or(stable_terminal)
                .or(shared_terminal);
            if let Some(outcome) = known_terminal {
                let outcome =
                    outcome.with_registry_identity(id, &entry.description, &entry.subagent_type);
                *entry
                    .terminal_outcome
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(outcome.clone());
                entry.result_rx.take();
                entry.abort_handle.take();
                entry.final_outcome = Some(outcome.clone());
                entry.observed |= observed;
                entry.notify.notify_waiters();
                return Some(outcome);
            }
        }

        // Take the result receiver.
        let (description, subagent_type, mut rx) = {
            let mut tasks = self.tasks.lock().await;
            let entry = tasks.get_mut(id)?;
            if let Some(outcome) = entry.final_outcome.clone() {
                entry.observed |= observed;
                return Some(outcome);
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

        enum WorkerResult {
            Terminal(BackgroundTaskOutcome),
            Pending,
            Closed,
        }

        // Await the result. A closed channel is not pending: the worker task
        // can never produce another value, so convert it to a stable terminal
        // result instead of reinserting the closed receiver forever.
        let result = if timeout.is_zero() {
            match rx.try_recv() {
                Ok(outcome) => WorkerResult::Terminal(outcome),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => WorkerResult::Pending,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => WorkerResult::Closed,
            }
        } else {
            match tokio::time::timeout(timeout, &mut rx).await {
                Ok(Ok(outcome)) => WorkerResult::Terminal(outcome),
                Ok(Err(_)) => WorkerResult::Closed,
                Err(_) => WorkerResult::Pending,
            }
        };

        match result {
            WorkerResult::Terminal(outcome) => {
                // Workers typically omit registry identity; stamp it before cache.
                let outcome = outcome.with_registry_identity(id, &description, &subagent_type);
                Some(
                    self.commit_worker_terminal(
                        id,
                        &description,
                        &subagent_type,
                        outcome,
                        observed,
                    )
                    .await,
                )
            }
            WorkerResult::Closed => {
                let outcome = self
                    .outcomes
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .get(id)
                    .filter(|outcome| outcome.state.is_terminal())
                    .cloned()
                    .unwrap_or_else(|| {
                        if *self.shutdown.borrow() {
                            registry_dropped_outcome(id)
                        } else {
                            BackgroundTaskOutcome {
                                id: id.to_string(),
                                description: String::new(),
                                subagent_type: String::new(),
                                state: BackgroundTaskState::Failed,
                                output: "Background task worker stopped without a result.".into(),
                                applied: false,
                                changed_files: Vec::new(),
                            }
                        }
                    })
                    .with_registry_identity(id, &description, &subagent_type);
                Some(
                    self.commit_worker_terminal(
                        id,
                        &description,
                        &subagent_type,
                        outcome,
                        observed,
                    )
                    .await,
                )
            }
            WorkerResult::Pending => {
                // Put the receiver back.
                let mut tasks = self.tasks.lock().await;
                if let Some(entry) = tasks.get_mut(id) {
                    if let Some(outcome) = &entry.final_outcome {
                        entry.observed |= observed;
                        return Some(outcome.clone());
                    }
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

    pub async fn poll_many(&self, ids: &[String], timeout: Duration) -> Vec<BackgroundTaskOutcome> {
        let results = self.poll_many_inner(ids, timeout, false).await;
        self.acknowledge_terminal_results(ids, &results).await;
        results
    }

    async fn poll_many_inner(
        &self,
        ids: &[String],
        timeout: Duration,
        observed: bool,
    ) -> Vec<BackgroundTaskOutcome> {
        join_all(ids.iter().map(|id| async move {
            self.poll_inner(id, timeout, observed)
                .await
                .unwrap_or_else(|| not_found_outcome(id))
        }))
        .await
    }

    async fn acknowledge_terminal_results(
        &self,
        ids: &[String],
        results: &[BackgroundTaskOutcome],
    ) {
        let mut tasks = self.tasks.lock().await;
        for (id, outcome) in ids.iter().zip(results) {
            if outcome.state.is_terminal()
                && let Some(entry) = tasks.get_mut(id)
            {
                entry.observed = true;
            }
        }
    }

    pub async fn wait_all(&self, ids: &[String], timeout: Duration) -> Vec<BackgroundTaskOutcome> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let notified = self.completed_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let results = self.poll_many_inner(ids, Duration::ZERO, false).await;
            if results.iter().all(|outcome| outcome.state.is_terminal()) {
                self.acknowledge_terminal_results(ids, &results).await;
                return results;
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                self.acknowledge_terminal_results(ids, &results).await;
                return results;
            }
            // Register before taking the snapshot so a completion between the
            // snapshot and the await cannot leave us sleeping until timeout.
            let _ = tokio::time::timeout(remaining, notified).await;
        }
    }

    pub async fn wait_any(&self, ids: &[String], timeout: Duration) -> Vec<BackgroundTaskOutcome> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            // Arm the notification before taking the snapshot. A completion
            // between the snapshot and the await must not be able to strand
            // this wait until the full timeout.
            let notified = self.completed_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // Snapshot all tasks non-blockingly.
            let all_snapshots = self.poll_many_inner(ids, Duration::ZERO, false).await;
            if all_snapshots
                .iter()
                .any(|outcome| outcome.state.is_terminal())
            {
                self.acknowledge_terminal_results(ids, &all_snapshots).await;
                return all_snapshots;
            }

            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                self.acknowledge_terminal_results(ids, &all_snapshots).await;
                return all_snapshots;
            }
            // Event-driven wake: notified when any task completes, or timeout.
            // Replaces the old 200ms busy-poll loop.
            let _ = tokio::time::timeout(remaining, notified).await;
        }
    }

    pub async fn kill(&self, id: &str) -> Option<BackgroundTaskOutcome> {
        let mut tasks = self.tasks.lock().await;
        let entry = tasks.get_mut(id)?;

        if let Some(outcome) = entry.final_outcome.clone() {
            entry.observed = true;
            return Some(outcome);
        }

        // Drop the result receiver — the worker task will eventually finish.
        entry.result_rx.take();
        let indexed_handle = self
            .abort_handles
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(id);
        if let Some(handle) = entry.abort_handle.take().or(indexed_handle) {
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
        *entry
            .terminal_outcome
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(outcome.clone());
        entry.final_outcome = Some(outcome.clone());
        entry.observed = true;
        self.outcomes
            .lock()
            .expect("outcome cache poisoned")
            .insert(id.to_string(), outcome.clone());
        entry.notify.notify_waiters();
        self.completed_notify.notify_waiters();
        Some(outcome)
    }

    pub fn list_now(&self) -> Vec<String> {
        match self.tasks.try_lock() {
            Ok(tasks) => tasks.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    pub async fn list(&self) -> Vec<String> {
        let tasks = self.tasks.lock().await;
        tasks.keys().cloned().collect()
    }

    /// Whether this session has ever registered a background task. The
    /// advertisement path is synchronous, so use a non-blocking probe; if a
    /// concurrent spawn holds the lock, fail open and keep the polling tools.
    pub fn has_tasks(&self) -> bool {
        match self.tasks.try_lock() {
            Ok(tasks) => !tasks.is_empty(),
            Err(_) => true,
        }
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

    struct DropFlag(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn stalled_worker_ack_releases_registry_lock_and_discards_late_command() {
        let registry = Arc::new(BackgroundTaskRegistry::new());
        let (worker_tx, mut worker_rx) = tokio::sync::mpsc::unbounded_channel::<WorkerCommand>();
        let factory_ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let factory_dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let factory_ran_in_task = factory_ran.clone();
        let factory_drop_flag = DropFlag(factory_dropped.clone());
        let spawn_registry = registry.clone();
        let spawning = tokio::spawn(async move {
            spawn_registry
                .spawn_after_on_worker(
                    "delayed-ack",
                    "delegate",
                    &[],
                    Box::new(move || {
                        let _factory_drop_flag = factory_drop_flag;
                        factory_ran_in_task.store(true, std::sync::atomic::Ordering::SeqCst);
                        Box::pin(async {
                            BackgroundTaskOutcome {
                                id: String::new(),
                                description: String::new(),
                                subagent_type: String::new(),
                                state: BackgroundTaskState::Completed,
                                output: "must not run".into(),
                                applied: false,
                                changed_files: Vec::new(),
                            }
                        })
                    }),
                    worker_tx,
                    Duration::from_millis(250),
                )
                .await
        });

        // Hold the command without acknowledging it, exactly as a worker
        // command loop stalled behind a non-yielding task would. Registry
        // operations must remain available throughout the bounded wait.
        let delayed_command = worker_rx.recv().await.expect("spawn command");
        let listed = tokio::time::timeout(Duration::from_millis(50), registry.list())
            .await
            .expect("worker acknowledgment must not hold the task-map mutex");
        assert_eq!(listed, vec!["task_1".to_string()]);

        let error = tokio::time::timeout(Duration::from_secs(1), spawning)
            .await
            .expect("worker acknowledgment wait must be bounded")
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("did not acknowledge"));
        assert!(registry.list().await.is_empty());
        assert!(
            factory_dropped.load(std::sync::atomic::Ordering::SeqCst),
            "rollback should release the queued future factory immediately"
        );
        assert!(
            !registry
                .outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key("task_1")
        );
        assert!(
            registry
                .abort_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
        );

        // Process the queued command after rollback. Its cancellation token
        // must make the worker discard it without invoking the future factory.
        let local_set = tokio::task::LocalSet::new();
        dispatch_worker_command(&local_set, delayed_command);
        local_set.run_until(tokio::task::yield_now()).await;
        assert!(!factory_ran.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn dropping_registry_aborts_an_active_task() {
        let registry = BackgroundTaskRegistry::new();
        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let started_in_task = started.clone();
        let dropped_in_task = dropped.clone();
        registry
            .spawn(
                "drop-owned",
                "delegate",
                Box::new(move || {
                    Box::pin(async move {
                        let _drop_flag = DropFlag(dropped_in_task);
                        started_in_task.store(true, std::sync::atomic::Ordering::SeqCst);
                        std::future::pending::<()>().await;
                        unreachable!("registry-owned task survived forever")
                    })
                }),
            )
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(2), async {
            while !started.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("background task should start");

        drop(registry);

        tokio::time::timeout(Duration::from_secs(2), async {
            while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropping the registry must drop its active task future");
    }

    #[tokio::test]
    async fn closed_worker_result_becomes_a_stable_terminal_failure() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn(
                "lost-worker",
                "delegate",
                Box::new(|| {
                    Box::pin(async {
                        std::future::pending::<()>().await;
                        unreachable!("test worker should be aborted")
                    })
                }),
            )
            .await
            .unwrap();
        let handle = registry
            .abort_handles
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .expect("worker abort handle should be indexed");
        handle.abort();

        let first = registry
            .poll(&id, Duration::from_secs(2))
            .await
            .expect("closed worker channel should remain addressable");
        assert_eq!(first.state, BackgroundTaskState::Failed);
        assert!(first.output.contains("stopped without a result"));

        let replay = registry.poll(&id, Duration::ZERO).await.unwrap();
        assert_eq!(replay.state, BackgroundTaskState::Failed);
        assert_eq!(replay.output, first.output);
    }

    #[tokio::test]
    async fn spawn_and_poll_completed() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn(
                "test",
                "explore",
                Box::new(|| {
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
                }),
            )
            .await
            .unwrap();

        let outcome = registry.poll(&id, Duration::from_secs(2)).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
        assert_eq!(outcome.output, "done");
    }

    #[tokio::test]
    async fn poll_stamps_registry_identity_when_worker_omits_it() {
        // Production run_bg_* paths return empty id/description; the registry
        // must fill them so completed polls keep the human label.
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn(
                "scan deps",
                "general-purpose",
                Box::new(|| {
                    Box::pin(async {
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: "general-purpose".into(),
                            state: BackgroundTaskState::Completed,
                            output: "ok".into(),
                            applied: true,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();

        let outcome = registry.poll(&id, Duration::from_secs(2)).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
        assert_eq!(outcome.id, id);
        assert_eq!(outcome.description, "scan deps");
        assert_eq!(outcome.subagent_type, "general-purpose");
        // Cached re-poll keeps identity.
        let again = registry.poll(&id, Duration::ZERO).await.unwrap();
        assert_eq!(again.id, id);
        assert_eq!(again.description, "scan deps");
    }

    #[tokio::test]
    async fn poll_non_blocking_returns_running() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn(
                "slow",
                "explore",
                Box::new(|| {
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
                }),
            )
            .await
            .unwrap();

        let result = registry.poll(&id, Duration::ZERO).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().state, BackgroundTaskState::Running);
    }

    #[tokio::test]
    async fn dependency_starts_when_parent_never_polls_prerequisite() {
        let registry = BackgroundTaskRegistry::new();
        let prerequisite = registry
            .spawn(
                "prerequisite",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "ready".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();

        let dependent = registry
            .spawn_after(
                "dependent",
                "explore",
                std::slice::from_ref(&prerequisite),
                Box::new(|| {
                    Box::pin(async {
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "ran".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();

        // Only poll the dependent task. Its prerequisite must wake the gate
        // directly when the worker finishes; callers should not need to poll
        // every dependency themselves.
        let outcome = registry
            .poll(&dependent, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
        assert_eq!(outcome.output, "ran");
    }

    #[tokio::test]
    async fn panicked_factory_publishes_failure_and_wakes_unpolled_dependent() {
        let registry = BackgroundTaskRegistry::new();
        let prerequisite = registry
            .spawn(
                "panicked-factory",
                "explore",
                Box::new(|| panic!("factory boom")),
            )
            .await
            .unwrap();
        let dependent_ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dependent_ran_in_task = dependent_ran.clone();
        let dependent = registry
            .spawn_after(
                "after-panic",
                "explore",
                std::slice::from_ref(&prerequisite),
                Box::new(move || {
                    Box::pin(async move {
                        dependent_ran_in_task.store(true, Ordering::SeqCst);
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "must not run".into(),
                            applied: false,
                            changed_files: Vec::new(),
                        }
                    })
                }),
            )
            .await
            .unwrap();

        // Poll only the dependent. The caught parent panic must publish to the
        // stable dependency gate instead of requiring an explicit parent poll.
        let outcome = registry
            .poll(&dependent, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Failed);
        assert!(outcome.output.contains("Dependency"), "{}", outcome.output);
        assert!(!dependent_ran.load(Ordering::SeqCst));

        let prerequisite = registry.poll(&prerequisite, Duration::ZERO).await.unwrap();
        assert_eq!(prerequisite.state, BackgroundTaskState::Failed);
        assert!(prerequisite.output.contains("factory boom"));
    }

    #[tokio::test]
    async fn kill_cancels_running_task() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn(
                "cancellable",
                "delegate",
                Box::new(|| {
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
                }),
            )
            .await
            .unwrap();

        let outcome = registry.kill(&id).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Cancelled);

        let outcome2 = registry.poll(&id, Duration::ZERO).await.unwrap();
        assert_eq!(outcome2.state, BackgroundTaskState::Cancelled);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn kill_remains_authoritative_after_poll_takes_completed_receiver() {
        let registry = Arc::new(BackgroundTaskRegistry::new());
        let release = Arc::new(Notify::new());
        let release_in_task = release.clone();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let id = registry
            .spawn(
                "racing-kill",
                "delegate",
                Box::new(move || {
                    Box::pin(async move {
                        started_tx.send(()).unwrap();
                        release_in_task.notified().await;
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "worker completed".into(),
                            applied: false,
                            changed_files: Vec::new(),
                        }
                    })
                }),
            )
            .await
            .unwrap();
        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("worker should start");

        let poll_registry = registry.clone();
        let poll_id = id.clone();
        let poll = tokio::spawn(async move {
            poll_registry
                .poll(&poll_id, Duration::from_secs(2))
                .await
                .unwrap()
        });

        // Let the poll future take exclusive ownership of result_rx, then
        // keep this single-thread executor occupied while the OS worker sends
        // its result. This deterministically places kill between the worker
        // send and poll's terminal-result commit.
        loop {
            let receiver_taken = registry
                .tasks
                .lock()
                .await
                .get(&id)
                .is_some_and(|entry| entry.result_rx.is_none());
            if receiver_taken {
                break;
            }
            tokio::task::yield_now().await;
        }
        release.notify_one();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            let worker_published = registry
                .outcomes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&id)
                .is_some_and(|outcome| outcome.state == BackgroundTaskState::Completed);
            if worker_published {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "worker should publish its result"
            );
            std::thread::yield_now();
        }

        let killed = registry.kill(&id).await.unwrap();
        assert_eq!(killed.state, BackgroundTaskState::Cancelled);
        let polled = poll.await.unwrap();
        assert_eq!(polled.state, BackgroundTaskState::Cancelled);
        assert_eq!(
            registry.poll(&id, Duration::ZERO).await.unwrap().state,
            BackgroundTaskState::Cancelled
        );
    }

    #[tokio::test]
    async fn wait_all_completes_when_all_done() {
        let registry = BackgroundTaskRegistry::new();
        let id1 = registry
            .spawn(
                "t1",
                "explore",
                Box::new(|| {
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
                }),
            )
            .await
            .unwrap();
        let id2 = registry
            .spawn(
                "t2",
                "explore",
                Box::new(|| {
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
                }),
            )
            .await
            .unwrap();

        let results = registry.wait_all(&[id1, id2], Duration::from_secs(2)).await;
        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|r| r.state == BackgroundTaskState::Completed)
        );
    }

    #[tokio::test]
    async fn kill_is_idempotent() {
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn(
                "idempotent",
                "explore",
                Box::new(|| {
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
                }),
            )
            .await
            .unwrap();

        let outcome = registry.poll(&id, Duration::from_secs(2)).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);

        let outcome = registry.kill(&id).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
    }

    #[tokio::test]
    async fn parallel_tasks_run_concurrently() {
        // Two tasks that each sleep 300ms. On a single-thread LocalSet they
        // would take ~600ms total. On the multi-worker pool they should
        // complete in ~300ms — well under 500ms. This verifies true
        // OS-thread parallelism among background tasks.
        let registry = BackgroundTaskRegistry::new();
        let id1 = registry
            .spawn(
                "sleep1",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        BackgroundTaskOutcome {
                            id: "sleep1".into(),
                            description: "sleep1".into(),
                            subagent_type: "explore".into(),
                            state: BackgroundTaskState::Completed,
                            output: "done1".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();
        let id2 = registry
            .spawn(
                "sleep2",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        BackgroundTaskOutcome {
                            id: "sleep2".into(),
                            description: "sleep2".into(),
                            subagent_type: "explore".into(),
                            state: BackgroundTaskState::Completed,
                            output: "done2".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();

        let start = std::time::Instant::now();
        let results = registry.wait_all(&[id1, id2], Duration::from_secs(5)).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 2);
        assert!(
            results
                .iter()
                .all(|r| r.state == BackgroundTaskState::Completed)
        );
        // If tasks ran sequentially on one thread, this would be ~600ms.
        // With the worker pool, both run concurrently → ~300ms.
        // Allow generous headroom for CI / scheduling jitter.
        assert!(
            elapsed < Duration::from_millis(550),
            "parallel tasks took {elapsed:?} — expected concurrent execution"
        );
    }

    #[tokio::test]
    async fn dispatches_up_to_worker_count_concurrently() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        assert_eq!(BG_WORKER_THREADS, MAX_BG_TASKS - 1);

        let registry = BackgroundTaskRegistry::new();
        let started = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Notify::new());
        let mut ids = Vec::with_capacity(BG_WORKER_THREADS);

        for task_idx in 0..BG_WORKER_THREADS {
            let started = started.clone();
            let release = release.clone();
            ids.push(
                registry
                    .spawn(
                        &format!("worker-{task_idx}"),
                        "explore",
                        Box::new(move || {
                            Box::pin(async move {
                                started.fetch_add(1, Ordering::SeqCst);
                                release.notified().await;
                                BackgroundTaskOutcome {
                                    id: format!("worker-{task_idx}"),
                                    description: format!("worker-{task_idx}"),
                                    subagent_type: "explore".into(),
                                    state: BackgroundTaskState::Completed,
                                    output: "done".into(),
                                    applied: false,
                                    changed_files: vec![],
                                }
                            })
                        }),
                    )
                    .await
                    .unwrap(),
            );
        }

        tokio::time::timeout(Duration::from_secs(5), async {
            while started.load(Ordering::SeqCst) < BG_WORKER_THREADS {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("all configured workers should start a task");

        release.notify_waiters();
        let outcomes = registry.wait_all(&ids, Duration::from_secs(5)).await;
        assert_eq!(outcomes.len(), BG_WORKER_THREADS);
        assert!(
            outcomes
                .iter()
                .all(|outcome| outcome.state == BackgroundTaskState::Completed)
        );
    }

    #[tokio::test]
    async fn wait_any_wakes_on_completion_without_busy_poll() {
        // A task that completes after 200ms. wait_any should return promptly
        // (well under the old 200ms poll interval × number of polls), proving
        // the Notify-driven wake works.
        let registry = BackgroundTaskRegistry::new();
        let id = registry
            .spawn(
                "notifier",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        BackgroundTaskOutcome {
                            id: "notifier".into(),
                            description: "notifier".into(),
                            subagent_type: "explore".into(),
                            state: BackgroundTaskState::Completed,
                            output: "woke".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();

        let start = std::time::Instant::now();
        let results = registry.wait_any(&[id], Duration::from_secs(5)).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].state, BackgroundTaskState::Completed);
        // Should complete in ~200ms (the task sleep) plus a small wake margin.
        // The old 200ms busy-poll could add up to 200ms of latency on top.
        assert!(
            elapsed < Duration::from_millis(400),
            "wait_any took {elapsed:?} — expected Notify-driven wake"
        );
    }

    #[tokio::test]
    async fn dependent_task_waits_for_success() {
        let registry = BackgroundTaskRegistry::new();
        let first = registry
            .spawn(
                "first",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "ok".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();
        let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ran_task = ran.clone();
        let second = registry
            .spawn_after(
                "second",
                "explore",
                std::slice::from_ref(&first),
                Box::new(move || {
                    Box::pin(async move {
                        ran_task.store(true, std::sync::atomic::Ordering::SeqCst);
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "done".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();
        assert!(!ran.load(std::sync::atomic::Ordering::SeqCst));
        let _ = registry.poll(&first, Duration::from_secs(1)).await;
        let results = registry.wait_all(&[second], Duration::from_secs(1)).await;
        assert_eq!(results[0].state, BackgroundTaskState::Completed);
        assert!(ran.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn capacity_pruning_preserves_dependency_entries() {
        let registry = BackgroundTaskRegistry::new();
        let mut completed = Vec::with_capacity(MAX_BG_TASKS);
        for index in 0..MAX_BG_TASKS {
            completed.push(
                registry
                    .spawn(
                        &format!("completed-{index}"),
                        "explore",
                        Box::new(|| {
                            Box::pin(async {
                                BackgroundTaskOutcome {
                                    id: String::new(),
                                    description: String::new(),
                                    subagent_type: String::new(),
                                    state: BackgroundTaskState::Completed,
                                    output: "done".into(),
                                    applied: false,
                                    changed_files: vec![],
                                }
                            })
                        }),
                    )
                    .await
                    .unwrap(),
            );
        }
        let _ = registry.wait_all(&completed, Duration::from_secs(2)).await;

        // The registry is at capacity, but the first completed task is still
        // a valid dependency. Pruning must not remove it between validation
        // and dependency-gate construction.
        let dependent = registry
            .spawn_after(
                "after-completed",
                "explore",
                std::slice::from_ref(&completed[0]),
                Box::new(|| {
                    Box::pin(async {
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "after".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .unwrap();
        let outcome = registry
            .poll(&dependent, Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
    }

    #[tokio::test]
    async fn queued_dependency_gate_survives_capacity_pruning() {
        let registry = BackgroundTaskRegistry::new();
        let release = Arc::new(Notify::new());
        let release_in_task = release.clone();
        let prerequisite = registry
            .spawn(
                "queued-prerequisite",
                "explore",
                Box::new(move || {
                    Box::pin(async move {
                        release_in_task.notified().await;
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "ready".into(),
                            applied: false,
                            changed_files: Vec::new(),
                        }
                    })
                }),
            )
            .await
            .unwrap();
        // Capture exactly the stable gate an already-queued dependent owns,
        // but intentionally do not poll it until after capacity pruning.
        let queued_gate = {
            let tasks = registry.tasks.lock().await;
            let entry = tasks.get(&prerequisite).unwrap();
            DependencyGate {
                id: prerequisite.clone(),
                terminal_outcome: entry.terminal_outcome.clone(),
                notify: entry.notify.clone(),
            }
        };

        let mut fillers = Vec::with_capacity(MAX_BG_TASKS - 1);
        for index in 0..(MAX_BG_TASKS - 1) {
            fillers.push(
                registry
                    .spawn(
                        &format!("prune-filler-{index}"),
                        "explore",
                        Box::new(|| {
                            Box::pin(async {
                                BackgroundTaskOutcome {
                                    id: String::new(),
                                    description: String::new(),
                                    subagent_type: String::new(),
                                    state: BackgroundTaskState::Completed,
                                    output: "done".into(),
                                    applied: false,
                                    changed_files: Vec::new(),
                                }
                            })
                        }),
                    )
                    .await
                    .unwrap(),
            );
        }
        let filler_results = registry.wait_all(&fillers, Duration::from_secs(2)).await;
        assert!(
            filler_results
                .iter()
                .all(|outcome| outcome.state.is_terminal())
        );

        release.notify_one();
        assert_eq!(
            registry
                .poll(&prerequisite, Duration::from_secs(2))
                .await
                .unwrap()
                .state,
            BackgroundTaskState::Completed
        );

        let replacement = registry
            .spawn(
                "forces-capacity-prune",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "replacement".into(),
                            applied: false,
                            changed_files: Vec::new(),
                        }
                    })
                }),
            )
            .await
            .unwrap();
        assert!(
            !registry.tasks.lock().await.contains_key(&prerequisite),
            "capacity pruning should remove the registry entry in this regression"
        );

        let gate_result = tokio::time::timeout(
            Duration::from_millis(250),
            wait_for_dependencies(vec![queued_gate]),
        )
        .await
        .expect("a pruned prerequisite must not strand an existing dependent");
        assert_eq!(gate_result, Ok(()));
        assert_eq!(
            registry
                .poll(&replacement, Duration::from_secs(2))
                .await
                .unwrap()
                .state,
            BackgroundTaskState::Completed
        );
    }

    #[tokio::test]
    async fn capacity_preserves_unobserved_completed_tasks_for_retrieval() {
        let registry = BackgroundTaskRegistry::new();
        let mut ids = Vec::with_capacity(MAX_BG_TASKS);
        for index in 0..MAX_BG_TASKS {
            ids.push(
                registry
                    .spawn(
                        &format!("unpolled-{index}"),
                        "explore",
                        Box::new(move || {
                            Box::pin(async move {
                                BackgroundTaskOutcome {
                                    id: String::new(),
                                    description: String::new(),
                                    subagent_type: String::new(),
                                    state: BackgroundTaskState::Completed,
                                    output: format!("done-{index}"),
                                    applied: false,
                                    changed_files: vec![],
                                }
                            })
                        }),
                    )
                    .await
                    .unwrap(),
            );
        }
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let terminal = registry
                    .outcomes
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .filter(|outcome| outcome.state.is_terminal())
                    .count();
                if terminal == MAX_BG_TASKS {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("all tasks should publish terminal results");
        let intermediate = registry.poll_many_inner(&ids, Duration::ZERO, false).await;
        assert!(
            intermediate
                .iter()
                .all(|outcome| outcome.state == BackgroundTaskState::Completed)
        );
        assert!(
            registry
                .tasks
                .lock()
                .await
                .values()
                .all(|entry| !entry.observed),
            "an internal wait snapshot must not acknowledge results before return"
        );

        let error = registry
            .spawn(
                "after-unpolled",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "ran".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .expect_err("unobserved terminal results must retain their slots");
        let capacity = error
            .downcast_ref::<BackgroundTaskCapacityError>()
            .expect("capacity admission should return its typed error");
        assert_eq!(capacity.maximum, MAX_BG_TASKS);
        assert_eq!(capacity.running, 0);
        assert_eq!(capacity.unobserved_terminal, MAX_BG_TASKS);
        assert!(error.to_string().contains("get_task_output or wait_tasks"));

        for (index, id) in ids.iter().enumerate() {
            let outcome = registry.poll(id, Duration::ZERO).await.unwrap();
            assert_eq!(outcome.state, BackgroundTaskState::Completed);
            assert_eq!(outcome.output, format!("done-{index}"));
        }
    }

    #[tokio::test]
    async fn acknowledged_completed_tasks_are_pruned_to_admit_later_work() {
        let registry = BackgroundTaskRegistry::new();
        let mut ids = Vec::with_capacity(MAX_BG_TASKS);
        for index in 0..MAX_BG_TASKS {
            ids.push(
                registry
                    .spawn(
                        &format!("observed-{index}"),
                        "explore",
                        Box::new(|| {
                            Box::pin(async {
                                BackgroundTaskOutcome {
                                    id: String::new(),
                                    description: String::new(),
                                    subagent_type: String::new(),
                                    state: BackgroundTaskState::Completed,
                                    output: "observed".into(),
                                    applied: false,
                                    changed_files: vec![],
                                }
                            })
                        }),
                    )
                    .await
                    .unwrap(),
            );
        }
        let observed = registry.wait_all(&ids, Duration::from_secs(2)).await;
        assert!(observed.iter().all(|outcome| outcome.state.is_terminal()));

        let next = registry
            .spawn(
                "seventeenth-cumulative-task",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "later".into(),
                            applied: false,
                            changed_files: vec![],
                        }
                    })
                }),
            )
            .await
            .expect("acknowledged terminal entries should be reclaimable");
        let outcome = registry.poll(&next, Duration::from_secs(1)).await.unwrap();
        assert_eq!(outcome.state, BackgroundTaskState::Completed);
        assert_eq!(outcome.output, "later");
    }

    #[tokio::test]
    async fn capacity_pruning_reclaims_observed_panicked_futures() {
        let registry = BackgroundTaskRegistry::new();
        let mut ids = Vec::with_capacity(MAX_BG_TASKS);
        for index in 0..MAX_BG_TASKS {
            ids.push(
                registry
                    .spawn(
                        &format!("panicked-{index}"),
                        "explore",
                        Box::new(|| Box::pin(async { panic!("future boom") })),
                    )
                    .await
                    .unwrap(),
            );
        }
        // Wait for the workers to publish every caught panic without polling
        // any registry entry. A fixed sleep made this regression sensitive to
        // slow or heavily loaded CI hosts.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let panicked = registry
                    .outcomes
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .values()
                    .filter(|outcome| {
                        outcome.state == BackgroundTaskState::Failed
                            && outcome.output.contains("future boom")
                    })
                    .count();
                if panicked == MAX_BG_TASKS {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("all panicked futures should publish terminal failures");
        let observed = registry.poll_many(&ids, Duration::ZERO).await;
        assert!(observed.iter().all(|outcome| {
            outcome.state == BackgroundTaskState::Failed && outcome.output.contains("future boom")
        }));

        let next = registry
            .spawn(
                "after-panics",
                "explore",
                Box::new(|| {
                    Box::pin(async {
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: String::new(),
                            state: BackgroundTaskState::Completed,
                            output: "ran".into(),
                            applied: false,
                            changed_files: Vec::new(),
                        }
                    })
                }),
            )
            .await
            .expect("observed panics should be reclaimable terminal tasks");
        assert_eq!(
            registry
                .poll(&next, Duration::from_secs(1))
                .await
                .unwrap()
                .state,
            BackgroundTaskState::Completed
        );
    }

    #[tokio::test]
    async fn rejects_unknown_dependency() {
        let registry = BackgroundTaskRegistry::new();
        let result = registry
            .spawn_after(
                "bad",
                "explore",
                &["task_missing".into()],
                Box::new(|| Box::pin(async { unreachable!() })),
            )
            .await;
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unknown dependency")
        );
    }
}
