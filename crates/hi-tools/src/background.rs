//! Background command execution with polling.
//!
//! `bash` with `run_in_background: true` starts a long-lived/blocking process
//! (a dev server, a file watcher, a slow build) and returns an id immediately
//! instead of waiting for it to exit. The agent then drains incremental output
//! with `bash_output` and stops it with `bash_kill`.
//!
//! Each background process is driven by a detached Tokio task that continuously
//! pumps stdout/stderr into a shared, size-bounded buffer and records the exit
//! status — so the pipes are always drained (never deadlocking) and a poll is a
//! cheap read of already-collected output.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use tokio::sync::Notify;

mod execution;
use execution::{drive, stop_and_reap};

#[path = "background_names.rs"]
mod names;

#[cfg(test)]
#[path = "background/lifecycle_tests.rs"]
mod lifecycle_tests;
use names::handle_id;
pub use names::shell_title;

/// Cap on retained per-process output. Beyond this we drop the oldest bytes (a
/// ring buffer): a chatty server left unpolled can't grow memory without bound.
const MAX_BG_BUFFER: usize = 256 * 1024;
/// Cap on retained processes. When exceeded, already-exited entries are pruned
/// oldest-first so a long session that starts many servers can't leak handles.
const MAX_BG_PROCS: usize = 64;
/// Workspace teardown must not race a killed process that still owns open
/// descriptors or can execute a final filesystem write while being reaped.
const QUIESCENT_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Clone, Copy, PartialEq, Eq)]
enum BgState {
    Running,
    Exited(Option<i32>),
    Killed,
    Failed,
}

/// How a background process came to exist. Turn-scoped cleanup keys on this:
/// a process the model *deliberately* started with `run_in_background: true`
/// (or a background download) is long-lived work the user is owed — turn end,
/// turn cancel, and pre-verification cleanup must not reap it (observed loss:
/// two ~800 GB downloads killed at turn end hours before completion). An
/// *auto-backgrounded* process — a foreground command that outgrew its timeout
/// and was adopted — is incidental turn state and is still reaped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BgOrigin {
    Requested,
    AutoBackgrounded,
}

/// Shared state for one background process: the command, its process-group id
/// (for tree-kill), and the mutable buffer/cursor/status the driver task fills.
struct BgProc {
    command: String,
    /// Short human label for UI / model status lines (never raw JSON).
    title: String,
    pgid: Option<i32>,
    origin: BgOrigin,
    effect_baseline: Option<Arc<EffectBaseline>>,
    managed_job: Option<crate::job_lifecycle::ManagedBackgroundJob>,
    inner: Mutex<BgInner>,
    reaped: Notify,
    /// Woken on every output append and lifecycle transition, so a blocking
    /// [`BackgroundRegistry::poll_wait`] sleeps instead of spinning.
    changed: Notify,
}

/// A handle the model named that this registry has never seen. The registry
/// records these so the agent can tell a *guessed* id (nothing has ever run
/// under it) from a *pruned* one (a real process was forgotten at capacity).
/// Guessed ids are the model's own invention — the agent can correct the
/// model without surfacing anything to the user; pruned ids are a real
/// limitation the user may need to know about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownHandle {
    pub id: String,
    /// Whether the registry was empty when the id was named. An empty
    /// registry means the id cannot have been pruned — it was never real.
    pub registry_was_empty: bool,
}

impl From<UnknownHandle> for crate::UnknownBackgroundHandle {
    fn from(handle: UnknownHandle) -> Self {
        crate::UnknownBackgroundHandle {
            id: handle.id,
            registry_was_empty: handle.registry_was_empty,
        }
    }
}

/// Workspace/runtime-owned background process registry. Separate registries do
/// not share handles or cleanup, so two agents cannot poll or kill each other's
/// processes.
pub struct BackgroundRegistry {
    processes: Mutex<HashMap<String, Arc<BgProc>>>,
    counter: AtomicU64,
    /// Slots reserved while a child is being spawned or adopted. The process
    /// map lock makes reservation and insertion a single capacity decision,
    /// so concurrent background launches cannot race past `MAX_BG_PROCS`.
    reserved_slots: AtomicUsize,
    /// Blocks new reservations while a workspace lifecycle operation takes a
    /// stable, fully-reaped snapshot.  Reservation always acquires the
    /// process-map lock before observing this flag; the quiescence path sets
    /// it before taking that same lock, closing the otherwise possible gap
    /// between the reserved-slot and map checks.
    quiescing: std::sync::atomic::AtomicBool,
    /// Handles named by callers that were not in the registry, with whether
    /// the registry was empty at the time. Bounded FIFO so a model that
    /// guesses ids in a loop cannot grow this without bound.
    unknown_handles: Mutex<VecDeque<UnknownHandle>>,
    /// Optional per-registry override used by embedded callers and tests. A
    /// registry-local value keeps timing controls out of process-global
    /// environment state.
    poll_wait_base_secs: AtomicU64,
    lifecycle: crate::job_lifecycle::BackgroundJobLifecycleSlot,
}

/// Cap on remembered unknown handles. Bounded so a guessing loop cannot grow
/// memory; the agent only needs the most recent misses.
const MAX_UNKNOWN_HANDLES: usize = 16;
const POLL_WAIT_USE_ENV: u64 = u64::MAX;

impl Default for BackgroundRegistry {
    fn default() -> Self {
        Self {
            processes: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(1),
            reserved_slots: AtomicUsize::new(0),
            quiescing: std::sync::atomic::AtomicBool::new(false),
            unknown_handles: Mutex::new(VecDeque::new()),
            poll_wait_base_secs: AtomicU64::new(POLL_WAIT_USE_ENV),
            lifecycle: crate::job_lifecycle::BackgroundJobLifecycleSlot::default(),
        }
    }
}

struct EffectBaseline {
    root: PathBuf,
    state_root: PathBuf,
    snapshot: crate::effects::WorkspaceSnapshot,
}

struct BgInner {
    /// Full retained combined stdout+stderr (front-trimmed past `MAX_BG_BUFFER`).
    output: String,
    /// Exact number of bytes evicted from the front of `output`. Together with
    /// `read_position`, this gives every byte an absolute stream position, so a
    /// poll can report precisely how much unread output was lost when the ring
    /// wrapped instead of silently rewinding a relative cursor.
    dropped_bytes: u64,
    /// Absolute byte position immediately after the output returned by the last
    /// poll. Only newer bytes are delivered next time. This deliberately is not
    /// clamped when the ring wraps: `dropped_bytes - read_position` is the exact
    /// unread omission the next poll must surface.
    read_position: u64,
    state: BgState,
    /// Native child/drain cleanup is complete; publication may still await
    /// the workspace callback. A later kill cannot cancel completed work.
    native_exited: bool,
    reaped: bool,
    /// Effects are sealed on the first observation after the process becomes
    /// terminal, so later unrelated workspace edits cannot be attributed to it.
    terminal_effects: Option<Result<crate::ToolEffects, String>>,
    /// Consecutive polls that returned no fresh output while running. Drives
    /// the escalating default wait in [`BackgroundRegistry::poll_wait_default`]
    /// — the quieter the process, the longer the next default poll parks.
    /// Reset whenever a poll delivers output.
    empty_polls: u32,
}

impl BgInner {
    fn running(output: String) -> Self {
        let mut inner = Self {
            output,
            dropped_bytes: 0,
            read_position: 0,
            state: BgState::Running,
            native_exited: false,
            reaped: false,
            terminal_effects: None,
            empty_polls: 0,
        };
        trim_output_to_cap(&mut inner);
        inner
    }

    fn native_running(&self) -> bool {
        self.state == BgState::Running && !self.native_exited
    }
}

impl Drop for BackgroundRegistry {
    fn drop(&mut self) {
        kill_all_from(self);
    }
}

#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));
#[cfg(test)]
static TEST_REGISTRY: std::sync::LazyLock<BackgroundRegistry> =
    std::sync::LazyLock::new(BackgroundRegistry::default);

/// Start `command` in the background and return its handle id — a
/// command-derived name like `cargo-test_3` (see [`handle_id`]).
#[cfg(test)]
pub(crate) fn spawn(command: &str) -> Result<String> {
    let runner = crate::ProcessRunner::from_current_dir()?;
    TEST_REGISTRY.spawn(&runner, command)
}

impl BackgroundRegistry {
    pub fn set_job_lifecycle(&self, lifecycle: Arc<dyn crate::BackgroundJobLifecycle>) {
        self.lifecycle.set(lifecycle);
    }

    pub async fn pending_job_settlements(&self) -> Vec<crate::BackgroundJobId> {
        self.lifecycle.pending().await
    }

    pub async fn settle_jobs_after_workspace(
        &self,
        pending: &[crate::BackgroundJobId],
    ) -> Result<()> {
        self.lifecycle
            .settle_after_workspace(pending)
            .await
            .map_err(anyhow::Error::msg)
    }

    pub fn spawn(&self, runner: &crate::ProcessRunner, command: &str) -> Result<String> {
        self.spawn_with_baseline(runner, command, None)
    }

    pub(crate) async fn spawn_tracked(
        &self,
        runner: &crate::ProcessRunner,
        command: &str,
        root: &Path,
        state_root: &Path,
        snapshot: crate::effects::WorkspaceSnapshot,
    ) -> Result<String> {
        self.spawn_managed(
            runner,
            command,
            Some(EffectBaseline {
                root: root.to_path_buf(),
                state_root: state_root.to_path_buf(),
                snapshot,
            }),
            None,
        )
        .await
    }

    /// Start a known live writer only after the workspace job lifecycle has
    /// admitted it. Callers that cannot take a complete workspace snapshot
    /// still use the same admission and terminal-settlement path; registration
    /// denial happens before `ProcessRunner::spawn_shell` is reached.
    pub(crate) async fn spawn_managed_live_writer(
        &self,
        runner: &crate::ProcessRunner,
        command: &str,
    ) -> Result<String> {
        self.spawn_managed(
            runner,
            command,
            None,
            Some(crate::BackgroundJobEffect::LiveWriter),
        )
        .await
    }

    /// Adopt an already-running child that a foreground command handed off
    /// because it exceeded its foreground budget (auto-background-on-timeout).
    /// The child keeps running under a fresh command-named handle (see
    /// [`handle_id`]), seeded with the output it produced while in the
    /// foreground so a later `bash_output` shows the whole run. The caller
    /// must have defused any process-group kill guard before handing the
    /// child over — this registry now owns its lifecycle. `pgid` is the
    /// child's process-group id for tree-kill.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn adopt(
        &self,
        command: &str,
        child: tokio::process::Child,
        stdout: Option<tokio::process::ChildStdout>,
        stderr: Option<tokio::process::ChildStderr>,
        pgid: Option<i32>,
        seed_output: String,
        baseline: (PathBuf, PathBuf, crate::effects::WorkspaceSnapshot),
    ) -> Result<String> {
        self.adopt_with_baseline(
            command,
            child,
            stdout,
            stderr,
            pgid,
            seed_output,
            Some(baseline),
        )
        .await
    }

    /// Adopt a definitely read-only command without retaining a workspace
    /// snapshot. It still gets the same lifecycle/output handling, but a
    /// terminal poll cannot attribute unrelated file changes to it.
    pub(crate) async fn adopt_read_only(
        &self,
        command: &str,
        child: tokio::process::Child,
        stdout: Option<tokio::process::ChildStdout>,
        stderr: Option<tokio::process::ChildStderr>,
        pgid: Option<i32>,
        seed_output: String,
    ) -> Result<String> {
        self.adopt_with_baseline(command, child, stdout, stderr, pgid, seed_output, None)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn adopt_with_baseline(
        &self,
        command: &str,
        child: tokio::process::Child,
        stdout: Option<tokio::process::ChildStdout>,
        stderr: Option<tokio::process::ChildStderr>,
        pgid: Option<i32>,
        seed_output: String,
        baseline: Option<(PathBuf, PathBuf, crate::effects::WorkspaceSnapshot)>,
    ) -> Result<String> {
        if let Err(error) = self.reserve_slot() {
            // The caller has already handed ownership of this child to us.
            // Kill and reap it before returning the capacity error so a
            // timed-out foreground command cannot escape the registry.
            if let Some(pgid) = pgid {
                crate::tools::kill_group(pgid);
            }
            let mut child = child;
            let _ = child.start_kill();
            tokio::spawn(async move {
                let _ = child.wait().await;
            });
            return Err(error);
        }
        let id = handle_id(command, self.counter.fetch_add(1, Ordering::Relaxed));
        let effect = if baseline.is_some() {
            crate::BackgroundJobEffect::LiveWriter
        } else {
            crate::BackgroundJobEffect::ReadOnly
        };
        let managed_job = match self
            .lifecycle
            .register(&id, crate::BackgroundJobKind::Process, effect, command)
            .await
        {
            Ok(job) => job,
            Err(error) => {
                stop_and_reap(child, pgid).await;
                self.release_slot();
                return Err(anyhow::Error::msg(error));
            }
        };
        if let Err(error) =
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::JobAfterSpawn)
        {
            stop_and_reap(child, pgid).await;
            if let Some(job) = &managed_job {
                let _ = job
                    .observe(
                        crate::BackgroundJobTerminal::Failed,
                        Some(error.to_string()),
                    )
                    .await;
            }
            self.release_slot();
            return Err(error.into());
        }
        let proc = Arc::new(BgProc {
            command: command.to_string(),
            title: shell_title(command),
            pgid,
            origin: BgOrigin::AutoBackgrounded,
            effect_baseline: baseline.map(|(root, state_root, snapshot)| {
                Arc::new(EffectBaseline {
                    root,
                    state_root,
                    snapshot,
                })
            }),
            managed_job,
            inner: Mutex::new(BgInner::running(seed_output)),
            reaped: Notify::new(),
            changed: Notify::new(),
        });
        {
            let mut reg = self.processes.lock().unwrap();
            reg.insert(id.clone(), proc.clone());
        }
        self.release_slot();
        // Every child gets its driver immediately — the driver only drains
        // pipes and reaps, which is cheap. Gating drivers behind a permit pool
        // meant the 5th+ concurrent job was never drained: it wedged on a full
        // pipe, reported "still running" forever after exiting, and leaked.
        tokio::spawn(async move {
            drive(proc, child, stdout, stderr).await;
        });
        Ok(id)
    }

    fn spawn_with_baseline(
        &self,
        runner: &crate::ProcessRunner,
        command: &str,
        effect_baseline: Option<EffectBaseline>,
    ) -> Result<String> {
        // Background commands get the same irreversible-op guard as foreground ones.
        if let Some(reason) = crate::guard::catastrophic_op(command) {
            bail!(
                "refused: this command {reason}. It's blocked as irreversible — the per-turn \
             checkpoint can't undo it. Ask the user to run it themselves if it's genuinely \
             needed (or set HI_ALLOW_DANGEROUS=1)."
            );
        }

        self.reserve_slot()?;
        let mut child = match runner.spawn_shell(command) {
            Ok(child) => child,
            Err(error) => {
                self.release_slot();
                return Err(error);
            }
        };
        let pgid = child.id().map(|p| p as i32);
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let id = handle_id(command, self.counter.fetch_add(1, Ordering::Relaxed));
        let proc = Arc::new(BgProc {
            command: command.to_string(),
            title: shell_title(command),
            pgid,
            origin: BgOrigin::Requested,
            effect_baseline: effect_baseline.map(Arc::new),
            managed_job: None,
            inner: Mutex::new(BgInner::running(String::new())),
            reaped: Notify::new(),
            changed: Notify::new(),
        });

        {
            let mut reg = self.processes.lock().unwrap();
            reg.insert(id.clone(), proc.clone());
        }
        self.release_slot();

        // Detached driver: drain both pipes to EOF, then reap and record the status.
        // Every child gets its driver immediately — the driver only drains
        // pipes and reaps, which is cheap. Gating drivers behind a permit pool
        // meant the 5th+ concurrent job was never drained: it wedged on a full
        // pipe, reported "still running" forever after exiting, and leaked.
        tokio::spawn(async move {
            drive(proc, child, stdout, stderr).await;
        });
        Ok(id)
    }

    async fn spawn_managed(
        &self,
        runner: &crate::ProcessRunner,
        command: &str,
        effect_baseline: Option<EffectBaseline>,
        effect_override: Option<crate::BackgroundJobEffect>,
    ) -> Result<String> {
        if let Some(reason) = crate::guard::catastrophic_op(command) {
            bail!("refused: this command {reason}");
        }
        self.reserve_slot()?;
        let id = handle_id(command, self.counter.fetch_add(1, Ordering::Relaxed));
        let effect = effect_override.unwrap_or_else(|| {
            if crate::shell_policy::classify_shell_command(command).is_proven_read_only() {
                crate::BackgroundJobEffect::ReadOnly
            } else {
                crate::BackgroundJobEffect::LiveWriter
            }
        });
        let managed_job = match self
            .lifecycle
            .register(&id, crate::BackgroundJobKind::Process, effect, command)
            .await
        {
            Ok(job) => job,
            Err(error) => {
                self.release_slot();
                return Err(anyhow::Error::msg(error));
            }
        };
        if let Err(error) =
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::ToolBeforeStart)
        {
            if let Some(job) = &managed_job {
                let _ = job
                    .observe(
                        crate::BackgroundJobTerminal::FailedBeforeStart,
                        Some(error.to_string()),
                    )
                    .await;
            }
            self.release_slot();
            return Err(error.into());
        }
        let mut child = match runner.spawn_shell(command) {
            Ok(child) => child,
            Err(error) => {
                if let Some(job) = &managed_job {
                    let _ = job
                        .observe(
                            crate::BackgroundJobTerminal::FailedBeforeStart,
                            Some(error.to_string()),
                        )
                        .await;
                }
                self.release_slot();
                return Err(error);
            }
        };
        let pgid = child.id().map(|pid| pid as i32);
        if let Err(error) =
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::JobAfterSpawn)
        {
            stop_and_reap(child, pgid).await;
            if let Some(job) = &managed_job {
                let _ = job
                    .observe(
                        crate::BackgroundJobTerminal::Failed,
                        Some(error.to_string()),
                    )
                    .await;
            }
            self.release_slot();
            return Err(error.into());
        }
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let proc = Arc::new(BgProc {
            command: command.to_string(),
            title: shell_title(command),
            pgid,
            origin: BgOrigin::Requested,
            effect_baseline: effect_baseline.map(Arc::new),
            managed_job,
            inner: Mutex::new(BgInner::running(String::new())),
            reaped: Notify::new(),
            changed: Notify::new(),
        });
        self.processes
            .lock()
            .unwrap()
            .insert(id.clone(), proc.clone());
        self.release_slot();
        tokio::spawn(async move { drive(proc, child, stdout, stderr).await });
        Ok(id)
    }

    /// Reserve a retained-process slot before starting a child. Exited
    /// entries are reclaimed first; live entries are never evicted because
    /// doing so would lose the only handle capable of stopping their process
    /// group.
    fn reserve_slot(&self) -> Result<()> {
        let mut reg = self.processes.lock().unwrap();
        if self.quiescing.load(Ordering::Acquire) {
            bail!("workspace lifecycle operation is waiting for background processes to stop");
        }
        prune(&mut reg);
        let reserved = self.reserved_slots.load(Ordering::Acquire);
        if reg.len().saturating_add(reserved) >= MAX_BG_PROCS {
            bail!(
                "background process capacity reached ({MAX_BG_PROCS} live or starting); \
                 stop a running background process before starting another"
            );
        }
        self.reserved_slots.fetch_add(1, Ordering::Release);
        Ok(())
    }

    fn release_slot(&self) {
        let previous = self.reserved_slots.fetch_sub(1, Ordering::Release);
        debug_assert!(
            previous > 0,
            "background slot released without a reservation"
        );
    }

    pub fn poll(&self, id: &str) -> Result<String> {
        poll_from(self, id)
    }

    /// [`poll_wait`](Self::poll_wait) with an adaptive budget — the default
    /// for a `bash_output` call that names no `wait_secs`. The registry's
    /// change notification is the watcher: an empty poll of a running process
    /// parks on it instead of returning instantly, so a model that never
    /// passes `wait_secs` still cannot turn waiting into an API-call-per-poll
    /// loop. Patience escalates with consecutive empty polls
    /// ([`default_poll_wait_budget`]) and any fresh output resets it; polls
    /// with output already pending (or a terminal process) return immediately
    /// as before.
    pub async fn poll_wait_default(&self, id: &str) -> Result<String> {
        self.poll_wait_default_streaming(id, &mut |_| {}).await
    }

    /// [`poll_wait_default`](Self::poll_wait_default) with a live output callback.
    pub async fn poll_wait_default_streaming(
        &self,
        id: &str,
        on_line: &mut (dyn FnMut(&str) + Send),
    ) -> Result<String> {
        let empty_polls = {
            let proc = lookup(self, id)?;
            let inner = proc.inner.lock().unwrap();
            inner.empty_polls
        };
        let base_override_secs = match self.poll_wait_base_secs.load(Ordering::Acquire) {
            POLL_WAIT_USE_ENV => None,
            base => Some(base),
        };
        self.poll_wait_streaming(
            id,
            default_poll_wait_budget(empty_polls, base_override_secs),
            on_line,
        )
        .await
    }

    /// Override the adaptive default wait for this registry. `None` restores
    /// the standalone environment-based default. This is registry-local, so
    /// changing one agent's polling policy cannot affect another agent.
    pub fn set_poll_wait_base_secs(&self, base_secs: Option<u64>) {
        self.poll_wait_base_secs
            .store(base_secs.unwrap_or(POLL_WAIT_USE_ENV), Ordering::Release);
    }

    /// Like [`poll`](Self::poll), but blocks up to `wait` until the process
    /// produces new output or reaches a terminal state — so one tool call can
    /// cover minutes of waiting instead of a tight model-round poll loop. On
    /// timeout it returns the normal idle status. The wait sleeps on a
    /// notification (no spinning) and holds no locks while parked.
    pub async fn poll_wait(&self, id: &str, wait: std::time::Duration) -> Result<String> {
        self.poll_wait_streaming(id, wait, &mut |_| {}).await
    }

    /// [`poll_wait`](Self::poll_wait) that also forwards newly buffered output
    /// through `on_line` as it arrives, so a UI can paint a live tail during a
    /// multi-minute wait instead of staying blank until the poll returns.
    pub async fn poll_wait_streaming(
        &self,
        id: &str,
        wait: std::time::Duration,
        on_line: &mut (dyn FnMut(&str) + Send),
    ) -> Result<String> {
        let proc = lookup(self, id)?;
        let mut streamed = {
            let inner = proc.inner.lock().unwrap();
            inner.read_position
        };
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            let notified = proc.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let (omitted, fresh, end, done) = {
                let inner = proc.inner.lock().unwrap();
                let (omitted, fresh, end) = output_since(&inner, streamed);
                let done = output_end(&inner) > inner.read_position
                    || !matches!(inner.state, BgState::Running);
                (omitted, fresh, end, done)
            };
            if omitted > 0 {
                on_line(&output_omission_marker(omitted));
            }
            if omitted > 0 || !fresh.is_empty() {
                streamed = end;
            }
            if !fresh.is_empty() {
                emit_stream_chunk(on_line, &fresh);
            }
            if done {
                break;
            }
            tokio::select! {
                () = &mut notified => {}
                () = tokio::time::sleep_until(deadline) => break,
            }
        }
        {
            let inner = proc.inner.lock().unwrap();
            let (omitted, fresh, _) = output_since(&inner, streamed);
            if omitted > 0 || !fresh.is_empty() {
                drop(inner);
                if omitted > 0 {
                    on_line(&output_omission_marker(omitted));
                }
                emit_stream_chunk(on_line, &fresh);
            }
        }
        poll_from(self, id)
    }

    pub fn kill(&self, id: &str) -> Result<String> {
        kill_from(self, id)
    }

    /// Request process-group cancellation and do not report the stop complete
    /// until the child is reaped and its workspace-job callback has run.
    pub async fn kill_and_reap(&self, id: &str) -> Result<String> {
        let result = kill_from(self, id)?;
        let process = lookup(self, id)?;
        let deadline = tokio::time::Instant::now() + QUIESCENT_REAP_TIMEOUT;
        wait_for_terminal_reap(&process, id, deadline).await?;
        Ok(result)
    }

    /// Verify that no tracked process is live and wait for every terminal child
    /// to be fully reaped before its workspace can be switched or removed.
    ///
    /// `kill` records the public `Killed` state before the detached driver has
    /// necessarily drained pipes and reaped the child. Looking only at that
    /// state leaves a small but real last-write race with a workspace rebind.
    pub async fn ensure_quiescent_and_reaped(&self) -> Result<()> {
        if self
            .quiescing
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            bail!("another workspace lifecycle operation is waiting for background processes");
        }
        let result = self.ensure_quiescent_and_reaped_inner().await;
        self.quiescing.store(false, Ordering::Release);
        result
    }

    async fn ensure_quiescent_and_reaped_inner(&self) -> Result<()> {
        // `reserve_slot` acquires `processes` and then observes `quiescing`.
        // Setting the flag before acquiring this lock means either an already
        // in-flight reservation is visible below or every later one is
        // rejected; never a hidden reservation between the two snapshots.
        if self.reserved_slots.load(Ordering::Acquire) != 0 {
            bail!("a background process is still being started");
        }

        let processes = self
            .processes
            .lock()
            .unwrap()
            .iter()
            .map(|(id, process)| (id.clone(), process.clone()))
            .collect::<Vec<_>>();
        let running = processes
            .iter()
            .filter(|(_, process)| process.inner.lock().unwrap().native_running())
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        if !running.is_empty() {
            bail!("background processes remain active: {}", running.join(", "));
        }

        let deadline = tokio::time::Instant::now() + QUIESCENT_REAP_TIMEOUT;
        for (id, process) in processes {
            wait_for_terminal_reap(&process, &id, deadline).await?;
        }

        // New reservations remain blocked until the outer method clears the
        // lifecycle gate, so this final snapshot only needs to account for
        // processes already present in the registry.
        if self.reserved_slots.load(Ordering::Acquire) != 0 {
            bail!("a background process started while waiting for workspace quiescence");
        }
        let running = self
            .processes
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(id, process)| {
                let inner = process.inner.lock().unwrap();
                (matches!(inner.state, BgState::Running) || !inner.reaped).then(|| id.clone())
            })
            .collect::<Vec<_>>();
        if !running.is_empty() {
            bail!(
                "background processes changed while waiting for workspace quiescence: {}",
                running.join(", ")
            );
        }
        Ok(())
    }

    pub fn outcome(&self, id: &str) -> Result<crate::BackgroundOutcome> {
        outcome_from(self, id)
    }

    /// Attribute changes since this process's launch baseline. For terminal
    /// processes the first complete result is cached; subsequent polls report
    /// the same effects even if unrelated workspace changes occur later.
    pub(crate) async fn effects(&self, id: &str) -> Result<crate::ToolEffects> {
        let proc = lookup(self, id)?;
        let Some(baseline) = proc.effect_baseline.clone() else {
            return Ok(crate::ToolEffects::default());
        };
        {
            let inner = proc.inner.lock().unwrap();
            if let Some(cached) = &inner.terminal_effects {
                return cached.clone().map_err(|error| anyhow::anyhow!(error));
            }
        }

        // `bash_kill` marks the public lifecycle state immediately, but exact
        // effects must be captured only after the SIGKILLed process group has
        // closed its pipes and the child has been reaped.
        let reap_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let wait_for_reap = {
                let inner = proc.inner.lock().unwrap();
                !matches!(inner.state, BgState::Running) && !inner.reaped
            };
            if !wait_for_reap {
                break;
            }
            tokio::select! {
                () = proc.reaped.notified() => {},
                () = tokio::time::sleep_until(reap_deadline) => {
                    bail!("timed out waiting to reap background process {id}");
                }
            }
        }

        // A running poll may race the process exit: its snapshot can begin
        // before the command mutates the tree, then finish after the driver has
        // marked the process exited. Remember the lifecycle state *before* the
        // snapshot so that stale running-state observations are never sealed as
        // the terminal effects. The next terminal poll will take a fresh
        // post-reap snapshot.
        let terminal_before_snapshot = {
            let inner = proc.inner.lock().unwrap();
            !matches!(inner.state, BgState::Running) && inner.reaped
        };

        let after =
            match crate::effects::workspace_snapshot(&baseline.root, &baseline.state_root).await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    let message = format!("{error:#}");
                    let mut inner = proc.inner.lock().unwrap();
                    if should_seal_terminal_effects(&inner, terminal_before_snapshot) {
                        inner.terminal_effects = Some(Err(message.clone()));
                    }
                    return Err(anyhow::anyhow!(message));
                }
            };
        let effects = crate::effects::process_effects(&baseline.snapshot, &after);
        let mut inner = proc.inner.lock().unwrap();
        if should_seal_terminal_effects(&inner, terminal_before_snapshot) {
            inner.terminal_effects = Some(Ok(effects.clone()));
        }
        Ok(effects)
    }

    pub fn kill_all(&self) {
        kill_all_from(self)
    }

    /// Stop only auto-backgrounded strays, sparing processes the model started
    /// deliberately with `run_in_background: true`. One-shot runs whose
    /// deliverable *is* a running service (a dev server the caller will use
    /// after `hi` exits) use this instead of [`Self::kill_all`].
    pub fn kill_auto_backgrounded(&self) {
        let reg = self.processes.lock().unwrap();
        for proc in reg.values() {
            if proc.origin != BgOrigin::AutoBackgrounded {
                continue;
            }
            let mut inner = proc.inner.lock().unwrap();
            if inner.native_running() {
                inner.state = BgState::Killed;
                if let Some(pgid) = proc.pgid {
                    crate::tools::kill_group(pgid);
                }
            }
        }
    }

    /// Forget every tracked process without signalling it, so the registry's
    /// `Drop` cannot reap survivors. Pairs with
    /// [`Self::kill_auto_backgrounded`] at one-shot exit.
    pub fn release_all(&self) {
        self.processes.lock().unwrap().clear();
    }

    /// The OS process id (process-group leader) behind a handle, when known.
    /// Lets callers sample live resource usage (e.g. RSS while a model
    /// server loads weights) for progress display.
    pub fn os_pid(&self, id: &str) -> Option<i32> {
        let processes = self.processes.lock().unwrap();
        processes.get(id).and_then(|proc| proc.pgid)
    }

    pub fn ids(&self) -> Vec<String> {
        ids_from(self)
    }

    /// Handles named by callers that were not in the registry, most recent
    /// first, with whether the registry was empty at the time. Lets the agent
    /// distinguish a model-guessed id (never real) from a pruned one (a real
    /// process was forgotten at capacity).
    pub fn unknown_handles(&self) -> Vec<crate::UnknownBackgroundHandle> {
        self.unknown_handles
            .lock()
            .unwrap()
            .iter()
            .rev()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// A non-consuming snapshot of every tracked job: `(id, command, status)`.
    /// Unlike [`poll`](Self::poll), this does not advance the read cursor — it
    /// is for read-only inspection (e.g. a session snapshot shown to the model).
    /// Status is a short label: `running`, `exited <code>`, `killed`, or `failed`.
    pub fn snapshot(&self) -> Vec<(String, String, String)> {
        self.processes
            .lock()
            .unwrap()
            .iter()
            .map(|(id, proc)| {
                let inner = proc.inner.lock().unwrap();
                let status = match inner.state {
                    BgState::Running => "running".to_string(),
                    BgState::Exited(Some(code)) => format!("exited {code}"),
                    BgState::Exited(None) => "exited".to_string(),
                    BgState::Killed => "killed".to_string(),
                    BgState::Failed => "failed".to_string(),
                };
                (id.clone(), proc.command.clone(), status)
            })
            .collect()
    }

    pub fn kill_started_after(&self, before: &[String]) -> usize {
        kill_started_after_from(self, before)
    }

    /// Stop auto-backgrounded processes created after `before` and wait until
    /// their detached drivers have reaped the native children and completed
    /// workspace-job lifecycle callbacks.
    ///
    /// Turn cancellation uses this stronger form before reconciling workspace
    /// bytes. Merely observing the public `Killed` state is insufficient: the
    /// child can still execute a final write until its driver has reaped it.
    pub async fn kill_started_after_and_reap(&self, before: &[String]) -> Result<usize> {
        let before: HashSet<&str> = before.iter().map(String::as_str).collect();
        let targets = {
            let processes = self.processes.lock().unwrap();
            processes
                .iter()
                .filter(|(id, process)| {
                    if before.contains(id.as_str()) || process.origin != BgOrigin::AutoBackgrounded
                    {
                        return false;
                    }
                    let inner = process.inner.lock().unwrap();
                    matches!(inner.state, BgState::Running) || !inner.reaped
                })
                .map(|(id, process)| (id.clone(), Arc::clone(process)))
                .collect::<Vec<_>>()
        };

        let mut signalled = 0;
        for (id, process) in &targets {
            let running = {
                let inner = process.inner.lock().unwrap();
                inner.native_running()
            };
            if running && kill_from(self, id).is_ok() {
                signalled += 1;
            }
        }

        let deadline = tokio::time::Instant::now() + QUIESCENT_REAP_TIMEOUT;
        for (id, process) in targets {
            wait_for_terminal_reap(&process, &id, deadline).await?;
        }
        Ok(signalled)
    }
}

fn emit_stream_chunk(on_line: &mut dyn FnMut(&str), chunk: &str) {
    for piece in chunk.split_inclusive('\n') {
        let line = piece.trim_end_matches(['\n', '\r']);
        if !line.is_empty() {
            on_line(line);
        }
    }
}

fn should_seal_terminal_effects(inner: &BgInner, terminal_before_snapshot: bool) -> bool {
    terminal_before_snapshot && !matches!(inner.state, BgState::Running) && inner.reaped
}

/// The adaptive default-wait budget: 15s on the first empty poll, doubling
/// per consecutive empty poll, capped at 4 minutes. Long enough that waiting
/// costs at most a handful of model rounds per hour instead of one every few
/// seconds; short enough that an Esc/interrupt (checked between tool
/// completions) stays responsive. An explicit `wait_secs` bypasses this;
/// `HI_BG_POLL_WAIT_BASE_SECS` rescales it for standalone callers; embedded
/// callers should prefer [`BackgroundRegistry::set_poll_wait_base_secs`] so
/// timing state stays local to one registry.
fn default_poll_wait_budget(
    empty_polls: u32,
    base_override_secs: Option<u64>,
) -> std::time::Duration {
    const CAP_SECS: u64 = 240;
    let base = base_override_secs
        .or_else(|| {
            std::env::var("HI_BG_POLL_WAIT_BASE_SECS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
        })
        .unwrap_or(15);
    if base == 0 {
        return std::time::Duration::ZERO;
    }
    let secs = base
        .saturating_mul(1u64 << empty_polls.min(6))
        .min(CAP_SECS);
    std::time::Duration::from_secs(secs)
}

/// Return output produced since the last poll, plus a status line. Non-blocking:
/// returns immediately with whatever is buffered.
#[cfg(test)]
pub(crate) fn poll(id: &str) -> Result<String> {
    poll_from(&TEST_REGISTRY, id)
}

fn poll_from(registry: &BackgroundRegistry, id: &str) -> Result<String> {
    let proc = lookup(registry, id)?;
    let mut inner = proc.inner.lock().unwrap();
    let (omitted, retained_fresh, end) = output_since(&inner, inner.read_position);
    inner.read_position = end;
    let delivered = omitted > 0 || !retained_fresh.is_empty();
    let fresh = if omitted > 0 {
        let marker = output_omission_marker(omitted);
        if retained_fresh.is_empty() {
            marker
        } else {
            format!("{marker}\n{retained_fresh}")
        }
    } else {
        retained_fresh
    };
    // Escalation state for the adaptive default wait: consecutive polls that
    // came back empty while running mean the process is quiet, so the next
    // defaulted poll should park longer before reporting "no new output".
    if !delivered && matches!(inner.state, BgState::Running) {
        inner.empty_polls = inner.empty_polls.saturating_add(1);
    } else {
        inner.empty_polls = 0;
    }
    // Status lines name the shell by title so the UI never has to show JSON
    // handle payloads. The model still gets the stable `id=` for tool calls.
    let title = proc.title.as_str();
    let status = match inner.state {
        BgState::Running if !delivered => {
            format!("[{id} · {title}: still running — no new output]")
        }
        BgState::Running => format!("[{id} · {title}: still running]"),
        BgState::Exited(Some(code)) => {
            format!("[{id} · {title}: exited with code {code}]")
        }
        BgState::Exited(None) => format!("[{id} · {title}: exited]"),
        BgState::Killed => format!("[{id} · {title}: stopped]"),
        BgState::Failed => format!("[{id} · {title}: failed]"),
    };
    // Idle running polls must stay a one-line status. Re-echoing the full
    // command on every empty poll makes the UI look like a hung loop,
    // especially for multi-line scripts that were auto-backgrounded.
    Ok(if !delivered {
        match inner.state {
            BgState::Running => status,
            // Terminal and drained: a bare status line here reads as "result
            // missing" and invites a re-poll loop (a live session stalled
            // exactly this way). Restate the tail so the caller can conclude
            // from this reply instead of hunting for the earlier one.
            _ if !inner.output.is_empty() => format!(
                "{status} (`{}`) — all output was already delivered by an earlier poll; \
                 re-polling cannot return more. Tail of that output:\n{}",
                proc.command,
                output_tail(&inner.output)
            ),
            _ => format!(
                "{status} (`{}`) — the process produced no output",
                proc.command
            ),
        }
    } else {
        format!("{status}\n{fresh}")
    })
}

/// Absolute byte position immediately after the retained output.
fn output_end(inner: &BgInner) -> u64 {
    inner
        .dropped_bytes
        .saturating_add(inner.output.len() as u64)
}

/// Return output newer than an absolute stream position, plus the exact number
/// of unread bytes that fell out of the bounded ring before they could be read.
/// All stored positions land on UTF-8 boundaries because appends contain valid
/// strings and front trimming uses [`char_boundary_at_or_after`].
fn output_since(inner: &BgInner, position: u64) -> (u64, String, u64) {
    let end = output_end(inner);
    let omitted = inner.dropped_bytes.saturating_sub(position);
    let retained_position = position.max(inner.dropped_bytes).min(end);
    let offset = retained_position.saturating_sub(inner.dropped_bytes) as usize;
    (omitted, inner.output[offset..].to_string(), end)
}

fn output_omission_marker(omitted: u64) -> String {
    format!(
        "… [background output omitted: {omitted} unread bytes exceeded the {MAX_BG_BUFFER}-byte retention limit] …"
    )
}

/// The last chunk of a finished process's output, for restating on drained
/// polls. Bounded and aligned to a line start so a huge log re-echoes as a
/// readable tail, not a mid-line splice.
fn output_tail(output: &str) -> String {
    const TAIL_BYTES: usize = 2000;
    let trimmed = output.trim_end();
    if trimmed.len() <= TAIL_BYTES {
        return trimmed.to_string();
    }
    let mut start = trimmed.len() - TAIL_BYTES;
    while !trimmed.is_char_boundary(start) {
        start += 1;
    }
    let tail = &trimmed[start..];
    let tail = tail.split_once('\n').map_or(tail, |(_, rest)| rest);
    format!("… (earlier output elided)\n{tail}")
}

/// Kill a background process (whole tree) and mark it killed. Idempotent: a
/// process that already exited reports that instead.
#[cfg(test)]
pub(crate) fn kill(id: &str) -> Result<String> {
    kill_from(&TEST_REGISTRY, id)
}

fn kill_from(registry: &BackgroundRegistry, id: &str) -> Result<String> {
    let proc = lookup(registry, id)?;
    {
        let mut inner = proc.inner.lock().unwrap();
        match inner.state {
            BgState::Running if inner.native_exited => {
                return Ok(format!(
                    "[{id} · {}] already exited; waiting for settlement",
                    proc.title
                ));
            }
            BgState::Exited(_) => {
                return Ok(format!("[{id} · {}] already exited", proc.title));
            }
            BgState::Killed => {
                return Ok(format!("[{id} · {}] already stopped", proc.title));
            }
            BgState::Failed => {
                return Ok(format!("[{id} · {}] already failed", proc.title));
            }
            BgState::Running => inner.state = BgState::Killed,
        }
    }
    if let Some(pgid) = proc.pgid {
        crate::tools::kill_group(pgid);
    }
    proc.changed.notify_waiters();
    hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::JobAfterCancelRequest)?;
    Ok(format!("[{id} · {}] stopped", proc.title))
}

/// Kill every still-running background process. Intended for session shutdown so
/// spawned servers/watchers don't outlive the agent.
fn kill_all_from(registry: &BackgroundRegistry) {
    let reg = registry.processes.lock().unwrap();
    for proc in reg.values() {
        let mut inner = proc.inner.lock().unwrap();
        if inner.native_running() {
            inner.state = BgState::Killed;
            if let Some(pgid) = proc.pgid {
                crate::tools::kill_group(pgid);
            }
        }
    }
}

/// Snapshot known background process ids. Used by frontends before a cancellable
/// turn so they can clean up only processes created by the discarded turn.
#[cfg(test)]
pub(crate) fn outcome(id: &str) -> Result<crate::BackgroundOutcome> {
    outcome_from(&TEST_REGISTRY, id)
}

fn outcome_from(registry: &BackgroundRegistry, id: &str) -> Result<crate::BackgroundOutcome> {
    let proc = lookup(registry, id)?;
    let state = proc.inner.lock().unwrap().state;
    let (state, exit_code) = match state {
        BgState::Running => (crate::BackgroundState::Running, None),
        BgState::Exited(code) => (crate::BackgroundState::Exited, code),
        BgState::Killed => (crate::BackgroundState::Killed, None),
        BgState::Failed => (crate::BackgroundState::Failed, None),
    };
    Ok(crate::BackgroundOutcome {
        id: id.to_string(),
        state,
        exit_code,
    })
}

fn ids_from(registry: &BackgroundRegistry) -> Vec<String> {
    let mut ids: Vec<String> = registry.processes.lock().unwrap().keys().cloned().collect();
    ids.sort_by_key(|id| id_num(id));
    ids
}

/// Kill running **auto-backgrounded** processes started after `before` —
/// foreground commands that outgrew their timeout and were adopted. These are
/// incidental turn state, so turn end / cancel / pre-verification cleanup may
/// reap them. Processes the model deliberately started with
/// `run_in_background: true` are spared: they are long-lived work (downloads,
/// servers) that must survive the turn that started them. They still die with
/// the session (`kill_all` on shutdown) or an explicit `bash_kill`.
/// Returns the number of processes signalled.
fn kill_started_after_from(registry: &BackgroundRegistry, before: &[String]) -> usize {
    let before: HashSet<&str> = before.iter().map(String::as_str).collect();
    let targets: Vec<String> = {
        let reg = registry.processes.lock().unwrap();
        reg.iter()
            .filter(|(id, proc)| {
                !before.contains(id.as_str())
                    && proc.origin == BgOrigin::AutoBackgrounded
                    && proc.inner.lock().unwrap().native_running()
            })
            .map(|(id, _)| id.clone())
            .collect()
    };
    let mut killed = 0;
    for id in targets {
        if kill_from(registry, &id).is_ok() {
            killed += 1;
        }
    }
    killed
}

fn lookup(registry: &BackgroundRegistry, id: &str) -> Result<Arc<BgProc>> {
    let processes = registry.processes.lock().unwrap();
    if let Some(proc) = processes.get(id) {
        return Ok(proc.clone());
    }
    // Remember the miss so the agent can tell a model-guessed id (registry
    // empty — nothing has ever run under it) from a pruned one (a real
    // process was forgotten at capacity). Bounded FIFO.
    let registry_was_empty = processes.is_empty();
    let known: Vec<String> = processes.keys().cloned().collect();
    drop(processes);
    {
        let mut unknown = registry.unknown_handles.lock().unwrap();
        if unknown.len() >= MAX_UNKNOWN_HANDLES {
            unknown.pop_front();
        }
        unknown.push_back(UnknownHandle {
            id: id.to_string(),
            registry_was_empty,
        });
    }
    // A missing handle with an EMPTY registry means the model invented the
    // id (observed on Multi-SWE-bench: `bash_output noop` / `bash_1`
    // guessed in a loop). Say so decisively — "may have been pruned"
    // invites retrying with the next guess.
    if registry_was_empty {
        Err(anyhow::anyhow!(
            "no background process `{id}` — no background processes are running at all. \
             Do not call this again; continue the task with other tools."
        ))
    } else {
        Err(anyhow::anyhow!(
            "no background process `{id}` (it may have been pruned). Running: {}",
            known.join(", ")
        ))
    }
}

async fn wait_for_terminal_reap(
    process: &BgProc,
    id: &str,
    deadline: tokio::time::Instant,
) -> Result<()> {
    loop {
        // Register before inspecting state so `notify_waiters` cannot land in
        // the check-to-await gap and strand teardown until the timeout.
        let notified = process.reaped.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        {
            let inner = process.inner.lock().unwrap();
            if inner.native_running() {
                bail!("background process {id} is still running");
            }
            if inner.reaped {
                return Ok(());
            }
        }
        tokio::time::timeout_at(deadline, notified)
            .await
            .map_err(|_| anyhow::anyhow!("timed out waiting to reap background process {id}"))?;
    }
}

/// Drop already-exited entries oldest-first once the registry is at capacity.
/// Ids end in the monotonic counter (`{slug}_{N}`), so ordering by that
/// number is insertion order.
fn prune(reg: &mut HashMap<String, Arc<BgProc>>) {
    if reg.len() < MAX_BG_PROCS {
        return;
    }
    let mut exited: Vec<(u64, String)> = reg
        .iter()
        .filter(|(_, process)| process.inner.lock().unwrap().reaped)
        .map(|(id, _)| (id_num(id), id.clone()))
        .collect();
    exited.sort_by_key(|(n, _)| *n);
    for (_, id) in exited {
        if reg.len() < MAX_BG_PROCS {
            break;
        }
        reg.remove(&id);
    }
}

fn id_num(id: &str) -> u64 {
    // Ids are `{slug}_{N}` (`cargo-test_3`, legacy `sh_1`/`bg_1`): the
    // insertion counter is always the segment after the last underscore.
    id.rsplit_once('_')
        .and_then(|(_, n)| n.parse().ok())
        .unwrap_or(0)
}

/// Enforce the retained-output cap while preserving an absolute byte origin.
/// The read position is intentionally left untouched: if unread bytes are
/// evicted, the next poll can compute and report the exact missing span.
fn trim_output_to_cap(inner: &mut BgInner) {
    if inner.output.len() > MAX_BG_BUFFER {
        let overflow = inner.output.len() - MAX_BG_BUFFER;
        let cut = char_boundary_at_or_after(&inner.output, overflow);
        inner.output.drain(..cut);
        inner.dropped_bytes = inner.dropped_bytes.saturating_add(cut as u64);
    }
}

/// Smallest valid UTF-8 char boundary at or after `idx` (so `drain(..idx)` is
/// always legal). `str::floor_char_boundary` is still unstable, hence this.
fn char_boundary_at_or_after(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    idx
}

#[cfg(test)]
#[path = "background_tests.rs"]
mod tests;
