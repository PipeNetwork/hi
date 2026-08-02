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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Notify;

/// Cap on retained per-process output. Beyond this we drop the oldest bytes (a
/// ring buffer): a chatty server left unpolled can't grow memory without bound.
const MAX_BG_BUFFER: usize = 256 * 1024;
/// Cap on retained processes. When exceeded, already-exited entries are pruned
/// oldest-first so a long session that starts many servers can't leak handles.
const MAX_BG_PROCS: usize = 64;

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
    /// Handles named by callers that were not in the registry, with whether
    /// the registry was empty at the time. Bounded FIFO so a model that
    /// guesses ids in a loop cannot grow this without bound.
    unknown_handles: Mutex<VecDeque<UnknownHandle>>,
}

/// Cap on remembered unknown handles. Bounded so a guessing loop cannot grow
/// memory; the agent only needs the most recent misses.
const MAX_UNKNOWN_HANDLES: usize = 16;

impl Default for BackgroundRegistry {
    fn default() -> Self {
        Self {
            processes: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(1),
            unknown_handles: Mutex::new(VecDeque::new()),
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
    /// Byte offset of output already returned by a poll; only newer bytes are
    /// delivered next time.
    read_offset: usize,
    state: BgState,
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
    pub fn spawn(&self, runner: &crate::ProcessRunner, command: &str) -> Result<String> {
        self.spawn_with_baseline(runner, command, None)
    }

    pub(crate) fn spawn_tracked(
        &self,
        runner: &crate::ProcessRunner,
        command: &str,
        root: &Path,
        state_root: &Path,
        snapshot: crate::effects::WorkspaceSnapshot,
    ) -> Result<String> {
        self.spawn_with_baseline(
            runner,
            command,
            Some(EffectBaseline {
                root: root.to_path_buf(),
                state_root: state_root.to_path_buf(),
                snapshot,
            }),
        )
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
    pub(crate) fn adopt(
        &self,
        command: &str,
        child: tokio::process::Child,
        stdout: Option<tokio::process::ChildStdout>,
        stderr: Option<tokio::process::ChildStderr>,
        pgid: Option<i32>,
        seed_output: String,
        baseline: (PathBuf, PathBuf, crate::effects::WorkspaceSnapshot),
    ) -> String {
        let (root, state_root, snapshot) = baseline;
        let id = handle_id(command, self.counter.fetch_add(1, Ordering::Relaxed));
        let proc = Arc::new(BgProc {
            command: command.to_string(),
            title: shell_title(command),
            pgid,
            origin: BgOrigin::AutoBackgrounded,
            effect_baseline: Some(Arc::new(EffectBaseline {
                root,
                state_root,
                snapshot,
            })),
            inner: Mutex::new(BgInner {
                output: seed_output,
                read_offset: 0,
                state: BgState::Running,
                reaped: false,
                terminal_effects: None,
                empty_polls: 0,
            }),
            reaped: Notify::new(),
            changed: Notify::new(),
        });
        {
            let mut reg = self.processes.lock().unwrap();
            prune(&mut reg);
            reg.insert(id.clone(), proc.clone());
        }
        // Every child gets its driver immediately — the driver only drains
        // pipes and reaps, which is cheap. Gating drivers behind a permit pool
        // meant the 5th+ concurrent job was never drained: it wedged on a full
        // pipe, reported "still running" forever after exiting, and leaked.
        tokio::spawn(async move {
            drive(proc, child, stdout, stderr).await;
        });
        id
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

        let mut child = runner.spawn_shell(command)?;
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
            inner: Mutex::new(BgInner {
                output: String::new(),
                read_offset: 0,
                state: BgState::Running,
                reaped: false,
                terminal_effects: None,
                empty_polls: 0,
            }),
            reaped: Notify::new(),
            changed: Notify::new(),
        });

        {
            let mut reg = self.processes.lock().unwrap();
            prune(&mut reg);
            reg.insert(id.clone(), proc.clone());
        }

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
        let empty_polls = {
            let proc = lookup(self, id)?;
            let inner = proc.inner.lock().unwrap();
            inner.empty_polls
        };
        self.poll_wait(id, default_poll_wait_budget(empty_polls))
            .await
    }

    /// Like [`poll`](Self::poll), but blocks up to `wait` until the process
    /// produces new output or reaches a terminal state — so one tool call can
    /// cover minutes of waiting instead of a tight model-round poll loop. On
    /// timeout it returns the normal idle status. The wait sleeps on a
    /// notification (no spinning) and holds no locks while parked.
    pub async fn poll_wait(&self, id: &str, wait: std::time::Duration) -> Result<String> {
        let proc = lookup(self, id)?;
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            // Register interest before checking the condition so an append or
            // exit that lands between the check and the await still wakes us.
            let notified = proc.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = proc.inner.lock().unwrap();
                let has_new_output = inner.output.len() > inner.read_offset;
                if has_new_output || !matches!(inner.state, BgState::Running) {
                    break;
                }
            }
            tokio::select! {
                () = &mut notified => {}
                () = tokio::time::sleep_until(deadline) => break,
            }
        }
        poll_from(self, id)
    }

    pub fn kill(&self, id: &str) -> Result<String> {
        kill_from(self, id)
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
            if inner.state == BgState::Running {
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
}

fn should_seal_terminal_effects(inner: &BgInner, terminal_before_snapshot: bool) -> bool {
    terminal_before_snapshot && !matches!(inner.state, BgState::Running) && inner.reaped
}

/// The adaptive default-wait budget: 15s on the first empty poll, doubling
/// per consecutive empty poll, capped at 4 minutes. Long enough that waiting
/// costs at most a handful of model rounds per hour instead of one every few
/// seconds; short enough that an Esc/interrupt (checked between tool
/// completions) stays responsive. An explicit `wait_secs` bypasses this;
/// `HI_BG_POLL_WAIT_BASE_SECS` rescales it (0 restores instant polls — used
/// by tests that exercise the instant-poll steering paths).
fn default_poll_wait_budget(empty_polls: u32) -> std::time::Duration {
    const CAP_SECS: u64 = 240;
    let base = std::env::var("HI_BG_POLL_WAIT_BASE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
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
    let fresh = inner.output[inner.read_offset..].to_string();
    inner.read_offset = inner.output.len();
    // Escalation state for the adaptive default wait: consecutive polls that
    // came back empty while running mean the process is quiet, so the next
    // defaulted poll should park longer before reporting "no new output".
    if fresh.is_empty() && matches!(inner.state, BgState::Running) {
        inner.empty_polls = inner.empty_polls.saturating_add(1);
    } else {
        inner.empty_polls = 0;
    }
    // Status lines name the shell by title so the UI never has to show JSON
    // handle payloads. The model still gets the stable `id=` for tool calls.
    let title = proc.title.as_str();
    let status = match inner.state {
        BgState::Running if fresh.is_empty() => {
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
    Ok(if fresh.is_empty() {
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

/// Short auto-name for a shell command (UI / status lines). Not the full
/// command string — just enough to recognize the job (`cargo test`, `sleep`,
/// `npm run build`). Never includes JSON.
pub fn shell_title(command: &str) -> String {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    if tokens.is_empty() {
        return "shell".into();
    }
    // Skip env assignments (`FOO=bar cmd …`).
    let mut i = 0usize;
    while i < tokens.len() && tokens[i].contains('=') && !tokens[i].starts_with('-') {
        i += 1;
    }
    if i >= tokens.len() {
        return "shell".into();
    }
    let head = tokens[i];
    let base = std::path::Path::new(head)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(head);
    // Keep a small useful phrase: `cargo test`, `npm run build`, `python -m pytest`.
    let mut parts = vec![base.to_string()];
    let mut j = i + 1;
    while j < tokens.len() && parts.len() < 3 {
        let t = tokens[j];
        if t.starts_with('-') && t != "-m" && t != "-c" {
            break;
        }
        // Stop before shell operators / paths that bloat the label.
        if matches!(t, "|" | "||" | "&&" | ";" | ">" | ">>" | "<") {
            break;
        }
        if t.contains('/') || t.contains('\\') {
            break;
        }
        // Skip bare numbers/timeouts (`sleep 600`) — they make status lines look
        // like the full command was re-echoed.
        if t.chars().all(|c| c.is_ascii_digit()) {
            j += 1;
            continue;
        }
        parts.push(t.to_string());
        j += 1;
        // After `run`/`test`/`build` take one more token if short.
        if parts.len() == 2 && matches!(parts[1].as_str(), "run" | "test" | "build" | "exec") {
            continue;
        }
        if parts.len() >= 2
            && !matches!(
                parts[0].as_str(),
                "npm" | "pnpm" | "yarn" | "cargo" | "go" | "python" | "python3" | "pip" | "uv"
            )
        {
            break;
        }
    }
    let title = parts.join(" ");
    const MAX: usize = 40;
    if title.chars().count() <= MAX {
        title
    } else {
        let kept: String = title.chars().take(MAX).collect();
        format!("{kept}…")
    }
}

/// Handle id for a background shell: a command-derived slug plus the
/// registry's monotonic counter (`cargo-test_3`, `git-push_7`). Real names
/// beat an opaque `sh_N`: polls, status lines, and kill calls read as the
/// job they name, and a model can't cold-guess a plausible handle the way
/// it guessed `sh_1` in live runs. The numeric suffix keeps ids unique and
/// preserves insertion order for pruning; the slug is never `task`, so a
/// handle can't collide with agent task ids (`task_N`).
fn handle_id(command: &str, n: u64) -> String {
    let mut slug = String::new();
    let mut prev_dash = true; // suppress a leading dash
    for c in shell_title(command).chars() {
        if slug.len() >= 24 {
            break;
        }
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    let slug = slug.trim_end_matches('-');
    let slug = if slug.is_empty() || slug == "task" {
        "sh"
    } else {
        slug
    };
    format!("{slug}_{n}")
}

fn kill_from(registry: &BackgroundRegistry, id: &str) -> Result<String> {
    let proc = lookup(registry, id)?;
    {
        let mut inner = proc.inner.lock().unwrap();
        match inner.state {
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
    Ok(format!("[{id} · {}] stopped", proc.title))
}

/// Kill every still-running background process. Intended for session shutdown so
/// spawned servers/watchers don't outlive the agent.
fn kill_all_from(registry: &BackgroundRegistry) {
    let reg = registry.processes.lock().unwrap();
    for proc in reg.values() {
        let mut inner = proc.inner.lock().unwrap();
        if inner.state == BgState::Running {
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
                    && matches!(proc.inner.lock().unwrap().state, BgState::Running)
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

/// Drop already-exited entries oldest-first once the registry is at capacity.
/// Ids end in the monotonic counter (`{slug}_{N}`), so ordering by that
/// number is insertion order.
fn prune(reg: &mut HashMap<String, Arc<BgProc>>) {
    if reg.len() < MAX_BG_PROCS {
        return;
    }
    let mut exited: Vec<(u64, String)> = reg
        .iter()
        .filter(|(_, p)| !matches!(p.inner.lock().unwrap().state, BgState::Running))
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

/// Drive one process to completion: pump both pipes into the shared buffer, then
/// reap. A kill recorded mid-flight is preserved (not clobbered by the status).
async fn drive(
    proc: Arc<BgProc>,
    mut child: tokio::process::Child,
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
) {
    tokio::join!(pump(stdout, &proc), pump(stderr, &proc));
    let state = match child.wait().await {
        Ok(status) => BgState::Exited(status.code()),
        Err(_) => BgState::Failed,
    };
    let mut inner = proc.inner.lock().unwrap();
    if inner.state == BgState::Running {
        inner.state = state;
    }
    inner.reaped = true;
    drop(inner);
    proc.reaped.notify_waiters();
    proc.changed.notify_waiters();
}

/// Append every line from one pipe into the shared buffer, enforcing the size
/// cap by front-trimming on a char boundary (and shifting the read cursor).
async fn pump<R: tokio::io::AsyncRead + Unpin>(pipe: Option<R>, proc: &BgProc) {
    let Some(pipe) = pipe else { return };
    // Read raw bytes and lossy-decode per line: `next_line()` errors on the
    // first invalid-UTF-8 byte, which would stop draining the pipe — output
    // after that point would be lost, and a child still writing would block on
    // a full pipe buffer.
    let mut reader = BufReader::new(pipe);
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        match reader.read_until(b'\n', &mut bytes).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&bytes);
        let mut inner = proc.inner.lock().unwrap();
        inner.output.push_str(line.trim_end_matches(['\r', '\n']));
        inner.output.push('\n');
        if inner.output.len() > MAX_BG_BUFFER {
            let overflow = inner.output.len() - MAX_BG_BUFFER;
            let cut = char_boundary_at_or_after(&inner.output, overflow);
            inner.output.drain(..cut);
            inner.read_offset = inner.read_offset.saturating_sub(cut);
        }
        drop(inner);
        proc.changed.notify_waiters();
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
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn running_effect_snapshot_is_not_sealed_when_process_exits_during_scan() {
        let inner = BgInner {
            output: String::new(),
            read_offset: 0,
            state: BgState::Exited(Some(0)),
            reaped: true,
            terminal_effects: None,
            empty_polls: 0,
        };
        assert!(should_seal_terminal_effects(&inner, true));
        assert!(
            !should_seal_terminal_effects(&inner, false),
            "a snapshot begun while running must be recomputed after reap"
        );
    }

    /// Poll until the process reports it is no longer running, or time out.
    async fn poll_until_done(id: &str) -> String {
        for _ in 0..200 {
            let out = poll(id).unwrap();
            if !out.contains("running") {
                return out;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("background process {id} never finished");
    }

    #[tokio::test]
    async fn background_captures_output_and_exit_code() {
        let _guard = TEST_LOCK.lock().await;
        let id = spawn("echo hi-bg").unwrap();
        let combined = poll_until_done(&id).await;
        // `poll_until_done` returns the poll that observed the exit; the echoed
        // line should be in that same drain (output is flushed before exit).
        assert!(
            combined.contains("hi-bg") || poll(&id).unwrap().contains("hi-bg"),
            "expected output, got: {combined:?}"
        );
        assert!(combined.contains("exited with code 0"), "got: {combined:?}");
        assert_eq!(outcome(&id).unwrap().state, crate::BackgroundState::Exited);
        assert_eq!(outcome(&id).unwrap().exit_code, Some(0));
    }

    #[tokio::test]
    async fn drained_terminal_poll_restates_output_tail() {
        let _guard = TEST_LOCK.lock().await;
        let id = spawn("echo tail-marker").unwrap();
        let first = poll_until_done(&id).await;
        // Drain any straggling flush so the next poll is genuinely empty.
        if !first.contains("tail-marker") {
            poll(&id).unwrap();
        }
        let drained = poll(&id).unwrap();
        assert!(drained.contains("exited with code 0"), "got: {drained:?}");
        assert!(
            drained.contains("already delivered") && drained.contains("tail-marker"),
            "a drained terminal poll must restate the output tail so the \
             caller can conclude without re-polling: {drained:?}"
        );
    }

    #[tokio::test]
    async fn drained_terminal_poll_of_silent_process_says_so() {
        let _guard = TEST_LOCK.lock().await;
        let id = spawn("true").unwrap();
        poll_until_done(&id).await;
        let drained = poll(&id).unwrap();
        assert!(drained.contains("produced no output"), "got: {drained:?}");
    }

    #[test]
    fn output_tail_bounds_and_aligns_to_line_start() {
        assert_eq!(output_tail("short\n"), "short");
        let long = format!("{}\nlast line", "x".repeat(5000));
        let tail = output_tail(&long);
        assert!(tail.starts_with("… (earlier output elided)\n"));
        assert!(tail.ends_with("last line"));
        assert!(tail.len() < 2100, "tail stays bounded: {}", tail.len());
    }

    #[test]
    fn handle_ids_name_the_job_and_order_by_suffix() {
        // Real names, not opaque `sh_N`: a live run showed the model
        // cold-guessing `sh_1` with nothing running; a slug it cannot
        // predict is not guessable, and polls/kills read as the job.
        assert_eq!(
            handle_id("cargo test --quiet -p hi-tools", 3),
            "cargo-test_3"
        );
        assert_eq!(
            handle_id("RUST_LOG=debug git push origin main", 7),
            "git-push_7"
        );
        assert_eq!(handle_id("sleep 600", 1), "sleep_1");
        // Unparseable → shell_title's generic label, never empty.
        assert_eq!(handle_id("", 2), "shell_2");
        assert_eq!(handle_id("---", 5), "sh_5");
        // A bare `task` command must not mint ids in the task_ namespace.
        assert_eq!(handle_id("task", 4), "sh_4");
        // The slug is bounded and the counter still parses for prune order.
        let long = handle_id("extraordinarily-long-command-name-beyond-any-cap xyz", 12);
        assert!(long.len() <= 28, "bounded: {long}");
        assert_eq!(id_num(&long), 12);
        assert_eq!(id_num("cargo-test_9"), 9);
        assert_eq!(id_num("sh_1"), 1);
        assert_eq!(id_num("bg_5"), 5);
    }

    #[tokio::test]
    async fn background_returns_immediately_for_long_process() {
        let _guard = TEST_LOCK.lock().await;
        // A 600s sleep must not block spawn; it returns an id at once.
        let id = tokio::time::timeout(Duration::from_secs(2), async { spawn("sleep 600") })
            .await
            .expect("spawn must not block")
            .unwrap();
        let out = poll(&id).unwrap();
        assert!(out.contains("running"), "got: {out:?}");
        assert_eq!(outcome(&id).unwrap().state, crate::BackgroundState::Running);
        kill(&id).unwrap();
    }

    #[tokio::test]
    async fn idle_running_poll_does_not_re_echo_command() {
        let _guard = TEST_LOCK.lock().await;
        let id = spawn("sleep 600").unwrap();
        let out = poll(&id).unwrap();
        assert!(
            out.contains("still running — no new output"),
            "idle poll status: {out:?}"
        );
        assert!(
            !out.contains("`sleep 600`") && !out.contains("sleep 600"),
            "idle running polls must not re-echo the full command (looks like a hung UI loop): {out:?}"
        );
        // Auto-name may include the program (`sleep`) — that is the title, not a dump.
        kill(&id).unwrap();
    }

    #[tokio::test]
    async fn snapshot_lists_each_job_with_command_and_status() {
        let _guard = TEST_LOCK.lock().await;
        let registry = BackgroundRegistry::default();
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        let id = registry.spawn(&runner, "sleep 600").unwrap();

        let snap = registry.snapshot();
        let entry = snap.iter().find(|(eid, _, _)| *eid == id);
        assert!(
            entry.is_some(),
            "snapshot includes the spawned job: {snap:?}"
        );
        let (_, command, status) = entry.unwrap();
        assert_eq!(command, "sleep 600");
        assert_eq!(status, "running");

        // Snapshot is non-consuming: polling afterwards still returns fresh output.
        registry.kill(&id).unwrap();
        let snap_after = registry.snapshot();
        let (_, _, status_after) = snap_after.iter().find(|(eid, _, _)| *eid == id).unwrap();
        assert_eq!(status_after, "killed");
    }

    #[tokio::test]
    async fn background_kill_stops_the_process() {
        let _guard = TEST_LOCK.lock().await;
        let id = spawn("sleep 600").unwrap();
        let killed = kill(&id).unwrap();
        assert!(killed.contains("stopped"), "got: {killed:?}");
        // After the kill propagates, a poll reports it is no longer running.
        let out = poll_until_done(&id).await;
        assert!(out.contains("stopped"), "got: {out:?}");
        // Killing again is idempotent.
        assert!(kill(&id).unwrap().contains("already"), "second kill");
    }

    #[tokio::test]
    async fn kill_started_after_reaps_auto_backgrounded_but_spares_deliberate_jobs() {
        let _guard = TEST_LOCK.lock().await;
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        let registry = BackgroundRegistry::default();
        let before = registry.ids();
        // Deliberate `run_in_background` work started after the baseline —
        // an 800 GB download must not die because its turn ended.
        let download = registry.spawn(&runner, "sleep 600").unwrap();
        // Auto-backgrounded: a foreground command adopted after outgrowing
        // its budget — incidental turn state, still reaped.
        let mut child = runner.spawn_shell("sleep 600").unwrap();
        let pgid = child.id().map(|p| p as i32);
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let root = std::env::current_dir().unwrap();
        let state = std::env::temp_dir().join(format!("hi-origin-state-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&state);
        let snapshot = crate::effects::workspace_snapshot(&root, &state)
            .await
            .unwrap();
        let adopted = registry.adopt(
            "sleep 600",
            child,
            stdout,
            stderr,
            pgid,
            String::new(),
            (root, state, snapshot),
        );

        let killed = registry.kill_started_after(&before);

        assert_eq!(killed, 1, "only the auto-backgrounded process is reaped");
        assert_eq!(
            registry.outcome(&download).unwrap().state,
            crate::BackgroundState::Running,
            "a deliberate run_in_background job survives turn-scoped cleanup"
        );
        assert_eq!(
            registry.outcome(&adopted).unwrap().state,
            crate::BackgroundState::Killed
        );
        registry.kill_all();
    }

    #[tokio::test]
    async fn poll_unknown_id_errors() {
        assert!(poll("sh_does_not_exist").is_err());
        assert!(kill("sh_does_not_exist").is_err());
    }

    #[tokio::test]
    async fn unknown_handles_are_recorded_with_registry_emptiness() {
        let registry = BackgroundRegistry::default();
        // Empty registry: the id cannot have been pruned — it was guessed.
        assert!(registry.poll("ghost_1").is_err());
        let unknown = registry.unknown_handles();
        assert_eq!(unknown.len(), 1);
        assert_eq!(unknown[0].id, "ghost_1");
        assert!(unknown[0].registry_was_empty);

        // A real process makes later misses ambiguous (possibly pruned).
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        let id = registry.spawn(&runner, "sleep 600").unwrap();
        assert!(registry.poll("ghost_2").is_err());
        let unknown = registry.unknown_handles();
        assert_eq!(unknown.len(), 2);
        assert_eq!(unknown[0].id, "ghost_2");
        assert!(!unknown[0].registry_was_empty);
        assert_eq!(unknown[1].id, "ghost_1");
        assert!(unknown[1].registry_was_empty);
        registry.kill(&id).unwrap();
    }

    #[tokio::test]
    async fn unknown_handle_log_is_bounded() {
        let registry = BackgroundRegistry::default();
        for n in 0..(MAX_UNKNOWN_HANDLES + 5) {
            assert!(registry.poll(&format!("ghost_{n}")).is_err());
        }
        let unknown = registry.unknown_handles();
        assert_eq!(unknown.len(), MAX_UNKNOWN_HANDLES);
        // Oldest misses are dropped first; the most recent is kept.
        assert_eq!(unknown[0].id, format!("ghost_{}", MAX_UNKNOWN_HANDLES + 4));
        assert_eq!(
            unknown[MAX_UNKNOWN_HANDLES - 1].id,
            format!("ghost_{}", 5)
        );
    }

    #[test]
    fn default_wait_budget_escalates_and_caps() {
        // SAFETY: single-threaded test scope; the var is read per call.
        unsafe { std::env::remove_var("HI_BG_POLL_WAIT_BASE_SECS") };
        assert_eq!(default_poll_wait_budget(0), Duration::from_secs(15));
        assert_eq!(default_poll_wait_budget(1), Duration::from_secs(30));
        assert_eq!(default_poll_wait_budget(2), Duration::from_secs(60));
        assert_eq!(default_poll_wait_budget(4), Duration::from_secs(240));
        assert_eq!(
            default_poll_wait_budget(63),
            Duration::from_secs(240),
            "cap holds for arbitrary streaks"
        );
        unsafe { std::env::set_var("HI_BG_POLL_WAIT_BASE_SECS", "0") };
        assert_eq!(default_poll_wait_budget(3), Duration::ZERO, "0 = instant");
        unsafe { std::env::remove_var("HI_BG_POLL_WAIT_BASE_SECS") };
    }

    #[tokio::test]
    async fn default_poll_parks_on_the_watcher_until_output() {
        let _guard = TEST_LOCK.lock().await;
        let registry = BackgroundRegistry::default();
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        // Quiet for 400ms, then emits: the defaulted poll must park on the
        // change notification and wake with the output — not return an
        // instant "no new output" that costs the caller a round-trip.
        let id = registry
            .spawn(&runner, "sleep 0.4; echo woke-the-watcher; sleep 600")
            .unwrap();

        let started = std::time::Instant::now();
        let out = registry.poll_wait_default(&id).await.unwrap();

        assert!(
            out.contains("woke-the-watcher"),
            "default poll returns the awaited output: {out:?}"
        );
        assert!(
            started.elapsed() >= Duration::from_millis(300),
            "must actually have parked: {:?}",
            started.elapsed()
        );
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "must wake on output, not sleep out the budget: {:?}",
            started.elapsed()
        );
        registry.kill(&id).unwrap();
    }

    #[tokio::test]
    async fn empty_polls_escalate_and_fresh_output_resets() {
        let _guard = TEST_LOCK.lock().await;
        let registry = BackgroundRegistry::default();
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        let id = registry.spawn(&runner, "echo first; sleep 600").unwrap();
        let strikes = |registry: &BackgroundRegistry, id: &str| {
            let processes = registry.processes.lock().unwrap();
            let proc = processes.get(id).unwrap();
            let inner = proc.inner.lock().unwrap();
            inner.empty_polls
        };

        // Wait for the first line, then drain it: counter resets on output.
        let drained = registry
            .poll_wait(&id, Duration::from_secs(5))
            .await
            .unwrap();
        assert!(drained.contains("first"), "got: {drained:?}");
        assert_eq!(strikes(&registry, &id), 0);

        // Two instant empty peeks escalate the streak.
        let _ = registry.poll(&id).unwrap();
        let _ = registry.poll(&id).unwrap();
        assert_eq!(strikes(&registry, &id), 2);
        registry.kill(&id).unwrap();
    }

    #[tokio::test]
    async fn poll_wait_blocks_until_new_output_arrives() {
        let _guard = TEST_LOCK.lock().await;
        let registry = BackgroundRegistry::default();
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        let id = registry
            .spawn(&runner, "sleep 0.4; echo late-line; sleep 600")
            .unwrap();

        let started = std::time::Instant::now();
        let out = registry
            .poll_wait(&id, Duration::from_secs(10))
            .await
            .unwrap();

        assert!(
            out.contains("late-line"),
            "the wait should return the fresh output: {out:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "must wake on output, not sleep out the full wait: {:?}",
            started.elapsed()
        );
        registry.kill(&id).unwrap();
    }

    #[tokio::test]
    async fn poll_wait_times_out_to_idle_status_on_a_quiet_process() {
        let _guard = TEST_LOCK.lock().await;
        let registry = BackgroundRegistry::default();
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        let id = registry.spawn(&runner, "sleep 600").unwrap();

        let started = std::time::Instant::now();
        let out = registry
            .poll_wait(&id, Duration::from_millis(200))
            .await
            .unwrap();

        assert!(
            out.contains("still running — no new output"),
            "a timed-out wait reports genuine idleness: {out:?}"
        );
        assert!(started.elapsed() >= Duration::from_millis(180));
        registry.kill(&id).unwrap();
    }

    #[tokio::test]
    async fn poll_wait_wakes_promptly_when_the_process_is_killed() {
        let _guard = TEST_LOCK.lock().await;
        let registry = std::sync::Arc::new(BackgroundRegistry::default());
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        let id = registry.spawn(&runner, "sleep 600").unwrap();

        let waiter = {
            let registry = registry.clone();
            let id = id.clone();
            tokio::spawn(async move { registry.poll_wait(&id, Duration::from_secs(30)).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        registry.kill(&id).unwrap();

        let out = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("kill must wake the waiter")
            .unwrap()
            .unwrap();
        assert!(out.contains("stopped"), "got: {out:?}");
    }

    #[tokio::test]
    async fn adopt_keeps_child_running_and_seeds_output() {
        let _guard = TEST_LOCK.lock().await;
        let runner = crate::ProcessRunner::from_current_dir().unwrap();
        // Simulate the auto-background handoff: spawn a still-running child and
        // adopt it with a seed capturing the "foreground" output so far.
        let mut child = runner.spawn_shell("sleep 600").unwrap();
        let pgid = child.id().map(|p| p as i32);
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let root = std::env::current_dir().unwrap();
        let state = std::env::temp_dir().join(format!("hi-adopt-state-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&state);
        let snapshot = crate::effects::workspace_snapshot(&root, &state)
            .await
            .unwrap();
        let id = TEST_REGISTRY.adopt(
            "sleep 600",
            child,
            stdout,
            stderr,
            pgid,
            "already-printed\n".to_string(),
            (root, state.clone(), snapshot),
        );

        let polled = poll(&id).unwrap();
        assert!(polled.contains("running"), "adopted child runs: {polled:?}");
        assert!(
            polled.contains("already-printed"),
            "seed output is visible on first poll: {polled:?}"
        );
        assert_eq!(outcome(&id).unwrap().state, crate::BackgroundState::Running);
        kill(&id).unwrap();
        let done = poll_until_done(&id).await;
        assert!(done.contains("stopped"), "got: {done:?}");
    }

    #[test]
    fn char_boundary_helper_lands_on_boundaries() {
        let s = "a😀b"; // 😀 is 4 bytes at index 1..5
        assert_eq!(char_boundary_at_or_after(s, 2), 5);
        assert_eq!(char_boundary_at_or_after(s, 1), 1);
        assert_eq!(char_boundary_at_or_after(s, 99), s.len());
    }
}
