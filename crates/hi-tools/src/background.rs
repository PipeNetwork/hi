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

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};
use tokio::io::{AsyncBufReadExt, BufReader};

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

/// Shared state for one background process: the command, its process-group id
/// (for tree-kill), and the mutable buffer/cursor/status the driver task fills.
struct BgProc {
    command: String,
    pgid: Option<i32>,
    effect_baseline: Option<Arc<EffectBaseline>>,
    inner: Mutex<BgInner>,
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
}

/// Workspace/runtime-owned background process registry. Separate registries do
/// not share handles or cleanup, so two agents cannot poll or kill each other's
/// processes.
pub struct BackgroundRegistry {
    processes: Mutex<HashMap<String, Arc<BgProc>>>,
    counter: AtomicU64,
}

impl Default for BackgroundRegistry {
    fn default() -> Self {
        Self {
            processes: Mutex::new(HashMap::new()),
            counter: AtomicU64::new(1),
        }
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

/// Start `command` in the background and return its handle id (e.g. `bg_1`).
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
    /// The child keeps running under a fresh `bg_N` handle, seeded with the
    /// output it produced while in the foreground so a later `bash_output`
    /// shows the whole run. The caller must have defused any process-group kill
    /// guard before handing the child over — this registry now owns its
    /// lifecycle. `pgid` is the child's process-group id for tree-kill.
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
        let id = format!("bg_{}", self.counter.fetch_add(1, Ordering::Relaxed));
        let proc = Arc::new(BgProc {
            command: command.to_string(),
            pgid,
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
            }),
        });
        {
            let mut reg = self.processes.lock().unwrap();
            prune(&mut reg);
            reg.insert(id.clone(), proc.clone());
        }
        tokio::spawn(drive(proc, child, stdout, stderr));
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

        let id = format!("bg_{}", self.counter.fetch_add(1, Ordering::Relaxed));
        let proc = Arc::new(BgProc {
            command: command.to_string(),
            pgid,
            effect_baseline: effect_baseline.map(Arc::new),
            inner: Mutex::new(BgInner {
                output: String::new(),
                read_offset: 0,
                state: BgState::Running,
                reaped: false,
                terminal_effects: None,
            }),
        });

        {
            let mut reg = self.processes.lock().unwrap();
            prune(&mut reg);
            reg.insert(id.clone(), proc.clone());
        }

        // Detached driver: drain both pipes to EOF, then reap and record the status.
        tokio::spawn(drive(proc, child, stdout, stderr));
        Ok(id)
    }

    pub fn poll(&self, id: &str) -> Result<String> {
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
            if tokio::time::Instant::now() >= reap_deadline {
                bail!("timed out waiting to reap background process {id}");
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
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

    pub fn ids(&self) -> Vec<String> {
        ids_from(self)
    }

    pub fn kill_started_after(&self, before: &[String]) -> usize {
        kill_started_after_from(self, before)
    }
}

fn should_seal_terminal_effects(inner: &BgInner, terminal_before_snapshot: bool) -> bool {
    terminal_before_snapshot && !matches!(inner.state, BgState::Running) && inner.reaped
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
    let status = match inner.state {
        BgState::Running if fresh.is_empty() => {
            format!("[{id}: running — no new output]")
        }
        BgState::Running => format!("[{id}: running]"),
        BgState::Exited(Some(code)) => format!("[{id}: exited code {code}]"),
        BgState::Exited(None) => format!("[{id}: exited]"),
        BgState::Killed => format!("[{id}: killed]"),
        BgState::Failed => format!("[{id}: failed]"),
    };
    Ok(if fresh.is_empty() {
        format!("{status} (`{}`)", proc.command)
    } else {
        format!("{status}\n{fresh}")
    })
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
            BgState::Exited(_) => return Ok(format!("[{id}] already exited")),
            BgState::Killed => return Ok(format!("[{id}] already killed")),
            BgState::Failed => return Ok(format!("[{id}] already failed")),
            BgState::Running => inner.state = BgState::Killed,
        }
    }
    if let Some(pgid) = proc.pgid {
        crate::tools::kill_group(pgid);
    }
    Ok(format!("[{id}] killed (`{}`)", proc.command))
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

#[cfg(test)]
fn ids() -> Vec<String> {
    ids_from(&TEST_REGISTRY)
}

/// Kill running background processes that were started after `before`.
/// Returns the number of processes signalled.
fn kill_started_after_from(registry: &BackgroundRegistry, before: &[String]) -> usize {
    let before: HashSet<&str> = before.iter().map(String::as_str).collect();
    let targets: Vec<String> = {
        let reg = registry.processes.lock().unwrap();
        reg.iter()
            .filter(|(id, proc)| {
                !before.contains(id.as_str())
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

#[cfg(test)]
fn kill_started_after(before: &[String]) -> usize {
    kill_started_after_from(&TEST_REGISTRY, before)
}

fn lookup(registry: &BackgroundRegistry, id: &str) -> Result<Arc<BgProc>> {
    registry
        .processes
        .lock()
        .unwrap()
        .get(id)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no background process `{id}` (it may have been pruned)"))
}

/// Drop already-exited entries oldest-first once the registry is at capacity.
/// Ids are monotonic (`bg_N`), so lexual-by-number ordering is insertion order.
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
    id.strip_prefix("bg_")
        .and_then(|n| n.parse().ok())
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
        assert!(combined.contains("exited code 0"), "got: {combined:?}");
        assert_eq!(outcome(&id).unwrap().state, crate::BackgroundState::Exited);
        assert_eq!(outcome(&id).unwrap().exit_code, Some(0));
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
    async fn background_kill_stops_the_process() {
        let _guard = TEST_LOCK.lock().await;
        let id = spawn("sleep 600").unwrap();
        let killed = kill(&id).unwrap();
        assert!(killed.contains("killed"), "got: {killed:?}");
        // After the kill propagates, a poll reports it is no longer running.
        let out = poll_until_done(&id).await;
        assert!(out.contains("killed"), "got: {out:?}");
        // Killing again is idempotent.
        assert!(kill(&id).unwrap().contains("already"), "second kill");
    }

    #[tokio::test]
    async fn kill_started_after_kills_only_new_running_processes() {
        let _guard = TEST_LOCK.lock().await;
        let keep = spawn("sleep 600").unwrap();
        let before = ids();
        let doomed = spawn("sleep 600").unwrap();

        let killed = kill_started_after(&before);

        assert_eq!(killed, 1);
        let doomed_out = poll_until_done(&doomed).await;
        assert!(doomed_out.contains("killed"), "got: {doomed_out:?}");
        let keep_out = poll(&keep).unwrap();
        assert!(
            keep_out.contains("running"),
            "pre-existing background process should survive: {keep_out:?}"
        );
        kill(&keep).unwrap();
    }

    #[tokio::test]
    async fn poll_unknown_id_errors() {
        assert!(poll("bg_does_not_exist").is_err());
        assert!(kill("bg_does_not_exist").is_err());
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
        assert!(done.contains("killed"), "got: {done:?}");
    }

    #[test]
    fn char_boundary_helper_lands_on_boundaries() {
        let s = "a😀b"; // 😀 is 4 bytes at index 1..5
        assert_eq!(char_boundary_at_or_after(s, 2), 5);
        assert_eq!(char_boundary_at_or_after(s, 1), 1);
        assert_eq!(char_boundary_at_or_after(s, 99), s.len());
    }
}
