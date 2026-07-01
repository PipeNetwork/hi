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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

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
}

/// Shared state for one background process: the command, its process-group id
/// (for tree-kill), and the mutable buffer/cursor/status the driver task fills.
struct BgProc {
    command: String,
    pgid: Option<i32>,
    inner: Mutex<BgInner>,
}

struct BgInner {
    /// Full retained combined stdout+stderr (front-trimmed past `MAX_BG_BUFFER`).
    output: String,
    /// Byte offset of output already returned by a poll; only newer bytes are
    /// delivered next time.
    read_offset: usize,
    state: BgState,
}

static REGISTRY: LazyLock<Mutex<HashMap<String, Arc<BgProc>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static COUNTER: AtomicU64 = AtomicU64::new(1);
#[cfg(test)]
pub(crate) static TEST_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

/// Start `command` in the background and return its handle id (e.g. `bg_1`).
pub(crate) fn spawn(command: &str) -> Result<String> {
    // Background commands get the same irreversible-op guard as foreground ones.
    if let Some(reason) = crate::guard::catastrophic_op(command) {
        bail!(
            "refused: this command {reason}. It's blocked as irreversible — the per-turn \
             checkpoint can't undo it. Ask the user to run it themselves if it's genuinely \
             needed (or set HI_ALLOW_DANGEROUS=1)."
        );
    }

    let mut child = crate::tools::spawn_shell(command)?;
    let pgid = child.id().map(|p| p as i32);
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let id = format!("bg_{}", COUNTER.fetch_add(1, Ordering::Relaxed));
    let proc = Arc::new(BgProc {
        command: command.to_string(),
        pgid,
        inner: Mutex::new(BgInner {
            output: String::new(),
            read_offset: 0,
            state: BgState::Running,
        }),
    });

    {
        let mut reg = REGISTRY.lock().unwrap();
        prune(&mut reg);
        reg.insert(id.clone(), proc.clone());
    }

    // Detached driver: drain both pipes to EOF, then reap and record the status.
    tokio::spawn(drive(proc, child, stdout, stderr));
    Ok(id)
}

/// Return output produced since the last poll, plus a status line. Non-blocking:
/// returns immediately with whatever is buffered.
pub(crate) fn poll(id: &str) -> Result<String> {
    let proc = lookup(id)?;
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
    };
    Ok(if fresh.is_empty() {
        format!("{status} (`{}`)", proc.command)
    } else {
        format!("{status}\n{fresh}")
    })
}

/// Kill a background process (whole tree) and mark it killed. Idempotent: a
/// process that already exited reports that instead.
pub(crate) fn kill(id: &str) -> Result<String> {
    let proc = lookup(id)?;
    {
        let mut inner = proc.inner.lock().unwrap();
        match inner.state {
            BgState::Exited(_) => return Ok(format!("[{id}] already exited")),
            BgState::Killed => return Ok(format!("[{id}] already killed")),
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
pub fn kill_all() {
    let reg = REGISTRY.lock().unwrap();
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
pub fn ids() -> Vec<String> {
    let mut ids: Vec<String> = REGISTRY.lock().unwrap().keys().cloned().collect();
    ids.sort_by_key(|id| id_num(id));
    ids
}

/// Kill running background processes that were started after `before`.
/// Returns the number of processes signalled.
pub fn kill_started_after(before: &[String]) -> usize {
    let before: HashSet<&str> = before.iter().map(String::as_str).collect();
    let targets: Vec<String> = {
        let reg = REGISTRY.lock().unwrap();
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
        if kill(&id).is_ok() {
            killed += 1;
        }
    }
    killed
}

fn lookup(id: &str) -> Result<Arc<BgProc>> {
    REGISTRY
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
    let code = child.wait().await.ok().and_then(|s| s.code());
    let mut inner = proc.inner.lock().unwrap();
    if inner.state == BgState::Running {
        inner.state = BgState::Exited(code);
    }
}

/// Append every line from one pipe into the shared buffer, enforcing the size
/// cap by front-trimming on a char boundary (and shifting the read cursor).
async fn pump<R: tokio::io::AsyncRead + Unpin>(pipe: Option<R>, proc: &BgProc) {
    let Some(pipe) = pipe else { return };
    let mut reader = BufReader::new(pipe).lines();
    while let Ok(Some(line)) = reader.next_line().await {
        let mut inner = proc.inner.lock().unwrap();
        inner.output.push_str(&line);
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

    #[test]
    fn char_boundary_helper_lands_on_boundaries() {
        let s = "a😀b"; // 😀 is 4 bytes at index 1..5
        assert_eq!(char_boundary_at_or_after(s, 2), 5);
        assert_eq!(char_boundary_at_or_after(s, 1), 1);
        assert_eq!(char_boundary_at_or_after(s, 99), s.len());
    }
}
