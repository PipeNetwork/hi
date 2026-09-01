//! Bounded process capture, adoptable children, and process-group cleanup.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::{ProcessOutcome, ToolStatus, TruncationState};

use super::ProcessExecution;

/// Maximum bytes retained from each process stream before the middle is
/// discarded. The reader continues draining after the cap so a noisy child can
/// never deadlock on a full pipe.
const MAX_CAPTURE_BYTES: usize = 2 * 1024 * 1024;
/// Maximum amount of one newline-delimited output record kept in memory while
/// streaming. A child can legally write a multi-megabyte line (or never write
/// a newline at all); `read_until` would grow its temporary buffer without
/// bound before `BoundedBuffer` ever gets a chance to cap the result.
const MAX_STREAM_LINE_BYTES: usize = 64 * 1024;

/// How long the pipe drains may keep reading after the direct child exits.
/// Long enough for buffered output to flush; short enough that a lingering
/// daemon holding the pipes cannot stall the command's result.
///
/// This must fit a legitimate multi-megabyte flush from a fast producer (the
/// reader emits in bounded pseudo-line chunks, redacting and locking per
/// chunk). 250 ms was enough on macOS but raced on faster Linux producers: a
/// 4 MB no-newline `dd` record could exit before the drains finished, leaving
/// the capture under the truncation cap and misreporting a truncated run as
/// `Complete`. 5 s keeps the leaked-descendant guard while comfortably
/// covering real buffered output.
const PIPE_DRAIN_GRACE: Duration = Duration::from_secs(5);

pub(super) async fn capture_child(
    mut child: tokio::process::Child,
    timeout: Duration,
    on_line: &mut (dyn FnMut(&str) + Send),
    started: Instant,
) -> Result<ProcessExecution> {
    let mut group_guard = ProcessGroupDropGuard::for_child(&child);
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

    let callback: &Mutex<&mut (dyn FnMut(&str) + Send)> = &Mutex::new(on_line);
    let stdout_buf = Mutex::new(BoundedBuffer::default());
    let stderr_buf = Mutex::new(BoundedBuffer::default());

    let (status, exit_code) = {
        // Race the reap against the pipe drains. A grandchild that inherited
        // the pipes (`sh -c "server &"`, a test leaking a helper) keeps them
        // open past the child's exit; strictly sequencing reads-then-wait
        // turned that into a full-budget timeout that discarded the real
        // exit status. Once the child exits, the drains get a short grace to
        // flush what's buffered, then the result is built.
        let combined = async {
            let drains = async {
                tokio::join!(
                    read_stream(&mut stdout, callback, &stdout_buf),
                    read_stream(&mut stderr, callback, &stderr_buf),
                );
            };
            let mut drains = std::pin::pin!(drains);
            let mut wait = std::pin::pin!(child.wait());
            tokio::select! {
                exit = &mut wait => {
                    let _ = tokio::time::timeout(PIPE_DRAIN_GRACE, &mut drains).await;
                    exit
                }
                _ = &mut drains => wait.await,
            }
        };
        match tokio::time::timeout(timeout, combined).await {
            Ok(Ok(exit)) if exit.success() => (ToolStatus::Succeeded, exit.code()),
            Ok(Ok(exit)) => (ToolStatus::Failed, exit.code()),
            Ok(Err(err)) => return Err(err).context("waiting for command"),
            Err(_) => {
                // SIGKILL the group, then bound the reap. `Child::kill()` waits
                // for exit; a D-state / wedged descendant would otherwise pin
                // the coding turn past the command deadline.
                kill_process_group(&child);
                let _ = child.start_kill();
                let _ = tokio::time::timeout(PIPE_DRAIN_GRACE, child.wait()).await;
                (ToolStatus::TimedOut, None)
            }
        }
    };
    // On clean completion the guard normally still tree-kills, so a foreground
    // command cannot leak strays (`sleep 600 &`). When the caller declares that
    // detached services are deliverables, that same kill is what murders an
    // intentionally started server the moment its launching shell returns —
    // so it is skipped. Timeout/drop paths always kill the group (above).
    if !matches!(status, ToolStatus::TimedOut) && detached_descendants_preserved() {
        group_guard.defuse();
    }
    drop(group_guard);

    Ok(build_execution(
        stdout_buf.into_inner().unwrap_or_default(),
        stderr_buf.into_inner().unwrap_or_default(),
        status,
        exit_code,
        started,
    ))
}

/// A live child handed back by [`ProcessRunner::run_shell_adoptable`] because it
/// exceeded its foreground budget while still running. The caller adopts it into
/// the background registry (keeping it alive) rather than killing it.
pub struct RunningChild {
    pub child: tokio::process::Child,
    pub stdout: Option<tokio::process::ChildStdout>,
    pub stderr: Option<tokio::process::ChildStderr>,
    pub pgid: Option<i32>,
    /// The combined stdout+stderr produced while in the foreground, to seed the
    /// background handle so a later poll shows the whole run.
    pub partial_output: String,
}

/// Either the command completed within the foreground budget, or it is still
/// running and eligible for adoption into the background.
pub enum AdoptableOutcome {
    Completed(ProcessExecution),
    StillRunning(RunningChild),
}

fn build_execution(
    stdout: BoundedBuffer,
    stderr: BoundedBuffer,
    status: ToolStatus,
    exit_code: Option<i32>,
    started: Instant,
) -> ProcessExecution {
    let stdout_total_bytes = stdout.total_bytes;
    let stderr_total_bytes = stderr.total_bytes;
    let stdout_truncated = stdout.truncated;
    let stderr_truncated = stderr.truncated;
    // The streaming reader redacts each emitted chunk for live UI safety, but
    // a credential can straddle two chunks. Scrub the reconstructed buffers
    // again so direct ProcessRunner/verification callers cannot receive a
    // split-token leak.
    let stdout_text = hi_secrets::redact_secrets(&stdout.into_text()).into_owned();
    let stderr_text = hi_secrets::redact_secrets(&stderr.into_text()).into_owned();
    let stdout_summary = crate::condense::condense(stdout_text.trim_end());
    let stderr_summary = crate::condense::condense(stderr_text.trim_end());
    let original_bytes = stdout_total_bytes.saturating_add(stderr_total_bytes) as u64;
    let retained_bytes = stdout_summary.len().saturating_add(stderr_summary.len()) as u64;
    let truncation = if stdout_truncated || stderr_truncated || retained_bytes < original_bytes {
        TruncationState::Truncated {
            original_bytes,
            retained_bytes,
        }
    } else {
        TruncationState::Complete
    };
    ProcessExecution {
        status,
        outcome: ProcessOutcome {
            exit_code,
            stdout_summary,
            stderr_summary,
            duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        },
        truncation,
    }
}

/// Like [`capture_child`], but on hitting the foreground budget the still-running
/// child is returned for adoption instead of being killed. The process-group
/// kill guard is defused on that path so the child survives the handoff.
pub(super) async fn capture_child_adoptable(
    mut child: tokio::process::Child,
    foreground_budget: Duration,
    on_line: &mut (dyn FnMut(&str) + Send),
    started: Instant,
) -> Result<AdoptableOutcome> {
    let mut group_guard = ProcessGroupDropGuard::for_child(&child);
    let pgid = child.id().map(|pid| pid as i32);
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();

    let callback: &Mutex<&mut (dyn FnMut(&str) + Send)> = &Mutex::new(on_line);
    let stdout_buf = Mutex::new(BoundedBuffer::default());
    let stderr_buf = Mutex::new(BoundedBuffer::default());

    let timed_out = {
        // Same wait-vs-drain race as `capture_child`: an inherited-pipe
        // descendant must not make a finished command look still-running at
        // the foreground budget (which would adopt an already-exited child).
        let combined = async {
            let drains = async {
                tokio::join!(
                    read_stream(&mut stdout, callback, &stdout_buf),
                    read_stream(&mut stderr, callback, &stderr_buf),
                );
            };
            let mut drains = std::pin::pin!(drains);
            let mut wait = std::pin::pin!(child.wait());
            tokio::select! {
                exit = &mut wait => {
                    let _ = tokio::time::timeout(PIPE_DRAIN_GRACE, &mut drains).await;
                    exit
                }
                _ = &mut drains => wait.await,
            }
        };
        match tokio::time::timeout(foreground_budget, combined).await {
            Ok(Ok(exit)) if exit.success() => {
                drop(group_guard);
                return Ok(AdoptableOutcome::Completed(build_execution(
                    stdout_buf.into_inner().unwrap_or_default(),
                    stderr_buf.into_inner().unwrap_or_default(),
                    ToolStatus::Succeeded,
                    exit.code(),
                    started,
                )));
            }
            Ok(Ok(exit)) => {
                drop(group_guard);
                return Ok(AdoptableOutcome::Completed(build_execution(
                    stdout_buf.into_inner().unwrap_or_default(),
                    stderr_buf.into_inner().unwrap_or_default(),
                    ToolStatus::Failed,
                    exit.code(),
                    started,
                )));
            }
            Ok(Err(err)) => return Err(err).context("waiting for command"),
            Err(_) => true,
        }
    };

    debug_assert!(timed_out);
    // Still running at the budget: hand the live child off. Defuse the guard so
    // dropping it here does not kill the group the registry is about to own.
    group_guard.defuse();
    let partial = {
        let stdout =
            hi_secrets::redact_secrets(&stdout_buf.into_inner().unwrap_or_default().into_text())
                .into_owned();
        let stderr =
            hi_secrets::redact_secrets(&stderr_buf.into_inner().unwrap_or_default().into_text())
                .into_owned();
        let mut combined = stdout;
        if !stderr.is_empty() {
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str(&stderr);
        }
        combined
    };
    Ok(AdoptableOutcome::StillRunning(RunningChild {
        child,
        stdout,
        stderr,
        pgid,
        partial_output: partial,
    }))
}

#[derive(Default)]
struct BoundedBuffer {
    /// Retained output before truncation. Once the cap is crossed, this is the
    /// fixed head and is never rebuilt again.
    data: String,
    /// Retained output after truncation. A deque makes retaining the moving tail
    /// O(bytes received), instead of copying the entire capture on every chunk.
    tail: VecDeque<u8>,
    total_bytes: usize,
    truncated: bool,
}

impl BoundedBuffer {
    fn push(&mut self, text: &str) {
        self.total_bytes = self.total_bytes.saturating_add(text.len());
        if !self.truncated && self.data.len().saturating_add(text.len()) <= MAX_CAPTURE_BYTES {
            self.data.push_str(text);
            return;
        }

        if !self.truncated {
            self.data.push_str(text);
            self.truncated = true;
            let head_target = MAX_CAPTURE_BYTES * 3 / 5;
            let tail_target = MAX_CAPTURE_BYTES - head_target;
            let head_end = char_boundary_at_or_before(&self.data, head_target);
            let tail_start =
                char_boundary_at_or_after(&self.data, self.data.len().saturating_sub(tail_target));
            let tail = self.data.as_bytes()[tail_start..].to_vec();
            self.data.truncate(head_end);
            self.tail.extend(tail);
            return;
        }

        let tail_target = MAX_CAPTURE_BYTES - (MAX_CAPTURE_BYTES * 3 / 5);
        self.tail.extend(text.as_bytes());
        while self.tail.len() > tail_target {
            self.tail.pop_front();
        }
    }

    fn into_text(self) -> String {
        if !self.truncated {
            return self.data;
        }
        let tail_bytes: Vec<u8> = self.tail.into_iter().collect();
        let tail = String::from_utf8_lossy(&tail_bytes);
        let mut data = format!(
            "{}\n… [process output middle truncated] …\n{}",
            self.data, tail,
        );
        if data.len() > MAX_CAPTURE_BYTES + 128 {
            let end = char_boundary_at_or_before(&data, MAX_CAPTURE_BYTES + 128);
            data.truncate(end);
        }
        data
    }
}

async fn read_stream<R: tokio::io::AsyncRead + Unpin>(
    pipe: &mut Option<R>,
    on_line: &Mutex<&mut (dyn FnMut(&str) + Send)>,
    buffer: &Mutex<BoundedBuffer>,
) {
    let Some(pipe) = pipe.as_mut() else { return };
    use tokio::io::AsyncReadExt;

    // Read fixed-size chunks and assemble bounded pseudo-lines. Keeping the
    // callback line-oriented preserves live terminal output, while flushing a
    // long line in chunks prevents a single malicious/noisy record from
    // defeating the total capture cap through an intermediate allocation.
    let mut chunk = [0_u8; 8 * 1024];
    let mut line = Vec::with_capacity(MAX_STREAM_LINE_BYTES);
    let emit = |bytes: &[u8]| {
        if bytes.is_empty() {
            return;
        }
        let text = hi_secrets::redact_secrets(&String::from_utf8_lossy(bytes)).into_owned();
        if let Ok(mut callback) = on_line.lock() {
            (*callback)(&text);
        }
        if let Ok(mut buffer) = buffer.lock() {
            buffer.push(&text);
        }
    };

    loop {
        let read = match pipe.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        let mut start = 0;
        while start < read {
            let newline = chunk[start..read]
                .iter()
                .position(|byte| *byte == b'\n')
                .map(|offset| start + offset + 1);
            let end = newline.unwrap_or(read);
            line.extend_from_slice(&chunk[start..end]);

            while line.len() > MAX_STREAM_LINE_BYTES {
                let prefix: Vec<u8> = line.drain(..MAX_STREAM_LINE_BYTES).collect();
                emit(&prefix);
            }
            if newline.is_some() {
                let complete = std::mem::take(&mut line);
                emit(&complete);
            }
            start = end;
        }
    }
    emit(&line);
}

fn char_boundary_at_or_before(text: &str, mut offset: usize) -> usize {
    offset = offset.min(text.len());
    while offset > 0 && !text.is_char_boundary(offset) {
        offset -= 1;
    }
    offset
}

fn char_boundary_at_or_after(text: &str, mut offset: usize) -> usize {
    offset = offset.min(text.len());
    while offset < text.len() && !text.is_char_boundary(offset) {
        offset += 1;
    }
    offset
}

/// Whether a *successfully completed* foreground command may leave detached
/// descendants running (`sh -c "server &"`, `nohup … &`).
///
/// Default false: strays from ordinary commands are leaks. Frontends running a
/// one-shot prompt whose deliverable is a live service (`hi --keep-background`)
/// set this true, because there the detached process is the point.
static PRESERVE_DETACHED_DESCENDANTS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Let successfully completed foreground commands leave detached descendants
/// alive. Process-global and intended to be set once during startup.
pub fn preserve_detached_descendants(preserve: bool) {
    PRESERVE_DETACHED_DESCENDANTS.store(preserve, std::sync::atomic::Ordering::Relaxed);
}

fn detached_descendants_preserved() -> bool {
    PRESERVE_DETACHED_DESCENDANTS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(unix)]
struct ProcessGroupDropGuard {
    pgid: Option<i32>,
}

#[cfg(unix)]
impl ProcessGroupDropGuard {
    fn for_child(child: &tokio::process::Child) -> Self {
        Self {
            pgid: child.id().map(|pid| pid as i32),
        }
    }

    /// Disarm the guard so dropping it does not kill the process group. Used
    /// when the still-running child is handed off (auto-background-on-timeout)
    /// — the new owner is now responsible for its lifecycle.
    fn defuse(&mut self) {
        self.pgid = None;
    }
}

#[cfg(unix)]
impl Drop for ProcessGroupDropGuard {
    fn drop(&mut self) {
        if let Some(pgid) = self.pgid {
            kill_group(pgid);
        }
    }
}

#[cfg(not(unix))]
struct ProcessGroupDropGuard;

#[cfg(not(unix))]
impl ProcessGroupDropGuard {
    fn for_child(_child: &tokio::process::Child) -> Self {
        Self
    }

    fn defuse(&mut self) {}
}

#[cfg(unix)]
pub(crate) fn kill_group(pgid: i32) {
    // SAFETY: a negative pid addresses the process group and has no memory
    // safety implications. A stale group simply returns an OS error.
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
pub(crate) fn kill_group(_pgid: i32) {}

#[cfg(unix)]
pub(super) fn kill_process_group(child: &tokio::process::Child) {
    if let Some(pid) = child.id() {
        kill_group(pid as i32);
    }
}

#[cfg(not(unix))]
pub(super) fn kill_process_group(_child: &tokio::process::Child) {}
