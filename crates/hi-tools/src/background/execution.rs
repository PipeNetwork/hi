use super::*;
use tokio::io::AsyncReadExt;

/// A noisy child must not allocate an unbounded newline-free record.
const MAX_BG_LINE_BYTES: usize = 64 * 1024;

/// Reap the direct child concurrently with output capture. Descendants may
/// inherit its pipes; they must not hide a completed command's exit status.
pub(super) fn drive(
    proc: Arc<BgProc>,
    child: tokio::process::Child,
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
) -> impl std::future::Future<Output = ()> + Send + 'static {
    // Construct the guard before returning the future. Even if Tokio drops the
    // task before its first poll, the captured guard retains cancellation
    // safety for a child whose own kill_on_drop is intentionally disabled.
    let ownership = DriverOwnershipGuard(Arc::clone(&proc));
    drive_owned(proc, child, stdout, stderr, ownership)
}

async fn drive_owned(
    proc: Arc<BgProc>,
    mut child: tokio::process::Child,
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
    _ownership: DriverOwnershipGuard,
) {
    let mut stdout_pending = Vec::new();
    let mut stderr_pending = Vec::new();
    let exit = {
        let drains = async {
            tokio::join!(
                pump(stdout, &proc, &mut stdout_pending),
                pump(stderr, &proc, &mut stderr_pending)
            );
        };
        let mut drains = std::pin::pin!(drains);
        let mut wait = std::pin::pin!(child.wait());
        tokio::select! {
            exit = &mut wait => {
                let _ = tokio::time::timeout(crate::process::PIPE_DRAIN_GRACE, &mut drains).await;
                exit
            }
            _ = &mut drains => wait.await,
        }
    };
    // A bounded drain can interrupt an unterminated diagnostic. Keep pending
    // bytes outside the cancelled pumps so settlement still publishes it.
    append_output(&proc, &stdout_pending);
    append_output(&proc, &stderr_pending);

    // Under --keep-background, an explicitly requested shell is only the
    // launcher: its process group can still contain the service the user asked
    // Hi to preserve. Keep the durable job live until that whole group exits
    // or the shutdown release transfers ownership. Publishing Succeeded here
    // would let a live writer escape both settlement and the Orphaned record.
    let preserve_requested_group =
        proc.origin == BgOrigin::Requested && crate::process::detached_descendants_preserved();
    if exit.is_ok() && preserve_requested_group && proc.pgid.is_some_and(process_group_is_alive) {
        wait_for_owned_process_group(&proc).await;
    }

    let cancelled = {
        let mut inner = proc.inner.lock().unwrap();
        let cancelled = matches!(inner.state, BgState::Killed);
        if (cancelled || exit.is_err() || !preserve_requested_group)
            && let Some(pgid) = proc.pgid
        {
            crate::process::kill_group(pgid);
        }
        // Serialize this boundary with every cancellation path. Keep public
        // success pending until observe() completes, but never claim a later
        // kill cancelled work whose native execution is already finished.
        inner.native_exited = true;
        cancelled
    };
    // Serialize the real terminal callback with `--keep-background` release.
    // `native_exited` is written first so release sees an already-observed
    // native exit even while this callback is waiting for the async gate. If
    // release guardedly observes a live process first, that is the intentional
    // handoff linearization point and its Orphaned publication wins.
    let _terminal_publication = proc.terminal_publication.lock().await;
    let raw_state = match exit {
        Ok(status) => BgState::Exited(status.code()),
        Err(_) => BgState::Failed,
    };
    if proc.ownership_released.load(Ordering::Acquire) {
        // Release already durably published Orphaned. The retained driver may
        // reap the native child later, but it must not publish a second,
        // contradictory terminal transition for a job the harness no longer
        // owns.
        let mut inner = proc.inner.lock().unwrap();
        inner.state = raw_state;
        inner.reaped = true;
        drop(inner);
        proc.reaped.notify_waiters();
        proc.changed.notify_waiters();
        return;
    }
    let failpoint_error = (!cancelled)
        .then(|| {
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::JobAfterNaturalExit)
                .err()
        })
        .flatten();
    let terminal = if cancelled {
        crate::BackgroundJobTerminal::Cancelled
    } else if failpoint_error.is_some() {
        crate::BackgroundJobTerminal::Failed
    } else {
        match raw_state {
            BgState::Exited(Some(0)) => crate::BackgroundJobTerminal::Succeeded,
            _ => crate::BackgroundJobTerminal::Failed,
        }
    };
    let detail = failpoint_error.as_ref().map(ToString::to_string);
    let lifecycle_error = match &proc.managed_job {
        Some(job) => job.observe(terminal, detail).await.err(),
        None => None,
    };
    let mut inner = proc.inner.lock().unwrap();
    inner.state = if cancelled {
        BgState::Killed
    } else if lifecycle_error.is_some() || failpoint_error.is_some() {
        BgState::Failed
    } else {
        raw_state
    };
    if let Some(error) = lifecycle_error {
        inner
            .output
            .push_str(&format!("workspace job settlement failed: {error}\n"));
        trim_output_to_cap(&mut inner);
    }
    inner.reaped = true;
    drop(inner);
    proc.reaped.notify_waiters();
    proc.changed.notify_waiters();
}

struct DriverOwnershipGuard(Arc<BgProc>);

impl Drop for DriverOwnershipGuard {
    fn drop(&mut self) {
        self.0.kill_native_if_owned();
    }
}

async fn wait_for_owned_process_group(proc: &BgProc) {
    let Some(pgid) = proc.pgid else { return };
    while !proc.ownership_released.load(Ordering::Acquire) && process_group_is_alive(pgid) {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

#[cfg(unix)]
fn process_group_is_alive(pgid: i32) -> bool {
    if unsafe { libc::kill(-pgid, 0) } == 0 {
        return true;
    }
    // A group we own should be signalable, but permission ambiguity must not
    // publish a false terminal state for a potentially live writer.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn process_group_is_alive(_pgid: i32) -> bool {
    false
}

pub(super) async fn stop_and_reap(mut child: tokio::process::Child, pgid: Option<i32>) {
    if let Some(pgid) = pgid {
        crate::tools::kill_group(pgid);
    }
    let _ = child.start_kill();
    let _ = child.wait().await;
}

/// Append every line from one pipe into the shared buffer, enforcing the size
/// cap by front-trimming on a char boundary (and shifting the read cursor).
async fn pump<R: tokio::io::AsyncRead + Unpin>(pipe: Option<R>, proc: &BgProc, line: &mut Vec<u8>) {
    let Some(pipe) = pipe else { return };
    // Read fixed-size chunks and assemble bounded pseudo-lines. A noisy child
    // must keep draining even when it never emits a newline.
    let mut reader = pipe;
    let mut chunk = [0_u8; 8 * 1024];
    loop {
        let read = match reader.read(&mut chunk).await {
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
            while line.len() > MAX_BG_LINE_BYTES {
                let prefix: Vec<u8> = line.drain(..MAX_BG_LINE_BYTES).collect();
                append_output(proc, &prefix);
            }
            if newline.is_some() {
                let complete = std::mem::take(line);
                append_output(proc, &complete);
            }
            start = end;
        }
    }
    append_output(proc, line);
    line.clear();
}

fn append_output(proc: &BgProc, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let line = String::from_utf8_lossy(bytes);
    let mut inner = proc.inner.lock().unwrap();
    inner.output.push_str(line.trim_end_matches(['\r', '\n']));
    inner.output.push('\n');
    trim_output_to_cap(&mut inner);
    drop(inner);
    proc.changed.notify_waiters();
}
