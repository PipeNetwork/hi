use super::*;
use tokio::io::AsyncReadExt;

/// A noisy child must not allocate an unbounded newline-free record.
const MAX_BG_LINE_BYTES: usize = 64 * 1024;

/// Reap the direct child concurrently with output capture. Descendants may
/// inherit its pipes; they must not hide a completed command's exit status.
pub(super) async fn drive(
    proc: Arc<BgProc>,
    mut child: tokio::process::Child,
    stdout: Option<tokio::process::ChildStdout>,
    stderr: Option<tokio::process::ChildStderr>,
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
    let cancelled = {
        let mut inner = proc.inner.lock().unwrap();
        let cancelled = matches!(inner.state, BgState::Killed);
        if (cancelled || exit.is_err() || !crate::process::detached_descendants_preserved())
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
    let raw_state = match exit {
        Ok(status) => BgState::Exited(status.code()),
        Err(_) => BgState::Failed,
    };
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
