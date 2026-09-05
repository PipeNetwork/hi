//! Synchronous Git commands with interruptible input/output and owned children.

use std::io::{Read, Seek, Write};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use tokio_util::sync::CancellationToken;

/// One optional deadline shared by preparation, lock acquisition, and apply.
#[derive(Clone, Copy)]
pub(super) struct Budget {
    started: Instant,
    timeout: Option<Duration>,
}

impl Budget {
    pub(super) fn new(timeout: Option<Duration>) -> Self {
        Self {
            started: Instant::now(),
            timeout,
        }
    }

    pub(super) fn remaining(self) -> Result<Option<Duration>> {
        let remaining = self
            .timeout
            .map(|timeout| timeout.saturating_sub(self.started.elapsed()));
        ensure!(
            remaining.is_none_or(|time| !time.is_zero()),
            "worktree command timed out"
        );
        Ok(remaining)
    }

    pub(super) fn check(self, cancellation: Option<&CancellationToken>) -> Result<Duration> {
        ensure!(
            cancellation.is_none_or(|token| !token.is_cancelled()),
            "worktree command cancelled"
        );
        Ok(self
            .remaining()?
            .unwrap_or(Duration::from_millis(50))
            .min(Duration::from_millis(50)))
    }
}

/// Use anonymous temporary files for process I/O. A large patch cannot block
/// the supervising thread behind stdin backpressure, and a noisy filter cannot
/// deadlock stderr or hold a capture pipe open after its launcher exits.
pub(super) fn run(
    command: &mut Command,
    input: Option<&[u8]>,
    budget: Budget,
    cancellation: Option<&CancellationToken>,
) -> Result<Output> {
    budget.check(cancellation)?;
    let mut stdout = tempfile::tempfile().context("creating worktree stdout capture")?;
    let mut stderr = tempfile::tempfile().context("creating worktree stderr capture")?;
    let stdin = match input {
        Some(input) => {
            let mut file = tempfile::tempfile().context("creating worktree input")?;
            for chunk in input.chunks(64 * 1024) {
                budget.check(cancellation)?;
                file.write_all(chunk).context("writing worktree input")?;
            }
            file.rewind().context("rewinding worktree input")?;
            Stdio::from(file)
        }
        None => Stdio::null(),
    };
    command
        .stdin(stdin)
        .stdout(stdout.try_clone()?)
        .stderr(stderr.try_clone()?);
    super::configure_private_process_group(command);
    budget.check(cancellation)?;
    let mut child = OwnedChild(command.spawn().context("starting worktree command")?);
    let status = wait(&mut child.0, budget, cancellation)?;
    // Cleanup precedes capture reads and success publication, including when
    // a successful filter left a background writer attached to a file handle.
    drop(child);
    budget.check(cancellation)?;
    stdout.rewind()?;
    stderr.rewind()?;
    let mut output = Output {
        status,
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    stdout
        .read_to_end(&mut output.stdout)
        .context("reading worktree stdout")?;
    stderr
        .read_to_end(&mut output.stderr)
        .context("reading worktree stderr")?;
    budget.check(cancellation)?;
    Ok(output)
}

pub(super) fn wait(
    child: &mut Child,
    budget: Budget,
    cancellation: Option<&CancellationToken>,
) -> Result<ExitStatus> {
    let mut backoff = Duration::from_millis(1);
    loop {
        let poll = budget.check(cancellation)?;
        if let Some(status) = child.try_wait().context("waiting for worktree command")? {
            return Ok(status);
        }
        // Most Git metadata commands finish in a few milliseconds. Keep those
        // fast without busy-polling a long-running filter or patch application.
        std::thread::sleep(backoff.min(poll));
        backoff = (backoff * 2).min(Duration::from_millis(50));
    }
}

struct OwnedChild(Child);

impl Drop for OwnedChild {
    fn drop(&mut self) {
        super::terminate_sync_child_group(&mut self.0);
    }
}
