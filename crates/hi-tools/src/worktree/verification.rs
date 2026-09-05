use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};

use super::{PYCACHE_EXCLUDES, configure_private_process_group, terminate_sync_child_group};

/// Ground-truth check: run the verify command inside the worktree.
pub fn verify_passes(worktree: &Path, verify: &str) -> bool {
    verify_passes_with_timeout_and_cancel(worktree, verify, crate::check_timeout(), None)
}

/// Async owner for a potentially unbounded verification process.
///
/// The blocking worker polls an operation-local cancellation token. Its drop
/// guard lives in this future, so dropping/cancelling the caller kills and
/// reaps the verifier's complete process group instead of detaching a
/// `spawn_blocking` worker that can run forever.
pub async fn verify_passes_async(
    worktree: &Path,
    verify: &str,
    parent_cancel: Option<&tokio_util::sync::CancellationToken>,
) -> bool {
    let operation_cancel = parent_cancel
        .map(tokio_util::sync::CancellationToken::child_token)
        .unwrap_or_default();
    let drop_guard = operation_cancel.clone().drop_guard();
    let worker_cancel = operation_cancel.clone();
    let worktree = worktree.to_path_buf();
    let verify = verify.to_string();
    let result = tokio::task::spawn_blocking(move || {
        verify_passes_with_timeout_and_cancel(
            &worktree,
            &verify,
            crate::check_timeout(),
            Some(&worker_cancel),
        )
    })
    .await
    .unwrap_or(false);
    let _ = drop_guard.disarm();
    result
}

pub(super) fn verification_timed_out(timeout: Option<Duration>, elapsed: Duration) -> bool {
    timeout.is_some_and(|timeout| elapsed >= timeout)
}

/// Run a worktree verification command with an optional operator deadline.
/// Verification only needs an exit status, so stdout/stderr are discarded.
/// This is intentionally synchronous because the public worktree API is also
/// used from blocking fleet workers.
#[cfg(test)]
pub(super) fn verify_passes_with_timeout(
    worktree: &Path,
    verify: &str,
    timeout: Option<Duration>,
) -> bool {
    verify_passes_with_timeout_and_cancel(worktree, verify, timeout, None)
}

fn verify_passes_with_timeout_and_cancel(
    worktree: &Path,
    verify: &str,
    timeout: Option<Duration>,
    cancellation: Option<&tokio_util::sync::CancellationToken>,
) -> bool {
    if cancellation.is_some_and(|token| token.is_cancelled()) {
        return false;
    }
    crate::prepare_verify_workdir(worktree);
    let budget = VerifyBudget {
        started: Instant::now(),
        timeout,
    };
    let Ok(mut stability) = StabilityCheck::capture(worktree, cancellation, budget) else {
        return false;
    };
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(verify)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .current_dir(worktree)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    configure_private_process_group(&mut command);
    let Ok(mut child) = command.spawn() else {
        return false;
    };
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Even a successful shell can leave a background test/server
                // running. Verification owns that group through completion:
                // no descendant may keep changing the tree after we return.
                terminate_sync_child_group(&mut child);
                return status.success()
                    && stability
                        .as_mut()
                        .is_none_or(|check| check.unchanged().unwrap_or(false))
                    && cancellation.is_none_or(|token| !token.is_cancelled())
                    && budget.remaining().is_ok();
            }
            Ok(None)
                if !verification_timed_out(timeout, budget.started.elapsed())
                    && cancellation.is_none_or(|token| !token.is_cancelled()) =>
            {
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(None) | Err(_) => {
                terminate_sync_child_group(&mut child);
                return false;
            }
        }
    }
}

/// Keep verification tied to the source revision that entered it. A private
/// index uses exactly the staging rules used by merge preparation, including
/// tracked ignored files, without disturbing the user's staged changes.
struct StabilityCheck {
    worktree: PathBuf,
    directory: PathBuf,
    index: PathBuf,
    tree: Vec<u8>,
    cancellation: Option<tokio_util::sync::CancellationToken>,
    budget: VerifyBudget,
}

impl StabilityCheck {
    fn capture(
        worktree: &Path,
        cancellation: Option<&tokio_util::sync::CancellationToken>,
        budget: VerifyBudget,
    ) -> Result<Option<Self>> {
        let repository = run_git(
            worktree,
            &["rev-parse", "--is-inside-work-tree"],
            None,
            cancellation,
            budget,
        )?;
        if !repository.status.success() {
            // Plain directories are supported by the standalone verifier; no
            // Git candidate can be merged from them. Other Git failures are
            // inconclusive and must not authorize a worktree merge.
            ensure!(
                String::from_utf8_lossy(&repository.stderr).contains("not a git repository"),
                "could not inspect verification repository"
            );
            return Ok(None);
        }
        ensure!(
            repository.stdout == b"true\n",
            "verification requires a working tree"
        );
        let directory =
            std::env::temp_dir().join(format!("hi-verify-index-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&directory).context("creating verification index directory")?;
        let mut check = Self {
            worktree: worktree.to_path_buf(),
            index: directory.join("index"),
            directory,
            tree: Vec::new(),
            cancellation: cancellation.cloned(),
            budget,
        };
        let original = run_git(
            worktree,
            &["rev-parse", "--path-format=absolute", "--git-path", "index"],
            None,
            cancellation,
            budget,
        )?;
        ensure!(
            original.status.success(),
            "could not locate the worktree index"
        );
        let original = std::str::from_utf8(&original.stdout)
            .context("non-UTF-8 index path")?
            .trim_end_matches('\n');
        match std::fs::copy(original, &check.index) {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let empty = check.git(&["read-tree", "--empty"])?;
                ensure!(
                    empty.status.success(),
                    "could not initialize verification index"
                );
            }
            Err(error) => return Err(error).context("copying the worktree index for verification"),
        }
        check.tree = check.capture_tree()?;
        Ok(Some(check))
    }

    fn git(&self, args: &[&str]) -> Result<Output> {
        run_git(
            &self.worktree,
            args,
            Some(&self.index),
            self.cancellation.as_ref(),
            self.budget,
        )
    }

    fn capture_tree(&self) -> Result<Vec<u8>> {
        let args = ["add", "-A", "--", "."]
            .into_iter()
            .chain(PYCACHE_EXCLUDES.iter().copied())
            .collect::<Vec<_>>();
        let staged = self.git(&args)?;
        ensure!(
            staged.status.success(),
            "could not stage verification inputs: {}",
            String::from_utf8_lossy(&staged.stderr).trim()
        );
        let tree = self.git(&["write-tree"])?;
        ensure!(
            tree.status.success() && !tree.stdout.is_empty(),
            "could not capture verification tree"
        );
        Ok(tree.stdout)
    }

    fn unchanged(&mut self) -> Result<bool> {
        Ok(self.capture_tree()? == self.tree)
    }
}

impl Drop for StabilityCheck {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn run_git(
    worktree: &Path,
    args: &[&str],
    index: Option<&Path>,
    cancellation: Option<&tokio_util::sync::CancellationToken>,
    budget: VerifyBudget,
) -> Result<Output> {
    ensure!(
        cancellation.is_none_or(|token| !token.is_cancelled()),
        "verification snapshot cancelled"
    );
    let timeout = budget.remaining()?;
    let mut command = Command::new("git");
    command
        .current_dir(worktree)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    // Content filters may leave background children behind. They must not
    // keep an inherited capture pipe open after Git itself has exited.
    if args.first() == Some(&"add") {
        command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
    }
    if let Some(index) = index {
        command.env("GIT_INDEX_FILE", index);
    }
    configure_private_process_group(&mut command);
    let child = command.spawn().context("starting verification snapshot")?;
    let pid = child.id();
    let result = super::wait_for_apply(child, timeout, cancellation);
    #[cfg(unix)]
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = pid;
    result
}

#[derive(Clone, Copy)]
struct VerifyBudget {
    started: Instant,
    timeout: Option<Duration>,
}

impl VerifyBudget {
    fn remaining(self) -> Result<Option<Duration>> {
        let remaining = self
            .timeout
            .map(|timeout| timeout.saturating_sub(self.started.elapsed()));
        ensure!(
            remaining.is_none_or(|remaining| !remaining.is_zero()),
            "verification snapshot timed out"
        );
        Ok(remaining)
    }
}
