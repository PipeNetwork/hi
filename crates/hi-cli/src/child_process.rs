//! Hardened execution for delegate and best-of child `hi` processes.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

static CHILD_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn child_runtime() -> Result<&'static tokio::runtime::Runtime> {
    if let Some(runtime) = CHILD_RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(2)
                .clamp(2, 8),
        )
        .enable_all()
        .build()
        .context("creating shared child-process runtime")?;
    let _ = CHILD_RUNTIME.set(runtime);
    Ok(CHILD_RUNTIME.get().expect("child runtime initialized"))
}

pub(crate) fn run_maybe_cancelled(
    workspace_root: &Path,
    executable: &Path,
    arguments: Vec<OsString>,
    environment: Vec<(OsString, OsString)>,
    timeout: Option<Duration>,
    log_path: &Path,
    cancellation: Option<hi_agent::TurnCancellation>,
) -> Result<hi_tools::ProcessExecution> {
    let workspace_root = workspace_root.to_path_buf();
    let executable = executable.to_path_buf();
    let log_path = log_path.to_path_buf();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating child log directory {}", parent.display()))?;
    }
    // Establish the durable artifact before launch. The typed result replaces
    // it after completion, including timeout/failure information when present.
    std::fs::write(&log_path, [])
        .with_context(|| format!("creating child log {}", log_path.display()))?;

    let execution = child_runtime()?.block_on(async move {
        let runner = hi_tools::ProcessRunner::new(&workspace_root)?;
        let started = Instant::now();
        let run =
            runner.run_program_with_env_maybe_timeout(executable, arguments, environment, timeout);
        match cancellation {
            Some(cancel) => {
                tokio::select! {
                    result = run => result,
                    _ = wait_for_cancel(cancel) => Ok(cancelled_execution(started)),
                }
            }
            None => run.await,
        }
    })?;

    write_log(&log_path, &execution)?;
    Ok(execution)
}

pub(crate) async fn wait_for_cancel(cancellation: hi_agent::TurnCancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn cancelled_execution(started: Instant) -> hi_tools::ProcessExecution {
    hi_tools::ProcessExecution {
        status: hi_tools::ToolStatus::Cancelled,
        outcome: hi_tools::ProcessOutcome {
            exit_code: None,
            stdout_summary: String::new(),
            stderr_summary: "cancelled".into(),
            duration_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        },
        truncation: hi_tools::TruncationState::Complete,
    }
}

fn write_log(path: &PathBuf, execution: &hi_tools::ProcessExecution) -> Result<()> {
    let text = format!(
        "status: {:?}\nexit_code: {:?}\nduration_ms: {}\ntruncation: {:?}\n\nstdout:\n{}\n\nstderr:\n{}\n",
        execution.status,
        execution.outcome.exit_code,
        execution.outcome.duration_ms,
        execution.truncation,
        execution.outcome.stdout_summary,
        execution.outcome.stderr_summary,
    );
    std::fs::write(path, text).with_context(|| format!("writing child log {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "hi-child-process-{label}-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn run_stops_when_cancelled() {
        let dir = temp_dir("cancel");
        let cancel = hi_agent::TurnCancellation::new();
        let cancel_thread = cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            cancel_thread.cancel();
        });
        let started = Instant::now();
        let execution = run_maybe_cancelled(
            &dir,
            Path::new("/bin/sleep"),
            vec![OsString::from("30")],
            Vec::new(),
            None,
            &dir.join("child.log"),
            Some(cancel),
        )
        .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(execution.status, hi_tools::ToolStatus::Cancelled);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "cancel must kill the child instead of waiting out the timeout: {:?}",
            started.elapsed()
        );
    }
}
