//! Hardened execution for delegate and best-of child `hi` processes.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

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

pub(crate) fn run(
    workspace_root: &Path,
    executable: &Path,
    arguments: Vec<OsString>,
    environment: Vec<(OsString, OsString)>,
    timeout: Duration,
    log_path: &Path,
) -> Result<hi_tools::ProcessExecution> {
    let workspace_root = workspace_root.to_path_buf();
    let executable = executable.to_path_buf();
    let log_path = log_path.to_path_buf();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating child log directory {}", parent.display()))?;
    }
    // Establish the durable artifact before launch. The bounded result replaces
    // it after completion, including typed timeout/failure information.
    std::fs::write(&log_path, [])
        .with_context(|| format!("creating child log {}", log_path.display()))?;

    let execution = child_runtime()?.block_on(async move {
        let runner = hi_tools::ProcessRunner::new(&workspace_root)?;
        runner
            .run_program_with_env(executable, arguments, environment, timeout)
            .await
    })?;

    write_log(&log_path, &execution)?;
    Ok(execution)
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
