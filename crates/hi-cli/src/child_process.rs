//! Hardened execution for delegate and best-of child `hi` processes.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};

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

    let execution = std::thread::spawn(move || -> Result<hi_tools::ProcessExecution> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("creating child-process runtime")?;
        runtime.block_on(async move {
            let runner = hi_tools::ProcessRunner::new(&workspace_root)?;
            runner
                .run_program_with_env(executable, arguments, environment, timeout)
                .await
        })
    })
    .join()
    .map_err(|_| anyhow!("child-process worker panicked"))??;

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
