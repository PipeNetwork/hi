//! Trusted repository lifecycle-hook process execution.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;

use super::workspace_trusted;

/// Typed terminal state returned to workspace coordination. A cancellation is
/// reported only after the direct child is reaped and its private process group
/// has been terminated. Cleanup uncertainty remains explicitly indeterminate.
#[derive(Debug)]
pub(crate) enum HookExecution {
    Completed(Result<String>),
    Cancelled,
    Indeterminate(anyhow::Error),
}

/// Versioned structured hook response. Hook stdout may be this JSON; plain text
/// remains a backwards-compatible informational response.
#[derive(Debug, Deserialize)]
struct HookResponse {
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    decision: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

/// Run a lifecycle hook script from `.hi/hooks/<name>`.
///
/// Input is passed on stdin; stdout is returned as a user-visible report. A
/// non-zero exit is a gate failure (callers decide whether to block an action).
pub async fn run_hook(workspace: &Path, name: &str, input: &str) -> Result<String> {
    validate_hook(workspace, name)?;
    run_hook_process(workspace, name, input, hook_timeout()).await
}

/// Cancellation-aware entry used by the turn coordinator after workspace
/// admission. Unlike cancelling a wrapper future, this retains ownership of
/// the child until process-group termination and direct-child reap complete.
pub(crate) async fn run_hook_cancellable(
    workspace: &Path,
    name: &str,
    input: &str,
    cancellation: &crate::TurnCancellation,
) -> HookExecution {
    if cancellation.is_cancelled() {
        return HookExecution::Cancelled;
    }
    if let Err(error) = validate_hook(workspace, name) {
        return HookExecution::Completed(Err(error));
    }
    run_hook_process_controlled(
        workspace,
        name,
        input,
        hook_timeout(),
        Some(cancellation.clone()),
    )
    .await
}

fn validate_hook(workspace: &Path, name: &str) -> Result<()> {
    if !workspace_trusted(workspace) {
        bail!("workspace is untrusted; run `/trust on` before executing project hooks");
    }
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
    {
        bail!("invalid hook name {name:?}");
    }
    let path = workspace.join(".hi").join("hooks").join(name);
    if !path.is_file() {
        bail!("hook not found: {}", path.display());
    }
    Ok(())
}

pub(super) fn hook_timeout_from_value(value: Option<&str>) -> Option<std::time::Duration> {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(std::time::Duration::from_secs)
}

/// Optional operator-selected lifecycle-hook timeout. Hooks are trusted,
/// productive project work and therefore have no wall-clock ceiling by
/// default. Whole-turn cancellation still tears down their process group.
fn hook_timeout() -> Option<std::time::Duration> {
    let configured = std::env::var("HI_HOOK_TIMEOUT_SECS").ok();
    hook_timeout_from_value(configured.as_deref())
}

pub(super) async fn run_hook_process(
    workspace: &Path,
    name: &str,
    input: &str,
    timeout: Option<std::time::Duration>,
) -> Result<String> {
    match run_hook_process_controlled(workspace, name, input, timeout, None).await {
        HookExecution::Completed(result) => result,
        HookExecution::Cancelled => bail!("hook {name} was cancelled"),
        HookExecution::Indeterminate(error) => Err(error),
    }
}

#[cfg(test)]
pub(crate) async fn run_hook_process_cancellable_for_test(
    workspace: &Path,
    name: &str,
    input: &str,
    cancellation: &crate::TurnCancellation,
) -> HookExecution {
    run_hook_process_controlled(workspace, name, input, None, Some(cancellation.clone())).await
}

async fn run_hook_process_controlled(
    workspace: &Path,
    name: &str,
    input: &str,
    timeout: Option<std::time::Duration>,
    cancellation: Option<crate::TurnCancellation>,
) -> HookExecution {
    let path = workspace.join(".hi").join("hooks").join(name);
    let mut command = tokio::process::Command::new(&path);
    command
        .current_dir(workspace)
        .env("HI_HOOK", name)
        .env("HI_WORKSPACE", workspace)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    command.kill_on_drop(true);
    #[cfg(unix)]
    command.process_group(0);
    let mut child = match command
        .spawn()
        .with_context(|| format!("spawning hook {}", path.display()))
    {
        Ok(child) => child,
        Err(error) => return HookExecution::Completed(Err(error)),
    };
    let mut process_group = HookProcessGroupGuard::for_child(&child);
    let mut stdin_task = child.stdin.take().map(|mut stdin| {
        let input = input.as_bytes().to_vec();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt as _;
            tolerate_closed_hook_stdin(stdin.write_all(&input).await)?;
            tolerate_closed_hook_stdin(stdin.shutdown().await)
        })
    });
    let Some(stdout) = child.stdout.take() else {
        return indeterminate_after_cleanup(
            &mut child,
            &mut process_group,
            &mut stdin_task,
            None,
            None,
            anyhow!("hook stdout was not piped"),
        )
        .await;
    };
    let Some(stderr) = child.stderr.take() else {
        return indeterminate_after_cleanup(
            &mut child,
            &mut process_group,
            &mut stdin_task,
            Some(tokio::spawn(read_hook_output(stdout))),
            None,
            anyhow!("hook stderr was not piped"),
        )
        .await;
    };
    let mut stdout_task = Some(tokio::spawn(read_hook_output(stdout)));
    let mut stderr_task = Some(tokio::spawn(read_hook_output(stderr)));

    let trigger = {
        let wait = child.wait();
        tokio::pin!(wait);
        tokio::select! {
            biased;
            _ = wait_for_cancellation(cancellation.clone()) => ProcessTrigger::Cancelled,
            _ = wait_for_timeout(timeout) => ProcessTrigger::TimedOut(timeout.expect("timeout waiter cannot complete without a timeout")),
            status = &mut wait => ProcessTrigger::Exited(status),
        }
    };

    match trigger {
        ProcessTrigger::Cancelled => {
            let cleanup = interrupt_and_reap(
                &mut child,
                &mut process_group,
                &mut stdin_task,
                &mut stdout_task,
                &mut stderr_task,
            )
            .await;
            match cleanup {
                Ok(()) => HookExecution::Cancelled,
                Err(error) => HookExecution::Indeterminate(
                    error.context(format!("cancelled hook {name} could not be proven reaped")),
                ),
            }
        }
        ProcessTrigger::TimedOut(limit) => {
            let cleanup = interrupt_and_reap(
                &mut child,
                &mut process_group,
                &mut stdin_task,
                &mut stdout_task,
                &mut stderr_task,
            )
            .await;
            match cleanup {
                Ok(()) => HookExecution::Completed(Err(anyhow!(
                    "hook {name} timed out after {}s",
                    limit.as_secs()
                ))),
                Err(error) => HookExecution::Indeterminate(error.context(format!(
                    "hook {name} timed out after {}s and could not be proven reaped",
                    limit.as_secs()
                ))),
            }
        }
        ProcessTrigger::Exited(status) => {
            let group_cleanup = process_group.terminate();
            let status = match status.context("waiting for hook process") {
                Ok(status) => status,
                Err(error) => {
                    abort_hook_io(&mut stdin_task, &mut stdout_task, &mut stderr_task).await;
                    return HookExecution::Indeterminate(error);
                }
            };
            if let Err(error) = group_cleanup {
                abort_hook_io(&mut stdin_task, &mut stdout_task, &mut stderr_task).await;
                return HookExecution::Indeterminate(
                    anyhow::Error::new(error)
                        .context("terminating hook descendants after direct-child exit"),
                );
            }
            match drain_hook_io(
                &mut stdin_task,
                &mut stdout_task,
                &mut stderr_task,
                cancellation,
            )
            .await
            {
                DrainOutcome::Completed(Ok((stdout, stderr))) => {
                    HookExecution::Completed(interpret_hook_output(name, status, stdout, stderr))
                }
                DrainOutcome::Completed(Err(error)) => HookExecution::Indeterminate(error),
                DrainOutcome::Cancelled => HookExecution::Cancelled,
                DrainOutcome::TimedOut => HookExecution::Indeterminate(anyhow!(
                    "hook output pipes did not close after process-group cleanup"
                )),
            }
        }
    }
}

enum ProcessTrigger {
    Exited(std::io::Result<std::process::ExitStatus>),
    Cancelled,
    TimedOut(std::time::Duration),
}

async fn wait_for_cancellation(cancellation: Option<crate::TurnCancellation>) {
    let Some(cancellation) = cancellation else {
        std::future::pending::<()>().await;
        return;
    };
    while !cancellation.is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

async fn wait_for_timeout(timeout: Option<std::time::Duration>) {
    match timeout {
        Some(timeout) => tokio::time::sleep(timeout).await,
        None => std::future::pending::<()>().await,
    }
}

async fn interrupt_and_reap(
    child: &mut tokio::process::Child,
    process_group: &mut HookProcessGroupGuard,
    stdin_task: &mut Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    stdout_task: &mut Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
    stderr_task: &mut Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
) -> Result<()> {
    let group_cleanup = process_group.terminate();
    let _ = child.start_kill();
    let reap = tokio::time::timeout(HOOK_REAP_GRACE, child.wait()).await;
    abort_hook_io(stdin_task, stdout_task, stderr_task).await;
    group_cleanup.context("terminating hook process group")?;
    match reap {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(error).context("waiting for terminated hook process"),
        Err(_) => bail!(
            "timed out after {}ms waiting for terminated hook process",
            HOOK_REAP_GRACE.as_millis()
        ),
    }
}

async fn indeterminate_after_cleanup(
    child: &mut tokio::process::Child,
    process_group: &mut HookProcessGroupGuard,
    stdin_task: &mut Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    mut stdout_task: Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
    mut stderr_task: Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
    error: anyhow::Error,
) -> HookExecution {
    let cleanup = interrupt_and_reap(
        child,
        process_group,
        stdin_task,
        &mut stdout_task,
        &mut stderr_task,
    )
    .await;
    HookExecution::Indeterminate(match cleanup {
        Ok(()) => error,
        Err(cleanup) => error.context(format!("hook cleanup also failed: {cleanup:#}")),
    })
}

enum DrainOutcome {
    Completed(Result<(Vec<u8>, Vec<u8>)>),
    Cancelled,
    TimedOut,
}

async fn drain_hook_io(
    stdin_task: &mut Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    stdout_task: &mut Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
    stderr_task: &mut Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
    cancellation: Option<crate::TurnCancellation>,
) -> DrainOutcome {
    let outcome = {
        let drains = async {
            if let Some(task) = stdin_task.as_mut() {
                task.await.context("joining hook stdin writer")??;
            }
            let stdout = stdout_task
                .as_mut()
                .expect("hook stdout task exists")
                .await
                .context("joining hook stdout reader")??;
            let stderr = stderr_task
                .as_mut()
                .expect("hook stderr task exists")
                .await
                .context("joining hook stderr reader")??;
            Ok::<_, anyhow::Error>((stdout, stderr))
        };
        tokio::pin!(drains);
        tokio::select! {
            biased;
            _ = wait_for_cancellation(cancellation) => DrainOutcome::Cancelled,
            result = &mut drains => DrainOutcome::Completed(result),
            _ = tokio::time::sleep(HOOK_PIPE_DRAIN_GRACE) => DrainOutcome::TimedOut,
        }
    };
    if !matches!(&outcome, DrainOutcome::Completed(Ok(_))) {
        abort_hook_io(stdin_task, stdout_task, stderr_task).await;
    }
    outcome
}

async fn abort_hook_io(
    stdin_task: &mut Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    stdout_task: &mut Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
    stderr_task: &mut Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
) {
    if let Some(task) = stdin_task.take() {
        task.abort();
        let _ = task.await;
    }
    if let Some(task) = stdout_task.take() {
        task.abort();
        let _ = task.await;
    }
    if let Some(task) = stderr_task.take() {
        task.abort();
        let _ = task.await;
    }
}

fn interpret_hook_output(
    name: &str,
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
) -> Result<String> {
    let stdout = String::from_utf8_lossy(&stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&stderr).trim().to_string();
    if !status.success() {
        bail!(
            "hook {name} failed ({}): {}",
            status.code().unwrap_or(-1),
            if stderr.is_empty() { stdout } else { stderr }
        );
    }
    if let Ok(response) = serde_json::from_str::<HookResponse>(&stdout) {
        if response.version.unwrap_or(1) != 1 {
            bail!("hook {name} returned unsupported protocol version");
        }
        let message = response.message.unwrap_or_default();
        match response.decision.as_deref().unwrap_or("allow") {
            "allow" | "warn" => {
                return Ok(if message.is_empty() {
                    format!("hook {name}: ok")
                } else {
                    format!("hook {name}: {message}")
                });
            }
            "deny" | "block" => bail!(
                "hook {name} denied action{}",
                if message.is_empty() {
                    String::new()
                } else {
                    format!(": {message}")
                }
            ),
            other => bail!("hook {name} returned unknown decision {other:?}"),
        }
    }
    Ok(if stdout.is_empty() {
        format!("hook {name}: ok")
    } else {
        format!("hook {name}:\n{stdout}")
    })
}

pub(super) const MAX_HOOK_OUTPUT_BYTES: usize = 1024 * 1024;
pub(super) const HOOK_PIPE_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);
const HOOK_REAP_GRACE: std::time::Duration = std::time::Duration::from_millis(500);

pub(super) fn tolerate_closed_hook_stdin(result: std::io::Result<()>) -> std::io::Result<()> {
    match result {
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        result => result,
    }
}

async fn read_hook_output(
    mut reader: impl tokio::io::AsyncRead + Unpin,
) -> std::io::Result<Vec<u8>> {
    use tokio::io::AsyncReadExt as _;

    const HEAD_BYTES: usize = MAX_HOOK_OUTPUT_BYTES / 2;
    const TAIL_BYTES: usize = MAX_HOOK_OUTPUT_BYTES - HEAD_BYTES;
    const OMITTED: &[u8] = b"\n[... hook output truncated ...]\n";
    let mut head = Vec::with_capacity(HEAD_BYTES);
    let mut tail = std::collections::VecDeque::with_capacity(TAIL_BYTES);
    let mut total = 0usize;
    let mut buffer = [0u8; 8 * 1024];
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        total = total.saturating_add(count);
        for &byte in &buffer[..count] {
            if head.len() < HEAD_BYTES {
                head.push(byte);
            } else {
                if tail.len() == TAIL_BYTES {
                    tail.pop_front();
                }
                tail.push_back(byte);
            }
        }
    }
    if total > MAX_HOOK_OUTPUT_BYTES {
        head.extend_from_slice(OMITTED);
    }
    head.extend(tail);
    Ok(head)
}

#[cfg(unix)]
struct HookProcessGroupGuard {
    process_group: Option<libc::pid_t>,
}

#[cfg(unix)]
impl HookProcessGroupGuard {
    fn for_child(child: &tokio::process::Child) -> Self {
        Self {
            process_group: child.id().map(|pid| pid as libc::pid_t),
        }
    }

    fn terminate(&mut self) -> std::io::Result<()> {
        let Some(process_group) = self.process_group.take() else {
            return Ok(());
        };
        // SAFETY: the child was spawned as leader of a private process group;
        // a negative PID targets only that group.
        let result = unsafe { libc::kill(-process_group, libc::SIGKILL) };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

#[cfg(unix)]
impl Drop for HookProcessGroupGuard {
    fn drop(&mut self) {
        let _ = self.terminate();
    }
}

#[cfg(not(unix))]
struct HookProcessGroupGuard;

#[cfg(not(unix))]
impl HookProcessGroupGuard {
    fn for_child(_child: &tokio::process::Child) -> Self {
        Self
    }

    fn terminate(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
