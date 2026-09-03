//! Hook command execution — spawns hook commands, captures output, enforces timeout.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::config::HookSpec;
use crate::event::HookEventEnvelope;
use crate::matcher::matcher_allows;
use crate::result::{HookDecision, HookRunResult};

/// Retain useful evidence without ever allowing a noisy hook to grow memory
/// without bound. Readers continue draining after this cap is reached.
const HOOK_OUTPUT_CAP_BYTES: usize = 64 * 1024;
/// A direct shell may exit after daemonizing a descendant that inherited its
/// pipes. Descendants are killed at the shell boundary, then buffered output
/// gets this short grace period to drain.
const PIPE_DRAIN_GRACE: Duration = Duration::from_secs(2);
/// Reap the direct process briefly after a timeout. This is cleanup, not an
/// additional productive execution deadline.
const PROCESS_REAP_GRACE: Duration = Duration::from_secs(2);

/// Context for hook execution.
pub struct RunContext<'a> {
    pub session_id: &'a str,
    pub workspace_root: &'a str,
}

/// JSON from `pre_tool_use` gate hooks: `{"decision": "allow" | "deny", "reason": "…"}`.
#[derive(Debug, Deserialize)]
struct GateHookJson {
    decision: String,
    #[serde(default)]
    reason: Option<String>,
}

/// Run a single hook command. Returns the result and elapsed time.
pub async fn run_hook(
    spec: &HookSpec,
    envelope: &HookEventEnvelope,
    ctx: &RunContext<'_>,
) -> (HookRunResult, Duration) {
    run_hook_with_timeout(
        spec,
        envelope,
        ctx,
        spec.timeout_secs.map(Duration::from_secs),
    )
    .await
}

async fn run_hook_with_timeout(
    spec: &HookSpec,
    envelope: &HookEventEnvelope,
    ctx: &RunContext<'_>,
    timeout: Option<Duration>,
) -> (HookRunResult, Duration) {
    let start = Instant::now();

    // Serialize the envelope as JSON for stdin.
    let stdin = match serde_json::to_string(envelope) {
        Ok(s) => s,
        Err(e) => {
            return (
                HookRunResult::Failed {
                    hook_name: spec.name.clone(),
                    error: format!("serializing envelope: {e}"),
                    elapsed: start.elapsed(),
                },
                start.elapsed(),
            );
        }
    };

    // Build the command. We use `sh -c` so the command can use shell features
    // (pipes, redirects, env vars). The envelope JSON is passed via stdin.
    let mut cmd = tokio::process::Command::new("sh");
    cmd.arg("-c").arg(&spec.command);
    cmd.env("HI_HOOK_EVENT", &envelope.hook_event);
    cmd.env("HI_HOOK_NAME", &spec.name);
    cmd.env("HI_SESSION_ID", ctx.session_id);
    cmd.env("HI_WORKSPACE_ROOT", ctx.workspace_root);
    cmd.env("HI_HOOK_PAYLOAD", &stdin);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    // Ensure dropping the timed-out wait future terminates the hook instead of
    // leaving its shell running after the timeout has been reported.
    cmd.kill_on_drop(true);

    #[cfg(unix)]
    {
        cmd.process_group(0);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return (
                HookRunResult::Failed {
                    hook_name: spec.name.clone(),
                    error: format!("spawning hook command: {e}"),
                    elapsed: start.elapsed(),
                },
                start.elapsed(),
            );
        }
    };

    let mut process_group = HookProcessGroupGuard::for_child(&child);
    let Some(stdout) = child.stdout.take() else {
        process_group.terminate();
        return failed_result(&spec.name, "hook stdout was not piped", start);
    };
    let Some(stderr) = child.stderr.take() else {
        process_group.terminate();
        return failed_result(&spec.name, "hook stderr was not piped", start);
    };
    let mut readers = tokio::task::JoinSet::new();
    readers.spawn(async move { (HookStream::Stdout, read_hook_output(stdout).await) });
    readers.spawn(async move { (HookStream::Stderr, read_hook_output(stderr).await) });

    let status = match timeout {
        Some(limit) => match tokio::time::timeout(limit, child.wait()).await {
            Ok(status) => status,
            Err(_) => {
                terminate_hook(&mut child, &mut process_group).await;
                drain_after_termination(&mut readers).await;
                return failed_result(
                    &spec.name,
                    &format!("hook timed out after {}", duration_label(limit)),
                    start,
                );
            }
        },
        None => child.wait().await,
    };

    // The hook's lifecycle ends with its direct shell. Kill any daemonized
    // descendants before awaiting inherited pipes, otherwise `wait_with_output`
    // can hang forever even though the command itself already exited.
    process_group.terminate();
    let output =
        match tokio::time::timeout(PIPE_DRAIN_GRACE, collect_hook_output(&mut readers)).await {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                readers.abort_all();
                return failed_result(&spec.name, &error, start);
            }
            Err(_) => {
                readers.abort_all();
                return failed_result(
                    &spec.name,
                    "hook output pipes did not close after process-group cleanup",
                    start,
                );
            }
        };

    match status {
        Ok(status) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if !status.success() {
                return (
                    HookRunResult::Failed {
                        hook_name: spec.name.clone(),
                        error: format!(
                            "exit code {}: {}",
                            status.code().unwrap_or(-1),
                            String::from_utf8_lossy(&output.stderr).trim()
                        ),
                        elapsed: start.elapsed(),
                    },
                    start.elapsed(),
                );
            }
            // For blocking hooks, parse the stdout as a gate decision.
            if envelope.hook_event == "pre_tool_use"
                && let Ok(gate) = serde_json::from_str::<GateHookJson>(&stdout)
                && gate.decision == "deny"
            {
                return (
                    HookRunResult::Denied {
                        hook_name: spec.name.clone(),
                        reason: gate
                            .reason
                            .filter(|reason| !reason.trim().is_empty())
                            .unwrap_or_else(|| format!("denied by hook '{}'", spec.name)),
                        elapsed: start.elapsed(),
                    },
                    start.elapsed(),
                );
            }
            (
                HookRunResult::Success {
                    hook_name: spec.name.clone(),
                    elapsed: start.elapsed(),
                },
                start.elapsed(),
            )
        }
        Err(e) => (
            HookRunResult::Failed {
                hook_name: spec.name.clone(),
                error: format!("waiting for hook: {e}"),
                elapsed: start.elapsed(),
            },
            start.elapsed(),
        ),
    }
}

fn failed_result(name: &str, error: &str, start: Instant) -> (HookRunResult, Duration) {
    (
        HookRunResult::Failed {
            hook_name: name.to_owned(),
            error: error.to_owned(),
            elapsed: start.elapsed(),
        },
        start.elapsed(),
    )
}

fn duration_label(duration: Duration) -> String {
    if duration.subsec_nanos() == 0 {
        format!("{}s", duration.as_secs())
    } else {
        format!("{duration:?}")
    }
}

#[derive(Clone, Copy)]
enum HookStream {
    Stdout,
    Stderr,
}

#[derive(Default)]
struct HookOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

async fn collect_hook_output(
    readers: &mut tokio::task::JoinSet<(HookStream, std::io::Result<Vec<u8>>)>,
) -> Result<HookOutput, String> {
    let mut output = HookOutput::default();
    while let Some(joined) = readers.join_next().await {
        let (stream, bytes) =
            joined.map_err(|error| format!("joining hook output reader: {error}"))?;
        let bytes = bytes.map_err(|error| format!("reading hook output: {error}"))?;
        match stream {
            HookStream::Stdout => output.stdout = bytes,
            HookStream::Stderr => output.stderr = bytes,
        }
    }
    Ok(output)
}

async fn read_hook_output(
    mut reader: impl tokio::io::AsyncRead + Unpin,
) -> std::io::Result<Vec<u8>> {
    use tokio::io::AsyncReadExt as _;

    const HEAD_BYTES: usize = HOOK_OUTPUT_CAP_BYTES / 2;
    const TAIL_BYTES: usize = HOOK_OUTPUT_CAP_BYTES - HEAD_BYTES;
    const OMITTED: &[u8] = b"\n[... hook output truncated ...]\n";

    let mut head = Vec::with_capacity(HEAD_BYTES);
    let mut tail = VecDeque::with_capacity(TAIL_BYTES);
    let mut total = 0usize;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read);
        for &byte in &buffer[..read] {
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
    if total > HOOK_OUTPUT_CAP_BYTES {
        head.extend_from_slice(OMITTED);
    }
    head.extend(tail);
    Ok(head)
}

async fn terminate_hook(
    child: &mut tokio::process::Child,
    process_group: &mut HookProcessGroupGuard,
) {
    process_group.terminate();
    // `kill_on_drop` is the final fallback. Explicitly request termination too
    // so non-Unix platforms reap the immediate shell on timeout.
    let _ = child.start_kill();
    let _ = tokio::time::timeout(PROCESS_REAP_GRACE, child.wait()).await;
}

async fn drain_after_termination(
    readers: &mut tokio::task::JoinSet<(HookStream, std::io::Result<Vec<u8>>)>,
) {
    let _ = tokio::time::timeout(PIPE_DRAIN_GRACE, collect_hook_output(readers)).await;
    readers.abort_all();
}

#[cfg(unix)]
struct HookProcessGroupGuard {
    process_group: Option<libc::pid_t>,
}

#[cfg(unix)]
impl HookProcessGroupGuard {
    fn for_child(child: &tokio::process::Child) -> Self {
        Self {
            process_group: child.id().and_then(|pid| libc::pid_t::try_from(pid).ok()),
        }
    }

    fn terminate(&mut self) {
        if let Some(process_group) = self.process_group.take() {
            // SAFETY: the child was spawned as leader of a private process
            // group; a negative PID targets only that group. ESRCH is benign.
            unsafe {
                libc::kill(-process_group, libc::SIGKILL);
            }
        }
    }
}

#[cfg(unix)]
impl Drop for HookProcessGroupGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(not(unix))]
struct HookProcessGroupGuard;

#[cfg(not(unix))]
impl HookProcessGroupGuard {
    fn for_child(_child: &tokio::process::Child) -> Self {
        Self
    }

    fn terminate(&mut self) {}
}

/// Dispatch a `pre_tool_use` event against all matching hooks.
///
/// Runs hooks sequentially in config order. Only an explicit `deny` decision
/// from a hook stops the chain and blocks the tool call. Hook failures are
/// fail-open: the failure is logged but the tool call continues.
pub async fn run_pre_tool_hooks(
    registry: &crate::HookRegistry,
    envelope: &HookEventEnvelope,
    ctx: &RunContext<'_>,
    tool_name: Option<&str>,
) -> (HookDecision, Vec<HookRunResult>) {
    let hooks = registry.hooks_for(crate::HookEvent::PreToolUse);
    if hooks.is_empty() {
        return (HookDecision::Allow, Vec::new());
    }

    let mut results = Vec::new();
    for spec in hooks {
        if !spec.enabled {
            results.push(HookRunResult::Skipped {
                hook_name: spec.name.clone(),
            });
            continue;
        }
        if !matcher_allows(spec.matcher.as_ref(), tool_name) {
            continue;
        }

        let (result, _) = run_hook(spec, envelope, ctx).await;
        let denial = match &result {
            HookRunResult::Denied { reason, .. } => Some(reason.clone()),
            _ => None,
        };
        results.push(result);

        if let Some(reason) = denial {
            return (
                HookDecision::Deny {
                    reason,
                    hook_name: spec.name.clone(),
                },
                results,
            );
        }
    }

    (HookDecision::Allow, results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HandlerType, HookSpec};
    use crate::event::{HookEvent, HookEventEnvelope, HookPayload};
    use std::collections::HashMap;

    fn make_spec(name: &str, command: &str) -> HookSpec {
        HookSpec {
            name: name.into(),
            event: HookEvent::PreToolUse,
            handler_type: HandlerType::Command,
            command: command.into(),
            matcher: None,
            timeout_secs: Some(5),
            enabled: true,
            source_dir: std::path::PathBuf::from("/tmp"),
            extra_env: HashMap::new(),
        }
    }

    fn make_envelope() -> HookEventEnvelope {
        HookEventEnvelope {
            hook_event: "pre_tool_use".into(),
            session_id: "test".into(),
            workspace_root: "/tmp".into(),
            timestamp: "2025-01-01T00:00:00Z".into(),
            payload: HookPayload::PreToolUse {
                tool_name: "bash".into(),
                arguments: serde_json::json!({}),
            },
        }
    }

    #[cfg(unix)]
    fn process_is_alive(pid: libc::pid_t) -> bool {
        // SAFETY: signal zero only probes a PID created by this test.
        unsafe { libc::kill(pid, 0) == 0 }
    }

    #[cfg(unix)]
    async fn read_pid(path: &std::path::Path) -> libc::pid_t {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if let Ok(text) = std::fs::read_to_string(path)
                && let Ok(pid) = text.trim().parse()
            {
                return pid;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "hook did not write descendant PID to {}",
                path.display()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    #[cfg(unix)]
    async fn assert_process_exits(pid: libc::pid_t) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while process_is_alive(pid) && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!process_is_alive(pid), "hook descendant {pid} leaked");
    }

    #[tokio::test]
    async fn run_hook_succeeds_on_exit_zero() {
        let spec = make_spec("ok", "true");
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let (result, _) = run_hook(&spec, &envelope, &ctx).await;
        assert!(matches!(result, HookRunResult::Success { .. }));
    }

    #[tokio::test]
    async fn run_hook_fails_on_nonzero_exit() {
        let spec = make_spec("fail", "exit 1");
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let (result, _) = run_hook(&spec, &envelope, &ctx).await;
        assert!(matches!(result, HookRunResult::Failed { .. }));
    }

    #[tokio::test]
    async fn noisy_stdout_and_stderr_are_drained_with_bounded_head_tail_evidence() {
        let command = r#"
            printf 'stderr-head\n' >&2
            i=0
            while [ "$i" -lt 8000 ]; do
                printf 'stdout-abcdefghijklmnopqrstuvwxyz-0123456789\n'
                printf 'stderr-abcdefghijklmnopqrstuvwxyz-0123456789\n' >&2
                i=$((i + 1))
            done
            printf 'stderr-tail\n' >&2
            exit 7
        "#;
        let spec = make_spec("noisy", command);
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let (result, _) = run_hook(&spec, &envelope, &ctx).await;
        let HookRunResult::Failed { error, .. } = result else {
            panic!("noisy failing hook should report failure");
        };
        assert!(error.contains("stderr-head"), "{error}");
        assert!(error.contains("stderr-tail"), "{error}");
        assert!(error.contains("hook output truncated"), "{error}");
        assert!(
            error.len() <= HOOK_OUTPUT_CAP_BYTES + 128,
            "retained hook evidence was not bounded: {} bytes",
            error.len()
        );
    }

    #[tokio::test]
    async fn pre_tool_hook_enforces_explicit_deny() {
        let spec = make_spec(
            "gate",
            r#"printf '%s' '{"decision":"deny","reason":"unsafe operation"}'"#,
        );
        let mut registry = crate::HookRegistry::default();
        registry.append_specs(vec![spec]);
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };

        let (decision, results) =
            run_pre_tool_hooks(&registry, &envelope, &ctx, Some("bash")).await;

        assert_eq!(
            decision,
            HookDecision::Deny {
                reason: "unsafe operation".into(),
                hook_name: "gate".into(),
            }
        );
        assert!(matches!(results.as_slice(), [HookRunResult::Denied { .. }]));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timed_out_hook_process_is_terminated() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survived");
        let spec = make_spec(
            "slow",
            &format!("sleep 0.2; printf survived > {}", marker.display()),
        );
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let (result, _) =
            run_hook_with_timeout(&spec, &envelope, &ctx, Some(Duration::from_millis(50))).await;
        assert!(matches!(result, HookRunResult::Failed { .. }));
        for _ in 0..25 {
            if marker.exists() {
                panic!("timed-out hook kept running");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(!marker.exists(), "timed-out hook kept running");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn normal_shell_exit_kills_descendants_and_does_not_wait_for_inherited_pipes() {
        let dir = tempfile::tempdir().unwrap();
        let pid_path = dir.path().join("child.pid");
        let spec = make_spec(
            "daemon",
            &format!(
                "sleep 30 & echo $! > {}; printf '%s' '{{\"decision\":\"allow\"}}'",
                pid_path.display()
            ),
        );
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let started = Instant::now();
        let (result, _) =
            run_hook_with_timeout(&spec, &envelope, &ctx, Some(Duration::from_secs(2))).await;
        assert!(matches!(result, HookRunResult::Success { .. }));
        assert!(started.elapsed() < Duration::from_secs(2));
        let pid = read_pid(&pid_path).await;
        assert_process_exits(pid).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_the_complete_descendant_group() {
        let dir = tempfile::tempdir().unwrap();
        let pid_path = dir.path().join("child.pid");
        let spec = make_spec(
            "timeout-tree",
            &format!("sleep 30 & echo $! > {}; wait", pid_path.display()),
        );
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let (result, _) =
            run_hook_with_timeout(&spec, &envelope, &ctx, Some(Duration::from_millis(250))).await;
        let HookRunResult::Failed { error, .. } = result else {
            panic!("hook should time out");
        };
        assert!(error.contains("timed out after 250ms"), "{error}");
        let pid = read_pid(&pid_path).await;
        assert_process_exits(pid).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_the_runner_future_kills_the_complete_descendant_group() {
        let dir = tempfile::tempdir().unwrap();
        let pid_path = dir.path().join("child.pid");
        let spec = make_spec(
            "cancel-tree",
            &format!("sleep 30 & echo $! > {}; wait", pid_path.display()),
        );
        let envelope = make_envelope();
        let task = tokio::spawn(async move {
            let ctx = RunContext {
                session_id: "test",
                workspace_root: "/tmp",
            };
            run_hook_with_timeout(&spec, &envelope, &ctx, None).await
        });
        let pid = read_pid(&pid_path).await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_process_exits(pid).await;
    }

    #[tokio::test]
    async fn run_hook_times_out() {
        let spec = make_spec("slow", "sleep 1");
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let (result, elapsed) =
            run_hook_with_timeout(&spec, &envelope, &ctx, Some(Duration::from_millis(50))).await;
        assert!(matches!(result, HookRunResult::Failed { .. }));
        assert!(
            elapsed < Duration::from_millis(500),
            "should have timed out"
        );
    }
}
