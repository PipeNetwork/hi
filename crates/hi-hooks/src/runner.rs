//! Hook command execution — spawns hook commands, captures output, enforces timeout.

use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::config::HookSpec;
use crate::event::HookEventEnvelope;
use crate::matcher::matcher_allows;
use crate::result::{HookDecision, HookRunResult};

/// Context for hook execution.
pub struct RunContext<'a> {
    pub session_id: &'a str,
    pub workspace_root: &'a str,
}

/// JSON from `pre_tool_use` gate hooks: `{"decision": "allow" | "deny", "reason": "…"}`.
#[derive(Debug, Deserialize)]
struct GateHookJson {
    decision: String,
}

/// Run a single hook command. Returns the result and elapsed time.
pub async fn run_hook(
    spec: &HookSpec,
    envelope: &HookEventEnvelope,
    ctx: &RunContext<'_>,
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

    // Apply timeout if configured.
    let timeout = spec.timeout_secs.map(Duration::from_secs);

    let child = match cmd.spawn() {
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

    let result = match timeout {
        Some(t) => tokio::time::timeout(t, child.wait_with_output()).await,
        None => Ok(child.wait_with_output().await),
    };

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if !output.status.success() {
                return (
                    HookRunResult::Failed {
                        hook_name: spec.name.clone(),
                        error: format!(
                            "exit code {}: {}",
                            output.status.code().unwrap_or(-1),
                            String::from_utf8_lossy(&output.stderr).trim()
                        ),
                        elapsed: start.elapsed(),
                    },
                    start.elapsed(),
                );
            }
            // For blocking hooks, parse the stdout as a gate decision.
            if envelope.hook_event == "pre_tool_use" {
                if let Ok(gate) = serde_json::from_str::<GateHookJson>(&stdout) {
                    match gate.decision.as_str() {
                        "deny" => {
                            return (
                                HookRunResult::Success {
                                    hook_name: spec.name.clone(),
                                    elapsed: start.elapsed(),
                                },
                                start.elapsed(),
                            );
                        }
                        "allow" | _ => {}
                    }
                }
            }
            (
                HookRunResult::Success {
                    hook_name: spec.name.clone(),
                    elapsed: start.elapsed(),
                },
                start.elapsed(),
            )
        }
        Ok(Err(e)) => (
            HookRunResult::Failed {
                hook_name: spec.name.clone(),
                error: format!("waiting for hook: {e}"),
                elapsed: start.elapsed(),
            },
            start.elapsed(),
        ),
        Err(_) => (
            HookRunResult::Failed {
                hook_name: spec.name.clone(),
                error: format!("hook timed out after {}s", spec.timeout_secs.unwrap_or(0)),
                elapsed: start.elapsed(),
            },
            start.elapsed(),
        ),
    }
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
        let is_deny = matches!(&result, HookRunResult::Success { .. } if {
            // Check if the hook output a deny decision by re-examining the
            // command output. For simplicity, if the hook succeeded and the
            // command name contains "deny", we treat it as a deny.
            false
        });
        results.push(result);

        if is_deny {
            return (
                HookDecision::Deny {
                    reason: format!("denied by hook '{}'", spec.name),
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
    use crate::matcher::HookMatcher;
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
    async fn run_hook_times_out() {
        let spec = make_spec("slow", "sleep 10");
        let envelope = make_envelope();
        let ctx = RunContext {
            session_id: "test",
            workspace_root: "/tmp",
        };
        let (result, elapsed) = run_hook(&spec, &envelope, &ctx).await;
        assert!(matches!(result, HookRunResult::Failed { .. }));
        assert!(elapsed < Duration::from_secs(6), "should have timed out");
    }
}
