//! Background subagent task handlers: `task`, `get_task_output`, `wait_tasks`,
//! `kill_task`.
//!
//! The `task` tool spawns an `explore` or `delegate` subagent as a detached
//! Tokio task that runs independently of the parent turn. It returns immediately
//! with a task handle. The parent polls results with `get_task_output`, waits
//! with `wait_tasks`, and cancels with `kill_task`.
//!
//! Unlike the synchronous `explore`/`delegate` tools (which block the parent
//! turn until the subagent completes), background tasks let the parent continue
//! working while subagents run in parallel. The trade-off: the parent must
//! explicitly poll for results, and background subagents don't get live UI
//! streaming (their output is collected and returned on poll).
//!
//! Depth is capped at 1: the child is built with `explore_subagents = false`
//! and `is_subagent = true`, so it never sees the `task`/`explore`/`delegate`
//! tools and cannot spawn further subagents.

use std::time::Duration;

use hi_ai::ToolMode;
use serde_json::Value;

use crate::AgentConfig;
use crate::Ui;
use crate::ui::NullUi;

/// Cap on background subagent tasks per session.
pub(crate) const MAX_BG_SUBAGENTS_PER_SESSION: u32 = 16;

fn bg_tool_outcome(
    content: impl Into<String>,
    status: hi_tools::ToolStatus,
) -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content: content.into(),
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation: hi_tools::TruncationState::Complete,
    }
}

impl crate::Agent {
    /// Handle the `task` tool — spawn a background subagent.
    ///
    /// Parses `description`, `prompt`, `subagent_type` ("explore" or "delegate"),
    /// and optional `verify` (for delegate). Spawns the subagent as a detached
    /// Tokio task and returns immediately with the task ID.
    pub(crate) async fn handle_task(
        &mut self,
        arguments: &str,
        ui: &mut dyn Ui,
    ) -> hi_tools::ToolOutcome {
        let parsed = match serde_json::from_str::<Value>(arguments) {
            Ok(v) => v,
            Err(_) => {
                return bg_tool_outcome(
                    "task error: invalid JSON arguments",
                    hi_tools::ToolStatus::Failed,
                );
            }
        };

        let prompt = parsed
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if prompt.trim().is_empty() {
            return bg_tool_outcome(
                "task error: missing required \"prompt\" argument",
                hi_tools::ToolStatus::Failed,
            );
        }

        let description = parsed
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if description.trim().is_empty() {
            return bg_tool_outcome(
                "task error: missing required \"description\" argument",
                hi_tools::ToolStatus::Failed,
            );
        }

        let subagent_type = parsed
            .get("subagent_type")
            .and_then(Value::as_str)
            .unwrap_or("explore")
            .to_string();

        let verify = parsed
            .get("verify")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string);

        // Budget check — use the explore counter for explore, delegate for delegate.
        let is_explore = subagent_type == "explore";
        let slot = if is_explore {
            self.subagents
                .try_begin_explore(crate::agent::explore_turn::MAX_EXPLORE_SUBAGENTS_PER_SESSION)
        } else {
            self.subagents
                .try_begin_delegate(crate::agent::delegate_turn::MAX_DELEGATE_SUBAGENTS_PER_SESSION)
        };
        let Some(n) = slot else {
            let max = if is_explore {
                crate::agent::explore_turn::MAX_EXPLORE_SUBAGENTS_PER_SESSION
            } else {
                crate::agent::delegate_turn::MAX_DELEGATE_SUBAGENTS_PER_SESSION
            };
            return bg_tool_outcome(
                format!(
                    "task error: {} subagent budget exhausted ({max} this session)",
                    subagent_type
                ),
                hi_tools::ToolStatus::Denied,
            );
        };

        // For delegate, check that a delegate runner is available.
        if !is_explore && self.subagents.delegate_runner.is_none() {
            self.subagents.release_delegate();
            return bg_tool_outcome(
                "task error: no delegate runner attached — write-capable background subagents are unavailable",
                hi_tools::ToolStatus::Denied,
            );
        }

        // UI callout.
        let summary: String = description.chars().take(72).collect();
        ui.subagent_note(&format!("↳ background {subagent_type} task {n}: {summary}"));

        // Build the future factory and spawn the task.
        let provider = self.provider.clone();
        let child_config = if is_explore {
            self.build_bg_explore_config(n)
        } else {
            self.build_bg_delegate_config(n)
        };

        // The future factory is `Send` (a closure), but the future it produces
        // does NOT need to be `Send` — it runs on a worker thread's `LocalSet`.
        let prompt_for_factory = prompt.clone();
        let verify_for_factory = verify.clone();
        let factory: Box<dyn FnOnce() -> hi_tools::BgFuture + Send + 'static> = if is_explore {
            Box::new(move || Box::pin(run_bg_explore(provider, child_config, prompt_for_factory)))
        } else {
            Box::new(move || {
                Box::pin(run_bg_delegate(
                    provider,
                    child_config,
                    prompt_for_factory,
                    verify_for_factory,
                ))
            })
        };

        let task_id = match self
            .bg_tasks
            .spawn(&description, &subagent_type, factory)
            .await
        {
            Ok(id) => id,
            Err(e) => {
                if is_explore {
                    self.subagents.release_explore();
                } else {
                    self.subagents.release_delegate();
                }
                return bg_tool_outcome(
                    format!("task error: failed to spawn background task: {e}"),
                    hi_tools::ToolStatus::Failed,
                );
            }
        };

        bg_tool_outcome(
            format!(
                "Background {subagent_type} task spawned: {task_id}\nDescription: {description}\nPoll results with get_task_output (task_ids: [\"{task_id}\"]) or wait_tasks."
            ),
            hi_tools::ToolStatus::Succeeded,
        )
    }

    /// Handle the `get_task_output` tool — poll one or more background tasks.
    pub(crate) async fn handle_get_task_output(&self, arguments: &str) -> hi_tools::ToolOutcome {
        let parsed = match serde_json::from_str::<Value>(arguments) {
            Ok(v) => v,
            Err(_) => {
                return bg_tool_outcome(
                    "get_task_output error: invalid JSON arguments",
                    hi_tools::ToolStatus::Failed,
                );
            }
        };

        // task_ids can be a string or array of strings.
        let ids: Vec<String> = match parsed.get("task_ids") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => {
                return bg_tool_outcome(
                    "get_task_output error: \"task_ids\" must be a string or array of strings",
                    hi_tools::ToolStatus::Failed,
                );
            }
        };

        if ids.is_empty() {
            return bg_tool_outcome(
                "get_task_output error: \"task_ids\" is empty",
                hi_tools::ToolStatus::Failed,
            );
        }

        let timeout_ms = parsed
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let timeout = if timeout_ms == 0 {
            Duration::ZERO
        } else {
            Duration::from_millis(timeout_ms).min(hi_tools::MAX_WAIT_TIMEOUT)
        };

        let results = self.bg_tasks.poll_many(&ids, timeout).await;
        let content = format_task_results(&results);
        bg_tool_outcome(content, hi_tools::ToolStatus::Succeeded)
    }

    /// Handle the `wait_tasks` tool — wait for multiple background tasks.
    pub(crate) async fn handle_wait_tasks(&self, arguments: &str) -> hi_tools::ToolOutcome {
        let parsed = match serde_json::from_str::<Value>(arguments) {
            Ok(v) => v,
            Err(_) => {
                return bg_tool_outcome(
                    "wait_tasks error: invalid JSON arguments",
                    hi_tools::ToolStatus::Failed,
                );
            }
        };

        let ids: Vec<String> = parsed
            .get("task_ids")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        if ids.is_empty() {
            return bg_tool_outcome(
                "wait_tasks error: \"task_ids\" is empty",
                hi_tools::ToolStatus::Failed,
            );
        }

        let mode = parsed
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or("wait_all");
        let timeout_ms = parsed
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(30_000);
        let timeout = Duration::from_millis(timeout_ms).min(hi_tools::MAX_WAIT_TIMEOUT);

        let results = if mode == "wait_any" {
            self.bg_tasks.wait_any(&ids, timeout).await
        } else {
            self.bg_tasks.wait_all(&ids, timeout).await
        };

        let content = format_task_results(&results);
        bg_tool_outcome(content, hi_tools::ToolStatus::Succeeded)
    }

    /// Handle the `kill_task` tool — cancel a background task.
    pub(crate) async fn handle_kill_task(&self, arguments: &str) -> hi_tools::ToolOutcome {
        let parsed = match serde_json::from_str::<Value>(arguments) {
            Ok(v) => v,
            Err(_) => {
                return bg_tool_outcome(
                    "kill_task error: invalid JSON arguments",
                    hi_tools::ToolStatus::Failed,
                );
            }
        };

        let task_id = match parsed.get("task_id").and_then(Value::as_str) {
            Some(s) => s.to_string(),
            None => {
                return bg_tool_outcome(
                    "kill_task error: missing required \"task_id\" argument",
                    hi_tools::ToolStatus::Failed,
                );
            }
        };

        match self.bg_tasks.kill(&task_id).await {
            Some(outcome) => {
                let content = format!(
                    "Task {} cancelled.\nState: {:?}\nOutput: {}",
                    outcome.id, outcome.state, outcome.output
                );
                bg_tool_outcome(content, hi_tools::ToolStatus::Succeeded)
            }
            None => bg_tool_outcome(
                format!("kill_task error: no task with id \"{task_id}\""),
                hi_tools::ToolStatus::Failed,
            ),
        }
    }

    /// Build a child config for a background explore subagent.
    fn build_bg_explore_config(&self, n: u32) -> AgentConfig {
        AgentConfig {
            paths: crate::AgentPaths {
                workspace_root: self.runtime.root().to_path_buf(),
                state_root: self
                    .runtime
                    .state_root()
                    .join("subagents")
                    .join(format!("bg-explore-{n}")),
                ..crate::AgentPaths::default()
            },
            routing: crate::AgentRouting {
                model: self.config.routing.model.clone(),
                requested_max_tokens: self.config.routing.requested_max_tokens,
                max_tokens: self.config.routing.max_tokens,
                max_tokens_explicit: self.config.routing.max_tokens_explicit,
                temperature: self.config.routing.temperature,
                thinking_budget: self.config.routing.thinking_budget,
                reasoning_effort: self.config.routing.reasoning_effort,
                compat: self.config.routing.compat,
                context_window: self.config.routing.context_window,
                tool_mode: ToolMode::ReadOnly,
                ..crate::AgentRouting::default()
            },
            gates: crate::AgentGates {
                verification: crate::VerificationMode::Disabled,
                read_only_preflight: false,
                lsp_mode: crate::LspMode::Off,
                ..crate::AgentGates::default()
            },
            loop_limits: crate::AgentLoopLimits {
                max_steps: 10,
                max_parallel_tools: 2,
                max_silent_continues: 0,
                ..crate::AgentLoopLimits::default()
            },
            subagents: crate::AgentSubagents {
                explore_subagents: false,
                write_subagents: crate::WriteSubagentPolicy::Off,
                is_subagent: true,
                ..crate::AgentSubagents::default()
            },
            ..self.config.clone()
        }
    }

    /// Build a child config for a background delegate subagent.
    fn build_bg_delegate_config(&self, n: u32) -> AgentConfig {
        AgentConfig {
            paths: crate::AgentPaths {
                workspace_root: self.runtime.root().to_path_buf(),
                state_root: self
                    .runtime
                    .state_root()
                    .join("subagents")
                    .join(format!("bg-delegate-{n}")),
                ..crate::AgentPaths::default()
            },
            routing: crate::AgentRouting {
                model: self.config.routing.model.clone(),
                requested_max_tokens: self.config.routing.requested_max_tokens,
                max_tokens: self.config.routing.max_tokens,
                max_tokens_explicit: self.config.routing.max_tokens_explicit,
                temperature: self.config.routing.temperature,
                thinking_budget: self.config.routing.thinking_budget,
                reasoning_effort: self.config.routing.reasoning_effort,
                compat: self.config.routing.compat,
                context_window: self.config.routing.context_window,
                tool_mode: ToolMode::Auto,
                ..crate::AgentRouting::default()
            },
            gates: crate::AgentGates {
                verification: crate::VerificationMode::Disabled,
                read_only_preflight: false,
                lsp_mode: self.config.gates.lsp_mode,
                ..crate::AgentGates::default()
            },
            loop_limits: crate::AgentLoopLimits {
                max_steps: 20,
                max_parallel_tools: 2,
                max_silent_continues: 0,
                ..crate::AgentLoopLimits::default()
            },
            subagents: crate::AgentSubagents {
                explore_subagents: false,
                write_subagents: crate::WriteSubagentPolicy::Off,
                is_subagent: true,
                ..crate::AgentSubagents::default()
            },
            ..self.config.clone()
        }
    }
}

/// Format task results for the model-facing tool output.
fn format_task_results(results: &[hi_tools::BackgroundTaskOutcome]) -> String {
    if results.is_empty() {
        return "No tasks found.".to_string();
    }
    let mut lines = Vec::with_capacity(results.len());
    for outcome in results {
        lines.push(format!(
            "Task {} ({}/{}) — {:?}\n  {}",
            outcome.id, outcome.description, outcome.subagent_type, outcome.state, outcome.output
        ));
    }
    lines.join("\n\n")
}

/// Run a background explore subagent to completion and return its outcome.
async fn run_bg_explore(
    provider: std::sync::Arc<dyn hi_ai::Provider>,
    config: AgentConfig,
    prompt: String,
) -> hi_tools::BackgroundTaskOutcome {
    let child_prompt = format!(
        "Answer this question about the codebase. Read and search the relevant files as needed, then \
         reply with a concise, self-contained answer that cites the specific files and locations \
         supporting it.\n\nQuestion: {prompt}"
    );

    let child = match crate::Agent::new(provider, config) {
        Ok(c) => c,
        Err(e) => {
            return hi_tools::BackgroundTaskOutcome {
                id: String::new(),
                description: String::new(),
                subagent_type: "explore".into(),
                state: hi_tools::BackgroundTaskState::Failed,
                output: format!("Failed to create explore subagent: {e}"),
                applied: false,
                changed_files: vec![],
            };
        }
    };

    let mut child = child;
    // Use a no-op UI for background subagents — their output is collected, not streamed.
    let mut ui = NullUi;
    let result = child.run_turn(&child_prompt, &mut ui).await;

    let (state, output) = match result {
        Ok(turn) => match turn.status {
            crate::TurnStatus::Completed => (
                hi_tools::BackgroundTaskState::Completed,
                child
                    .last_assistant_text()
                    .unwrap_or_else(|| "explore subagent produced no answer".to_string()),
            ),
            crate::TurnStatus::Blocked => (
                hi_tools::BackgroundTaskState::Failed,
                "explore subagent was blocked".to_string(),
            ),
            crate::TurnStatus::Cancelled => (
                hi_tools::BackgroundTaskState::Cancelled,
                "explore subagent was cancelled".to_string(),
            ),
            _ => (
                hi_tools::BackgroundTaskState::Failed,
                child
                    .last_assistant_text()
                    .unwrap_or_else(|| "explore subagent failed".to_string()),
            ),
        },
        Err(e) => (
            hi_tools::BackgroundTaskState::Failed,
            format!("explore subagent error: {e}"),
        ),
    };

    let _ = child.kill_background_processes();

    hi_tools::BackgroundTaskOutcome {
        id: String::new(),
        description: String::new(),
        subagent_type: "explore".into(),
        state,
        output,
        applied: false,
        changed_files: vec![],
    }
}

/// Run a background delegate subagent to completion and return its outcome.
///
/// For background delegate tasks, we run a write-capable child agent in-process
/// (like explore). The child's changes are applied directly to the working tree
/// (no worktree isolation for background tasks — the parent is still working and
/// can observe changes as they happen). If a `verify` command is provided, it's
/// run after the child completes; if it fails, the outcome is marked as failed
/// but changes are NOT rolled back (background tasks don't have the same
/// transactional guarantees as synchronous delegate).
async fn run_bg_delegate(
    provider: std::sync::Arc<dyn hi_ai::Provider>,
    config: AgentConfig,
    prompt: String,
    verify: Option<String>,
) -> hi_tools::BackgroundTaskOutcome {
    let child = match crate::Agent::new(provider, config) {
        Ok(c) => c,
        Err(e) => {
            return hi_tools::BackgroundTaskOutcome {
                id: String::new(),
                description: String::new(),
                subagent_type: "delegate".into(),
                state: hi_tools::BackgroundTaskState::Failed,
                output: format!("Failed to create delegate subagent: {e}"),
                applied: false,
                changed_files: vec![],
            };
        }
    };

    let mut child = child;
    let mut ui = NullUi;
    let result = child.run_turn(&prompt, &mut ui).await;

    let (state, output) = match result {
        Ok(turn) => match turn.status {
            crate::TurnStatus::Completed => (
                hi_tools::BackgroundTaskState::Completed,
                child
                    .last_assistant_text()
                    .unwrap_or_else(|| "delegate subagent completed".to_string()),
            ),
            crate::TurnStatus::Blocked => (
                hi_tools::BackgroundTaskState::Failed,
                "delegate subagent was blocked".to_string(),
            ),
            crate::TurnStatus::Cancelled => (
                hi_tools::BackgroundTaskState::Cancelled,
                "delegate subagent was cancelled".to_string(),
            ),
            _ => (
                hi_tools::BackgroundTaskState::Failed,
                child
                    .last_assistant_text()
                    .unwrap_or_else(|| "delegate subagent failed".to_string()),
            ),
        },
        Err(e) => (
            hi_tools::BackgroundTaskState::Failed,
            format!("delegate subagent error: {e}"),
        ),
    };

    let _ = child.kill_background_processes();

    // If a verify command was provided, run it.
    let (final_state, final_output) = if let Some(verify_cmd) = verify {
        if state == hi_tools::BackgroundTaskState::Completed {
            match hi_tools::run_check_in(child.runtime.root(), &verify_cmd).await {
                Ok(exec) if exec.status == hi_tools::ToolStatus::Succeeded => (
                    state,
                    format!("{output}\n\nVerification passed: {verify_cmd}"),
                ),
                Ok(exec) => (
                    hi_tools::BackgroundTaskState::Failed,
                    format!(
                        "{output}\n\nVerification failed: {verify_cmd}\n{}",
                        exec.outcome.stdout_summary
                    ),
                ),
                Err(e) => (
                    hi_tools::BackgroundTaskState::Failed,
                    format!("{output}\n\nVerification error: {e}"),
                ),
            }
        } else {
            (state, output)
        }
    } else {
        (state, output)
    };

    hi_tools::BackgroundTaskOutcome {
        id: String::new(),
        description: String::new(),
        subagent_type: "delegate".into(),
        state: final_state,
        output: final_output,
        applied: final_state == hi_tools::BackgroundTaskState::Completed,
        changed_files: vec![],
    }
}
