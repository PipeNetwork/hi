//! Background subagent task handlers: `task`, `get_task_output`, `wait_tasks`,
//! `kill_task`.
//!
//! The `task` tool spawns a background subagent as a detached Tokio task that
//! runs independently of the parent turn. It returns immediately with a task
//! handle. The parent polls results with `get_task_output`, waits with
//! `wait_tasks`, and cancels with `kill_task`.
//!
//! Built-in kinds match grok-build's task catalog:
//! - `explore` — fast read-only codebase investigation
//! - `plan` — read-only architecture / implementation planning
//! - `general-purpose` — full write-capable multi-step work
//!
//! `delegate` is accepted as a legacy alias for `general-purpose`.
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

/// Canonical background task kinds (grok-build naming).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BgTaskKind {
    Explore,
    Plan,
    GeneralPurpose,
}

impl BgTaskKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Explore => "explore",
            Self::Plan => "plan",
            Self::GeneralPurpose => "general-purpose",
        }
    }

    fn is_read_only(self) -> bool {
        matches!(self, Self::Explore | Self::Plan)
    }

    /// Parse model-supplied `subagent_type`, accepting legacy aliases.
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "explore" => Some(Self::Explore),
            "plan" => Some(Self::Plan),
            // Write-capable: grok-build name + hi's older label.
            "general-purpose" | "general_purpose" | "generalpurpose" | "delegate" | "code" => {
                Some(Self::GeneralPurpose)
            }
            // Common harness synonym for read-only review passes.
            "review" => Some(Self::Explore),
            _ => None,
        }
    }
}

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
    /// Parses `description`, `prompt`, `subagent_type` (`explore` / `plan` /
    /// `general-purpose`, plus legacy aliases), and optional `verify` (for
    /// write-capable kinds). Spawns the subagent as a detached Tokio task and
    /// returns immediately with the task ID.
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

        let raw_type = parsed
            .get("subagent_type")
            .and_then(Value::as_str)
            .unwrap_or("explore");
        let Some(kind) = BgTaskKind::parse(raw_type) else {
            return bg_tool_outcome(
                format!(
                    "task error: unknown subagent_type \"{raw_type}\" — use explore, plan, or general-purpose"
                ),
                hi_tools::ToolStatus::Failed,
            );
        };
        let subagent_type = kind.as_str().to_string();
        let cost = parsed
            .get("cost")
            .and_then(Value::as_str)
            .unwrap_or("normal");
        let dependency_values = parsed.get("depends_on").and_then(Value::as_array);
        let dependencies: Vec<String> = dependency_values
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if dependency_values.is_some_and(|values| values.len() != dependencies.len()) {
            return bg_tool_outcome(
                "task error: depends_on must contain only task ID strings",
                hi_tools::ToolStatus::Failed,
            );
        }
        let prompt = if !kind.is_read_only() && cost == "tiny" {
            format!(
                "Complete this tiny task as one cohesive job, including any closely related cleanup needed to verify the result:\n\n{prompt}"
            )
        } else {
            prompt
        };

        let verify = parsed
            .get("verify")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string);

        // Budget check — read-only kinds share the explore counter; write kinds
        // share the delegate counter.
        let is_read_only = kind.is_read_only();
        let slot = if is_read_only {
            self.subagents
                .try_begin_explore(crate::agent::explore_turn::MAX_EXPLORE_SUBAGENTS_PER_TURN)
        } else {
            self.subagents
                .try_begin_delegate(crate::agent::delegate_turn::delegate_turn_limit())
        };
        let Some(n) = slot else {
            let max = if is_read_only {
                crate::agent::explore_turn::MAX_EXPLORE_SUBAGENTS_PER_TURN
            } else {
                crate::agent::delegate_turn::delegate_turn_limit()
            };
            return bg_tool_outcome(
                format!("task error: {subagent_type} subagent budget exhausted ({max} this turn)"),
                hi_tools::ToolStatus::Denied,
            );
        };

        // Write-capable kinds need a delegate runner.
        if !is_read_only && self.subagents.delegate_runner.is_none() {
            self.subagents.release_delegate();
            return bg_tool_outcome(
                "task error: no delegate runner attached — write-capable background subagents are unavailable",
                hi_tools::ToolStatus::Denied,
            );
        }

        // UI callout — short harness-style label: "↳ explore: Review crate boundaries".
        let summary: String = description.chars().take(72).collect();
        ui.subagent_note(&format!("↳ {subagent_type}: {summary}"));

        // Build the future factory and spawn the task. Each role runs on its
        // configured route (team roles): explore/delegate children may use a
        // different model or endpoint than the driver.
        let provider = if is_read_only {
            self.explore_child_provider()
        } else {
            self.delegate_child_provider()
        };
        let child_config = if is_read_only {
            self.build_bg_explore_config(n, kind)
        } else {
            self.build_bg_delegate_config(n)
        };

        // The future factory is `Send` (a closure), but the future it produces
        // does NOT need to be `Send` — it runs on a worker thread's `LocalSet`.
        let prompt_for_factory = prompt.clone();
        let verify_for_factory = verify.clone();
        let factory: Box<dyn FnOnce() -> hi_tools::BgFuture + Send + 'static> = if is_read_only {
            Box::new(move || {
                Box::pin(run_bg_readonly(
                    provider,
                    child_config,
                    kind,
                    prompt_for_factory,
                ))
            })
        } else {
            Box::new(move || {
                Box::pin(run_bg_general_purpose(
                    provider,
                    child_config,
                    prompt_for_factory,
                    verify_for_factory,
                ))
            })
        };

        let task_id = match self
            .bg_tasks
            .spawn_after(&description, &subagent_type, &dependencies, factory)
            .await
        {
            Ok(id) => id,
            Err(e) => {
                if is_read_only {
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

        // The full text is model-facing protocol (how to poll); the UI only
        // needs the short kind+description — the subagent note already announced it.
        let mut outcome = bg_tool_outcome(
            format!(
                "{subagent_type} task spawned: {task_id}\nDescription: {description}\nPoll results with get_task_output (task_ids: [\"{task_id}\"]) or wait_tasks."
            ),
            hi_tools::ToolStatus::Succeeded,
        );
        outcome.display = Some(format!("{subagent_type}: {summary}"));
        outcome
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

    /// Build a child config for a background read-only subagent (`explore` / `plan`).
    fn build_bg_explore_config(&self, n: u32, kind: BgTaskKind) -> AgentConfig {
        let explore_model = crate::agent::explore_turn::explore_child_model(&self.config);
        let dir_name = match kind {
            BgTaskKind::Plan => format!("bg-plan-{n}"),
            _ => format!("bg-explore-{n}"),
        };
        AgentConfig {
            paths: crate::AgentPaths {
                workspace_root: self.runtime.root().to_path_buf(),
                state_root: self.runtime.state_root().join("subagents").join(dir_name),
            },
            routing: crate::AgentRouting {
                model: explore_model,
                requested_max_tokens: self.config.routing.requested_max_tokens,
                max_tokens: self.config.routing.max_tokens,
                max_tokens_explicit: self.config.routing.max_tokens_explicit,
                temperature: self.config.routing.temperature,
                thinking_budget: self.config.routing.thinking_budget,
                reasoning_effort: self.config.routing.reasoning_effort,
                compat: self.config.routing.compat,
                deepseek_compat: self.config.routing.deepseek_compat,
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
                // Inherit the parent's step setting (off unless the operator
                // capped the session). Hard child ceilings branded finished
                // work "Failed — step limit" in live runs; runaway children
                // are ended by the repeat/no-progress/stall budgets instead.
                max_steps: self.config.loop_limits.max_steps,
                max_parallel_tools: 4,
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
        let delegate_model = self
            .config
            .subagents
            .delegate_model
            .clone()
            .filter(|model| !model.trim().is_empty())
            .unwrap_or_else(|| self.config.routing.model.clone());
        AgentConfig {
            paths: crate::AgentPaths {
                workspace_root: self.runtime.root().to_path_buf(),
                state_root: self
                    .runtime
                    .state_root()
                    .join("subagents")
                    .join(format!("bg-general-purpose-{n}")),
            },
            routing: crate::AgentRouting {
                model: delegate_model,
                requested_max_tokens: self.config.routing.requested_max_tokens,
                max_tokens: self.config.routing.max_tokens,
                max_tokens_explicit: self.config.routing.max_tokens_explicit,
                temperature: self.config.routing.temperature,
                thinking_budget: self.config.routing.thinking_budget,
                reasoning_effort: self.config.routing.reasoning_effort,
                compat: self.config.routing.compat,
                deepseek_compat: self.config.routing.deepseek_compat,
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
                // Inherit the parent's step setting (off unless the operator
                // capped the session). The old hard cap of 20 made every
                // `cost: large` delegate end "Incomplete due to step limit" —
                // work done, verification unrun, outcome branded Failed.
                max_steps: self.config.loop_limits.max_steps,
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
        let state_label = match outcome.state {
            hi_tools::BackgroundTaskState::Running => "Running",
            hi_tools::BackgroundTaskState::Completed => "Completed",
            hi_tools::BackgroundTaskState::Failed => "Failed",
            hi_tools::BackgroundTaskState::Cancelled => "Cancelled",
        };
        // Build "id — {State}: {description} [{type}]", omitting empty pieces so a
        // missing description never renders as a bare "/unknown" fragment.
        let mut header = format!("{} — {}", outcome.id, state_label);
        if !outcome.description.is_empty() {
            header = format!("{}: {}", header, outcome.description);
        }
        if !outcome.subagent_type.is_empty() && outcome.subagent_type != "unknown" {
            header = format!("{} [{}]", header, outcome.subagent_type);
        }
        if outcome.output.is_empty() {
            lines.push(header);
        } else {
            lines.push(format!("{}\n  {}", header, outcome.output));
        }
    }
    lines.join("\n")
}

fn readonly_child_prompt(kind: BgTaskKind, prompt: &str) -> String {
    match kind {
        BgTaskKind::Plan => format!(
            "You are a read-only software architect. Explore the codebase and design an \
             implementation plan. Do not edit files. Cite specific files and locations that \
             support the plan.\n\nTask: {prompt}"
        ),
        // explore (and review alias)
        _ => format!(
            "You are a fast, read-only codebase exploration agent. Read and search the relevant \
             files as needed, then reply with a concise, self-contained answer that cites the \
             specific files and locations supporting it.\n\nQuestion: {prompt}"
        ),
    }
}

/// Run a background read-only subagent (`explore` / `plan`) to completion.
async fn run_bg_readonly(
    provider: std::sync::Arc<dyn hi_ai::Provider>,
    config: AgentConfig,
    kind: BgTaskKind,
    prompt: String,
) -> hi_tools::BackgroundTaskOutcome {
    let kind_label = kind.as_str();
    let child_prompt = readonly_child_prompt(kind, &prompt);

    let child = match crate::Agent::new(provider, config) {
        Ok(c) => c,
        Err(e) => {
            return hi_tools::BackgroundTaskOutcome {
                id: String::new(),
                description: String::new(),
                subagent_type: kind_label.into(),
                state: hi_tools::BackgroundTaskState::Failed,
                output: format!("Failed to create {kind_label} subagent: {e}"),
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
                    .unwrap_or_else(|| format!("{kind_label} subagent produced no answer")),
            ),
            crate::TurnStatus::Blocked => (
                hi_tools::BackgroundTaskState::Failed,
                format!("{kind_label} subagent was blocked"),
            ),
            crate::TurnStatus::Cancelled => (
                hi_tools::BackgroundTaskState::Cancelled,
                format!("{kind_label} subagent was cancelled"),
            ),
            _ => (
                hi_tools::BackgroundTaskState::Failed,
                child
                    .last_assistant_text()
                    .unwrap_or_else(|| format!("{kind_label} subagent failed")),
            ),
        },
        Err(e) => (
            hi_tools::BackgroundTaskState::Failed,
            format!("{kind_label} subagent error: {e}"),
        ),
    };

    child.kill_background_processes();

    hi_tools::BackgroundTaskOutcome {
        id: String::new(),
        description: String::new(),
        subagent_type: kind_label.into(),
        state,
        output,
        applied: false,
        changed_files: vec![],
    }
}

/// Run a background `general-purpose` (write-capable) subagent to completion.
///
/// Changes apply directly to the working tree (no worktree isolation for
/// background tasks — the parent is still working and can observe changes as
/// they happen). If a `verify` command is provided, it's run after the child
/// completes; if it fails, the outcome is marked failed but changes are NOT
/// rolled back (background tasks don't have the same transactional guarantees
/// as synchronous delegate).
async fn run_bg_general_purpose(
    provider: std::sync::Arc<dyn hi_ai::Provider>,
    config: AgentConfig,
    prompt: String,
    verify: Option<String>,
) -> hi_tools::BackgroundTaskOutcome {
    let kind_label = BgTaskKind::GeneralPurpose.as_str();
    let child = match crate::Agent::new(provider, config) {
        Ok(c) => c,
        Err(e) => {
            return hi_tools::BackgroundTaskOutcome {
                id: String::new(),
                description: String::new(),
                subagent_type: kind_label.into(),
                state: hi_tools::BackgroundTaskState::Failed,
                output: format!("Failed to create {kind_label} subagent: {e}"),
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
                    .unwrap_or_else(|| format!("{kind_label} subagent completed")),
            ),
            crate::TurnStatus::Blocked => (
                hi_tools::BackgroundTaskState::Failed,
                format!("{kind_label} subagent was blocked"),
            ),
            crate::TurnStatus::Cancelled => (
                hi_tools::BackgroundTaskState::Cancelled,
                format!("{kind_label} subagent was cancelled"),
            ),
            _ => (
                hi_tools::BackgroundTaskState::Failed,
                child
                    .last_assistant_text()
                    .unwrap_or_else(|| format!("{kind_label} subagent failed")),
            ),
        },
        Err(e) => (
            hi_tools::BackgroundTaskState::Failed,
            format!("{kind_label} subagent error: {e}"),
        ),
    };

    child.kill_background_processes();

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
        subagent_type: kind_label.into(),
        state: final_state,
        output: final_output,
        applied: final_state == hi_tools::BackgroundTaskState::Completed,
        changed_files: vec![],
    }
}

#[cfg(test)]
mod format_tests {
    use super::{BgTaskKind, format_task_results};
    use hi_tools::{BackgroundTaskOutcome, BackgroundTaskState};

    fn outcome(
        id: &str,
        desc: &str,
        kind: &str,
        state: BackgroundTaskState,
        output: &str,
    ) -> BackgroundTaskOutcome {
        BackgroundTaskOutcome {
            id: id.into(),
            description: desc.into(),
            subagent_type: kind.into(),
            state,
            output: output.into(),
            applied: false,
            changed_files: Vec::new(),
        }
    }

    #[test]
    fn empty_results_say_no_tasks() {
        assert_eq!(format_task_results(&[]), "No tasks found.");
    }

    #[test]
    fn running_task_with_description_renders_compactly() {
        let r = format_task_results(&[outcome(
            "task_1",
            "find user type",
            "explore",
            BackgroundTaskState::Running,
            "",
        )]);
        assert_eq!(r, "task_1 — Running: find user type [explore]");
    }

    #[test]
    fn missing_description_does_not_render_bare_slash_fragment() {
        // The old format produced "Task task_2 (/unknown) — Running" for a
        // not-found task. The new format omits empty pieces.
        let r = format_task_results(&[outcome(
            "task_2",
            "",
            "unknown",
            BackgroundTaskState::Running,
            "",
        )]);
        assert_eq!(r, "task_2 — Running");
        assert!(!r.contains("/unknown"));
        assert!(!r.contains("(/"));
    }

    #[test]
    fn completed_task_with_output_indents_output() {
        let r = format_task_results(&[outcome(
            "task_3",
            "scan deps",
            "general-purpose",
            BackgroundTaskState::Completed,
            "found 3 issues",
        )]);
        assert_eq!(
            r,
            "task_3 — Completed: scan deps [general-purpose]\n  found 3 issues"
        );
    }

    #[test]
    fn bg_task_kind_parses_grok_names_and_aliases() {
        assert_eq!(BgTaskKind::parse("explore"), Some(BgTaskKind::Explore));
        assert_eq!(BgTaskKind::parse("plan"), Some(BgTaskKind::Plan));
        assert_eq!(
            BgTaskKind::parse("general-purpose"),
            Some(BgTaskKind::GeneralPurpose)
        );
        // Legacy / harness aliases.
        assert_eq!(
            BgTaskKind::parse("delegate"),
            Some(BgTaskKind::GeneralPurpose)
        );
        assert_eq!(BgTaskKind::parse("code"), Some(BgTaskKind::GeneralPurpose));
        assert_eq!(BgTaskKind::parse("review"), Some(BgTaskKind::Explore));
        assert_eq!(BgTaskKind::parse(""), Some(BgTaskKind::Explore));
        assert_eq!(BgTaskKind::parse("unknown-kind"), None);
    }

    #[test]
    fn multiple_results_join_with_single_newline_no_blank_lines() {
        let r = format_task_results(&[
            outcome("task_4", "a", "explore", BackgroundTaskState::Running, ""),
            outcome("task_5", "b", "explore", BackgroundTaskState::Running, ""),
        ]);
        assert_eq!(
            r,
            "task_4 — Running: a [explore]\ntask_5 — Running: b [explore]"
        );
        assert!(!r.contains("\n\n"));
    }
}
