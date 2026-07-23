//! Read-only `explore` subagent: delegate a bounded investigation to a child
//! agent that shares the parent's provider (via `Arc`) but runs with read-only
//! tools, its own fresh context, and a small step budget, then returns a concise
//! answer to the parent.
//!
//! Depth is capped at 1: the child is built with `explore_subagents = false`, and
//! because it runs in `ToolMode::ReadOnly` it never sees the (deliberately
//! non-read-only) `explore` tool — so a subagent cannot spawn another.

use hi_ai::ToolMode;
use serde_json::Value;

use crate::AgentConfig;
use crate::Ui;

pub(crate) fn explore_tool_outcome(
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

/// Cap on `explore` subagents per session, to bound cost if the model over-delegates.
pub(crate) const MAX_EXPLORE_SUBAGENTS_PER_SESSION: u32 = 8;

/// Budgets for one child explore turn. Parallel fan-out is intentionally tiny:
/// a model can otherwise emit a dozen reads per round and turn a nominal
/// 20-step child into hundreds of tool calls before the parent can intervene.
const EXPLORE_MAX_STEPS: u32 = 10;
const EXPLORE_MAX_PARALLEL_TOOLS: usize = 2;

/// Maximum number of explore subagents to run concurrently within a single
/// tool batch. Explores are read-only and independent, so they can overlap —
/// but each spawns a child `Agent` (memory + subprocess), so cap the fan-out.
pub(crate) const MAX_PARALLEL_EXPLORES: usize = 3;

/// A prepared-but-not-yet-running explore subagent job. Extracted from the
/// parent `Agent` so the heavy work (child `run_turn`) can run concurrently
/// across multiple explores without holding `&mut self`.
pub(crate) struct ExploreJob {
    pub(crate) slot: u32,
    pub(crate) task: String,
    pub(crate) provider: std::sync::Arc<dyn hi_ai::Provider>,
    pub(crate) child_config: AgentConfig,
}

/// The result of running an explore job — the tool outcome plus the child's
/// token usage (to fold into the parent's totals) and the slot (for budget
/// release on failure).
pub(crate) struct ExploreResult {
    pub(crate) slot: u32,
    pub(crate) outcome: hi_tools::ToolOutcome,
    pub(crate) usage: hi_ai::Usage,
}
impl crate::Agent {
    /// Prepare an explore subagent job: check budget, build the child config,
    /// and extract the provider. Returns `None` if the budget is exhausted or
    /// the task is empty. The returned job owns everything it needs to run
    /// concurrently with other explore jobs.
    pub(crate) fn prepare_explore(&mut self, arguments: &str) -> Option<ExploreJob> {
        let task = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| v.get("task").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();
        if task.trim().is_empty() {
            return None;
        }
        let n = self
            .subagents
            .try_begin_explore(MAX_EXPLORE_SUBAGENTS_PER_SESSION)?;
        let child_config = AgentConfig {
            paths: crate::AgentPaths {
                workspace_root: self.runtime.root().to_path_buf(),
                state_root: self
                    .runtime
                    .state_root()
                    .join("subagents")
                    .join(format!("explore-{n}")),
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
                max_steps: EXPLORE_MAX_STEPS,
                max_steps_explicit: true,
                max_parallel_tools: EXPLORE_MAX_PARALLEL_TOOLS,
                // A read-only explorer's text output IS its answer — don't nudge it to
                // keep going after it stops with text.
                max_silent_continues: 0,
                ..crate::AgentLoopLimits::default()
            },
            memory: crate::AgentMemory {
                project_context: self.config.memory.project_context.clone(),
                finalize: false,
                curate_skills: false,
                ..crate::AgentMemory::default()
            },
            subagents: crate::AgentSubagents {
                explore_subagents: false,
                long_horizon: false,
                // Depth guard: a subagent is never advertised `explore`, so it can't
                // spawn another (depth ≤ 1), even in read-only mode.
                is_subagent: true,
                ..crate::AgentSubagents::default()
            },
            ..AgentConfig::default()
        };
        Some(ExploreJob {
            slot: n,
            task,
            provider: self.provider.clone(),
            child_config,
        })
    }

    /// Run one read-only `explore` subagent for the `{task}` argument and return
    /// its answer as the tool result. Best-effort: a provider/parse error becomes
    /// an error string fed back to the model, never fatal to the parent turn.
    ///
    /// This is the synchronous single-explore path. When multiple explore calls
    /// are ready in the same tool batch, the batch scheduler uses
    /// [`prepare_explore`] + [`run_explore_job`] + [`finish_explore`] to run
    /// them concurrently.
    pub(crate) async fn handle_explore(
        &mut self,
        arguments: &str,
        ui: &mut dyn Ui,
    ) -> hi_tools::ToolOutcome {
        let task = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| v.get("task").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();
        if task.trim().is_empty() {
            return explore_tool_outcome(
                "explore error: missing required \"task\" argument",
                hi_tools::ToolStatus::Failed,
            );
        }
        let Some(job) = self.prepare_explore(arguments) else {
            return explore_tool_outcome(
                format!(
                    "explore budget exhausted ({MAX_EXPLORE_SUBAGENTS_PER_SESSION} subagents this \
                     session); investigate directly instead."
                ),
                hi_tools::ToolStatus::Denied,
            );
        };
        // Prominent callout so the user clearly sees a nested agent run start.
        let summary: String = task.chars().take(72).collect();
        let ellipsis = if task.chars().count() > 72 { "…" } else { "" };
        ui.subagent_note(&format!(
            "↳ explore subagent {}/{MAX_EXPLORE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}",
            job.slot,
        ));

        let result = {
            let mut buf_ui = BufferingUi::new();
            let r = run_explore_job(job, &mut buf_ui).await;
            buf_ui.replay_to(&mut *ui);
            r
        };
        // Fold the child's token usage into the parent's session totals.
        self.add_side_usage(result.usage);
        ui.subagent_note(&format!("↳ explore subagent {} done", result.slot));
        result.outcome
    }

    /// Finish a completed explore job: fold usage into the parent's totals.
    /// Called after parallel explores complete in the batch scheduler.
    pub(crate) fn finish_explore(&mut self, result: ExploreResult) -> hi_tools::ToolOutcome {
        self.add_side_usage(result.usage);
        result.outcome
    }

    /// Release an explore budget slot when the job failed before running
    /// (e.g. child agent initialization error in the parallel path).
    pub(crate) fn release_explore_slot(&mut self) {
        self.subagents.release_explore();
    }
}

/// Run a prepared explore job to completion. This is a free function (not a
/// method on `Agent`) so it can run concurrently across multiple jobs without
/// holding `&mut self`. UI output is buffered into the provided `BufferingUi`
/// and replayed to the real UI by the caller after the job completes — this
/// avoids needing to share `&mut dyn Ui` across concurrent futures.
pub(crate) async fn run_explore_job(
    job: ExploreJob,
    ui: &mut BufferingUi,
) -> ExploreResult {
    let ExploreJob {
        slot,
        task,
        provider,
        child_config,
    } = job;

    let mut child = match crate::Agent::new(provider, child_config) {
        Ok(child) => child,
        Err(error) => {
            return ExploreResult {
                slot,
                outcome: explore_tool_outcome(
                    format!("explore subagent runtime initialization failed: {error:#}"),
                    hi_tools::ToolStatus::Failed,
                ),
                usage: hi_ai::Usage::default(),
            };
        }
    };
    // `Box::pin` breaks the async-recursion cycle (`run_turn` → `handle_explore`
    // → child `run_turn`) that would otherwise make the future infinitely sized.
    // The child writes to the `BufferingUi`; the caller replays to the real UI.
    let outcome = {
        match Box::pin(child.run_turn(&explore_child_prompt(&task), ui)).await {
            Ok(outcome) => {
                let answer = child.last_assistant_text();
                let mut status = match outcome.status {
                    crate::TurnStatus::Completed => hi_tools::ToolStatus::Succeeded,
                    crate::TurnStatus::Blocked => hi_tools::ToolStatus::Denied,
                    crate::TurnStatus::Cancelled => hi_tools::ToolStatus::Cancelled,
                    crate::TurnStatus::Incomplete | crate::TurnStatus::Failed => {
                        hi_tools::ToolStatus::Failed
                    }
                };
                let answer = answer.unwrap_or_else(|| {
                    status = hi_tools::ToolStatus::Failed;
                    "explore subagent produced no answer".to_string()
                });
                explore_tool_outcome(answer, status)
            }
            Err(err) => {
                // Nested escapes: typed fail cleanup (turn-scoped bg kill).
                let _ = child.cleanup_turn(crate::TurnCleanupKind::Fail).await;
                explore_tool_outcome(
                    format!("explore subagent error: {err}"),
                    hi_tools::ToolStatus::Failed,
                )
            }
        }
    };
    // Throwaway child runtime: full kill (local skeptic + any leftover bg).
    child.kill_background_processes();
    let usage = *child.totals();
    ExploreResult {
        slot,
        outcome,
        usage,
    }
}

fn explore_child_prompt(task: &str) -> String {
    // Deliberately plain phrasing: the child's read-only restriction and
    // inspection-sprawl cap come from its task contract and capability scope,
    // not legacy review-intent prompt shaping.
    format!(
        "Answer this question about the codebase. Read and search the relevant files as needed, then \
         reply with a concise, self-contained answer that cites the specific files and locations \
         supporting it.\n\nQuestion: {task}"
    )
}

/// A buffered UI event — collected by `BufferingUi` and replayed later.
enum BufferedUiEvent {
    ToolCall { name: String, args: String },
    ToolResult { name: String, result: String },
    Status { text: String },
}

/// A `Ui` that buffers events into a `Vec` instead of emitting them live.
/// Used by the parallel-explore path: each concurrent explore writes to its
/// own `BufferingUi`, and the batch scheduler replays the events to the real
/// UI sequentially after each explore completes. This avoids sharing
/// `&mut dyn Ui` across concurrent futures.
pub(crate) struct BufferingUi {
    events: Vec<BufferedUiEvent>,
}

impl BufferingUi {
    pub(crate) fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Drain and replay all buffered events to the real UI, prefixing tool
    /// names with `explore:` and status text with `explore: ` — matching the
    /// `ExploreUi` forwarding convention.
    pub(crate) fn replay_to(&mut self, ui: &mut dyn Ui) {
        for event in self.events.drain(..) {
            match event {
                BufferedUiEvent::ToolCall { name, args } => {
                    ui.tool_call(&format!("explore:{name}"), &args);
                }
                BufferedUiEvent::ToolResult { name, result } => {
                    ui.tool_result(&format!("explore:{name}"), &result);
                }
                BufferedUiEvent::Status { text } => {
                    ui.status(&format!("explore: {text}"));
                }
            }
        }
    }
}

impl Ui for BufferingUi {
    fn assistant_text(&mut self, _text: &str) {}
    fn assistant_reasoning(&mut self, _text: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, name: &str, args: &str) {
        self.events.push(BufferedUiEvent::ToolCall {
            name: name.to_string(),
            args: args.to_string(),
        });
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.events.push(BufferedUiEvent::ToolResult {
            name: name.to_string(),
            result: result.to_string(),
        });
    }
    fn status(&mut self, text: &str) {
        self.events.push(BufferedUiEvent::Status {
            text: text.to_string(),
        });
    }
    fn turn_end(&mut self, _summary: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_prompt_stays_plain_but_has_a_read_only_task_contract() {
        let prompt = explore_child_prompt("count the Rust source lines");
        assert!(crate::steering::classify_read_only_intent(&prompt).is_none());
        assert_eq!(
            crate::TaskContract::derive(&prompt, crate::VerificationMode::Disabled).intent,
            crate::TaskIntent::ReadOnly
        );
        assert_eq!(EXPLORE_MAX_STEPS, 10);
        assert_eq!(EXPLORE_MAX_PARALLEL_TOOLS, 2);
    }
}
