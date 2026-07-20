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

fn explore_tool_outcome(
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

impl crate::Agent {
    /// Run one read-only `explore` subagent for the `{task}` argument and return
    /// its answer as the tool result. Best-effort: a provider/parse error becomes
    /// an error string fed back to the model, never fatal to the parent turn.
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
        if self.subagents.explore_subagents_used >= MAX_EXPLORE_SUBAGENTS_PER_SESSION {
            return explore_tool_outcome(
                format!(
                    "explore budget exhausted ({MAX_EXPLORE_SUBAGENTS_PER_SESSION} subagents this \
                     session); investigate directly instead."
                ),
                hi_tools::ToolStatus::Denied,
            );
        }
        self.subagents.explore_subagents_used += 1;
        let n = self.subagents.explore_subagents_used;
        // Prominent callout so the user clearly sees a nested agent run start.
        let summary: String = task.chars().take(72).collect();
        let ellipsis = if task.chars().count() > 72 { "…" } else { "" };
        ui.subagent_note(&format!(
            "↳ explore subagent {n}/{MAX_EXPLORE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}"
        ));

        // A read-only, bounded child config derived from the parent's model/token
        // settings. `explore_subagents = false` + `ToolMode::ReadOnly` cap depth at 1.
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

        // Share the provider (Arc) — same HTTP client / connection pool, no rebuild.
        let mut child = match crate::Agent::new(self.provider.clone(), child_config) {
            Ok(child) => child,
            Err(error) => {
                return explore_tool_outcome(
                    format!("explore subagent runtime initialization failed: {error:#}"),
                    hi_tools::ToolStatus::Failed,
                );
            }
        };
        // Scope the child UI (a reborrow of `ui`) so `ui` is free again for the
        // completion callout below. `Box::pin` breaks the async-recursion cycle
        // (`run_turn` → `handle_explore` → child `run_turn`) that would otherwise
        // make the future infinitely sized.
        let result = {
            let mut child_ui = ExploreUi { parent: &mut *ui };
            match Box::pin(child.run_turn(&explore_child_prompt(&task), &mut child_ui)).await {
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
                    // Mirror parent failed-turn cleanup: kill child backgrounds and
                    // type an infrastructure outcome so nested escapes don't leak.
                    let _ = child.finalize_failed_turn();
                    explore_tool_outcome(
                        format!("explore subagent error: {err}"),
                        hi_tools::ToolStatus::Failed,
                    )
                }
            }
        };
        // Always tear down child-owned backgrounds (read-only explore should be
        // quiet, but bash-in-readonly denial paths still share the kill API).
        child.kill_background_processes();
        // Fold the child's token usage into the parent's session totals.
        self.add_side_usage(*child.totals());
        ui.subagent_note(&format!("↳ explore subagent {n} done"));
        result
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

/// A `Ui` for a child explore turn: forwards its tool activity and status to the
/// parent UI with an `explore:` prefix so the subagent's work is visible live, and
/// swallows its streamed prose (the final answer returns via the tool result).
struct ExploreUi<'a> {
    parent: &'a mut dyn Ui,
}

impl Ui for ExploreUi<'_> {
    fn assistant_text(&mut self, _text: &str) {}
    fn assistant_reasoning(&mut self, _text: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, name: &str, args: &str) {
        self.parent.tool_call(&format!("explore:{name}"), args);
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.parent.tool_result(&format!("explore:{name}"), result);
    }
    fn status(&mut self, text: &str) {
        self.parent.status(&format!("explore: {text}"));
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
