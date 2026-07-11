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

/// Cap on `explore` subagents per session, to bound cost if the model over-delegates.
pub(crate) const MAX_EXPLORE_SUBAGENTS_PER_SESSION: u32 = 8;

/// Step budget for one child explore turn.
const EXPLORE_MAX_STEPS: u32 = 20;

impl crate::Agent {
    /// Run one read-only `explore` subagent for the `{task}` argument and return
    /// its answer as the tool result. Best-effort: a provider/parse error becomes
    /// an error string fed back to the model, never fatal to the parent turn.
    pub(crate) async fn handle_explore(&mut self, arguments: &str, ui: &mut dyn Ui) -> String {
        let task = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| v.get("task").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default();
        if task.trim().is_empty() {
            return "explore error: missing required \"task\" argument".to_string();
        }
        if self.explore_subagents_used >= MAX_EXPLORE_SUBAGENTS_PER_SESSION {
            return format!(
                "explore budget exhausted ({MAX_EXPLORE_SUBAGENTS_PER_SESSION} subagents this \
                 session); investigate directly instead."
            );
        }
        self.explore_subagents_used += 1;
        let n = self.explore_subagents_used;
        // Prominent callout so the user clearly sees a nested agent run start.
        let summary: String = task.chars().take(72).collect();
        let ellipsis = if task.chars().count() > 72 { "…" } else { "" };
        ui.subagent_note(&format!(
            "↳ explore subagent {n}/{MAX_EXPLORE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}"
        ));

        // A read-only, bounded child config derived from the parent's model/token
        // settings. `explore_subagents = false` + `ToolMode::ReadOnly` cap depth at 1.
        let child_config = AgentConfig {
            model: self.config.model.clone(),
            requested_max_tokens: self.config.requested_max_tokens,
            max_tokens: self.config.max_tokens,
            max_tokens_explicit: self.config.max_tokens_explicit,
            temperature: self.config.temperature,
            thinking_budget: self.config.thinking_budget,
            reasoning_effort: self.config.reasoning_effort,
            compat: self.config.compat,
            context_window: self.config.context_window,
            project_context: self.config.project_context.clone(),
            tool_mode: ToolMode::ReadOnly,
            verify: Vec::new(),
            max_steps: EXPLORE_MAX_STEPS,
            max_steps_explicit: true,
            finalize: false,
            // A read-only explorer's text output IS its answer — don't nudge it to
            // keep going after it stops with text.
            max_silent_continues: 0,
            curate_skills: false,
            explore_subagents: false,
            // Depth guard: a subagent is never advertised `explore`, so it can't
            // spawn another (depth ≤ 1), even in read-only mode.
            is_subagent: true,
            long_horizon: false,
            read_only_preflight: false,
            lsp: false,
            ..AgentConfig::default()
        };

        // Share the provider (Arc) — same HTTP client / connection pool, no rebuild.
        let mut child = crate::Agent::new(self.provider.clone(), child_config);
        // Scope the child UI (a reborrow of `ui`) so `ui` is free again for the
        // completion callout below. `Box::pin` breaks the async-recursion cycle
        // (`run_turn` → `handle_explore` → child `run_turn`) that would otherwise
        // make the future infinitely sized.
        let result = {
            let mut child_ui = ExploreUi { parent: &mut *ui };
            match Box::pin(child.run_turn(&explore_child_prompt(&task), &mut child_ui)).await {
                Ok(()) => child
                    .last_assistant_text()
                    .unwrap_or_else(|| "(the explore subagent produced no answer)".to_string()),
                Err(err) => format!("explore subagent error: {err}"),
            }
        };
        // Fold the child's token usage into the parent's session totals.
        self.add_side_usage(*child.totals());
        ui.subagent_note(&format!("↳ explore subagent {n} done"));
        result
    }
}

fn explore_child_prompt(task: &str) -> String {
    // Deliberately plain phrasing: the child's read-only restriction is enforced by
    // `ToolMode::ReadOnly` (edit/shell tools simply aren't advertised), so the prompt
    // avoids "read-only / don't edit" wording that would trip the heavier review-intent
    // machinery (which enforces a rigid cite-findings-and-limits answer format).
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
