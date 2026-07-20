//! Domain-scoped holders for cross-cutting `Agent` state.
//!
//! These keep goals/plans and RSI observe-only fields grouped so the root
//! `Agent` surface reads as composition rather than a flat bag of peers.
//! Plan/snapshot mutations go through [`GoalState`] methods; field projection
//! remains available for read-heavy call sites.

use hi_tools::{PlanStatus, PlanStep};

use crate::goal::Goal;
use crate::heuristics::plan_has_pending_steps;

/// Session goal + plan state owned by the interactive agent.
#[derive(Clone, Debug, Default)]
pub(crate) struct GoalState {
    /// Transient free-text goal (prompt injection; not the durable structured goal).
    pub(crate) free_text: Option<String>,
    /// Durable hierarchical goal when long-horizon mode is on.
    pub(crate) structured: Option<Goal>,
    /// Latest `update_plan` steps for incomplete-plan steering.
    pub(crate) last_plan: Vec<PlanStep>,
}

impl GoalState {
    /// Current plan steps (possibly empty).
    pub(crate) fn plan(&self) -> &[PlanStep] {
        &self.last_plan
    }

    /// Whether any plan step is still pending/active.
    pub(crate) fn plan_incomplete(&self) -> bool {
        plan_has_pending_steps(&self.last_plan)
    }

    /// Drop the in-memory plan.
    pub(crate) fn clear_plan(&mut self) {
        self.last_plan.clear();
    }

    /// Clear the plan unless `preserve` is set (e.g. goal-drive / "continue").
    /// Returns whether the plan was cleared.
    pub(crate) fn clear_plan_unless(&mut self, preserve: bool) -> bool {
        if preserve || self.last_plan.is_empty() {
            return false;
        }
        self.clear_plan();
        true
    }

    /// Install a plan only when it still has unfinished work; completed-only
    /// plans are dropped so they don't re-trigger incomplete-plan steering.
    pub(crate) fn set_plan_if_pending(&mut self, plan: Vec<PlanStep>) {
        self.last_plan = if plan.iter().any(|step| step.status != PlanStatus::Done) {
            plan
        } else {
            Vec::new()
        };
    }

    /// Replace the plan from an `update_plan` tool result. Returns whether the
    /// steps actually changed.
    pub(crate) fn replace_plan(&mut self, plan: &[PlanStep]) -> bool {
        let changed = self.last_plan.as_slice() != plan;
        self.last_plan = plan.to_vec();
        changed
    }

    /// Snapshot the triple stored on [`crate::AgentStateSnapshot`] (decisions
    /// stay outside this holder).
    pub(crate) fn snapshot_triple(&self) -> (Option<String>, Option<Goal>, Vec<PlanStep>) {
        (
            self.free_text.clone(),
            self.structured.clone(),
            self.last_plan.clone(),
        )
    }

    /// Restore free-text, structured goal, and plan from a prior snapshot triple.
    pub(crate) fn restore_triple(
        &mut self,
        free_text: Option<String>,
        structured: Option<Goal>,
        last_plan: Vec<PlanStep>,
    ) {
        self.free_text = free_text;
        self.structured = structured;
        self.last_plan = last_plan;
    }

    /// Set or clear the transient free-text goal (trim; empty → `None`).
    pub(crate) fn set_free_text(&mut self, goal: Option<String>) {
        self.free_text = goal.and_then(|g| {
            let g = g.trim().to_string();
            (!g.is_empty()).then_some(g)
        });
    }

    /// Clone the durable structured goal (turn-start baseline for revert).
    pub(crate) fn clone_structured(&self) -> Option<Goal> {
        self.structured.clone()
    }

    /// Replace the durable structured goal.
    pub(crate) fn set_structured(&mut self, goal: Option<Goal>) {
        self.structured = goal;
    }
}

/// Live RSI observation state that is *not* config (`AgentRsi`).
///
/// Interactive code may observe RSI; it must not drive the RSI workflow SM.
#[derive(Clone, Debug, Default)]
pub(crate) struct RsiObserveState {
    /// Frontend observation result for the latest completed turn.
    pub(crate) last_fully_observed: Option<bool>,
    /// Validated worker-provided conversation reference for managed RSI.
    pub(crate) managed_context: Option<String>,
}

impl RsiObserveState {
    /// Record whether the latest turn was fully observed by the frontend.
    pub(crate) fn set_last_fully_observed(&mut self, observed: Option<bool>) {
        self.last_fully_observed = observed;
    }

    /// Install or clear the validated managed-RSI conversation reference.
    pub(crate) fn set_managed_context(&mut self, context: Option<String>) {
        self.managed_context = context.filter(|s| !s.trim().is_empty());
    }

    /// Take the managed context for one-shot injection (clears the slot).
    pub(crate) fn take_managed_context(&mut self) -> Option<String> {
        self.managed_context.take()
    }
}

/// Per-turn control flags shared across Model / Tools / Steer.
///
/// Not stored on [`crate::Agent`] — constructed at turn start and passed through
/// the phase helpers so the turn loop does not grow an ever-longer local list
/// without a name. Field projection keeps call sites direct.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)] // wired into loop in a follow-up
pub(crate) struct TurnControlFlags {
    pub force_tools_next: bool,
    pub text_tool_fallback_next: bool,
    pub force_text_answer_next: bool,
    pub force_no_progress_final_answer_next: bool,
    pub suppress_bookkeeping_tools_next: bool,
    pub made_tool_call: bool,
    pub stalled_repeating: bool,
    pub stalled_unfinished: bool,
    pub ended_at_cap: bool,
    pub obligation_nudge_fired: bool,
}

impl TurnControlFlags {
    /// Clear one-shot force flags that apply only to the next Model request.
    #[allow(dead_code)] // available for loop flag bag adoption
    pub(crate) fn clear_one_shot_forces(&mut self) {
        self.force_tools_next = false;
        self.text_tool_fallback_next = false;
        self.force_text_answer_next = false;
        self.force_no_progress_final_answer_next = false;
        self.suppress_bookkeeping_tools_next = false;
    }
}
