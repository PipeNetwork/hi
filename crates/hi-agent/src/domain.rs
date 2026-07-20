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
