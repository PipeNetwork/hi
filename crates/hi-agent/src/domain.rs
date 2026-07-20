//! Domain-scoped holders for cross-cutting `Agent` state.
//!
//! These keep goals/plans and RSI observe-only fields grouped so the root
//! `Agent` surface reads as composition rather than a flat bag of peers.
//! Access remains `pub(crate)` field projection (`agent.goals.last_plan`) so
//! existing call sites stay direct without a large accessor layer.

use hi_tools::PlanStep;

use crate::goal::Goal;

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
