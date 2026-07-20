//! Domain-scoped holders for cross-cutting `Agent` state.
//!
//! Root `Agent` composes these so session/runtime/goals/report/workspace concerns
//! stay separated. Plan/snapshot mutations go through [`GoalState`] methods; other
//! holders expose fields for hot turn-loop projection. Cross-domain reads happen
//! at the composition layer (`Agent` methods), not inside holders.

use hi_ai::Usage;
use hi_tools::{PlanStatus, PlanStep};

use crate::goal::Goal;
use crate::heuristics::plan_has_pending_steps;
use crate::outcome::{EffectiveModelRoute, TurnOutcome};
use crate::subagent::DelegateRunner;
use crate::task_contract::TaskContract;
use crate::agent::turn::TurnPhase;
use crate::TurnTelemetry;
use std::sync::Arc;

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

/// Per-turn ranked task / memory prompt assembly state.
#[derive(Clone, Debug, Default)]
pub(crate) struct TaskContextState {
    /// Per-turn ranked repository data and scoped instructions.
    pub(crate) task_context: Option<String>,
    /// Live hierarchical memory section (task-ranked).
    pub(crate) memory_context: Option<String>,
    /// Latest user/goal task text used for memory ranking.
    pub(crate) last_task_prompt: Option<String>,
    pub(crate) last_task_contract: Option<TaskContract>,
}

/// Post-turn report surface: usage, verify, telemetry, phase, route.
#[derive(Clone, Debug)]
pub(crate) struct TurnReportState {
    pub(crate) last_turn_usage: Usage,
    pub(crate) last_user_prompt_tokens: u64,
    pub(crate) last_verify: Option<bool>,
    pub(crate) context_used: u64,
    pub(crate) last_compat_fallbacks: Vec<String>,
    pub(crate) last_turn_telemetry: TurnTelemetry,
    pub(crate) last_turn_outcome: Option<TurnOutcome>,
    pub(crate) turn_phase: TurnPhase,
    pub(crate) last_effective_route: EffectiveModelRoute,
}

impl TurnReportState {
    pub(crate) fn new(route: EffectiveModelRoute) -> Self {
        Self {
            last_turn_usage: Usage::default(),
            last_user_prompt_tokens: 0,
            last_verify: None,
            context_used: 0,
            last_compat_fallbacks: Vec::new(),
            last_turn_telemetry: TurnTelemetry::default(),
            last_turn_outcome: None,
            turn_phase: TurnPhase::Setup,
            last_effective_route: route,
        }
    }
}

impl Default for TurnReportState {
    fn default() -> Self {
        Self::new(EffectiveModelRoute {
            provider: None,
            model: String::new(),
        })
    }
}

/// Mutation/undo/reconcile state for the in-flight and last turn.
#[derive(Clone, Debug, Default)]
pub(crate) struct WorkspaceTurnState {
    /// Per-turn git checkpoints (working-tree snapshots), for `/undo`.
    pub(crate) checkpoints: Vec<String>,
    /// Files whose content or presence changed in the most recent turn.
    pub(crate) last_changed_files: Vec<String>,
    /// Structured effects reported by mutating tools in the most recent turn.
    pub(crate) last_file_changes: Vec<hi_tools::FileChange>,
    /// Per-turn cache of the checkpoint diff (`turn_diff`).
    pub(crate) turn_diff_cache: Option<(u64, String)>,
    /// Per-turn cache of the stub scan over changed files.
    pub(crate) turn_stub_scan_cache: Option<(u64, Vec<hi_tools::stub_scan::StubFinding>)>,
    /// Ledger baseline while a turn future is in flight (cancel-safe).
    pub(crate) active_turn_ledger_revision: Option<u64>,
    /// Message-len baseline while a turn future is in flight (cancel-safe).
    pub(crate) active_turn_message_start: Option<usize>,
    /// Background process ids at turn start so failed/cancelled finalizers can
    /// kill only processes this turn started (mirrors frontend cancel cleanup).
    pub(crate) active_turn_background_baseline: Option<Vec<String>>,
}

impl WorkspaceTurnState {
    /// Clear cancel-safe active-turn baselines after a turn settles.
    pub(crate) fn clear_active_baselines(&mut self) {
        self.active_turn_ledger_revision = None;
        self.active_turn_message_start = None;
        self.active_turn_background_baseline = None;
    }
}

/// Session-scoped subagent caps and the optional write-capable runner.
#[derive(Default)]
pub(crate) struct SubagentSessionState {
    /// Frontend-supplied runner for the write-capable `delegate` subagent.
    pub(crate) delegate_runner: Option<Arc<dyn DelegateRunner>>,
    /// Count of skills auto-curated this session (verifier-gated).
    pub(crate) auto_skills_written: u32,
    /// Count of coding facts auto-recorded this session (green-verify gate).
    pub(crate) coding_facts_written: u32,
    /// Count of read-only `explore` subagents run this session.
    pub(crate) explore_subagents_used: u32,
    /// Count of write-capable `delegate` subagents run this session.
    pub(crate) delegate_subagents_used: u32,
}

/// Per-turn control flags shared across Model / Tools / Steer.
///
/// Not stored on [`crate::Agent`] — constructed at turn start and passed through
/// the phase helpers so the turn loop does not grow an ever-longer local list
/// without a name. Field projection keeps call sites direct.
#[derive(Clone, Debug, Default)]
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
    pub(crate) fn clear_one_shot_forces(&mut self) {
        self.force_tools_next = false;
        self.text_tool_fallback_next = false;
        self.force_text_answer_next = false;
        self.force_no_progress_final_answer_next = false;
        self.suppress_bookkeeping_tools_next = false;
    }
}
