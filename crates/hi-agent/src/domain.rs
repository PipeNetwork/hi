//! Domain-scoped holders for cross-cutting `Agent` state.
//!
//! Root `Agent` composes these so session/runtime/goals/report/workspace concerns
//! stay separated. Plan/snapshot mutations go through [`GoalState`] methods; other
//! holders expose fields for hot turn-loop projection. Cross-domain reads happen
//! at the composition layer (`Agent` methods), not inside holders.

use hi_ai::Usage;
use hi_tools::{PlanStatus, PlanStep};

use crate::TurnTelemetry;
use crate::agent::turn::TurnPhase;
use crate::goal::Goal;
use crate::heuristics::{leftover_plan_summary, plan_has_pending_steps};
use crate::outcome::{EffectiveModelRoute, TurnOutcome};
use crate::subagent::DelegateRunner;
use crate::task_contract::TaskContract;
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

    /// Title of the first active, else pending, **checklist** step.
    pub(crate) fn next_checklist_step_title(&self) -> Option<&str> {
        self.last_plan
            .iter()
            .find(|step| step.status == PlanStatus::Active)
            .or_else(|| {
                self.last_plan
                    .iter()
                    .find(|step| step.status == PlanStatus::Pending)
            })
            .map(|step| step.title.as_str())
    }

    /// Title of the active plan/goal step, else the first pending one.
    pub(crate) fn next_step_title(&self) -> Option<&str> {
        if let Some(active) = self
            .structured
            .as_ref()
            .and_then(crate::goal::Goal::active_sub_goal)
        {
            return Some(active.description.as_str());
        }
        self.next_checklist_step_title()
    }

    /// Checklist leftover only, e.g. `3/9 remaining — wire the scheduler`.
    pub(crate) fn plan_leftover_work(&self) -> Option<String> {
        leftover_plan_summary(&self.last_plan)
    }

    /// Structured-goal leftover only.
    pub(crate) fn goal_leftover_work(&self) -> Option<String> {
        let goal = self.structured.as_ref()?;
        let remaining = goal
            .sub_goals
            .iter()
            .filter(|step| step.status != crate::goal::GoalStatus::Done)
            .count();
        if remaining == 0 {
            return None;
        }
        let title = self.next_step_title()?;
        Some(format!(
            "{remaining}/{} remaining — {title}",
            goal.sub_goals.len()
        ))
    }

    /// User-facing leftover: goal line if that goal would auto-drive, else plan.
    pub(crate) fn leftover_work(&self) -> Option<String> {
        if self
            .structured
            .as_ref()
            .is_some_and(crate::goal::Goal::should_auto_drive)
        {
            return self.goal_leftover_work();
        }
        self.plan_leftover_work()
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

    /// Choose which plan to keep when discarding a turn.
    ///
    /// Interrupt/`/retry` rewinds must not roll checklist progress backward: if
    /// the abandoned turn already advanced `update_plan` (including finishing
    /// every step), that progress is what the user sees and what a restarted
    /// prompt should build on. A completed live plan is kept in memory so the
    /// UI can show finished; callers persist it as cleared so resume does not
    /// resurrect a done checklist.
    pub(crate) fn prefer_plan_progress(snapshot: &[PlanStep], live: &[PlanStep]) -> Vec<PlanStep> {
        if live.is_empty() {
            return snapshot.to_vec();
        }
        // Live completion wins even when the pre-turn snapshot was incomplete —
        // otherwise Esc after the final update_plan reverts "all done" to N-1.
        if !plan_has_pending_steps(live) {
            return live.to_vec();
        }
        if snapshot.is_empty() {
            return live.to_vec();
        }
        let live_done = live.iter().filter(|s| s.status == PlanStatus::Done).count();
        let snap_done = snapshot
            .iter()
            .filter(|s| s.status == PlanStatus::Done)
            .count();
        if live_done > snap_done || (live_done == snap_done && live.len() >= snapshot.len()) {
            live.to_vec()
        } else {
            snapshot.to_vec()
        }
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
        self.free_text = normalize_free_text_goal(free_text);
        self.structured = structured;
        self.last_plan = last_plan;
    }

    /// Set or clear the transient free-text goal (trim; empty → `None`).
    pub(crate) fn set_free_text(&mut self, goal: Option<String>) {
        self.free_text = normalize_free_text_goal(goal);
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

const MAX_FREE_TEXT_GOAL_CHARS: usize = 500;

fn normalize_free_text_goal(goal: Option<String>) -> Option<String> {
    goal.and_then(|g| {
        let g = g.trim();
        if g.is_empty() {
            return None;
        }
        if g.chars().count() <= MAX_FREE_TEXT_GOAL_CHARS {
            return Some(g.to_string());
        }
        let clipped: String = g
            .chars()
            .take(MAX_FREE_TEXT_GOAL_CHARS.saturating_sub(1))
            .collect();
        Some(format!("{clipped}…"))
    })
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
    /// Failure shape of the findings-ledger steering hint currently in the
    /// memory context, if any — stamped onto findings so hint efficacy is
    /// measurable.
    pub(crate) active_hint_shape: Option<String>,
    /// Latest user/goal task text used for memory ranking.
    pub(crate) last_task_prompt: Option<String>,
    pub(crate) last_task_contract: Option<TaskContract>,
}

/// Post-turn report surface: usage, verify, telemetry, phase, route.
/// The verification verdict for a turn fused with the workspace evidence it is
/// bound to. A `Passed` verdict cannot exist without its bound
/// `(ledger_revision, workspace_digest)` — the pairing invariant the
/// settlement/classify logic used to hold by convention is now enforced by
/// construction: the only way to produce `Passed` is [`VerifyEvidence::pass`],
/// which requires the evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum VerifyEvidence {
    /// No verification ran this turn (or was cleared by a cancel/fail finalizer).
    None,
    /// A verification stage ran and failed.
    Failed,
    /// A verification stage ran and passed, bound to the workspace ledger
    /// revision and digest at the moment it passed.
    Passed { revision: u64, digest: String },
}

impl VerifyEvidence {
    /// The only constructor for `Passed` — requires the bound evidence, so a
    /// pass can never exist without a revision/digest to vouch for it.
    pub(crate) fn pass(revision: u64, digest: String) -> Self {
        Self::Passed { revision, digest }
    }

    pub(crate) fn fail() -> Self {
        Self::Failed
    }

    pub(crate) fn none() -> Self {
        Self::None
    }

    /// `true` iff a verification stage passed (and therefore carries evidence).
    pub(crate) fn passed(&self) -> bool {
        matches!(self, Self::Passed { .. })
    }

    /// `true` iff a verification stage ran and failed.
    pub(crate) fn failed(&self) -> bool {
        matches!(self, Self::Failed)
    }

    /// Map to the historical `Option<bool>` verdict shape: `Some(true)` for
    /// Passed, `Some(false)` for Failed, `None` for no verification. Preserves
    /// the read semantics of the old `last_verify: Option<bool>` field.
    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Passed { .. } => Some(true),
            Self::Failed => Some(false),
            Self::None => None,
        }
    }

    /// The bound `(revision, digest)` evidence, or `None` when not Passed.
    /// Replaces reads of the old `turn.verified_at` field.
    pub(crate) fn bound_revision_digest(&self) -> Option<(u64, String)> {
        match self {
            Self::Passed { revision, digest } => Some((*revision, digest.clone())),
            _ => None,
        }
    }

    /// The bound workspace digest, or `None` when not Passed.
    pub(crate) fn digest(&self) -> Option<&str> {
        match self {
            Self::Passed { digest, .. } => Some(digest),
            _ => None,
        }
    }

    /// Reset to no verification (cancel/fail finalizers, settlement invalidation).
    pub(crate) fn clear(&mut self) {
        *self = Self::None;
    }
}

impl Default for VerifyEvidence {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TurnReportState {
    pub(crate) last_turn_usage: Usage,
    pub(crate) last_user_prompt_tokens: u64,
    pub(crate) verify: VerifyEvidence,
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
            verify: VerifyEvidence::None,
            context_used: 0,
            last_compat_fallbacks: Vec::new(),
            last_turn_telemetry: TurnTelemetry::default(),
            last_turn_outcome: None,
            turn_phase: TurnPhase::Setup,
            last_effective_route: route,
        }
    }

    /// Stamp the typed outcome and keep the effective route in sync.
    pub(crate) fn set_outcome(&mut self, outcome: TurnOutcome) {
        self.last_effective_route = outcome.effective_route.clone();
        self.last_turn_outcome = Some(outcome);
    }

    /// Clear verification evidence (cancel/fail finalizers, settlement
    /// invalidation). The verdict and its bound revision/digest are fused in
    /// [`VerifyEvidence`], so this clears both atomically — a `Passed` verdict
    /// can no longer survive without its evidence.
    pub(crate) fn clear_verify(&mut self) {
        self.verify.clear();
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

    /// Install ledger-derived change lists for the last turn.
    pub(crate) fn record_changes(
        &mut self,
        changes: Vec<hi_tools::FileChange>,
        clear_verify: bool,
    ) {
        self.last_changed_files = changes.iter().map(|c| c.path.clone()).collect();
        self.last_file_changes = changes;
        let _ = clear_verify; // verify clear lives on TurnReportState; callers pair both.
    }

    /// Clear per-turn diff/stub caches at turn start.
    pub(crate) fn clear_turn_caches(&mut self) {
        self.turn_diff_cache = None;
        self.turn_stub_scan_cache = None;
    }

    /// Begin a turn: record ledger + background baselines.
    pub(crate) fn begin_turn(&mut self, ledger_revision: u64, background_ids: Vec<String>) {
        self.active_turn_ledger_revision = Some(ledger_revision);
        self.active_turn_message_start = None;
        self.active_turn_background_baseline = Some(background_ids);
        self.clear_turn_caches();
    }

    /// Mark the transcript index of the user message that opened this turn.
    pub(crate) fn set_message_start(&mut self, start: usize) {
        self.active_turn_message_start = Some(start);
    }
}

impl TaskContextState {
    /// Refresh the ranked task context string when it changed.
    pub(crate) fn set_task_context(&mut self, context: Option<String>) {
        self.task_context = context;
    }

    /// Store the latest task prompt + derived contract.
    pub(crate) fn set_task(&mut self, prompt: Option<String>, contract: Option<TaskContract>) {
        self.last_task_prompt = prompt;
        self.last_task_contract = contract;
    }

    /// Refresh live memory injection text.
    pub(crate) fn set_memory_context(&mut self, context: Option<String>) {
        self.memory_context = context;
    }
}

impl SubagentSessionState {
    /// Reset the per-turn subagent budgets. Called at turn start: the caps
    /// guard against within-turn runaway delegation, so a session that runs
    /// many turns must not starve — each new turn refills the budget while
    /// the lifetime counters keep slot numbers (and child state dirs) unique
    /// across turns, including background tasks that outlive their turn.
    pub(crate) fn begin_turn(&mut self) {
        self.explore_turn_used = 0;
        self.delegate_turn_used = 0;
    }

    /// Try to consume one explore slot; returns the 1-based lifetime slot
    /// number or `None` if this turn's budget is exhausted.
    pub(crate) fn try_begin_explore(&mut self, max: u32) -> Option<u32> {
        if self.explore_turn_used >= max {
            return None;
        }
        self.explore_turn_used += 1;
        self.explore_subagents_used += 1;
        Some(self.explore_subagents_used)
    }

    /// Return an explore slot when startup failed before a child could run.
    pub(crate) fn release_explore(&mut self) {
        self.explore_turn_used = self.explore_turn_used.saturating_sub(1);
        self.explore_subagents_used = self.explore_subagents_used.saturating_sub(1);
    }

    /// Return a delegate slot when startup failed before a child could run.
    pub(crate) fn release_delegate(&mut self) {
        self.delegate_turn_used = self.delegate_turn_used.saturating_sub(1);
        self.delegate_subagents_used = self.delegate_subagents_used.saturating_sub(1);
    }

    /// Try to consume one delegate slot; returns the 1-based lifetime slot
    /// number or `None` if this turn's budget is exhausted.
    pub(crate) fn try_begin_delegate(&mut self, max: u32) -> Option<u32> {
        if self.delegate_turn_used >= max {
            return None;
        }
        self.delegate_turn_used += 1;
        self.delegate_subagents_used += 1;
        Some(self.delegate_subagents_used)
    }
}

/// Subagent budgets and the optional write-capable runner. Budgets are
/// per-turn (refilled by [`Self::begin_turn`]); the lifetime counters exist
/// for unique slot naming, not budgeting.
#[derive(Default)]
pub(crate) struct SubagentSessionState {
    /// Frontend-supplied runner for the write-capable `delegate` subagent.
    pub(crate) delegate_runner: Option<Arc<dyn DelegateRunner>>,
    /// Count of skills auto-curated this session (verifier-gated).
    pub(crate) auto_skills_written: u32,
    /// Count of coding facts auto-recorded this session (green-verify gate).
    pub(crate) coding_facts_written: u32,
    /// Lifetime count of read-only `explore` subagents run this session.
    pub(crate) explore_subagents_used: u32,
    /// Lifetime count of write-capable `delegate` subagents run this session.
    pub(crate) delegate_subagents_used: u32,
    /// `explore` subagents consumed this turn (budget counter).
    pub(crate) explore_turn_used: u32,
    /// `delegate` subagents consumed this turn (budget counter).
    pub(crate) delegate_turn_used: u32,
}

/// Per-turn control flags shared across Model / Tools / Steer.
///
/// Not stored on [`crate::Agent`] — constructed at turn start and passed through
/// Prompt-cache health: compares each model request's message list against
/// the previous request's. A provider prompt cache (Anthropic breakpoints or
/// OpenAI-style implicit prefix caching) can only reuse the unchanged prefix,
/// so an append-only request is a cache hit and any rewrite of an
/// already-sent message is a break — measured here so `hi metrics` can see
/// prefix-cache health instead of guessing from bills.
#[derive(Clone, Debug, Default)]
pub(crate) struct PrefixStability {
    /// Per-message content hashes of the most recent request sent.
    prev_hashes: Vec<u64>,
    /// This turn's append-only request count.
    pub(crate) stable_rounds: u32,
    /// This turn's prefix-breaking request count.
    pub(crate) break_rounds: u32,
    /// Smallest divergence index observed this turn (0 = system message).
    pub(crate) earliest_break: Option<u32>,
}

impl PrefixStability {
    /// Reset the per-turn counters; the previous request's hashes persist so
    /// the first request of a new turn is measured against the last request
    /// of the previous one (turn boundaries are where the context block
    /// strip legitimately breaks the prefix once).
    pub(crate) fn begin_turn(&mut self) {
        self.stable_rounds = 0;
        self.break_rounds = 0;
        self.earliest_break = None;
    }

    /// Record the request about to be sent. Returns nothing; counters are
    /// read into turn telemetry at turn end.
    pub(crate) fn record_request(&mut self, messages: &[hi_ai::Message]) {
        let hashes = message_hashes(messages);
        if !self.prev_hashes.is_empty() {
            let shared = self.prev_hashes.len().min(hashes.len());
            let divergence = (0..shared)
                .find(|&i| self.prev_hashes[i] != hashes[i])
                .or_else(|| (hashes.len() < self.prev_hashes.len()).then_some(hashes.len()));
            match divergence {
                None => self.stable_rounds = self.stable_rounds.saturating_add(1),
                Some(index) => {
                    self.break_rounds = self.break_rounds.saturating_add(1);
                    let index = index as u32;
                    self.earliest_break = Some(match self.earliest_break {
                        Some(previous) => previous.min(index),
                        None => index,
                    });
                }
            }
        }
        self.prev_hashes = hashes;
    }
}

fn message_hashes(messages: &[hi_ai::Message]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    messages
        .iter()
        .map(|message| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::mem::discriminant(&message.role).hash(&mut hasher);
            for block in &message.content {
                match block {
                    hi_ai::Content::Text(text) => {
                        0u8.hash(&mut hasher);
                        text.hash(&mut hasher);
                    }
                    hi_ai::Content::Thinking { text, signature } => {
                        1u8.hash(&mut hasher);
                        text.hash(&mut hasher);
                        signature.hash(&mut hasher);
                    }
                    hi_ai::Content::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        2u8.hash(&mut hasher);
                        id.hash(&mut hasher);
                        name.hash(&mut hasher);
                        arguments.hash(&mut hasher);
                    }
                    hi_ai::Content::ToolResult { call_id, output } => {
                        3u8.hash(&mut hasher);
                        call_id.hash(&mut hasher);
                        output.hash(&mut hasher);
                    }
                    hi_ai::Content::Image { data, media_type } => {
                        4u8.hash(&mut hasher);
                        data.hash(&mut hasher);
                        media_type.hash(&mut hasher);
                    }
                }
            }
            hasher.finish()
        })
        .collect()
}

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
    /// Whether the turn stopped starting new work because its soft wall-clock
    /// deadline expired (see `AgentLoopLimits::turn_soft_deadline`).
    pub ended_at_deadline: bool,
    /// Whether this turn already granted the one tool-free wrap-up round after
    /// reaching the step cap. Sticky: the next cap hit ends the turn for real.
    pub cap_wrap_up_requested: bool,
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

#[cfg(test)]
mod plan_progress_tests {
    use super::GoalState;
    use hi_tools::{PlanStatus, PlanStep};

    fn step(title: &str, status: PlanStatus) -> PlanStep {
        PlanStep {
            title: title.into(),
            status,
        }
    }

    #[test]
    fn interrupt_keeps_live_completion_over_stale_snapshot() {
        let snapshot = vec![
            step("a", PlanStatus::Done),
            step("b", PlanStatus::Active),
            step("c", PlanStatus::Pending),
        ];
        let live = vec![
            step("a", PlanStatus::Done),
            step("b", PlanStatus::Done),
            step("c", PlanStatus::Done),
        ];
        let kept = GoalState::prefer_plan_progress(&snapshot, &live);
        assert!(
            kept.iter().all(|s| s.status == PlanStatus::Done),
            "finished live plan must not roll back on interrupt: {kept:?}"
        );
    }

    #[test]
    fn interrupt_keeps_advanced_incomplete_progress() {
        let snapshot = vec![
            step("a", PlanStatus::Active),
            step("b", PlanStatus::Pending),
        ];
        let live = vec![step("a", PlanStatus::Done), step("b", PlanStatus::Active)];
        let kept = GoalState::prefer_plan_progress(&snapshot, &live);
        assert_eq!(kept[0].status, PlanStatus::Done);
        assert_eq!(kept[1].status, PlanStatus::Active);
    }

    #[test]
    fn empty_live_falls_back_to_snapshot() {
        let snapshot = vec![step("a", PlanStatus::Pending)];
        let kept = GoalState::prefer_plan_progress(&snapshot, &[]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].title, "a");
    }
}

#[cfg(test)]
mod prefix_stability_tests {
    use super::PrefixStability;
    use hi_ai::{Content, Message};

    #[test]
    fn counts_append_only_requests_and_prefix_breaks() {
        let mut tracker = PrefixStability::default();
        let first = vec![Message::system("sys"), Message::user("question")];
        tracker.record_request(&first);
        assert_eq!(
            tracker.stable_rounds + tracker.break_rounds,
            0,
            "the first request has no predecessor to compare against"
        );

        let mut appended = first.clone();
        appended.push(Message::assistant(vec![Content::Text("answer".into())]));
        tracker.record_request(&appended);
        assert_eq!(tracker.stable_rounds, 1, "pure append is cache-friendly");
        assert_eq!(tracker.break_rounds, 0);

        let mut rewritten = appended.clone();
        rewritten[0] = Message::system("sys CHANGED");
        tracker.record_request(&rewritten);
        assert_eq!(tracker.break_rounds, 1);
        assert_eq!(
            tracker.earliest_break,
            Some(0),
            "a system rewrite breaks the prefix at message 0"
        );

        // Truncation (messages dropped) breaks at the new, shorter length.
        let truncated = rewritten[..2].to_vec();
        tracker.record_request(&truncated);
        assert_eq!(tracker.break_rounds, 2);
        assert_eq!(tracker.earliest_break, Some(0), "keeps the minimum");

        tracker.begin_turn();
        assert_eq!(tracker.stable_rounds, 0);
        assert_eq!(tracker.earliest_break, None);
        let mut next_turn = truncated.clone();
        next_turn.push(Message::user("next"));
        tracker.record_request(&next_turn);
        assert_eq!(
            tracker.stable_rounds, 1,
            "previous hashes survive the per-turn counter reset"
        );
    }
}
