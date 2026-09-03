//! Owned per-turn state bag for the interactive loop.
//!
//! Built once in `run_turn_body` and passed through Model / Tools / Steer /
//! WorkspaceRepair instead of a long list of locals + borrow bags.

use std::collections::BTreeSet;

use crate::agent::turn::fast_feedback::FastFeedbackState;
use crate::agent::turn::progress::ProgressTracker;
use crate::agent::turn::retry::{ReviewRepairState, TurnRetryState};
use crate::agent::turn::speculation::SpeculationRegistry;
use crate::domain::TurnControlFlags;
use crate::steering::{
    EvidenceTracker, ImplementationIntent, ImplementationTracker, MutationRecovery, ReviewIntent,
};
use crate::verify::{Snapshot, WorkspaceRepairVerifier};
use crate::{ReviewStatus, TaskContract, TurnPhaseLatencies};

use super::retention::ToolTimeline;

/// All mutable state that lives for one `run_turn` invocation.
pub(super) struct TurnState {
    // --- identity / setup ---
    pub user_prompt_tokens: u64,
    pub turn_ledger_revision: u64,
    pub turn_background_baseline: Vec<String>,
    pub context_task: String,
    pub task_contract: TaskContract,
    pub repository_context_enabled: bool,
    pub ranked_context_paths: BTreeSet<String>,
    pub context_generation_seen: u64,
    pub indexed_ledger_revision: u64,
    pub read_only_intent: Option<ReviewIntent>,
    pub implementation_intent: Option<ImplementationIntent>,
    pub expected_mutation: bool,
    pub requested_validation: bool,
    /// When set, inspection-sprawl caps apply (same type as read-only intent).
    pub inspection_sprawl_intent: Option<ReviewIntent>,
    pub read_only_inspection_cap: Option<u32>,
    pub turn_input: String,

    // --- checkpoints / verify harness ---
    pub turn_checkpoint_allowed: Option<bool>,
    pub turn_checkpoint_created: bool,
    pub verifier: WorkspaceRepairVerifier,
    pub fast_feedback: FastFeedbackState,
    pub max_steps: u32,
    pub max_parallel_tools: usize,

    // --- loop budgets ---
    pub steps: u32,
    pub empty_retries: u32,
    pub truncation_retries: u32,
    pub truncation_total_retries: u32,
    pub silent_continues: u32,
    /// Bounded repair for canned text such as "Completed the requested
    /// action." Kept separate from empty/protocol retries because the provider
    /// returned syntactically valid output; it was semantically unusable.
    pub generic_completion_retries: u32,
    pub continue_total_nudges: u32,
    pub repeat_nudges: u32,

    // --- control / trackers ---
    pub flags: TurnControlFlags,
    pub mutation_recovery: MutationRecovery,
    pub plan_updated_goal: bool,
    pub proposed_goal: Option<crate::Goal>,
    pub goal_before: Option<crate::Goal>,
    pub progress_tracker: ProgressTracker,
    pub evidence: EvidenceTracker,
    pub implementation_tracker: ImplementationTracker,
    pub review_repair: ReviewRepairState,
    pub empty_tui_needs_project: bool,

    // --- scheduler / tools ---
    pub sched_tool_calls: u32,
    pub sched_max_concurrent: u32,
    pub sched_serial_runs: u32,
    pub tool_timeline: ToolTimeline,
    pub advertised_tool_names: BTreeSet<String>,
    pub tool_schema_tokens: u64,
    pub speculation_registry: SpeculationRegistry,
    /// Remove `run_program` from the next request after one malformed or
    /// rejected program, giving the model exactly one ordinary-tool recovery.
    pub program_fallback_next: bool,
    /// Prevent a provider that ignores the fallback tool catalog from
    /// repeatedly re-entering the rejected program path.
    pub program_fallback_used: bool,
    /// Keep the one-time DeepSeek strict-schema fallback active for the rest
    /// of this tool loop. Some gateways alternate between valid and malformed
    /// arguments when strict mode is re-enabled on the next request.
    pub deepseek_strict_fallback_active: bool,
    pub deepseek_strict_fallback_used: bool,

    // --- provider retry ---
    pub retry_state: TurnRetryState,
    pub request_max_tokens_override: Option<u32>,
    pub compat_fallbacks: Vec<String>,
    pub effective_fallback_route: Option<String>,

    // --- verify / settle ---
    pub independent_review_status: ReviewStatus,
    pub independent_review_repairs: u32,
    /// Why the independent review produced no verdict (provider error, empty
    /// bounded diff, post-pass invalidation). Persisted with the outcome so a
    /// post-mortem can recover what the transient status line said.
    pub review_unavailable_reason: Option<String>,
    pub verification_infrastructure_error: bool,
    pub verification_unstable: bool,
    pub last_verify_attributions: Vec<hi_tools::Attribution>,
    pub turn_snapshot: Option<Snapshot>,
    pub turn_start: usize,
    pub phase_latencies: TurnPhaseLatencies,
}

impl TurnState {
    /// Project model-round mutables from this owned bag.
    pub(super) fn as_model_round_state(&mut self) -> super::model_round::ModelRoundState<'_> {
        super::model_round::ModelRoundState {
            steps: &mut self.steps,
            empty_retries: &mut self.empty_retries,
            truncation_retries: &mut self.truncation_retries,
            truncation_total_retries: &mut self.truncation_total_retries,
            silent_continues: &mut self.silent_continues,
            generic_completion_retries: &mut self.generic_completion_retries,
            continue_total_nudges: &mut self.continue_total_nudges,
            repeat_nudges: &mut self.repeat_nudges,
            force_tools_next: &mut self.flags.force_tools_next,
            text_tool_fallback_next: &mut self.flags.text_tool_fallback_next,
            force_text_answer_next: &mut self.flags.force_text_answer_next,
            suppress_bookkeeping_tools_next: &mut self.flags.suppress_bookkeeping_tools_next,
            made_tool_call: &mut self.flags.made_tool_call,
            provider_exhausted: &mut self.flags.provider_exhausted,
            ended_at_cap: &mut self.flags.ended_at_cap,
            cap_wrap_up_requested: &mut self.flags.cap_wrap_up_requested,
            cap_kind: &mut self.flags.cap_kind,
            turn_start: &mut self.turn_start,
            context_generation_seen: &mut self.context_generation_seen,
            indexed_ledger_revision: &mut self.indexed_ledger_revision,
            sched_tool_calls: &mut self.sched_tool_calls,
            sched_max_concurrent: &mut self.sched_max_concurrent,
            sched_serial_runs: &mut self.sched_serial_runs,
            tool_schema_tokens: &mut self.tool_schema_tokens,
            deepseek_strict_fallback_active: &mut self.deepseek_strict_fallback_active,
            retry_state: &mut self.retry_state,
            request_max_tokens_override: &mut self.request_max_tokens_override,
            compat_fallbacks: &mut self.compat_fallbacks,
            effective_fallback_route: &mut self.effective_fallback_route,
            ranked_context_paths: &mut self.ranked_context_paths,
            progress_tracker: &mut self.progress_tracker,
            evidence: &mut self.evidence,
            implementation_tracker: &mut self.implementation_tracker,
            review_repair: &mut self.review_repair,
            last_verify_attributions: &mut self.last_verify_attributions,
            tool_timeline: &mut self.tool_timeline,
            speculation_registry: &self.speculation_registry,
            program_fallback_next: &mut self.program_fallback_next,
            program_fallback_used: &mut self.program_fallback_used,
            advertised_tool_names: &mut self.advertised_tool_names,
            turn_snapshot: &mut self.turn_snapshot,
            max_steps: self.max_steps,
            context_task: &self.context_task,
            task_intent: self.task_contract.intent,
            repository_context_enabled: self.repository_context_enabled,
            turn_ledger_revision: self.turn_ledger_revision,
            read_only_intent: self.read_only_intent,
            implementation_intent: self.implementation_intent,
            read_only_inspection_cap: self.read_only_inspection_cap,
            expected_mutation: self.expected_mutation,
            requested_validation: self.requested_validation,
            input: &self.turn_input,
            user_prompt_tokens: self.user_prompt_tokens,
            inspection_sprawl_intent: self.inspection_sprawl_intent,
            verifier: &self.verifier,
        }
    }
}

impl Drop for TurnState {
    fn drop(&mut self) {
        // A turn can end through cancellation, disconnect, or an early
        // provider error before the normal model-round cleanup runs.  The
        // registry owns spawned shadow tasks, so make the turn boundary an
        // unconditional final cancellation barrier as well.
        self.speculation_registry.cancel_all();
    }
}
