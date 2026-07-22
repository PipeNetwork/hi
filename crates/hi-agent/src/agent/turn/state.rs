//! Owned per-turn state bag for the interactive loop.
//!
//! Built once in `run_turn_body` and passed through Model / Tools / Steer /
//! WorkspaceRepair instead of a long list of locals + borrow bags.

use std::collections::BTreeSet;

use crate::agent::turn::fast_feedback::FastFeedbackState;
use crate::agent::turn::progress::ProgressTracker;
use crate::agent::turn::retry::{ReviewRepairState, TurnRetryState};
use crate::domain::TurnControlFlags;
use crate::steering::{
    EvidenceTracker, ImplementationIntent, ImplementationTracker, MutationRecovery, ReviewIntent,
    ToolLoopGuardrail,
};
use crate::verify::{Snapshot, WorkspaceRepairVerifier};
use crate::{ReviewStatus, TaskContract, ToolCallEntry};

/// All mutable state that lives for one `run_turn` invocation.
pub(super) struct TurnState {
    // --- identity / setup ---
    pub user_prompt_tokens: u64,
    pub turn_ledger_revision: u64,
    pub turn_background_baseline: Vec<String>,
    pub context_task: String,
    pub goal_drive_turn: bool,
    pub task_contract: TaskContract,
    pub repository_context_enabled: bool,
    pub ranked_context_paths: BTreeSet<String>,
    pub context_generation_seen: u64,
    pub indexed_ledger_revision: u64,
    pub read_only_intent: Option<ReviewIntent>,
    pub implementation_intent: Option<ImplementationIntent>,
    pub expected_mutation: bool,
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
    pub continue_total_nudges: u32,
    pub repeat_nudges: u32,
    pub repeat_sampling_rounds: u32,

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
    pub tool_guardrail: ToolLoopGuardrail,
    pub empty_tui_needs_project: bool,

    // --- scheduler / tools ---
    pub sched_tool_calls: u32,
    pub sched_max_concurrent: u32,
    pub sched_serial_runs: u32,
    pub tool_timeline: Vec<ToolCallEntry>,
    pub advertised_tool_names: BTreeSet<String>,
    pub tool_schema_tokens: u64,
    pub prev_call_sig: Option<Vec<(String, String)>>,
    pub prev_added_no_evidence: bool,

    // --- provider retry ---
    pub retry_state: TurnRetryState,
    pub request_max_tokens_override: Option<u32>,
    pub compat_fallbacks: Vec<String>,
    pub effective_fallback_route: Option<String>,

    // --- verify / settle ---
    pub independent_review_status: ReviewStatus,
    pub independent_review_repairs: u32,
    pub verification_infrastructure_error: bool,
    pub verification_unstable: bool,
    pub verified_at: Option<(u64, String)>,
    pub last_verify_attributions: Vec<hi_tools::Attribution>,
    pub turn_snapshot: Option<Snapshot>,
    pub turn_start: usize,
}

impl TurnState {
    /// Project verify-outcome mutables without a separate lifetime bag.
    pub(super) fn as_verify_outcome_state(
        &mut self,
    ) -> super::verify_outcome::VerifyOutcomeState<'_> {
        super::verify_outcome::VerifyOutcomeState {
            obligation_nudge_fired: &mut self.flags.obligation_nudge_fired,
            force_tools_next: &mut self.flags.force_tools_next,
            verified_at: &mut self.verified_at,
            independent_review_status: &mut self.independent_review_status,
            independent_review_repairs: &mut self.independent_review_repairs,
            stalled_unfinished: &mut self.flags.stalled_unfinished,
            verification_infrastructure_error: &mut self.verification_infrastructure_error,
            verification_unstable: &mut self.verification_unstable,
            last_verify_attributions: &mut self.last_verify_attributions,
            ranked_context_paths: &mut self.ranked_context_paths,
            context_generation_seen: &mut self.context_generation_seen,
            indexed_ledger_revision: &mut self.indexed_ledger_revision,
        }
    }

    /// Project model-round mutables from this owned bag.
    pub(super) fn as_model_round_state(&mut self) -> super::model_round::ModelRoundState<'_> {
        super::model_round::ModelRoundState {
            steps: &mut self.steps,
            empty_retries: &mut self.empty_retries,
            truncation_retries: &mut self.truncation_retries,
            truncation_total_retries: &mut self.truncation_total_retries,
            silent_continues: &mut self.silent_continues,
            continue_total_nudges: &mut self.continue_total_nudges,
            repeat_nudges: &mut self.repeat_nudges,
            repeat_sampling_rounds: &mut self.repeat_sampling_rounds,
            force_tools_next: &mut self.flags.force_tools_next,
            text_tool_fallback_next: &mut self.flags.text_tool_fallback_next,
            force_text_answer_next: &mut self.flags.force_text_answer_next,
            force_no_progress_final_answer_next: &mut self
                .flags
                .force_no_progress_final_answer_next,
            suppress_bookkeeping_tools_next: &mut self.flags.suppress_bookkeeping_tools_next,
            made_tool_call: &mut self.flags.made_tool_call,
            stalled_repeating: &mut self.flags.stalled_repeating,
            stalled_unfinished: &mut self.flags.stalled_unfinished,
            ended_at_cap: &mut self.flags.ended_at_cap,
            prev_added_no_evidence: &mut self.prev_added_no_evidence,
            turn_start: &mut self.turn_start,
            context_generation_seen: &mut self.context_generation_seen,
            indexed_ledger_revision: &mut self.indexed_ledger_revision,
            sched_tool_calls: &mut self.sched_tool_calls,
            sched_max_concurrent: &mut self.sched_max_concurrent,
            sched_serial_runs: &mut self.sched_serial_runs,
            tool_schema_tokens: &mut self.tool_schema_tokens,
            prev_call_sig: &mut self.prev_call_sig,
            retry_state: &mut self.retry_state,
            request_max_tokens_override: &mut self.request_max_tokens_override,
            compat_fallbacks: &mut self.compat_fallbacks,
            effective_fallback_route: &mut self.effective_fallback_route,
            ranked_context_paths: &mut self.ranked_context_paths,
            progress_tracker: &mut self.progress_tracker,
            evidence: &mut self.evidence,
            implementation_tracker: &mut self.implementation_tracker,
            review_repair: &mut self.review_repair,
            tool_guardrail: &mut self.tool_guardrail,
            last_verify_attributions: &mut self.last_verify_attributions,
            tool_timeline: &mut self.tool_timeline,
            advertised_tool_names: &mut self.advertised_tool_names,
            turn_snapshot: &mut self.turn_snapshot,
            max_steps: self.max_steps,
            context_task: &self.context_task,
            repository_context_enabled: self.repository_context_enabled,
            turn_ledger_revision: self.turn_ledger_revision,
            read_only_intent: self.read_only_intent,
            implementation_intent: self.implementation_intent,
            read_only_inspection_cap: self.read_only_inspection_cap,
            expected_mutation: self.expected_mutation,
            input: &self.turn_input,
            user_prompt_tokens: self.user_prompt_tokens,
            inspection_sprawl_intent: self.inspection_sprawl_intent,
            verifier: &self.verifier,
        }
    }
}
