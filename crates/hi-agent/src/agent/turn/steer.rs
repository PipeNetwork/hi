//! Steer-phase policy after a model round (no tools or post-tools).
//!
//! Review-answer repair, implementation completeness, silent continues, and
//! post-tool mutation recovery / repeat guards. Workspace compile/lint/test
//! repair stays in [`super::verify_run`] under WorkspaceRepair.

use hi_ai::Content;

use crate::agent::mutation_recovery_turn::MutationRecoveryControl;
use crate::heuristics::looks_like_unfinished_step;
use crate::heuristics::plan_has_pending_steps;
use crate::steering::{
    CONCRETE_REVIEW_NUDGE,
    EvidenceTracker,
    GAP_SEARCH_OVERCLAIM_NUDGE,
    IMPLEMENTATION_NO_CHANGES_NUDGE,
    IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE,
    ImplementationIntent,
    ImplementationTracker,
    MutationRecovery,
    READ_AFTER_SEARCH_NUDGE,
    REREAD_NUDGE,
    ReviewIntent,
    ReviewRepairMode,
    SECURITY_BROAD_SEARCH_NUDGE,
    SECURITY_SCOPE_NUDGE,
    WAIT_POLL_STATIC_NUDGE,
    answer_says_insufficient_evidence,
    bash_call_waits,
    concrete_review_answer_problem,
    deepen_review_nudge,
    implementation_missing_validation_nudge,
    implementation_text_tool_nudge,
    no_evidence_review_nudge,
    repair_nudge_with_required_next,
    should_deepen_review,
    should_nudge_gap_search_overclaim,
    should_nudge_no_evidence_review,
    should_nudge_read_after_search_final,
    should_nudge_security_broad_search,
    should_nudge_security_scope,
    should_reject_review_repair_template,
    summarize_inspected_evidence_nudge,
};
use crate::transcript::NudgeKind;
use crate::{PLAN_CONTINUE_NUDGE, SILENT_CONTINUE_NUDGE, Ui};

use super::phase::TurnPhase;
use super::progress::{
    NO_PROGRESS_FINAL_ANSWER_NUDGE,
    ProgressKind,
    ProgressTracker,
    no_progress_signature_for_calls,
};
use super::retry::{INCOMPLETE_STATUS, ReviewRepairState};
use super::tools::ToolBatchOutcome;

/// Whether the inner Model→Tools→Steer loop should continue or stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RoundControl {
    Continue,
    /// `true` means step-cap; `false` means natural end / stalled end of tools loop.
    BreakInner(bool),
}

impl crate::Agent {
    /// Post-model Steer when the model returned text and no tool calls.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn steer_without_tools(
        &mut self,
        assistant_text: &str,
        completion_content: &mut Vec<Content>,
        read_only_intent: Option<ReviewIntent>,
        implementation_intent: Option<ImplementationIntent>,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        review_repair: &mut ReviewRepairState,
        progress_tracker: &mut ProgressTracker,
        silent_continues: &mut u32,
        continue_total_nudges: &mut u32,
        force_tools_next: &mut bool,
        force_text_answer_next: &mut bool,
        text_tool_fallback_next: &mut bool,
        stalled_unfinished: &mut bool,
        buffered_assistant_text: &mut String,
        buffer_read_only_review_text: bool,
        _steps: u32,
        ui: &mut dyn Ui,
    ) -> RoundControl {
        self.set_turn_phase(TurnPhase::Steer);
        let budgets = &self.config.loop_limits.review_repair;
// Text but no tool call (the content-less case was handled
// above). Silently re-prompt the model to continue — no
// status line, no steer counter, no visible nudge.
//
// Two signals detect an unfinished turn:
// 1. The text looks like an announced-but-unperformed next
//    step ("Let me start by…", "Now I'll rewrite main.rs:").
// 2. The plan has pending/active steps — the model posted a
//    plan via `update_plan` and it's not complete, even if
//    the text reads like a finished recap ("I've implemented
//    proof.rs."). The plan state is unambiguous and catches
//    the common case where the model does one sub-task,
//    writes a recap, and stops — leaving the plan at 2/9.
//
// A *finished* response ends the turn cleanly: a final recap
// after a multi-step task with a complete plan, or a plain
// Q&A answer. Bounded so it can't loop forever.
let looks_unfinished = looks_like_unfinished_step(assistant_text);
let plan_incomplete = plan_has_pending_steps(&self.goals.last_plan);
if let Some(intent) = read_only_intent
    && (looks_unfinished || plan_incomplete)
{
    if evidence.inspection_sprawl_nudges > 0 {
        if evidence.quality_repair_nudges < 3 {
            evidence.quality_repair_nudges += 1;
            *continue_total_nudges += 1;
            *force_text_answer_next = true;
            ui.nudge(
                "review tried to continue inspecting after the sprawl limit; forcing a bounded answer from existing evidence",
            );
            self.messages
                .push_assistant(std::mem::take(completion_content));
            self.messages.push_nudge(
                NudgeKind::Continue,
                summarize_inspected_evidence_nudge(intent, &evidence),
            );
            return RoundControl::Continue;
        }

        *stalled_unfinished = true;
        let _ = intent;
        ui.status(INCOMPLETE_STATUS);
        return RoundControl::BreakInner(false);
    }

    if *silent_continues < self.config.loop_limits.max_silent_continues {
        self.messages
            .push_assistant(std::mem::take(completion_content));
        *silent_continues += 1;
        *continue_total_nudges += 1;
        *force_tools_next = true;
        let nudge = if plan_incomplete && !looks_unfinished {
            PLAN_CONTINUE_NUDGE
        } else {
            SILENT_CONTINUE_NUDGE
        };
        self.messages.push_nudge(NudgeKind::Continue, nudge);
        return RoundControl::Continue;
    }
}
if implementation_intent.is_some() && !implementation_tracker.mutation_seen {
    if implementation_tracker.no_change_nudges < 2 {
        implementation_tracker.no_change_nudges += 1;
        evidence.quality_repair_nudges =
            evidence.quality_repair_nudges.saturating_add(1);
        let use_text_fallback = implementation_tracker.no_change_nudges >= 2;
        *force_tools_next = !use_text_fallback;
        *text_tool_fallback_next = use_text_fallback;
        ui.nudge(
	                                "implementation answer had no file changes; nudging the model to edit or scaffold",
	                            );
        self.messages
            .push_assistant(std::mem::take(completion_content));
        let nudge = if use_text_fallback {
            implementation_text_tool_nudge(IMPLEMENTATION_NO_CHANGES_NUDGE)
        } else {
            IMPLEMENTATION_NO_CHANGES_NUDGE.to_string()
        };
        self.messages.push_nudge(NudgeKind::Continue, nudge);
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    ui.nudge("implementation still had no file changes after repair");
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if implementation_intent.is_some()
    && implementation_tracker.mutation_seen
    && !implementation_tracker.substantive_edit_seen
{
    if implementation_tracker.scaffold_only_nudges < 2 {
        implementation_tracker.scaffold_only_nudges += 1;
        evidence.quality_repair_nudges =
            evidence.quality_repair_nudges.saturating_add(1);
        let use_text_fallback =
            implementation_tracker.scaffold_only_nudges >= 2;
        *force_tools_next = !use_text_fallback;
        *text_tool_fallback_next = use_text_fallback;
        ui.nudge(
	                                "implementation only scaffolded setup files; nudging the model to edit source files",
	                            );
        self.messages
            .push_assistant(std::mem::take(completion_content));
        let nudge = if use_text_fallback {
            implementation_text_tool_nudge(IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE)
        } else {
            IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE.to_string()
        };
        self.messages.push_nudge(NudgeKind::Continue, nudge);
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    ui.nudge(
        "implementation still only had scaffold/setup changes after repair",
    );
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if implementation_intent.is_some()
    && implementation_tracker.mutation_seen
    && !implementation_tracker.validation_after_last_mutation
{
    if implementation_tracker.missing_validation_nudges < 2 {
        implementation_tracker.missing_validation_nudges += 1;
        evidence.quality_repair_nudges =
            evidence.quality_repair_nudges.saturating_add(1);
        let use_text_fallback =
            implementation_tracker.missing_validation_nudges >= 2;
        *force_tools_next = !use_text_fallback;
        *text_tool_fallback_next = use_text_fallback;
        ui.nudge(
	                                "implementation changed files without validation; nudging the model to run tests or build",
	                            );
        self.messages
            .push_assistant(std::mem::take(completion_content));
        let validation_nudge =
            implementation_missing_validation_nudge(&implementation_tracker);
        let nudge = if use_text_fallback {
            implementation_text_tool_nudge(&validation_nudge)
        } else {
            validation_nudge
        };
        self.messages.push_nudge(NudgeKind::Continue, nudge);
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    ui.nudge("implementation still lacked validation after repair");
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if should_nudge_no_evidence_review(read_only_intent, &evidence, assistant_text)
{
    let mode = ReviewRepairMode::NoEvidence;
    if review_repair.spend(mode, evidence, budgets) {
        *force_tools_next = true;
        ui.nudge(
            "review answer had no inspected evidence; nudging the model to inspect before answering",
        );
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(
                mode,
                no_evidence_review_nudge(
                    read_only_intent.expect("checked above"),
                ),
            ),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.nudge(
        "review still had no inspected evidence after repair; stopping incomplete",
    );
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if let Some(intent) = read_only_intent
    && evidence.saw_read
    && answer_says_insufficient_evidence(assistant_text)
{
    if matches!(intent, ReviewIntent::Security)
        && evidence.saw_search
        && !evidence.security_search_complete()
        && review_repair
            .spend(ReviewRepairMode::SecurityBroadSearch, evidence, budgets)
    {
        *force_tools_next = true;
        ui.nudge(
            "security review gave a generic evidence disclaimer before searching all required pattern families; nudging the model to broaden the search",
        );
        self.messages
            .push_assistant_repair_note(ReviewRepairMode::SecurityBroadSearch);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(
                ReviewRepairMode::SecurityBroadSearch,
                SECURITY_BROAD_SEARCH_NUDGE,
            ),
        );
        return RoundControl::Continue;
    }
    let mode = ReviewRepairMode::InspectedDisclaimer;
    let chat_mode = ReviewRepairMode::InspectedDisclaimerChatAttempt;
    let has_disclaimer_budget = review_repair.has_budget(mode, budgets);
    let has_chat_attempt_budget = review_repair.has_budget(chat_mode, budgets);
    if has_disclaimer_budget || has_chat_attempt_budget {
        if has_disclaimer_budget {
            review_repair.spend(mode, evidence, budgets);
        } else {
            evidence.quality_repair_nudges =
                evidence.quality_repair_nudges.saturating_add(1);
        }
        review_repair.note(chat_mode);
        *force_text_answer_next = true;
        *force_tools_next = false;
        ui.nudge(
            "review gave a generic evidence disclaimer after inspection; nudging the model to answer from inspected files",
        );
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(
                mode,
                summarize_inspected_evidence_nudge(intent, &evidence),
            ),
        );
        return RoundControl::Continue;
    }
    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.status(
        "review kept returning a generic evidence disclaimer after inspection; stopping incomplete",
    );
    let _ = (intent, &evidence);
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
// (`saw_read` is implied here: the previous disjunct already
// catches search-without-read, so boolean-equivalently drop it.)
let needs_evidence_depth_repair = evidence.listing_only()
    || (evidence.saw_search && !evidence.saw_read)
    || (matches!(read_only_intent, Some(ReviewIntent::Security))
        && evidence.saw_search
        && !evidence.security_search_complete());
if !needs_evidence_depth_repair
    && should_reject_review_repair_template(read_only_intent, assistant_text)
{
    if let Some(intent) = read_only_intent
        && review_repair.spend(ReviewRepairMode::GenericTemplate, evidence, budgets)
    {
        let mode = ReviewRepairMode::GenericTemplate;
        let has_inspected_evidence = evidence.saw_read || evidence.saw_search;
        *force_text_answer_next = has_inspected_evidence;
        *force_tools_next = !has_inspected_evidence;
        ui.nudge(
            "review answer was a generic repair template; nudging the model to produce a concrete bounded review",
        );
        self.messages.push_assistant_repair_note(mode);
        let nudge = if has_inspected_evidence {
            summarize_inspected_evidence_nudge(intent, &evidence)
        } else {
            deepen_review_nudge(intent).to_string()
        };
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(mode, nudge),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(ReviewRepairMode::GenericTemplate);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.status("review answer stayed generic after repair; stopping incomplete");
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if should_deepen_review(read_only_intent, &evidence, assistant_text) {
    let mode = ReviewRepairMode::ListingOnly;
    if review_repair.spend(mode, evidence, budgets) {
        *force_tools_next = true;
        ui.nudge(
            "review evidence was only a listing; nudging the model to inspect files or search results",
        );
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(
                mode,
                deepen_review_nudge(read_only_intent.expect("checked above")),
            ),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.nudge(
        "review still had only listing evidence after repair; stopping incomplete",
    );
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if should_nudge_read_after_search_final(
    read_only_intent,
    &evidence,
    assistant_text,
) {
    let mode = ReviewRepairMode::ReadAfterSearch;
    if review_repair.spend(mode, evidence, budgets) {
        *force_tools_next = true;
        ui.nudge(
            "review had targeted search but no file reads; nudging the model to read matching files",
        );
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(mode, READ_AFTER_SEARCH_NUDGE),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.nudge(
        "review still had targeted search but no file reads after repair; stopping incomplete",
    );
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if should_nudge_security_broad_search(
    read_only_intent,
    &evidence,
    assistant_text,
) {
    let mode = ReviewRepairMode::SecurityBroadSearch;
    if review_repair.spend(mode, evidence, budgets) {
        *force_tools_next = true;
        ui.nudge(
            "security review missed required pattern families; nudging the model to broaden the search",
        );
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(mode, SECURITY_BROAD_SEARCH_NUDGE),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.nudge(
        "security review still missed required pattern families after repair; stopping incomplete",
    );
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if should_nudge_security_scope(read_only_intent, &evidence, assistant_text) {
    let mode = ReviewRepairMode::SecurityScope;
    if review_repair.spend(mode, evidence, budgets) {
        ui.status(
            "security answer overclaimed repo-wide safety; nudging the model to bound findings to evidence",
        );
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(mode, SECURITY_SCOPE_NUDGE),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.status(
        "security answer still overclaimed after repair; stopping incomplete",
    );
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if should_nudge_gap_search_overclaim(
    read_only_intent,
    &evidence,
    assistant_text,
) {
    let mode = ReviewRepairMode::GapSearchOverclaim;
    if review_repair.spend(mode, evidence, budgets) {
        ui.nudge(
            "gap answer contradicted search matches; nudging the model to bound claims to inspected evidence",
        );
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(mode, GAP_SEARCH_OVERCLAIM_NUDGE),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.nudge(
        "gap answer still overclaimed after search matches; stopping incomplete",
    );
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if let Some(problem) =
    concrete_review_answer_problem(read_only_intent, &evidence, assistant_text)
{
    let mode = ReviewRepairMode::ConcreteAnswer;
    if review_repair.spend(mode, evidence, budgets) {
        *force_text_answer_next = true;
        ui.nudge(problem.status());
        self.messages.push_assistant_repair_note(mode);
        self.messages.push_nudge(
            NudgeKind::Continue,
            repair_nudge_with_required_next(mode, CONCRETE_REVIEW_NUDGE),
        );
        return RoundControl::Continue;
    }

    *stalled_unfinished = true;
    let reason = review_repair.exhausted(mode);
    progress_tracker.record(ProgressKind::None, reason, None);
    ui.nudge(problem.exhausted_status());
    ui.status(INCOMPLETE_STATUS);
    return RoundControl::BreakInner(false);
}
if buffer_read_only_review_text {
    let text_to_emit = if buffered_assistant_text.is_empty() {
        assistant_text
    } else {
        buffered_assistant_text
    };
    ui.assistant_text(text_to_emit);
    ui.assistant_end();
}
self.messages
    .push_assistant(std::mem::take(completion_content));
if (looks_unfinished || plan_incomplete)
    && *silent_continues < self.config.loop_limits.max_silent_continues
{
    *silent_continues += 1;
    *continue_total_nudges += 1;
    // Force the next round to actually call a tool, so the
    // nudge can't be answered with yet another narration or an
    // empty completion.
    *force_tools_next = true;
    // Use a plan-aware nudge when the plan is incomplete, so
    // the model knows to continue the next step rather than
    // just "continue from where you stopped".
    let nudge = if plan_incomplete && !looks_unfinished {
        PLAN_CONTINUE_NUDGE
    } else {
        SILENT_CONTINUE_NUDGE
    };
    self.messages.push_nudge(NudgeKind::Continue, nudge);
    return RoundControl::Continue;
}
// If we exhausted the silent-continue budget (at least one
// continue was attempted) on a turn that looked unfinished,
// let the user know. Don't warn when max_silent_continues
// is 0 (no continue was attempted — the feature is off).
if (looks_unfinished || plan_incomplete) && *silent_continues > 0 {
    ui.status(
        "⚠ the model kept narrating without acting — the task may be \
         incomplete. /retry, or send 'continue'.",
    );
}
if looks_unfinished || plan_incomplete {
    progress_tracker.record(
        ProgressKind::Weak,
        "text answer looked unfinished",
        None,
    );
} else {
    progress_tracker.record_final_answer();
}
RoundControl::BreakInner(false)
    }

    /// Post-tool Steer: mutation recovery, repeat/idempotent guards, sprawl.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn steer_after_tools(
        &mut self,
        calls: &[(String, String, String)],
        batch: &ToolBatchOutcome,
        expected_mutation: bool,
        read_only_intent: Option<ReviewIntent>,
        implementation_intent: Option<ImplementationIntent>,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        mutation_recovery: &mut MutationRecovery,
        progress_tracker: &mut ProgressTracker,
        repeat_nudges: &mut u32,
        force_tools_next: &mut bool,
        text_tool_fallback_next: &mut bool,
        force_no_progress_final_answer_next: &mut bool,
        prev_added_no_evidence: &mut bool,
        stalled_repeating: &mut bool,
        stalled_unfinished: &mut bool,
        ui: &mut dyn Ui,
    ) -> RoundControl {
        let ToolBatchOutcome {
            hash_guard_applies,
            hashable_idempotent_results,
            repeated_idempotent_results,
            ref tool_progress_labels,
            plan_changed_this_batch,
        } = *batch;
        let plan_changed_this_batch = plan_changed_this_batch;
        let hashable_idempotent_results = hashable_idempotent_results;
        let repeated_idempotent_results = repeated_idempotent_results;
        let hash_guard_applies = hash_guard_applies;
// Post-tool policy (mutation recovery, inspection sprawl, …) is Steer.
self.set_turn_phase(TurnPhase::Steer);
match self.handle_mutation_recovery(
    mutation_recovery,
    expected_mutation,
    implementation_tracker,
    evidence,
    plan_changed_this_batch, force_tools_next,
    ui,
) {
    MutationRecoveryControl::None => {}
    MutationRecoveryControl::Continue => return RoundControl::Continue,
}
let repeated_result_no_progress = hash_guard_applies
    && hashable_idempotent_results == calls.len()
    && repeated_idempotent_results == calls.len();
if repeated_result_no_progress {
    *prev_added_no_evidence = true;
    let repeat_budget_available = *repeat_nudges < self.config.loop_limits.max_repeat_nudges;
    let no_new_after_mutation = implementation_tracker.mutation_seen;
    if repeat_budget_available {
        *repeat_nudges += 1;
        *stalled_repeating = true;
        let waiting_round = calls
            .iter()
            .any(|(_, name, args)| name == "bash" && bash_call_waits(args));
        let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
            if waiting_round {
                "wait poll returned static output"
            } else {
                "repeated idempotent tool output"
            },
            no_progress_signature_for_calls(&calls),
        ) && implementation_intent.is_none();
        if waiting_round {
            ui.nudge(&format!(
                "the wait-and-check poll returned the same output — nudging the model to diagnose the stalled process ({repeat_nudges}/{})",
                self.config.loop_limits.max_repeat_nudges
            ));
        } else {
            ui.nudge(&format!(
                "the model got the same inspection output again — nudging it to act on already-returned evidence ({repeat_nudges}/{})",
                self.config.loop_limits.max_repeat_nudges
            ));
        }
        let base_nudge = if waiting_round {
            WAIT_POLL_STATIC_NUDGE
        } else {
            REREAD_NUDGE
        };
        let nudge = if force_final_after_nudge {
            *force_no_progress_final_answer_next = true;
            *force_tools_next = false;
            format!("{base_nudge}\n\n{NO_PROGRESS_FINAL_ANSWER_NUDGE}")
        } else {
            base_nudge.to_string()
        };
        self.messages.push_nudge(NudgeKind::Repeat, nudge);
        return RoundControl::Continue;
    }
    progress_tracker.record(
        ProgressKind::None,
        "repeated idempotent tool output",
        no_progress_signature_for_calls(&calls),
    );
    if !no_new_after_mutation {
        if let Some(intent) = read_only_intent {
            *stalled_unfinished = true;
            ui.nudge(
                "review kept getting the same inspection output; stopping incomplete",
            );
            let _ = intent;
            ui.status(INCOMPLETE_STATUS);
            return RoundControl::BreakInner(false);
        }
        if implementation_intent.is_some() && !implementation_tracker.mutation_seen
        {
            if implementation_tracker.no_change_nudges < 2 {
                implementation_tracker.no_change_nudges += 1;
                evidence.quality_repair_nudges =
                    evidence.quality_repair_nudges.saturating_add(1);
                let use_text_fallback =
                    implementation_tracker.no_change_nudges >= 2;
                *force_tools_next = !use_text_fallback;
                *text_tool_fallback_next = use_text_fallback;
                ui.nudge(
                    "implementation repeated equivalent inspection output without editing; nudging the model to edit or scaffold",
                );
                let nudge = if use_text_fallback {
                    implementation_text_tool_nudge(IMPLEMENTATION_NO_CHANGES_NUDGE)
                } else {
                    IMPLEMENTATION_NO_CHANGES_NUDGE.to_string()
                };
                self.messages.push_nudge(NudgeKind::Continue, nudge);
                return RoundControl::Continue;
            }

            *stalled_unfinished = true;
            ui.nudge(
                "implementation repeated equivalent inspection output without editing",
            );
            ui.status(INCOMPLETE_STATUS);
            return RoundControl::BreakInner(false);
        }
    }
} else if !tool_progress_labels.is_empty() {
    progress_tracker.record_round_from_tools(&tool_progress_labels);
}

        RoundControl::Continue
    }
}
