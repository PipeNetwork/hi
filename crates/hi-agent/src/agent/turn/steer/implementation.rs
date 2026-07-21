//! Post-tool Steer: mutation recovery, repeat/no-progress, implementation stalls.

use crate::agent::mutation_recovery_turn::MutationRecoveryControl;
use crate::steering::{
    EvidenceTracker, IMPLEMENTATION_NO_CHANGES_NUDGE, ImplementationIntent, ImplementationTracker,
    MutationRecovery, REREAD_NUDGE, ReviewIntent, WAIT_POLL_STATIC_NUDGE, bash_call_waits,
    implementation_text_tool_nudge,
};
use crate::transcript::NudgeKind;
use crate::ui::Ui;

use super::super::phase::TurnPhase;
use super::super::progress::{
    NO_PROGRESS_FINAL_ANSWER_NUDGE, ProgressKind, ProgressTracker, no_progress_signature_for_calls,
};
use super::super::retry::INCOMPLETE_STATUS;
use super::super::tools::ToolBatchOutcome;
use super::RoundControl;

impl crate::Agent {
    /// Post-tool Steer: mutation recovery, repeat/idempotent guards, sprawl.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent::turn) fn steer_after_tools(
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
        if (implementation_intent.is_some() || expected_mutation)
            && !implementation_tracker.mutation_seen
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

