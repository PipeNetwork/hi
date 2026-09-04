//! Post-tool Steer: mutation recovery, repeat/no-progress, implementation stalls.

use crate::agent::mutation_recovery_turn::MutationRecoveryControl;
use crate::steering::{
    BACKGROUND_WAIT_FINAL_NUDGE, BACKGROUND_WAIT_STATUS_NUDGE, EvidenceTracker,
    IMPLEMENTATION_NO_CHANGES_NUDGE, ImplementationIntent, ImplementationTracker, MutationRecovery,
    REREAD_NUDGE, WAIT_POLL_STATIC_NUDGE, bash_call_waits, implementation_text_tool_nudge,
    tool_validation_retry_nudge, unavailable_tool_retry_nudge,
};
use crate::transcript::NudgeKind;
use crate::ui::Ui;

use super::super::phase::TurnPhase;
use super::super::progress::{
    AWAITING_BACKGROUND_REASON, NO_PROGRESS_FINAL_ANSWER_NUDGE, ProgressKind, ProgressTracker,
    REPEATED_VALIDATION_DIAGNOSIS_NUDGE, WAITING_ROUND_BUDGET, no_progress_signature_for_calls,
};
use super::super::tools::{ToolBatchOutcome, ToolProtocolFailureKind};
use super::RoundControl;

const COMPLETED_PLAN_TOOL_CLOSEOUT: &str =
    "The plan is complete and the successful tool results were retained.";

impl crate::Agent {
    /// Post-tool Steer: mutation recovery and repeat/idempotent guards.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent::turn) fn steer_after_tools(
        &mut self,
        batch: &ToolBatchOutcome,
        expected_mutation: bool,
        implementation_intent: Option<ImplementationIntent>,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        mutation_recovery: &mut MutationRecovery,
        progress_tracker: &mut ProgressTracker,
        repeat_nudges: &mut u32,
        force_tools_next: &mut bool,
        suppress_bookkeeping_tools_next: &mut bool,
        text_tool_fallback_next: &mut bool,
        tool_validation_text_fallback_used: &mut bool,
        deepseek_strict_fallback_active: &mut bool,
        deepseek_strict_fallback_used: &mut bool,
        ui: &mut dyn Ui,
    ) -> RoundControl {
        let ToolBatchOutcome {
            hash_guard_applies,
            hashable_idempotent_results,
            repeated_idempotent_results,
            running_background_poll_results,
            actionable_poll_results,
            wait_flavored_results,
            ref tool_progress_labels,
            plan_changed_this_batch,
            interrupted_calls,
            interrupted_coordination_calls,
            ref unknown_background_handles,
            ..
        } = *batch;
        let protocol_validation_errors = &batch.protocol_validation_errors;
        let calls = batch.calls.as_slice();
        let read_only_intent = batch.read_only_intent;
        // Post-tool policy (mutation recovery, inspection sprawl, …) is Steer.
        self.set_turn_phase(TurnPhase::Steer);
        if interrupted_calls > 0 {
            let coordination_only = interrupted_calls == interrupted_coordination_calls;
            *force_tools_next = true;
            // If a bookkeeping call was interrupted as part of a mixed batch,
            // withhold the whole coordination family too. Otherwise the model
            // can immediately replay the skipped plan/decision call while the
            // concrete work is still waiting for recovery.
            *suppress_bookkeeping_tools_next |= interrupted_coordination_calls > 0;
            progress_tracker.prev_added_no_evidence = false;
            progress_tracker.record(
                ProgressKind::Weak,
                "user skipped a tool call; task remains active",
                None,
            );
            let nudge = if coordination_only {
                "The user skipped the preceding bookkeeping tool call, not the overall task. Do not stop or merely report the interruption, and do not issue another planning/bookkeeping call now. Continue the original task with a concrete inspection, edit, or validation tool."
            } else {
                "The user skipped the preceding tool call, not the overall task. Do not stop or merely report the interruption. Continue the original task now using a different appropriate tool."
            };
            ui.nudge("tool call skipped — steering the model to continue the active task");
            self.messages.push_nudge(NudgeKind::Continue, nudge);
            return RoundControl::Continue;
        }
        if !protocol_validation_errors.is_empty() {
            let validation_summary = protocol_validation_errors
                .iter()
                .take(3)
                .map(|failure| format!("{}: {}", failure.tool, failure.message))
                .collect::<Vec<_>>()
                .join("; ");
            if protocol_validation_errors
                .iter()
                .any(|failure| failure.kind == ToolProtocolFailureKind::EnvelopeIntegrity)
            {
                progress_tracker.record(
                    ProgressKind::None,
                    "sealed tool envelope failed integrity validation",
                    None,
                );
                *force_tools_next = false;
                *text_tool_fallback_next = false;
                ui.status(&format!(
                    "sealed tool envelope failed integrity validation; stopping tool recovery ({validation_summary})"
                ));
                return RoundControl::BreakInner(false);
            }
            let unavailable_tools = protocol_validation_errors
                .iter()
                .filter(|failure| failure.kind == ToolProtocolFailureKind::UnavailableTool)
                .map(|failure| failure.tool.as_str())
                .collect::<Vec<_>>();
            if !unavailable_tools.is_empty() {
                if *repeat_nudges < self.config.loop_limits.max_repeat_nudges {
                    *repeat_nudges += 1;
                    *force_tools_next = !batch.admitted_tool_names.is_empty();
                    *text_tool_fallback_next = false;
                    progress_tracker.force_no_progress_final_answer_next = false;
                    progress_tracker.prev_added_no_evidence = false;
                    progress_tracker.prev_call_sig = None;
                    ui.nudge(&format!(
                        "the model called a tool outside this request's sealed envelope; retrying with the admitted tool list ({repeat_nudges}/{})",
                        self.config.loop_limits.max_repeat_nudges,
                    ));
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        unavailable_tool_retry_nudge(
                            &unavailable_tools,
                            &batch.admitted_tool_names,
                        ),
                    );
                    return RoundControl::Continue;
                }
                ui.status(&format!(
                    "the model kept calling tools outside the sealed envelope ({validation_summary})"
                ));
                return RoundControl::BreakInner(false);
            }
            if *repeat_nudges < self.config.loop_limits.max_repeat_nudges {
                if !*deepseek_strict_fallback_used
                    && self.config.routing.deepseek_compat != hi_ai::DeepSeekCompat::Off
                {
                    *deepseek_strict_fallback_used = true;
                    *deepseek_strict_fallback_active = true;
                    ui.status(
                        "DeepSeek tool arguments failed client validation; retrying once without strict schemas",
                    );
                }
                *repeat_nudges += 1;
                *force_tools_next = true;
                *text_tool_fallback_next = false;
                progress_tracker.force_no_progress_final_answer_next = false;
                progress_tracker.prev_added_no_evidence = false;
                progress_tracker.prev_call_sig = None;
                let guidance = protocol_validation_errors
                    .iter()
                    .take(3)
                    .map(|failure| tool_validation_retry_nudge(&failure.tool, &failure.message))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                ui.nudge(&format!(
                    "the model emitted invalid tool arguments ({validation_summary}); requesting a schema-corrected call ({repeat_nudges}/{})",
                    self.config.loop_limits.max_repeat_nudges,
                ));
                self.messages.push_nudge(NudgeKind::Continue, guidance);
                return RoundControl::Continue;
            }
            if (implementation_intent.is_some() || expected_mutation)
                && !*tool_validation_text_fallback_used
            {
                *tool_validation_text_fallback_used = true;
                *force_tools_next = false;
                *text_tool_fallback_next = true;
                progress_tracker.force_no_progress_final_answer_next = false;
                progress_tracker.prev_added_no_evidence = false;
                progress_tracker.prev_call_sig = None;
                let guidance = protocol_validation_errors
                    .iter()
                    .take(3)
                    .map(|failure| tool_validation_retry_nudge(&failure.tool, &failure.message))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                ui.nudge(
                    "structured tool arguments kept failing validation; trying one plain-text tool call",
                );
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    implementation_text_tool_nudge(&guidance),
                );
                return RoundControl::Continue;
            }
            if self.try_no_progress_recovery(progress_tracker, force_tools_next, None, ui) {
                return RoundControl::Continue;
            }
            ui.status(&format!(
                "tool arguments kept failing validation ({validation_summary})"
            ));
            return RoundControl::BreakInner(false);
        }

        // A final, successful bookkeeping-only batch is itself a terminal
        // signal once concrete work has landed. Asking the main provider for
        // one more prose recap adds no execution value and can turn completed
        // work into a false infrastructure failure if that optional request is
        // empty or the route becomes unavailable. Settle through the normal
        // outer reconciliation/verification path, with a deterministic visible
        // closeout so finalization does not issue another provider request.
        let completed_plan_after_bookkeeping = plan_changed_this_batch
            && calls.iter().all(|(_, name, _)| name == "update_plan")
            // Structured long-horizon goals have an independent completion
            // auditor at settlement. Keep their normal model-answer round so
            // the audit receives the intended response and can commit the
            // durable goal transition after verification.
            && self.goals.structured.is_none()
            && !self.goals.plan().is_empty()
            && !self.goals.plan_incomplete()
            && (implementation_tracker.mutation_seen || implementation_tracker.validation_seen);
        if completed_plan_after_bookkeeping {
            self.emit_assistant_text(ui, COMPLETED_PLAN_TOOL_CLOSEOUT);
            ui.assistant_end();
            self.messages.push_assistant(vec![hi_ai::Content::Text(
                COMPLETED_PLAN_TOOL_CLOSEOUT.into(),
            )]);
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_no_progress_reason.clear();
            progress_tracker.record_final_answer();
            ui.status("plan completed; settling from successful tool evidence");
            return RoundControl::BreakInner(false);
        }
        match self.handle_mutation_recovery(
            mutation_recovery,
            expected_mutation,
            implementation_tracker,
            evidence,
            plan_changed_this_batch,
            force_tools_next,
            ui,
        ) {
            MutationRecoveryControl::None => {}
            MutationRecoveryControl::Continue => return RoundControl::Continue,
            MutationRecoveryControl::Break => {
                ui.nudge(
                    "implementation discovery budget was exhausted; settling with current evidence",
                );
                return RoundControl::BreakInner(false);
            }
        }
        // Waiting-round detection keys on the process *lifecycle*, not output
        // novelty: a live progress bar makes every poll deliver fresh bytes,
        // which defeated the byte-identical idle guard for hours while the
        // turn burned a model round per poll. A round that only watched
        // still-running background work is waiting, full stop. After the
        // budget, steer to a terminal status answer; a quiet-but-running
        // process is not a no-progress turn, so no repeat budgets are consumed.
        //
        // Exception: a poll whose fresh output carried failure diagnostics
        // (compiler errors, test failures, panics) is new work arriving, not
        // waiting — falling through to the else arm resets the whole streak
        // so the model may act on the evidence. A live turn was once forced
        // tool-free one round after its poll finally surfaced the compile
        // error it needed to fix; that must not happen again.
        let waiting_round = running_background_poll_results > 0
            && wait_flavored_results == calls.len()
            && actionable_poll_results == 0;
        if waiting_round {
            progress_tracker.waiting_rounds = progress_tracker.waiting_rounds.saturating_add(1);
        } else {
            progress_tracker.waiting_rounds = 0;
            progress_tracker.awaiting_background = false;
        }
        if waiting_round && progress_tracker.waiting_rounds >= WAITING_ROUND_BUDGET {
            let first_request = !progress_tracker.awaiting_background;
            progress_tracker.awaiting_background = true;
            *repeat_nudges = 0;
            *force_tools_next = false;
            progress_tracker.force_no_progress_final_answer_next = false;
            progress_tracker.record(
                ProgressKind::Weak,
                AWAITING_BACKGROUND_REASON,
                no_progress_signature_for_calls(calls),
            );
            if first_request {
                ui.nudge(
                    "the background process is still running; asking the model to wait once with wait_secs or wrap up with a status report",
                );
                self.messages
                    .push_nudge(NudgeKind::Continue, BACKGROUND_WAIT_STATUS_NUDGE);
            } else {
                // Still polling after the wrap-up request — force the next
                // round tool-free so the status answer actually lands.
                progress_tracker.force_no_progress_final_answer_next = true;
                ui.nudge("still polling after the wrap-up request — forcing a final status answer");
                self.messages
                    .push_nudge(NudgeKind::Continue, BACKGROUND_WAIT_FINAL_NUDGE);
            }
            return RoundControl::Continue;
        }
        // A handle the model named that the registry has never seen. The
        // registry records whether it was empty at the time, so a *guessed*
        // id (nothing has ever run under it) is distinguishable from a
        // *pruned* one (a real process was forgotten at capacity). Guessed
        // ids are the model's own invention — correct the model without
        // surfacing anything to the user; pruned ids are a real limitation
        // the user may need to know about.
        let guessed_handle = unknown_background_handles
            .iter()
            .find(|handle| handle.registry_was_empty);
        if let Some(guessed) = guessed_handle {
            progress_tracker.prev_added_no_evidence = true;
            ui.nudge(&format!(
                "the model named background handle `{}`, which has never existed this session — steering it away from the invented handle",
                guessed.id
            ));
            self.messages.push_nudge(
                NudgeKind::Repeat,
                format!(
                    "The background process handle `{}` you just used has never existed this session — no background process has ever run under that id, so polling or killing it again cannot produce new output or change anything. Do not call bash_output or bash_kill for `{}` again. Continue from the available output, restart the command if you still need it, or finish with the current result.",
                    guessed.id, guessed.id
                ),
            );
            return RoundControl::Continue;
        }
        let repeated_result_no_progress = hash_guard_applies
            && hashable_idempotent_results == calls.len()
            && repeated_idempotent_results == calls.len();
        if repeated_result_no_progress {
            progress_tracker.prev_added_no_evidence = true;
            let repeat_budget_available =
                *repeat_nudges < self.config.loop_limits.max_repeat_nudges;
            let no_new_after_mutation = implementation_tracker.mutation_seen;
            if repeat_budget_available {
                *repeat_nudges += 1;
                let waiting_round = calls
                    .iter()
                    .any(|(_, name, args)| name == "bash" && bash_call_waits(args));
                let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
                    if waiting_round {
                        "wait poll returned static output"
                    } else {
                        "repeated idempotent tool output"
                    },
                    no_progress_signature_for_calls(calls),
                ) && implementation_intent.is_none()
                    && !(expected_mutation
                        && !implementation_tracker.mutation_seen
                        && !implementation_tracker.validation_seen);
                if waiting_round {
                    ui.nudge(&format!(
                "the wait-and-check poll returned the same output — nudging the model to diagnose the idle process ({repeat_nudges}/{})",
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
                    progress_tracker.force_no_progress_final_answer_next = true;
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
                no_progress_signature_for_calls(calls),
            );
            if !no_new_after_mutation {
                if let Some(intent) = read_only_intent {
                    // Prefer one force-text recovery when inspection already happened.
                    if !progress_tracker.force_no_progress_final_answer_next {
                        progress_tracker.force_no_progress_final_answer_next = true;
                        *force_tools_next = false;
                        *repeat_nudges = 0;
                        ui.nudge(
                            "review kept getting the same inspection output; forcing a bounded answer",
                        );
                        self.messages.push_nudge(
                            NudgeKind::Continue,
                            crate::steering::repair_nudge_with_required_next(
                                crate::steering::ReviewRepairMode::SprawlForceAnswer,
                                crate::steering::summarize_inspected_evidence_nudge(
                                    intent, evidence,
                                ),
                            ),
                        );
                        return RoundControl::Continue;
                    }
                    if self.try_no_progress_recovery(progress_tracker, force_tools_next, None, ui) {
                        progress_tracker.prev_call_sig = None;
                        return RoundControl::Continue;
                    }
                    progress_tracker.record(
                        ProgressKind::None,
                        "repeat_same_inspection_output",
                        None,
                    );
                    ui.nudge("review kept getting the same inspection output");
                    let _ = intent;
                    return RoundControl::BreakInner(false);
                }
                if (implementation_intent.is_some() || expected_mutation)
                    && !implementation_tracker.mutation_seen
                {
                    if implementation_tracker.no_change_nudges < 2 {
                        implementation_tracker.no_change_nudges += 1;
                        evidence.quality_repair_nudges =
                            evidence.quality_repair_nudges.saturating_add(1);
                        let use_text_fallback = implementation_tracker.no_change_nudges >= 2;
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

                    if self.try_no_progress_recovery(progress_tracker, force_tools_next, None, ui) {
                        progress_tracker.prev_call_sig = None;
                        return RoundControl::Continue;
                    }
                    progress_tracker.record(
                        ProgressKind::None,
                        "implementation_repeat_no_edit",
                        None,
                    );
                    ui.nudge(
                        "implementation repeated equivalent inspection output without editing",
                    );
                    return RoundControl::BreakInner(false);
                }
            }
        } else if !tool_progress_labels.is_empty() {
            progress_tracker.record_round_from_tools(tool_progress_labels);
            if implementation_tracker.mutation_seen
                && progress_tracker.take_repeated_validation_diagnosis()
            {
                *force_tools_next = true;
                progress_tracker.force_no_progress_final_answer_next = false;
                ui.nudge(
                    "the same validation failure survived two repair cycles — requesting a focused root-cause diagnosis",
                );
                self.messages
                    .push_nudge(NudgeKind::Continue, REPEATED_VALIDATION_DIAGNOSIS_NUDGE);
                return RoundControl::Continue;
            }
            if implementation_tracker.mutation_seen
                && let Some(failure_signature) =
                    progress_tracker.repeated_validation_repair_exhausted()
            {
                *force_tools_next = false;
                progress_tracker.record(
                    ProgressKind::None,
                    "same validation failure survived focused repair",
                    Some(failure_signature),
                );
                ui.status("the same validation failure persisted after focused repair");
                return RoundControl::BreakInner(false);
            }
        }

        RoundControl::Continue
    }
}
