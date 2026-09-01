//! Text-answer Steer path: unfinished continues, **answer-repair** quality
//! nudges, and implementation completeness gates.
//!
//! Answer repair (`ReviewRepairMode` / `ReviewRepairBudgets`) is distinct from
//! post-mutation **completion review** (`ReviewPolicy` → `ReviewStatus`) and
//! the long-horizon **goal skeptic**.

use hi_ai::Content;

use crate::steering::{
    EvidenceTracker, ImplementationIntent, ImplementationTracker, ReviewIntent,
    repair_nudge_with_required_next, summarize_inspected_evidence_nudge,
};
use crate::transcript::NudgeKind;
use crate::{GOAL_CONTINUE_NUDGE, PLAN_CONTINUE_NUDGE, Ui};

use super::super::phase::TurnPhase;
use super::super::progress::{AWAITING_BACKGROUND_REASON, ProgressKind, ProgressTracker};
use super::super::retry::ReviewRepairState;
use super::RoundControl;

impl crate::Agent {
    /// Post-model Steer when the model returned text and no tool calls this round.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent::turn) fn steer_without_tools(
        &mut self,
        assistant_text: &str,
        completion_content: &mut Vec<Content>,
        read_only_intent: Option<ReviewIntent>,
        implementation_intent: Option<ImplementationIntent>,
        expected_mutation: bool,
        requested_validation: bool,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        review_repair: &mut ReviewRepairState,
        progress_tracker: &mut ProgressTracker,
        silent_continues: &mut u32,
        generic_completion_retries: &mut u32,
        continue_total_nudges: &mut u32,
        force_tools_next: &mut bool,
        force_text_answer_next: &mut bool,
        text_tool_fallback_next: &mut bool,
        stalled_repeating: &mut bool,
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
        // Structured state detects an unfinished turn: the plan has
        // pending/active steps, or a structured long-horizon goal still has
        // leftover drive work. The
        //    model's `update_plan` checklist can be empty or all-done
        //    while the Goal still has remaining sub-goals — treating
        //    that recap as finished parks the drive.
        //
        // A *finished* response ends the turn cleanly: a final recap
        // after a multi-step task with a complete plan, or a plain
        // Q&A answer. Bounded so it can't loop forever.
        let leftover_goal = self
            .goals
            .structured
            .as_ref()
            .is_some_and(crate::goal::Goal::should_auto_drive);
        let plan_incomplete = self.goals.plan_incomplete() || leftover_goal;
        // Prefer a goal-aware continue when leftover Goal work is why the
        // turn is unfinished — the model's `update_plan` checklist may already
        // look complete.
        let continue_nudge = if leftover_goal {
            GOAL_CONTINUE_NUDGE
        } else {
            PLAN_CONTINUE_NUDGE
        };
        // A plan can be structurally incomplete while every remaining step is
        // blocked on a live background process. A status answer is then the
        // correct terminal outcome — re-nudging "continue with the next
        // pending step" only makes the model poll again (observed in real
        // transcripts: 85 plan nudges in one turn babysitting two downloads,
        // each cycle a full model round). The waiting classifier in
        // steer_after_tools sets `awaiting_background` after consecutive
        // waiting rounds; any non-waiting tool round clears it.
        if progress_tracker.awaiting_background && !assistant_text.trim().is_empty() {
            if buffer_read_only_review_text {
                let text_to_emit = if buffered_assistant_text.is_empty() {
                    assistant_text
                } else {
                    buffered_assistant_text
                };
                self.emit_assistant_text(ui, text_to_emit);
                ui.assistant_end();
            }
            self.messages
                .push_assistant(std::mem::take(completion_content));
            *stalled_repeating = false;
            *stalled_unfinished = false;
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_stall_reason.clear();
            progress_tracker.record(ProgressKind::Weak, AWAITING_BACKGROUND_REASON, None);
            ui.status("background work continues; ending the turn with the status report");
            return RoundControl::BreakInner(false);
        }
        if let Some(intent) = read_only_intent
            && plan_incomplete
        {
            if evidence.inspection_sprawl_nudges > 0 {
                let sprawl_mode = crate::steering::AnswerRepairMode::SprawlForceAnswer;
                if review_repair.has_budget(sprawl_mode, budgets) {
                    assert!(
                        review_repair.spend(sprawl_mode, evidence, budgets),
                        "sprawl force-answer spend must succeed after has_budget"
                    );
                    *continue_total_nudges += 1;
                    *force_text_answer_next = true;
                    ui.nudge(
                "review tried to continue inspecting after the sprawl limit; forcing a bounded answer from existing evidence",
            );
                    self.messages
                        .push_assistant(std::mem::take(completion_content));
                    self.messages.push_assistant_repair_note(sprawl_mode);
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        crate::steering::repair_nudge_with_required_next(
                            sprawl_mode,
                            summarize_inspected_evidence_nudge(intent, evidence),
                        ),
                    );
                    return RoundControl::Continue;
                }

                // Budget spent: accept a non-empty forced answer instead of stalling.
                if !assistant_text.trim().is_empty() && !plan_incomplete {
                    let _ = review_repair.exhausted(sprawl_mode);
                    let _ = intent;
                    ui.status(
                        "review sprawl force-answer budget spent; accepting the bounded answer",
                    );
                    // Fall through to normal emit/accept path below.
                } else if !assistant_text.trim().is_empty() {
                    // Still looks unfinished, but we have text — emit and end cleanly.
                    let _ = review_repair.exhausted(sprawl_mode);
                    let _ = intent;
                    if buffer_read_only_review_text {
                        let text_to_emit = if buffered_assistant_text.is_empty() {
                            assistant_text
                        } else {
                            buffered_assistant_text
                        };
                        self.emit_assistant_text(ui, text_to_emit);
                        ui.assistant_end();
                    }
                    self.messages
                        .push_assistant(std::mem::take(completion_content));
                    *stalled_repeating = false;
                    *stalled_unfinished = false;
                    progress_tracker.no_progress_streak = 0;
                    progress_tracker.last_stall_reason.clear();
                    progress_tracker.record_final_answer();
                    ui.status("review sprawl force-answer budget spent; accepting the last answer");
                    return RoundControl::BreakInner(false);
                } else {
                    if self.keep_working_after_stall(
                        progress_tracker,
                        force_tools_next,
                        stalled_unfinished,
                        stalled_repeating,
                        Some(continue_total_nudges),
                        ui,
                    ) {
                        self.messages
                            .push_assistant(std::mem::take(completion_content));
                        return RoundControl::Continue;
                    }
                    *stalled_unfinished = true;
                    let reason = review_repair.exhausted(sprawl_mode);
                    progress_tracker.record(ProgressKind::None, reason, None);
                    let _ = intent;
                    ui.status(&self.incomplete_turn_status(reason));
                    return RoundControl::BreakInner(false);
                }
            }

            if *silent_continues < self.config.loop_limits.max_silent_continues {
                self.messages
                    .push_assistant(std::mem::take(completion_content));
                *silent_continues += 1;
                *continue_total_nudges += 1;
                *force_tools_next = true;
                self.messages
                    .push_nudge(NudgeKind::Continue, continue_nudge);
                return RoundControl::Continue;
            }
        }
        // Table-driven implementation completeness (order = IMPLEMENTATION_COMPLETENESS_CASCADE).
        // Ordinary expected_mutation turns get the no-change gate for finished
        // answers, including after read/fetch/wait tools. Unfinished narration
        // and incomplete plans take the existing continuation paths below.
        let finished_text_answer = !plan_incomplete;
        // Escape hatch: the no-change nudge asks the model to either edit or
        // state plainly that no file changes are needed. A challenged model
        // that explicitly declines mutation has answered the challenge —
        // accept the finished text as the deliverable instead of exhausting
        // the cascade into a stall. A stall therefore always means "the model
        // agreed work was owed and did not do it", never "the model disagreed
        // with the prompt classifier".
        //
        // Do not take this hatch while a structured goal still has leftover
        // drive work — "already done" with 9/9 remaining is a stall, not a
        // finished answer.
        let mutation_declined_after_challenge = implementation_tracker.no_change_nudges > 0
            && !implementation_tracker.mutation_seen
            && finished_text_answer
            && crate::steering::answer_declines_mutation(assistant_text);
        if mutation_declined_after_challenge {
            ui.status("model states no file changes are needed; accepting the text answer");
        }
        // Declining an edit resolves only the mutation obligation. A separate
        // explicit request to run tests/checks still needs tool evidence.
        let gated_implementation_intent = (!mutation_declined_after_challenge)
            .then_some(implementation_intent)
            .flatten();
        let gated_expected_mutation = expected_mutation && !mutation_declined_after_challenge;
        {
            match super::impl_cascade::select_implementation_completeness(
                gated_implementation_intent,
                gated_expected_mutation,
                requested_validation,
                finished_text_answer,
                implementation_tracker,
            ) {
                Some(super::impl_cascade::ImplementationCascadeAction::Repair {
                    gate,
                    status,
                    nudge_body,
                    force_tools,
                    text_tool_fallback,
                }) => {
                    super::impl_cascade::spend_implementation_gate(gate, implementation_tracker);
                    evidence.quality_repair_nudges =
                        evidence.quality_repair_nudges.saturating_add(1);
                    *continue_total_nudges = continue_total_nudges.saturating_add(1);
                    *force_tools_next = force_tools;
                    *text_tool_fallback_next = text_tool_fallback;
                    ui.nudge(status);
                    self.messages
                        .push_assistant(std::mem::take(completion_content));
                    self.messages.push_nudge(NudgeKind::Continue, nudge_body);
                    return RoundControl::Continue;
                }
                Some(super::impl_cascade::ImplementationCascadeAction::Exhausted { status }) => {
                    if self.keep_working_after_stall(
                        progress_tracker,
                        force_tools_next,
                        stalled_unfinished,
                        stalled_repeating,
                        Some(continue_total_nudges),
                        ui,
                    ) {
                        self.messages
                            .push_assistant(std::mem::take(completion_content));
                        return RoundControl::Continue;
                    }
                    *stalled_unfinished = true;
                    ui.nudge(status);
                    ui.status(
                        &self.incomplete_turn_status("implementation_completeness_exhausted"),
                    );
                    return RoundControl::BreakInner(false);
                }
                None => {}
            }
        }
        // Table-driven review quality cascade (order = REVIEW_QUALITY_CASCADE).
        match super::cascade::select_review_quality_repair(
            read_only_intent,
            evidence,
            assistant_text,
            review_repair,
            budgets,
        ) {
            Some(super::cascade::QualityCascadeAction::Repair {
                mode,
                status,
                nudge_body,
                force_tools,
                force_text,
            }) => {
                assert!(
                    review_repair.spend(mode, evidence, budgets),
                    "answer-repair spend must succeed after cascade has_budget for {}",
                    mode.key()
                );
                *force_tools_next = force_tools;
                *force_text_answer_next = force_text;
                ui.nudge(&status);
                // Some modes use ui.status historically; keep nudge for all for visibility.
                self.messages.push_assistant_repair_note(mode);
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    repair_nudge_with_required_next(mode, nudge_body),
                );
                return RoundControl::Continue;
            }
            Some(super::cascade::QualityCascadeAction::Exhausted { mode, status }) => {
                if self.keep_working_after_stall(
                    progress_tracker,
                    force_tools_next,
                    stalled_unfinished,
                    stalled_repeating,
                    Some(continue_total_nudges),
                    ui,
                ) {
                    self.messages
                        .push_assistant(std::mem::take(completion_content));
                    return RoundControl::Continue;
                }
                *stalled_unfinished = true;
                let reason = review_repair.exhausted(mode);
                progress_tracker.record(ProgressKind::None, reason, None);
                ui.nudge(&status);
                ui.status(&self.incomplete_turn_status(reason));
                return RoundControl::BreakInner(false);
            }
            None => {}
        }
        // A syntactically valid but content-free completion claim is never a
        // user answer. Give ordinary Q&A and already-satisfied implementation
        // turns one compact retry; if the provider repeats the same canned
        // phrase, stop incomplete instead of presenting a false success. Review
        // turns normally reach this only after their evidence-specific repair
        // cascade has declined to act.
        if crate::steering::answer_is_generic_completion_placeholder(assistant_text)
            && !self
                .task
                .last_task_prompt
                .as_deref()
                .is_some_and(crate::task_contract::prompt_requests_exact_text_response)
        {
            const MAX_GENERIC_COMPLETION_RETRIES: u32 = 1;
            if *generic_completion_retries < MAX_GENERIC_COMPLETION_RETRIES {
                *generic_completion_retries += 1;
                *continue_total_nudges = continue_total_nudges.saturating_add(1);
                // Do not replay the rejected phrase: weak models imitate it.
                self.messages.push_assistant(vec![Content::Text(
                    "[answer retry: generic completion placeholder rejected; provide the actual result]"
                        .into(),
                )]);
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    "The previous response only claimed completion and did not answer the user's request. Provide the concrete answer or result now. If repository evidence is genuinely needed, inspect it with an available tool; otherwise answer directly. Do not repeat a generic completion phrase.",
                );
                // A completed mutation already has its evidence. Keep its retry
                // text-only so the model summarizes instead of starting new
                // work; unanswered Q&A remains Auto and may inspect if needed.
                *force_text_answer_next = implementation_tracker.mutation_seen;
                *force_tools_next = false;
                ui.nudge(
                    "model returned only a completion placeholder; requesting the actual result",
                );
                return RoundControl::Continue;
            }

            self.messages.push_assistant(vec![Content::Text(
                "[answer rejected: generic completion placeholder repeated]".into(),
            )]);
            *stalled_unfinished = true;
            progress_tracker.record(ProgressKind::None, "generic_completion_placeholder", None);
            ui.nudge("model repeated a completion placeholder without providing a result");
            ui.status(&self.incomplete_turn_status("generic_completion_placeholder"));
            return RoundControl::BreakInner(false);
        }
        if buffer_read_only_review_text {
            let text_to_emit = if buffered_assistant_text.is_empty() {
                assistant_text
            } else {
                buffered_assistant_text
            };
            self.emit_assistant_text(ui, text_to_emit);
            ui.assistant_end();
        }
        self.messages
            .push_assistant(std::mem::take(completion_content));
        if plan_incomplete && *silent_continues < self.config.loop_limits.max_silent_continues {
            // A real final answer after a forced no-progress recovery resolves
            // pending unfinished state from the preceding tool round.
            *stalled_repeating = false;
            *stalled_unfinished = false;
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_stall_reason.clear();
            *silent_continues += 1;
            *continue_total_nudges += 1;
            // Force the next round to actually call a tool, so the
            // nudge can't be answered with yet another narration or an
            // empty completion.
            *force_tools_next = true;
            // Use a goal-aware or plan-aware nudge so the model knows to
            // continue leftover drive work rather than recap and stop.
            self.messages
                .push_nudge(NudgeKind::Continue, continue_nudge);
            return RoundControl::Continue;
        }
        // Stall budgets spent: keep working in-turn instead of asking the
        // user to `/retry`. If that recovery is also spent, settle without
        // a retry prompt.
        if plan_incomplete {
            if self.keep_working_after_stall(
                progress_tracker,
                force_tools_next,
                stalled_unfinished,
                stalled_repeating,
                Some(continue_total_nudges),
                ui,
            ) {
                return RoundControl::Continue;
            }
            progress_tracker.record(
                ProgressKind::Weak,
                "structured plan remains incomplete",
                None,
            );
            if *silent_continues > 0 {
                ui.status(&self.incomplete_turn_status("plan_incomplete_without_progress"));
            }
        } else {
            *stalled_repeating = false;
            *stalled_unfinished = false;
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_stall_reason.clear();
            progress_tracker.record_final_answer();
        }
        RoundControl::BreakInner(false)
    }
}
