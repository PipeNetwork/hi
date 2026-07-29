//! Text-only Steer path: unfinished continues, **answer-repair** quality nudges,
//! and implementation completeness gates when no tools were called.
//!
//! Answer repair (`ReviewRepairMode` / `ReviewRepairBudgets`) is distinct from
//! post-mutation **completion review** (`ReviewPolicy` → `ReviewStatus`) and
//! the long-horizon **goal skeptic**.

use hi_ai::Content;

use crate::heuristics::looks_like_unfinished_step;
use crate::steering::{
    EvidenceTracker, ImplementationIntent, ImplementationTracker, ReviewIntent,
    repair_nudge_with_required_next, summarize_inspected_evidence_nudge,
};
use crate::transcript::NudgeKind;
use crate::{PLAN_CONTINUE_NUDGE, SILENT_CONTINUE_NUDGE, Ui};

use super::super::phase::TurnPhase;
use super::super::progress::{AWAITING_BACKGROUND_REASON, ProgressKind, ProgressTracker};
use super::super::retry::{INCOMPLETE_STATUS, ReviewRepairState};
use super::RoundControl;

impl crate::Agent {
    /// Post-model Steer when the model returned text and no tool calls.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent::turn) fn steer_without_tools(
        &mut self,
        assistant_text: &str,
        completion_content: &mut Vec<Content>,
        read_only_intent: Option<ReviewIntent>,
        implementation_intent: Option<ImplementationIntent>,
        expected_mutation: bool,
        made_tool_call: bool,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        review_repair: &mut ReviewRepairState,
        progress_tracker: &mut ProgressTracker,
        silent_continues: &mut u32,
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
        let plan_incomplete = self.goals.plan_incomplete();
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
            && (looks_unfinished || plan_incomplete)
        {
            if evidence.inspection_sprawl_nudges > 0 {
                let sprawl_mode = crate::steering::ReviewRepairMode::SprawlForceAnswer;
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

                *stalled_unfinished = true;
                let reason = review_repair.exhausted(sprawl_mode);
                progress_tracker.record(ProgressKind::None, reason, None);
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
        // Table-driven implementation completeness (order = IMPLEMENTATION_COMPLETENESS_CASCADE).
        // For ordinary expected_mutation turns: finished text + never used tools only.
        // Unfinished narration, incomplete plans, and tool-using turns take the paths below.
        let finished_text_answer = !looks_unfinished && !plan_incomplete;
        let text_only_turn = !made_tool_call;
        // Escape hatch: the no-change nudge asks the model to either edit or
        // state plainly that no file changes are needed. A challenged model
        // that explicitly declines mutation has answered the challenge —
        // accept the finished text as the deliverable instead of exhausting
        // the cascade into a stall. A stall therefore always means "the model
        // agreed work was owed and did not do it", never "the model disagreed
        // with the prompt classifier".
        let mutation_declined_after_challenge = implementation_tracker.no_change_nudges > 0
            && !implementation_tracker.mutation_seen
            && finished_text_answer
            && crate::steering::answer_declines_mutation(assistant_text);
        if mutation_declined_after_challenge {
            ui.status("model states no file changes are needed; accepting the text answer");
        } else {
            match super::impl_cascade::select_implementation_completeness(
                implementation_intent,
                expected_mutation,
                finished_text_answer,
                text_only_turn,
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
                    *stalled_unfinished = true;
                    ui.nudge(status);
                    ui.status(INCOMPLETE_STATUS);
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
                note_mode,
                spend,
            }) => {
                if spend {
                    assert!(
                        review_repair.spend(mode, evidence, budgets),
                        "answer-repair spend must succeed after cascade has_budget for {}",
                        mode.key()
                    );
                    // Disclaimer dual-spend: primary spend *and* note(chat_attempt)
                    // on one nudge. Chat-attempt is a secondary ceiling that
                    // counts every disclaimer repair, including primary ones.
                    if let Some(note) = note_mode {
                        review_repair.note(note);
                    }
                } else {
                    // Primary disclaimer budget exhausted: nudge via chat-attempt
                    // accounting only (no primary spend).
                    review_repair.note_quality(note_mode, evidence);
                }
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
                *stalled_unfinished = true;
                let reason = review_repair.exhausted(mode);
                progress_tracker.record(ProgressKind::None, reason, None);
                ui.nudge(&status);
                ui.status(INCOMPLETE_STATUS);
                return RoundControl::BreakInner(false);
            }
            None => {}
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
        if (looks_unfinished || plan_incomplete)
            && *silent_continues < self.config.loop_limits.max_silent_continues
        {
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
            progress_tracker.record(ProgressKind::Weak, "text answer looked unfinished", None);
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
