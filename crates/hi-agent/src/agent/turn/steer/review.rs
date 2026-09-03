//! Text-answer Steer path: unfinished continues, **answer-repair** quality
//! nudges, and implementation completeness gates.
//!
//! Answer repair (`ReviewRepairMode` / `ReviewRepairBudgets`) is distinct from
//! post-mutation **completion review** (`ReviewPolicy` → `ReviewStatus`) and
//! the long-horizon **goal skeptic**.

use hi_ai::Content;

use crate::steering::{
    EvidenceTracker, ImplementationIntent, ImplementationTracker, ReviewIntent,
    repair_nudge_with_required_next,
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
        buffered_assistant_text: &mut String,
        buffer_read_only_review_text: bool,
        _steps: u32,
        ui: &mut dyn Ui,
    ) -> anyhow::Result<RoundControl> {
        self.set_turn_phase(TurnPhase::Steer);
        let budgets = self.config.loop_limits.review_repair.clone();
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
        // Plan mode is deliberately a planning-only turn. Pending checklist
        // items are the successful output of that turn, not evidence that the
        // turn is unfinished. Feeding PLAN_CONTINUE_NUDGE here creates an
        // impossible instruction cycle ("do the work" while mutating tools are
        // unavailable) and invites the model to self-certify every step as
        // done just to escape the loop.
        let plan_incomplete = !self.plan_mode && (self.goals.plan_incomplete() || leftover_goal);
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
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_no_progress_reason.clear();
            progress_tracker.record(ProgressKind::Weak, AWAITING_BACKGROUND_REASON, None);
            ui.status("background work continues; ending the turn with the status report");
            return Ok(RoundControl::BreakInner(false));
        }
        if read_only_intent.is_some() && plan_incomplete {
            if *silent_continues < self.config.loop_limits.max_silent_continues {
                self.messages
                    .push_assistant(std::mem::take(completion_content));
                *silent_continues += 1;
                *continue_total_nudges += 1;
                *force_tools_next = true;
                self.messages
                    .push_nudge(NudgeKind::Continue, continue_nudge);
                return Ok(RoundControl::Continue);
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
                    return Ok(RoundControl::Continue);
                }
                Some(super::impl_cascade::ImplementationCascadeAction::Exhausted { status }) => {
                    if self.try_no_progress_recovery(
                        progress_tracker,
                        force_tools_next,
                        Some(continue_total_nudges),
                        ui,
                    ) {
                        self.messages
                            .push_assistant(std::mem::take(completion_content));
                        return Ok(RoundControl::Continue);
                    }
                    ui.nudge(status);
                    // A bounded heuristic may challenge the answer, but it is
                    // not a terminal correctness oracle. Accept the best text
                    // below and let deterministic verification decide.
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
            &budgets,
        ) {
            Some(super::cascade::QualityCascadeAction::Repair {
                mode,
                status,
                nudge_body,
                force_tools,
                force_text,
            }) => {
                assert!(
                    review_repair.spend(mode, evidence, &budgets),
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
                return Ok(RoundControl::Continue);
            }
            Some(super::cascade::QualityCascadeAction::Exhausted { mode, status }) => {
                if self.try_no_progress_recovery(
                    progress_tracker,
                    force_tools_next,
                    Some(continue_total_nudges),
                    ui,
                ) {
                    self.messages
                        .push_assistant(std::mem::take(completion_content));
                    return Ok(RoundControl::Continue);
                }
                let reason = review_repair.exhausted(mode);
                progress_tracker.record(ProgressKind::None, reason, None);
                ui.nudge(&status);
                if matches!(mode, crate::steering::AnswerRepairMode::GapSearchOverclaim) {
                    // A contradiction between search evidence and the model's
                    // final claim is different from a merely format-weak
                    // answer. Never surface the overclaim as the result after
                    // its bounded repair budget is spent. Review text is
                    // buffered for read-only turns, so replacing it here keeps
                    // the user-visible answer honest and single-owned.
                    self.emit_deterministic_closeout(ui);
                    return Ok(RoundControl::BreakInner(false));
                }
                // Repair exhaustion is advisory. Preserve and return the
                // model's answer instead of manufacturing a failed turn.
            }
            None => {}
        }
        // A syntactically valid but content-free completion claim is never a
        // user answer. Give ordinary Q&A and already-satisfied implementation
        // turns one compact retry. If the provider repeats the same canned
        // phrase, return the available response; this heuristic must not create
        // a synthetic turn failure.
        if crate::steering::generic_completion_guards_enabled()
            && crate::steering::answer_is_generic_completion_placeholder(assistant_text)
            && !self
                .task
                .last_task_prompt
                .as_deref()
                .is_some_and(crate::task_contract::prompt_requests_exact_text_response)
        {
            // The implementation cascade has already challenged a no-op
            // twice. A third model request can only produce another canned
            // completion (and used to consume the provider's next response),
            // so settle with one truthful deterministic answer instead of
            // extending the turn or displaying the same phrase again.
            let no_change_recovery_exhausted = implementation_tracker.no_change_nudges >= 2
                && !implementation_tracker.mutation_seen
                && (expected_mutation || implementation_intent.is_some());
            if no_change_recovery_exhausted {
                const NO_CHANGE_FALLBACK: &str =
                    "No file changes were made; the requested implementation was not applied.";
                if buffer_read_only_review_text || !*force_text_answer_next {
                    self.emit_assistant_text(ui, NO_CHANGE_FALLBACK);
                    ui.assistant_end();
                }
                self.messages
                    .push_assistant(vec![Content::Text(NO_CHANGE_FALLBACK.into())]);
                progress_tracker.record_final_answer();
                ui.status("no file changes were made; ending the turn without another retry");
                return Ok(RoundControl::BreakInner(false));
            }
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
                return Ok(RoundControl::Continue);
            }

            progress_tracker.record(ProgressKind::None, "generic_completion_placeholder", None);
            if plan_incomplete {
                if implementation_tracker.mutation_seen || implementation_tracker.validation_seen {
                    // The model failed only at summarizing a productive turn.
                    // Keep the landed edits and let turn-end verification plus
                    // the next plan drive own the remaining checklist; treating
                    // this as a provider failure discards real progress from the
                    // frontend's control flow and stops auto-drive at N/M done.
                    const PARTIAL_PROGRESS_FALLBACK: &str = "Made concrete progress on the current step; the remaining plan is still pending.";
                    self.emit_assistant_text(ui, PARTIAL_PROGRESS_FALLBACK);
                    ui.assistant_end();
                    self.messages
                        .push_assistant(vec![Content::Text(PARTIAL_PROGRESS_FALLBACK.into())]);
                    progress_tracker.record(
                        ProgressKind::Weak,
                        "generic completion after plan progress",
                        None,
                    );
                    ui.status(
                        "model summary was unusable; keeping the completed work and continuing the plan",
                    );
                    return Ok(RoundControl::BreakInner(false));
                }
                // A repeated canned completion is not a usable result for an
                // unchanged unfinished checklist. Accepting it as a successful
                // turn makes the frontend enqueue the same synthetic drive
                // again until the cross-turn stall guard parks it. Preserve
                // the durable plan, but fail this bounded provider attempt now
                // so the frontend stops auto-driving and reports the real cause.
                self.messages.push_assistant(vec![Content::Text(
                    "[answer rejected: generic completion placeholder repeated]".into(),
                )]);
                ui.nudge("model repeated a generic completion response without advancing the plan");
                return Err(anyhow::anyhow!(
                    "model returned no usable final answer for the unfinished plan after bounded recovery"
                ));
            }
            ui.nudge("model repeated a generic completion response; returning the available text");
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
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_no_progress_reason.clear();
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
            return Ok(RoundControl::Continue);
        }
        // Once the plan-continuation budget is spent, try one different action.
        // If that is also spent, settle and leave the remaining plan durable for
        // the next drive turn.
        if plan_incomplete {
            if self.try_no_progress_recovery(
                progress_tracker,
                force_tools_next,
                Some(continue_total_nudges),
                ui,
            ) {
                return Ok(RoundControl::Continue);
            }
            progress_tracker.record(
                ProgressKind::Weak,
                "structured plan has remaining steps",
                None,
            );
        } else {
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_no_progress_reason.clear();
            progress_tracker.record_final_answer();
        }
        Ok(RoundControl::BreakInner(false))
    }
}
