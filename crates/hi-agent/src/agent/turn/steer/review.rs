//! Text-answer steering: unfinished work and implementation completeness.

use hi_ai::Content;

use crate::steering::{
    BAIL_CONTINUE_NUDGE, EvidenceTracker, GoalKind, IMPLEMENTATION_REVIEW_WRAPUP_NUDGE,
    ImplementationIntent, ImplementationTracker, LazinessCategory, LazinessConfig,
    LazinessDecision, NoNudgeReason, ReviewIntent, TodoGateDecision, build_laziness_nudge,
    claim_evidence_category, evaluate_laziness, evaluate_todo_gate, matched_bail_out,
    todo_gate_input_from_plan,
};
use crate::transcript::NudgeKind;
use crate::{GOAL_CONTINUE_NUDGE, PLAN_CONTINUE_NUDGE, Ui};

use super::super::phase::TurnPhase;
use super::super::progress::{AWAITING_BACKGROUND_REASON, ProgressKind, ProgressTracker};
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
        goal_kind: GoalKind,
        requested_validation: bool,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
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
        // ChatOnly wrap-up can still parse a textual tool call into
        // `completion_content`. This path records text only; pairing those
        // calls would require results that never executed.
        completion_content.retain(|content| !matches!(content, Content::ToolCall { .. }));
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
        //
        // Grok-build kind lens: analysis/research judge the written
        // deliverable. A leftover implementer checklist from a prior
        // code-change must not keep this turn alive. Structured goal
        // auto-drive still continues.
        let checklist_incomplete = !self.plan_mode && self.goals.plan_incomplete();
        let plan_incomplete = leftover_goal
            || (checklist_incomplete
                && (goal_kind.requires_workspace_evidence() || progress_tracker.plan_updated));
        // Prefer a goal-aware continue when leftover Goal work is why the
        // turn is unfinished — the model's `update_plan` checklist may already
        // look complete. A last-paragraph bail-out gets grok-build's preface.
        let continue_nudge = {
            let body = if leftover_goal {
                self.goals
                    .structured
                    .as_ref()
                    .map(crate::goal::continuation::mid_turn_continue_nudge)
                    .unwrap_or_else(|| GOAL_CONTINUE_NUDGE.to_string())
            } else {
                PLAN_CONTINUE_NUDGE.to_string()
            };
            if plan_incomplete && matched_bail_out(assistant_text).is_some() {
                format!("{BAIL_CONTINUE_NUDGE}\n\n{body}")
            } else {
                body
            }
        };
        // A plan can be structurally incomplete while every remaining step is
        // blocked on a live background process. A status answer is then the
        // correct terminal outcome — re-nudging "continue with the next
        // pending step" only makes the model poll again (observed in real
        // transcripts: 85 plan nudges in one turn babysitting two downloads,
        // each cycle a full model round). The waiting classifier in
        // steer_after_tools sets `awaiting_background` after consecutive
        // waiting rounds; any non-waiting tool round clears it.
        // Live: "review for any major issues and fix", 4 unique reads, withhold,
        // this dump, outstanding todos continue, cargo test green, no_progress.
        // Reject it on a fix turn even with no mutation and even if a plan is
        // still incomplete — todo-gate must not keep the dump alive.
        if read_only_intent.is_none()
            && (expected_mutation || implementation_intent.is_some())
            && crate::steering::answer_is_review_shaped_insufficient_evidence(assistant_text)
        {
            return self.reject_review_shaped_wrapup_on_fix(
                assistant_text,
                completion_content,
                implementation_tracker,
                evidence,
                progress_tracker,
                continue_total_nudges,
                force_tools_next,
                text_tool_fallback_next,
                buffer_read_only_review_text,
                force_text_answer_next,
                ui,
            );
        }
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
            return Ok(RoundControl::Finish(
                crate::agent::turn::ModelLoopDecision::Verify,
            ));
        }
        if read_only_intent.is_some()
            && plan_incomplete
            && *silent_continues < self.config.loop_limits.max_silent_continues
        {
            return self.fire_todo_gate(
                completion_content,
                assistant_text,
                progress_tracker,
                silent_continues,
                continue_total_nudges,
                force_tools_next,
                leftover_goal,
                &continue_nudge,
                ui,
            );
        }
        // Table-driven implementation completeness (order = IMPLEMENTATION_COMPLETENESS_CASCADE).
        // Ordinary expected_mutation turns get the no-change gate for finished
        // answers, including after read/fetch/wait tools. Unfinished narration
        // and incomplete plans take the existing continuation paths below.
        let finished_text_answer = !plan_incomplete;
        if finished_text_answer
            && self.laziness_should_nudge(
                assistant_text,
                goal_kind,
                implementation_tracker.tests_seen,
                false,
                progress_tracker.awaiting_background,
                progress_tracker.laziness_nudges,
            )
        {
            self.messages
                .push_assistant(std::mem::take(completion_content));
            self.maybe_fire_laziness(
                assistant_text,
                goal_kind,
                implementation_tracker.tests_seen,
                false,
                progress_tracker.awaiting_background,
                progress_tracker,
                force_tools_next,
                ui,
            );
            return Ok(RoundControl::Continue);
        }
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
                    self.messages.push_nudge(NudgeKind::Steer, nudge_body);
                    return Ok(RoundControl::Continue);
                }
                Some(super::impl_cascade::ImplementationCascadeAction::Exhausted {
                    gate,
                    status,
                }) => {
                    if gate == super::impl_cascade::ImplementationGate::NoChanges
                        && self.try_no_progress_recovery(
                            progress_tracker,
                            force_tools_next,
                            Some(continue_total_nudges),
                            ui,
                        )
                    {
                        self.messages
                            .push_assistant(std::mem::take(completion_content));
                        return Ok(RoundControl::Continue);
                    }
                    if !implementation_tracker.mutation_seen
                        && (gated_expected_mutation || gated_implementation_intent.is_some())
                    {
                        implementation_tracker.no_mutation_exhausted = true;
                        progress_tracker.record(
                            ProgressKind::None,
                            "implementation repair exhausted without a mutation",
                            None,
                        );
                    }
                    ui.nudge(status);
                    // Preserve the best text for context, but keep the sticky
                    // semantic failure above. With no landed mutation there is
                    // no verification stage that can turn this into success.
                }
                None => {}
            }
        }
        // A review answer is model-authored content. Do not reject or rewrite
        // it based on evidence counts, disclaimer phrases, or required headings.
        // Execution and verification outcomes remain independently enforced.
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
                implementation_tracker.no_mutation_exhausted = true;
                progress_tracker.record(
                    ProgressKind::None,
                    "implementation answer exhausted without a mutation",
                    None,
                );
                ui.status("no file changes were made; ending the turn without another retry");
                return Ok(RoundControl::Finish(
                    crate::agent::turn::ModelLoopDecision::Verify,
                ));
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
                    return Ok(RoundControl::Finish(
                        crate::agent::turn::ModelLoopDecision::Verify,
                    ));
                }
                // A repeated canned completion is not a usable result for an
                // unchanged unfinished checklist. Accepting it as a successful
                // turn makes the frontend enqueue the same synthetic drive
                // again until the cross-turn stall guard parks it. Preserve
                // the durable plan and settle this bounded semantic failure as
                // no-progress. Returning Err here used to make frontend cleanup
                // mislabel it as verification infrastructure failure.
                progress_tracker.bounded_plan_answer_recovery_exhausted = true;
                progress_tracker.record(
                    ProgressKind::None,
                    "generic completion after bounded plan recovery",
                    None,
                );
                self.messages.push_assistant(vec![Content::Text(
                    "[answer rejected: generic completion placeholder repeated]".into(),
                )]);
                ui.nudge("model repeated a generic completion response without advancing the plan");
                ui.status("model did not produce a usable plan result after bounded recovery");
                return Ok(RoundControl::Finish(
                    crate::agent::turn::ModelLoopDecision::Verify,
                ));
            }
            progress_tracker.record(ProgressKind::None, "generic_completion_placeholder", None);
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
            return self.fire_todo_gate_after_assistant(
                assistant_text,
                progress_tracker,
                silent_continues,
                continue_total_nudges,
                force_tools_next,
                leftover_goal,
                &continue_nudge,
                ui,
            );
        }
        if plan_incomplete {
            progress_tracker.record(
                ProgressKind::Weak,
                "structured plan has remaining steps",
                None,
            );
            progress_tracker.record_final_answer();
            ui.status("todo gate spent; leaving remaining plan steps for the next turn");
            return Ok(RoundControl::Finish(
                crate::agent::turn::ModelLoopDecision::Verify,
            ));
        }
        if self.maybe_fire_laziness(
            assistant_text,
            goal_kind,
            implementation_tracker.tests_seen,
            false,
            progress_tracker.awaiting_background,
            progress_tracker,
            force_tools_next,
            ui,
        ) {
            return Ok(RoundControl::Continue);
        }
        if !implementation_tracker.no_mutation_exhausted {
            progress_tracker.no_progress_streak = 0;
            progress_tracker.last_no_progress_reason.clear();
            progress_tracker.record_final_answer();
        }
        Ok(RoundControl::Finish(
            crate::agent::turn::ModelLoopDecision::Verify,
        ))
    }

    fn reject_review_shaped_wrapup_on_fix(
        &mut self,
        _assistant_text: &str,
        completion_content: &mut Vec<Content>,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        progress_tracker: &mut ProgressTracker,
        continue_total_nudges: &mut u32,
        force_tools_next: &mut bool,
        text_tool_fallback_next: &mut bool,
        buffer_read_only_review_text: bool,
        force_text_answer_next: &mut bool,
        ui: &mut dyn Ui,
    ) -> anyhow::Result<RoundControl> {
        if implementation_tracker.unusable_wrapup_nudges < 1 {
            implementation_tracker.unusable_wrapup_nudges = implementation_tracker
                .unusable_wrapup_nudges
                .saturating_add(1);
            evidence.quality_repair_nudges = evidence.quality_repair_nudges.saturating_add(1);
            *continue_total_nudges = continue_total_nudges.saturating_add(1);
            *force_tools_next = true;
            *text_tool_fallback_next = false;
            ui.nudge("review-shaped wrap-up is not an implementation; requesting the actual fix");
            self.messages.push_assistant(vec![Content::Text(
                "[answer retry: review-shaped wrap-up rejected on a fix request; continue implementing]"
                    .into(),
            )]);
            self.messages
                .push_nudge(NudgeKind::Steer, IMPLEMENTATION_REVIEW_WRAPUP_NUDGE);
            return Ok(RoundControl::Continue);
        }
        const FALLBACK: &str = "The requested fix was not completed. A review-style 'insufficient evidence' wrap-up is not an implementation.";
        if buffer_read_only_review_text || !*force_text_answer_next {
            self.emit_assistant_text(ui, FALLBACK);
            ui.assistant_end();
        }
        self.messages
            .push_assistant(vec![Content::Text(FALLBACK.into())]);
        if !implementation_tracker.mutation_seen {
            implementation_tracker.no_mutation_exhausted = true;
        }
        let _ = completion_content;
        progress_tracker.record(
            ProgressKind::None,
            "review-shaped wrap-up on a fix request",
            None,
        );
        ui.status(
            "review-shaped wrap-up is not an implementation; ending without treating the turn as complete",
        );
        Ok(RoundControl::Finish(
            crate::agent::turn::ModelLoopDecision::Verify,
        ))
    }

    fn fire_todo_gate(
        &mut self,
        completion_content: &mut Vec<Content>,
        assistant_text: &str,
        progress_tracker: &mut ProgressTracker,
        silent_continues: &mut u32,
        continue_total_nudges: &mut u32,
        force_tools_next: &mut bool,
        leftover_goal: bool,
        continue_nudge: &str,
        ui: &mut dyn Ui,
    ) -> anyhow::Result<RoundControl> {
        self.messages
            .push_assistant(std::mem::take(completion_content));
        self.fire_todo_gate_after_assistant(
            assistant_text,
            progress_tracker,
            silent_continues,
            continue_total_nudges,
            force_tools_next,
            leftover_goal,
            continue_nudge,
            ui,
        )
    }

    fn fire_todo_gate_after_assistant(
        &mut self,
        assistant_text: &str,
        progress_tracker: &mut ProgressTracker,
        silent_continues: &mut u32,
        continue_total_nudges: &mut u32,
        force_tools_next: &mut bool,
        leftover_goal: bool,
        continue_nudge: &str,
        ui: &mut dyn Ui,
    ) -> anyhow::Result<RoundControl> {
        let input =
            todo_gate_input_from_plan(&self.goals.last_plan, progress_tracker.awaiting_background);
        let decision = if leftover_goal {
            TodoGateDecision::Nudge {
                reminder: continue_nudge.to_string(),
                reason: crate::steering::TodoGateReason::InFlight,
            }
        } else {
            evaluate_todo_gate(&input)
        };
        match decision {
            TodoGateDecision::Nudge { reminder, .. } => {
                progress_tracker.no_progress_streak = 0;
                progress_tracker.last_no_progress_reason.clear();
                *silent_continues += 1;
                *continue_total_nudges += 1;
                // Analysis/research leftover work is a write-up, not a mutation.
                // Forcing a tool call after a cited recap only pads the turn.
                *force_tools_next = !leftover_goal
                    || self
                        .goals
                        .structured
                        .as_ref()
                        .is_some_and(|goal| goal.kind.requires_workspace_evidence());
                ui.status("outstanding todos remain; continuing this turn");
                let reminder = if matched_bail_out(assistant_text).is_some() {
                    format!("{BAIL_CONTINUE_NUDGE}\n\n{reminder}")
                } else {
                    reminder
                };
                self.messages.push_nudge(NudgeKind::TodoGate, reminder);
                Ok(RoundControl::Continue)
            }
            TodoGateDecision::Continue => {
                let _ = assistant_text;
                Ok(RoundControl::Finish(
                    crate::agent::turn::ModelLoopDecision::Verify,
                ))
            }
        }
    }

    fn laziness_should_nudge(
        &self,
        assistant_text: &str,
        goal_kind: GoalKind,
        tests_seen: bool,
        plan_incomplete: bool,
        awaiting_background: bool,
        laziness_nudges: u32,
    ) -> bool {
        matches!(
            self.laziness_decision(
                assistant_text,
                goal_kind,
                tests_seen,
                plan_incomplete,
                awaiting_background,
                laziness_nudges,
            ),
            LazinessDecision::Nudge { .. }
        )
    }

    fn laziness_decision(
        &self,
        assistant_text: &str,
        goal_kind: GoalKind,
        tests_seen: bool,
        plan_incomplete: bool,
        awaiting_background: bool,
        laziness_nudges: u32,
    ) -> LazinessDecision {
        if !goal_kind.requires_workspace_evidence() {
            return LazinessDecision::NoNudge {
                category: LazinessCategory::NotStalledComplete,
                confidence: 1.0,
                reason: NoNudgeReason::NotStalled,
            };
        }
        let category = claim_evidence_category(
            assistant_text,
            goal_kind,
            tests_seen,
            plan_incomplete,
            awaiting_background,
        );
        let parsed = crate::steering::ClassifierOutput {
            category,
            confidence: 1.0,
            evidence: "turn-end claim vs tool evidence".into(),
        };
        evaluate_laziness(&parsed, &LazinessConfig::default(), laziness_nudges, 0.7)
    }

    fn maybe_fire_laziness(
        &mut self,
        assistant_text: &str,
        goal_kind: GoalKind,
        tests_seen: bool,
        plan_incomplete: bool,
        awaiting_background: bool,
        progress_tracker: &mut ProgressTracker,
        force_tools_next: &mut bool,
        ui: &mut dyn Ui,
    ) -> bool {
        let LazinessDecision::Nudge {
            category, evidence, ..
        } = self.laziness_decision(
            assistant_text,
            goal_kind,
            tests_seen,
            plan_incomplete,
            awaiting_background,
            progress_tracker.laziness_nudges,
        )
        else {
            return false;
        };
        progress_tracker.laziness_nudges = progress_tracker.laziness_nudges.saturating_add(1);
        *force_tools_next = true;
        ui.status("idle-stall detector: continuing without branding no-progress");
        self.messages.push_nudge(
            NudgeKind::Laziness,
            build_laziness_nudge(category, &evidence),
        );
        true
    }
}
