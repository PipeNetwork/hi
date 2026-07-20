//! Text-only Steer path: unfinished continues, review-answer repairs,
//! and implementation completeness gates when no tools were called.

use hi_ai::Content;

use crate::heuristics::looks_like_unfinished_step;
use crate::steering::{
    EvidenceTracker, IMPLEMENTATION_NO_CHANGES_NUDGE, IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE,
    ImplementationIntent, ImplementationTracker, ReviewIntent,
    implementation_missing_validation_nudge, implementation_text_tool_nudge,
    repair_nudge_with_required_next, summarize_inspected_evidence_nudge,
};
use crate::transcript::NudgeKind;
use crate::{PLAN_CONTINUE_NUDGE, SILENT_CONTINUE_NUDGE, Ui};

use super::super::phase::TurnPhase;
use super::super::progress::{ProgressKind, ProgressTracker};
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
let plan_incomplete = self.goals.plan_incomplete();
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
            let _ = review_repair.spend(mode, evidence, budgets);
        } else {
            evidence.quality_repair_nudges =
                evidence.quality_repair_nudges.saturating_add(1);
        }
        if let Some(note) = note_mode {
            review_repair.note(note);
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
}

