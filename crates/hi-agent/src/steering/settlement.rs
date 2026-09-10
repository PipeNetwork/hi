//! Turn-end settlement vocabulary ported from grok-build's laziness detector.
//!
//! Grok-build classifies a content-only ending as one of:
//! `stalled_narration`, `stalled_permission_asking`, `stalled_false_completion`,
//! `not_stalled_complete`, `not_stalled_waiting_on_background`,
//! `not_stalled_waiting_on_user`.
//!
//! hi applies the same categories deterministically from [`GoalKind`] plus the
//! last-paragraph bail-out panel. Analysis/research treat the written answer as
//! the deliverable — offering to implement next is `not_stalled_waiting_on_user`,
//! not an unusable forced final.

use hi_ai::Content;

use super::goal_kind::GoalKind;
use super::stop_detector::matched_bail_out;
use crate::heuristics::parse_text_tool_calls;

/// Closed set matching grok-build's `LazinessCategory` (minus the todo-list
/// variant, which hi already covers via plan-incomplete continue).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StallCategory {
    StalledNarration,
    StalledPermissionAsking,
    StalledFalseCompletion,
    NotStalledComplete,
    NotStalledWaitingOnBackground,
    NotStalledWaitingOnUser,
}

impl StallCategory {
    pub(crate) fn is_stalled(self) -> bool {
        matches!(
            self,
            Self::StalledNarration | Self::StalledPermissionAsking | Self::StalledFalseCompletion
        )
    }
}

/// Classify a tool-free wrap-up / forced-final answer the way grok-build's
/// kind lens + last-paragraph stop detector would.
pub(crate) fn classify_text_answer(
    kind: GoalKind,
    text: &str,
    plan_incomplete: bool,
    awaiting_background: bool,
) -> StallCategory {
    let trimmed = text.trim();
    if awaiting_background && !trimmed.is_empty() {
        return StallCategory::NotStalledWaitingOnBackground;
    }
    if trimmed.is_empty() {
        return StallCategory::StalledNarration;
    }
    if parse_text_tool_calls(trimmed, 0)
        .iter()
        .any(|content| matches!(content, Content::ToolCall { .. }))
    {
        return StallCategory::StalledNarration;
    }
    if !kind.requires_workspace_evidence() {
        // Grok-build analysis/research: FINAL_RESPONSE is the deliverable.
        // A trailing "let me implement" is waiting on the user, not a stall.
        if offers_next_work(trimmed) {
            return StallCategory::NotStalledWaitingOnUser;
        }
        return StallCategory::NotStalledComplete;
    }
    if plan_incomplete {
        return StallCategory::StalledFalseCompletion;
    }
    match matched_bail_out(trimmed) {
        Some(super::stop_detector::PATTERN_PLEASE_DEFLECTION) => {
            StallCategory::StalledPermissionAsking
        }
        Some(_) => StallCategory::StalledNarration,
        None => StallCategory::NotStalledComplete,
    }
}

/// Whether a forced-final wrap-up should be rejected as unusable.
pub(crate) fn forced_final_answer_is_unusable(
    text: &str,
    plan_incomplete: bool,
    kind: GoalKind,
) -> bool {
    classify_text_answer(kind, text, plan_incomplete, false).is_stalled()
}

/// Grok-build's TodoGate fires to *continue*; once the harness already forced
/// a final, leftover "Fix issues" checklist items must not reject the wrap-up.
/// Green tests with no mutation are the deliverable when the model omitted
/// prose (provider-invisible / empty) after that force.
pub(crate) fn no_progress_forced_final_is_unusable(
    text: &str,
    kind: GoalKind,
    tests_seen: bool,
    mutation_seen: bool,
) -> bool {
    if text.trim().is_empty() {
        return !(tests_seen && !mutation_seen);
    }
    forced_final_answer_is_unusable(text, false, kind)
}

pub(crate) fn offers_next_work(text: &str) -> bool {
    let last = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .last()
        .unwrap_or(text);
    let lower = last.to_ascii_lowercase();
    [
        "let me ",
        "let's ",
        "i'll ",
        "i will ",
        "i can implement",
        "want me to",
        "should i ",
    ]
    .iter()
    .any(|cue| lower.contains(cue))
        && ![
            "let me know",
            "i'll be happy",
            "i'll let you",
            "i'll wait",
            "i'll stop",
        ]
        .iter()
        .any(|closing| lower.contains(closing))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_review_that_offers_to_implement_is_waiting_on_user() {
        let review = "I have all the source in context now. Here's my review.\n\n\
## Major issues found\n\n\
1. PRIVMSG leaks whether a username exists.\n\
2. NICK rename leaks rate-limiter entries.\n\
3. KICK broadcasts KICKED to every subscriber.\n\n\
Those are the highest-value fixes.\n\n\
Let me implement fixes for #1, #2, and #3.";
        assert_eq!(
            classify_text_answer(GoalKind::Analysis, review, false, false),
            StallCategory::NotStalledWaitingOnUser
        );
        assert!(!forced_final_answer_is_unusable(
            review,
            false,
            GoalKind::Analysis
        ));
        assert!(!forced_final_answer_is_unusable(
            "Let me implement the parser.",
            false,
            GoalKind::Analysis
        ));
    }

    #[test]
    fn code_change_bail_out_is_stalled() {
        assert!(forced_final_answer_is_unusable(
            "I can't proceed.",
            false,
            GoalKind::CodeChange
        ));
        assert!(forced_final_answer_is_unusable(
            "The parser is fixed.",
            true,
            GoalKind::CodeChange
        ));
        assert!(!forced_final_answer_is_unusable(
            "The parser is fixed.",
            false,
            GoalKind::CodeChange
        ));
        assert!(!forced_final_answer_is_unusable(
            "Ready for review.",
            false,
            GoalKind::CodeChange
        ));
    }

    #[test]
    fn empty_and_in_text_tool_calls_are_unusable() {
        assert!(forced_final_answer_is_unusable(
            "",
            false,
            GoalKind::Analysis
        ));
        assert!(forced_final_answer_is_unusable(
            r#"{"name": "read", "arguments": {"path": "src/a.rs"}}"#,
            false,
            GoalKind::Analysis
        ));
    }

    #[test]
    fn no_progress_forced_final_accepts_green_tests_without_a_fix_checklist() {
        assert!(!no_progress_forced_final_is_unusable(
            "",
            GoalKind::CodeChange,
            true,
            false
        ));
        assert!(no_progress_forced_final_is_unusable(
            "",
            GoalKind::CodeChange,
            false,
            false
        ));
        assert!(!no_progress_forced_final_is_unusable(
            "No major issues. Tests already pass.",
            GoalKind::CodeChange,
            true,
            false
        ));
        assert!(no_progress_forced_final_is_unusable(
            "I can't proceed.",
            GoalKind::CodeChange,
            true,
            false
        ));
    }

    #[test]
    fn please_run_is_permission_asking_on_code_change_not_analysis() {
        let text = "Please run the tests for me.";
        assert_eq!(
            classify_text_answer(GoalKind::CodeChange, text, false, false),
            StallCategory::StalledPermissionAsking
        );
        assert!(forced_final_answer_is_unusable(
            text,
            false,
            GoalKind::CodeChange
        ));
        assert_eq!(
            classify_text_answer(GoalKind::Analysis, text, false, false),
            StallCategory::NotStalledComplete
        );
        assert!(!forced_final_answer_is_unusable(
            text,
            false,
            GoalKind::Analysis
        ));
    }

    #[test]
    fn awaiting_background_is_not_a_stall() {
        assert_eq!(
            classify_text_answer(
                GoalKind::CodeChange,
                "Waiting for the compile to finish.",
                true,
                true
            ),
            StallCategory::NotStalledWaitingOnBackground
        );
        assert!(
            !classify_text_answer(
                GoalKind::CodeChange,
                "Waiting for the compile to finish.",
                true,
                true
            )
            .is_stalled()
        );
    }

    #[test]
    fn research_kind_treats_the_written_answer_as_the_deliverable() {
        let text = "RFC 1459 NICK is unique per server. Let me implement a local cache next.";
        assert_eq!(
            classify_text_answer(GoalKind::Research, text, false, false),
            StallCategory::NotStalledWaitingOnUser
        );
        assert!(!forced_final_answer_is_unusable(
            text,
            true,
            GoalKind::Research
        ));
    }
}
