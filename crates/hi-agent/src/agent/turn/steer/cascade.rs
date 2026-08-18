//! Table-driven **answer-repair** quality cascade (Steer phase).
//!
//! Priority is:
//! 1. [`crate::steering::REVIEW_QUALITY_PREFACE`] (SecurityBroad insufficient-evidence)
//! 2. [`crate::steering::REVIEW_QUALITY_CASCADE`]
//!
//! This module walks those tables instead of an open-coded if-ladder so
//! reorder/regressions fail at the cascade constant + selector tests, not only
//! in integration. [`AnswerRepairMode::SprawlForceAnswer`] is outside this
//! cascade (dedicated force-answer budget in the text-only Steer path).

use crate::config::ReviewRepairBudgets;
use crate::steering::{
    AnswerRepairMode, CONCRETE_REVIEW_NUDGE, EvidenceTracker, GAP_SEARCH_OVERCLAIM_NUDGE,
    READ_AFTER_SEARCH_NUDGE, REVIEW_QUALITY_CASCADE, REVIEW_QUALITY_PREFACE, ReviewIntent,
    SECURITY_BROAD_SEARCH_NUDGE, SECURITY_SCOPE_NUDGE, answer_says_insufficient_evidence,
    concrete_review_answer_problem, deepen_review_nudge, no_evidence_review_nudge,
    should_deepen_review, should_nudge_gap_search_overclaim, should_nudge_no_evidence_review,
    should_nudge_read_after_search_final, should_nudge_security_broad_search,
    should_nudge_security_scope, should_reject_review_repair_template,
    summarize_inspected_evidence_nudge,
};

use super::super::retry::ReviewRepairState;

/// What the quality cascade wants the Steer phase to do next.
#[derive(Debug)]
pub(super) enum QualityCascadeAction {
    /// Spend budget and continue the model loop with a repair nudge.
    Repair {
        mode: AnswerRepairMode,
        /// UI status / nudge line (short).
        status: String,
        /// Full nudge body (already includes required-next when applied).
        nudge_body: String,
        force_tools: bool,
        force_text: bool,
    },
    /// Budget exhausted — stall incomplete.
    Exhausted {
        mode: AnswerRepairMode,
        status: String,
    },
}

/// Walk preface then [`REVIEW_QUALITY_CASCADE`] and return the first applicable action.
///
/// Returns `None` when no quality repair applies (caller emits the answer).
pub(super) fn select_review_quality_repair(
    read_only_intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    assistant_text: &str,
    review_repair: &ReviewRepairState,
    budgets: &ReviewRepairBudgets,
) -> Option<QualityCascadeAction> {
    // Preface (documented in REVIEW_QUALITY_PREFACE): SecurityBroad can fire on
    // insufficient-evidence-after-read *before* the disclaimer branch.
    for &mode in REVIEW_QUALITY_PREFACE {
        if let Some(action) = evaluate_preface_mode(
            mode,
            read_only_intent,
            evidence,
            assistant_text,
            review_repair,
            budgets,
        ) {
            return Some(action);
        }
    }

    for &mode in REVIEW_QUALITY_CASCADE {
        if let Some(action) = evaluate_cascade_mode(
            mode,
            read_only_intent,
            evidence,
            assistant_text,
            review_repair,
            budgets,
        ) {
            return Some(action);
        }
    }
    None
}

/// Preface-only predicates. Kept separate from the cascade table walk so the
/// SecurityBroad insufficient-evidence priority is explicit and testable.
fn evaluate_preface_mode(
    mode: AnswerRepairMode,
    read_only_intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    assistant_text: &str,
    review_repair: &ReviewRepairState,
    budgets: &ReviewRepairBudgets,
) -> Option<QualityCascadeAction> {
    match mode {
        AnswerRepairMode::SecurityBroadSearch => {
            if !(matches!(read_only_intent, Some(ReviewIntent::Security))
                && evidence.saw_read
                && evidence.saw_search
                && !evidence.security_search_complete()
                && answer_says_insufficient_evidence(assistant_text))
            {
                return None;
            }
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "security review gave a generic evidence disclaimer before searching all required pattern families; nudging the model to broaden the search".into(),
                    nudge_body: SECURITY_BROAD_SEARCH_NUDGE.to_string(),
                    force_tools: true,
                    force_text: false,
                })
            } else {
                // Budget exhausted: fall through so the cascade disclaimer
                // path (or later modes) can still act. This is the intended
                // degradation: once the broad-search budget is spent, the
                // `InspectedDisclaimer` mode below may accept a bounded answer
                // from already-inspected files without requiring the full
                // security pattern-family sweep to complete.
                None
            }
        }
        // Only `SecurityBroadSearch` is listed in `REVIEW_QUALITY_PREFACE`.
        // Match explicitly (no catch-all) so adding a new preface mode fails
        // to compile here instead of silently doing nothing.
        AnswerRepairMode::NoEvidence
        | AnswerRepairMode::InspectedDisclaimer
        | AnswerRepairMode::InspectedDisclaimerChatAttempt
        | AnswerRepairMode::GenericTemplate
        | AnswerRepairMode::ListingOnly
        | AnswerRepairMode::ReadAfterSearch
        | AnswerRepairMode::SecurityScope
        | AnswerRepairMode::GapSearchOverclaim
        | AnswerRepairMode::ConcreteAnswer
        | AnswerRepairMode::SprawlForceAnswer => None,
    }
}

fn evaluate_cascade_mode(
    mode: AnswerRepairMode,
    read_only_intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    assistant_text: &str,
    review_repair: &ReviewRepairState,
    budgets: &ReviewRepairBudgets,
) -> Option<QualityCascadeAction> {
    match mode {
        AnswerRepairMode::NoEvidence => {
            if !should_nudge_no_evidence_review(read_only_intent, evidence, assistant_text) {
                return None;
            }
            let intent = read_only_intent?;
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "review answer had no inspected evidence; nudging the model to inspect before answering".into(),
                    nudge_body: no_evidence_review_nudge(intent).to_string(),
                    force_tools: true,
                    force_text: false,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status:
                        "review still had no inspected evidence after repair; stopping incomplete"
                            .into(),
                })
            }
        }
        AnswerRepairMode::InspectedDisclaimer
        | AnswerRepairMode::InspectedDisclaimerChatAttempt => {
            // `InspectedDisclaimerChatAttempt` is a cascade slot / wire key for
            // telemetry only — it never produces a repair action. It shares
            // this arm so the exhaustiveness match stays complete, but
            // short-circuits to `None` before evaluating any predicate.
            if mode == AnswerRepairMode::InspectedDisclaimerChatAttempt {
                return None;
            }
            let intent = read_only_intent?;
            if !(evidence.saw_read && answer_says_insufficient_evidence(assistant_text)) {
                return None;
            }
            // Nudge while budget remains. After that, accept the answer: the
            // model already inspected files and is choosing a bounded hedge.
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "review gave a generic evidence disclaimer after inspection; nudging the model to answer from inspected files".into(),
                    nudge_body: summarize_inspected_evidence_nudge(intent, evidence),
                    force_tools: false,
                    force_text: true,
                })
            } else {
                None
            }
        }
        AnswerRepairMode::GenericTemplate => {
            let needs_evidence_depth_repair = evidence.listing_only()
                || (evidence.saw_search && !evidence.saw_read)
                || (matches!(read_only_intent, Some(ReviewIntent::Security))
                    && evidence.saw_search
                    && !evidence.security_search_complete());
            if needs_evidence_depth_repair
                || !should_reject_review_repair_template(read_only_intent, assistant_text)
            {
                return None;
            }
            let intent = read_only_intent?;
            let has_inspected_evidence = evidence.saw_read || evidence.saw_search;
            if review_repair.has_budget(mode, budgets) {
                let nudge = if has_inspected_evidence {
                    summarize_inspected_evidence_nudge(intent, evidence)
                } else {
                    deepen_review_nudge(intent).to_string()
                };
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "review answer was a generic repair template; nudging the model to produce a concrete bounded review".into(),
                    nudge_body: nudge,
                    force_tools: !has_inspected_evidence,
                    force_text: has_inspected_evidence,
                })
            } else if has_inspected_evidence && !assistant_text.trim().is_empty() {
                // Evidence exists; weak template after budget is still a deliverable.
                None
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "review answer stayed generic after repair; stopping incomplete".into(),
                })
            }
        }
        AnswerRepairMode::ListingOnly => {
            if !should_deepen_review(read_only_intent, evidence, assistant_text) {
                return None;
            }
            let intent = read_only_intent?;
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "review evidence was only a listing; nudging the model to inspect files or search results".into(),
                    nudge_body: deepen_review_nudge(intent).to_string(),
                    force_tools: true,
                    force_text: false,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status:
                        "review still had only listing evidence after repair; stopping incomplete"
                            .into(),
                })
            }
        }
        AnswerRepairMode::ReadAfterSearch => {
            if !should_nudge_read_after_search_final(read_only_intent, evidence, assistant_text) {
                return None;
            }
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "review had targeted search but no file reads; nudging the model to read matching files".into(),
                    nudge_body: READ_AFTER_SEARCH_NUDGE.to_string(),
                    force_tools: true,
                    force_text: false,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "review still had targeted search but no file reads after repair; stopping incomplete".into(),
                })
            }
        }
        AnswerRepairMode::SecurityBroadSearch => {
            // The insufficient-evidence special case may already have handled this.
            if !should_nudge_security_broad_search(read_only_intent, evidence, assistant_text) {
                return None;
            }
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "security review missed required pattern families; nudging the model to broaden the search".into(),
                    nudge_body: SECURITY_BROAD_SEARCH_NUDGE.to_string(),
                    force_tools: true,
                    force_text: false,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "security review still missed required pattern families after repair; stopping incomplete".into(),
                })
            }
        }
        AnswerRepairMode::SecurityScope => {
            if !should_nudge_security_scope(read_only_intent, evidence, assistant_text) {
                return None;
            }
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "security answer overclaimed repo-wide safety; nudging the model to bound findings to evidence".into(),
                    nudge_body: SECURITY_SCOPE_NUDGE.to_string(),
                    force_tools: false,
                    force_text: false,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "security answer still overclaimed after repair; stopping incomplete"
                        .into(),
                })
            }
        }
        AnswerRepairMode::GapSearchOverclaim => {
            if !should_nudge_gap_search_overclaim(read_only_intent, evidence, assistant_text) {
                return None;
            }
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "gap answer contradicted search matches; nudging the model to bound claims to inspected evidence".into(),
                    nudge_body: GAP_SEARCH_OVERCLAIM_NUDGE.to_string(),
                    force_tools: false,
                    force_text: false,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status:
                        "gap answer still overclaimed after search matches; stopping incomplete"
                            .into(),
                })
            }
        }
        AnswerRepairMode::ConcreteAnswer => {
            let problem =
                concrete_review_answer_problem(read_only_intent, evidence, assistant_text)?;
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: problem.status().to_string(),
                    nudge_body: CONCRETE_REVIEW_NUDGE.to_string(),
                    // Keep the read-only catalog. Preflight + a thinking
                    // model (DeepSeek V4 especially) often answers from the
                    // seed evidence without calling tools; a chat-only
                    // follow-up then cannot cite or re-read those files.
                    force_tools: false,
                    force_text: false,
                })
            } else if evidence.saw_read && !assistant_text.trim().is_empty() {
                // Format-weak answer after inspection: accept rather than stall.
                None
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: problem.exhausted_status().to_string(),
                })
            }
        }
        // Handled outside the cascade (text-only Steer force-answer path).
        AnswerRepairMode::SprawlForceAnswer => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steering::{REVIEW_QUALITY_CASCADE, REVIEW_QUALITY_PREFACE};

    #[test]
    fn selector_visits_cascade_in_spec_order() {
        // The evaluator is consulted in REVIEW_QUALITY_CASCADE order; ChatAttempt
        // is skipped as a primary step (accounting-only).
        let primary: Vec<_> = REVIEW_QUALITY_CASCADE
            .iter()
            .copied()
            .filter(|m| *m != AnswerRepairMode::InspectedDisclaimerChatAttempt)
            .collect();
        assert_eq!(primary.first(), Some(&AnswerRepairMode::NoEvidence));
        assert_eq!(primary.last(), Some(&AnswerRepairMode::ConcreteAnswer));
        let idx = |m: AnswerRepairMode| primary.iter().position(|x| *x == m).unwrap();
        assert!(idx(AnswerRepairMode::NoEvidence) < idx(AnswerRepairMode::InspectedDisclaimer));
        assert!(
            idx(AnswerRepairMode::InspectedDisclaimer) < idx(AnswerRepairMode::GenericTemplate)
        );
        assert!(idx(AnswerRepairMode::GenericTemplate) < idx(AnswerRepairMode::ListingOnly));
        assert!(idx(AnswerRepairMode::ListingOnly) < idx(AnswerRepairMode::ReadAfterSearch));
        assert!(
            idx(AnswerRepairMode::ReadAfterSearch) < idx(AnswerRepairMode::SecurityBroadSearch)
        );
        assert!(idx(AnswerRepairMode::SecurityBroadSearch) < idx(AnswerRepairMode::SecurityScope));
        assert!(idx(AnswerRepairMode::SecurityScope) < idx(AnswerRepairMode::GapSearchOverclaim));
        assert!(idx(AnswerRepairMode::GapSearchOverclaim) < idx(AnswerRepairMode::ConcreteAnswer));
        assert!(
            !REVIEW_QUALITY_CASCADE.contains(&AnswerRepairMode::SprawlForceAnswer),
            "sprawl force-answer is outside the quality cascade"
        );
    }

    #[test]
    fn preface_lists_security_broad_before_cascade() {
        assert_eq!(
            REVIEW_QUALITY_PREFACE,
            &[AnswerRepairMode::SecurityBroadSearch]
        );
    }

    #[test]
    fn security_broad_preface_beats_disclaimer_on_insufficient_evidence() {
        let evidence = EvidenceTracker {
            saw_read: true,
            saw_search: true,
            ..Default::default()
        };
        // Incomplete security families → SecurityBroad eligible.
        assert!(!evidence.security_search_complete());
        let budgets = ReviewRepairBudgets::default();
        let state = ReviewRepairState::default();
        let action = select_review_quality_repair(
            Some(ReviewIntent::Security),
            &evidence,
            "Insufficient evidence to assess the security posture.",
            &state,
            &budgets,
        )
        .expect("preface should fire");
        match action {
            QualityCascadeAction::Repair { mode, .. } => {
                assert_eq!(mode, AnswerRepairMode::SecurityBroadSearch);
            }
            other => panic!("expected SecurityBroad repair, got {other:?}"),
        }
    }

    #[test]
    fn security_broad_preface_skips_when_budget_exhausted_then_disclaimer_may_fire() {
        let evidence = EvidenceTracker {
            saw_read: true,
            saw_search: true,
            inspected_paths: vec!["src/auth.rs".into()],
            ..Default::default()
        };
        let budgets = ReviewRepairBudgets {
            security_broad_search: 0,
            ..ReviewRepairBudgets::default()
        };
        let state = ReviewRepairState::default();
        let action = select_review_quality_repair(
            Some(ReviewIntent::Security),
            &evidence,
            "Insufficient evidence to assess the security posture.",
            &state,
            &budgets,
        )
        .expect("disclaimer cascade should still fire");
        match action {
            QualityCascadeAction::Repair { mode, .. } => {
                assert_eq!(mode, AnswerRepairMode::InspectedDisclaimer);
            }
            other => panic!("expected disclaimer repair, got {other:?}"),
        }
    }

    /// Predicate matrix: intent × evidence × answer shape → first cascade action.
    #[test]
    fn answer_repair_predicate_matrix_selects_expected_mode() {
        let budgets = ReviewRepairBudgets::default();
        let state = ReviewRepairState::default();

        // No discovery at all → NoEvidence.
        {
            let evidence = EvidenceTracker::default();
            let action = select_review_quality_repair(
                Some(ReviewIntent::Review),
                &evidence,
                "Looks fine overall.",
                &state,
                &budgets,
            )
            .expect("no-evidence");
            assert!(matches!(
                action,
                QualityCascadeAction::Repair {
                    mode: AnswerRepairMode::NoEvidence,
                    ..
                }
            ));
        }

        // Listing only → ListingOnly (after NoEvidence/disclaimer/template skip).
        {
            let evidence = EvidenceTracker {
                saw_listing: true,
                ..Default::default()
            };
            assert!(evidence.listing_only());
            let action = select_review_quality_repair(
                Some(ReviewIntent::Review),
                &evidence,
                "Here is the tree structure of the project.",
                &state,
                &budgets,
            )
            .expect("listing");
            assert!(matches!(
                action,
                QualityCascadeAction::Repair {
                    mode: AnswerRepairMode::ListingOnly,
                    ..
                }
            ));
        }

        // Search without read → ReadAfterSearch.
        {
            let evidence = EvidenceTracker {
                saw_search: true,
                ..Default::default()
            };
            let action = select_review_quality_repair(
                Some(ReviewIntent::Gaps),
                &evidence,
                "No gaps found in the codebase.",
                &state,
                &budgets,
            )
            .expect("read-after-search");
            assert!(matches!(
                action,
                QualityCascadeAction::Repair {
                    mode: AnswerRepairMode::ReadAfterSearch,
                    ..
                }
            ));
        }

        // Security with incomplete families (non-disclaimer answer) → SecurityBroad via cascade.
        {
            let evidence = EvidenceTracker {
                saw_search: true,
                saw_read: true,
                inspected_paths: vec!["src/lib.rs".into()],
                ..Default::default()
            };
            let action = select_review_quality_repair(
                Some(ReviewIntent::Security),
                &evidence,
                "Findings:\n- src/lib.rs uses unwrap in one place.\nLimits: partial scan.",
                &state,
                &budgets,
            )
            .expect("security broad cascade");
            assert!(matches!(
                action,
                QualityCascadeAction::Repair {
                    mode: AnswerRepairMode::SecurityBroadSearch,
                    ..
                }
            ));
        }

        // Generic repair template → GenericTemplate.
        {
            let evidence = EvidenceTracker {
                saw_read: true,
                inspected_paths: vec!["src/a.rs".into()],
                ..Default::default()
            };
            let action = select_review_quality_repair(
                Some(ReviewIntent::Review),
                &evidence,
                "The inspected context points to these concrete review targets: src/a.rs. \
                 Review observations should stay tied to those files or modules.",
                &state,
                &budgets,
            )
            .expect("generic template");
            assert!(matches!(
                action,
                QualityCascadeAction::Repair {
                    mode: AnswerRepairMode::GenericTemplate,
                    ..
                }
            ));
        }

        // Clean bounded review with citations → no repair.
        {
            let evidence = EvidenceTracker {
                saw_read: true,
                inspected_paths: vec!["src/parser.rs".into()],
                ..Default::default()
            };
            let action = select_review_quality_repair(
                Some(ReviewIntent::Review),
                &evidence,
                "Based on inspected src/parser.rs:\n\
                 Findings:\n- src/parser.rs: missing error path on EOF.\n\
                 Limits: only inspected src/parser.rs.",
                &state,
                &budgets,
            );
            assert!(
                action.is_none(),
                "clean answer should not repair: {action:?}"
            );
        }
    }
}
