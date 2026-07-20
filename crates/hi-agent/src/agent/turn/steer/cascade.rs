//! Table-driven review quality-repair cascade.
//!
//! Order is frozen by [`crate::steering::REVIEW_QUALITY_CASCADE`]. This module
//! walks that table instead of an open-coded if-ladder so reorder/regressions
//! fail at the cascade constant + selector tests, not only in integration.

use crate::config::ReviewRepairBudgets;
use crate::steering::{
    CONCRETE_REVIEW_NUDGE, EvidenceTracker, GAP_SEARCH_OVERCLAIM_NUDGE, READ_AFTER_SEARCH_NUDGE,
    REVIEW_QUALITY_CASCADE, ReviewIntent, ReviewRepairMode, SECURITY_BROAD_SEARCH_NUDGE,
    SECURITY_SCOPE_NUDGE, answer_says_insufficient_evidence, concrete_review_answer_problem,
    deepen_review_nudge, no_evidence_review_nudge, should_deepen_review,
    should_nudge_gap_search_overclaim, should_nudge_no_evidence_review,
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
        mode: ReviewRepairMode,
        /// UI status / nudge line (short).
        status: String,
        /// Full nudge body (already includes required-next when applied).
        nudge_body: String,
        force_tools: bool,
        force_text: bool,
        /// When set, also `note` this mode (disclaimer chat-attempt accounting).
        note_mode: Option<ReviewRepairMode>,
        /// When true, call `spend` on `mode`; when false, only bump quality counter.
        spend: bool,
    },
    /// Budget exhausted — stall incomplete.
    Exhausted {
        mode: ReviewRepairMode,
        status: String,
    },
}

/// Walk [`REVIEW_QUALITY_CASCADE`] and return the first applicable action.
///
/// Returns `None` when no quality repair applies (caller emits the answer).
pub(super) fn select_review_quality_repair(
    read_only_intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    assistant_text: &str,
    review_repair: &ReviewRepairState,
    budgets: &ReviewRepairBudgets,
) -> Option<QualityCascadeAction> {
    // Special pre-step: insufficient-evidence-after-read can fire SecurityBroad
    // *before* the disclaimer branch (historical order inside that arm).
    if let Some(intent) = read_only_intent
        && evidence.saw_read
        && answer_says_insufficient_evidence(assistant_text)
    {
        if matches!(intent, ReviewIntent::Security)
            && evidence.saw_search
            && !evidence.security_search_complete()
            && review_repair.has_budget(ReviewRepairMode::SecurityBroadSearch, budgets)
        {
            return Some(QualityCascadeAction::Repair {
                mode: ReviewRepairMode::SecurityBroadSearch,
                status: "security review gave a generic evidence disclaimer before searching all required pattern families; nudging the model to broaden the search".into(),
                nudge_body: SECURITY_BROAD_SEARCH_NUDGE.to_string(),
                force_tools: true,
                force_text: false,
                note_mode: None,
                spend: true,
            });
        }
        // Fall through into cascade; InspectedDisclaimer predicate will match.
    }

    for &mode in REVIEW_QUALITY_CASCADE {
        if let Some(action) =
            evaluate_cascade_mode(mode, read_only_intent, evidence, assistant_text, review_repair, budgets)
        {
            return Some(action);
        }
    }
    None
}

fn evaluate_cascade_mode(
    mode: ReviewRepairMode,
    read_only_intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    assistant_text: &str,
    review_repair: &ReviewRepairState,
    budgets: &ReviewRepairBudgets,
) -> Option<QualityCascadeAction> {
    match mode {
        ReviewRepairMode::NoEvidence => {
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
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "review still had no inspected evidence after repair; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::InspectedDisclaimer | ReviewRepairMode::InspectedDisclaimerChatAttempt => {
            // ChatAttempt is accounting-only; selection is driven by InspectedDisclaimer.
            if mode == ReviewRepairMode::InspectedDisclaimerChatAttempt {
                return None;
            }
            let intent = read_only_intent?;
            if !(evidence.saw_read && answer_says_insufficient_evidence(assistant_text)) {
                return None;
            }
            let chat_mode = ReviewRepairMode::InspectedDisclaimerChatAttempt;
            let has_disclaimer_budget = review_repair.has_budget(mode, budgets);
            let has_chat_attempt_budget = review_repair.has_budget(chat_mode, budgets);
            if has_disclaimer_budget || has_chat_attempt_budget {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: "review gave a generic evidence disclaimer after inspection; nudging the model to answer from inspected files".into(),
                    nudge_body: summarize_inspected_evidence_nudge(intent, evidence),
                    force_tools: false,
                    force_text: true,
                    note_mode: Some(chat_mode),
                    spend: has_disclaimer_budget,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "review kept returning a generic evidence disclaimer after inspection; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::GenericTemplate => {
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
            if review_repair.has_budget(mode, budgets) {
                let has_inspected_evidence = evidence.saw_read || evidence.saw_search;
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
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "review answer stayed generic after repair; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::ListingOnly => {
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
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "review still had only listing evidence after repair; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::ReadAfterSearch => {
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
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "review still had targeted search but no file reads after repair; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::SecurityBroadSearch => {
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
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "security review still missed required pattern families after repair; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::SecurityScope => {
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
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "security answer still overclaimed after repair; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::GapSearchOverclaim => {
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
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: "gap answer still overclaimed after search matches; stopping incomplete".into(),
                })
            }
        }
        ReviewRepairMode::ConcreteAnswer => {
            let problem =
                concrete_review_answer_problem(read_only_intent, evidence, assistant_text)?;
            if review_repair.has_budget(mode, budgets) {
                Some(QualityCascadeAction::Repair {
                    mode,
                    status: problem.status().to_string(),
                    nudge_body: CONCRETE_REVIEW_NUDGE.to_string(),
                    force_tools: false,
                    force_text: true,
                    note_mode: None,
                    spend: true,
                })
            } else {
                Some(QualityCascadeAction::Exhausted {
                    mode,
                    status: problem.exhausted_status().to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steering::REVIEW_QUALITY_CASCADE;

    #[test]
    fn selector_visits_cascade_in_spec_order() {
        // The evaluator is consulted in REVIEW_QUALITY_CASCADE order; ChatAttempt
        // is skipped as a primary step (accounting-only).
        let primary: Vec<_> = REVIEW_QUALITY_CASCADE
            .iter()
            .copied()
            .filter(|m| *m != ReviewRepairMode::InspectedDisclaimerChatAttempt)
            .collect();
        assert_eq!(primary.first(), Some(&ReviewRepairMode::NoEvidence));
        assert_eq!(primary.last(), Some(&ReviewRepairMode::ConcreteAnswer));
        let idx = |m: ReviewRepairMode| primary.iter().position(|x| *x == m).unwrap();
        assert!(idx(ReviewRepairMode::NoEvidence) < idx(ReviewRepairMode::InspectedDisclaimer));
        assert!(idx(ReviewRepairMode::InspectedDisclaimer) < idx(ReviewRepairMode::GenericTemplate));
        assert!(idx(ReviewRepairMode::GenericTemplate) < idx(ReviewRepairMode::ListingOnly));
        assert!(idx(ReviewRepairMode::ListingOnly) < idx(ReviewRepairMode::ReadAfterSearch));
        assert!(idx(ReviewRepairMode::ReadAfterSearch) < idx(ReviewRepairMode::SecurityBroadSearch));
        assert!(idx(ReviewRepairMode::SecurityBroadSearch) < idx(ReviewRepairMode::SecurityScope));
        assert!(idx(ReviewRepairMode::SecurityScope) < idx(ReviewRepairMode::GapSearchOverclaim));
        assert!(idx(ReviewRepairMode::GapSearchOverclaim) < idx(ReviewRepairMode::ConcreteAnswer));
    }
}
