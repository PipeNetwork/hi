//! Centralized **review-answer** repair mode metadata (Steer phase).
//!
//! These modes nudge the model when a read-only review answer is weak (no
//! evidence, generic template, …). They never run shell stages.
//!
//! Contrast with [`crate::verify::WorkspaceRepairVerifier`] (WorkspaceRepair
//! phase), which runs compile/lint/test and feeds failures back into the loop.

/// Local repair modes for read-only review turns (answer quality, not tests).
///
/// The string keys are report/telemetry wire values. Keep them stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ReviewRepairMode {
    NoEvidence,
    ListingOnly,
    GenericTemplate,
    InspectedDisclaimer,
    InspectedDisclaimerChatAttempt,
    ConcreteAnswer,
    ReadAfterSearch,
    SecurityBroadSearch,
    SecurityScope,
    GapSearchOverclaim,
}

impl ReviewRepairMode {
    pub(crate) const ALL: &'static [Self] = &[
        Self::NoEvidence,
        Self::ListingOnly,
        Self::GenericTemplate,
        Self::InspectedDisclaimer,
        Self::InspectedDisclaimerChatAttempt,
        Self::ConcreteAnswer,
        Self::ReadAfterSearch,
        Self::SecurityBroadSearch,
        Self::SecurityScope,
        Self::GapSearchOverclaim,
    ];

    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::NoEvidence => "review_no_evidence",
            Self::ListingOnly => "review_listing_only",
            Self::GenericTemplate => "review_generic_template",
            Self::InspectedDisclaimer => "review_inspected_disclaimer",
            Self::InspectedDisclaimerChatAttempt => "review_inspected_disclaimer_chat_attempt",
            Self::ConcreteAnswer => "review_concrete_answer",
            Self::ReadAfterSearch => "review_read_after_search",
            Self::SecurityBroadSearch => "review_security_broad_search",
            Self::SecurityScope => "review_security_scope",
            Self::GapSearchOverclaim => "review_gap_search_overclaim",
        }
    }

    pub(crate) fn exhaustion_key(self) -> &'static str {
        match self {
            Self::NoEvidence => "review_no_evidence_exhausted",
            Self::ListingOnly => "review_listing_only_exhausted",
            Self::GenericTemplate
            | Self::InspectedDisclaimer
            | Self::InspectedDisclaimerChatAttempt => "review_generic_disclaimer_exhausted",
            Self::ConcreteAnswer => "review_concrete_answer_exhausted",
            Self::ReadAfterSearch => "review_read_after_search_exhausted",
            Self::SecurityBroadSearch => "review_security_broad_search_exhausted",
            Self::SecurityScope => "review_security_scope_exhausted",
            Self::GapSearchOverclaim => "review_gap_search_overclaim_exhausted",
        }
    }

    pub(crate) fn required_next(self) -> &'static str {
        match self {
            Self::NoEvidence => "inspect_files_before_answering",
            Self::ListingOnly => "inspect_one_concrete_file_before_answering",
            Self::GenericTemplate => "produce_concrete_bounded_review",
            Self::InspectedDisclaimer | Self::InspectedDisclaimerChatAttempt => {
                "chat_only_bounded_answer_from_inspected_files"
            }
            Self::ConcreteAnswer => "cite_findings_plus_limits",
            Self::ReadAfterSearch => "read_one_matching_file_before_answering",
            Self::SecurityBroadSearch => "search_required_security_patterns_before_answering",
            Self::SecurityScope => "bound_security_claims_to_inspected_evidence",
            Self::GapSearchOverclaim => "cite_search_matches_plus_limits",
        }
    }

    pub(crate) fn required_next_instruction(self) -> &'static str {
        match self {
            Self::NoEvidence => "inspect concrete files with read-only tools before answering.",
            Self::ListingOnly => {
                "inspect at least one concrete relevant file or targeted search result before answering."
            }
            Self::GenericTemplate => {
                "produce a concrete bounded review tied to inspected evidence."
            }
            Self::InspectedDisclaimer | Self::InspectedDisclaimerChatAttempt => {
                "answer in chat only from the inspected files, with findings and limits."
            }
            Self::ConcreteAnswer => {
                "cite inspected files in concrete findings and include bounded limits."
            }
            Self::ReadAfterSearch => {
                "read one matching file from the search results before answering."
            }
            Self::SecurityBroadSearch => {
                "search the required security pattern families before answering."
            }
            Self::SecurityScope => {
                "bound security claims to inspected files and searched patterns."
            }
            Self::GapSearchOverclaim => {
                "cite search matches and inspected files, then state limits for broader claims."
            }
        }
    }

    pub(crate) fn compact_label(self) -> &'static str {
        match self {
            Self::NoEvidence => "no_evidence",
            Self::ListingOnly => "listing",
            Self::GenericTemplate => "generic",
            Self::InspectedDisclaimer => "disclaimer",
            Self::InspectedDisclaimerChatAttempt => "disclaimer_chat",
            Self::ConcreteAnswer => "concrete",
            Self::ReadAfterSearch => "read_after_search",
            Self::SecurityBroadSearch => "security_broad",
            Self::SecurityScope => "security_scope",
            Self::GapSearchOverclaim => "gap_overclaim",
        }
    }

    /// Built-in default budget for this mode (also the
    /// [`crate::config::ReviewRepairBudgets`] default). Prefer
    /// [`Self::limit_with`] when an [`crate::AgentConfig`] is available.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn default_limit(self) -> u32 {
        crate::config::ReviewRepairBudgets::default().limit_for_key(self.key())
    }

    /// Budget for this mode from operator-tunable loop limits.
    pub(crate) fn limit_with(self, budgets: &crate::config::ReviewRepairBudgets) -> u32 {
        budgets.limit_for_key(self.key())
    }

    pub(crate) fn from_key(key: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|mode| mode.key() == key)
    }
}

pub(crate) fn repair_nudge_with_required_next(
    mode: ReviewRepairMode,
    body: impl AsRef<str>,
) -> String {
    format!(
        "{}\n\nRequired next action `{}`: {}",
        body.as_ref(),
        mode.required_next(),
        mode.required_next_instruction()
    )
}

pub(crate) fn compact_review_repair_label(label: &str) -> String {
    if let Some(mode) = ReviewRepairMode::from_key(label) {
        return mode.compact_label().to_string();
    }

    let label = label.strip_prefix("review_").unwrap_or(label);
    let label = label.strip_suffix("_exhausted").unwrap_or(label);
    match label {
        "no_evidence" => "no_evidence",
        "listing_only" => "listing",
        "generic_disclaimer" => "generic",
        "generic_template" => "generic",
        "inspected_disclaimer" => "disclaimer",
        "inspected_disclaimer_chat_attempt" => "disclaimer_chat",
        "concrete_answer" => "concrete",
        "read_after_search" => "read_after_search",
        "security_broad_search" => "security_broad",
        "security_scope" => "security_scope",
        "gap_search_overclaim" => "gap_overclaim",
        other => other,
    }
    .to_string()
}

/// Text-only Steer quality-repair cascade order (after unfinished/plan and
/// implementation-completeness gates). Keep this list aligned with
/// `steer/review.rs` — tests freeze the order so a casual reorder fails loudly.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const REVIEW_QUALITY_CASCADE: &[ReviewRepairMode] = &[
    ReviewRepairMode::NoEvidence,
    ReviewRepairMode::InspectedDisclaimer,
    ReviewRepairMode::InspectedDisclaimerChatAttempt,
    ReviewRepairMode::GenericTemplate,
    ReviewRepairMode::ListingOnly,
    ReviewRepairMode::ReadAfterSearch,
    ReviewRepairMode::SecurityBroadSearch,
    ReviewRepairMode::SecurityScope,
    ReviewRepairMode::GapSearchOverclaim,
    ReviewRepairMode::ConcreteAnswer,
];

#[cfg(test)]
mod cascade_tests {
    use super::*;

    #[test]
    fn quality_cascade_is_unique_and_covers_known_modes() {
        let mut seen = std::collections::BTreeSet::new();
        for mode in REVIEW_QUALITY_CASCADE {
            assert!(seen.insert(mode.key()), "duplicate cascade entry {}", mode.key());
            assert!(
                ReviewRepairMode::ALL.contains(mode),
                "{} missing from ReviewRepairMode::ALL",
                mode.key()
            );
        }
        // Disclaimer family shares exhaustion key but remains distinct cascade steps.
        assert!(REVIEW_QUALITY_CASCADE.contains(&ReviewRepairMode::InspectedDisclaimer));
        assert!(REVIEW_QUALITY_CASCADE.contains(&ReviewRepairMode::InspectedDisclaimerChatAttempt));
    }

    #[test]
    fn cascade_runs_no_evidence_before_concrete_and_security_before_gap() {
        let idx = |m: ReviewRepairMode| {
            REVIEW_QUALITY_CASCADE
                .iter()
                .position(|x| *x == m)
                .expect("mode in cascade")
        };
        assert!(idx(ReviewRepairMode::NoEvidence) < idx(ReviewRepairMode::ConcreteAnswer));
        assert!(idx(ReviewRepairMode::ReadAfterSearch) < idx(ReviewRepairMode::ConcreteAnswer));
        assert!(idx(ReviewRepairMode::SecurityBroadSearch) < idx(ReviewRepairMode::SecurityScope));
        assert!(idx(ReviewRepairMode::SecurityScope) < idx(ReviewRepairMode::GapSearchOverclaim));
        assert!(idx(ReviewRepairMode::ListingOnly) < idx(ReviewRepairMode::ConcreteAnswer));
    }
}
