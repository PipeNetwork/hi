//! Nudge decision functions and answer-shape checks. Uses nudge strings from
//! [`constants`](super::constants), [`contains_any`] from [`intent`](super::intent),
//! and tracker types from [`types`](super::types).

use super::constants::{
    GAP_DEEPEN_NUDGE, IMPLEMENTATION_MISSING_VALIDATION_NUDGE, NO_EVIDENCE_GAP_NUDGE,
    NO_EVIDENCE_REVIEW_NUDGE, NO_EVIDENCE_SECURITY_NUDGE, NO_EVIDENCE_STATUS_NUDGE,
    REVIEW_DEEPEN_NUDGE, SECURITY_DEEPEN_NUDGE, STATUS_DEEPEN_NUDGE,
    TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE,
};
use super::intent::contains_any;
use super::types::{EvidenceTracker, ImplementationTracker, ReviewIntent};
pub(crate) fn implementation_missing_validation_nudge(tracker: &ImplementationTracker) -> String {
    let preferred = tracker
        .preferred_validation
        .as_deref()
        .map(|command| format!(" Prefer `{command}`."))
        .unwrap_or_default();
    format!("{IMPLEMENTATION_MISSING_VALIDATION_NUDGE}{preferred}")
}

pub(crate) fn implementation_text_tool_nudge(reason: &str) -> String {
    format!("{reason}\n\n{TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE}")
}

pub(crate) fn answer_says_insufficient_evidence(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "insufficient evidence",
            "not enough evidence",
            "not enough information",
            "only a directory listing",
            "only saw a listing",
            "need to inspect",
            "need file reads",
            "need targeted search",
        ],
    )
}

/// Exact low-information completion phrases observed from weak/local models.
///
/// This intentionally does not reject short answers in general (`ok`, `done`,
/// `yes`) because those may be the user's requested payload. It targets only
/// canned claims that assert completion while conveying no result or evidence.
pub fn answer_is_generic_completion_placeholder(content: &str) -> bool {
    let normalized = content
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    matches!(
        normalized.as_str(),
        "completed the requested action"
            | "the requested action is complete"
            | "the requested action has been completed"
            | "the requested task is complete"
            | "the requested task has been completed"
    )
}

/// Whether production generic-completion answer-integrity guards are active.
///
/// The false branch is compiled only by the executable smoke negative-control
/// script. Keeping the switch compile-time-only prevents scenarios, providers,
/// or end users from disabling the guards at runtime.
pub(crate) const fn generic_completion_guards_enabled() -> bool {
    !cfg!(feature = "smoke-negative-control-disable-generic-completion-guards")
}

/// Whether the model, challenged by the no-change nudge, explicitly declines
/// to mutate: the reply commits to "no file changes are needed" rather than
/// narrating pending work. The nudge prescribes this phrasing, so detection
/// stays a tight phrase match instead of a broad heuristic.
pub(crate) fn answer_declines_mutation(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "no file changes are needed",
            "no file changes are required",
            "no file changes needed",
            "no code changes are needed",
            "no code changes needed",
            "no changes are needed",
            "no changes are required",
            "no edits are needed",
            "no edits are required",
            "requires no file changes",
            "requires no code changes",
            // Broader decline / out-of-scope language after a no-change challenge.
            "out of scope",
            "outside the scope",
            "no changes necessary",
            "no change is needed",
            "already implemented",
            "already done",
            "nothing to change",
            "nothing needs to change",
            "i won't modify",
            "i will not modify",
            "i won't edit",
            "i will not edit",
            "leaving the code unchanged",
            "leave the code unchanged",
            "without modifying files",
            "without changing files",
            "no modifications are needed",
            "no modifications needed",
        ],
    )
}

pub(crate) fn should_deepen_review(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    _answer: &str,
) -> bool {
    intent.is_some() && evidence.listing_only()
}

pub(crate) fn should_nudge_no_evidence_review(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    _answer: &str,
) -> bool {
    intent.is_some() && !evidence.has_discovery()
}

pub(crate) fn answer_looks_like_review_repair_template(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    contains_any(
        &lower,
        &[
            "the inspected context points to these concrete review targets",
            "review observations should stay tied to those files or modules",
            "convert any broad status claims into file-specific findings",
            "the inspected context identifies these concrete targets as the likely ownership surface",
            "gap claims should be tied to those inspected files or modules",
            "convert broad recommendations into scoped work items tied to the inspected files",
        ],
    )
}

pub(crate) fn should_reject_review_repair_template(
    intent: Option<ReviewIntent>,
    answer: &str,
) -> bool {
    intent.is_some()
        && !answer_says_insufficient_evidence(answer)
        && answer_looks_like_review_repair_template(answer)
}

#[cfg(test)]
pub(crate) fn should_nudge_concrete_review_answer(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    concrete_review_answer_problem(intent, evidence, answer).is_some()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConcreteReviewAnswerProblem {
    GenericCompletionPlaceholder,
    MissingInspectedCitation,
    GenericInventorySummary,
    MissingReviewShape,
}

impl ConcreteReviewAnswerProblem {
    pub(crate) fn status(self) -> &'static str {
        match self {
            Self::GenericCompletionPlaceholder => {
                "review answer was only a completion placeholder; nudging the model to provide concrete findings"
            }
            Self::MissingInspectedCitation => {
                "review answer lacked concrete inspected files; nudging the model to tie findings to evidence"
            }
            Self::GenericInventorySummary => {
                "review answer was a generic inventory summary; nudging the model to tie findings to inspected evidence"
            }
            Self::MissingReviewShape => {
                "review answer cited inspected evidence but lacked bounded review findings; nudging the model to answer with findings and limits"
            }
        }
    }

    pub(crate) fn exhausted_status(self) -> &'static str {
        match self {
            Self::GenericCompletionPlaceholder => {
                "review answer remained a completion placeholder after repair; returning the available answer"
            }
            Self::MissingInspectedCitation => {
                "review answer still lacked concrete inspected files after repair; returning the available answer"
            }
            Self::GenericInventorySummary => {
                "review answer stayed generic after repair; returning the available answer"
            }
            Self::MissingReviewShape => {
                "review answer still lacked bounded review findings after repair; returning the available answer"
            }
        }
    }
}

pub(crate) fn concrete_review_answer_problem(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> Option<ConcreteReviewAnswerProblem> {
    let intent = intent?;
    if answer_is_generic_completion_placeholder(answer) {
        return Some(ConcreteReviewAnswerProblem::GenericCompletionPlaceholder);
    }
    if evidence.inspected_paths.is_empty() || answer_says_insufficient_evidence(answer) {
        return None;
    }
    if !answer_cites_inspected_path(evidence, answer) {
        return Some(ConcreteReviewAnswerProblem::MissingInspectedCitation);
    }
    if answer_looks_like_generic_inventory_summary(answer) {
        return Some(ConcreteReviewAnswerProblem::GenericInventorySummary);
    }
    if answer_is_concise_bounded_review(intent, answer) {
        return None;
    }
    if answer_lacks_review_shape(intent, answer) {
        return Some(ConcreteReviewAnswerProblem::MissingReviewShape);
    }
    None
}

pub(crate) fn answer_cites_inspected_path(evidence: &EvidenceTracker, answer: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    evidence.inspected_paths.iter().any(|path| {
        inspected_path_aliases(path)
            .iter()
            .any(|alias| lower.contains(alias))
    })
}

fn inspected_path_aliases(path: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    let lower_path = path.to_ascii_lowercase();
    if !lower_path.is_empty() {
        aliases.push(lower_path);
    }

    let file_name = std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if file_name.len() >= 5 && !aliases.iter().any(|alias| alias == &file_name) {
        aliases.push(file_name.clone());
    }

    let stem = std::path::Path::new(&file_name)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if is_distinctive_file_stem(&stem) && !aliases.iter().any(|alias| alias == &stem) {
        aliases.push(stem);
    }
    aliases
}

fn is_distinctive_file_stem(stem: &str) -> bool {
    stem.len() >= 5
        && (stem.contains('-') || stem.contains('_'))
        && !matches!(stem, "index" | "route" | "page" | "main" | "mod" | "lib")
}

pub(crate) fn answer_lacks_review_shape(intent: ReviewIntent, answer: &str) -> bool {
    if answer_has_priority_findings(answer) {
        return false;
    }
    let lower = answer.to_ascii_lowercase();
    let has_evidence_language = contains_any(
        &lower,
        &[
            "inspected",
            "reviewed",
            "evidence",
            "findings:",
            "based on",
            "limited to",
            "from the inspected",
            "in the inspected",
        ],
    );
    let has_review_language = match intent {
        ReviewIntent::Security => contains_any(
            &lower,
            &[
                "finding",
                "security",
                "unsafe",
                "unwrap",
                "expect",
                "panic",
                "secret",
                "token",
                "auth",
                "risk",
                "follow-up",
                "follow up",
            ],
        ),
        ReviewIntent::Status => contains_any(
            &lower,
            &[
                "status",
                "state",
                "build next",
                "risk",
                "validation",
                "follow-up",
                "follow up",
            ],
        ),
        ReviewIntent::Roadmap | ReviewIntent::Gaps => contains_any(
            &lower,
            &[
                "missing",
                "gap",
                "roadmap",
                "build next",
                "risk",
                "coverage",
                "follow-up",
                "follow up",
            ],
        ),
        ReviewIntent::Review => contains_any(
            &lower,
            &[
                "finding",
                "reviewed",
                "status",
                "risk",
                "validation",
                "tests pass",
                "test pass",
                "follow-up",
                "follow up",
            ],
        ),
    };
    !(has_evidence_language && has_review_language)
}

fn answer_has_priority_findings(answer: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    lower.contains("[p0]")
        || lower.contains("[p1]")
        || lower.contains("[p2]")
        || lower.contains("[p3]")
        || lower.contains("no findings")
}

fn answer_is_concise_bounded_review(intent: ReviewIntent, answer: &str) -> bool {
    if answer_has_priority_findings(answer) {
        return true;
    }
    let lower = answer.to_ascii_lowercase();
    let has_evidence_language = contains_any(
        &lower,
        &[
            "inspected",
            "reviewed",
            "read",
            "evidence",
            "based on",
            "from the inspected",
            "in the inspected",
        ],
    );
    let has_bounded_limit = contains_any(
        &lower,
        &[
            "limit:",
            "limits:",
            "limited to",
            "not a complete",
            "only inspected",
            "from this file alone",
            "outside the inspected",
            "cannot rule out",
        ],
    );
    let has_review_signal = match intent {
        ReviewIntent::Security => contains_any(
            &lower,
            &[
                "security",
                "unsafe",
                "unwrap",
                "expect",
                "panic",
                "secret",
                "token",
                "auth",
                "risk",
                "finding",
                "issue",
                "confirmed",
            ],
        ),
        ReviewIntent::Status => contains_any(
            &lower,
            &[
                "status",
                "state",
                "blocker",
                "validation",
                "risk",
                "current",
                "review",
                "confirmed",
            ],
        ),
        ReviewIntent::Roadmap | ReviewIntent::Gaps => contains_any(
            &lower,
            &[
                "missing",
                "gap",
                "roadmap",
                "build next",
                "coverage",
                "risk",
                "work",
                "confirmed",
            ],
        ),
        ReviewIntent::Review => contains_any(
            &lower,
            &[
                "finding",
                "review",
                "issue",
                "risk",
                "follow-up",
                "follow up",
                "confirmed",
            ],
        ),
    };
    has_evidence_language && has_bounded_limit && has_review_signal
}

pub(crate) fn answer_looks_like_generic_inventory_summary(answer: &str) -> bool {
    let lower = answer.to_ascii_lowercase();
    let inventory_markers = [
        "codebase is",
        "project is",
        "repository is",
        "structured with",
        "consists of",
        "main components",
        "main functionality",
        "key features",
        "workspace setup",
        "entry point",
        "support for multiple",
        "supports multiple",
        "the exact count can be determined",
        "approximately ",
    ];
    let marker_count = inventory_markers
        .iter()
        .filter(|marker| lower.contains(**marker))
        .count();
    let has_bounded_review_language = contains_any(
        &lower,
        &[
            "findings:",
            "status:",
            "evidence:",
            "inspected evidence",
            "risks/validation",
            "build next",
            "missing/gaps",
            "limits:",
            "based on the inspected",
            "from the inspected",
            "not a complete",
        ],
    );
    marker_count >= 2 && !has_bounded_review_language
}

pub(crate) fn should_nudge_read_after_repeated_search(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
) -> bool {
    intent.is_some() && evidence.saw_search && !evidence.saw_read
}

pub(crate) fn should_nudge_read_after_search_final(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    _answer: &str,
) -> bool {
    intent.is_some() && evidence.saw_search && !evidence.saw_read
}

pub(crate) fn should_nudge_security_broad_search(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    matches!(intent, Some(ReviewIntent::Security))
        && evidence.saw_search
        && evidence.saw_read
        && !evidence.security_search_complete()
        && !answer_says_insufficient_evidence(answer)
}

pub(crate) fn should_nudge_security_scope(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    matches!(intent, Some(ReviewIntent::Security))
        && evidence.saw_search
        && evidence.saw_read
        && security_answer_overclaims_scope(answer)
}

pub(crate) fn should_nudge_gap_search_overclaim(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    matches!(intent, Some(ReviewIntent::Gaps | ReviewIntent::Roadmap))
        && evidence.grep_match_lines > 0
        && gap_answer_overclaims_absence(answer)
}

pub(crate) fn security_answer_overclaims_scope(answer: &str) -> bool {
    if answer_says_insufficient_evidence(answer) {
        return false;
    }
    let lower = answer.to_ascii_lowercase();
    let broad_all_clear = contains_any(
        &lower,
        &[
            "the codebase does not contain",
            "the codebase doesn't contain",
            "the codebase appears to be secure",
            "codebase appears secure",
            "secure against common unsafe patterns",
            "there are no hardcoded secrets",
            "no hardcoded secrets",
            "no direct command execution",
            "does not contain any unsafe",
            "doesn't contain any unsafe",
            "no security issues",
            "no security-critical issues",
        ],
    );
    let bounded = contains_any(
        &lower,
        &[
            "insufficient evidence",
            "limited to",
            "based on the inspected",
            "based on searched",
            "based on the searched",
            "from the inspected",
            "in the inspected",
            "i only inspected",
            "not a complete audit",
            "cannot rule out",
            "cannot make broad",
        ],
    );
    broad_all_clear && !bounded
}

pub(crate) fn gap_answer_overclaims_absence(answer: &str) -> bool {
    if answer_says_insufficient_evidence(answer) {
        return false;
    }
    let lower = answer.to_ascii_lowercase();
    let broad_absence = contains_any(
        &lower,
        &[
            "no todo",
            "no todos",
            "no todo/fixme",
            "no fixme",
            "no fixmes",
            "no missing implementations",
            "no obvious gaps",
            "no obvious missing",
            "no obvious gaps in functionality",
            "appears mature with no obvious gaps",
            "shows no obvious gaps",
        ],
    );
    let bounded = contains_any(
        &lower,
        &[
            "based on the inspected",
            "based on searched",
            "based on the searched",
            "from the inspected",
            "in the inspected",
            "i only inspected",
            "not a complete",
            "cannot rule out",
            "cannot make broad",
        ],
    );
    broad_absence && !bounded
}

pub(crate) fn read_only_intent_label(intent: ReviewIntent) -> &'static str {
    match intent {
        ReviewIntent::Security => "security review",
        ReviewIntent::Status => "status review",
        ReviewIntent::Roadmap => "roadmap review",
        ReviewIntent::Gaps => "gap review",
        ReviewIntent::Review => "review",
    }
}

pub(crate) fn inspected_paths_for_prompt(evidence: &EvidenceTracker) -> String {
    if evidence.inspected_paths.is_empty() {
        return "none".to_string();
    }
    const LIMIT: usize = 8;
    let mut paths = evidence
        .inspected_paths
        .iter()
        .take(LIMIT)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let omitted = evidence.inspected_paths.len().saturating_sub(LIMIT);
    if omitted > 0 {
        paths.push_str(&format!(" (+{omitted} more)"));
    }
    paths
}

pub(crate) fn summarize_inspected_evidence_nudge(
    intent: ReviewIntent,
    evidence: &EvidenceTracker,
) -> String {
    let label = read_only_intent_label(intent);
    let paths = inspected_paths_for_prompt(evidence);
    match intent {
        ReviewIntent::Security => format!(
            "You already have inspected evidence for this {label}. Do not answer with a generic refusal or evidence disclaimer. Produce a bounded security review from only the inspected searches/files. Cite concrete inspected files from this set: {paths}. Include: Findings, Inspected Evidence, Limits, and Follow-up. If a pattern match is only a review target, say that it is not confirmed."
        ),
        ReviewIntent::Status => format!(
            "You already have inspected evidence for this {label}. Do not answer with a generic refusal or evidence disclaimer. Produce a bounded status review from only the inspected files. Cite concrete inspected files from this set: {paths}. Include: Status, Evidence, Build Next, and Risks/Validation. Do not claim repo-wide completeness."
        ),
        ReviewIntent::Roadmap | ReviewIntent::Gaps => format!(
            "You already have inspected evidence for this {label}. Do not answer with a generic refusal or evidence disclaimer. Produce bounded gaps/build-next notes from only the inspected files and searches. Cite concrete inspected files from this set: {paths}. Include: Missing/Gaps, Build Next, Evidence, and Risks. Do not claim repo-wide completeness."
        ),
        ReviewIntent::Review => format!(
            "You already have inspected evidence for this {label}. Do not answer with a generic refusal or evidence disclaimer. Produce bounded findings from only the inspected files/searches. Cite concrete inspected files from this set: {paths}. Include findings, evidence, follow-up, and limits."
        ),
    }
}

pub(crate) fn no_evidence_review_nudge(intent: ReviewIntent) -> &'static str {
    match intent {
        ReviewIntent::Security => NO_EVIDENCE_SECURITY_NUDGE,
        ReviewIntent::Status => NO_EVIDENCE_STATUS_NUDGE,
        ReviewIntent::Roadmap | ReviewIntent::Gaps => NO_EVIDENCE_GAP_NUDGE,
        ReviewIntent::Review => NO_EVIDENCE_REVIEW_NUDGE,
    }
}

pub(crate) fn deepen_review_nudge(intent: ReviewIntent) -> &'static str {
    match intent {
        ReviewIntent::Security => SECURITY_DEEPEN_NUDGE,
        ReviewIntent::Status => STATUS_DEEPEN_NUDGE,
        ReviewIntent::Roadmap | ReviewIntent::Gaps => GAP_DEEPEN_NUDGE,
        ReviewIntent::Review => REVIEW_DEEPEN_NUDGE,
    }
}

pub(crate) fn read_only_blocks_tool(intent: Option<ReviewIntent>, name: &str) -> bool {
    // `explore` isn't classified read-only (so a read-only child can't spawn one),
    // but it only ever launches a read-only subagent — so it's allowed to run in a
    // review turn. A subagent is never advertised `explore`, so it can't reach here.
    intent.is_some() && !hi_tools::is_read_only(name) && name != "explore"
}

pub(crate) fn read_only_blocked_tool_result(name: &str) -> String {
    format!(
        "Tool `{name}` blocked: this is a read-only review/discuss-only turn. Use read-only inspection tools and answer from inspected evidence; do not modify files."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_findings_count_as_review_shape() {
        assert!(
            !answer_lacks_review_shape(
                ReviewIntent::Review,
                "[P1] Missing nil check — src/parser.rs:40\nThe new branch unwraps on empty input."
            ),
            "P-findings must not trip generic-template repair"
        );
        assert!(!answer_lacks_review_shape(
            ReviewIntent::Review,
            "No findings. Residual risk is low."
        ));
        assert!(answer_is_concise_bounded_review(
            ReviewIntent::Review,
            "[P0] Auth bypass — crates/hi-cli/src/auth.rs:12"
        ));
    }

    #[test]
    fn generic_completion_guard_feature_state_is_compile_time_scoped() {
        #[cfg(feature = "smoke-negative-control-disable-generic-completion-guards")]
        assert!(!generic_completion_guards_enabled());
        #[cfg(not(feature = "smoke-negative-control-disable-generic-completion-guards"))]
        assert!(generic_completion_guards_enabled());
    }
}
