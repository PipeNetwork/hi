//! Nudge decision functions and answer-shape checks. Uses nudge strings from
//! [`constants`](super::constants), [`contains_any`] from [`intent`](super::intent),
//! and tracker types from [`types`](super::types).

use super::intent::contains_any;
use super::types::{EvidenceTracker, ReviewIntent};
pub(crate) fn implementation_text_tool_nudge(reason: &str) -> String {
    format!(
        "{reason}\n\nThe next request will describe the plain-text call format from its sealed tool envelope."
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
    let explicit_no_change = contains_any(
        &lower,
        &[
            "no file changes are needed",
            "no file changes are required",
            "no file changes were needed",
            "no file changes were required",
            "no further file changes are needed",
            "no further file changes were needed",
            "no further file changes are required",
            "no further file changes were required",
            "no file changes needed",
            "no code changes are needed",
            "no code changes were needed",
            "no code changes needed",
            "no changes are needed",
            "no changes are required",
            "no edits are needed",
            "no edits were needed",
            "no edits are required",
            "requires no file changes",
            "requires no code changes",
            "no changes necessary",
            "no change is needed",
            "nothing needs to change",
            "no modifications are needed",
            "no modifications needed",
        ],
    );
    // A bare refusal is not evidence that an explicit fix is already
    // satisfied. Require the no-change conclusion to carry a concrete reason;
    // otherwise weak models can escape the mutation obligation with "I won't
    // edit" or an unsupported "out of scope" response.
    let evidence_backed = contains_any(
        &lower,
        &[
            " because ",
            " already ",
            "already correct",
            "already rejects",
            "already handles",
            "does not reproduce",
            "doesn't reproduce",
            "current implementation",
            "existing implementation",
            "reported bug",
            "report was",
        ],
    );
    explicit_no_change && evidence_backed
}

pub(crate) fn should_nudge_read_after_repeated_search(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
) -> bool {
    intent.is_some() && evidence.saw_search && !evidence.saw_read
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
    format!(
        "Repeated inspection is no longer producing new results for this {label}. Give your answer from the available results, including any unresolved questions. Inspected paths: {paths}."
    )
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
    fn generic_completion_guard_feature_state_is_compile_time_scoped() {
        #[cfg(feature = "smoke-negative-control-disable-generic-completion-guards")]
        assert!(!generic_completion_guards_enabled());
        #[cfg(not(feature = "smoke-negative-control-disable-generic-completion-guards"))]
        assert!(generic_completion_guards_enabled());
    }

    #[test]
    fn mutation_decline_requires_a_no_change_conclusion_and_reason() {
        assert!(answer_declines_mutation(
            "No file changes are needed because the current implementation already handles it."
        ));
        assert!(answer_declines_mutation(
            "No further file changes were needed — the request was already satisfied by the prior turn's work, and the current state passes all checks."
        ));
        for unsupported in [
            "I won't modify the files.",
            "That is out of scope.",
            "No file changes are needed.",
            "The work is already done.",
        ] {
            assert!(!answer_declines_mutation(unsupported), "{unsupported:?}");
        }
    }
}
