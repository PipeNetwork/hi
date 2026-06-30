use super::*;
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

pub(crate) fn should_deepen_review(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    intent.is_some() && evidence.listing_only() && !answer_says_insufficient_evidence(answer)
}

pub(crate) fn should_nudge_no_evidence_review(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    intent.is_some() && !evidence.has_discovery() && !answer_says_insufficient_evidence(answer)
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

pub(crate) fn should_reject_review_repair_template(intent: Option<ReviewIntent>, answer: &str) -> bool {
    intent.is_some()
        && !answer_says_insufficient_evidence(answer)
        && answer_looks_like_review_repair_template(answer)
}

pub(crate) fn should_nudge_concrete_review_answer(
    intent: Option<ReviewIntent>,
    evidence: &EvidenceTracker,
    answer: &str,
) -> bool {
    let Some(intent) = intent else {
        return false;
    };
    if evidence.inspected_paths.is_empty() || answer_says_insufficient_evidence(answer) {
        return false;
    }
    let cites_inspected_path = evidence
        .inspected_paths
        .iter()
        .any(|path| answer.contains(path));
    !cites_inspected_path
        || answer_looks_like_generic_inventory_summary(answer)
        || answer_lacks_review_shape(intent, answer)
}

pub(crate) fn answer_lacks_review_shape(intent: ReviewIntent, answer: &str) -> bool {
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
    answer: &str,
) -> bool {
    intent.is_some()
        && evidence.saw_search
        && !evidence.saw_read
        && !answer_says_insufficient_evidence(answer)
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

pub(crate) fn insufficient_after_repeated_search(evidence: &EvidenceTracker) -> Option<&'static str> {
    if evidence.saw_search && !evidence.saw_read {
        Some(
            "Insufficient evidence: targeted search ran, but no matching file was read, so I cannot make file-specific review findings.",
        )
    } else {
        None
    }
}

pub(crate) fn insufficient_after_incomplete_security_search(evidence: &EvidenceTracker) -> Option<String> {
    if !evidence.saw_search || !evidence.saw_read || evidence.security_search_complete() {
        return None;
    }
    let mut missing = Vec::new();
    if !evidence.security_unsafe_search {
        missing.push("unsafe/unwrap/expect/panic");
    }
    if !evidence.security_execution_search {
        missing.push("command execution/filesystem/env");
    }
    if !evidence.security_secret_search {
        missing.push("secret/token/auth");
    }
    Some(format!(
        "Insufficient evidence: the security review did not search all required pattern families (missing {}), so I cannot make broad security claims.",
        missing.join(", ")
    ))
}

pub(crate) fn insufficient_after_security_scope_overclaim() -> &'static str {
    "Insufficient evidence: the security answer made repo-wide all-clear claims that were broader than the inspected files and search results support."
}

pub(crate) fn insufficient_after_no_review_evidence() -> &'static str {
    "Insufficient evidence: no files, searches, diffs, or directory listings were inspected, so I cannot present this as a completed review."
}

pub(crate) fn insufficient_after_review_repair_template() -> &'static str {
    "Insufficient evidence: the answer was a generic review-repair template instead of concrete findings tied to inspected files, so I cannot present this as a completed review."
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

pub(crate) fn bounded_review_repair_exhaustion_answer(
    intent: ReviewIntent,
    evidence: &EvidenceTracker,
    reason: &str,
) -> String {
    let label = read_only_intent_label(intent);
    let mut lines = vec![
        format!(
            "Bounded evidence summary for an incomplete {label}: the model inspected evidence but did not produce acceptable file-specific findings after repair."
        ),
        String::new(),
        "Inspected evidence:".to_string(),
        format!("- Targeted searches: {}", evidence.targeted_searches),
        format!("- File reads: {}", evidence.file_reads),
    ];

    if matches!(intent, ReviewIntent::Security) {
        let mut families = Vec::new();
        if evidence.security_unsafe_search {
            families.push("unsafe/unwrap/expect/panic");
        }
        if evidence.security_execution_search {
            families.push("command execution/filesystem/env");
        }
        if evidence.security_secret_search {
            families.push("secret/token/auth");
        }
        let searched = if families.is_empty() {
            "none".to_string()
        } else {
            families.join(", ")
        };
        lines.push(format!("- Security pattern families searched: {searched}"));
    }

    if evidence.inspected_paths.is_empty() {
        lines.push("- Inspected files: none".to_string());
    } else {
        const INSPECTED_PATH_FALLBACK_LIMIT: usize = 6;
        let mut paths = evidence
            .inspected_paths
            .iter()
            .take(INSPECTED_PATH_FALLBACK_LIMIT)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let omitted = evidence
            .inspected_paths
            .len()
            .saturating_sub(INSPECTED_PATH_FALLBACK_LIMIT);
        if omitted > 0 {
            paths.push_str(&format!(" (+{omitted} more)"));
        }
        lines.push(format!("- Inspected files: {paths}"));
    }

    if !evidence.search_hit_snippets.is_empty() {
        const SEARCH_HIT_FALLBACK_LIMIT: usize = 6;
        lines.push(String::new());
        lines.push("Concrete search matches from inspected evidence:".to_string());
        for snippet in evidence
            .search_hit_snippets
            .iter()
            .take(SEARCH_HIT_FALLBACK_LIMIT)
        {
            lines.push(format!("- {snippet}"));
        }
        let omitted = evidence
            .search_hit_snippets
            .len()
            .saturating_sub(SEARCH_HIT_FALLBACK_LIMIT);
        if omitted > 0 {
            lines.push(format!("- (+{omitted} more search match target(s))"));
        }
        lines.push(
            "These are pattern-match review targets, not confirmed vulnerabilities or all-clear findings."
                .to_string(),
        );
    }

    lines.push(String::new());
    lines.push(format!("Why this stopped: {reason}."));
    lines.push(
        "No file is being changed; this turn remains read-only and no broader repo-wide claim is being made."
            .to_string(),
    );
    lines.join("\n")
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

pub(crate) fn summarize_inspected_evidence_nudge(intent: ReviewIntent, evidence: &EvidenceTracker) -> String {
    let label = read_only_intent_label(intent);
    let paths = inspected_paths_for_prompt(evidence);
    match intent {
        ReviewIntent::Security => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce a bounded security review from only the inspected searches/files. Cite concrete inspected files from this set: {paths}. Include: Findings, Inspected Evidence, Limits, and Follow-up. If a pattern match is only a review target, say that it is not confirmed."
        ),
        ReviewIntent::Status => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce a bounded status review from only the inspected files. Cite concrete inspected files from this set: {paths}. Include: Status, Evidence, Build Next, and Risks/Validation. Do not claim repo-wide completeness."
        ),
        ReviewIntent::Roadmap | ReviewIntent::Gaps => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce bounded gaps/build-next notes from only the inspected files and searches. Cite concrete inspected files from this set: {paths}. Include: Missing/Gaps, Build Next, Evidence, and Risks. Do not claim repo-wide completeness."
        ),
        ReviewIntent::Review => format!(
            "You already have inspected evidence for this {label}. Do not answer with generic insufficient-evidence text. Produce bounded findings from only the inspected files/searches. Cite concrete inspected files from this set: {paths}. Include findings, evidence, follow-up, and limits."
        ),
    }
}

pub(crate) fn inspected_insufficient_repair_limit(intent: ReviewIntent) -> u32 {
    match intent {
        ReviewIntent::Security => 3,
        ReviewIntent::Status
        | ReviewIntent::Roadmap
        | ReviewIntent::Gaps
        | ReviewIntent::Review => 2,
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
    intent.is_some() && !hi_tools::is_read_only(name)
}

pub(crate) fn read_only_blocked_tool_result(name: &str) -> String {
    format!(
        "Tool `{name}` blocked: this is a read-only review/discuss-only turn. Use read-only inspection tools and answer from inspected evidence; do not modify files."
    )
}

