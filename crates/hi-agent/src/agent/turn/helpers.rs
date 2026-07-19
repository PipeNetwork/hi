//! Shared free helpers for the turn loop: telemetry, routing, tool entries.

use crate::heuristics::humanize_count;
use hi_ai::{RateLimitBucket, RateLimitState};

use crate::steering::{EvidenceTracker, ReviewIntent};
use crate::{
    EffectiveModelRoute, ReviewStatus, TaskContract, TaskIntent, ToolCallEntry, TurnAttribution,
    TurnTelemetry,
};

use super::progress::{ProgressTracker, ToolProgressLabel};
use super::retry::ReviewRepairState;

#[allow(clippy::too_many_arguments)]
pub(super) fn build_turn_telemetry(
    effective_max_steps: u32,
    verify_rounds: u32,
    recovery_retries: u32,
    repeat_nudges: u32,
    continue_nudges: u32,
    truncation_retries: u32,
    progress: &ProgressTracker,
    hit_step_cap: bool,
    stalled_unfinished: bool,
    stalled_repeating: bool,
    verify_attributions: &[hi_tools::Attribution],
    verification_executions: &[crate::VerificationExecution],
    tool_calls: u32,
    max_concurrent_batch: u32,
    serial_runs: u32,
    tool_timeline: &[ToolCallEntry],
    evidence: &EvidenceTracker,
    review_repair: &ReviewRepairState,
) -> TurnTelemetry {
    TurnTelemetry {
        effective_max_steps,
        verify_rounds,
        recovery_retries,
        repeat_nudges,
        continue_nudges,
        truncation_retries,
        no_progress_streak: progress.no_progress_streak,
        forced_final_answer_attempts: progress.forced_final_answer_attempts,
        last_progress_reason: progress.last_progress_reason.clone(),
        last_stall_reason: progress.last_stall_reason.clone(),
        hit_step_cap,
        stalled_unfinished,
        stalled_repeating,
        verify_attributions: verify_attributions
            .iter()
            .map(TurnAttribution::from)
            .collect(),
        verification_executions: verification_executions.to_vec(),
        tool_calls,
        max_concurrent_batch,
        serial_runs,
        tool_timeline: tool_timeline.to_vec(),
        progress_events: progress.events.clone(),
        file_reads: evidence.file_reads,
        targeted_searches: evidence.targeted_searches,
        listing_only: evidence.listing_only(),
        first_tool_kind: evidence.first_tool_kind().to_string(),
        discovery_depth: evidence.discovery_depth().to_string(),
        quality_repair_nudges: evidence.quality_repair_nudges,
        review_repair_exhaustion_reason: review_repair.exhaustion_reason.clone(),
        review_repair_counts: review_repair.counts.clone(),
        review_repair_stopped_by_exhaustion: !review_repair.exhaustion_reason.is_empty(),
        skeptic_unavailable_count: 0,
        skeptic_last_status: None,
        checkpoint_available: None,
        advertised_tools: Vec::new(),
        tool_schema_tokens: 0,
    }
}

pub(super) fn effective_max_steps_for_turn(
    config: &crate::AgentConfig,
    contract_intent: TaskIntent,
    read_only_intent: Option<ReviewIntent>,
    implementation_intent: Option<crate::steering::ImplementationIntent>,
) -> u32 {
    if config.max_steps_explicit {
        return config.max_steps.max(1);
    }
    // Intent-aware per-turn cap, regardless of `long_horizon`. A long-horizon goal
    // spans many turns (each advancing one sub-goal), so each turn gets the normal
    // per-intent budget — not a flat 200 that would also apply when no goal is set.
    if contract_intent == TaskIntent::ReadOnly || read_only_intent.is_some() {
        80
    } else if implementation_intent.is_some() {
        120
    } else {
        200
    }
}

pub(super) fn task_needs_repository_context(task: &str, contract: &TaskContract) -> bool {
    if !contract.referenced_paths.is_empty() {
        return true;
    }
    let lower = format!(" {} ", task.to_ascii_lowercase());
    [
        " add ",
        " build ",
        " change ",
        " code",
        " config",
        " create ",
        " debug",
        " delete ",
        " edit ",
        " file",
        " fix ",
        " implement ",
        " migrate ",
        " refactor ",
        " repo",
        " remove ",
        " rename ",
        " replace ",
        " src/",
        " test",
        " update ",
        " write ",
        ".go",
        ".js",
        ".py",
        ".rs",
        ".ts",
        // Comprehension/orientation markers. Omitting these caused a live
        // regression: "what does this program do" matched no marker, so the
        // turn ran with NO task context index — a repo-blind model has
        // nothing to anchor on and (observed across two different models)
        // falls back to re-posting its plan instead of exploring. Questions
        // about "this program/project" are exactly the tasks that need the
        // repository map most.
        " program",
        " project",
        " codebase",
        " architecture",
        " explain",
        " describe",
        " overview",
        " understand",
        " summarize",
        " what ",
        " how ",
        " where ",
        " why ",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
}

pub(super) fn tool_satisfies_validation(output: &hi_tools::ToolOutcome) -> bool {
    output.satisfies_validation()
}

pub(super) fn tool_entry(
    tool: String,
    path: String,
    duration_ms: u64,
    output: &hi_tools::ToolOutcome,
    progress: &ToolProgressLabel,
) -> ToolCallEntry {
    ToolCallEntry {
        tool,
        path,
        duration_ms,
        status: output.status,
        background: output.background.clone(),
        process: output.process.clone(),
        effects: output.effects.clone(),
        truncation: output.truncation.clone(),
        error: output.status != hi_tools::ToolStatus::Succeeded,
        progress_kind: progress.kind.as_str().to_string(),
        progress_reason: progress.reason.clone(),
        normalized_signature: progress.signature.clone(),
    }
}

pub(super) fn synthetic_tool_outcome(
    content: String,
    status: hi_tools::ToolStatus,
) -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content,
        display: None,
        plan: None,
        status,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation: hi_tools::TruncationState::Complete,
    }
}

pub(super) fn effective_model_route(
    config: &crate::AgentConfig,
    fallback_route: Option<&str>,
) -> EffectiveModelRoute {
    if let Some(route) = fallback_route {
        let (provider, model) = route
            .split_once('/')
            .map(|(provider, model)| (Some(provider.to_string()), model.to_string()))
            .unwrap_or_else(|| (None, route.to_string()));
        EffectiveModelRoute { provider, model }
    } else {
        EffectiveModelRoute {
            provider: config.provider_route.clone(),
            model: config.model.clone(),
        }
    }
}

/// Fold the independent completion reviewer and the optional long-horizon
/// skeptic into the single public review status. Any concrete objection is
/// fail-closed; infrastructure unavailability remains visible; otherwise a
/// pass from either configured reviewer is retained.
pub(super) fn combined_review_status(
    independent: ReviewStatus,
    skeptic: ReviewStatus,
) -> ReviewStatus {
    use ReviewStatus::{NotRequired, Objected, Passed, Unavailable};
    if independent == Objected || skeptic == Objected {
        Objected
    } else if independent == Unavailable || skeptic == Unavailable {
        Unavailable
    } else if independent == Passed || skeptic == Passed {
        Passed
    } else {
        NotRequired
    }
}

/// Late workspace deltas that only touch prose (docs, learned skills under
/// `.hi/skills/`, etc.) must not wipe a deterministic verification pass. The
/// auto-pipeline never covers those paths (`SkippedProseOnly`), so treating a
/// skill-curation write as "unverified changes" is a false alarm users hate.
pub(super) fn post_verify_delta_is_benign(changes: &[hi_tools::FileChange]) -> bool {
    !changes.is_empty()
        && changes
            .iter()
            .all(|change| crate::verify::is_prose_only_path(&change.path))
}

/// Conservative fallback used only when a checkpoint-backed unified diff is
/// unavailable (for example, the user explicitly allowed mutation without an
/// undo snapshot). It prevents that escape hatch from also bypassing the
/// risk-review threshold. The reviewer still receives `Unavailable` rather
/// than an invented diff; this count is solely a trigger.
pub(super) fn fallback_review_line_count(
    root: &std::path::Path,
    changes: &[hi_tools::FileChange],
) -> usize {
    const TRIGGER: usize = 301;
    let mut lines = 0usize;
    for change in changes {
        let path = root.join(&change.path);
        if let Ok(metadata) = std::fs::symlink_metadata(&path)
            && metadata.is_file()
            && let Ok(mut file) = std::fs::File::open(&path)
        {
            let mut buffer = [0_u8; 16 * 1024];
            let mut scanned = 0usize;
            while lines < TRIGGER && scanned < 2 * 1024 * 1024 {
                let Ok(read) = std::io::Read::read(&mut file, &mut buffer) else {
                    break;
                };
                if read == 0 {
                    // A non-empty final line has no terminating newline.
                    if metadata.len() > 0 {
                        lines = lines.saturating_add(1).min(TRIGGER);
                    }
                    break;
                }
                scanned = scanned.saturating_add(read);
                lines = lines
                    .saturating_add(buffer[..read].iter().filter(|byte| **byte == b'\n').count())
                    .min(TRIGGER);
            }
        } else if change.after_digest.is_none() {
            // Deleted contents are unavailable without a checkpoint. Treat a
            // sufficiently large deletion as review-worthy instead of silently
            // under-counting it.
            lines = lines
                .saturating_add(change.before_len.unwrap_or_default().min(TRIGGER as u64) as usize);
        }
        if lines >= TRIGGER {
            return TRIGGER;
        }
    }
    lines
}

pub(super) fn rate_limit_summary(limits: RateLimitState) -> Option<String> {
    if !limits.has_data() {
        return None;
    }
    let mut parts = Vec::new();
    if let Some(part) = rate_limit_bucket_summary("req", limits.requests_min) {
        parts.push(part);
    } else if let Some(part) = rate_limit_bucket_summary("req/hr", limits.requests_hour) {
        parts.push(part);
    }
    if let Some(part) = rate_limit_bucket_summary("tok", limits.tokens_min) {
        parts.push(part);
    } else if let Some(part) = rate_limit_bucket_summary("tok/hr", limits.tokens_hour) {
        parts.push(part);
    }
    (!parts.is_empty()).then(|| format!("limits {}", parts.join(" · ")))
}

pub(super) fn rate_limit_bucket_summary(label: &str, bucket: RateLimitBucket) -> Option<String> {
    if bucket.limit == 0 {
        return None;
    }
    let reset = if bucket.reset_seconds > 0 {
        format!(" reset {}", format_rate_limit_reset(bucket.reset_seconds))
    } else {
        String::new()
    };
    Some(format!(
        "{label} {}/{}{reset}",
        humanize_count(bucket.remaining),
        humanize_count(bucket.limit)
    ))
}

pub(super) fn format_rate_limit_reset(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3599 => {
            let minutes = seconds / 60;
            let secs = seconds % 60;
            if secs == 0 {
                format!("{minutes}m")
            } else {
                format!("{minutes}m {secs}s")
            }
        }
        _ => {
            let hours = seconds / 3600;
            let minutes = (seconds % 3600) / 60;
            if minutes == 0 {
                format!("{hours}h")
            } else {
                format!("{hours}h {minutes}m")
            }
        }
    }
}

#[cfg(test)]
mod step_cap_tests {
    use super::*;
    use crate::steering::ImplementationIntent;
    use hi_tools::{FileChange, FileChangeKind};

    fn cfg(long_horizon: bool) -> crate::AgentConfig {
        crate::AgentConfig {
            long_horizon,
            max_steps_explicit: false,
            ..Default::default()
        }
    }

    fn change(path: &str) -> FileChange {
        FileChange {
            path: path.into(),
            kind: FileChangeKind::Modify,
            before_digest: None,
            after_digest: None,
            before_len: None,
            after_len: None,
            before_mode: None,
            after_mode: None,
        }
    }

    #[test]
    fn post_verify_prose_delta_is_benign_code_is_not() {
        assert!(post_verify_delta_is_benign(&[change(
            ".hi/skills/retry/SKILL.md"
        )]));
        assert!(post_verify_delta_is_benign(&[change("README.md")]));
        assert!(!post_verify_delta_is_benign(&[change("src/lib.rs")]));
        assert!(!post_verify_delta_is_benign(&[
            change("README.md"),
            change("src/lib.rs"),
        ]));
        assert!(!post_verify_delta_is_benign(&[]));
    }

    #[test]
    fn max_steps_is_intent_aware_even_with_long_horizon() {
        // Decoupled from `long_horizon`: each turn gets its per-intent cap whether
        // or not a long-horizon goal is active (the goal spans many turns).
        for lh in [false, true] {
            assert_eq!(
                effective_max_steps_for_turn(
                    &cfg(lh),
                    TaskIntent::Mutation,
                    None,
                    Some(ImplementationIntent::default())
                ),
                120,
                "implementation intent (long_horizon={lh})"
            );
            assert_eq!(
                effective_max_steps_for_turn(
                    &cfg(lh),
                    TaskIntent::ReadOnly,
                    Some(ReviewIntent::Security),
                    None
                ),
                80,
                "read-only intent (long_horizon={lh})"
            );
            assert_eq!(
                effective_max_steps_for_turn(&cfg(lh), TaskIntent::Mutation, None, None),
                200,
                "no intent (long_horizon={lh})"
            );
        }
    }

    #[test]
    fn explicit_max_steps_always_wins() {
        let mut c = cfg(true);
        c.max_steps_explicit = true;
        c.max_steps = 42;
        assert_eq!(
            effective_max_steps_for_turn(&c, TaskIntent::ReadOnly, None, None),
            42
        );
    }

    #[test]
    fn independent_and_skeptic_review_statuses_are_combined_fail_closed() {
        assert_eq!(
            combined_review_status(ReviewStatus::Passed, ReviewStatus::NotRequired),
            ReviewStatus::Passed
        );
        assert_eq!(
            combined_review_status(ReviewStatus::Passed, ReviewStatus::Unavailable),
            ReviewStatus::Unavailable
        );
        assert_eq!(
            combined_review_status(ReviewStatus::Unavailable, ReviewStatus::Objected),
            ReviewStatus::Objected
        );
        assert_eq!(
            combined_review_status(ReviewStatus::NotRequired, ReviewStatus::Passed),
            ReviewStatus::Passed
        );
    }
}
