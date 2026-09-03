//! Shared free helpers for the turn loop: telemetry, routing, tool entries.

use crate::heuristics::humanize_count;
use hi_ai::{RateLimitBucket, RateLimitState};

use crate::steering::EvidenceTracker;
use crate::{
    EffectiveModelRoute, ReviewStatus, TaskContract, ToolCallEntry, TurnAttribution, TurnTelemetry,
};

use super::progress::{ProgressTracker, ToolProgressLabel};
use super::retention::ToolTimeline;
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
    hit_tool_cap: bool,
    verify_attributions: &[hi_tools::Attribution],
    verification_executions: &[crate::VerificationExecution],
    verification_executions_dropped: u64,
    verification_execution_count: u64,
    successful_test_verification: bool,
    tool_calls: u32,
    max_concurrent_batch: u32,
    serial_runs: u32,
    tool_timeline: &ToolTimeline,
    evidence: &EvidenceTracker,
    review_repair: &ReviewRepairState,
    prefix_stability: &crate::prefix_stability::PrefixStability,
) -> TurnTelemetry {
    TurnTelemetry {
        phase_latencies: crate::TurnPhaseLatencies::default(),
        effective_max_steps,
        verify_rounds,
        recovery_retries,
        repeat_nudges,
        continue_nudges,
        truncation_retries,
        no_progress_streak: progress.no_progress_streak,
        forced_final_answer_attempts: progress.forced_final_answer_attempts,
        last_progress_reason: progress.last_progress_reason.clone(),
        last_no_progress_reason: progress.last_no_progress_reason.clone(),
        hit_step_cap,
        hit_tool_cap,
        verify_attributions: verify_attributions
            .iter()
            .map(TurnAttribution::from)
            .collect(),
        verification_executions: verification_executions.to_vec(),
        tool_calls,
        max_concurrent_batch,
        serial_runs,
        tool_timeline: tool_timeline.to_vec(),
        progress_events: progress.retained_events(),
        drive_evidence_hashes: progress.drive_evidence_hashes(),
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
        review_unavailable_reason: None,
        checkpoint_available: None,
        advertised_tools: Vec::new(),
        tool_schema_tokens: 0,
        prefix_stable_rounds: prefix_stability.stable_rounds,
        prefix_break_rounds: prefix_stability.break_rounds,
        tool_prefix_break_rounds: prefix_stability.tool_break_rounds,
        earliest_prefix_break: prefix_stability.earliest_break,
        model_requests: 0,
        accepted_completions: 0,
        last_stop_reason: None,
        tool_call_channel: "none".to_string(),
        reasoning_requested: false,
        reasoning_received: false,
        reasoning_replayed: false,
        reasoning_signature_replayed: false,
        reasoning_fallback: false,
        refusal_source: None,
        wire_audit: Vec::new(),
        requests: Vec::new(),
        compaction: Vec::new(),
        diagnostic_retention: crate::TurnDiagnosticRetention {
            progress_events_dropped: progress.retained_events_dropped(),
            tool_timeline_dropped: tool_timeline.dropped(),
            verification_executions_dropped,
            wire_audit_dropped: 0,
            requests_dropped: 0,
            compaction_events_dropped: 0,
            verification_executions_total: verification_execution_count,
            successful_test_verification,
        },
    }
}

/// The per-turn model-call cap. `u32::MAX` is the ordinary unlimited default;
/// finite values come from an explicit user or internal budget.
pub(super) fn effective_max_steps_for_turn(config: &crate::AgentConfig) -> u32 {
    config.loop_limits.max_steps.max(1)
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
    tool_entry_with_args(tool, path, duration_ms, output, progress, "")
}

pub(super) fn tool_entry_with_args(
    tool: String,
    path: String,
    duration_ms: u64,
    output: &hi_tools::ToolOutcome,
    progress: &ToolProgressLabel,
    arguments: &str,
) -> ToolCallEntry {
    let command = bash_command_preview(&tool, arguments);
    let kind = crate::tool_kind(&tool).to_string();
    let truncated = !matches!(output.truncation, hi_tools::TruncationState::Complete);
    let path = {
        let all = hi_tools::target_paths(&tool, arguments);
        if all.len() > 1 { all.join("\n") } else { path }
    };
    ToolCallEntry {
        tool,
        path,
        duration_ms,
        queue_delay_ms: 0,
        completion_index: 0,
        status: output.status,
        background: output.background.clone(),
        process: output.process.clone(),
        effects: output.effects.clone(),
        truncation: output.truncation.clone(),
        error: output.status != hi_tools::ToolStatus::Succeeded,
        progress_kind: progress.kind.as_str().to_string(),
        progress_reason: progress.reason.clone(),
        normalized_signature: progress.signature.clone(),
        command,
        arg_chars: arguments.chars().count() as u64,
        result_chars: output.content.chars().count() as u64,
        truncated,
        kind,
    }
}

fn bash_command_preview(tool: &str, arguments: &str) -> Option<String> {
    if tool != "bash" {
        return None;
    }
    crate::steering::bash_command(arguments).map(|command| {
        let trimmed = command.trim();
        if trimmed.chars().count() > 240 {
            format!("{}…", trimmed.chars().take(239).collect::<String>())
        } else {
            trimmed.to_string()
        }
    })
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
        images: Vec::new(),
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
            provider: config.routing.provider_route.clone(),
            model: config.routing.model.clone(),
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
    use ReviewStatus::{Escalated, NotRequired, Objected, Passed, Unavailable};
    // Objections fail-closed over everything else.
    if independent == Objected || skeptic == Objected {
        Objected
    } else if independent == Escalated || skeptic == Escalated {
        // Escalated is weaker than Objected: visible scar, not a defect block.
        Escalated
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
///
/// An **empty** delta is also benign: the ledger revision/digest can move from
/// reconcile bookkeeping without any file change. Treating that as a wipe was
/// flipping green turns into failed, unverified outcomes.
pub(super) fn post_verify_delta_is_benign(changes: &[hi_tools::FileChange]) -> bool {
    changes
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
    use hi_tools::{FileChange, FileChangeKind};

    fn cfg(long_horizon: bool) -> crate::AgentConfig {
        crate::AgentConfig {
            subagents: crate::AgentSubagents {
                long_horizon,
                ..crate::AgentSubagents::default()
            },
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
        assert!(post_verify_delta_is_benign(&[change(".hi/memory.md")]));
        assert!(post_verify_delta_is_benign(&[change(".hi/memory.undo.md")]));
        assert!(!post_verify_delta_is_benign(&[change("src/lib.rs")]));
        assert!(!post_verify_delta_is_benign(&[
            change("README.md"),
            change("src/lib.rs"),
        ]));
        // No files changed after verify — keep the pass (revision-only drift).
        assert!(post_verify_delta_is_benign(&[]));
    }

    #[test]
    fn default_step_budget_is_unlimited_and_intent_independent() {
        // Intent classification and horizon must not silently introduce a cap.
        for lh in [false, true] {
            assert_eq!(
                effective_max_steps_for_turn(&cfg(lh)),
                u32::MAX,
                "default is unlimited (long_horizon={lh})"
            );
        }
    }

    #[test]
    fn configured_max_steps_is_honored_and_zero_is_clamped() {
        let mut c = cfg(true);
        c.loop_limits.max_steps = 42;
        assert_eq!(effective_max_steps_for_turn(&c), 42);
        c.loop_limits.max_steps = 0;
        assert_eq!(effective_max_steps_for_turn(&c), 1);
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
