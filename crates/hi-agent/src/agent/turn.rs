//! The main turn loop and its helpers: `run_turn` (user message → model →
//! tool calls → results → repeat, then verify), `finalize_turn`, and the
//! per-turn steering/tool-selection helpers.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use anyhow::Result;
use futures_util::StreamExt;
use hi_ai::{
    ChatRequest, Content, Message, OutputCapError, ProviderErrorKind, RateLimitBucket,
    RateLimitState, RequestProfile, Role, StreamEvent, ToolMode, ToolSpec, estimate_text_tokens,
    provider_error_kind,
};
use hi_tools::{
    PlanStatus, execute_in_runtime, execute_prepared_in_runtime, execute_streaming_in_runtime,
    prepare_mutation_in_with_state,
};

use super::mutation_recovery_turn::MutationRecoveryControl;
use crate::command;
use crate::compaction;
use crate::heuristics::{
    RECOVERY_SAMPLING, StallMode, emit_tool_output, humanize_count, looks_like_continue,
    looks_like_unfinished_step, looks_mutating, mode_blocks_tool, parse_text_tool_calls,
    plan_has_pending_steps, recovery_sampling, recovery_telemetry, respects_deps,
    textcall_id_offset, tool_deps, tool_mode_label,
};
use crate::snapshot::changed_files_between;
use crate::steering::{
    BOOKKEEPING_REPOST_NUDGE, CONCRETE_REVIEW_NUDGE, EvidenceTracker, GAP_SEARCH_OVERCLAIM_NUDGE,
    IMPLEMENTATION_EMPTY_TUI_NUDGE, IMPLEMENTATION_NO_CHANGES_NUDGE,
    IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE, ImplementationIntent, ImplementationTracker,
    MutationRecovery, PLAN_REPOST_NUDGE, POST_TOOL_EMPTY_RESPONSE_NUDGE, READ_AFTER_SEARCH_NUDGE,
    READ_ONLY_SAFE_CONTEXT_WINDOW, REPEAT_NUDGE, REREAD_NUDGE, ReviewIntent, ReviewRepairMode,
    SECURITY_BROAD_SEARCH_NUDGE, SECURITY_SCOPE_NUDGE, SKIPPED_BOOKKEEPING_REPOST_RESULT,
    SKIPPED_PLAN_REPOST_RESULT, SKIPPED_REPEATED_CALL_RESULT, TOOL_PROTOCOL_RETRY_NUDGE,
    TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE, ToolLoopGuardrail, WAIT_POLL_STATIC_NUDGE,
    active_read_only_inspection_cap, answer_says_insufficient_evidence, bash_call_waits,
    bash_command, bash_no_progress_signature, classify_bash_command,
    classify_implementation_intent, classify_read_only_intent, concrete_review_answer_problem,
    deepen_review_nudge, evidence_kind_for_tool, implementation_mentions_tui,
    implementation_missing_validation_nudge, implementation_text_tool_nudge,
    implementation_tool_call_mutates, implementation_tool_call_validates,
    implementation_tool_result_landed_mutation, implementation_tool_result_landed_substantive_edit,
    implementation_turn_prompt, inspected_paths_for_prompt, inspection_signature,
    inspection_sprawl_exhausted, inspection_sprawl_nudge, no_evidence_review_nudge,
    read_only_blocked_tool_result, read_only_blocks_tool, read_only_turn_prompt,
    repair_nudge_with_required_next, should_deepen_review, should_nudge_gap_search_overclaim,
    should_nudge_inspection_sprawl, should_nudge_no_evidence_review,
    should_nudge_read_after_repeated_search, should_nudge_read_after_search_final,
    should_nudge_security_broad_search, should_nudge_security_scope,
    should_reject_review_repair_template, summarize_inspected_evidence_nudge,
};
use crate::transcript::{NudgeKind, repair_invalid_tool_call_arguments_in_messages};
use crate::verify::{
    Snapshot, Verifier, VerifyOutcome, VerifyWorkspace, is_prose_only_path, stage_guidance,
};
use crate::{
    AUTO_KEEP_RECENT, ConfirmationRequest, ConfirmationResult, EffectiveModelRoute,
    FINALIZE_PROMPT, MAX_TOOL_PROTOCOL_RETRIES, PLAN_CONTINUE_NUDGE, ProgressEvent, ReviewStatus,
    SILENT_CONTINUE_NUDGE, TRUNCATED_TOOL_CALL_NUDGE, TRUNCATION_NUDGE, TaskContract, TaskIntent,
    ToolCallEntry, TurnAttribution, TurnOutcome, TurnStatus, TurnStopReason, TurnTelemetry, Ui,
    VerificationMode, VerificationStatus, apply_plan_to_goal, partial_text_tool_call_start,
};

#[allow(clippy::too_many_arguments)]
fn build_turn_telemetry(
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

const PROGRESS_EVENT_LIMIT: usize = 20;
const NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD: u32 = 2;
const NO_PROGRESS_FINAL_ANSWER_NUDGE: &str = "You have not made new progress after repeated tool-use nudges. Stop using tools now and give the best final answer from the evidence already in the conversation. If the task cannot be completed from that evidence, say exactly what is missing.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProgressKind {
    Meaningful,
    Weak,
    None,
}

impl ProgressKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Meaningful => "meaningful",
            Self::Weak => "weak",
            Self::None => "none",
        }
    }
}

#[derive(Clone, Debug)]
struct ToolProgressLabel {
    kind: ProgressKind,
    reason: String,
    signature: Option<String>,
}

impl ToolProgressLabel {
    fn new(kind: ProgressKind, reason: impl Into<String>, signature: Option<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
            signature,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ProgressTracker {
    no_progress_streak: u32,
    no_progress_nudges: u32,
    forced_final_answer_attempts: u32,
    last_progress_reason: String,
    last_stall_reason: String,
    events: Vec<ProgressEvent>,
}

impl ProgressTracker {
    fn push_event(
        &mut self,
        kind: ProgressKind,
        reason: impl Into<String>,
        signature: Option<String>,
    ) {
        self.events.push(ProgressEvent {
            kind: kind.as_str().to_string(),
            reason: reason.into(),
            signature,
        });
        if self.events.len() > PROGRESS_EVENT_LIMIT {
            let excess = self.events.len() - PROGRESS_EVENT_LIMIT;
            self.events.drain(0..excess);
        }
    }

    fn record(&mut self, kind: ProgressKind, reason: impl Into<String>, signature: Option<String>) {
        let reason = reason.into();
        match kind {
            ProgressKind::Meaningful | ProgressKind::Weak => {
                self.no_progress_streak = 0;
                self.last_progress_reason = reason.clone();
            }
            ProgressKind::None => {
                self.no_progress_streak = self.no_progress_streak.saturating_add(1);
                self.last_stall_reason = reason.clone();
            }
        }
        self.push_event(kind, reason, signature);
    }

    fn record_no_progress_nudge(
        &mut self,
        reason: impl Into<String>,
        signature: Option<String>,
    ) -> bool {
        self.no_progress_nudges = self.no_progress_nudges.saturating_add(1);
        self.record(ProgressKind::None, reason, signature);
        self.no_progress_nudges >= NO_PROGRESS_FINAL_ANSWER_NUDGE_THRESHOLD
            && self.forced_final_answer_attempts == 0
    }

    fn record_tool(&mut self, label: &ToolProgressLabel) {
        self.push_event(label.kind, label.reason.clone(), label.signature.clone());
    }

    fn record_round_from_tools(&mut self, labels: &[ToolProgressLabel]) {
        if let Some(label) = labels
            .iter()
            .find(|label| label.kind == ProgressKind::Meaningful)
        {
            self.record(
                ProgressKind::Meaningful,
                label.reason.clone(),
                label.signature.clone(),
            );
        } else if labels.iter().all(|label| label.kind == ProgressKind::None) {
            self.record(ProgressKind::None, "tool round made no progress", None);
        } else if let Some(label) = labels.first() {
            self.record(
                ProgressKind::Weak,
                label.reason.clone(),
                label.signature.clone(),
            );
        }
    }

    fn record_final_answer(&mut self) {
        self.record(ProgressKind::Meaningful, "accepted final answer", None);
    }

    fn record_forced_final_answer_attempt(&mut self) {
        self.forced_final_answer_attempts = self.forced_final_answer_attempts.saturating_add(1);
    }
}

fn effective_max_steps_for_turn(
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

fn task_needs_repository_context(task: &str, contract: &TaskContract) -> bool {
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

fn no_progress_signature_for_calls(calls: &[(String, String, String)]) -> Option<String> {
    calls.iter().find_map(|(_, name, args)| {
        inspection_signature(name, args)
            .or_else(|| bash_no_progress_signature(args).map(|sig| format!("bash:{sig}")))
    })
}

fn forced_final_answer_is_unusable(text: &str, plan_incomplete: bool) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || plan_incomplete || looks_like_unfinished_step(trimmed) {
        return true;
    }
    parse_text_tool_calls(trimmed, 0)
        .iter()
        .any(|content| matches!(content, Content::ToolCall { .. }))
}

fn signature_seen(evidence: &EvidenceTracker, signature: &Option<String>) -> bool {
    signature
        .as_ref()
        .is_some_and(|sig| evidence.seen_signatures.iter().any(|seen| seen == sig))
}

fn background_handle_terminal(name: &str, output: &str) -> bool {
    match name {
        "bash_output" => output
            .lines()
            .next()
            .is_some_and(|status| status.contains(": exited") || status.contains(": killed")),
        "bash_kill" => {
            output.starts_with('[')
                && (output.contains("] killed")
                    || output.contains("] already exited")
                    || output.contains("] already killed"))
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn classify_tool_progress(
    name: &str,
    arguments: &str,
    output: &str,
    error: bool,
    validation_succeeded: bool,
    signature: Option<String>,
    signature_was_seen: bool,
    repeated_idempotent_result: bool,
    tracker_before: &ImplementationTracker,
    plan_changed: bool,
) -> ToolProgressLabel {
    if plan_changed {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "changed plan state", signature);
    }
    if repeated_idempotent_result {
        return ToolProgressLabel::new(
            ProgressKind::None,
            "repeated idempotent tool output",
            signature,
        );
    }
    if name == "bash" && bash_no_progress_signature(arguments).is_some() {
        return ToolProgressLabel::new(
            ProgressKind::None,
            "semantic no-op bash command",
            signature,
        );
    }
    if signature_was_seen {
        let reason = if matches!(name, "bash_output" | "bash_kill")
            && background_handle_terminal(name, output)
        {
            "stale background handle"
        } else {
            "repeated inspection signature"
        };
        return ToolProgressLabel::new(ProgressKind::None, reason, signature);
    }
    if error {
        return ToolProgressLabel::new(ProgressKind::Weak, "tool returned an error", signature);
    }
    if implementation_tool_result_landed_substantive_edit(name, arguments, output) {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "substantive edit", signature);
    }
    if implementation_tool_result_landed_mutation(name, arguments, output) {
        return ToolProgressLabel::new(ProgressKind::Meaningful, "successful mutation", signature);
    }
    if tracker_before.mutation_seen
        && validation_succeeded
        && implementation_tool_call_validates(name, arguments)
    {
        return ToolProgressLabel::new(
            ProgressKind::Meaningful,
            "successful validation after mutation",
            signature,
        );
    }
    if let Some(kind) = evidence_kind_for_tool(name, arguments) {
        let (progress_kind, reason) = match kind {
            crate::steering::EvidenceKind::FileRead => {
                (ProgressKind::Meaningful, "new file evidence")
            }
            crate::steering::EvidenceKind::TargetedSearch => {
                (ProgressKind::Meaningful, "new targeted search evidence")
            }
            crate::steering::EvidenceKind::Listing => (ProgressKind::Weak, "new listing evidence"),
        };
        return ToolProgressLabel::new(progress_kind, reason, signature);
    }
    if name == "bash" {
        let Some(command) = crate::steering::bash_command(arguments) else {
            return ToolProgressLabel::new(ProgressKind::Weak, "bash command completed", signature);
        };
        let kind = classify_bash_command(&command);
        let reason = format!("bash {} command completed", kind.as_str());
        return ToolProgressLabel::new(ProgressKind::Weak, reason, signature);
    }
    ToolProgressLabel::new(ProgressKind::Weak, "tool completed", signature)
}

fn tool_satisfies_validation(output: &hi_tools::ToolOutcome) -> bool {
    output.satisfies_validation()
}

fn tool_entry(
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

fn synthetic_tool_outcome(content: String, status: hi_tools::ToolStatus) -> hi_tools::ToolOutcome {
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

fn effective_model_route(
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
fn combined_review_status(independent: ReviewStatus, skeptic: ReviewStatus) -> ReviewStatus {
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

/// Conservative fallback used only when a checkpoint-backed unified diff is
/// unavailable (for example, the user explicitly allowed mutation without an
/// undo snapshot). It prevents that escape hatch from also bypassing the
/// risk-review threshold. The reviewer still receives `Unavailable` rather
/// than an invented diff; this count is solely a trigger.
fn fallback_review_line_count(root: &std::path::Path, changes: &[hi_tools::FileChange]) -> usize {
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

const MAX_TRANSIENT_ROUTE_RETRIES: u32 = 2;
const TRANSIENT_ROUTE_RETRY_DELAYS: [u64; 2] = [2, 5];
const MAX_TRANSIENT_ROUTE_RETRY_DELAY_SECS: u64 = 30;
const MAX_PROVIDER_OVERLOAD_RETRIES: u32 = 4;
const PROVIDER_OVERLOAD_RETRY_DELAYS: [u64; 4] = [5, 15, 30, 60];
const MAX_PROVIDER_OVERLOAD_RETRY_DELAY_SECS: u64 = 90;
const MIN_OUTPUT_CAP_RETRY_TOKENS: u32 = 512;
const INCOMPLETE_STATUS: &str = "turn stopped incomplete";

#[derive(Default)]
struct ReviewRepairState {
    counts: BTreeMap<String, u32>,
    exhaustion_reason: String,
}

impl ReviewRepairState {
    fn count(&self, mode: ReviewRepairMode) -> u32 {
        self.counts.get(mode.key()).copied().unwrap_or(0)
    }

    fn has_budget(&self, mode: ReviewRepairMode) -> bool {
        self.count(mode) < mode.default_limit()
    }

    fn spend(&mut self, mode: ReviewRepairMode, evidence: &mut EvidenceTracker) -> bool {
        if !self.has_budget(mode) {
            return false;
        }
        let entry = self.counts.entry(mode.key().to_string()).or_insert(0);
        *entry = (*entry).saturating_add(1);
        evidence.quality_repair_nudges = evidence.quality_repair_nudges.saturating_add(1);
        true
    }

    fn note(&mut self, mode: ReviewRepairMode) {
        let entry = self.counts.entry(mode.key().to_string()).or_insert(0);
        *entry = (*entry).saturating_add(1);
    }

    fn exhausted(&mut self, mode: ReviewRepairMode) -> &'static str {
        let reason = mode.exhaustion_key();
        self.exhaustion_reason = reason.to_string();
        reason
    }
}

#[derive(Default)]
struct TurnRetryState {
    request_too_large_retried: bool,
    output_cap_retry_attempted: bool,
    transient_route_retries: u32,
    provider_overload_retries: u32,
    protocol_retries: u32,
    protocol_text_fallbacks: u32,
}

impl TurnRetryState {
    fn record_provider_success(&mut self) {
        self.output_cap_retry_attempted = false;
        self.transient_route_retries = 0;
        self.provider_overload_retries = 0;
    }
}

fn output_cap_retry_tokens(current: u32, cap: OutputCapError) -> Option<u32> {
    let next = if let Some(available) = cap.available_output_tokens {
        available.min(current.saturating_sub(1))
    } else if current > 1024 {
        (current / 2).max(1024)
    } else {
        return None;
    };
    (next >= MIN_OUTPUT_CAP_RETRY_TOKENS && next < current).then_some(next)
}

fn transient_route_retry_delay(retry: u32, err: &anyhow::Error) -> std::time::Duration {
    provider_retry_delay(
        retry,
        err,
        &TRANSIENT_ROUTE_RETRY_DELAYS,
        MAX_TRANSIENT_ROUTE_RETRY_DELAY_SECS,
    )
}

fn provider_overload_retry_delay(retry: u32, err: &anyhow::Error) -> std::time::Duration {
    provider_retry_delay(
        retry,
        err,
        &PROVIDER_OVERLOAD_RETRY_DELAYS,
        MAX_PROVIDER_OVERLOAD_RETRY_DELAY_SECS,
    )
}

fn provider_retry_delay(
    retry: u32,
    err: &anyhow::Error,
    default_delays: &[u64],
    max_delay_secs: u64,
) -> std::time::Duration {
    let default = default_delays
        .get(retry.saturating_sub(1) as usize)
        .copied()
        .unwrap_or(*default_delays.last().unwrap_or(&5));
    let secs = hi_ai::provider_retry_after_seconds(err)
        .unwrap_or(default)
        .min(max_delay_secs);
    if secs == 0 {
        return std::time::Duration::ZERO;
    }
    let jitter_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u64::from(duration.subsec_millis()) % 250)
        .unwrap_or(0);
    std::time::Duration::from_secs(secs) + std::time::Duration::from_millis(jitter_ms)
}

fn delay_label(delay: std::time::Duration) -> String {
    if delay.is_zero() {
        "now".to_string()
    } else {
        format!("{}s", delay.as_secs())
    }
}

fn estimate_tool_schema_tokens(tools: &[ToolSpec]) -> u64 {
    tools
        .iter()
        .map(|tool| {
            hi_ai::estimate_text_tokens(&tool.name)
                + hi_ai::estimate_text_tokens(&tool.description)
                + hi_ai::estimate_text_tokens(&tool.parameters.to_string())
        })
        .sum()
}

impl crate::Agent {
    /// Refresh the active task index at most once per context generation.
    /// Workspace edits advance both the ledger and the generation, while a
    /// transcript-only compaction advances only the generation. That
    /// distinction avoids rescanning the repository after compaction while
    /// still replacing the system message at the new transcript boundary.
    fn refresh_active_task_context(
        &mut self,
        task: &str,
        repository_context_enabled: bool,
        turn_ledger_revision: u64,
        ranked_paths: &mut BTreeSet<String>,
        seen_generation: &mut u64,
        indexed_ledger_revision: &mut u64,
    ) {
        let generation = self.runtime.context_generation();
        if generation == *seen_generation {
            return;
        }

        let (ledger_revision, touched_paths, current_paths) = {
            let ledger = self.runtime.ledger();
            (
                ledger.revision(),
                ledger.touched_paths_since(turn_ledger_revision),
                ledger.changed_paths_since(turn_ledger_revision),
            )
        };
        ranked_paths.extend(touched_paths);
        ranked_paths.extend(current_paths);

        if repository_context_enabled && ledger_revision != *indexed_ledger_revision {
            let paths = ranked_paths.iter().cloned().collect::<Vec<_>>();
            let refreshed = crate::context_index::build_task_context_index(
                self.runtime.root(),
                task,
                &paths,
                &self.config.context_exclusions,
            );
            if refreshed != self.task_context {
                self.task_context = refreshed;
            }
        }

        // `replace_system` changes only slot zero (or creates it for an empty
        // transcript), preserving the alternating user/assistant/tool tail.
        // Do this even for a transcript-only compaction so the new boundary is
        // guaranteed to carry the current task index.
        self.refresh_system_message();
        debug_assert!(self.messages.validate_for_provider().is_ok());
        *seen_generation = generation;
        *indexed_ledger_revision = ledger_revision;
    }

    fn reconcile_error_turn_changes(&mut self, turn_revision: u64) -> Result<()> {
        self.reconcile_workspace_changes()?;
        let changes = self.runtime.ledger().changes_since(turn_revision);
        self.last_changed_files = changes.iter().map(|change| change.path.clone()).collect();
        self.last_file_changes = changes;
        Ok(())
    }

    fn nudge_after_post_tool_empty_response(
        &mut self,
        force_tools_next: &mut bool,
        force_tool_call: bool,
    ) {
        self.messages
            .push_nudge_or_fold(NudgeKind::Continue, POST_TOOL_EMPTY_RESPONSE_NUDGE);
        if force_tool_call {
            *force_tools_next = true;
        }
    }

    async fn ensure_turn_checkpoint(
        &mut self,
        checkpoint_allowed: &mut Option<bool>,
        checkpoint_created: &mut bool,
        ui: &mut dyn Ui,
    ) -> bool {
        if let Some(allowed) = *checkpoint_allowed {
            return allowed;
        }

        // Snapshot lazily, immediately before the first mutating tool. YOLO
        // mode means checkpoint failure never asks for permission; it does not
        // mean skipping a recoverable /undo point when one can be created.
        let reason = match hi_tools::checkpoint::create_detailed_with_state(
            self.runtime.root(),
            self.runtime.state_root(),
        )
        .await
        {
            hi_tools::checkpoint::CreateResult::Created(sha) => {
                let mut next = self.checkpoints.clone();
                next.push(sha);
                if next.len() > crate::MAX_CHECKPOINTS {
                    next.drain(0..next.len() - crate::MAX_CHECKPOINTS);
                }
                if let Some(session) = self.session.as_mut()
                    && let Err(err) = session.record_checkpoints(&next)
                {
                    format!(
                        "checkpoint was created but its reference could not be persisted: {err:#}"
                    )
                } else {
                    self.checkpoints = next;
                    *checkpoint_created = true;
                    *checkpoint_allowed = Some(true);
                    return true;
                }
            }
            hi_tools::checkpoint::CreateResult::Unavailable(reason)
            | hi_tools::checkpoint::CreateResult::Failed(reason) => reason,
        };
        let allowed = self.config.allow_no_checkpoint;
        *checkpoint_allowed = Some(allowed);
        if !allowed {
            ui.status(&format!(
                "mutation skipped: a checkpoint is required but unavailable: {reason}"
            ));
        }
        allowed
    }

    /// Bind the pre-turn checkpoint to the exact post-turn workspace state.
    /// `/undo` will refuse this record if an editor or another process changes
    /// any tracked path after the turn completes.
    async fn seal_turn_checkpoint(&mut self, ui: &mut dyn Ui) -> Result<bool> {
        let Some(target) = self.checkpoints.last().cloned() else {
            return Ok(false);
        };
        match hi_tools::checkpoint::create_detailed_with_state(
            self.runtime.root(),
            self.runtime.state_root(),
        )
        .await
        {
            hi_tools::checkpoint::CreateResult::Created(expected_current) => {
                let sealed = hi_tools::checkpoint::sealed_reference(&target, &expected_current);
                if let Some(last) = self.checkpoints.last_mut() {
                    *last = sealed;
                }
                if let Some(session) = self.session.as_mut() {
                    session.record_checkpoints(&self.checkpoints)?;
                }
                Ok(true)
            }
            hi_tools::checkpoint::CreateResult::Unavailable(reason)
            | hi_tools::checkpoint::CreateResult::Failed(reason) => {
                // An unsealed 0.2 undo record could overwrite edits made after
                // this turn, so always drop it. Strict mode becomes incomplete;
                // YOLO continues silently and exposes the loss in telemetry.
                self.checkpoints.pop();
                if let Some(session) = self.session.as_mut() {
                    session.record_checkpoints(&self.checkpoints)?;
                }
                if !self.config.allow_no_checkpoint {
                    ui.checkpoint_warning(&format!(
                        "⚠ could not seal this turn's undo record: {reason}"
                    ));
                }
                Ok(false)
            }
        }
    }

    async fn ensure_turn_snapshot(
        &mut self,
        turn_snapshot: &mut Option<Snapshot>,
    ) -> Result<Snapshot> {
        if let Some(snapshot) = turn_snapshot.as_ref() {
            return Ok(snapshot.clone());
        }
        let snapshot = self.snapshot_cached().await?;
        *turn_snapshot = Some(snapshot.clone());
        Ok(snapshot)
    }

    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// After the model stops calling tools, an optional verification command is
    /// run; if it fails, its output is fed back and the model iterates, up to
    /// one initial check plus `max_verify_repairs` repair/check cycles.
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<TurnOutcome> {
        let user_prompt_tokens = estimate_text_tokens(input);
        // Reset the per-turn file-read cache. It's invalidated per-key by the
        // edit tools and wholesale after `bash`, but clearing it here restores
        // its documented per-turn contract — so a file changed outside `hi`
        // between turns is re-read fresh, not served from a prior turn's cache.
        self.runtime.clear_read_cache();
        // Reconcile user/external edits before establishing this turn's
        // baseline so they are not attributed to the agent.
        self.runtime.ledger().reconcile()?;
        let turn_ledger_revision = self.runtime.ledger().revision();
        self.active_turn_ledger_revision = Some(turn_ledger_revision);
        self.active_turn_message_start = None;
        let turn_background_baseline = self.runtime.background().ids();
        let expanded_input =
            command::expand_prompt_macro(input).unwrap_or_else(|| input.to_string());
        // Synthetic goal-drive text is only transport. Contracts, context
        // ranking, review, and implementation guards need the real objective
        // and active milestone—especially explicit paths such as plan.md.
        let goal_context = self.goal_continuation_context(&expanded_input);
        let goal_drive_turn = goal_context.is_some();
        let context_task = goal_context.unwrap_or_else(|| expanded_input.clone());
        let structurally_read_only_subagent =
            self.config.is_subagent && self.config.tool_mode == ToolMode::ReadOnly;
        let mut task_contract =
            TaskContract::derive(&context_task, self.config.verification.clone());
        // Capability scope is authoritative for an explore child. Its quoted
        // question may contain mutation verbs ("what should we build next"),
        // but the child is an investigator, not an implementer. Letting prompt
        // wording override that scope activates mutation completion guards that
        // it can never satisfy and previously turned valid reads into denials.
        if structurally_read_only_subagent {
            task_contract.intent = TaskIntent::ReadOnly;
            task_contract.explicit_mutation = false;
        }
        self.refresh_tools_for_task(&context_task, task_contract.intent);
        let repository_context_enabled =
            task_needs_repository_context(&context_task, &task_contract);
        let mut ranked_context_paths = self
            .last_changed_files
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        self.task_context = repository_context_enabled
            .then(|| {
                crate::context_index::build_task_context_index(
                    self.runtime.root(),
                    &context_task,
                    &ranked_context_paths.iter().cloned().collect::<Vec<_>>(),
                    &self.config.context_exclusions,
                )
            })
            .flatten();
        let mut context_generation_seen = self.runtime.context_generation();
        let mut indexed_ledger_revision = self.runtime.ledger().revision();
        self.last_task_contract = Some(task_contract.clone());
        self.refresh_system_message();
        let implementation_candidate = if structurally_read_only_subagent {
            None
        } else if goal_drive_turn && task_contract.intent == TaskIntent::Mutation {
            Some(ImplementationIntent {
                tui: implementation_mentions_tui(&context_task),
            })
        } else {
            classify_implementation_intent(&context_task)
        };
        let read_only_intent = if implementation_candidate.is_some() {
            None
        } else {
            classify_read_only_intent(&context_task)
        };
        let implementation_intent = if read_only_intent.is_none() {
            implementation_candidate
        } else {
            None
        };
        // A turn is *expected* to mutate — and ends "incomplete · stalled"
        // when it changes no files — only for an explicit mutation request
        // ("fix the login bug"), a structured implementation task, or a goal
        // drive turn. The mutation-capable intent that ambiguous wording
        // ("how do users use it?") and tool nouns ("does cargo build build
        // hi-mlx?") default into still advertises mutating tools, but must
        // not brand a correct text-only answer as a stall.
        let expected_mutation = task_contract.explicit_mutation
            || implementation_intent.is_some()
            || (goal_drive_turn && task_contract.intent == TaskIntent::Mutation);
        // Keep the legacy read-only classifier responsible for review prompt
        // shaping. A plain repository question can still have a read-only task
        // contract, and an `explore` child is structurally read-only even when its
        // wording is ambiguous. Apply the sprawl limit to either structural case
        // without imposing the rigid review response format.
        let structural_read_only_inspection = (task_contract.intent == TaskIntent::ReadOnly
            && repository_context_enabled)
            || structurally_read_only_subagent;
        let inspection_sprawl_intent = read_only_intent
            .or_else(|| structural_read_only_inspection.then_some(ReviewIntent::Review));
        let read_only_inspection_cap = inspection_sprawl_intent
            .map(|intent| active_read_only_inspection_cap(&context_task, intent));
        let turn_input = if let Some(intent) = read_only_intent {
            read_only_turn_prompt(&context_task, intent)
        } else if let Some(intent) = implementation_intent {
            implementation_turn_prompt(&context_task, intent)
        } else {
            context_task.clone()
        };
        let input = turn_input.as_str();
        self.reset_last_turn_usage(user_prompt_tokens);
        self.last_turn_outcome = None;
        self.last_effective_route = effective_model_route(&self.config, None);

        // A top-level session the user restricted to ChatOnly/ReadOnly gets a
        // clear early "your mode blocks edits" error when the prompt clearly asks
        // for mutation. This must NOT fire for a subagent: an `explore` child
        // runs ReadOnly as internal capability-scoping (not a user restriction),
        // and its task text naturally contains verbs like "find where X creates
        // Y" — pattern-matching that as a mutating request would abort the child
        // before its first model call and return "(no answer)". The child simply
        // isn't advertised mutating tools, so it's safe to let it run and answer.
        if read_only_intent.is_none()
            && !self.config.is_subagent
            && self.tools_unavailable_for(input)
        {
            self.last_verify = None;
            self.last_changed_files.clear();
            self.last_file_changes.clear();
            self.last_compat_fallbacks.clear();
            self.last_turn_telemetry = TurnTelemetry::default();
            let preserve_plan = (goal_drive_turn || looks_like_continue(&context_task))
                && plan_has_pending_steps(&self.last_plan);
            if !preserve_plan && !self.last_plan.is_empty() {
                self.last_plan.clear();
                if let Some(session) = self.session.as_mut() {
                    session.clear_plan()?;
                }
                ui.plan(&[]);
            }
            self.messages.strip_trailing_nudges();
            self.persisted = self.persisted.min(self.messages.len());
            self.persist()?;
            ui.turn_error(
                "tools",
                &format!(
                    "tool mode {} blocks file edits and shell commands",
                    tool_mode_label(self.config.tool_mode)
                ),
                "",
            );
            let outcome = TurnOutcome {
                status: TurnStatus::Blocked,
                verification: VerificationStatus::NotApplicable,
                review: ReviewStatus::NotRequired,
                stop_reason: TurnStopReason::ToolModeDenied,
                changed_files: Vec::new(),
                verified_workspace_revision: None,
                effective_route: effective_model_route(&self.config, None),
            };
            self.last_turn_outcome = Some(outcome.clone());
            self.active_turn_ledger_revision = None;
            self.active_turn_message_start = None;
            return Ok(outcome);
        }
        let mut turn_checkpoint_allowed = None;
        let mut turn_checkpoint_created = false;

        // If the context window is filling up, reclaim room before adding more,
        // so the session keeps going instead of overflowing. Two tiers: a free,
        // deterministic elision of old tool output first; then, only if still
        // heavy, the configured summarizing strategy. Best-effort — a failed
        // model call just leaves the (already elided) history as-is.
        //
        // The outer trigger uses the provider-reported `context_used` (the last
        // request's occupancy — the most accurate signal, and only meaningful
        // once a real request has happened, so a fresh session isn't
        // over-eagerly compacted). Tier 2 below gates on a local token estimate
        // instead, because `context_used` is stale by then.
        if self.config.auto_compact
            && let Some(window) = self.config.context_window
            && window > 0
            && self.context_used * 100 >= u64::from(window) * self.config.auto_compact_percent
        {
            ui.status(&format!(
                "context ~{}% full — compacting to free room",
                self.context_used * 100 / u64::from(window)
            ));
            // Tier 1: deterministic, no model call. Only old turns are eligible.
            if let Some(split) =
                compaction::recent_split(self.messages.as_slice(), AUTO_KEEP_RECENT)
                && compaction::elide_tool_outputs(self.messages.mutate_slice(), split) > 0
            {
                self.runtime.invalidate_context_after_compaction();
            }
            // Tier 2: only if still heavy. `context_used` reflects the
            // pre-elision request and is now stale, so gate on a local estimate.
            let target = u64::from(window) * self.config.compact_target_percent / 100;
            if compaction::estimate_tokens(self.messages.as_slice()) > target {
                let _ = self.compact(ui).await;
            }
            self.context_used = 0;
        }

        self.messages.strip_trailing_nudges();
        self.persisted = self.persisted.min(self.messages.len());
        let mut turn_start = self.messages.len();
        self.active_turn_message_start = Some(turn_start);
        self.messages.push_user_or_fold(input);
        self.last_verify = None;
        self.last_changed_files.clear();
        self.last_file_changes.clear();
        self.last_compat_fallbacks.clear();
        self.last_turn_telemetry.verification_executions.clear();
        // Preserve only an unfinished plan that the user explicitly continues.
        // Clearing must also be emitted: the TUI owns a pinned copy and cannot
        // infer that the agent cleared its internal state.
        let preserve_plan = (goal_drive_turn || looks_like_continue(&context_task))
            && plan_has_pending_steps(&self.last_plan);
        if !preserve_plan && !self.last_plan.is_empty() {
            self.last_plan.clear();
            if let Some(session) = self.session.as_mut() {
                session.clear_plan()?;
            }
            ui.plan(&[]);
        }
        let mut compat_fallbacks = Vec::new();
        let mut effective_fallback_route: Option<String> = None;

        let resolved_verify_stages = self
            .config
            .verification
            .resolved_stages(self.runtime.root());
        let verify_rounds = self.config.max_verify_repairs.saturating_add(1);
        let mut verifier = if matches!(&self.config.verification, VerificationMode::Auto) {
            Verifier::automatic(resolved_verify_stages, verify_rounds)
        } else {
            Verifier::new(resolved_verify_stages, verify_rounds)
        };
        let max_steps = effective_max_steps_for_turn(
            &self.config,
            task_contract.intent,
            read_only_intent,
            implementation_intent,
        );
        let max_parallel_tools = self.config.max_parallel_tools.max(1);
        let mut steps = 0u32;
        let mut empty_retries = 0u32;
        // Consecutive output-limit continuations. This is a stall budget, so it
        // resets after any non-truncated model response/tool progress.
        let mut truncation_retries = 0u32;
        // Cumulative truncation nudges for telemetry/UI summaries. Unlike the
        // consecutive budget above, this should not reset mid-turn.
        let mut truncation_total_retries = 0u32;
        let mut silent_continues = 0u32;
        let mut continue_total_nudges = 0u32;
        let mut repeat_nudges = 0u32;
        let mut progress_tracker = ProgressTracker::default();
        // Set after a silent-continue nudge: force the *next* round to call a
        // tool (`tool_choice: required`) instead of letting the model narrate
        // again or return an empty completion. Some models (e.g. weaker
        // OpenAI-compat coders) intermittently emit text-only or empty responses
        // when asked to continue; backing the "use your tools; act, don't
        // narrate" nudge with a hard tool-choice makes them actually act. Stays
        // set across empty-retries and re-nudges until the model emits a tool
        // call, then clears (see the made_tool_call path). Only takes effect when
        // tools are otherwise freely available (config tool_mode Auto).
        let mut force_tools_next = false;
        // Bounded discovery narrows the advertised catalog until the model
        // records a plan or makes the requested edit.
        let mut mutation_recovery = MutationRecovery::default();
        // A model-authored plan is only a proposal until deterministic
        // verification passes for the settled workspace revision. Keeping it
        // turn-local prevents failed, unverified, cancelled, or infrastructure-
        // error turns from leaking goal progress into the live session.
        let mut plan_updated_goal = false;
        let mut proposed_goal: Option<crate::Goal> = None;
        // The goal as it stood at turn start — so the skeptic gate can review
        // against the sub-goal that was active *before* the turn (update_plan may
        // have marked it done mid-turn) and, on an objection, revert the turn's
        // goal progress.
        let goal_before = self.structured_goal.clone();
        // Scheduler parallelism counters: how many calls ran this turn, the
        // largest concurrent ready-batch, and how many ran serially (bash or a
        // lone ready call). Flushed into telemetry so the dep-aware scheduler's
        // concurrency is measurable, not shipped on faith.
        let mut sched_tool_calls = 0u32;
        let mut sched_max_concurrent = 0u32;
        let mut sched_serial_runs = 0u32;
        // Per-tool-call timeline: each call's name, path, duration, and error
        // status, flushed into telemetry so `--report` can diagnose where time
        // went and which calls failed.
        let mut tool_timeline: Vec<ToolCallEntry> = Vec::new();
        let mut advertised_tool_names = BTreeSet::new();
        let mut tool_schema_tokens = 0_u64;
        let mut evidence = EvidenceTracker::default();
        let mut review_repair = ReviewRepairState::default();
        let mut independent_review_status = ReviewStatus::NotRequired;
        let mut independent_review_repairs = 0_u32;
        let mut verification_infrastructure_error = false;
        let mut verification_unstable = false;
        // A pass is bound to both the ledger event number and the full content
        // digest observed immediately after the verifier. Later workspace
        // activity must never inherit that pass.
        let mut verified_at: Option<(u64, String)> = None;
        // Whether the model or deterministic preflight has run a tool this
        // turn (kept for finalization gating — a plain Q&A turn doesn't need a
        // recap).
        let mut made_tool_call = false;
        let mut implementation_tracker = ImplementationTracker::default();
        let mut empty_tui_needs_project = false;
        if let Some(intent) = read_only_intent
            && self.config.read_only_preflight
            && !matches!(self.config.tool_mode, ToolMode::ChatOnly)
        {
            let preflight = self
                .run_read_only_preflight(
                    intent,
                    read_only_inspection_cap.unwrap_or_else(|| evidence.inspection_attempt_count()),
                    ui,
                    &mut evidence,
                    &mut tool_timeline,
                )
                .await;
            if preflight.executed > 0 {
                made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight.executed);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight.serial_runs);
                sched_max_concurrent = sched_max_concurrent.max(preflight.max_concurrent_batch);
            }
        }
        if implementation_intent.is_some() && !matches!(self.config.tool_mode, ToolMode::ChatOnly) {
            let preflight_calls = self
                .run_implementation_preflight(ui, &mut implementation_tracker, &mut tool_timeline)
                .await;
            if preflight_calls > 0 {
                made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight_calls);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight_calls);
                sched_max_concurrent = sched_max_concurrent.max(1);
            }
            empty_tui_needs_project = implementation_intent.is_some_and(|intent| intent.tui)
                && implementation_tracker.preferred_validation.is_none();
        }
        // Signature (name, arguments) of the previous round's tool calls, to
        // spot a model re-issuing the exact same call and looping on it.
        let mut prev_call_sig: Option<Vec<(String, String)>> = None;
        // Whether the previous executed round added no new evidence (every call
        // was a read-only inspection already seen). Used by the no-new-evidence
        // cycle guard to fire only on the *second* consecutive wasted round,
        // preserving a single legitimate re-inspection after new evidence.
        let mut prev_added_no_evidence = false;
        let mut retry_state = TurnRetryState::default();
        let mut request_max_tokens_override: Option<u32> = None;
        let mut text_tool_fallback_next = false;
        let mut force_text_answer_next = false;
        let mut force_no_progress_final_answer_next = false;
        // After a bookkeeping-repost nudge, withhold the bookkeeping tools
        // (`update_plan`, `record_decision`) from the next request's tool
        // list. A bookkeeping-fixated model (observed live) keeps re-posting
        // meta-work through every nudge — and when only `update_plan` was
        // withheld it slid to repeating `record_decision` instead. Clear
        // feedback alone doesn't break the loop; removing the whole family
        // for one round forces a tool that does real work.
        let mut suppress_bookkeeping_tools_next = false;
        // Consecutive rounds skipped by the repeat guard, driving recovery
        // sampling: a model re-emitting the identical call each round is stuck
        // in a token-level loop that only hotter sampling breaks. Resets as
        // soon as the model issues a different round, so later rounds run at
        // the configured sampling again (unlike the cumulative
        // `repeat_nudges` budget, which never resets within a turn).
        let mut repeat_sampling_rounds = 0u32;
        let mut tool_guardrail = ToolLoopGuardrail::default();
        // Whether the turn ended because the model kept re-issuing the exact
        // same tool call through the whole repeat-nudge budget (drives the
        // stalled telemetry and skips the finalization recap).
        let mut stalled_repeating = false;
        // Whether the turn ended without enough evidence for a read-only review.
        let mut stalled_unfinished = false;
        // Whether the turn was cut short by the per-turn step cap, so the
        // finalization recap is skipped (the work may be incomplete).
        let mut ended_at_cap = false;
        // Attributions parsed from the most recent verify failure — captured
        // here so they survive to turn end and can be flushed into telemetry.
        let mut last_verify_attributions: Vec<hi_tools::Attribution> = Vec::new();
        // Snapshot the turn baseline lazily. Read-only/chat turns should not
        // walk the whole workspace just to prove nothing changed; the baseline
        // is captured before the first actual mutation, or before verification
        // when verify stages are configured.
        let mut turn_snapshot: Option<Snapshot> = None;
        // Snapshot from the most recent verify check. Reused at turn end to
        // avoid a second full tree walk when verify already took one.

        if empty_tui_needs_project {
            force_tools_next = true;
            self.messages
                .push_nudge(NudgeKind::Continue, IMPLEMENTATION_EMPTY_TUI_NUDGE);
        }

        'turn: loop {
            // Inner loop: model + tools until the model stops calling tools, or
            // the per-turn step cap is hit.
            let hit_cap = loop {
                if steps >= max_steps {
                    break true;
                }
                steps += 1;

                // Mid-turn steering: inject any messages the user typed while
                // the turn was running, as genuine user messages, before the
                // next model round. This is a safe transcript boundary — the
                // prior round's tool calls are all resolved — so the folding
                // nudge push keeps provider alternation valid. The model
                // decides how to weigh them; we add no deferral directive.
                let interjected = self.interjections.drain();
                if !interjected.is_empty() {
                    for message in &interjected {
                        self.messages.push_nudge_or_fold(
                            NudgeKind::Interjection,
                            format!(
                                "The user sent this message while you were working — take it into account now:\n{message}"
                            ),
                        );
                    }
                    ui.status(&format!(
                        "✉ received {} message(s) from you mid-turn — factoring them in",
                        interjected.len()
                    ));
                }

                // After a content-less/garbled round, resample hotter and with
                // nucleus + frequency penalty on the retry to break out of the
                // low-entropy attractor that produced it (cf. minion's recovery
                // sampling). Bounded, and only while consecutively stalling —
                // `empty_retries` resets on real output, so a normal round runs at
                // the configured sampling. Toggleable via HI_RECOVERY_SAMPLING for
                // A/B-ing on the eval harness.
                let sampling_retries = empty_retries
                    .max(retry_state.protocol_retries)
                    .max(repeat_sampling_rounds);
                let (sampling_mode, sampling_budget) = if repeat_sampling_rounds > 0
                    && repeat_sampling_rounds >= empty_retries
                    && repeat_sampling_rounds >= retry_state.protocol_retries
                {
                    // The model is deterministically re-emitting the same tool
                    // call round after round (observed live: four byte-identical
                    // `update_plan` calls despite nudges and withheld tools).
                    // Hotter sampling + a frequency penalty is what actually
                    // breaks a token-level loop; nudge text alone doesn't.
                    (StallMode::Repeat, self.config.max_repeat_nudges)
                } else if retry_state.protocol_retries > empty_retries {
                    (StallMode::Empty, MAX_TOOL_PROTOCOL_RETRIES)
                } else {
                    (StallMode::Empty, self.config.max_empty_retries)
                };
                let (temperature, top_p, frequency_penalty) = recovery_sampling(
                    sampling_retries,
                    self.config.temperature,
                    *RECOVERY_SAMPLING,
                );

                // Telemetry for the recovery-sampling A/B: emit a concise debug
                // line only when sampling is actually being changed (recovery on
                // and this is a retry), so ordinary runs stay quiet.
                if let Some(line) = recovery_telemetry(
                    sampling_mode,
                    sampling_retries,
                    sampling_budget,
                    temperature,
                    top_p,
                    frequency_penalty,
                    *RECOVERY_SAMPLING,
                ) {
                    ui.nudge(&line);
                }

                let context_safety_window = read_only_intent
                    .is_some()
                    .then_some(READ_ONLY_SAFE_CONTEXT_WINDOW);
                self.elide_in_turn_context_if_needed(ui, context_safety_window);

                self.refresh_active_task_context(
                    &context_task,
                    repository_context_enabled,
                    turn_ledger_revision,
                    &mut ranked_context_paths,
                    &mut context_generation_seen,
                    &mut indexed_ledger_revision,
                );

                self.messages.repair_invalid_tool_call_arguments();

                // Debug-mode invariant check: the transcript we're about to send
                // must be provider-safe (every tool_use answered, no consecutive
                // user messages). Cheap in release builds; in debug it catches
                // the orphan-tool_use class of bug at the source.
                debug_assert!(
                    self.messages.validate_for_provider().is_ok(),
                    "transcript invariant violated before provider send"
                );

                let request_text_tool_fallback = text_tool_fallback_next;
                text_tool_fallback_next = false;
                let request_text_answer = force_text_answer_next;
                force_text_answer_next = false;
                let request_no_progress_final_answer = force_no_progress_final_answer_next;
                if request_no_progress_final_answer {
                    progress_tracker.record_forced_final_answer_attempt();
                }
                force_no_progress_final_answer_next = false;

                // After a continue-nudge, force this round to call a tool rather
                // than narrate again or come back empty. Only when tools are
                // freely available (Auto): never override an intentional
                // ChatOnly/ReadOnly restriction, and Required already forces.
                let tool_mode = if request_text_tool_fallback
                    || request_text_answer
                    || request_no_progress_final_answer
                {
                    ToolMode::ChatOnly
                } else if force_tools_next && self.config.tool_mode == ToolMode::Auto {
                    ToolMode::Required
                } else {
                    self.config.tool_mode
                };
                let tool_availability_mode = if request_text_tool_fallback
                    || request_text_answer
                    || request_no_progress_final_answer
                {
                    ToolMode::ChatOnly
                } else if read_only_intent.is_some()
                    && !matches!(self.config.tool_mode, ToolMode::ChatOnly)
                {
                    ToolMode::ReadOnly
                } else {
                    self.config.tool_mode
                };
                let requested_request_max_tokens =
                    request_max_tokens_override.unwrap_or(self.config.max_tokens);
                let mut request_tools = self.request_tools_for(tool_availability_mode);
                if suppress_bookkeeping_tools_next {
                    suppress_bookkeeping_tools_next = false;
                    // Only withhold when other tools remain — an empty tool
                    // list with tool_choice=required would be a provider error.
                    if request_tools
                        .iter()
                        .any(|tool| !hi_tools::is_coordination(&tool.name))
                    {
                        request_tools = request_tools
                            .iter()
                            .filter(|tool| !hi_tools::is_coordination(&tool.name))
                            .cloned()
                            .collect();
                    }
                }
                advertised_tool_names.extend(request_tools.iter().map(|tool| tool.name.clone()));
                let request_tool_schema_tokens = estimate_tool_schema_tokens(&request_tools);
                tool_schema_tokens = tool_schema_tokens.max(request_tool_schema_tokens);
                let context_preflight = match self.ensure_request_fits_context(
                    input,
                    turn_start,
                    requested_request_max_tokens,
                    request_tool_schema_tokens,
                    context_safety_window,
                    ui,
                ) {
                    Ok(context_preflight) => context_preflight,
                    Err(err) => {
                        self.reconcile_error_turn_changes(turn_ledger_revision)?;
                        self.truncate_messages(turn_start);
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            max_steps,
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            &progress_tracker,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            verifier.executions(),
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
                            &review_repair,
                        );
                        let _ = self.persist();
                        let (kind, guidance) = crate::ui::classify_error(&err);
                        ui.turn_error(kind, &err.to_string(), guidance);
                        self.last_effective_route = effective_model_route(
                            &self.config,
                            effective_fallback_route.as_deref(),
                        );
                        return Err(err);
                    }
                };
                if context_preflight.dropped_prior_context {
                    turn_start = self.messages.len().saturating_sub(1);
                }
                // Context fitting may itself compact or elide the transcript.
                // Consume that generation before constructing the request.
                self.refresh_active_task_context(
                    &context_task,
                    repository_context_enabled,
                    turn_ledger_revision,
                    &mut ranked_context_paths,
                    &mut context_generation_seen,
                    &mut indexed_ledger_revision,
                );
                let request_max_tokens = context_preflight.max_tokens;
                if request_max_tokens != requested_request_max_tokens {
                    request_max_tokens_override = Some(request_max_tokens);
                }
                let request = ChatRequest {
                    model: self.config.model.clone(),
                    messages: self.messages.arc(),
                    tools: request_tools,
                    max_tokens: request_max_tokens,
                    temperature,
                    top_p,
                    frequency_penalty,
                    thinking_budget: self.config.thinking_budget,
                    reasoning_effort: self.config.reasoning_effort,
                    profile: RequestProfile {
                        compat: self.config.compat,
                        tool_mode,
                        stream_usage: None,
                    },
                };

                let buffer_read_only_review_text =
                    read_only_intent.is_some() || implementation_intent.is_some();
                let mut buffered_assistant_text = String::new();
                let mut streamed_assistant_text = false;
                let mut sink = |event: StreamEvent| match event {
                    StreamEvent::Text(text) => {
                        if buffer_read_only_review_text {
                            buffered_assistant_text.push_str(&text);
                        } else {
                            streamed_assistant_text = true;
                            ui.assistant_text(&text);
                        }
                    }
                    StreamEvent::Reasoning(text) => ui.assistant_reasoning(&text),
                    StreamEvent::Status(text) => {
                        if let Some(fallback) = text.strip_prefix("compat: ") {
                            compat_fallbacks.push(fallback.to_string());
                        }
                        if let Some(route) = text.rsplit_once("falling back to ").map(|(_, r)| r) {
                            effective_fallback_route = Some(route.trim().to_string());
                        }
                        ui.status(&text);
                    }
                };
                let mut completion = match self.provider.stream(request, &mut sink).await {
                    Ok(completion) => {
                        retry_state.record_provider_success();
                        completion
                    }
                    Err(err)
                        if !retry_state.output_cap_retry_attempted
                            && hi_ai::provider_output_cap_error(&err)
                                .and_then(|cap| output_cap_retry_tokens(request_max_tokens, cap))
                                .is_some() =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.output_cap_retry_attempted = true;
                        let new_max = hi_ai::provider_output_cap_error(&err)
                            .and_then(|cap| output_cap_retry_tokens(request_max_tokens, cap))
                            .expect("guard checked retry tokens");
                        request_max_tokens_override = Some(new_max);
                        ui.nudge(&format!(
                            "provider rejected the output budget; retrying this turn with max_tokens={new_max}"
                        ));
                        continue;
                    }
                    Err(err)
                        if retry_state.provider_overload_retries
                            < MAX_PROVIDER_OVERLOAD_RETRIES
                            && hi_ai::provider_error_is_temporary_overload(&err) =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.provider_overload_retries += 1;
                        let retry = retry_state.provider_overload_retries;
                        let delay = provider_overload_retry_delay(retry, &err);
                        ui.nudge(&format!(
                            "request did not complete; retrying {} ({retry}/{MAX_PROVIDER_OVERLOAD_RETRIES})",
                            delay_label(delay)
                        ));
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                        continue;
                    }
                    Err(err)
                        if retry_state.transient_route_retries < MAX_TRANSIENT_ROUTE_RETRIES
                            && hi_ai::provider_route_error_is_retryable(&err) =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.transient_route_retries += 1;
                        let retry = retry_state.transient_route_retries;
                        let delay = transient_route_retry_delay(retry, &err);
                        ui.nudge(&format!(
                            "request did not complete; retrying {} ({retry}/{MAX_TRANSIENT_ROUTE_RETRIES})",
                            delay_label(delay)
                        ));
                        if !delay.is_zero() {
                            tokio::time::sleep(delay).await;
                        }
                        continue;
                    }
                    Err(err)
                        if provider_error_kind(&err)
                            == Some(ProviderErrorKind::RequestTooLarge) =>
                    {
                        let mut context_drop_persistence_failed = false;
                        if !retry_state.request_too_large_retried {
                            match self.retry_after_request_too_large(input, turn_start, ui) {
                                Ok(true) => {
                                    retry_state.request_too_large_retried = true;
                                    turn_start = self.messages.len().saturating_sub(1);
                                    continue;
                                }
                                Ok(false) => {}
                                Err(persist_err) => {
                                    ui.status(&format!(
                                        "couldn't persist dropped-context retry state: {persist_err}"
                                    ));
                                    context_drop_persistence_failed = true;
                                }
                            }
                        }
                        self.truncate_messages(turn_start);
                        if context_drop_persistence_failed {
                            ui.status(
                                "request exceeds the provider limit, and prior context could not be \
                                 safely dropped because the session boundary was not persisted; fix \
                                 session storage or start a fresh/cleared session, then retry",
                            );
                        } else {
                            ui.status(
                                "request still exceeds the provider limit with prior context removed; \
                                 shorten the prompt or attached input, then retry",
                            );
                        }
                        self.add_error_usage(&err);
                        self.reconcile_error_turn_changes(turn_ledger_revision)?;
                        self.emit_usage(ui);
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            max_steps,
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            &progress_tracker,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            verifier.executions(),
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
                            &review_repair,
                        );
                        let _ = self.persist();
                        let (kind, guidance) = crate::ui::classify_error(&err);
                        ui.turn_error(kind, &err.to_string(), guidance);
                        self.last_effective_route = effective_model_route(
                            &self.config,
                            effective_fallback_route.as_deref(),
                        );
                        return Err(err);
                    }
                    Err(err)
                        if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                            && retry_state.protocol_retries < MAX_TOOL_PROTOCOL_RETRIES =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.protocol_retries += 1;
                        let protocol_retries = retry_state.protocol_retries;
                        if implementation_intent.is_some() || made_tool_call {
                            force_tools_next = true;
                        }
                        ui.nudge(&format!(
                            "⚠ the model emitted an invalid tool turn — retrying with tool-format guidance ({protocol_retries}/{MAX_TOOL_PROTOCOL_RETRIES})"
                        ));
                        if self
                            .messages
                            .as_slice()
                            .last()
                            .is_some_and(|message| message.role == Role::User)
                        {
                            self.messages.push_user_or_fold(TOOL_PROTOCOL_RETRY_NUDGE);
                        } else {
                            self.messages
                                .push_nudge(NudgeKind::Continue, TOOL_PROTOCOL_RETRY_NUDGE);
                        }
                        continue;
                    }
                    Err(err)
                        if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                            && implementation_intent.is_some()
                            && retry_state.protocol_text_fallbacks < 1 =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        retry_state.protocol_text_fallbacks += 1;
                        text_tool_fallback_next = true;
                        force_tools_next = false;
                        ui.status(
                            "structured tool calls kept failing; falling back to plain-text tool-call parsing",
                        );
                        if self
                            .messages
                            .as_slice()
                            .last()
                            .is_some_and(|message| message.role == Role::User)
                        {
                            self.messages
                                .push_user_or_fold(TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE);
                        } else {
                            self.messages
                                .push_nudge(NudgeKind::Continue, TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE);
                        }
                        continue;
                    }
                    // A transient generation flake — a malformed/garbled stream or
                    // an empty completion. Treat it like a content-less response:
                    // flush, then silently re-run with hotter recovery sampling (a
                    // fresh request, with its own transport retries) up to the same
                    // budget, instead of failing the turn. Terminal errors (auth,
                    // rate limits, ...) fall through to the abort below. Invalid tool turns
                    // use the protocol-specific nudge path above.
                    Err(err)
                        if empty_retries < self.config.max_empty_retries
                            && matches!(
                                provider_error_kind(&err),
                                Some(
                                    ProviderErrorKind::MalformedStream
                                        | ProviderErrorKind::EmptyCompletion
                                )
                            ) =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        self.emit_usage(ui);
                        empty_retries += 1;
                        if made_tool_call {
                            self.nudge_after_post_tool_empty_response(
                                &mut force_tools_next,
                                implementation_intent.is_some(),
                            );
                        }
                        ui.nudge(&format!(
                            "⚠ the model's response didn't come through cleanly — \
                             retrying ({empty_retries}/{})",
                            self.config.max_empty_retries
                        ));
                        continue;
                    }
                    Err(err) => {
                        self.add_error_usage(&err);
                        self.reconcile_error_turn_changes(turn_ledger_revision)?;
                        self.emit_usage(ui);
                        if self.last_changed_files.is_empty()
                            && let Some(turn_snapshot) = turn_snapshot.as_ref()
                        {
                            self.messages.strip_trailing_nudges();
                            if let Ok(end_snapshot) = self.snapshot_cached().await {
                                self.last_changed_files =
                                    changed_files_between(turn_snapshot, &end_snapshot);
                            }
                        }
                        // With no model tool call, any concurrent workspace
                        // change was external to this failed attempt. Preserve
                        // it in the report, but never retain the failed user
                        // prompt or retry guidance in conversation history.
                        if !made_tool_call {
                            self.truncate_messages(turn_start);
                        }
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            max_steps,
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            &progress_tracker,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            verifier.executions(),
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
                            &review_repair,
                        );
                        let _ = self.persist();
                        let (kind, guidance) = crate::ui::classify_error(&err);
                        ui.turn_error(kind, &err.to_string(), guidance);
                        self.last_effective_route = effective_model_route(
                            &self.config,
                            effective_fallback_route.as_deref(),
                        );
                        return Err(err);
                    }
                };
                if !buffer_read_only_review_text {
                    ui.assistant_end();
                }

                self.add_usage(completion.usage);
                // Let the frontend show the running total climb mid-turn.
                self.emit_usage(ui);

                // Truncation recovery: the model hit the output token cap
                // (`stop_reason: "length"` / `"max_tokens"`) mid-generation.
                // The response was cut off, not finished — record what it
                // produced and nudge it to continue from the cutoff, instead
                // of treating the truncation as a natural stop (which would
                // end the turn on a half-finished output and leave the model
                // "picking up where it stalled" on the next prompt). Bounded
                // by a *dedicated* truncation budget (separate from
                // `empty_retries`) so a big task that legitimately hits the
                // cap several times can still finish without the user typing
                // "continue".
                let truncated = matches!(
                    completion.stop_reason.as_deref(),
                    Some("length" | "max_tokens")
                );
                if truncated && truncation_retries < self.config.max_truncation_retries {
                    truncation_retries += 1;
                    truncation_total_retries += 1;
                    ui.nudge(&format!(
                        "⚠ the model hit the output token limit — continuing ({truncation_retries}/{})",
                        self.config.max_truncation_retries
                    ));
                    // Clean text-embedded tool-call JSON (local models) from the
                    // truncated content before recording. Complete tool calls are
                    // extracted and stripped; partial JSON (cut off mid-generation)
                    // stays as text so the model can continue from the cutoff.
                    // Structured ToolCall blocks are stripped: a truncated tool call
                    // has partial/malformed arguments and was never executed, so it
                    // has no matching tool_result. Leaving it in would create an
                    // orphan tool_use that providers reject on the next request.
                    let partial_tool_call =
                        self.clean_text_tool_calls_from_content(&mut completion.content);
                    let truncated_text = completion
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            Content::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let active_tool_work = read_only_intent.is_none()
                        && (implementation_intent.is_some()
                            || made_tool_call
                            || implementation_tracker.mutation_seen
                            || plan_has_pending_steps(&self.last_plan)
                            || looks_like_unfinished_step(&truncated_text));
                    if (partial_tool_call || active_tool_work)
                        && self.config.tool_mode == ToolMode::Auto
                    {
                        force_tools_next = true;
                    }
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    self.messages.push_nudge(
                        NudgeKind::Truncation,
                        if partial_tool_call || active_tool_work {
                            TRUNCATED_TOOL_CALL_NUDGE
                        } else {
                            TRUNCATION_NUDGE
                        },
                    );
                    continue;
                }
                // Truncation budget exhausted: the model kept hitting the output
                // token cap through the whole retry budget. Record the truncated
                // output (stripping partial tool calls, as above) and warn the
                // user — the task may be incomplete. Don't silently end the turn
                // on a half-finished output without surfacing what happened.
                if truncated {
                    self.clean_text_tool_calls_from_content(&mut completion.content);
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    stalled_unfinished = true;
                    ui.nudge(&format!(
                        "⚠ the model hit the output token limit {max} times — the task may be \
                         incomplete. /retry, or send 'continue'.",
                        max = self.config.max_truncation_retries,
                    ));
                    break false;
                }

                let calls: Vec<(String, String, String)> =
                    if request_text_answer || request_no_progress_final_answer {
                        Vec::new()
                    } else {
                        completion
                            .tool_calls()
                            .into_iter()
                            .map(|c| {
                                (
                                    c.id.to_string(),
                                    c.name.to_string(),
                                    c.arguments.to_string(),
                                )
                            })
                            .collect()
                    };

                // Fallback for local models (Ollama, llama.cpp, etc.) that emit
                // tool calls as text — raw JSON like {"name":"bash","arguments":…}
                // — instead of using the structured `tool_calls` API field. When
                // the API returned no structured calls, scan the assistant text
                // for tool-call JSON and promote any matches to real ToolCall
                // blocks so they actually execute. The raw JSON is stripped from
                // the recorded text so history stays clean.
                let calls = if calls.is_empty()
                    && !request_text_answer
                    && !request_no_progress_final_answer
                {
                    let full_text: String = completion
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            Content::Text(t) => Some(t.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let parsed =
                        parse_text_tool_calls(&full_text, textcall_id_offset(&self.messages));
                    if parsed.iter().any(|c| matches!(c, Content::ToolCall { .. })) {
                        // Replace text blocks with the interleaved content
                        // (prose segments + ToolCall blocks in emission order),
                        // preserving any Thinking blocks from the original.
                        let mut new_content = Vec::new();
                        let mut parsed_iter = parsed.into_iter().peekable();
                        for c in completion.content.iter() {
                            match c {
                                Content::Text(_) => {
                                    // Drain the parsed content that corresponds to
                                    // this text block (all of it — the original had
                                    // one Text block with the full raw text).
                                    for p in parsed_iter.by_ref() {
                                        new_content.push(p);
                                    }
                                }
                                Content::Thinking { .. } => new_content.push(c.clone()),
                                _ => {}
                            }
                        }
                        // If the original had no Text block (shouldn't happen for
                        // the local-model path, but be safe), drain remaining.
                        for p in parsed_iter {
                            new_content.push(p);
                        }
                        completion.content = new_content;
                        completion
                            .tool_calls()
                            .into_iter()
                            .map(|c| {
                                (
                                    c.id.to_string(),
                                    c.name.to_string(),
                                    c.arguments.to_string(),
                                )
                            })
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    calls
                };

                // Repetition guard: the model re-issued the exact same tool
                // calls (same names, same arguments, same order) as the previous
                // round. Re-running most tools can only reproduce the same
                // output, so don't execute — nudge the model to act on the output
                // it already has. `bash_output` is intentionally excluded from
                // this exact-match shortcut because a live background process is
                // time-dependent and can emit new output between identical polls;
                // completed/missing/pruned handles are caught below by the
                // stale-background no-new-evidence path. Bounded; past the
                // budget the turn ends with an honest "stuck repeating" notice
                // rather than looping until `max_steps`.
                let call_sig: Vec<(String, String)> = calls
                    .iter()
                    .map(|(_, name, args)| (name.clone(), args.clone()))
                    .collect();
                let has_background_output_poll = calls
                    .iter()
                    .any(|(_, name, _)| name.as_str() == "bash_output");
                let has_background_handle_call = calls
                    .iter()
                    .any(|(_, name, _)| matches!(name.as_str(), "bash_output" | "bash_kill"));
                let has_no_progress_bash = calls.iter().any(|(_, name, args)| {
                    name == "bash" && bash_no_progress_signature(args).is_some()
                });
                // A bash command that deliberately waits before sampling state
                // ("sleep 300 && du -sh models/") is time-dependent the same
                // way a `bash_output` poll is: re-running it verbatim is how
                // the model watches a slow external process (a download, a
                // long build, a warming server), and each run can return new
                // output. Exempt such rounds from the signature-based repeat
                // guards; the result-hash guard below still catches the
                // static case (the same poll returning byte-identical output),
                // so a wait loop stays bounded without punishing legitimate
                // progress-watching.
                let has_wait_poll_bash = calls
                    .iter()
                    .any(|(_, name, args)| name == "bash" && bash_call_waits(args));
                let exact_repeat = !calls.is_empty()
                    && !has_background_output_poll
                    && !has_wait_poll_bash
                    && prev_call_sig.as_ref() == Some(&call_sig);
                // No-new-evidence cycle guard: a round whose every call is a
                // read-only inspection (read/list/grep/glob) or stale background
                // handle operation already performed earlier this turn. This
                // catches multi-step cycles like
                // A→B→C→A→B→C — including grep/list cycles, not just re-reads —
                // that evade the exact-match check because each round differs
                // from the one right before it. On large workspaces such a cycle
                // can otherwise loop until `max_steps` without ever re-issuing an
                // identical round. `EvidenceTracker::round_adds_evidence` keys on
                // a stable per-inspection signature (read path/page, list path,
                // grep pattern/glob/path/context, stale background handle id), so
                // any re-inspection is caught regardless of cycle length or tool
                // mix. Shares the same
                // `repeat_nudges` budget as the exact-match guard so it stays
                // bounded.
                //
                // Fires only on the *second* consecutive no-new-evidence round
                // (`prev_added_no_evidence`): a single re-inspection right after
                // new evidence is allowed through (e.g. re-reading a file once a
                // broader search has surfaced something to re-examine, or paging
                // further into a file). Once the turn has made a successful
                // mutation, this guard is advisory only: after the nudge budget
                // is spent, execute the inspection rather than hard-stalling a
                // long implementation harness in the middle of a later plan step.
                let no_new_evidence = !calls.is_empty() && !evidence.round_adds_evidence(&calls);
                let stale_background_handle_call = no_new_evidence && has_background_handle_call;
                // A wait-poll round re-runs a seen inspection signature by
                // design, so it must not trip the no-new-evidence cycle guard
                // either — its staleness is judged by output, below.
                let is_repeat = exact_repeat
                    || (no_new_evidence
                        && !has_wait_poll_bash
                        && (prev_added_no_evidence || stale_background_handle_call));
                let no_new_after_mutation = is_repeat
                    && no_new_evidence
                    && implementation_tracker.mutation_seen
                    && !stale_background_handle_call;
                let repeat_budget_available = repeat_nudges < self.config.max_repeat_nudges;
                let should_skip_for_repeat =
                    is_repeat && (!no_new_after_mutation || repeat_budget_available);
                if should_skip_for_repeat {
                    // We deliberately do NOT execute the repeated tool calls,
                    // but the calls stay in the transcript, each paired with a
                    // synthetic result that says why it was skipped. Stripping
                    // them (as this path once did) left the model's turn as a
                    // bare placeholder with no result for the call it just
                    // made — weak models concluded the tool layer was broken
                    // ("my tool calls aren't producing visible output") and
                    // gave up instead of correcting course. Pairing every
                    // skipped `tool_use` with a `tool_result` also keeps the
                    // transcript in the shape providers require.
                    let all_plan_reposts = calls.iter().all(|(_, name, _)| name == "update_plan");
                    let all_bookkeeping_reposts = calls
                        .iter()
                        .all(|(_, name, _)| hi_tools::is_coordination(name));
                    let skip_results: Vec<(String, String)> = calls
                        .iter()
                        .map(|(id, name, _)| {
                            let note = if name == "update_plan" {
                                SKIPPED_PLAN_REPOST_RESULT
                            } else if hi_tools::is_coordination(name) {
                                SKIPPED_BOOKKEEPING_REPOST_RESULT
                            } else {
                                SKIPPED_REPEATED_CALL_RESULT
                            };
                            (id.clone(), note.to_string())
                        })
                        .collect();
                    self.messages.push_assistant_with_results(
                        std::mem::take(&mut completion.content),
                        skip_results,
                    );
                    if repeat_budget_available {
                        repeat_nudges += 1;
                        repeat_sampling_rounds += 1;
                        stalled_repeating = true;
                        let stall_reason = if all_plan_reposts {
                            "unchanged plan repost"
                        } else if all_bookkeeping_reposts {
                            "repeated bookkeeping call"
                        } else if stale_background_handle_call {
                            "stale background handle"
                        } else if has_no_progress_bash {
                            "semantic no-op bash command"
                        } else if no_new_evidence {
                            "repeated inspection signature"
                        } else {
                            "skipped repeated calls"
                        };
                        let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
                            stall_reason,
                            no_progress_signature_for_calls(&calls),
                        ) && !no_new_after_mutation
                            && implementation_intent.is_none();
                        let nudge = if all_bookkeeping_reposts {
                            if all_plan_reposts {
                                ui.nudge(&format!(
                                    "the model re-posted an unchanged plan — withholding \
                                     bookkeeping tools for a round and nudging it to execute \
                                     the next step ({repeat_nudges}/{})",
                                    self.config.max_repeat_nudges
                                ));
                            } else {
                                ui.nudge(&format!(
                                    "the model repeated bookkeeping calls without real work — \
                                     withholding bookkeeping tools for a round \
                                     ({repeat_nudges}/{})",
                                    self.config.max_repeat_nudges
                                ));
                            }
                            suppress_bookkeeping_tools_next = true;
                            force_tools_next = true;
                            if all_plan_reposts {
                                PLAN_REPOST_NUDGE.to_string()
                            } else {
                                BOOKKEEPING_REPOST_NUDGE.to_string()
                            }
                        } else if stale_background_handle_call {
                            if has_background_output_poll {
                                ui.nudge(&format!(
                                    "the model kept polling stale background process handles — \
                                     nudging it to stop polling them ({repeat_nudges}/{})",
                                    self.config.max_repeat_nudges
                                ));
                                "The background process handle you just polled is completed, missing, or pruned, so polling it again cannot produce new output. Do not call bash_output for that handle again. Continue from the available output, restart the command if you still need it, or finish with the current result.".to_string()
                            } else {
                                ui.nudge(&format!(
                                    "the model kept using stale background process handles — \
                                     nudging it to stop using them ({repeat_nudges}/{})",
                                    self.config.max_repeat_nudges
                                ));
                                "The background process handle you just used is already killed, already exited, missing, or pruned, so calling bash_kill for it again cannot change anything. Do not call bash_kill for that handle again. Continue from the available output, restart the command if you still need it, or finish with the current result.".to_string()
                            }
                        } else if should_nudge_read_after_repeated_search(
                            read_only_intent,
                            &evidence,
                        ) {
                            ui.nudge(&format!(
                                        "the model re-ran the same search — nudging it to read a matching file ({repeat_nudges}/{})",
                                        self.config.max_repeat_nudges
                                    ));
                            READ_AFTER_SEARCH_NUDGE.to_string()
                        } else if implementation_intent.is_some()
                            && no_new_evidence
                            && (evidence.saw_read || evidence.saw_search)
                        {
                            // Concrete, actionable nudge for implementation tasks:
                            // name the inspected files and the next plan step (if
                            // any) so the model has a specific action to take
                            // instead of a generic "start editing." A strong model
                            // responds to one concrete nudge; a weak one won't
                            // respond to any number, so the budget stays tight (2).
                            // Only fires for no-new-evidence cycles (re-reading
                            // already-inspected files); exact repeats of non-read
                            // tools (e.g. re-running a bash command) fall through
                            // to the generic REPEAT_NUDGE below, which says "don't
                            // re-run that command" — the right message for that case.
                            ui.nudge(&format!(
                                "the model re-read files it already inspected — their contents are \
                                 already above; nudging it to act on them ({repeat_nudges}/{})",
                                self.config.max_repeat_nudges
                            ));
                            let paths = inspected_paths_for_prompt(&evidence);
                            let plan_step = self
                                .last_plan
                                .iter()
                                .find(|s| {
                                    s.status == PlanStatus::Pending
                                        || s.status == PlanStatus::Active
                                })
                                .map(|s| s.title.as_str());
                            if let Some(step) = plan_step {
                                format!(
                                    "You already inspected these files: {paths}. Their contents are in the conversation above — do not re-read them. \
Your plan's next step is: \"{step}\". Execute it now with write/edit/multi_edit/apply_patch. \
Do not read more files first — you have enough context. Act on the next plan step immediately."
                                )
                            } else {
                                format!(
                                    "You already inspected these files: {paths}. Their contents are in the conversation above — do not re-read them. \
You have enough context to make progress. Edit one of the inspected files now with write/edit/multi_edit/apply_patch. \
If the task is already complete, stop and give your final recap."
                                )
                            }
                        } else if has_no_progress_bash {
                            ui.nudge(&format!(
                                "the model kept running no-op shell commands — nudging it to finish without more bash calls ({repeat_nudges}/{})",
                                self.config.max_repeat_nudges
                            ));
                            "The bash command you just called only says stop/quit/done or otherwise does no work. Do not call bash for that. If the task is complete, finish with a text answer; otherwise use a tool that inspects or changes the workspace.".to_string()
                        } else if no_new_evidence && !exact_repeat {
                            ui.nudge(&format!(
                                "the model re-read files it already inspected — their contents are \
                                 already above; nudging it to act on them ({repeat_nudges}/{})",
                                self.config.max_repeat_nudges
                            ));
                            REREAD_NUDGE.to_string()
                        } else {
                            ui.nudge(&format!(
                                "the model re-ran the same command — its output is already above; \
                                     nudging it to act on it ({repeat_nudges}/{})",
                                self.config.max_repeat_nudges
                            ));
                            REPEAT_NUDGE.to_string()
                        };
                        let nudge = if force_final_after_nudge {
                            force_no_progress_final_answer_next = true;
                            force_tools_next = false;
                            format!("{nudge}\n\n{NO_PROGRESS_FINAL_ANSWER_NUDGE}")
                        } else {
                            nudge
                        };
                        self.messages.push_nudge(NudgeKind::Repeat, nudge);
                        // Keep prev_call_sig as-is so a further repeat is still
                        // detected against the same signature.
                        continue;
                    }
                    if stale_background_handle_call {
                        ui.status(
                            "background process handles were completed, missing, or pruned (or already killed) and the model kept using them — the task may be incomplete. /retry, or send 'continue'.",
                        );
                        break false;
                    }
                    if has_no_progress_bash {
                        stalled_unfinished = true;
                        ui.nudge("model repeated no-op shell commands; stopping incomplete");
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if read_only_intent.is_some() && evidence.saw_search && !evidence.saw_read {
                        stalled_unfinished = true;
                        ui.nudge(
                            "review repeated the same search without reading files; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if let Some(intent) = read_only_intent
                        && (evidence.saw_read || evidence.saw_search)
                    {
                        stalled_unfinished = true;
                        ui.nudge(
                            "review repeated the same command after inspection; stopping incomplete",
                        );
                        let _ = (intent, &evidence);
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if implementation_intent.is_some()
                        && (evidence.saw_read || evidence.saw_search)
                        && !implementation_tracker.mutation_seen
                    {
                        // The model inspected the workspace but kept
                        // repeating non-mutating calls through the repeat
                        // budget. Route this through the implementation
                        // repair budget instead of the generic repeat failure:
                        // a chat-only final answer is not useful until a
                        // mutation actually exists.
                        if implementation_tracker.no_change_nudges < 2 {
                            implementation_tracker.no_change_nudges += 1;
                            evidence.quality_repair_nudges =
                                evidence.quality_repair_nudges.saturating_add(1);
                            let use_text_fallback = implementation_tracker.no_change_nudges >= 2;
                            force_tools_next = !use_text_fallback;
                            text_tool_fallback_next = use_text_fallback;
                            ui.nudge(
                                "implementation kept repeating without editing; nudging the model to edit or scaffold",
                            );
                            let nudge = if use_text_fallback {
                                implementation_text_tool_nudge(IMPLEMENTATION_NO_CHANGES_NUDGE)
                            } else {
                                IMPLEMENTATION_NO_CHANGES_NUDGE.to_string()
                            };
                            self.messages.push_nudge(NudgeKind::Continue, nudge);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "implementation kept repeating without editing; no file changes were made",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    ui.status(
                        "⚠ the model kept re-running the same command without acting on the \
                         result — the task may be incomplete. /retry, or send 'continue'.",
                    );
                    break false;
                }
                // A different set of calls (or none) this round — the model moved
                // on, so clear any pending repeat-stall state. A wait-poll
                // round is not counted as the first wasted round of a cycle:
                // waiting on external state is progress-neutral, not evidence
                // of a loop.
                stalled_repeating = false;
                repeat_sampling_rounds = 0;
                prev_call_sig = Some(call_sig);
                prev_added_no_evidence = no_new_evidence && !has_wait_poll_bash;

                // Inspection-sprawl guard: a read-only review turn that keeps
                // reading *distinct* files (each a new inspection signature, so
                // the repeat/cycle guard above never fires) without ever
                // producing findings. Once enough evidence has accumulated,
                // nudge the model to answer; if it keeps sprawling past the
                // budget, stop incomplete rather than fabricate an answer. This is
                // the only guard that catches the "read 100 files, never
                // answer" failure mode — all review-quality guards fire only
                // on a final text answer, which never comes while the model
                // keeps issuing tool calls.
                if inspection_sprawl_exhausted(
                    inspection_sprawl_intent,
                    &evidence,
                    &calls,
                    read_only_inspection_cap,
                ) {
                    stalled_unfinished = true;
                    ui.nudge(
                            "review kept inspecting new files without producing findings; stopping incomplete",
                        );
                    ui.status(INCOMPLETE_STATUS);
                    break false;
                }
                if should_nudge_inspection_sprawl(
                    inspection_sprawl_intent,
                    &evidence,
                    &calls,
                    read_only_inspection_cap,
                ) {
                    evidence.inspection_sprawl_nudges =
                        evidence.inspection_sprawl_nudges.saturating_add(1);
                    force_text_answer_next = true;
                    let cap = read_only_inspection_cap
                        .unwrap_or_else(|| evidence.inspection_attempt_count());
                    ui.nudge(&format!(
                        "review inspected {} files/searches without answering; nudging it to produce findings",
                        evidence.inspection_attempt_count()
                    ));
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        inspection_sprawl_nudge(cap, evidence.inspection_attempt_count()),
                    );
                    continue;
                }

                // This round's assistant text, joined and captured before the
                // content is moved into history. Used both to detect a content-less
                // response (a reasoning model can return only reasoning tokens or
                // whitespace) and to spot an announced-but-unperformed next step.
                let assistant_text: String = completion
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_text = !assistant_text.trim().is_empty();

                if request_no_progress_final_answer {
                    let unusable = forced_final_answer_is_unusable(
                        &assistant_text,
                        plan_has_pending_steps(&self.last_plan),
                    );
                    if has_text && (buffer_read_only_review_text || !streamed_assistant_text) {
                        let text_to_emit = if buffered_assistant_text.is_empty() {
                            assistant_text.as_str()
                        } else {
                            buffered_assistant_text.as_str()
                        };
                        ui.assistant_text(text_to_emit);
                        ui.assistant_end();
                    }
                    if unusable {
                        self.messages
                            .push_assistant_text_only(std::mem::take(&mut completion.content));
                        stalled_unfinished = true;
                        progress_tracker.record(
                            ProgressKind::None,
                            "forced final-answer attempt was unusable",
                            None,
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    self.messages
                        .push_assistant(std::mem::take(&mut completion.content));
                    progress_tracker.record_final_answer();
                    break false;
                }

                // Auto-recover from a content-less response — no tool calls and no
                // text, i.e. a flaky provider returning only reasoning or an empty
                // message. Silently re-run a few times before giving up, each
                // retry resampling hotter (see the temperature bump above). The
                // dead round isn't recorded, so each retry re-runs with the
                // original context.
                if calls.is_empty() && !has_text {
                    if empty_retries < self.config.max_empty_retries {
                        empty_retries += 1;
                        if made_tool_call {
                            self.nudge_after_post_tool_empty_response(
                                &mut force_tools_next,
                                implementation_intent.is_some(),
                            );
                        }
                        ui.status(&format!(
                            "⚠ the model returned no response — retrying ({empty_retries}/{})",
                            self.config.max_empty_retries
                        ));
                        continue;
                    }
                    ui.status("⚠ the model returned no response after retrying — try /retry.");
                    break false;
                }
                // Real output this round — clear the retry counter so the
                // temperature bump is transient: a later, unrelated stall gets
                // its own budget rather than inheriting this one's elevation.
                empty_retries = 0;
                retry_state.protocol_retries = 0;
                truncation_retries = 0;

                if calls.is_empty() {
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
                    let looks_unfinished = looks_like_unfinished_step(&assistant_text);
                    let plan_incomplete = plan_has_pending_steps(&self.last_plan);
                    if let Some(intent) = read_only_intent
                        && (looks_unfinished || plan_incomplete)
                    {
                        if evidence.inspection_sprawl_nudges > 0 {
                            if evidence.quality_repair_nudges < 3 {
                                evidence.quality_repair_nudges += 1;
                                continue_total_nudges += 1;
                                force_text_answer_next = true;
                                ui.nudge(
                                    "review tried to continue inspecting after the sprawl limit; forcing a bounded answer from existing evidence",
                                );
                                self.messages
                                    .push_assistant(std::mem::take(&mut completion.content));
                                self.messages.push_nudge(
                                    NudgeKind::Continue,
                                    summarize_inspected_evidence_nudge(intent, &evidence),
                                );
                                continue;
                            }

                            stalled_unfinished = true;
                            let _ = intent;
                            ui.status(INCOMPLETE_STATUS);
                            break false;
                        }

                        if silent_continues < self.config.max_silent_continues {
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            silent_continues += 1;
                            continue_total_nudges += 1;
                            force_tools_next = true;
                            let nudge = if plan_incomplete && !looks_unfinished {
                                PLAN_CONTINUE_NUDGE
                            } else {
                                SILENT_CONTINUE_NUDGE
                            };
                            self.messages.push_nudge(NudgeKind::Continue, nudge);
                            continue;
                        }
                    }
                    if implementation_intent.is_some() && !implementation_tracker.mutation_seen {
                        if implementation_tracker.no_change_nudges < 2 {
                            implementation_tracker.no_change_nudges += 1;
                            evidence.quality_repair_nudges =
                                evidence.quality_repair_nudges.saturating_add(1);
                            let use_text_fallback = implementation_tracker.no_change_nudges >= 2;
                            force_tools_next = !use_text_fallback;
                            text_tool_fallback_next = use_text_fallback;
                            ui.nudge(
	                                "implementation answer had no file changes; nudging the model to edit or scaffold",
	                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            let nudge = if use_text_fallback {
                                implementation_text_tool_nudge(IMPLEMENTATION_NO_CHANGES_NUDGE)
                            } else {
                                IMPLEMENTATION_NO_CHANGES_NUDGE.to_string()
                            };
                            self.messages.push_nudge(NudgeKind::Continue, nudge);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge("implementation still had no file changes after repair");
                        ui.status(INCOMPLETE_STATUS);
                        break false;
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
                            force_tools_next = !use_text_fallback;
                            text_tool_fallback_next = use_text_fallback;
                            ui.nudge(
	                                "implementation only scaffolded setup files; nudging the model to edit source files",
	                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            let nudge = if use_text_fallback {
                                implementation_text_tool_nudge(IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE)
                            } else {
                                IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE.to_string()
                            };
                            self.messages.push_nudge(NudgeKind::Continue, nudge);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "implementation still only had scaffold/setup changes after repair",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
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
                            force_tools_next = !use_text_fallback;
                            text_tool_fallback_next = use_text_fallback;
                            ui.nudge(
	                                "implementation changed files without validation; nudging the model to run tests or build",
	                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            let validation_nudge =
                                implementation_missing_validation_nudge(&implementation_tracker);
                            let nudge = if use_text_fallback {
                                implementation_text_tool_nudge(&validation_nudge)
                            } else {
                                validation_nudge
                            };
                            self.messages.push_nudge(NudgeKind::Continue, nudge);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge("implementation still lacked validation after repair");
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if should_nudge_no_evidence_review(read_only_intent, &evidence, &assistant_text)
                    {
                        let mode = ReviewRepairMode::NoEvidence;
                        if review_repair.spend(mode, &mut evidence) {
                            force_tools_next = true;
                            ui.nudge(
                                "review answer had no inspected evidence; nudging the model to inspect before answering",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(
                                    mode,
                                    no_evidence_review_nudge(
                                        read_only_intent.expect("checked above"),
                                    ),
                                ),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.nudge(
                            "review still had no inspected evidence after repair; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if let Some(intent) = read_only_intent
                        && evidence.saw_read
                        && answer_says_insufficient_evidence(&assistant_text)
                    {
                        if matches!(intent, ReviewIntent::Security)
                            && evidence.saw_search
                            && !evidence.security_search_complete()
                            && review_repair
                                .spend(ReviewRepairMode::SecurityBroadSearch, &mut evidence)
                        {
                            force_tools_next = true;
                            ui.nudge(
                                "security review gave a generic evidence disclaimer before searching all required pattern families; nudging the model to broaden the search",
                            );
                            self.messages
                                .push_assistant_repair_note(ReviewRepairMode::SecurityBroadSearch);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(
                                    ReviewRepairMode::SecurityBroadSearch,
                                    SECURITY_BROAD_SEARCH_NUDGE,
                                ),
                            );
                            continue;
                        }
                        let mode = ReviewRepairMode::InspectedDisclaimer;
                        let chat_mode = ReviewRepairMode::InspectedDisclaimerChatAttempt;
                        let has_disclaimer_budget = review_repair.has_budget(mode);
                        let has_chat_attempt_budget = review_repair.has_budget(chat_mode);
                        if has_disclaimer_budget || has_chat_attempt_budget {
                            if has_disclaimer_budget {
                                review_repair.spend(mode, &mut evidence);
                            } else {
                                evidence.quality_repair_nudges =
                                    evidence.quality_repair_nudges.saturating_add(1);
                            }
                            review_repair.note(chat_mode);
                            force_text_answer_next = true;
                            force_tools_next = false;
                            ui.nudge(
                                "review gave a generic evidence disclaimer after inspection; nudging the model to answer from inspected files",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(
                                    mode,
                                    summarize_inspected_evidence_nudge(intent, &evidence),
                                ),
                            );
                            continue;
                        }
                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.status(
                            "review kept returning a generic evidence disclaimer after inspection; stopping incomplete",
                        );
                        let _ = (intent, &evidence);
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    // (`saw_read` is implied here: the previous disjunct already
                    // catches search-without-read, so boolean-equivalently drop it.)
                    let needs_evidence_depth_repair = evidence.listing_only()
                        || (evidence.saw_search && !evidence.saw_read)
                        || (matches!(read_only_intent, Some(ReviewIntent::Security))
                            && evidence.saw_search
                            && !evidence.security_search_complete());
                    if !needs_evidence_depth_repair
                        && should_reject_review_repair_template(read_only_intent, &assistant_text)
                    {
                        if let Some(intent) = read_only_intent
                            && review_repair.spend(ReviewRepairMode::GenericTemplate, &mut evidence)
                        {
                            let mode = ReviewRepairMode::GenericTemplate;
                            let has_inspected_evidence = evidence.saw_read || evidence.saw_search;
                            force_text_answer_next = has_inspected_evidence;
                            force_tools_next = !has_inspected_evidence;
                            ui.nudge(
                                "review answer was a generic repair template; nudging the model to produce a concrete bounded review",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            let nudge = if has_inspected_evidence {
                                summarize_inspected_evidence_nudge(intent, &evidence)
                            } else {
                                deepen_review_nudge(intent).to_string()
                            };
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(mode, nudge),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(ReviewRepairMode::GenericTemplate);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.status("review answer stayed generic after repair; stopping incomplete");
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if should_deepen_review(read_only_intent, &evidence, &assistant_text) {
                        let mode = ReviewRepairMode::ListingOnly;
                        if review_repair.spend(mode, &mut evidence) {
                            force_tools_next = true;
                            ui.nudge(
                                "review evidence was only a listing; nudging the model to inspect files or search results",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(
                                    mode,
                                    deepen_review_nudge(read_only_intent.expect("checked above")),
                                ),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.nudge(
                            "review still had only listing evidence after repair; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if should_nudge_read_after_search_final(
                        read_only_intent,
                        &evidence,
                        &assistant_text,
                    ) {
                        let mode = ReviewRepairMode::ReadAfterSearch;
                        if review_repair.spend(mode, &mut evidence) {
                            force_tools_next = true;
                            ui.nudge(
                                "review had targeted search but no file reads; nudging the model to read matching files",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(mode, READ_AFTER_SEARCH_NUDGE),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.nudge(
                            "review still had targeted search but no file reads after repair; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if should_nudge_security_broad_search(
                        read_only_intent,
                        &evidence,
                        &assistant_text,
                    ) {
                        let mode = ReviewRepairMode::SecurityBroadSearch;
                        if review_repair.spend(mode, &mut evidence) {
                            force_tools_next = true;
                            ui.nudge(
                                "security review missed required pattern families; nudging the model to broaden the search",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(mode, SECURITY_BROAD_SEARCH_NUDGE),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.nudge(
                            "security review still missed required pattern families after repair; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if should_nudge_security_scope(read_only_intent, &evidence, &assistant_text) {
                        let mode = ReviewRepairMode::SecurityScope;
                        if review_repair.spend(mode, &mut evidence) {
                            ui.status(
                                "security answer overclaimed repo-wide safety; nudging the model to bound findings to evidence",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(mode, SECURITY_SCOPE_NUDGE),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.status(
                            "security answer still overclaimed after repair; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if should_nudge_gap_search_overclaim(
                        read_only_intent,
                        &evidence,
                        &assistant_text,
                    ) {
                        let mode = ReviewRepairMode::GapSearchOverclaim;
                        if review_repair.spend(mode, &mut evidence) {
                            ui.nudge(
                                "gap answer contradicted search matches; nudging the model to bound claims to inspected evidence",
                            );
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(mode, GAP_SEARCH_OVERCLAIM_NUDGE),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.nudge(
                            "gap answer still overclaimed after search matches; stopping incomplete",
                        );
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if let Some(problem) =
                        concrete_review_answer_problem(read_only_intent, &evidence, &assistant_text)
                    {
                        let mode = ReviewRepairMode::ConcreteAnswer;
                        if review_repair.spend(mode, &mut evidence) {
                            force_text_answer_next = true;
                            ui.nudge(problem.status());
                            self.messages.push_assistant_repair_note(mode);
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                repair_nudge_with_required_next(mode, CONCRETE_REVIEW_NUDGE),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        let reason = review_repair.exhausted(mode);
                        progress_tracker.record(ProgressKind::None, reason, None);
                        ui.nudge(problem.exhausted_status());
                        ui.status(INCOMPLETE_STATUS);
                        break false;
                    }
                    if buffer_read_only_review_text {
                        let text_to_emit = if buffered_assistant_text.is_empty() {
                            assistant_text.as_str()
                        } else {
                            buffered_assistant_text.as_str()
                        };
                        ui.assistant_text(text_to_emit);
                        ui.assistant_end();
                    }
                    self.messages
                        .push_assistant(std::mem::take(&mut completion.content));
                    if (looks_unfinished || plan_incomplete)
                        && silent_continues < self.config.max_silent_continues
                    {
                        silent_continues += 1;
                        continue_total_nudges += 1;
                        // Force the next round to actually call a tool, so the
                        // nudge can't be answered with yet another narration or an
                        // empty completion.
                        force_tools_next = true;
                        // Use a plan-aware nudge when the plan is incomplete, so
                        // the model knows to continue the next step rather than
                        // just "continue from where you stopped".
                        let nudge = if plan_incomplete && !looks_unfinished {
                            PLAN_CONTINUE_NUDGE
                        } else {
                            SILENT_CONTINUE_NUDGE
                        };
                        self.messages.push_nudge(NudgeKind::Continue, nudge);
                        continue;
                    }
                    // If we exhausted the silent-continue budget (at least one
                    // continue was attempted) on a turn that looked unfinished,
                    // let the user know. Don't warn when max_silent_continues
                    // is 0 (no continue was attempted — the feature is off).
                    if (looks_unfinished || plan_incomplete) && silent_continues > 0 {
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
                    break false;
                }
                // The model requested tool calls — it's actively working.
                made_tool_call = true;
                // Real progress this round, so clear the silent-continue counter:
                // the budget bounds *consecutive* narrate-without-acting stalls,
                // not their total across the turn. A long, productive turn that
                // reads many files but occasionally narrates a step without the
                // tool call (a quirk of some models) recovers each time via the
                // nudge — without this reset the counter would creep up across
                // the whole turn and kill the turn mid-progress on the Nth stall
                // even though the model acted between every one. Mirrors the
                // `empty_retries = 0` reset above (a later stall gets its own
                // budget rather than inheriting an earlier one's).
                silent_continues = 0;
                // The model acted, so drop the forced-tool-choice we may have set
                // after a nudge — the next round is free to narrate or finish.
                force_tools_next = false;
                let hash_guard_applies = calls.iter().all(|(_, name, args)| {
                    matches!(name.as_str(), "read" | "list" | "grep" | "glob")
                        || (name == "bash" && bash_call_waits(args))
                });
                let mut hashable_idempotent_results = 0usize;
                let mut repeated_idempotent_results = 0usize;
                let mut tool_progress_labels: Vec<ToolProgressLabel> = Vec::new();
                let mut plan_changed_this_batch = false;
                // Infer within-batch dependencies (a read of a file a mutating
                // call earlier in the batch targeted must observe that mutation;
                // mutating calls serialize). The scheduler below runs ready
                // calls concurrently respecting this graph, so independent reads
                // can overlap with an independent later write — while a read
                // whose path matches an earlier write waits for it.
                let deps = tool_deps(&calls);
                // Execute via a ready-queue scheduler over the dep graph. A call
                // is ready when all its deps are complete. Ready non-bash calls
                // run concurrently; bash runs alone this round (its line-by-line
                // UI streaming can't be reordered, and `tool_deps` already makes
                // it depend on all prior calls via the unknown-path fallback, so
                // it's never ready alongside a dependent). Results are collected
                // and recorded together via `push_assistant_with_results` so the
                // transcript never carries an orphan tool_use; results are
                // ordered by emission index so the transcript reads in model
                // order. UI streaming and snapshot invalidation still happen
                // during execution.
                let mut results: Vec<Option<(String, String)>> = vec![None; calls.len()];
                let mut completed = vec![false; calls.len()];
                let mut completion_order: Vec<usize> = Vec::with_capacity(calls.len());
                let mut scheduler_forced_skip = false;
                // Pre-pass: resolve calls blocked by read-only intent up front.
                // They produce instant synthetic error results and mutate
                // nothing, so completing them out of dep order is safe.
                // (`explore`/`delegate`/`record_decision` used to run here too,
                // but they *do* have deps that matter — running a subagent
                // before an earlier `write` in the same batch handed it a stale
                // tree — so they now dispatch inside the dep-aware scheduler
                // loop below.)
                for (i, (id, name, arguments)) in calls.iter().enumerate() {
                    // Block calls forbidden by the review intent (read-only
                    // prompt) OR the session tool_mode. The tool_mode check is
                    // essential for the text-promoted tool-call path above: a
                    // local model can emit `{"name":"write",…}` as prose, which
                    // bypasses tool *advertisement*, so without an execution-time
                    // guard a ChatOnly/ReadOnly session — every `explore` subagent
                    // included — could still run a mutating `write`/`bash`.
                    let blocked = if read_only_blocks_tool(read_only_intent, name) {
                        Some(read_only_blocked_tool_result(name))
                    } else {
                        mode_blocks_tool(self.config.tool_mode, name)
                    };
                    if let Some(content) = blocked {
                        ui.tool_call(name, arguments);
                        let mut output =
                            synthetic_tool_outcome(content.clone(), hi_tools::ToolStatus::Denied);
                        output.effects.mutation_attempted =
                            implementation_tool_call_mutates(name, arguments);
                        emit_tool_output(&mut *ui, name, &output);
                        let progress_label = ToolProgressLabel::new(
                            ProgressKind::Weak,
                            "tool denied by active mode",
                            inspection_signature(name, arguments),
                        );
                        progress_tracker.record_tool(&progress_label);
                        tool_progress_labels.push(progress_label.clone());
                        tool_timeline.push(tool_entry(
                            name.clone(),
                            hi_tools::target_path(name, arguments).unwrap_or_default(),
                            0,
                            &output,
                            &progress_label,
                        ));
                        results[i] = Some((id.clone(), content));
                        completed[i] = true;
                        completion_order.push(i);
                    }
                }
                let mut done = completion_order.len();
                if done > 0 {
                    sched_tool_calls = sched_tool_calls.saturating_add(done as u32);
                    sched_serial_runs = sched_serial_runs.saturating_add(done as u32);
                    sched_max_concurrent = sched_max_concurrent.max(1);
                }
                // Proactive per-edit checks: kicked off in the background as
                // mutating calls complete, awaited after the batch so any
                // syntax/lint error surfaces during the turn (before turn-end
                // verify) while the edit is still the model's focus. Each entry
                // is (path, join handle of the check).
                let mut pending_checks: Vec<(String, tokio::task::JoinHandle<(bool, String)>)> =
                    Vec::new();
                while done < calls.len() {
                    // Check the interrupt flag: if the user pressed Esc to skip
                    // the current tool, mark all uncompleted calls as interrupted
                    // and break out of the execution loop so the model gets a
                    // "interrupted by user" result and can adapt.
                    if self
                        .interrupt
                        .swap(false, std::sync::atomic::Ordering::Relaxed)
                    {
                        let mut interrupted = 0_u32;
                        for i in 0..calls.len() {
                            if !completed[i] {
                                let (id, name, arguments) = &calls[i];
                                ui.tool_call(name, arguments);
                                let msg = "Tool call interrupted by user.".to_string();
                                let mut output = synthetic_tool_outcome(
                                    msg.clone(),
                                    hi_tools::ToolStatus::Cancelled,
                                );
                                output.effects.mutation_attempted =
                                    implementation_tool_call_mutates(name, arguments);
                                emit_tool_output(&mut *ui, name, &output);
                                let progress_label = ToolProgressLabel::new(
                                    ProgressKind::None,
                                    "tool interrupted by user",
                                    inspection_signature(name, arguments),
                                );
                                progress_tracker.record_tool(&progress_label);
                                tool_progress_labels.push(progress_label.clone());
                                tool_timeline.push(tool_entry(
                                    name.clone(),
                                    hi_tools::target_path(name, arguments).unwrap_or_default(),
                                    0,
                                    &output,
                                    &progress_label,
                                ));
                                results[i] = Some((id.clone(), msg));
                                completed[i] = true;
                                completion_order.push(i);
                                done += 1;
                                interrupted = interrupted.saturating_add(1);
                            }
                        }
                        sched_tool_calls = sched_tool_calls.saturating_add(interrupted);
                        sched_serial_runs = sched_serial_runs.saturating_add(interrupted);
                        sched_max_concurrent = sched_max_concurrent.max(1);
                        ui.status("⚠ tool call interrupted by user — the model will adapt");
                        break;
                    }
                    // Ready: deps all complete.
                    let ready: Vec<usize> = (0..calls.len())
                        .filter(|&i| !completed[i] && deps[i].iter().all(|&d| completed[d]))
                        .collect();
                    if ready.is_empty() {
                        // Shouldn't happen (deps point backward), but if this
                        // ever regresses in release builds, do not record an
                        // assistant tool_use without a visible tool_result/UI
                        // result for each call. That corrupts the next provider
                        // request and looks like the model/tool harness stalled.
                        let unresolved: Vec<usize> =
                            (0..calls.len()).filter(|&i| !completed[i]).collect();
                        scheduler_forced_skip = true;
                        ui.status(
                            "⚠ tool scheduler could not make progress; marking unresolved calls as skipped",
                        );
                        sched_tool_calls += unresolved.len() as u32;
                        for i in unresolved {
                            let (id, name, arguments) = &calls[i];
                            ui.tool_call(name, arguments);
                            let msg = "Tool scheduler could not make progress; this call was skipped to keep the transcript valid.".to_string();
                            let mut output = synthetic_tool_outcome(
                                msg.clone(),
                                hi_tools::ToolStatus::Cancelled,
                            );
                            output.effects.mutation_attempted =
                                implementation_tool_call_mutates(name, arguments);
                            emit_tool_output(&mut *ui, name, &output);
                            results[i] = Some((id.clone(), msg));
                            completed[i] = true;
                            completion_order.push(i);
                            done += 1;
                            let progress_label = ToolProgressLabel::new(
                                ProgressKind::None,
                                "scheduler forced skip",
                                inspection_signature(name, arguments),
                            );
                            progress_tracker.record_tool(&progress_label);
                            tool_progress_labels.push(progress_label.clone());
                            tool_timeline.push(tool_entry(
                                name.clone(),
                                hi_tools::target_path(name, arguments).unwrap_or_default(),
                                0,
                                &output,
                                &progress_label,
                            ));
                        }
                        break;
                    }
                    // If any ready call is bash, run it alone (streaming UI).
                    let bash_idx = ready.iter().copied().find(|&i| calls[i].1 == "bash");
                    if let Some(i) = bash_idx {
                        let (id, name, arguments) = &calls[i];
                        let bash_mutates = implementation_tool_call_mutates(name, arguments);
                        if self.config.confirm_edits && bash_mutates {
                            let command =
                                bash_command(arguments).unwrap_or_else(|| arguments.clone());
                            let cwd = self.runtime.root().display().to_string();
                            let decision = ui
                                .confirm(ConfirmationRequest::ShellMutation { command, cwd })
                                .await;
                            if decision != ConfirmationResult::Approved {
                                ui.tool_call(name, arguments);
                                let msg = if decision == ConfirmationResult::Unavailable {
                                    "Shell mutation skipped: confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits."
                                } else {
                                    "Shell mutation skipped by user (not run)."
                                }
                                .to_string();
                                let mut output = synthetic_tool_outcome(
                                    msg.clone(),
                                    hi_tools::ToolStatus::Denied,
                                );
                                output.effects.mutation_attempted = true;
                                emit_tool_output(&mut *ui, name, &output);
                                let progress_label = ToolProgressLabel::new(
                                    ProgressKind::Weak,
                                    "shell mutation denied by confirmation",
                                    inspection_signature(name, arguments),
                                );
                                progress_tracker.record_tool(&progress_label);
                                tool_progress_labels.push(progress_label.clone());
                                tool_timeline.push(tool_entry(
                                    name.clone(),
                                    String::new(),
                                    0,
                                    &output,
                                    &progress_label,
                                ));
                                results[i] = Some((id.clone(), msg));
                                completed[i] = true;
                                completion_order.push(i);
                                done += 1;
                                sched_tool_calls += 1;
                                sched_serial_runs += 1;
                                sched_max_concurrent = sched_max_concurrent.max(1);
                                continue;
                            }
                        }
                        // Bash is opaque: an apparently read-only script or test
                        // can still rewrite files. Capture both the change
                        // baseline and undo checkpoint before every shell run;
                        // the mutation classifier is only a confirmation hint.
                        self.ensure_turn_snapshot(&mut turn_snapshot).await?;
                        if !self
                            .ensure_turn_checkpoint(
                                &mut turn_checkpoint_allowed,
                                &mut turn_checkpoint_created,
                                ui,
                            )
                            .await
                        {
                            ui.tool_call(name, arguments);
                            let msg = "Shell mutation skipped because strict mode requires an available checkpoint.".to_string();
                            let mut output =
                                synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                            output.effects.mutation_attempted = true;
                            emit_tool_output(&mut *ui, name, &output);
                            let progress_label = ToolProgressLabel::new(
                                ProgressKind::Weak,
                                "shell mutation denied without checkpoint",
                                inspection_signature(name, arguments),
                            );
                            progress_tracker.record_tool(&progress_label);
                            tool_progress_labels.push(progress_label.clone());
                            tool_timeline.push(tool_entry(
                                name.clone(),
                                String::new(),
                                0,
                                &output,
                                &progress_label,
                            ));
                            results[i] = Some((id.clone(), msg));
                            completed[i] = true;
                            completion_order.push(i);
                            done += 1;
                            sched_tool_calls += 1;
                            sched_serial_runs += 1;
                            sched_max_concurrent = sched_max_concurrent.max(1);
                            continue;
                        }
                        ui.tool_started(name, arguments);
                        ui.tool_call(name, arguments);
                        let path = hi_tools::target_path(name, arguments).unwrap_or_default();
                        let started = std::time::Instant::now();
                        let ui_ref: &mut dyn Ui = &mut *ui;
                        let lsp = self.runtime.lsp();
                        let output = execute_streaming_in_runtime(
                            self.runtime.root(),
                            self.runtime.state_root(),
                            &lsp,
                            self.runtime.background(),
                            self.runtime.read_cache(),
                            name,
                            arguments,
                            &mut |line: &str| ui_ref.tool_stream(name, line),
                        )
                        .await;
                        let duration_ms = started.elapsed().as_millis() as u64;
                        self.record_tool_effects(&output.effects)?;
                        self.reconcile_workspace_changes()?;
                        let error = output.status != hi_tools::ToolStatus::Succeeded;
                        let semantic_output = if error && !output.content.starts_with("Error:") {
                            std::borrow::Cow::Owned(format!("Error: {}", output.content))
                        } else {
                            std::borrow::Cow::Borrowed(output.content.as_str())
                        };
                        let signature = inspection_signature(name, arguments);
                        let signature_was_seen = signature_seen(&evidence, &signature);
                        let tracker_before = implementation_tracker.clone();
                        let validation_succeeded = tool_satisfies_validation(&output);
                        evidence.record_success(name, arguments, &semantic_output);
                        implementation_tracker.record_tool_result(
                            name,
                            arguments,
                            &semantic_output,
                            validation_succeeded,
                        );
                        let progress =
                            tool_guardrail.record_tool_result(name, arguments, &semantic_output);
                        if progress.hashable_idempotent {
                            hashable_idempotent_results += 1;
                            if progress.repeated_idempotent_result {
                                repeated_idempotent_results += 1;
                            }
                        }
                        let progress_label = classify_tool_progress(
                            name,
                            arguments,
                            &semantic_output,
                            error,
                            validation_succeeded,
                            signature,
                            signature_was_seen,
                            progress.repeated_idempotent_result,
                            &tracker_before,
                            false,
                        );
                        progress_tracker.record_tool(&progress_label);
                        tool_progress_labels.push(progress_label.clone());
                        tool_timeline.push(tool_entry(
                            name.clone(),
                            path,
                            duration_ms,
                            &output,
                            &progress_label,
                        ));
                        emit_tool_output(&mut *ui, name, &output);
                        results[i] = Some((id.clone(), output.content));
                        self.invalidate_snapshot();
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                        // Bash runs alone → a serial run and a batch of size 1.
                        sched_tool_calls += 1;
                        sched_serial_runs += 1;
                        sched_max_concurrent = sched_max_concurrent.max(1);
                        continue;
                    }
                    // Self-dispatched calls: `explore`/`delegate` run a child
                    // agent turn and `record_decision` mutates agent state, so
                    // all three need `&mut self` and can't join the parallel
                    // `execute` stream. Run one alone when it's ready — the dep
                    // graph then guarantees earlier mutations in the batch have
                    // landed before a subagent sees the tree.
                    let self_idx = ready.iter().copied().find(|&i| {
                        matches!(
                            calls[i].1.as_str(),
                            "explore" | "delegate" | "record_decision"
                        )
                    });
                    if let Some(i) = self_idx {
                        let (id, name, arguments) = &calls[i];
                        if name == "delegate" {
                            if self.config.confirm_edits {
                                let summary = serde_json::from_str::<serde_json::Value>(arguments)
                                    .ok()
                                    .and_then(|value| {
                                        value
                                            .get("task")
                                            .and_then(|v| v.as_str())
                                            .map(str::to_string)
                                    })
                                    .unwrap_or_else(|| arguments.clone());
                                let decision = ui
                                    .confirm(ConfirmationRequest::DelegateApply {
                                        summary: format!("Allow a write-capable delegate to apply verified changes for:\n{summary}"),
                                        diff: "The exact diff will be produced in an isolated worktree.".to_string(),
                                    })
                                    .await;
                                if decision != ConfirmationResult::Approved {
                                    ui.tool_call(name, arguments);
                                    let msg = if decision == ConfirmationResult::Unavailable {
                                        "Delegate skipped: confirmation required, but this frontend cannot answer it."
                                    } else {
                                        "Delegate skipped by user (no changes applied)."
                                    }
                                    .to_string();
                                    let mut output = synthetic_tool_outcome(
                                        msg.clone(),
                                        hi_tools::ToolStatus::Denied,
                                    );
                                    output.effects.mutation_attempted = true;
                                    emit_tool_output(&mut *ui, name, &output);
                                    let progress_label = ToolProgressLabel::new(
                                        ProgressKind::Weak,
                                        "delegate skipped by confirmation",
                                        inspection_signature(name, arguments),
                                    );
                                    progress_tracker.record_tool(&progress_label);
                                    tool_progress_labels.push(progress_label.clone());
                                    tool_timeline.push(tool_entry(
                                        name.clone(),
                                        String::new(),
                                        0,
                                        &output,
                                        &progress_label,
                                    ));
                                    results[i] = Some((id.clone(), msg));
                                    completed[i] = true;
                                    completion_order.push(i);
                                    done += 1;
                                    sched_tool_calls += 1;
                                    sched_serial_runs += 1;
                                    sched_max_concurrent = sched_max_concurrent.max(1);
                                    continue;
                                }
                            }
                            // Write-capable subagent: capture the turn baseline +
                            // checkpoint BEFORE it mutates the tree — otherwise the
                            // later lazy snapshot (verify gate) would record
                            // delegate's own output as the baseline, making the
                            // parent's verify + changed-files see "no changes", and
                            // leaving no pre-delegate checkpoint for `/undo` to
                            // isolate this turn.
                            self.ensure_turn_snapshot(&mut turn_snapshot).await?;
                            if !self
                                .ensure_turn_checkpoint(
                                    &mut turn_checkpoint_allowed,
                                    &mut turn_checkpoint_created,
                                    ui,
                                )
                                .await
                            {
                                ui.tool_call(name, arguments);
                                let msg = "Delegate skipped because strict mode requires an available checkpoint.".to_string();
                                let output = synthetic_tool_outcome(
                                    msg.clone(),
                                    hi_tools::ToolStatus::Denied,
                                );
                                emit_tool_output(&mut *ui, name, &output);
                                let progress_label = ToolProgressLabel::new(
                                    ProgressKind::Weak,
                                    "delegate skipped without checkpoint",
                                    inspection_signature(name, arguments),
                                );
                                progress_tracker.record_tool(&progress_label);
                                tool_progress_labels.push(progress_label.clone());
                                tool_timeline.push(tool_entry(
                                    name.clone(),
                                    String::new(),
                                    0,
                                    &output,
                                    &progress_label,
                                ));
                                results[i] = Some((id.clone(), msg));
                                completed[i] = true;
                                completion_order.push(i);
                                done += 1;
                                sched_tool_calls += 1;
                                sched_serial_runs += 1;
                                sched_max_concurrent = sched_max_concurrent.max(1);
                                continue;
                            }
                        }
                        ui.tool_call(name, arguments);
                        let started = std::time::Instant::now();
                        let output = match name.as_str() {
                            "explore" => self.handle_explore(arguments, &mut *ui).await,
                            "delegate" => self.handle_delegate(arguments, &mut *ui).await,
                            _ => self.handle_record_decision(arguments),
                        };
                        let duration_ms = started.elapsed().as_millis() as u64;
                        if name == "delegate" {
                            // The handler reconciles and attributes the exact
                            // delegate paths before returning its typed outcome.
                            self.invalidate_snapshot();
                        }
                        let error = output.status != hi_tools::ToolStatus::Succeeded;
                        let semantic_output = if error && !output.content.starts_with("Error:") {
                            std::borrow::Cow::Owned(format!("Error: {}", output.content))
                        } else {
                            std::borrow::Cow::Borrowed(output.content.as_str())
                        };
                        let signature = inspection_signature(name, arguments);
                        let signature_was_seen = signature_seen(&evidence, &signature);
                        let tracker_before = implementation_tracker.clone();
                        let validation_succeeded = tool_satisfies_validation(&output);
                        evidence.record_success(name, arguments, &semantic_output);
                        implementation_tracker.record_tool_result(
                            name,
                            arguments,
                            &semantic_output,
                            validation_succeeded,
                        );
                        let progress =
                            tool_guardrail.record_tool_result(name, arguments, &semantic_output);
                        if progress.hashable_idempotent {
                            hashable_idempotent_results += 1;
                            if progress.repeated_idempotent_result {
                                repeated_idempotent_results += 1;
                            }
                        }
                        let progress_label = if output.effects.mutation_applied {
                            ToolProgressLabel::new(
                                ProgressKind::Meaningful,
                                "successful delegated mutation",
                                signature,
                            )
                        } else {
                            classify_tool_progress(
                                name,
                                arguments,
                                &semantic_output,
                                error,
                                validation_succeeded,
                                signature,
                                signature_was_seen,
                                progress.repeated_idempotent_result,
                                &tracker_before,
                                false,
                            )
                        };
                        progress_tracker.record_tool(&progress_label);
                        tool_progress_labels.push(progress_label.clone());
                        tool_timeline.push(tool_entry(
                            name.clone(),
                            String::new(),
                            duration_ms,
                            &output,
                            &progress_label,
                        ));
                        emit_tool_output(&mut *ui, name, &output);
                        results[i] = Some((id.clone(), output.content));
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                        // Runs alone, like bash.
                        sched_tool_calls += 1;
                        sched_serial_runs += 1;
                        sched_max_concurrent = sched_max_concurrent.max(1);
                        continue;
                    }
                    // Run all ready non-bash calls concurrently. Record the
                    // completion order as the ready order (within a concurrent
                    // batch, relative order doesn't matter — none depend on
                    // each other, or they wouldn't all be ready).
                    let batch_size = ready.len() as u32;
                    let actual_concurrency = ready.len().min(max_parallel_tools) as u32;
                    // Signal each call as started so the live TUI can show a
                    // "running {tool}" timer. The transcript header is emitted
                    // later, paired with its result, so headers and results
                    // never drift apart in a concurrent batch.
                    for &i in &ready {
                        ui.tool_started(&calls[i].1, &calls[i].2);
                    }
                    // In --confirm-edits mode, check each mutating call with
                    // the UI before executing. Denied calls get a "skipped"
                    // result instead of running.
                    let mut denied: Vec<usize> = Vec::new();
                    let mut checkpoint_denied = BTreeSet::new();
                    let mut prepared_mutations = BTreeMap::new();
                    let mut preparation_failures = BTreeMap::new();
                    if self.config.confirm_edits {
                        for &i in &ready {
                            let name = &calls[i].1;
                            if matches!(
                                name.as_str(),
                                "write" | "edit" | "multi_edit" | "apply_patch"
                            ) {
                                let path = hi_tools::target_path(name, &calls[i].2)
                                    .unwrap_or_else(|| "(unknown)".to_string());
                                // Parse and materialize the complete mutation before
                                // confirmation. Approval consumes this same digest-sealed
                                // plan; it is never reparsed or rebuilt afterward.
                                let prepared = match prepare_mutation_in_with_state(
                                    self.runtime.root(),
                                    self.runtime.state_root(),
                                    name,
                                    &calls[i].2,
                                )
                                .await
                                {
                                    Ok(prepared) => prepared,
                                    Err(error) => {
                                        let mut output = synthetic_tool_outcome(
                                            format!("Error: {error:#}"),
                                            hi_tools::ToolStatus::Failed,
                                        );
                                        output.effects.mutation_attempted = true;
                                        preparation_failures.insert(i, output);
                                        continue;
                                    }
                                };
                                let preview = prepared.preview();
                                let decision = ui
                                    .confirm(ConfirmationRequest::FileEdit {
                                        path,
                                        diff: preview,
                                    })
                                    .await;
                                if decision != ConfirmationResult::Approved {
                                    if decision == ConfirmationResult::Unavailable {
                                        ui.status("confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits");
                                    }
                                    denied.push(i);
                                } else {
                                    prepared_mutations.insert(i, prepared);
                                }
                            }
                        }
                    }
                    let batch_started = std::time::Instant::now();
                    // Split ready into approved and denied; only execute approved.
                    let mut approved: Vec<usize> = ready
                        .iter()
                        .copied()
                        .filter(|i| !denied.contains(i))
                        .collect();
                    if approved.iter().any(|&i| {
                        !preparation_failures.contains_key(&i)
                            && implementation_tool_call_mutates(&calls[i].1, &calls[i].2)
                    }) {
                        self.ensure_turn_snapshot(&mut turn_snapshot).await?;
                        if !self
                            .ensure_turn_checkpoint(
                                &mut turn_checkpoint_allowed,
                                &mut turn_checkpoint_created,
                                ui,
                            )
                            .await
                        {
                            let blocked: Vec<usize> = approved
                                .iter()
                                .copied()
                                .filter(|&i| {
                                    !preparation_failures.contains_key(&i)
                                        && implementation_tool_call_mutates(
                                            &calls[i].1,
                                            &calls[i].2,
                                        )
                                })
                                .collect();
                            denied.extend(blocked.iter().copied());
                            checkpoint_denied.extend(blocked.iter().copied());
                            approved.retain(|i| !blocked.contains(i));
                        }
                    }
                    let root = self.runtime.root().to_path_buf();
                    let state_root = self.runtime.state_root().to_path_buf();
                    let lsp = self.runtime.lsp();
                    let executions = approved
                        .iter()
                        .map(|&i| {
                            (
                                i,
                                prepared_mutations.remove(&i),
                                preparation_failures.remove(&i),
                            )
                        })
                        .collect::<Vec<_>>();
                    let outputs: Vec<_> = futures_util::stream::iter(executions.into_iter().map(
                        |(i, prepared, failure)| {
                            let root = &root;
                            let state_root = &state_root;
                            let lsp = &lsp;
                            let background = self.runtime.background();
                            let read_cache = self.runtime.read_cache();
                            let calls = &calls;
                            async move {
                                if let Some(failure) = failure {
                                    failure
                                } else if let Some(prepared) = prepared {
                                    execute_prepared_in_runtime(lsp, read_cache, prepared).await
                                } else {
                                    execute_in_runtime(
                                        root,
                                        state_root,
                                        lsp,
                                        background,
                                        read_cache,
                                        &calls[i].1,
                                        &calls[i].2,
                                    )
                                    .await
                                }
                            }
                        },
                    ))
                    .buffered(max_parallel_tools)
                    .collect()
                    .await;
                    let batch_duration_ms = batch_started.elapsed().as_millis() as u64;
                    // Scheduler telemetry: count every call in the ready batch,
                    // but report actual concurrency after the configured cap.
                    sched_tool_calls += batch_size;
                    sched_max_concurrent = sched_max_concurrent.max(actual_concurrency);
                    if actual_concurrency == 1 {
                        sched_serial_runs += batch_size;
                    }
                    // Handle denied calls first: emit their headers and "skipped" results.
                    for &i in &denied {
                        let name = &calls[i].1;
                        ui.tool_call(name, &calls[i].2);
                        let skipped_msg = if checkpoint_denied.contains(&i) {
                            "Mutation skipped because strict mode requires an available checkpoint."
                                .to_string()
                        } else {
                            "Edit skipped by user (not applied).".to_string()
                        };
                        let mut output = synthetic_tool_outcome(
                            skipped_msg.clone(),
                            hi_tools::ToolStatus::Denied,
                        );
                        output.effects.mutation_attempted = true;
                        emit_tool_output(&mut *ui, name, &output);
                        results[i] = Some((calls[i].0.clone(), skipped_msg));
                        self.invalidate_snapshot();
                        let progress_label = ToolProgressLabel::new(
                            ProgressKind::Weak,
                            "tool skipped by user",
                            inspection_signature(name, &calls[i].2),
                        );
                        progress_tracker.record_tool(&progress_label);
                        tool_progress_labels.push(progress_label.clone());
                        tool_timeline.push(tool_entry(
                            name.clone(),
                            hi_tools::target_path(name, &calls[i].2).unwrap_or_default(),
                            0,
                            &output,
                            &progress_label,
                        ));
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                    }
                    for (&i, output) in approved.iter().zip(outputs) {
                        let name = &calls[i].1;
                        // Emit the transcript header immediately before its
                        // result — in a concurrent batch this pairs each header
                        // with its own result in completion order.
                        ui.tool_call(name, &calls[i].2);
                        let path = hi_tools::target_path(name, &calls[i].2).unwrap_or_default();
                        self.record_tool_effects(&output.effects)?;
                        if matches!(name.as_str(), "bash" | "bash_output" | "bash_kill") {
                            self.reconcile_workspace_changes()?;
                        }
                        let error = output.status != hi_tools::ToolStatus::Succeeded;
                        let semantic_output = if error && !output.content.starts_with("Error:") {
                            std::borrow::Cow::Owned(format!("Error: {}", output.content))
                        } else {
                            std::borrow::Cow::Borrowed(output.content.as_str())
                        };
                        let signature = inspection_signature(name, &calls[i].2);
                        let signature_was_seen = signature_seen(&evidence, &signature);
                        let tracker_before = implementation_tracker.clone();
                        let validation_succeeded = tool_satisfies_validation(&output);
                        let plan_changed = calls[i].1 == "update_plan"
                            && output
                                .plan
                                .as_deref()
                                .is_some_and(|plan| self.last_plan.as_slice() != plan);
                        plan_changed_this_batch |= plan_changed;
                        evidence.record_success(name, &calls[i].2, &semantic_output);
                        implementation_tracker.record_tool_result(
                            name,
                            &calls[i].2,
                            &semantic_output,
                            validation_succeeded,
                        );
                        let progress =
                            tool_guardrail.record_tool_result(name, &calls[i].2, &semantic_output);
                        if progress.hashable_idempotent {
                            hashable_idempotent_results += 1;
                            if progress.repeated_idempotent_result {
                                repeated_idempotent_results += 1;
                            }
                        }
                        let progress_label = classify_tool_progress(
                            name,
                            &calls[i].2,
                            &semantic_output,
                            error,
                            validation_succeeded,
                            signature,
                            signature_was_seen,
                            progress.repeated_idempotent_result,
                            &tracker_before,
                            plan_changed,
                        );
                        progress_tracker.record_tool(&progress_label);
                        tool_progress_labels.push(progress_label.clone());
                        tool_timeline.push(tool_entry(
                            name.clone(),
                            path,
                            batch_duration_ms,
                            &output,
                            &progress_label,
                        ));
                        emit_tool_output(&mut *ui, name, &output);
                        results[i] = Some((calls[i].0.clone(), output.content));
                        // Track the latest plan state so the continue logic can
                        // detect an incomplete plan when the model stops calling
                        // tools. The model resubmits the whole list on every
                        // call, so the last one is always current.
                        if calls[i].1 == "update_plan"
                            && let Some(plan) = output.plan.as_deref()
                        {
                            self.last_plan = plan.to_vec();
                            if let Some(session) = self.session.as_mut() {
                                if plan_has_pending_steps(plan) {
                                    session.record_plan(plan)?;
                                } else {
                                    // Keep the completed checklist visible for this live
                                    // turn, but do not resurrect it after a restart.
                                    session.clear_plan()?;
                                }
                            }
                            // Stage long-horizon progress without changing the
                            // live/durable goal. The turn-end gate commits this
                            // proposal only after current-revision verification
                            // and review succeed. The anchor comes from the
                            // durable goal (stable across the turn), so repeated
                            // update_plan calls can't compound past one advance.
                            if self.config.long_horizon
                                && let Some(current_goal) = self.structured_goal.as_ref()
                            {
                                let turn_start_active = current_goal.active_index();
                                let goal =
                                    proposed_goal.get_or_insert_with(|| current_goal.clone());
                                apply_plan_to_goal(goal, plan, turn_start_active);
                                plan_updated_goal = true;
                            }
                        }
                        // A filesystem-mutating tool may have changed files —
                        // invalidate the snapshot cache so a dependent read
                        // (guaranteed to run after by the dep graph) re-walks.
                        // `bash` also invalidates but always runs alone (above).
                        if hi_tools::is_filesystem_mutating(&calls[i].1) || calls[i].1 == "bash" {
                            self.invalidate_snapshot();
                            // Proactive per-edit verify: kick off a background
                            // fast check for the edited file so a syntax/lint
                            // error surfaces during the turn. The check is
                            // awaited after the batch; failures are non-fatal.
                            if self.config.proactive_verify
                                && let Some(path) = hi_tools::target_path(&calls[i].1, &calls[i].2)
                                && let Some(cmd) = hi_tools::fast_check_for(&path)
                            {
                                let root = self.runtime.root().to_path_buf();
                                let check = cmd.to_string();
                                let check_path = std::path::PathBuf::from(&path);
                                pending_checks.push((
                                    path,
                                    tokio::spawn(async move {
                                        hi_tools::run_fast_check_in(&root, &check, &check_path)
                                            .await
                                    }),
                                ));
                            }
                        }
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                    }
                }
                // Consume an interrupt that landed while (or just after) the
                // batch's last call finished — the loop above only polls the
                // flag between rounds, so a leftover flag would spuriously
                // cancel the next round's (or even the next turn's) batch.
                self.interrupt
                    .store(false, std::sync::atomic::Ordering::Relaxed);
                debug_assert_eq!(
                    done,
                    calls.len(),
                    "tool scheduler must account for every call"
                );
                // The completion order must respect the dep graph — a real
                // guarantee now (the scheduler only runs a call after its deps),
                // not just an emission-order coincidence.
                debug_assert!(
                    scheduler_forced_skip || respects_deps(&deps, &completion_order),
                    "scheduler completion must respect inferred tool deps: {:?} vs {:?}",
                    deps,
                    completion_order
                );
                let results: Vec<(String, String)> = results.into_iter().flatten().collect();
                self.messages
                    .push_assistant_with_results(std::mem::take(&mut completion.content), results);
                // Await the proactive per-edit checks kicked off during the
                // batch and surface each as a status line — a syntax/lint error
                // appears here, during the turn, before turn-end verify. A pass
                // is silent (no need to noise a clean edit); a failure names the
                // file and shows the check output so the model can fix it now.
                for (path, handle) in pending_checks {
                    if let Ok((passed, output)) = handle.await {
                        if passed {
                            continue;
                        }
                        ui.status(&format!("⚠ proactive check failed for {path}:\n{output}"));
                    }
                }
                implementation_tracker.record_tool_round();
                match self.handle_mutation_recovery(
                    &mut mutation_recovery,
                    expected_mutation,
                    &mut implementation_tracker,
                    &mut evidence,
                    plan_changed_this_batch,
                    &mut force_tools_next,
                    ui,
                ) {
                    MutationRecoveryControl::None => {}
                    MutationRecoveryControl::Continue => continue,
                }
                let repeated_result_no_progress = hash_guard_applies
                    && hashable_idempotent_results == calls.len()
                    && repeated_idempotent_results == calls.len();
                if repeated_result_no_progress {
                    prev_added_no_evidence = true;
                    let repeat_budget_available = repeat_nudges < self.config.max_repeat_nudges;
                    let no_new_after_mutation = implementation_tracker.mutation_seen;
                    if repeat_budget_available {
                        repeat_nudges += 1;
                        stalled_repeating = true;
                        let waiting_round = calls
                            .iter()
                            .any(|(_, name, args)| name == "bash" && bash_call_waits(args));
                        let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
                            if waiting_round {
                                "wait poll returned static output"
                            } else {
                                "repeated idempotent tool output"
                            },
                            no_progress_signature_for_calls(&calls),
                        ) && implementation_intent.is_none();
                        if waiting_round {
                            ui.nudge(&format!(
                                "the wait-and-check poll returned the same output — nudging the model to diagnose the stalled process ({repeat_nudges}/{})",
                                self.config.max_repeat_nudges
                            ));
                        } else {
                            ui.nudge(&format!(
                                "the model got the same inspection output again — nudging it to act on already-returned evidence ({repeat_nudges}/{})",
                                self.config.max_repeat_nudges
                            ));
                        }
                        let base_nudge = if waiting_round {
                            WAIT_POLL_STATIC_NUDGE
                        } else {
                            REREAD_NUDGE
                        };
                        let nudge = if force_final_after_nudge {
                            force_no_progress_final_answer_next = true;
                            force_tools_next = false;
                            format!("{base_nudge}\n\n{NO_PROGRESS_FINAL_ANSWER_NUDGE}")
                        } else {
                            base_nudge.to_string()
                        };
                        self.messages.push_nudge(NudgeKind::Repeat, nudge);
                        continue;
                    }
                    progress_tracker.record(
                        ProgressKind::None,
                        "repeated idempotent tool output",
                        no_progress_signature_for_calls(&calls),
                    );
                    if !no_new_after_mutation {
                        if let Some(intent) = read_only_intent {
                            stalled_unfinished = true;
                            ui.nudge(
                                "review kept getting the same inspection output; stopping incomplete",
                            );
                            let _ = intent;
                            ui.status(INCOMPLETE_STATUS);
                            break false;
                        }
                        if implementation_intent.is_some() && !implementation_tracker.mutation_seen
                        {
                            if implementation_tracker.no_change_nudges < 2 {
                                implementation_tracker.no_change_nudges += 1;
                                evidence.quality_repair_nudges =
                                    evidence.quality_repair_nudges.saturating_add(1);
                                let use_text_fallback =
                                    implementation_tracker.no_change_nudges >= 2;
                                force_tools_next = !use_text_fallback;
                                text_tool_fallback_next = use_text_fallback;
                                ui.nudge(
                                    "implementation repeated equivalent inspection output without editing; nudging the model to edit or scaffold",
                                );
                                let nudge = if use_text_fallback {
                                    implementation_text_tool_nudge(IMPLEMENTATION_NO_CHANGES_NUDGE)
                                } else {
                                    IMPLEMENTATION_NO_CHANGES_NUDGE.to_string()
                                };
                                self.messages.push_nudge(NudgeKind::Continue, nudge);
                                continue;
                            }

                            stalled_unfinished = true;
                            ui.nudge(
                                "implementation repeated equivalent inspection output without editing",
                            );
                            ui.status(INCOMPLETE_STATUS);
                            break false;
                        }
                    }
                } else if !tool_progress_labels.is_empty() {
                    progress_tracker.record_round_from_tools(&tool_progress_labels);
                }
            };

            if hit_cap {
                ui.status(&format!("reached step limit ({max_steps}); stopping turn"));
                ended_at_cap = true;
                break 'turn;
            }

            // Verification gate: run the stages in order (cheap compile/typecheck
            // first, then lint, then tests); the first to fail stops the turn and
            // its output is fed back. A passing pipeline ends the turn. The state
            // machine (round counter, change gating, stage execution) lives in the
            // `Verifier`; this loop just reacts to its outcome.
            let killed_backgrounds = self
                .runtime
                .background()
                .kill_started_after(&turn_background_baseline);
            if killed_backgrounds > 0 {
                ui.status(&format!(
                    "stopped {killed_backgrounds} live background process(es) before final verification"
                ));
                // Process-group termination is signalled synchronously. Yield so
                // the driver tasks can observe it before the final filesystem
                // reconciliation and verifier snapshot.
                tokio::task::yield_now().await;
                self.invalidate_snapshot();
                self.reconcile_workspace_changes()?;
            }
            let outcome = if verifier.is_on() {
                let baseline = self.ensure_turn_snapshot(&mut turn_snapshot).await?;
                let pre_turn_checkpoint = turn_checkpoint_created
                    .then(|| self.checkpoints.last())
                    .flatten()
                    .and_then(|reference| {
                        hi_tools::checkpoint::parse_reference(reference)
                            .ok()
                            .map(|(target, _)| target.to_string())
                    });
                let lsp = self.runtime.lsp();
                self.reconcile_workspace_changes()?;
                let (ledger_touched_files, ledger_mutation_seen) = {
                    let ledger = self.runtime.ledger();
                    (
                        ledger.touched_paths_since(turn_ledger_revision),
                        ledger.had_mutation_since(turn_ledger_revision),
                    )
                };
                let workspace = VerifyWorkspace::new(
                    self.runtime.root(),
                    self.runtime.state_root(),
                    pre_turn_checkpoint.as_deref(),
                    &lsp,
                )
                .with_changed_files(&ledger_touched_files)
                .with_mutation_seen(ledger_mutation_seen);
                verifier
                    .check(&workspace, &baseline, &mut self.snapshot_cache, ui)
                    .await
            } else {
                VerifyOutcome::NotRun
            };
            // Retain evidence immediately, not only in the common finalizer:
            // reconciliation or persistence can still fail after a successful
            // check, and reports for those error turns need the stages that
            // actually ran.
            self.last_turn_telemetry.verification_executions = verifier.executions().to_vec();
            match outcome {
                VerifyOutcome::NotRun => {
                    if self.last_verify == Some(false) {
                        stalled_unfinished = true;
                        ui.status(
                            "verification still failed after the retry budget; the task may be incomplete. /retry, or send 'continue'.",
                        );
                    }
                    break 'turn;
                }
                VerifyOutcome::SkippedNoChanges { first } => {
                    if first {
                        ui.status("verification skipped — no files changed this turn");
                    }
                    break 'turn;
                }
                VerifyOutcome::SkippedProseOnly { first } => {
                    if first {
                        ui.status("verification skipped — prose-only files changed this turn");
                    }
                    break 'turn;
                }
                VerifyOutcome::Passed => {
                    ui.status("✓ verification passed");
                    self.last_verify = Some(true);
                    self.reconcile_workspace_changes()?;
                    let (verified_revision, verified_digest, current_changes) = {
                        let ledger = self.runtime.ledger();
                        (
                            ledger.revision(),
                            ledger.workspace_revision(),
                            ledger.changes_since(turn_ledger_revision),
                        )
                    };
                    verified_at = Some((verified_revision, verified_digest.clone()));
                    let current_files = current_changes
                        .iter()
                        .map(|change| change.path.clone())
                        .collect::<Vec<_>>();
                    let mut diff = self.turn_diff().await;
                    let diff_lines = if diff.trim().is_empty() {
                        fallback_review_line_count(self.runtime.root(), &current_changes)
                    } else {
                        diff.lines().count()
                    };
                    let review_required =
                        self.last_task_contract.as_ref().is_some_and(|contract| {
                            contract.requires_review(
                                self.config.review,
                                &current_files,
                                diff_lines,
                                self.config.long_horizon || self.config.write_subagents,
                            )
                        });
                    if review_required {
                        self.refresh_active_task_context(
                            &context_task,
                            repository_context_enabled,
                            turn_ledger_revision,
                            &mut ranked_context_paths,
                            &mut context_generation_seen,
                            &mut indexed_ledger_revision,
                        );
                        if diff.chars().count() > 50_000 {
                            diff = diff.chars().take(50_000).collect();
                            diff.push_str("\n… (bounded review diff truncated)");
                        }
                        let contract = self
                            .last_task_contract
                            .as_ref()
                            .and_then(|contract| serde_json::to_string_pretty(contract).ok())
                            .unwrap_or_else(|| "(task contract unavailable)".into());
                        let instructions = self.task_context.as_deref().unwrap_or("(none)");
                        let stages = verifier.stages_summary().unwrap_or_else(|| "(none)".into());
                        let context = format!(
                            "Task contract:\n{contract}\n\nScoped instructions and relevant repository context:\n{instructions}\n\nChanged files:\n{}\n\nDeterministic verification: PASSED\nStages: {stages}\nVerified workspace revision: {}\n\nComplete bounded turn diff:\n{diff}",
                            current_files.join("\n"),
                            verified_digest,
                        );
                        ui.status("running independent completion review");
                        let verdict = if diff.trim().is_empty() && !current_files.is_empty() {
                            super::skeptic::SkepticVerdict::Unavailable(
                                "a complete turn diff was unavailable for the current changes"
                                    .into(),
                            )
                        } else {
                            self.independent_review(&context).await
                        };
                        match verdict {
                            super::skeptic::SkepticVerdict::Approve => {
                                independent_review_status = ReviewStatus::Passed;
                            }
                            super::skeptic::SkepticVerdict::Unavailable(reason) => {
                                independent_review_status = ReviewStatus::Unavailable;
                                ui.status(&format!(
                                    "independent review unavailable after deterministic pass: {reason}"
                                ));
                            }
                            super::skeptic::SkepticVerdict::Object(objections)
                                if independent_review_repairs == 0 =>
                            {
                                independent_review_repairs = 1;
                                independent_review_status = ReviewStatus::Objected;
                                self.last_verify = None;
                                verified_at = None;
                                verifier.allow_review_revalidation();
                                self.messages.push_nudge(
                                    NudgeKind::Review,
                                    format!(
                                        "Independent review found concrete completion defects. Repair them now, then re-run deterministic validation.\n\n{}",
                                        objections
                                            .iter()
                                            .map(|objection| format!("- {objection}"))
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    ),
                                );
                                ui.nudge("independent review objected; allowing one repair cycle");
                                continue 'turn;
                            }
                            super::skeptic::SkepticVerdict::Object(objections) => {
                                independent_review_status = ReviewStatus::Objected;
                                stalled_unfinished = true;
                                ui.status(&format!(
                                    "independent review objected again after repair: {}",
                                    objections.join("; ")
                                ));
                            }
                            // The independent-review prompt defines no ESCALATE
                            // verdict; treat a stray one as a final objection
                            // (no extra repair cycle — escalation means
                            // retrying can't fix it).
                            super::skeptic::SkepticVerdict::Escalate(objections) => {
                                independent_review_status = ReviewStatus::Objected;
                                stalled_unfinished = true;
                                ui.status(&format!(
                                    "independent review escalated — needs your judgment: {}",
                                    objections.join("; ")
                                ));
                            }
                        }
                    }
                    break 'turn;
                }
                VerifyOutcome::Failed {
                    stage,
                    output,
                    round,
                } => {
                    ui.status(&format!("✗ {} failed; iterating", stage.name));
                    self.last_verify = Some(false);
                    verified_at = None;
                    let guidance = stage_guidance(&stage);
                    // Attribution: parse the (already-condensed) failure output
                    // into structured file/line/symbol hints and prepend a
                    // "Likely cause" section so the model is pointed at the
                    // right region first. Enrich-only — the raw `Output:` block
                    // stays unchanged, so nothing the model could see before is
                    // hidden. Empty when nothing parseable is found (the nudge
                    // then keeps its original shape).
                    let causes = hi_tools::parse_attributions(&output, 3);
                    // Capture for telemetry (flushed to the Agent at turn end).
                    last_verify_attributions = causes.clone();
                    let cause_section = if causes.is_empty() {
                        String::new()
                    } else {
                        let lines: Vec<String> = causes
                            .iter()
                            .map(|a| {
                                let kind = match a.kind {
                                    hi_tools::AttrKind::Compile => "compile",
                                    hi_tools::AttrKind::Test => "test",
                                    hi_tools::AttrKind::Lint => "lint",
                                    hi_tools::AttrKind::Other => "other",
                                };
                                let loc = match (a.line, a.column) {
                                    (Some(l), Some(c)) => format!("{}:{}:{}", a.path, l, c),
                                    (Some(l), None) => format!("{}:{}", a.path, l),
                                    _ => a.path.clone(),
                                };
                                if loc.is_empty() {
                                    format!("- [{kind}] {}", a.message)
                                } else {
                                    format!("- [{kind}] {loc} — {}", a.message)
                                }
                            })
                            .collect();
                        format!(
                            "Likely cause (verify and fix first):\n{}\n\n",
                            lines.join("\n")
                        )
                    };
                    let nudge_body = format!(
                        "{cause_section}Verification stage `{}` failed (`{}`).\n\nOutput:\n{}\n\n{} \
                         If a previous fix didn't work, reconsider rather than repeat it.",
                        stage.name, stage.command, output, guidance
                    );
                    // Replace the previous verify nudge instead of accumulating.
                    // Only the latest verification output belongs in context.
                    // `replace_last_nudge` pops trailing tool/assistant messages
                    // from the prior verify cycle and the prior nudge itself
                    // (located by typed kind, not string-matching), then pushes
                    // the new one. On the first round there's no prior nudge, so
                    // nothing is popped — the model's just-finished turn stays.
                    self.messages
                        .replace_last_nudge(NudgeKind::Verify { round }, nudge_body);
                }
                VerifyOutcome::InfrastructureError {
                    stage,
                    output,
                    round,
                } => {
                    verification_infrastructure_error = true;
                    self.last_verify = None;
                    verified_at = None;
                    ui.status(&format!(
                        "verification infrastructure failed at {} (round {round}): {output}",
                        stage.name,
                    ));
                    break 'turn;
                }
                VerifyOutcome::Unstable {
                    stage,
                    changed_files,
                    round,
                } => {
                    verification_unstable = true;
                    stalled_unfinished = true;
                    self.last_verify = Some(false);
                    verified_at = None;
                    ui.status(&format!(
                        "verification is unstable in round {round}: stage {} modified {}",
                        stage.name,
                        changed_files.join(", ")
                    ));
                    break 'turn;
                }
            }
        }

        // Seal first: checkpoint creation may take long enough for an owned
        // process or editor to move the tree. The authoritative reconciliation
        // below therefore happens after this final asynchronous safety step.
        if turn_checkpoint_created && !self.seal_turn_checkpoint(ui).await? {
            turn_checkpoint_created = false;
            // Default YOLO permits checkpoint-free mutation. A seal failure
            // must be silent and non-terminal there; strict confirmation mode
            // still treats loss of its promised undo record as incomplete.
            stalled_unfinished |= !self.config.allow_no_checkpoint;
        }
        // The ledger is the authoritative source for exact effects, including
        // shell/delegate/background changes that did not flow through a file
        // mutation tool. Its revision is content-based and workspace-local.
        self.reconcile_workspace_changes()?;
        let (final_ledger_revision, final_workspace_revision, ledger_changes) = {
            let ledger = self.runtime.ledger();
            (
                ledger.revision(),
                ledger.workspace_revision(),
                ledger.changes_since(turn_ledger_revision),
            )
        };
        if self.last_verify == Some(true)
            && verified_at.as_ref().is_none_or(|(revision, digest)| {
                *revision != final_ledger_revision || digest != &final_workspace_revision
            })
        {
            self.last_verify = None;
            verified_at = None;
            if independent_review_status == ReviewStatus::Passed {
                independent_review_status = ReviewStatus::Unavailable;
            }
            ui.status("workspace changed after verification; the previous pass was invalidated");
        }
        self.last_changed_files = ledger_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.last_file_changes = ledger_changes;
        self.last_compat_fallbacks = compat_fallbacks;
        // Flush the per-turn counters (otherwise discarded locals) into
        // telemetry so `--report` / the eval harness can diagnose the turn's
        // trajectory: how many verify rounds, recovery retries, nudges fired,
        // and where the last verify failure pointed.
        self.last_turn_telemetry = build_turn_telemetry(
            max_steps,
            verifier.round(),
            empty_retries,
            repeat_nudges,
            continue_total_nudges,
            truncation_total_retries,
            &progress_tracker,
            ended_at_cap,
            stalled_unfinished,
            stalled_repeating,
            &last_verify_attributions,
            verifier.executions(),
            sched_tool_calls,
            sched_max_concurrent,
            sched_serial_runs,
            &tool_timeline,
            &evidence,
            &review_repair,
        );
        self.last_turn_telemetry.checkpoint_available =
            turn_checkpoint_allowed.map(|_| turn_checkpoint_created);
        self.last_turn_telemetry.advertised_tools = advertised_tool_names.into_iter().collect();
        self.last_turn_telemetry.tool_schema_tokens = tool_schema_tokens;

        // Verifier-gated skill auto-curation: after a turn that PASSED verification
        // and actually changed files, optionally distill a reusable technique into a
        // learned skill. The ground-truth verifier is the gate (safe with weak local
        // models); opt-in via `curate_skills`, and capped per session.
        if self.config.curate_skills
            && self.last_verify == Some(true)
            && !self.last_changed_files.is_empty()
            && self.auto_skills_written < super::MAX_AUTO_SKILLS_PER_SESSION
        {
            self.curate_turn_end(turn_start, ui).await;
        }

        // Surface the files this turn changed, so the user sees what was touched
        // without needing /diff. Skipped for read-only/Q&A turns (empty list).
        // Emitted BEFORE the finalize recap so the recap is the last text the
        // user sees (the "✓ done" marker follows it).
        if !self.last_changed_files.is_empty() {
            ui.changed_files(&self.last_changed_files);
        }

        // Finalization: after a turn where the model used its tools to change
        // files, make one dedicated tool-free call so the user always gets a
        // structured recap, even from a model that wouldn't summarize on its
        // own. Requiring `made_tool_call` keeps a plain Q&A turn (whose answer is
        // already the response) from triggering it. Skipped when the turn
        // hit the step cap or stalled repeating (the work may be incomplete).
        if self.config.finalize
            && made_tool_call
            && !ended_at_cap
            && !stalled_unfinished
            && !stalled_repeating
            && !self.last_changed_files.is_empty()
        {
            self.finalize_turn(turn_start, ui).await;
            // finalize_turn appended a [user: finalize-nudge][assistant: recap]
            // pair. Strip it from the persisted transcript so the FINALIZE_PROMPT
            // ("don't take any further action") doesn't bleed into the next turn
            // and make the model emit summary text instead of executing the new
            // prompt. The recap was already shown to the user via the UI.
            self.messages.strip_finalize_pair();
        }

        // Tool-free curation/finalization calls and external editors can take
        // time after the first final reconciliation. Reconcile once more before
        // any long-horizon progress or typed outcome is committed.
        self.reconcile_workspace_changes()?;
        let (settled_revision, settled_digest, settled_changes) = {
            let ledger = self.runtime.ledger();
            (
                ledger.revision(),
                ledger.workspace_revision(),
                ledger.changes_since(turn_ledger_revision),
            )
        };
        if self.last_verify == Some(true)
            && verified_at.as_ref().is_none_or(|(revision, digest)| {
                *revision != settled_revision || digest != &settled_digest
            })
        {
            self.last_verify = None;
            verified_at = None;
            if independent_review_status == ReviewStatus::Passed {
                independent_review_status = ReviewStatus::Unavailable;
            }
            ui.status("workspace changed after verification; the previous pass was invalidated");
        }
        self.last_changed_files = settled_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.last_file_changes = settled_changes;

        // Long-horizon progress happens only after the final settled revision
        // still matches deterministic verification.
        // Keep the pre-turn goal until every user/session callback has
        // finished. A late workspace mutation must also roll back progress
        // that this hook tentatively advances.
        let goal_before_final_settlement = goal_before.clone();
        let goal_invalidated_verification = self
            .goal_turn_end(
                super::goal_turn::GoalTurnState {
                    stalled_unfinished,
                    stalled_repeating,
                    hit_step_cap: ended_at_cap,
                    plan_updated_goal,
                    proposed_goal,
                    goal_before,
                    verified_at: verified_at.as_ref(),
                    turn_ledger_revision,
                },
                ui,
            )
            .await;
        if goal_invalidated_verification {
            verified_at = None;
            if independent_review_status == ReviewStatus::Passed {
                independent_review_status = ReviewStatus::Unavailable;
            }
        }

        // Report the user-prompt estimate and all turn-local model output; full request
        // context remains visible as the `ctx` gauge below.
        ui.turn_end(&self.usage_summary(&self.totals));
        // Strip any trailing synthetic nudge so it doesn't absorb the next
        // real prompt via `push_user_or_fold` (which folds a new user message
        // into a trailing user message). A stall (repeat-nudge, continue-
        // nudge, verify-fail, truncation) can leave a nudge as the last
        // entry; removing it here gives the next turn a clean transcript.
        self.messages.strip_trailing_nudges();
        self.persist()?;

        // `goal_turn_end`, `Ui::turn_end`, and a session sink are extension
        // points outside the verifier. Reconcile after all of them and before
        // constructing the typed outcome so none can create a false current-
        // revision pass. There are deliberately no callbacks after this
        // settlement point.
        self.reconcile_workspace_changes()?;
        let (outcome_revision, outcome_digest) = {
            let ledger = self.runtime.ledger();
            (ledger.revision(), ledger.workspace_revision())
        };
        let changed_after_final_hooks = self.last_verify == Some(true)
            && verified_at.as_ref().is_none_or(|(revision, digest)| {
                *revision != outcome_revision || digest != &outcome_digest
            });
        if changed_after_final_hooks {
            self.last_verify = None;
            verified_at = None;
            if independent_review_status == ReviewStatus::Passed {
                independent_review_status = ReviewStatus::Unavailable;
            }
            ui.status(
                "workspace changed during turn finalization; the previous pass and goal progress were invalidated",
            );
            if self.config.long_horizon
                && let Some(previous) = goal_before_final_settlement
            {
                self.structured_goal = Some(previous);
                self.refresh_system_message();
                // The earlier persist may contain tentatively advanced goal
                // state. Rewrite the goal record itself (message persistence
                // does not include side-channel goal state) before returning.
                if let Some(session) = self.session.as_mut()
                    && let Some(goal) = self.structured_goal.as_ref()
                {
                    session.record_goal(goal)?;
                }
            }
            // Capture any additional effects of the invalidation notification
            // or corrective persistence. No UI/session callback follows this.
            self.reconcile_workspace_changes()?;
        }
        let (final_changes, turn_had_mutation) = {
            let ledger = self.runtime.ledger();
            (
                ledger.changes_since(turn_ledger_revision),
                ledger.had_mutation_since(turn_ledger_revision),
            )
        };
        self.last_changed_files = final_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        self.last_file_changes = final_changes;

        let verification = if verification_infrastructure_error {
            VerificationStatus::InfrastructureError
        } else if self.last_verify == Some(true) {
            VerificationStatus::Passed
        } else if self.last_verify == Some(false) {
            VerificationStatus::Failed
        } else if (self.last_changed_files.is_empty() && !turn_had_mutation)
            || (!self.last_changed_files.is_empty()
                && self
                    .last_changed_files
                    .iter()
                    .all(|path| is_prose_only_path(path))
                && self.last_turn_telemetry.verification_executions.is_empty())
        {
            VerificationStatus::NotApplicable
        } else {
            VerificationStatus::Unverified
        };
        let skeptic_review = match self.last_turn_telemetry.skeptic_last_status {
            Some(crate::SkepticStatus::Approved) => ReviewStatus::Passed,
            Some(crate::SkepticStatus::Objected | crate::SkepticStatus::Escalated) => {
                ReviewStatus::Objected
            }
            Some(crate::SkepticStatus::Unavailable) => ReviewStatus::Unavailable,
            None => ReviewStatus::NotRequired,
        };
        let review = combined_review_status(independent_review_status, skeptic_review);
        let status = if verification_infrastructure_error {
            TurnStatus::Failed
        } else if ended_at_cap
            || stalled_unfinished
            || stalled_repeating
            || (expected_mutation && self.last_changed_files.is_empty())
            || verification == VerificationStatus::Failed
            || review == ReviewStatus::Objected
            || (verification == VerificationStatus::Unverified && !self.config.allow_unverified)
        {
            TurnStatus::Incomplete
        } else {
            TurnStatus::Completed
        };
        let stop_reason = if verification_infrastructure_error {
            TurnStopReason::InfrastructureFailure
        } else if verification_unstable {
            TurnStopReason::VerificationUnstable
        } else if ended_at_cap {
            TurnStopReason::StepLimit
        } else if review == ReviewStatus::Objected {
            TurnStopReason::ReviewObjected
        } else if verification == VerificationStatus::Failed {
            TurnStopReason::VerificationFailed
        } else if stalled_unfinished
            || stalled_repeating
            || (expected_mutation && self.last_changed_files.is_empty())
        {
            TurnStopReason::Stalled
        } else if verification == VerificationStatus::Unverified {
            TurnStopReason::VerificationUnavailable
        } else if verification == VerificationStatus::NotApplicable {
            TurnStopReason::NoApplicableVerification
        } else {
            TurnStopReason::Completed
        };
        let outcome = TurnOutcome {
            status,
            verification,
            review,
            stop_reason,
            changed_files: self.last_changed_files.clone(),
            verified_workspace_revision: (verification == VerificationStatus::Passed)
                .then(|| verified_at.as_ref().map(|(_, digest)| digest.clone()))
                .flatten(),
            effective_route: effective_model_route(
                &self.config,
                effective_fallback_route.as_deref(),
            ),
        };
        self.last_effective_route = outcome.effective_route.clone();
        self.last_turn_outcome = Some(outcome.clone());
        self.active_turn_ledger_revision = None;
        self.active_turn_message_start = None;
        Ok(outcome)
    }

    /// Make one dedicated, tool-free model call asking for a structured recap of
    /// the turn, and append it to the conversation as the closing assistant
    /// message. Best-effort: a provider error here doesn't fail the turn (the
    /// work is already done), it just leaves the turn without the extra summary.
    ///
    /// The synthetic request prompt is folded into history as a user turn so the
    /// roles stay alternating (some providers reject two assistant messages in a
    /// row) and the recap is part of the saved session.
    async fn finalize_turn(&mut self, turn_start: usize, ui: &mut dyn Ui) {
        // Only send the current turn's messages (plus the system prompt for
        // context), not the entire session history. The recap only needs to
        // know what happened *this turn* — sending 40K tokens of old context
        // to produce a 200-token summary is pure waste.
        let turn = &self.messages.as_slice()[turn_start..];
        let mut messages = Vec::with_capacity(turn.len() + 2);
        messages.push(self.minimal_system_message());
        messages.extend_from_slice(turn);
        messages.push(Message::user(FINALIZE_PROMPT));
        repair_invalid_tool_call_arguments_in_messages(&mut messages);

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: Arc::from(messages),
            tools: Arc::new([]), // recap only — no tool use
            max_tokens: 2048,    // throwaway call — recaps can be detailed
            temperature: self.config.temperature,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                compat: self.config.compat,
                tool_mode: ToolMode::ChatOnly,
                stream_usage: None,
            },
        };

        let mut recap = String::new();
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                recap.push_str(&text);
                ui.assistant_text(&text);
            }
            StreamEvent::Status(text) => ui.status(&text),
            StreamEvent::Reasoning(_) => {}
        };
        let completion = match self.provider.stream(request, &mut sink).await {
            Ok(completion) => completion,
            Err(err) => {
                // Finalize is a side call — book its error usage without resetting
                // the main conversation's `context_used` gauge.
                self.add_side_error_usage(&err);
                self.emit_usage(ui);
                // Flush any partially-streamed recap text before the status
                // line, so it isn't left dangling in the UI's pending buffer.
                ui.assistant_end();
                ui.status(&format!("(couldn't generate the final summary: {err})"));
                return;
            }
        };

        // Side call: spend counts, but its small request must not clobber the
        // main conversation's context gauge (see add_side_usage).
        self.add_side_usage(completion.usage);
        self.emit_usage(ui);

        // Fall back to the final content if the provider didn't stream text.
        // Emit it through the UI before assistant_end so the user actually sees
        // the recap — without this, a provider that returns text only in the
        // completion object (not via stream deltas) would have its summary
        // recorded in history but never displayed, so the turn appears to end
        // without its closing message.
        if recap.trim().is_empty() {
            for c in &completion.content {
                if let Content::Text(t) = c {
                    recap.push_str(t);
                    ui.assistant_text(t);
                }
            }
        }
        ui.assistant_end();

        if recap.trim().is_empty() {
            return; // nothing to record
        }
        // Record both the synthetic request and the recap so roles alternate.
        // The recap is a text-only assistant message (no tool calls).
        self.messages
            .push_nudge(NudgeKind::Finalize, FINALIZE_PROMPT);
        self.messages.push_assistant(vec![Content::Text(recap)]);
    }

    /// Format the completed-turn usage marker with explicitly scoped metrics.
    pub(crate) fn usage_summary(&self, usage: &hi_ai::Usage) -> String {
        // User-facing prompt size first. The full request can include system,
        // tool, and history context, so putting it first made a short question
        // like "what's your name?" appear to be a 1.5k-token user prompt.
        let mut summary = format!(
            "[user prompt estimate {} · output across all model calls {}{}",
            humanize_count(self.last_user_prompt_tokens),
            if self.last_turn_usage.estimated {
                "~"
            } else {
                ""
            },
            humanize_count(self.last_turn_usage.output_tokens),
        );
        if self.last_turn_usage.cache_read_tokens > 0 {
            summary.push_str(&format!(
                " ⟲{}",
                humanize_count(self.last_turn_usage.cache_read_tokens)
            ));
        }
        // The context gauge is the point-in-time full request size, which is
        // the number providers generally bill as input and the number that
        // drives context-window pressure.
        if let Some(window) = self.config.context_window
            && window > 0
        {
            let pct = (self.context_used * 100 / u64::from(window)).min(100);
            summary.push_str(&format!(
                " · ctx {}{pct}% ({}/{})",
                if self.last_turn_usage.estimated {
                    "~"
                } else {
                    ""
                },
                humanize_count(self.context_used),
                humanize_count(u64::from(window)),
            ));
        } else if self.context_used > 0 {
            summary.push_str(&format!(
                " · ctx {}{}",
                if self.last_turn_usage.estimated {
                    "~"
                } else {
                    ""
                },
                humanize_count(self.context_used)
            ));
        }
        if let Some(limits) = usage.rate_limits.and_then(rate_limit_summary) {
            summary.push_str(&format!(" · {limits}"));
        }
        // Per-turn trajectory: a terse "steer" suffix when the turn needed
        // more than one shot, so a noisy success reads differently from a clean
        // one. Clean turns (no verify rounds, no recovery retries, no nudges,
        // no stalls) add nothing. See `TurnTelemetry`.
        if let Some(steer) = self.turn_steer() {
            summary.push_str(&format!(" · {steer}"));
        }
        summary.push(']');
        summary
    }

    /// A terse per-turn steering summary for the usage line, or `None` when the
    /// turn was clean (no extra rounds of any kind, no stall). Format:
    /// `steer: 2 verify · 1 retry · stalled` — components omitted when zero.
    pub(crate) fn turn_steer(&self) -> Option<String> {
        let t = &self.last_turn_telemetry;
        let mut parts: Vec<String> = Vec::new();
        if t.verify_rounds > 0 {
            parts.push(format!("{} verify", t.verify_rounds));
        }
        if t.recovery_retries > 0 {
            parts.push(format!("{} retry", t.recovery_retries));
        }
        if t.repeat_nudges > 0 {
            parts.push(format!("{} repeat", t.repeat_nudges));
        }
        if t.continue_nudges > 0 {
            parts.push(format!("{} continue", t.continue_nudges));
        }
        if t.quality_repair_nudges > 0 {
            parts.push(format!("{} review-repair", t.quality_repair_nudges));
        }
        if t.truncation_retries > 0 {
            parts.push(format!("{} trunc", t.truncation_retries));
        }
        if t.stalled_unfinished || t.stalled_repeating {
            parts.push("stalled".to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(format!("steer: {}", parts.join(" · ")))
        }
    }

    pub(crate) fn request_tools_for(&self, mode: ToolMode) -> Arc<[ToolSpec]> {
        match mode {
            ToolMode::ChatOnly => Arc::new([]),
            // `explore` isn't classified read-only (that keeps a read-only *child*
            // from ever seeing it), but delegating a read-only investigation is
            // itself read-only — so a top-level agent keeps `explore` in a
            // read-only/review turn. A subagent never has it in `self.tools`.
            ToolMode::ReadOnly => self
                .tools
                .iter()
                .filter(|tool| {
                    hi_tools::is_read_only(&tool.name)
                        || (tool.name == "explore" && !self.config.is_subagent)
                })
                .cloned()
                .collect::<Vec<_>>()
                .into(),
            ToolMode::Auto | ToolMode::Required => self.tools.clone(),
        }
    }

    fn tools_unavailable_for(&self, input: &str) -> bool {
        matches!(
            self.config.tool_mode,
            ToolMode::ChatOnly | ToolMode::ReadOnly
        ) && looks_mutating(input)
    }

    /// Clean text-embedded tool-call JSON from `Content::Text` blocks in
    /// `content`. Used on the truncation path (before `parse_text_tool_calls`
    /// would normally run) so raw tool-call JSON doesn't leak into recorded
    /// history. Complete tool calls are extracted and stripped; partial JSON
    /// stays as text. `ToolCall` blocks are left in place — the caller
    /// (`push_assistant_text_only`) strips them.
    fn clean_text_tool_calls_from_content(&self, content: &mut Vec<Content>) -> bool {
        let mut new_content = Vec::new();
        let mut saw_partial_tool_call = false;
        for c in content.drain(..) {
            match c {
                Content::Text(t) => {
                    let parsed = parse_text_tool_calls(&t, textcall_id_offset(&self.messages));
                    if parsed.iter().any(|p| matches!(p, Content::ToolCall { .. })) {
                        // Tool calls found — keep only the Text blocks (drop
                        // the extracted ToolCalls; they're partial/truncated
                        // and have no matching results).
                        new_content.extend(
                            parsed.into_iter().filter(|p| {
                                matches!(p, Content::Text(_) | Content::Thinking { .. })
                            }),
                        );
                    } else if let Some(index) = partial_text_tool_call_start(&t) {
                        let prose = t[..index].trim_end();
                        if !prose.is_empty() {
                            new_content.push(Content::Text(prose.to_string()));
                        }
                        saw_partial_tool_call = true;
                    } else {
                        new_content.push(Content::Text(t));
                    }
                }
                Content::ToolCall { .. } => saw_partial_tool_call = true,
                other => new_content.push(other),
            }
        }
        *content = new_content;
        saw_partial_tool_call
    }
}

fn rate_limit_summary(limits: RateLimitState) -> Option<String> {
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

fn rate_limit_bucket_summary(label: &str, bucket: RateLimitBucket) -> Option<String> {
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

fn format_rate_limit_reset(seconds: u64) -> String {
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

    fn cfg(long_horizon: bool) -> crate::AgentConfig {
        crate::AgentConfig {
            long_horizon,
            max_steps_explicit: false,
            ..Default::default()
        }
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
