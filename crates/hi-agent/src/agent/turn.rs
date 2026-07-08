//! The main turn loop and its helpers: `run_turn` (user message → model →
//! tool calls → results → repeat, then verify), `finalize_turn`, and the
//! per-turn steering/tool-selection helpers.

use std::{collections::BTreeMap, sync::Arc};

use anyhow::Result;
use futures_util::StreamExt;
use hi_ai::{
    ChatRequest, Content, Message, OutputCapError, ProviderErrorKind, RateLimitBucket,
    RateLimitState, RequestProfile, Role, StreamEvent, ToolMode, ToolSpec, provider_error_kind,
};
use hi_tools::{PlanStatus, execute, execute_streaming};

use crate::command;
use crate::compaction;
use crate::heuristics::{
    RECOVERY_SAMPLING, StallMode, emit_tool_output, humanize_count, looks_like_continue,
    looks_like_unfinished_step, looks_mutating, parse_text_tool_calls, plan_has_pending_steps,
    recovery_sampling, recovery_telemetry, respects_deps, textcall_id_offset, tool_deps,
    tool_mode_label,
};
use crate::snapshot::changed_files_between;
use crate::steering::{
    CONCRETE_REVIEW_NUDGE, EvidenceTracker, GAP_SEARCH_OVERCLAIM_NUDGE,
    IMPLEMENTATION_EMPTY_TUI_NUDGE, IMPLEMENTATION_NO_CHANGES_NUDGE,
    IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE, ImplementationTracker, POST_TOOL_EMPTY_RESPONSE_NUDGE,
    READ_AFTER_SEARCH_NUDGE, READ_ONLY_SAFE_CONTEXT_WINDOW, REPEAT_NUDGE, REREAD_NUDGE,
    ReviewIntent, ReviewRepairMode, SECURITY_BROAD_SEARCH_NUDGE, SECURITY_SCOPE_NUDGE,
    TOOL_PROTOCOL_RETRY_NUDGE, TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE, ToolLoopGuardrail,
    active_read_only_inspection_cap, answer_says_insufficient_evidence, bash_no_progress_signature,
    classify_bash_command, classify_implementation_intent, classify_read_only_intent,
    concrete_review_answer_problem, deepen_review_nudge, evidence_kind_for_tool,
    implementation_missing_validation_nudge, implementation_text_tool_nudge,
    implementation_tool_call_mutates, implementation_tool_call_validates,
    implementation_tool_result_landed_mutation, implementation_tool_result_landed_substantive_edit,
    implementation_turn_prompt, inspected_paths_for_prompt, inspection_signature,
    inspection_sprawl_exhausted, inspection_sprawl_nudge, no_evidence_review_nudge,
    read_only_blocked_tool_result, read_only_blocks_tool, read_only_turn_prompt,
    repair_nudge_with_required_next, should_bootstrap_gpu_training_estimator, should_deepen_review,
    should_nudge_gap_search_overclaim, should_nudge_inspection_sprawl,
    should_nudge_no_evidence_review, should_nudge_read_after_repeated_search,
    should_nudge_read_after_search_final, should_nudge_security_broad_search,
    should_nudge_security_scope, should_reject_review_repair_template,
    summarize_inspected_evidence_nudge,
};
use crate::transcript::{NudgeKind, repair_invalid_tool_call_arguments_in_messages};
use crate::verify::{Snapshot, Verifier, VerifyOutcome, stage_guidance};
use crate::{
    AUTO_KEEP_RECENT, FINALIZE_PROMPT, MAX_TOOL_PROTOCOL_RETRIES, PLAN_CONTINUE_NUDGE,
    ProgressEvent, SILENT_CONTINUE_NUDGE, TRUNCATED_TOOL_CALL_NUDGE, TRUNCATION_NUDGE,
    ToolCallEntry, TurnAttribution, TurnTelemetry, Ui, apply_plan_to_goal,
    partial_text_tool_call_start,
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

#[derive(Clone, Debug)]
struct ProgressTracker {
    no_progress_streak: u32,
    no_progress_nudges: u32,
    forced_final_answer_attempts: u32,
    last_progress_reason: String,
    last_stall_reason: String,
    events: Vec<ProgressEvent>,
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self {
            no_progress_streak: 0,
            no_progress_nudges: 0,
            forced_final_answer_attempts: 0,
            last_progress_reason: String::new(),
            last_stall_reason: String::new(),
            events: Vec::new(),
        }
    }
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
    read_only_intent: Option<ReviewIntent>,
    implementation_intent: Option<crate::steering::ImplementationIntent>,
) -> u32 {
    if config.max_steps_explicit {
        return config.max_steps.max(1);
    }
    if config.long_horizon {
        200
    } else if implementation_intent.is_some() {
        120
    } else if read_only_intent.is_some() {
        80
    } else {
        200
    }
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
    if tracker_before.mutation_seen && implementation_tool_call_validates(name, arguments) {
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

fn tool_entry(
    tool: String,
    path: String,
    duration_ms: u64,
    error: bool,
    progress: &ToolProgressLabel,
) -> ToolCallEntry {
    ToolCallEntry {
        tool,
        path,
        duration_ms,
        error,
        progress_kind: progress.kind.as_str().to_string(),
        progress_reason: progress.reason.clone(),
        normalized_signature: progress.signature.clone(),
    }
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

    async fn ensure_turn_checkpoint(&mut self, checkpoint_created: &mut bool, _ui: &mut dyn Ui) {
        if *checkpoint_created {
            return;
        }
        *checkpoint_created = true;
        #[cfg(test)]
        return;
        #[cfg(not(test))]
        {
            // Snapshot lazily, immediately before the first approved mutating
            // tool. Creating a checkpoint requires walking non-ignored
            // untracked files so `/undo` does not delete pre-existing user
            // files. On large worktrees this is too expensive for read-only or
            // conversational turns.
            if let Some(sha) = hi_tools::checkpoint::create(std::path::Path::new(".")).await {
                self.checkpoints.push(sha);
                if self.checkpoints.len() > crate::MAX_CHECKPOINTS {
                    self.checkpoints
                        .drain(0..self.checkpoints.len() - crate::MAX_CHECKPOINTS);
                }
                if let Some(session) = self.session.as_mut()
                    && let Err(err) = session.record_checkpoints(&self.checkpoints)
                {
                    _ui.status(&format!("(couldn't persist checkpoint refs: {err})"));
                }
            }
        }
    }

    async fn ensure_turn_snapshot(&mut self, turn_snapshot: &mut Option<Snapshot>) -> Snapshot {
        if let Some(snapshot) = turn_snapshot.as_ref() {
            return snapshot.clone();
        }
        let snapshot = self.snapshot_cached().await;
        *turn_snapshot = Some(snapshot.clone());
        snapshot
    }

    /// Run one user turn to completion, emitting output through `ui`.
    ///
    /// After the model stops calling tools, an optional verification command is
    /// run; if it fails, its output is fed back and the model iterates, up to
    /// `max_verify_iterations` rounds.
    pub async fn run_turn(&mut self, input: &str, ui: &mut dyn Ui) -> Result<()> {
        let expanded_input =
            command::expand_prompt_macro(input).unwrap_or_else(|| input.to_string());
        let implementation_candidate = classify_implementation_intent(&expanded_input);
        let read_only_intent = if implementation_candidate.is_some() {
            None
        } else {
            classify_read_only_intent(&expanded_input)
        };
        let implementation_intent = if read_only_intent.is_none() {
            implementation_candidate
        } else {
            None
        };
        let read_only_inspection_cap =
            read_only_intent.map(|intent| active_read_only_inspection_cap(&expanded_input, intent));
        let turn_input = if let Some(intent) = read_only_intent {
            read_only_turn_prompt(&expanded_input, intent)
        } else if let Some(intent) = implementation_intent {
            implementation_turn_prompt(&expanded_input, intent)
        } else {
            expanded_input
        };
        let input = turn_input.as_str();

        if read_only_intent.is_none() && self.tools_unavailable_for(input) {
            self.last_verify = None;
            self.last_changed_files.clear();
            self.last_compat_fallbacks.clear();
            self.last_turn_telemetry = TurnTelemetry::default();
            if !looks_like_continue(input) {
                self.last_plan.clear();
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
            return Ok(());
        }
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
            {
                compaction::elide_tool_outputs(self.messages.mutate_slice(), split);
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
        self.messages.push_user_or_fold(input);
        self.last_verify = None;
        self.last_changed_files.clear();
        self.last_compat_fallbacks.clear();
        // Clear the plan from the previous turn unless the user's input looks
        // like a "continue" command. When the user types "continue" on an
        // incomplete plan, the plan state should persist so the plan-aware
        // continue logic can fire. For any other input, clear it so a stale
        // plan from a previous task doesn't cause spurious nudges.
        if !looks_like_continue(input) {
            self.last_plan.clear();
        }
        let mut compat_fallbacks = Vec::new();

        let mut verifier = Verifier::new(
            self.config.verify.clone(),
            self.config.max_verify_iterations,
        );
        let max_steps =
            effective_max_steps_for_turn(&self.config, read_only_intent, implementation_intent);
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
        // Whether the model's update_plan call already advanced the structured
        // goal during this turn (so goal_turn_end doesn't advance again and
        // skip the next sub-goal).
        let mut plan_updated_goal = false;
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
        let mut evidence = EvidenceTracker::default();
        let mut review_repair = ReviewRepairState::default();
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
                    read_only_inspection_cap.unwrap_or_else(|| evidence.inspection_count()),
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
        let mut verify_snapshot: Option<Snapshot> = None;

        if let Some(intent) = implementation_intent
            && !matches!(self.config.tool_mode, ToolMode::ChatOnly)
            && implementation_tracker.preferred_validation.is_none()
            && should_bootstrap_gpu_training_estimator(intent)
        {
            self.ensure_turn_snapshot(&mut turn_snapshot).await;
            self.ensure_turn_checkpoint(&mut turn_checkpoint_created, ui)
                .await;
            let bootstrap_calls = self
                .run_gpu_training_estimator_bootstrap(
                    ui,
                    &mut implementation_tracker,
                    &mut tool_timeline,
                    intent,
                )
                .await;
            if bootstrap_calls > 0 {
                made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(bootstrap_calls);
                sched_serial_runs = sched_serial_runs.saturating_add(bootstrap_calls);
                sched_max_concurrent = sched_max_concurrent.max(1);
                empty_tui_needs_project = false;
            }
        }
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

                // After a content-less/garbled round, resample hotter and with
                // nucleus + frequency penalty on the retry to break out of the
                // low-entropy attractor that produced it (cf. minion's recovery
                // sampling). Bounded, and only while consecutively stalling —
                // `empty_retries` resets on real output, so a normal round runs at
                // the configured sampling. Toggleable via HI_RECOVERY_SAMPLING for
                // A/B-ing on the eval harness.
                let sampling_retries = empty_retries.max(retry_state.protocol_retries);
                let sampling_budget = if retry_state.protocol_retries > empty_retries {
                    MAX_TOOL_PROTOCOL_RETRIES
                } else {
                    self.config.max_empty_retries
                };
                let (temperature, top_p, frequency_penalty) = recovery_sampling(
                    sampling_retries,
                    self.config.temperature,
                    *RECOVERY_SAMPLING,
                );

                // Telemetry for the recovery-sampling A/B: emit a concise debug
                // line only when sampling is actually being changed (recovery on
                // and this is a retry), so ordinary runs stay quiet. The empty
                // path is the only mode that escalates sampling today; repeat and
                // continue nudges re-run at the configured sampling.
                if let Some(line) = recovery_telemetry(
                    StallMode::Empty,
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
                let request_tools = self.request_tools_for(tool_availability_mode);
                let context_preflight = match self.ensure_request_fits_context(
                    input,
                    turn_start,
                    requested_request_max_tokens,
                    estimate_tool_schema_tokens(&request_tools),
                    context_safety_window,
                    ui,
                ) {
                    Ok(context_preflight) => context_preflight,
                    Err(err) => {
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
                        return Err(err);
                    }
                };
                if context_preflight.dropped_prior_context {
                    turn_start = self.messages.len().saturating_sub(1);
                }
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
                        self.emit_usage(ui);
                        if let Some(turn_snapshot) = turn_snapshot.as_ref() {
                            self.messages.strip_trailing_nudges();
                            let end_snapshot = self.snapshot_cached().await;
                            self.last_changed_files =
                                changed_files_between(turn_snapshot, &end_snapshot);
                        } else if made_tool_call {
                            self.last_changed_files.clear();
                        } else {
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
                let exact_repeat = !calls.is_empty()
                    && !has_background_output_poll
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
                let is_repeat = exact_repeat
                    || (no_new_evidence
                        && (prev_added_no_evidence || stale_background_handle_call));
                let no_new_after_mutation = is_repeat
                    && no_new_evidence
                    && implementation_tracker.mutation_seen
                    && !stale_background_handle_call;
                let repeat_budget_available = repeat_nudges < self.config.max_repeat_nudges;
                let should_skip_for_repeat =
                    is_repeat && (!no_new_after_mutation || repeat_budget_available);
                if should_skip_for_repeat {
                    // Record this round's assistant text (the model did emit
                    // something) before nudging, so the history stays coherent.
                    // We deliberately do NOT execute the repeated tool calls, so
                    // strip their `ToolCall` blocks from the recorded message:
                    // `push_assistant_text_only` is the intentional "calls
                    // skipped, not executed" path — leaving `tool_use` blocks
                    // without matching `tool_result` blocks puts the transcript
                    // in a state most providers reject on the next request.
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    if repeat_budget_available {
                        repeat_nudges += 1;
                        stalled_repeating = true;
                        let stall_reason = if stale_background_handle_call {
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
                        ) && !no_new_after_mutation;
                        let nudge = if stale_background_handle_call {
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
                        // re-reading instead of editing, through the whole
                        // repeat budget. This is the "explore forever, never
                        // edit" failure mode: report it as
                        // implementation-incomplete (matching the no-changes
                        // path) rather than the generic "stuck repeating"
                        // notice, so the user knows the issue is that no edit
                        // was made, not that a command failed.
                        stalled_unfinished = true;
                        ui.nudge(
                            "implementation kept re-reading without editing; no file changes were made",
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
                // on, so clear any pending repeat-stall state.
                stalled_repeating = false;
                prev_call_sig = Some(call_sig);
                prev_added_no_evidence = no_new_evidence;

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
                    read_only_intent,
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
                    read_only_intent,
                    &evidence,
                    &calls,
                    read_only_inspection_cap,
                ) {
                    evidence.inspection_sprawl_nudges =
                        evidence.inspection_sprawl_nudges.saturating_add(1);
                    force_text_answer_next = true;
                    let cap =
                        read_only_inspection_cap.unwrap_or_else(|| evidence.inspection_count());
                    ui.nudge(&format!(
                        "review inspected {} files/searches without answering; nudging it to produce findings",
                        evidence.inspection_count()
                    ));
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        inspection_sprawl_nudge(cap, evidence.inspection_count()),
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
                    let needs_evidence_depth_repair = evidence.listing_only()
                        || (evidence.saw_search && !evidence.saw_read)
                        || (matches!(read_only_intent, Some(ReviewIntent::Security))
                            && evidence.saw_search
                            && evidence.saw_read
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
                let hash_guard_applies = calls
                    .iter()
                    .all(|(_, name, _)| matches!(name.as_str(), "read" | "list" | "grep" | "glob"));
                let mut hashable_idempotent_results = 0usize;
                let mut repeated_idempotent_results = 0usize;
                let mut tool_progress_labels: Vec<ToolProgressLabel> = Vec::new();
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
                // Pre-pass: handle `record_decision` calls serially. They mutate
                // agent state (`self.decisions`) and aren't real tool dispatches,
                // so they can't run in the parallel `execute` stream (no `&mut
                // self` there). They're instantaneous and have no deps that
                // matter, so handling them up front is safe.
                for (i, (id, name, arguments)) in calls.iter().enumerate() {
                    if read_only_blocks_tool(read_only_intent, name) {
                        ui.tool_call(name, arguments);
                        let content = read_only_blocked_tool_result(name);
                        emit_tool_output(
                            &mut *ui,
                            name,
                            &hi_tools::ToolOutput {
                                content: content.clone(),
                                display: None,
                                plan: None,
                            },
                        );
                        results[i] = Some((id.clone(), content));
                        completed[i] = true;
                        completion_order.push(i);
                        continue;
                    }
                    if name != "record_decision" {
                        continue;
                    }
                    ui.tool_call(name, arguments);
                    let content = self.handle_record_decision(arguments);
                    ui.tool_result(name, &content);
                    results[i] = Some((id.clone(), content));
                    completed[i] = true;
                    completion_order.push(i);
                }
                let mut done = completion_order.len();
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
                        for i in 0..calls.len() {
                            if !completed[i] {
                                let (id, name, _) = &calls[i];
                                ui.tool_call(name, "[]");
                                let msg = "Tool call interrupted by user.".to_string();
                                ui.tool_result(name, &msg);
                                results[i] = Some((id.clone(), msg));
                                completed[i] = true;
                                completion_order.push(i);
                                done += 1;
                            }
                        }
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
                            emit_tool_output(
                                &mut *ui,
                                name,
                                &hi_tools::ToolOutput {
                                    content: msg.clone(),
                                    display: None,
                                    plan: None,
                                },
                            );
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
                                true,
                                &progress_label,
                            ));
                        }
                        break;
                    }
                    // If any ready call is bash, run it alone (streaming UI).
                    let bash_idx = ready.iter().copied().find(|&i| calls[i].1 == "bash");
                    if let Some(i) = bash_idx {
                        let (id, name, arguments) = &calls[i];
                        // Bash can mutate the workspace in ways static shell
                        // heuristics miss (redirection, scripts, build tools),
                        // so capture the baseline/checkpoint before it runs.
                        // The guard layer still blocks catastrophic commands.
                        self.ensure_turn_snapshot(&mut turn_snapshot).await;
                        self.ensure_turn_checkpoint(&mut turn_checkpoint_created, ui)
                            .await;
                        ui.tool_started(name, arguments);
                        ui.tool_call(name, arguments);
                        let path = hi_tools::target_path(name, arguments).unwrap_or_default();
                        let started = std::time::Instant::now();
                        let ui_ref: &mut dyn Ui = &mut *ui;
                        let output = execute_streaming(name, arguments, &mut |line: &str| {
                            ui_ref.tool_stream(name, line);
                        })
                        .await;
                        let duration_ms = started.elapsed().as_millis() as u64;
                        let error = output.content.starts_with("Error:");
                        let signature = inspection_signature(name, arguments);
                        let signature_was_seen = signature_seen(&evidence, &signature);
                        let tracker_before = implementation_tracker.clone();
                        evidence.record_success(name, arguments, &output.content);
                        implementation_tracker.record_tool_result(name, arguments, &output.content);
                        let progress =
                            tool_guardrail.record_tool_result(name, arguments, &output.content);
                        if progress.hashable_idempotent {
                            hashable_idempotent_results += 1;
                            if progress.repeated_idempotent_result {
                                repeated_idempotent_results += 1;
                            }
                        }
                        let progress_label = classify_tool_progress(
                            name,
                            arguments,
                            &output.content,
                            error,
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
                            error,
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
                    if self.config.confirm_edits {
                        for &i in &ready {
                            let name = &calls[i].1;
                            if matches!(
                                name.as_str(),
                                "write" | "edit" | "multi_edit" | "apply_patch"
                            ) {
                                let path = hi_tools::target_path(name, &calls[i].2)
                                    .unwrap_or_else(|| "(unknown)".to_string());
                                // Generate a diff preview for edit/multi_edit/apply_patch.
                                let preview = match name.as_str() {
                                    "edit" | "multi_edit" | "apply_patch" => {
                                        "(diff preview unavailable in concurrent batch)".to_string()
                                    }
                                    _ => String::new(),
                                };
                                if !ui.confirm_edit(&path, &preview) {
                                    denied.push(i);
                                }
                            }
                        }
                    }
                    let batch_started = std::time::Instant::now();
                    // Split ready into approved and denied; only execute approved.
                    let approved: Vec<usize> = ready
                        .iter()
                        .copied()
                        .filter(|i| !denied.contains(i))
                        .collect();
                    if approved
                        .iter()
                        .any(|&i| implementation_tool_call_mutates(&calls[i].1, &calls[i].2))
                    {
                        self.ensure_turn_snapshot(&mut turn_snapshot).await;
                        self.ensure_turn_checkpoint(&mut turn_checkpoint_created, ui)
                            .await;
                    }
                    let outputs: Vec<_> = futures_util::stream::iter(
                        approved.iter().map(|&i| execute(&calls[i].1, &calls[i].2)),
                    )
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
                        let skipped_msg = "Edit skipped by user (not applied).".to_string();
                        emit_tool_output(
                            &mut *ui,
                            name,
                            &hi_tools::ToolOutput {
                                content: skipped_msg.clone(),
                                display: None,
                                plan: None,
                            },
                        );
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
                            true,
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
                        let error = output.content.starts_with("Error:");
                        let signature = inspection_signature(name, &calls[i].2);
                        let signature_was_seen = signature_seen(&evidence, &signature);
                        let tracker_before = implementation_tracker.clone();
                        let plan_changed = calls[i].1 == "update_plan"
                            && output
                                .plan
                                .as_deref()
                                .is_some_and(|plan| self.last_plan.as_slice() != plan);
                        evidence.record_success(name, &calls[i].2, &output.content);
                        implementation_tracker.record_tool_result(
                            name,
                            &calls[i].2,
                            &output.content,
                        );
                        let progress =
                            tool_guardrail.record_tool_result(name, &calls[i].2, &output.content);
                        if progress.hashable_idempotent {
                            hashable_idempotent_results += 1;
                            if progress.repeated_idempotent_result {
                                repeated_idempotent_results += 1;
                            }
                        }
                        let progress_label = classify_tool_progress(
                            name,
                            &calls[i].2,
                            &output.content,
                            error,
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
                            error,
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
                        }
                        // Long-horizon: the model's `update_plan` statuses map
                        // onto the structured goal's sub-goals, so the agent
                        // advances/skips in lockstep with the model's stated
                        // progress. Only when long_horizon is on and a goal is
                        // set; the plan UI still renders via the ToolOutput.
                        if self.config.long_horizon
                            && calls[i].1 == "update_plan"
                            && let Some(goal) = self.structured_goal.as_mut()
                        {
                            apply_plan_to_goal(goal, &calls[i].2);
                            plan_updated_goal = true;
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
                                let cmd = format!("{cmd} {path}");
                                pending_checks.push((
                                    path,
                                    tokio::spawn(async move { hi_tools::run_check(&cmd).await }),
                                ));
                            }
                        }
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                    }
                }
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
                        let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
                            "repeated idempotent tool output",
                            no_progress_signature_for_calls(&calls),
                        );
                        ui.nudge(&format!(
                            "the model got the same inspection output again — nudging it to act on already-returned evidence ({repeat_nudges}/{})",
                            self.config.max_repeat_nudges
                        ));
                        let nudge = if force_final_after_nudge {
                            force_no_progress_final_answer_next = true;
                            force_tools_next = false;
                            format!("{REREAD_NUDGE}\n\n{NO_PROGRESS_FINAL_ANSWER_NUDGE}")
                        } else {
                            REREAD_NUDGE.to_string()
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
            let outcome = if verifier.is_on() {
                let baseline = self.ensure_turn_snapshot(&mut turn_snapshot).await;
                verifier
                    .check(&baseline, &mut self.snapshot_cache, ui)
                    .await
            } else {
                VerifyOutcome::NotRun
            };
            // Capture the verify snapshot for turn-end reuse whenever the
            // verifier actually walked the tree (i.e. it didn't bail before
            // snapshotting). On a failure we drop it: the model is about to edit
            // again, so it's no longer current.
            if matches!(
                outcome,
                VerifyOutcome::Passed
                    | VerifyOutcome::Failed { .. }
                    | VerifyOutcome::SkippedProseOnly { .. }
            ) {
                verify_snapshot = Some(self.snapshot_cached().await);
                if matches!(outcome, VerifyOutcome::Failed { .. }) {
                    verify_snapshot = None;
                }
            }
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
                    break 'turn;
                }
                VerifyOutcome::Failed {
                    stage,
                    output,
                    round,
                } => {
                    ui.status(&format!("✗ {} failed; iterating", stage.name));
                    self.last_verify = Some(false);
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
            }
        }

        // Reuse the verify snapshot when available (verify passed or found no
        // changes — no model work happened since). Otherwise take a fresh one.
        if let Some(turn_snapshot) = turn_snapshot.as_ref() {
            let end_snapshot = match verify_snapshot.take() {
                Some(s) => s,
                None => self.snapshot_cached().await,
            };
            self.last_changed_files = changed_files_between(turn_snapshot, &end_snapshot);
        } else {
            self.last_changed_files.clear();
        }
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
            sched_tool_calls,
            sched_max_concurrent,
            sched_serial_runs,
            &tool_timeline,
            &evidence,
            &review_repair,
        );

        // Long-horizon driver: when a structured goal is set and long_horizon
        // is on, advance or retry the active sub-goal based on this turn's
        // outcome — so the next turn resumes coherently at the right sub-goal
        // (and with prior-attempt notes if it stalled). See `goal_turn_end`.
        self.goal_turn_end(
            stalled_unfinished,
            stalled_repeating,
            ended_at_cap,
            plan_updated_goal,
            ui,
        );

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

        // Report cumulative session usage — the same number the live working
        // line and `/tokens` show, so the three never disagree.
        ui.turn_end(&self.usage_summary(&self.totals));
        // Strip any trailing synthetic nudge so it doesn't absorb the next
        // real prompt via `push_user_or_fold` (which folds a new user message
        // into a trailing user message). A stall (repeat-nudge, continue-
        // nudge, verify-fail, truncation) can leave a nudge as the last
        // entry; removing it here gives the next turn a clean transcript.
        self.messages.strip_trailing_nudges();
        self.persist()?;
        Ok(())
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
                self.add_error_usage(&err);
                self.emit_usage(ui);
                // Flush any partially-streamed recap text before the status
                // line, so it isn't left dangling in the UI's pending buffer.
                ui.assistant_end();
                ui.status(&format!("(couldn't generate the final summary: {err})"));
                return;
            }
        };

        self.add_usage(completion.usage);
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

    /// Format a usage line. `usage` carries the cumulative in/out/total;
    /// the context gauge instead uses `context_used` (the live conversation
    /// size), since cumulative input sums re-sent context across rounds and so
    /// isn't a measure of how full the window is.
    pub(crate) fn usage_summary(&self, usage: &hi_ai::Usage) -> String {
        // Cumulative session tokens, ↑ sent / ↓ received — these match the
        // live working line. Abbreviated in the same units as the
        // context gauge below so the two never read as raw-vs-rounded.
        let mut summary = format!(
            "[↑{} ↓{}",
            humanize_count(usage.input_tokens),
            humanize_count(usage.output_tokens),
        );
        if usage.cache_read_tokens > 0 {
            summary.push_str(&format!(" ⟲{}", humanize_count(usage.cache_read_tokens)));
        }
        // The context gauge is a *point-in-time* measure (the last request's
        // size), not cumulative input — so it is correctly smaller than ↑.
        if let Some(window) = self.config.context_window
            && window > 0
        {
            let pct = (self.context_used * 100 / u64::from(window)).min(100);
            summary.push_str(&format!(
                " · ctx {pct}% ({}/{})",
                humanize_count(self.context_used),
                humanize_count(u64::from(window)),
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

    fn request_tools_for(&self, mode: ToolMode) -> Arc<[ToolSpec]> {
        match mode {
            ToolMode::ChatOnly => Arc::new([]),
            ToolMode::ReadOnly => self
                .tools
                .iter()
                .filter(|tool| hi_tools::is_read_only(&tool.name))
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
