//! The main turn loop and its helpers: `run_turn` (user message → model →
//! tool calls → results → repeat, then verify), `finalize_turn`, and the
//! per-turn steering/tool-selection helpers.

use std::sync::Arc;

use anyhow::Result;
use futures_util::StreamExt;
use hi_ai::{
    ChatRequest, Content, Message, ProviderErrorKind, RequestProfile, Role, StreamEvent, ToolMode,
    ToolSpec, provider_error_kind,
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
    IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE, INSPECTION_SPRAWL_NUDGE, ImplementationTracker,
    READ_AFTER_SEARCH_NUDGE, READ_ONLY_SAFE_CONTEXT_WINDOW, REPEAT_NUDGE, REREAD_NUDGE,
    ReviewIntent, SECURITY_BROAD_SEARCH_NUDGE, SECURITY_SCOPE_NUDGE, TOOL_PROTOCOL_RETRY_NUDGE,
    TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE, answer_says_insufficient_evidence,
    bounded_review_repair_exhaustion_answer, classify_implementation_intent,
    classify_read_only_intent, concrete_review_answer_problem, deepen_review_nudge,
    implementation_missing_validation_nudge, implementation_text_tool_nudge,
    implementation_turn_prompt, inspected_insufficient_repair_limit, inspected_paths_for_prompt,
    inspection_sprawl_exhausted, insufficient_after_incomplete_security_search,
    insufficient_after_no_review_evidence, insufficient_after_repeated_search,
    insufficient_after_review_repair_template, insufficient_after_security_scope_overclaim,
    no_evidence_review_nudge, read_only_blocked_tool_result, read_only_blocks_tool,
    read_only_turn_prompt, should_bootstrap_gpu_training_estimator, should_deepen_review,
    should_nudge_gap_search_overclaim, should_nudge_inspection_sprawl,
    should_nudge_no_evidence_review, should_nudge_read_after_repeated_search,
    should_nudge_read_after_search_final, should_nudge_security_broad_search,
    should_nudge_security_scope, should_reject_review_repair_template,
    summarize_inspected_evidence_nudge,
};
use crate::transcript::NudgeKind;
use crate::verify::{Snapshot, Verifier, VerifyOutcome, stage_guidance};
use crate::{
    AUTO_KEEP_RECENT, FINALIZE_PROMPT, MAX_CHECKPOINTS, MAX_TOOL_PROTOCOL_RETRIES,
    PLAN_CONTINUE_NUDGE, SILENT_CONTINUE_NUDGE, TRUNCATED_TOOL_CALL_NUDGE, TRUNCATION_NUDGE,
    ToolCallEntry, TurnAttribution, TurnTelemetry, Ui, apply_plan_to_goal,
    partial_text_tool_call_start,
};

#[allow(clippy::too_many_arguments)]
fn build_turn_telemetry(
    verify_rounds: u32,
    recovery_retries: u32,
    repeat_nudges: u32,
    continue_nudges: u32,
    truncation_retries: u32,
    hit_step_cap: bool,
    stalled_unfinished: bool,
    stalled_repeating: bool,
    verify_attributions: &[hi_tools::Attribution],
    tool_calls: u32,
    max_concurrent_batch: u32,
    serial_runs: u32,
    tool_timeline: &[ToolCallEntry],
    evidence: &EvidenceTracker,
) -> TurnTelemetry {
    TurnTelemetry {
        verify_rounds,
        recovery_retries,
        repeat_nudges,
        continue_nudges,
        truncation_retries,
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
        file_reads: evidence.file_reads,
        targeted_searches: evidence.targeted_searches,
        listing_only: evidence.listing_only(),
        first_tool_kind: evidence.first_tool_kind().to_string(),
        discovery_depth: evidence.discovery_depth().to_string(),
        quality_repair_nudges: evidence.quality_repair_nudges,
    }
}

impl crate::Agent {
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
            let response = format!(
                "I cannot perform coding actions in {} mode because file-edit and shell tools are unavailable. Switch to `--tool-mode auto` or `--tool-mode required` to let me modify the workspace.",
                tool_mode_label(self.config.tool_mode)
            );
            ui.status(&format!(
                "tool mode {} does not allow file edits or shell commands for this turn",
                tool_mode_label(self.config.tool_mode)
            ));
            ui.assistant_text(&response);
            ui.assistant_end();
            self.messages.strip_trailing_nudges();
            self.persisted = self.persisted.min(self.messages.len());
            self.messages.push_user_or_fold(input);
            self.messages.push_assistant(vec![Content::Text(response)]);
            ui.turn_end(&self.usage_summary(&self.totals));
            self.persist()?;
            return Ok(());
        }
        // Snapshot the working tree before this turn touches anything, so `/undo`
        // can revert it. Best-effort: no-op outside a git repo.
        if let Some(sha) = hi_tools::checkpoint::create(std::path::Path::new(".")).await {
            self.checkpoints.push(sha);
            // Drop oldest checkpoints beyond the cap so the vec doesn't grow
            // without bound over a very long session. `/undo` only needs the
            // most recent few.
            if self.checkpoints.len() > MAX_CHECKPOINTS {
                self.checkpoints
                    .drain(0..self.checkpoints.len() - MAX_CHECKPOINTS);
            }
            if let Some(session) = self.session.as_mut()
                && let Err(err) = session.record_checkpoints(&self.checkpoints)
            {
                ui.status(&format!("(couldn't persist checkpoint refs: {err})"));
            }
        }

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
        let max_steps = self.config.max_steps.max(1);
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
            let preflight_calls = self
                .run_read_only_preflight(intent, ui, &mut evidence, &mut tool_timeline)
                .await;
            if preflight_calls > 0 {
                made_tool_call = true;
                sched_tool_calls = sched_tool_calls.saturating_add(preflight_calls);
                sched_serial_runs = sched_serial_runs.saturating_add(preflight_calls);
                sched_max_concurrent = sched_max_concurrent.max(1);
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
        let mut request_too_large_retried = false;
        let mut protocol_retries = 0u32;
        let mut protocol_text_fallbacks = 0u32;
        let mut text_tool_fallback_next = false;
        let mut force_text_answer_next = false;
        // Whether the turn ended because the model kept re-issuing the exact
        // same tool call through the whole repeat-nudge budget (drives the
        // incomplete notice and skips the finalization recap).
        let mut stalled_repeating = false;
        // Whether the turn ended without enough evidence for a read-only review.
        let mut stalled_unfinished = false;
        // Whether the turn was cut short by the per-turn step cap, so the
        // finalization recap is skipped (the work may be incomplete).
        let mut ended_at_cap = false;
        // Attributions parsed from the most recent verify failure — captured
        // here so they survive to turn end and can be flushed into telemetry.
        let mut last_verify_attributions: Vec<hi_tools::Attribution> = Vec::new();
        // Snapshot the turn baseline so verification only runs when the
        // workspace ends up changed. This catches `bash` edits too, while
        // skipping verify when a turn makes no net file changes.
        let turn_snapshot: Snapshot = self.snapshot_cached().await;
        // Snapshot from the most recent verify check. Reused at turn end to
        // avoid a second full tree walk when verify already took one.
        let mut verify_snapshot: Option<Snapshot> = None;

        if let Some(intent) = implementation_intent
            && !matches!(self.config.tool_mode, ToolMode::ChatOnly)
            && implementation_tracker.preferred_validation.is_none()
            && should_bootstrap_gpu_training_estimator(intent)
        {
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
                let sampling_retries = empty_retries.max(protocol_retries);
                let sampling_budget = if protocol_retries > empty_retries {
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
                    ui.status(&line);
                }

                let context_safety_window = read_only_intent
                    .is_some()
                    .then_some(READ_ONLY_SAFE_CONTEXT_WINDOW);
                self.elide_in_turn_context_if_needed(ui, context_safety_window);

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

                // After a continue-nudge, force this round to call a tool rather
                // than narrate again or come back empty. Only when tools are
                // freely available (Auto): never override an intentional
                // ChatOnly/ReadOnly restriction, and Required already forces.
                let tool_mode = if request_text_tool_fallback || request_text_answer {
                    ToolMode::ChatOnly
                } else if force_tools_next && self.config.tool_mode == ToolMode::Auto {
                    ToolMode::Required
                } else {
                    self.config.tool_mode
                };
                let tool_availability_mode = if request_text_tool_fallback || request_text_answer {
                    ToolMode::ChatOnly
                } else if read_only_intent.is_some()
                    && !matches!(self.config.tool_mode, ToolMode::ChatOnly)
                {
                    ToolMode::ReadOnly
                } else {
                    self.config.tool_mode
                };
                let request = ChatRequest {
                    model: self.config.model.clone(),
                    messages: self.messages.arc(),
                    tools: self.request_tools_for(tool_availability_mode),
                    max_tokens: self.config.max_tokens,
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
                let mut sink = |event: StreamEvent| match event {
                    StreamEvent::Text(text) => {
                        if buffer_read_only_review_text {
                            buffered_assistant_text.push_str(&text);
                        } else {
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
                    Ok(completion) => completion,
                    Err(err)
                        if provider_error_kind(&err)
                            == Some(ProviderErrorKind::RequestTooLarge) =>
                    {
                        let mut context_drop_persistence_failed = false;
                        if !request_too_large_retried {
                            match self.retry_after_request_too_large(input, turn_start, ui) {
                                Ok(true) => {
                                    request_too_large_retried = true;
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
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
                        );
                        let _ = self.persist();
                        let (kind, guidance) = crate::ui::classify_error(&err);
                        ui.turn_error(kind, &err.to_string(), guidance);
                        return Err(err);
                    }
                    Err(err)
                        if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                            && protocol_retries < MAX_TOOL_PROTOCOL_RETRIES =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        protocol_retries += 1;
                        if implementation_intent.is_some() || made_tool_call {
                            force_tools_next = true;
                        }
                        ui.status(&format!(
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
                            && protocol_text_fallbacks < 1 =>
                    {
                        ui.assistant_end();
                        self.add_error_usage(&err);
                        protocol_text_fallbacks += 1;
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
                    // outage, …) fall through to the abort below. Invalid tool turns
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
                        empty_retries += 1;
                        ui.status(&format!(
                            "⚠ the model's response didn't come through cleanly — \
                             retrying ({empty_retries}/{})",
                            self.config.max_empty_retries
                        ));
                        continue;
                    }
                    Err(err) => {
                        self.add_error_usage(&err);
                        if made_tool_call {
                            self.messages.strip_trailing_nudges();
                            let end_snapshot = self.snapshot_cached().await;
                            self.last_changed_files =
                                changed_files_between(&turn_snapshot, &end_snapshot);
                        } else {
                            self.truncate_messages(turn_start);
                        }
                        self.last_compat_fallbacks = compat_fallbacks.clone();
                        self.last_turn_telemetry = build_turn_telemetry(
                            verifier.round(),
                            empty_retries,
                            repeat_nudges,
                            continue_total_nudges,
                            truncation_total_retries,
                            ended_at_cap,
                            stalled_unfinished,
                            stalled_repeating,
                            &last_verify_attributions,
                            sched_tool_calls,
                            sched_max_concurrent,
                            sched_serial_runs,
                            &tool_timeline,
                            &evidence,
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
                ui.usage(
                    self.totals.input_tokens,
                    self.totals.output_tokens,
                    self.context_used,
                    self.config.context_window,
                );

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
                    ui.status(&format!(
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
                    ui.status(&format!(
                        "⚠ the model hit the output token limit {max} times — the task may be \
                         incomplete. /retry, or send 'continue'.",
                        max = self.config.max_truncation_retries,
                    ));
                    break false;
                }

                let calls: Vec<(String, String, String)> = if request_text_answer {
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
                let calls = if calls.is_empty() && !request_text_answer {
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
                    if read_only_intent.is_some()
                        && let Some(insufficient) = insufficient_after_repeated_search(&evidence)
                    {
                        stalled_unfinished = true;
                        ui.nudge(
                            "review repeated the same search without reading files; returning an insufficient-evidence answer",
                        );
                        ui.assistant_text(insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient.to_string())]);
                        break false;
                    }
                    if let Some(intent) = read_only_intent
                        && (evidence.saw_read || evidence.saw_search)
                    {
                        stalled_unfinished = true;
                        ui.nudge(
                            "review repeated the same command after inspection; returning a bounded evidence summary",
                        );
                        let insufficient = bounded_review_repair_exhaustion_answer(
                            intent,
                            &evidence,
                            "the model kept repeating the same tool call instead of producing findings tied to inspected evidence",
                        );
                        ui.assistant_text(&insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient)]);
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
                        let incomplete = "Implementation incomplete: the model inspected the workspace \
                        but kept re-reading files instead of making edits, so no file changes were made. \
                        /retry, or send 'continue' to resume.";
                        ui.nudge(
                            "implementation kept re-reading without editing; no file changes were made",
                        );
                        ui.assistant_text(incomplete);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(incomplete.to_string())]);
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
                // budget, hard-stop with a bounded-evidence summary. This is
                // the only guard that catches the "read 100 files, never
                // answer" failure mode — all review-quality guards fire only
                // on a final text answer, which never comes while the model
                // keeps issuing tool calls.
                if inspection_sprawl_exhausted(read_only_intent, &evidence, &calls) {
                    stalled_unfinished = true;
                    ui.nudge(
                        "review kept inspecting new files without producing findings; returning a bounded evidence summary",
                    );
                    let insufficient = bounded_review_repair_exhaustion_answer(
                        read_only_intent.expect("sprawl guard only fires on read-only review"),
                        &evidence,
                        "the model kept inspecting new files without producing findings tied to the evidence already gathered",
                    );
                    ui.assistant_text(&insufficient);
                    ui.assistant_end();
                    self.messages
                        .push_assistant(vec![Content::Text(insufficient)]);
                    break false;
                }
                if should_nudge_inspection_sprawl(read_only_intent, &evidence, &calls) {
                    evidence.inspection_sprawl_nudges =
                        evidence.inspection_sprawl_nudges.saturating_add(1);
                    force_text_answer_next = true;
                    ui.nudge(&format!(
                        "review inspected {} files/searches without answering; nudging it to produce findings",
                        evidence.inspection_count()
                    ));
                    self.messages
                        .push_assistant_text_only(std::mem::take(&mut completion.content));
                    self.messages
                        .push_nudge(NudgeKind::Continue, INSPECTION_SPRAWL_NUDGE);
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

                // Auto-recover from a content-less response — no tool calls and no
                // text, i.e. a flaky provider returning only reasoning or an empty
                // message. Silently re-run a few times before giving up, each
                // retry resampling hotter (see the temperature bump above). The
                // dead round isn't recorded, so each retry re-runs with the
                // original context.
                if calls.is_empty() && !has_text {
                    if empty_retries < self.config.max_empty_retries {
                        empty_retries += 1;
                        ui.status(&format!(
                            "⚠ the model returned no response — retrying ({empty_retries}/{})",
                            self.config.max_empty_retries
                        ));
                        continue;
                    }
                    ui.status(
                        "⚠ the model returned no response after retrying — try /retry, or \
                         /model to switch.",
                    );
                    break false;
                }
                // Real output this round — clear the retry counter so the
                // temperature bump is transient: a later, unrelated stall gets
                // its own budget rather than inheriting this one's elevation.
                empty_retries = 0;
                protocol_retries = 0;
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
                            if evidence.quality_repair_nudges < 2 {
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
                            let insufficient = bounded_review_repair_exhaustion_answer(
                                intent,
                                &evidence,
                                "the final answer kept proposing further inspection after the inspection-sprawl limit",
                            );
                            ui.assistant_text(&insufficient);
                            ui.assistant_end();
                            self.messages
                                .push_assistant(vec![Content::Text(insufficient)]);
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
                        let incomplete = "Implementation incomplete: the model inspected the workspace but did not make successful file changes, so I am not treating this as completed.";
                        ui.nudge("implementation still had no file changes after repair");
                        ui.assistant_text(incomplete);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(incomplete.to_string())]);
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
                        let incomplete = "Implementation incomplete: only project scaffolding or setup changes were detected, with no source/config edit implementing the requested behavior.";
                        ui.nudge(
                            "implementation still only had scaffold/setup changes after repair",
                        );
                        ui.assistant_text(incomplete);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(incomplete.to_string())]);
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
                        let incomplete = "Implementation incomplete: files were changed, but no successful validation command ran after the last change, so I am not treating this as completed.";
                        ui.nudge("implementation still lacked validation after repair");
                        ui.assistant_text(incomplete);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(incomplete.to_string())]);
                        break false;
                    }
                    if should_nudge_no_evidence_review(read_only_intent, &evidence, &assistant_text)
                    {
                        if evidence.quality_repair_nudges == 0 {
                            evidence.quality_repair_nudges += 1;
                            force_tools_next = true;
                            ui.nudge(
                                "review answer had no inspected evidence; nudging the model to inspect before answering",
                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                no_evidence_review_nudge(read_only_intent.expect("checked above")),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "review still had no inspected evidence after repair; returning an insufficient-evidence answer",
                        );
                        let insufficient = insufficient_after_no_review_evidence();
                        ui.assistant_text(insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient.to_string())]);
                        break false;
                    }
                    if let Some(intent) = read_only_intent
                        && evidence.saw_read
                        && answer_says_insufficient_evidence(&assistant_text)
                    {
                        if matches!(intent, ReviewIntent::Security)
                            && evidence.saw_search
                            && !evidence.security_search_complete()
                            && evidence.quality_repair_nudges < 3
                        {
                            evidence.quality_repair_nudges += 1;
                            force_tools_next = true;
                            ui.nudge(
                                "security review reported insufficient evidence before searching all required pattern families; nudging the model to broaden the search",
                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages
                                .push_nudge(NudgeKind::Continue, SECURITY_BROAD_SEARCH_NUDGE);
                            continue;
                        }
                        if evidence.quality_repair_nudges
                            < inspected_insufficient_repair_limit(intent)
                        {
                            evidence.quality_repair_nudges += 1;
                            force_text_answer_next = true;
                            ui.nudge(
                                "review reported insufficient evidence after inspection; nudging the model to summarize inspected files",
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
                        ui.status(
                            "review ended with generic insufficient-evidence text after inspection; returning a bounded evidence summary",
                        );
                        let insufficient = bounded_review_repair_exhaustion_answer(
                            intent,
                            &evidence,
                            "the final answer reported insufficient evidence after inspection instead of summarizing the inspected evidence",
                        );
                        ui.assistant_text(&insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient)]);
                        break false;
                    }
                    if should_reject_review_repair_template(read_only_intent, &assistant_text) {
                        stalled_unfinished = true;
                        ui.status(
                            "review answer was a generic repair template; returning an insufficient-evidence answer",
                        );
                        let insufficient = if let Some(intent) = read_only_intent
                            && (evidence.saw_read || evidence.saw_search)
                        {
                            bounded_review_repair_exhaustion_answer(
                                intent,
                                &evidence,
                                "the final answer was a generic review-repair template instead of findings tied to inspected files",
                            )
                        } else {
                            insufficient_after_review_repair_template().to_string()
                        };
                        ui.assistant_text(&insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient)]);
                        break false;
                    }
                    if should_deepen_review(read_only_intent, &evidence, &assistant_text) {
                        if evidence.quality_repair_nudges == 0 {
                            evidence.quality_repair_nudges += 1;
                            force_tools_next = true;
                            ui.nudge(
                                "review evidence was only a listing; nudging the model to inspect files or search results",
                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages.push_nudge(
                                NudgeKind::Continue,
                                deepen_review_nudge(read_only_intent.expect("checked above")),
                            );
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "review still had only listing evidence after repair; returning an insufficient-evidence answer",
                        );
                        let insufficient = "Insufficient evidence: only a directory listing was inspected, so I cannot make file-specific review findings without targeted searches or file reads.";
                        ui.assistant_text(insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient.to_string())]);
                        break false;
                    }
                    if should_nudge_read_after_search_final(
                        read_only_intent,
                        &evidence,
                        &assistant_text,
                    ) {
                        if evidence.quality_repair_nudges < 1 {
                            evidence.quality_repair_nudges += 1;
                            force_tools_next = true;
                            ui.nudge(
                                "review had targeted search but no file reads; nudging the model to read matching files",
                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages
                                .push_nudge(NudgeKind::Continue, READ_AFTER_SEARCH_NUDGE);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "review still had targeted search but no file reads after repair; returning an insufficient-evidence answer",
                        );
                        let insufficient = insufficient_after_repeated_search(&evidence)
                            .expect("search without read checked above");
                        ui.assistant_text(insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient.to_string())]);
                        break false;
                    }
                    if should_nudge_security_broad_search(
                        read_only_intent,
                        &evidence,
                        &assistant_text,
                    ) {
                        if evidence.quality_repair_nudges < 3 {
                            evidence.quality_repair_nudges += 1;
                            force_tools_next = true;
                            ui.nudge(
                                "security review missed required pattern families; nudging the model to broaden the search",
                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages
                                .push_nudge(NudgeKind::Continue, SECURITY_BROAD_SEARCH_NUDGE);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "security review still missed required pattern families after repair; returning an insufficient-evidence answer",
                        );
                        let reason = insufficient_after_incomplete_security_search(&evidence)
                            .expect("incomplete security search checked above");
                        let insufficient = if let Some(intent) = read_only_intent {
                            bounded_review_repair_exhaustion_answer(
                                intent,
                                &evidence,
                                reason.trim_start_matches("Insufficient evidence: "),
                            )
                        } else {
                            reason
                        };
                        ui.assistant_text(&insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient)]);
                        break false;
                    }
                    if should_nudge_security_scope(read_only_intent, &evidence, &assistant_text) {
                        if evidence.quality_repair_nudges < 4 {
                            evidence.quality_repair_nudges += 1;
                            ui.status(
                                "security answer overclaimed repo-wide safety; nudging the model to bound findings to evidence",
                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages
                                .push_nudge(NudgeKind::Continue, SECURITY_SCOPE_NUDGE);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.status(
                            "security answer still overclaimed after repair; returning an insufficient-evidence answer",
                        );
                        let insufficient = if let Some(intent) = read_only_intent {
                            bounded_review_repair_exhaustion_answer(
                                intent,
                                &evidence,
                                "the final answer made repo-wide all-clear claims broader than the inspected files and searches support",
                            )
                        } else {
                            insufficient_after_security_scope_overclaim().to_string()
                        };
                        ui.assistant_text(&insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient)]);
                        break false;
                    }
                    if should_nudge_gap_search_overclaim(
                        read_only_intent,
                        &evidence,
                        &assistant_text,
                    ) {
                        if evidence.quality_repair_nudges < 2 {
                            evidence.quality_repair_nudges += 1;
                            ui.nudge(
                                "gap answer contradicted search matches; nudging the model to bound claims to inspected evidence",
                            );
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages
                                .push_nudge(NudgeKind::Continue, GAP_SEARCH_OVERCLAIM_NUDGE);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(
                            "gap answer still overclaimed after search matches; returning a bounded evidence summary",
                        );
                        let insufficient = if let Some(intent) = read_only_intent {
                            bounded_review_repair_exhaustion_answer(
                                intent,
                                &evidence,
                                "the final answer claimed there were no TODO/FIXME or missing gaps even though targeted search returned matches",
                            )
                        } else {
                            "Insufficient evidence: the final answer overclaimed the absence of gaps despite search matches.".to_string()
                        };
                        ui.assistant_text(&insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient)]);
                        break false;
                    }
                    if let Some(problem) =
                        concrete_review_answer_problem(read_only_intent, &evidence, &assistant_text)
                    {
                        if evidence.quality_repair_nudges < 2 {
                            evidence.quality_repair_nudges += 1;
                            force_text_answer_next = true;
                            ui.nudge(problem.status());
                            self.messages
                                .push_assistant(std::mem::take(&mut completion.content));
                            self.messages
                                .push_nudge(NudgeKind::Continue, CONCRETE_REVIEW_NUDGE);
                            continue;
                        }

                        stalled_unfinished = true;
                        ui.nudge(problem.exhausted_status());
                        let insufficient = if let Some(intent) = read_only_intent {
                            bounded_review_repair_exhaustion_answer(
                                intent,
                                &evidence,
                                problem.reason(),
                            )
                        } else {
                            "Insufficient evidence: the inspected context was not tied to concrete file-specific findings, so I cannot present this as a completed review.".to_string()
                        };
                        ui.assistant_text(&insufficient);
                        ui.assistant_end();
                        self.messages
                            .push_assistant(vec![Content::Text(insufficient)]);
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
                            tool_timeline.push(ToolCallEntry {
                                tool: name.clone(),
                                path: hi_tools::target_path(name, arguments).unwrap_or_default(),
                                duration_ms: 0,
                                error: true,
                            });
                        }
                        break;
                    }
                    // If any ready call is bash, run it alone (streaming UI).
                    let bash_idx = ready.iter().copied().find(|&i| calls[i].1 == "bash");
                    if let Some(i) = bash_idx {
                        let (id, name, arguments) = &calls[i];
                        // Confirm edit if in --confirm-edits mode and this is a
                        // mutating tool. Bash is mutating but we let it run
                        // (the guard layer handles catastrophic ops).
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
                        evidence.record_success(name, arguments, &output.content);
                        implementation_tracker.record_tool_result(name, arguments, &output.content);
                        tool_timeline.push(ToolCallEntry {
                            tool: name.clone(),
                            path,
                            duration_ms,
                            error,
                        });
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
                        evidence.record_success(name, &calls[i].2, &output.content);
                        implementation_tracker.record_tool_result(
                            name,
                            &calls[i].2,
                            &output.content,
                        );
                        tool_timeline.push(ToolCallEntry {
                            tool: name.clone(),
                            path,
                            duration_ms: batch_duration_ms,
                            error,
                        });
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
            let outcome = verifier
                .check(&turn_snapshot, &mut self.snapshot_cache, ui)
                .await;
            // Capture the verify snapshot for turn-end reuse whenever the
            // verifier actually walked the tree (i.e. it didn't bail before
            // snapshotting). On a failure we drop it: the model is about to edit
            // again, so it's no longer current.
            if matches!(
                outcome,
                VerifyOutcome::Passed | VerifyOutcome::Failed { .. }
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
        let end_snapshot = match verify_snapshot.take() {
            Some(s) => s,
            None => self.snapshot_cached().await,
        };
        self.last_changed_files = changed_files_between(&turn_snapshot, &end_snapshot);
        self.last_compat_fallbacks = compat_fallbacks;
        // Flush the per-turn counters (otherwise discarded locals) into
        // telemetry so `--report` / the eval harness can diagnose the turn's
        // trajectory: how many verify rounds, recovery retries, nudges fired,
        // and where the last verify failure pointed.
        self.last_turn_telemetry = build_turn_telemetry(
            verifier.round(),
            empty_retries,
            repeat_nudges,
            continue_total_nudges,
            truncation_total_retries,
            ended_at_cap,
            stalled_unfinished,
            stalled_repeating,
            &last_verify_attributions,
            sched_tool_calls,
            sched_max_concurrent,
            sched_serial_runs,
            &tool_timeline,
            &evidence,
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

        // Cost warning: if the session has exceeded the configured spending
        // limit, surface a notice so the user can decide whether to continue.
        if let Some(limit) = self.config.max_cost_warn
            && let Some(cost) = self.cost_usd
            && cost >= limit
        {
            ui.status(&format!(
                "⚠ session cost ${cost:.4} has exceeded the --max-cost limit of ${limit:.2}"
            ));
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
                // Flush any partially-streamed recap text before the status
                // line, so it isn't left dangling in the UI's pending buffer.
                ui.assistant_end();
                ui.status(&format!("(couldn't generate the final summary: {err})"));
                return;
            }
        };

        self.add_usage(completion.usage);
        ui.usage(
            self.totals.input_tokens,
            self.totals.output_tokens,
            self.context_used,
            self.config.context_window,
        );

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

    /// Format a usage line. `usage` carries the cumulative in/out/total/cost;
    /// the context gauge instead uses `context_used` (the live conversation
    /// size), since cumulative input sums re-sent context across rounds and so
    /// isn't a measure of how full the window is.
    pub(crate) fn usage_summary(&self, usage: &hi_ai::Usage) -> String {
        // Cumulative session tokens, ↑ sent / ↓ received — these drive cost and
        // match the live working line. Abbreviated in the same units as the
        // context gauge below so the two never read as raw-vs-rounded.
        let mut summary = format!(
            "[↑{} ↓{}",
            humanize_count(usage.input_tokens),
            humanize_count(usage.output_tokens),
        );
        if usage.cache_read_tokens > 0 {
            summary.push_str(&format!(" ⟲{}", humanize_count(usage.cache_read_tokens)));
        }
        if let Some(cost) = self.cost_usd {
            summary.push_str(&format!(" · ${cost:.4}"));
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
