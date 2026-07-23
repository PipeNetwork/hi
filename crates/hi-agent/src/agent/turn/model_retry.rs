//! Provider stream handling: success path, retryable failures, fatal errors.

use anyhow::Result;
use hi_ai::{ChatRequest, Completion, ProviderErrorKind, Role, StreamEvent, provider_error_kind};

use crate::snapshot::changed_files_between;
use crate::steering::{
    EvidenceTracker, ImplementationIntent, TOOL_PROTOCOL_RETRY_NUDGE,
    TOOL_PROTOCOL_TEXT_FALLBACK_NUDGE,
};
use crate::transcript::NudgeKind;
use crate::verify::WorkspaceRepairVerifier;
use crate::{MAX_TOOL_PROTOCOL_RETRIES, ToolCallEntry, Ui};

use super::helpers::{build_turn_telemetry, effective_model_route};
use super::progress::ProgressTracker;
use super::retry::{
    MAX_PROVIDER_OVERLOAD_RETRIES, MAX_TRANSIENT_ROUTE_RETRIES, ReviewRepairState, TurnRetryState,
    delay_label, output_cap_retry_tokens, provider_error_is_backoff_retryable,
    provider_overload_retry_delay, transient_route_retry_delay,
};

pub(super) enum ProviderStreamResult {
    Ready {
        completion: Completion,
        buffered_assistant_text: String,
        buffer_read_only_review_text: bool,
        streamed_assistant_text: bool,
    },
    Continue,
    BreakInner(bool),
}

impl crate::Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_provider_stream(
        &mut self,
        request: ChatRequest,
        read_only_intent: Option<crate::steering::ReviewIntent>,
        implementation_intent: Option<ImplementationIntent>,
        request_max_tokens: u32,
        retry_state: &mut TurnRetryState,
        request_max_tokens_override: &mut Option<u32>,
        empty_retries: &mut u32,
        force_tools_next: &mut bool,
        text_tool_fallback_next: &mut bool,
        made_tool_call: bool,
        turn_start: &mut usize,
        turn_ledger_revision: u64,
        turn_snapshot: &Option<crate::verify::Snapshot>,
        input: &str,
        max_steps: u32,
        verifier: &WorkspaceRepairVerifier,
        repeat_nudges: u32,
        continue_total_nudges: u32,
        truncation_total_retries: u32,
        progress_tracker: &ProgressTracker,
        ended_at_cap: bool,
        stalled_unfinished: bool,
        stalled_repeating: bool,
        last_verify_attributions: &[hi_tools::Attribution],
        sched_tool_calls: u32,
        sched_max_concurrent: u32,
        sched_serial_runs: u32,
        tool_timeline: &[ToolCallEntry],
        evidence: &EvidenceTracker,
        review_repair: &ReviewRepairState,
        compat_fallbacks: &mut Vec<String>,
        effective_fallback_route: &mut Option<String>,
        ui: &mut dyn Ui,
    ) -> Result<ProviderStreamResult> {
        let buffer_read_only_review_text =
            read_only_intent.is_some() || implementation_intent.is_some();
        let mut buffered_assistant_text = String::new();
        let mut streamed_assistant_text = false;
        let btw_pending = &mut self.btw_answer_pending;
        let mut sink = |event: StreamEvent| match event {
            StreamEvent::Text(text) => {
                if buffer_read_only_review_text {
                    buffered_assistant_text.push_str(&text);
                } else {
                    streamed_assistant_text = true;
                    if *btw_pending {
                        *btw_pending = false;
                        ui.btw_answer(&text);
                    } else {
                        ui.assistant_text(&text);
                    }
                }
            }
            StreamEvent::Reasoning(text) => ui.assistant_reasoning(&text),
            StreamEvent::Status(text) => {
                if let Some(fallback) = text.strip_prefix("compat: ") {
                    compat_fallbacks.push(fallback.to_string());
                }
                if let Some(route) = text.rsplit_once("falling back to ").map(|(_, r)| r) {
                    *effective_fallback_route = Some(route.trim().to_string());
                }
                ui.status(&text);
            }
        };
        let provider_result = self.provider.stream(request, &mut sink).await;
        match provider_result {
            Ok(completion) => {
                retry_state.record_provider_success();
                Ok(ProviderStreamResult::Ready {
                    completion,
                    buffered_assistant_text,
                    buffer_read_only_review_text,
                    streamed_assistant_text,
                })
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
                retry_state.reset_request_id();
                let new_max = hi_ai::provider_output_cap_error(&err)
                    .and_then(|cap| output_cap_retry_tokens(request_max_tokens, cap))
                    .expect("guard checked retry tokens");
                *request_max_tokens_override = Some(new_max);
                ui.nudge(&format!(
                    "provider rejected the output budget; retrying this turn with max_tokens={new_max}"
                ));
                return Ok(ProviderStreamResult::Continue);
            }
            Err(err)
                if retry_state.provider_overload_retries < MAX_PROVIDER_OVERLOAD_RETRIES
                    && provider_error_is_backoff_retryable(&err) =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.provider_overload_retries += 1;
                let retry = retry_state.provider_overload_retries;
                let delay = provider_overload_retry_delay(retry, &err);
                let reason = if provider_error_kind(&err) == Some(ProviderErrorKind::RateLimit) {
                    "rate limited"
                } else {
                    "request did not complete"
                };
                ui.nudge(&format!(
                    "{reason}; retrying {} ({retry}/{MAX_PROVIDER_OVERLOAD_RETRIES})",
                    delay_label(delay)
                ));
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                return Ok(ProviderStreamResult::Continue);
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
                return Ok(ProviderStreamResult::Continue);
            }
            Err(err) if provider_error_kind(&err) == Some(ProviderErrorKind::RequestTooLarge) => {
                let mut context_drop_persistence_failed = false;
                if !retry_state.request_too_large_retried {
                    match self.retry_after_request_too_large(input, *turn_start, ui) {
                        Ok(true) => {
                            retry_state.request_too_large_retried = true;
                            retry_state.reset_request_id();
                            *turn_start = self.messages.len().saturating_sub(1);
                            return Ok(ProviderStreamResult::Continue);
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
                self.truncate_messages(*turn_start);
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
                self.reconcile_error_turn_changes(turn_ledger_revision)
                    .await?;
                self.emit_usage(ui);
                self.report.last_compat_fallbacks = compat_fallbacks.clone();
                self.report.last_turn_telemetry = build_turn_telemetry(
                    max_steps,
                    verifier.round(),
                    *empty_retries,
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
                self.report.last_effective_route =
                    effective_model_route(&self.config, effective_fallback_route.as_deref());
                return Err(err);
            }
            Err(err)
                if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                    && hi_ai::provider_error_retryable(&err) != Some(false)
                    && retry_state.protocol_retries < MAX_TOOL_PROTOCOL_RETRIES
                    && retry_state.protocol_failures_total < crate::MAX_TOOL_PROTOCOL_FAILURES =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.protocol_retries += 1;
                retry_state.protocol_failures_total += 1;
                retry_state.reset_request_id();
                let protocol_retries = retry_state.protocol_retries;
                if implementation_intent.is_some() || made_tool_call {
                    *force_tools_next = true;
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
                return Ok(ProviderStreamResult::Continue);
            }
            Err(err)
                if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                    && hi_ai::provider_error_retryable(&err) != Some(false)
                    && implementation_intent.is_some()
                    && retry_state.protocol_text_fallbacks < 1 =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.protocol_text_fallbacks += 1;
                retry_state.reset_request_id();
                *text_tool_fallback_next = true;
                *force_tools_next = false;
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
                return Ok(ProviderStreamResult::Continue);
            }
            Err(err)
                if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                    && hi_ai::provider_error_retryable(&err) != Some(false) =>
            {
                // Both the consecutive and cumulative invalid-tool-turn
                // budgets are spent. A model that alternates a valid tool
                // call with an invalid turn keeps resetting the consecutive
                // counter, so without the cumulative cap this nudge-and-retry
                // loop runs forever (spinning CPU, burning tokens). End the
                // turn instead so the driver/user regains control; on a
                // long-horizon drive the next turn resumes with a fresh budget.
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                ui.status(
                    "⚠ the model kept emitting invalid tool turns — ending the turn; /retry or continue to resume",
                );
                return Ok(ProviderStreamResult::BreakInner(false));
            }
            // A transient generation flake — a malformed/garbled stream or
            // an empty completion. Treat it like a content-less response:
            // flush, then silently re-run with hotter recovery sampling (a
            // fresh request, with its own transport retries) up to the same
            // budget, instead of failing the turn. Terminal errors (auth,
            // rate limits, ...) fall through to the abort below. Invalid tool turns
            // use the protocol-specific nudge path above.
            Err(err)
                if *empty_retries < self.config.loop_limits.max_empty_retries
                    && matches!(
                        provider_error_kind(&err),
                        Some(
                            ProviderErrorKind::MalformedStream | ProviderErrorKind::EmptyCompletion
                        )
                    ) =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                *empty_retries += 1;
                retry_state.reset_request_id();
                if made_tool_call {
                    self.nudge_after_post_tool_empty_response(
                        force_tools_next,
                        implementation_intent.is_some(),
                    );
                }
                ui.nudge(&format!(
                    "⚠ the model's response didn't come through cleanly — \
                     retrying ({empty_retries}/{})",
                    self.config.loop_limits.max_empty_retries
                ));
                return Ok(ProviderStreamResult::Continue);
            }
            Err(err) => {
                self.add_error_usage(&err);
                self.reconcile_error_turn_changes(turn_ledger_revision)
                    .await?;
                self.emit_usage(ui);
                if self.workspace.last_changed_files.is_empty()
                    && let Some(turn_snapshot) = turn_snapshot.as_ref()
                {
                    self.messages.strip_trailing_nudges();
                    if let Ok(end_snapshot) = self.snapshot_cached().await {
                        self.workspace.last_changed_files =
                            changed_files_between(turn_snapshot, &end_snapshot);
                    }
                }
                // With no model tool call, any concurrent workspace
                // change was external to this failed attempt. Preserve
                // it in the report, but never retain the failed user
                // prompt or retry guidance in conversation history.
                if !made_tool_call {
                    self.truncate_messages(*turn_start);
                }
                self.report.last_compat_fallbacks = compat_fallbacks.clone();
                self.report.last_turn_telemetry = build_turn_telemetry(
                    max_steps,
                    verifier.round(),
                    *empty_retries,
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
                self.report.last_effective_route =
                    effective_model_route(&self.config, effective_fallback_route.as_deref());
                return Err(err);
            }
        }
    }
}
