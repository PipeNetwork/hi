//! Provider stream handling: success path, retryable failures, fatal errors.

use std::collections::BTreeMap;

use anyhow::Result;
use hi_ai::{ChatRequest, Completion, ProviderErrorKind, Role, StreamEvent, provider_error_kind};
use hi_workflow::extract_partial_program_source;

use crate::snapshot::changed_files_between;
use crate::steering::{EvidenceTracker, ImplementationIntent, tool_protocol_retry_nudge};
use crate::transcript::NudgeKind;
use crate::verify::WorkspaceRepairVerifier;
use crate::{MAX_TOOL_PROTOCOL_RETRIES, Ui};

use super::helpers::{build_turn_telemetry, effective_model_route};
use super::progress::ProgressTracker;
use super::retry::{
    MAX_CAPACITY_RETRIES, MAX_PROVIDER_ROUTE_RETRIES, ReviewRepairState, TurnRetryState,
    capacity_retry_delay, delay_label, output_cap_retry_tokens,
    provider_error_is_backoff_retryable, provider_error_is_capacity_retryable,
    provider_overload_retry_delay, transient_route_retry_delay,
};
use super::speculation::SpeculationRegistry;

const COMPAT_FALLBACK_LIMIT: usize = 64;
const COMPAT_FALLBACK_PREFIX: usize = 62;
const COMPAT_FALLBACK_OMITTED_PREFIX: &str = "[diagnostic truncation: ";
pub(super) const COMPLETED_PLAN_EMPTY_RECAP_FALLBACK: &str = "The plan is complete and the successful tool results were retained. The provider did not return a final recap.";

fn record_compat_fallback(fallbacks: &mut Vec<String>, fallback: String) {
    if fallbacks.iter().any(|seen| seen == &fallback) {
        return;
    }
    if fallbacks.len() < COMPAT_FALLBACK_LIMIT {
        fallbacks.push(fallback);
        return;
    }

    let already_compacted = fallbacks
        .last()
        .and_then(|marker| marker.strip_prefix(COMPAT_FALLBACK_OMITTED_PREFIX))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|count| count.parse::<u64>().ok());
    let dropped = match already_compacted {
        Some(dropped) => {
            fallbacks[COMPAT_FALLBACK_PREFIX] = fallback;
            dropped.saturating_add(1)
        }
        None => {
            // The marker itself consumes one slot: retain the first 62 and the
            // newest event, and explicitly account for the two displaced rows.
            fallbacks.truncate(COMPAT_FALLBACK_PREFIX);
            fallbacks.push(fallback);
            fallbacks.push(String::new());
            2
        }
    };
    let last = fallbacks
        .last_mut()
        .expect("bounded compatibility trail always retains a marker slot");
    *last = format!(
        "{COMPAT_FALLBACK_OMITTED_PREFIX}{dropped} additional compatibility events omitted]"
    );
}

#[allow(
    clippy::large_enum_variant,
    reason = "this control-flow result is short-lived and boxing Completion would complicate ownership"
)]
pub(super) enum ProviderStreamResult {
    Ready {
        completion: Completion,
        buffered_assistant_text: String,
        buffer_read_only_review_text: bool,
        streamed_assistant_text: bool,
    },
    Continue,
    /// End the bounded model loop without turning a recoverable protocol
    /// exhaustion into an unhandled provider error. The outer turn still owns
    /// settlement, telemetry, and the deterministic user-facing closeout.
    BreakInner(bool),
}

impl crate::Agent {
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_provider_stream(
        &mut self,
        request: ChatRequest,
        tool_envelope: std::sync::Arc<hi_tools::envelope::ToolEnvelope>,
        read_only_intent: Option<crate::steering::ReviewIntent>,
        implementation_intent: Option<ImplementationIntent>,
        buffer_text_for_steering: bool,
        request_max_tokens: u32,
        request_no_progress_final_answer: bool,
        retry_state: &mut TurnRetryState,
        request_max_tokens_override: &mut Option<u32>,
        empty_retries: &mut u32,
        force_tools_next: &mut bool,
        text_tool_fallback_next: &mut bool,
        made_tool_call: bool,
        productive_tool_evidence: bool,
        provider_exhausted: &mut bool,
        turn_start: &mut usize,
        turn_ledger_revision: u64,
        turn_snapshot: &Option<crate::verify::Snapshot>,
        input: &str,
        max_steps: u32,
        verifier: &WorkspaceRepairVerifier,
        repeat_nudges: u32,
        continue_total_nudges: &mut u32,
        truncation_total_retries: u32,
        progress_tracker: &mut ProgressTracker,
        hit_step_cap: bool,
        hit_tool_cap: bool,
        last_verify_attributions: &[hi_tools::Attribution],
        sched_tool_calls: u32,
        sched_max_concurrent: u32,
        sched_serial_runs: u32,
        tool_timeline: &super::retention::ToolTimeline,
        speculation_registry: &SpeculationRegistry,
        evidence: &EvidenceTracker,
        review_repair: &ReviewRepairState,
        compat_fallbacks: &mut Vec<String>,
        effective_fallback_route: &mut Option<String>,
        ui: &mut dyn Ui,
    ) -> Result<ProviderStreamResult> {
        self.validate_and_audit_request_envelope(&request, &tool_envelope)?;
        let buffer_read_only_review_text = buffer_text_for_steering
            || read_only_intent.is_some()
            || implementation_intent.is_some();
        let mut buffered_assistant_text = String::new();
        let mut streamed_assistant_text = false;
        let mut program_delta_arguments = BTreeMap::<usize, String>::new();
        let mut program_delta_ids = BTreeMap::<usize, String>::new();
        let mut program_delta_names = BTreeMap::<usize, String>::new();
        let program_speculator = request
            .tools
            .iter()
            .any(|tool| tool.name == "run_program")
            .then(|| self.program_speculator(&tool_envelope));
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
            StreamEvent::WireAudit(audit) => {
                // Deliver wire evidence before execution or approval can
                // suspend settlement. The report copy is separately redacted.
                ui.provider_request(audit.as_ref());
                let mut value = serde_json::to_value(audit.as_ref()).unwrap_or_default();
                if let Some(object) = value.as_object_mut() {
                    object.remove("request_body");
                    super::model_request::attach_tool_envelope_audit(object, &tool_envelope);
                }
                self.report.last_turn_telemetry.record_wire_audit(value);
            }
            StreamEvent::Status(text) => {
                if let Some(fallback) = text.strip_prefix("compat: ") {
                    record_compat_fallback(compat_fallbacks, fallback.to_string());
                }
                if let Some(route) = text.rsplit_once("falling back to ").map(|(_, r)| r) {
                    *effective_fallback_route = Some(route.trim().to_string());
                }
                ui.status(&text);
            }
            StreamEvent::Warning(text) => ui.top_status(&text),
            StreamEvent::ToolCallDelta {
                index,
                id_delta,
                name_delta,
                arguments_delta,
            } => {
                if let Some(id_delta) = id_delta {
                    program_delta_ids
                        .entry(index)
                        .or_default()
                        .push_str(&id_delta);
                }
                if let Some(name_delta) = name_delta {
                    program_delta_names
                        .entry(index)
                        .or_default()
                        .push_str(&name_delta);
                }
                if !arguments_delta.is_empty() {
                    program_delta_arguments
                        .entry(index)
                        .or_default()
                        .push_str(&arguments_delta);
                }

                // The event is deliberately internal-only. Once a provider
                // has identified this call as run_program, a complete source
                // prefix is enough to launch safe literal reads in the
                // shadow executor while the rest of the program streams.
                if program_delta_names
                    .get(&index)
                    .is_some_and(|name| name == "run_program")
                    && let Some(program_speculator) = program_speculator.as_ref()
                    && let Some(arguments) = program_delta_arguments.get(&index)
                    && let Some(source) = extract_partial_program_source(arguments)
                {
                    let program_id = program_delta_ids
                        .get(&index)
                        .filter(|id| !id.is_empty())
                        .cloned()
                        .unwrap_or_else(|| format!("stream-program-{index}"));
                    program_speculator.launch(speculation_registry, &program_id, &source);
                }
            }
        };
        let protocol_retry_nudge =
            tool_protocol_retry_nudge(&request.tools, request.profile.tool_mode);
        let provider_result = self.provider.stream(request, &mut sink).await;
        // A retry, fatal provider error, or a completed non-program response
        // invalidates shadow work. Keeping it alive across a changed request
        // identity could leak a network/read task into the next round and let
        // a later exact claim observe stale context.
        if provider_result.as_ref().is_err()
            || provider_result.as_ref().is_ok_and(|completion| {
                !completion.content.iter().any(|content| {
                    matches!(
                        content,
                        hi_ai::Content::ToolCall { name, .. } if name == "run_program"
                    )
                })
            })
        {
            speculation_registry.cancel_all();
        }
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
                retry_state.record_recovery_attempt();
                let new_max = hi_ai::provider_output_cap_error(&err)
                    .and_then(|cap| output_cap_retry_tokens(request_max_tokens, cap))
                    .expect("guard checked retry tokens");
                *request_max_tokens_override = Some(new_max);
                ui.nudge(&format!(
                    "provider rejected the output budget; retrying this turn with max_tokens={new_max}"
                ));
                Ok(ProviderStreamResult::Continue)
            }
            Err(err)
                if retry_state.capacity_retries < MAX_CAPACITY_RETRIES
                    && provider_error_is_capacity_retryable(&err) =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.capacity_retries += 1;
                retry_state.record_recovery_attempt();
                let retry = retry_state.capacity_retries;
                let delay = capacity_retry_delay(retry, &err);
                let reason =
                    if provider_error_kind(&err) == Some(ProviderErrorKind::CapacityUnavailable) {
                        "capacity limited"
                    } else {
                        "provider overloaded"
                    };
                ui.nudge(&format!(
                    "{reason}; retrying {} ({retry}/{MAX_CAPACITY_RETRIES})",
                    delay_label(delay)
                ));
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                Ok(ProviderStreamResult::Continue)
            }
            Err(err)
                if retry_state.provider_route_retries < MAX_PROVIDER_ROUTE_RETRIES
                    && provider_error_is_backoff_retryable(&err) =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.provider_route_retries += 1;
                retry_state.record_recovery_attempt();
                let retry = retry_state.provider_route_retries;
                let delay = provider_overload_retry_delay(retry, &err);
                let reason = if provider_error_kind(&err) == Some(ProviderErrorKind::RateLimit) {
                    "rate limited"
                } else {
                    "request did not complete"
                };
                ui.nudge(&format!(
                    "{reason}; retrying {} ({retry}/{MAX_PROVIDER_ROUTE_RETRIES})",
                    delay_label(delay)
                ));
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                Ok(ProviderStreamResult::Continue)
            }
            Err(err)
                if retry_state.provider_route_retries < MAX_PROVIDER_ROUTE_RETRIES
                    && hi_ai::provider_route_error_is_retryable(&err)
                    && !provider_error_is_capacity_retryable(&err) =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.provider_route_retries += 1;
                retry_state.record_recovery_attempt();
                let retry = retry_state.provider_route_retries;
                let delay = transient_route_retry_delay(retry, &err);
                ui.nudge(&format!(
                    "request did not complete; retrying {} ({retry}/{MAX_PROVIDER_ROUTE_RETRIES})",
                    delay_label(delay)
                ));
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                Ok(ProviderStreamResult::Continue)
            }
            Err(err) if provider_error_kind(&err) == Some(ProviderErrorKind::RequestTooLarge) => {
                let mut context_recovery_persistence_failed = false;
                if !retry_state.request_too_large_compacted {
                    match self.retry_after_request_too_large_compact(ui) {
                        Ok(true) => {
                            retry_state.request_too_large_compacted = true;
                            retry_state.record_recovery_attempt();
                            *turn_start = self
                                .messages
                                .as_slice()
                                .iter()
                                .rposition(|message| message.role == Role::User)
                                .unwrap_or(1);
                            return Ok(ProviderStreamResult::Continue);
                        }
                        Ok(false) => {
                            retry_state.request_too_large_compacted = true;
                        }
                        Err(persist_err) => {
                            ui.status(&format!(
                                "couldn't persist compacted-context retry state: {persist_err}"
                            ));
                            context_recovery_persistence_failed = true;
                        }
                    }
                }
                if !context_recovery_persistence_failed && !retry_state.request_too_large_retried {
                    match self.retry_after_request_too_large(input, *turn_start, ui) {
                        Ok(true) => {
                            retry_state.request_too_large_retried = true;
                            retry_state.record_recovery_attempt();
                            *turn_start = self.messages.len().saturating_sub(1);
                            return Ok(ProviderStreamResult::Continue);
                        }
                        Ok(false) => {}
                        Err(persist_err) => {
                            ui.status(&format!(
                                "couldn't persist dropped-context retry state: {persist_err}"
                            ));
                            context_recovery_persistence_failed = true;
                        }
                    }
                }
                self.truncate_messages(*turn_start);
                if context_recovery_persistence_failed {
                    ui.status(
                        "request exceeds the provider limit, and prior context could not be \
                         safely compacted or dropped because the session boundary was not persisted; fix \
                         session storage or start a fresh/cleared session, then retry",
                    );
                } else {
                    ui.status(
                        "request still exceeds the provider limit after compacting and dropping \
                         prior context; shorten the prompt or attached input, then retry",
                    );
                }
                self.add_error_usage(&err);
                self.reconcile_error_turn_changes(turn_ledger_revision)
                    .await?;
                self.emit_usage(ui);
                self.report.last_compat_fallbacks = compat_fallbacks.clone();
                let model_telemetry = self.report.last_turn_telemetry.clone();
                let wire_audit = std::mem::take(&mut self.report.last_turn_telemetry.wire_audit);
                let requests = std::mem::take(&mut self.report.last_turn_telemetry.requests);
                let requests_dropped = self
                    .report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .requests_dropped;
                let compaction = std::mem::take(&mut self.report.last_turn_telemetry.compaction);
                let compaction_events_dropped = self
                    .report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .compaction_events_dropped;
                self.report.last_turn_telemetry = build_turn_telemetry(
                    max_steps,
                    verifier.round(),
                    *empty_retries,
                    repeat_nudges,
                    *continue_total_nudges,
                    truncation_total_retries,
                    progress_tracker,
                    hit_step_cap,
                    hit_tool_cap,
                    last_verify_attributions,
                    verifier.executions(),
                    verifier.executions_dropped(),
                    verifier.execution_count(),
                    verifier.successful_test_stage(),
                    sched_tool_calls,
                    sched_max_concurrent,
                    sched_serial_runs,
                    tool_timeline,
                    evidence,
                    review_repair,
                    &self.prefix_stability,
                );
                self.report.last_turn_telemetry.wire_audit = wire_audit;
                self.report.last_turn_telemetry.requests = requests;
                self.report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .requests_dropped = requests_dropped;
                self.report.last_turn_telemetry.compaction = compaction;
                self.report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .compaction_events_dropped = compaction_events_dropped;
                self.report
                    .last_turn_telemetry
                    .inherit_model_diagnostics(model_telemetry);
                let _ = self.persist();
                let (kind, guidance) = crate::ui::classify_error(&err);
                ui.turn_error(kind, &err.to_string(), guidance);
                self.report.last_effective_route =
                    effective_model_route(&self.config, effective_fallback_route.as_deref());
                Err(err)
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
                retry_state.record_recovery_attempt();
                let protocol_retries = retry_state.protocol_retries;
                if request_no_progress_final_answer {
                    // The live no-progress flag remains sticky in the caller,
                    // so retry the same ChatOnly request without appending the
                    // generic tool-protocol nudge. That nudge advertises tools
                    // and directly contradicts the existing "stop using
                    // tools" forced-final instruction.
                    *force_tools_next = false;
                    ui.nudge(&format!(
                        "⚠ the forced final answer was not a valid plain-text turn — retrying tool-free ({protocol_retries}/{MAX_TOOL_PROTOCOL_RETRIES})"
                    ));
                    return Ok(ProviderStreamResult::Continue);
                }
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
                    self.messages.push_user_or_fold(&protocol_retry_nudge);
                } else {
                    self.messages
                        .push_nudge(NudgeKind::Continue, &protocol_retry_nudge);
                }
                Ok(ProviderStreamResult::Continue)
            }
            Err(err)
                if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                    && hi_ai::provider_error_retryable(&err) != Some(false)
                    && !request_no_progress_final_answer
                    && implementation_intent.is_some()
                    && retry_state.protocol_text_fallbacks < 1 =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.protocol_text_fallbacks += 1;
                retry_state.record_recovery_attempt();
                *text_tool_fallback_next = true;
                *force_tools_next = false;
                ui.status(
                    "structured tool calls kept failing; falling back to plain-text tool-call parsing",
                );
                Ok(ProviderStreamResult::Continue)
            }
            Err(err)
                if provider_error_kind(&err) == Some(ProviderErrorKind::ToolProtocol)
                    && hi_ai::provider_error_retryable(&err) != Some(false) =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                retry_state.protocol_failures_total =
                    retry_state.protocol_failures_total.saturating_add(1);
                progress_tracker.record(
                    super::progress::ProgressKind::None,
                    "provider kept returning invalid tool turns",
                    None,
                );

                // A bounded change of approach is useful after the
                // consecutive protocol budget is spent. Reset only the
                // consecutive counter; the cumulative circuit breaker still
                // limits an alternating invalid/valid provider trajectory.
                if self.try_no_progress_recovery(
                    progress_tracker,
                    force_tools_next,
                    Some(continue_total_nudges),
                    ui,
                ) {
                    retry_state.protocol_retries = 0;
                    return Ok(ProviderStreamResult::Continue);
                }

                ui.status("invalid tool turns exhausted; ending this bounded turn");
                *provider_exhausted = true;
                Ok(ProviderStreamResult::BreakInner(false))
            }
            Err(err)
                if matches!(
                    provider_error_kind(&err),
                    Some(
                        ProviderErrorKind::MalformedStream
                            | ProviderErrorKind::EmptyCompletion
                            | ProviderErrorKind::QualityRejected
                    )
                ) =>
            {
                ui.assistant_end();
                self.add_error_usage(&err);
                self.emit_usage(ui);
                let empty_or_malformed = matches!(
                    provider_error_kind(&err),
                    Some(ProviderErrorKind::MalformedStream | ProviderErrorKind::EmptyCompletion)
                );
                if empty_or_malformed && *empty_retries < self.config.loop_limits.max_empty_retries
                {
                    *empty_retries += 1;
                    retry_state.record_recovery_attempt();
                    if made_tool_call && !request_no_progress_final_answer {
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
                progress_tracker.record(
                    super::progress::ProgressKind::None,
                    "provider returned no usable response",
                    None,
                );
                if self.try_no_progress_recovery(
                    progress_tracker,
                    force_tools_next,
                    Some(continue_total_nudges),
                    ui,
                ) {
                    *empty_retries = 0;
                    return Ok(ProviderStreamResult::Continue);
                }
                // If the checklist is complete and the turn has concrete
                // mutation/validation evidence, the missing prose recap is
                // optional. Pipe-compatible routes occasionally return an
                // accepted but empty stream at exactly this boundary. Preserve
                // the completed work and synthesize a truthful closeout instead
                // of misclassifying the entire turn as infrastructure failure.
                let completed_plan_with_tool_evidence = productive_tool_evidence
                    && !self.goals.plan().is_empty()
                    && !self.goals.plan_incomplete();
                if completed_plan_with_tool_evidence {
                    self.emit_assistant_text(ui, COMPLETED_PLAN_EMPTY_RECAP_FALLBACK);
                    ui.assistant_end();
                    self.messages.push_assistant(vec![hi_ai::Content::Text(
                        COMPLETED_PLAN_EMPTY_RECAP_FALLBACK.into(),
                    )]);
                    progress_tracker.no_progress_streak = 0;
                    progress_tracker.last_no_progress_reason.clear();
                    progress_tracker.record_final_answer();
                    ui.status(
                        "provider returned no final recap; closing from the completed plan and tool evidence",
                    );
                    return Ok(ProviderStreamResult::BreakInner(false));
                }
                // Once a tool has already produced a result, an exhausted
                // empty-response retry is a bounded settling condition, not a
                // provider failure. The outer turn still owns finalization
                // (including changed-file reconciliation and verification),
                // and returning through that path keeps a partially completed
                // implementation usable instead of surfacing a raw scripted
                // provider error or making the caller retry indefinitely.
                if made_tool_call {
                    ui.status("model returned no response after tool results; settling the turn");
                    *provider_exhausted = true;
                    return Ok(ProviderStreamResult::BreakInner(false));
                }
                self.reconcile_error_turn_changes(turn_ledger_revision)
                    .await?;
                if self.workspace.last_changed_files.is_empty()
                    && let Some(turn_snapshot) = turn_snapshot.as_ref()
                {
                    self.messages.strip_trailing_nudges();
                    if let Ok(end_snapshot) = self.snapshot_cached().await {
                        self.workspace.last_changed_files =
                            changed_files_between(turn_snapshot, &end_snapshot);
                    }
                }
                if !made_tool_call {
                    self.truncate_messages(*turn_start);
                }
                self.report.last_compat_fallbacks = compat_fallbacks.clone();
                let model_telemetry = self.report.last_turn_telemetry.clone();
                let wire_audit = std::mem::take(&mut self.report.last_turn_telemetry.wire_audit);
                let requests = std::mem::take(&mut self.report.last_turn_telemetry.requests);
                let requests_dropped = self
                    .report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .requests_dropped;
                let compaction = std::mem::take(&mut self.report.last_turn_telemetry.compaction);
                let compaction_events_dropped = self
                    .report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .compaction_events_dropped;
                self.report.last_turn_telemetry = build_turn_telemetry(
                    max_steps,
                    verifier.round(),
                    *empty_retries,
                    repeat_nudges,
                    *continue_total_nudges,
                    truncation_total_retries,
                    progress_tracker,
                    hit_step_cap,
                    hit_tool_cap,
                    last_verify_attributions,
                    verifier.executions(),
                    verifier.executions_dropped(),
                    verifier.execution_count(),
                    verifier.successful_test_stage(),
                    sched_tool_calls,
                    sched_max_concurrent,
                    sched_serial_runs,
                    tool_timeline,
                    evidence,
                    review_repair,
                    &self.prefix_stability,
                );
                self.report.last_turn_telemetry.wire_audit = wire_audit;
                self.report.last_turn_telemetry.requests = requests;
                self.report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .requests_dropped = requests_dropped;
                self.report.last_turn_telemetry.compaction = compaction;
                self.report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .compaction_events_dropped = compaction_events_dropped;
                self.report
                    .last_turn_telemetry
                    .inherit_model_diagnostics(model_telemetry);
                let _ = self.persist();
                let (kind, guidance) = crate::ui::classify_error(&err);
                ui.turn_error(kind, &err.to_string(), guidance);
                self.report.last_effective_route =
                    effective_model_route(&self.config, effective_fallback_route.as_deref());
                Err(err)
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
                let model_telemetry = self.report.last_turn_telemetry.clone();
                let wire_audit = std::mem::take(&mut self.report.last_turn_telemetry.wire_audit);
                let requests = std::mem::take(&mut self.report.last_turn_telemetry.requests);
                let requests_dropped = self
                    .report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .requests_dropped;
                let compaction = std::mem::take(&mut self.report.last_turn_telemetry.compaction);
                let compaction_events_dropped = self
                    .report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .compaction_events_dropped;
                self.report.last_turn_telemetry = build_turn_telemetry(
                    max_steps,
                    verifier.round(),
                    *empty_retries,
                    repeat_nudges,
                    *continue_total_nudges,
                    truncation_total_retries,
                    progress_tracker,
                    hit_step_cap,
                    hit_tool_cap,
                    last_verify_attributions,
                    verifier.executions(),
                    verifier.executions_dropped(),
                    verifier.execution_count(),
                    verifier.successful_test_stage(),
                    sched_tool_calls,
                    sched_max_concurrent,
                    sched_serial_runs,
                    tool_timeline,
                    evidence,
                    review_repair,
                    &self.prefix_stability,
                );
                self.report.last_turn_telemetry.wire_audit = wire_audit;
                self.report.last_turn_telemetry.requests = requests;
                self.report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .requests_dropped = requests_dropped;
                self.report.last_turn_telemetry.compaction = compaction;
                self.report
                    .last_turn_telemetry
                    .diagnostic_retention
                    .compaction_events_dropped = compaction_events_dropped;
                self.report
                    .last_turn_telemetry
                    .inherit_model_diagnostics(model_telemetry);
                let _ = self.persist();
                let (kind, guidance) = crate::ui::classify_error(&err);
                ui.turn_error(kind, &err.to_string(), guidance);
                self.report.last_effective_route =
                    effective_model_route(&self.config, effective_fallback_route.as_deref());
                Err(err)
            }
        }
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;

    #[test]
    fn compatibility_fallbacks_are_deduplicated_and_bounded() {
        let mut fallbacks = Vec::new();
        for index in 0..100 {
            record_compat_fallback(&mut fallbacks, format!("fallback-{index}"));
        }
        record_compat_fallback(&mut fallbacks, "fallback-0".into());

        assert_eq!(fallbacks.len(), COMPAT_FALLBACK_LIMIT);
        assert_eq!(fallbacks.first().map(String::as_str), Some("fallback-0"));
        assert_eq!(
            fallbacks.get(COMPAT_FALLBACK_PREFIX).map(String::as_str),
            Some("fallback-99")
        );
        assert!(
            fallbacks
                .last()
                .is_some_and(|marker| marker.contains("37 additional compatibility events omitted")),
            "exact dropped count is surfaced in the bounded diagnostic: {fallbacks:?}"
        );
    }
}
