//! One Model→(text)Steer iteration of the inner turn loop.

mod state;

pub(super) use state::{ModelRoundControl, ModelRoundState};
use state::{
    collapse_duplicate_inspection_calls, deepseek_thinking_for_round, merge_tool_call_channel,
};

use anyhow::Result;
use hi_ai::{ChatRequest, Content, RequestProfile, ToolMode};
use hi_tools::PlanStatus;

use crate::heuristics::{
    RECOVERY_SAMPLING, StallMode, looks_like_unfinished_step, parse_text_tool_calls,
    recovery_sampling, recovery_telemetry, textcall_id_offset,
};
use crate::steering::{
    BOOKKEEPING_REPOST_NUDGE, IMPLEMENTATION_NO_CHANGES_NUDGE, MUTATION_SAFE_CONTEXT_WINDOW,
    PLAN_REPOST_NUDGE, READ_AFTER_SEARCH_NUDGE, READ_ONLY_SAFE_CONTEXT_WINDOW, REPEAT_NUDGE,
    REREAD_NUDGE, SKIPPED_BOOKKEEPING_REPOST_RESULT, SKIPPED_COMPLETED_FILE_REREAD_RESULT,
    SKIPPED_PLAN_REPOST_RESULT, SKIPPED_REPEATED_CALL_RESULT, bash_call_waits,
    bash_no_progress_signature, implementation_text_tool_nudge, inspected_paths_for_prompt,
    inspection_round_cap_nudge, inspection_sprawl_exhausted, inspection_sprawl_nudge,
    should_nudge_inspection_sprawl, should_nudge_read_after_repeated_search,
};
use crate::transcript::NudgeKind;
use crate::{MAX_TOOL_PROTOCOL_RETRIES, TRUNCATED_TOOL_CALL_NUDGE, TRUNCATION_NUDGE, Ui};

use super::helpers::{build_turn_telemetry, effective_model_route};
use super::phase::TurnPhase;
use super::progress::{
    NO_PROGRESS_FINAL_ANSWER_NUDGE, ProgressKind, STEP_LIMIT_WRAP_UP_NUDGE,
    TOOL_LIMIT_WRAP_UP_NUDGE, forced_final_answer_is_unusable, no_progress_signature_for_calls,
};

/// `u32::MAX` is the public "unlimited" sentinel, not a finite cap that can be
/// reached. Keeping that distinction here also prevents the cap's extra wrap-up
/// request from overflowing the model-round counter at the sentinel boundary.
fn model_step_cap_reached(steps: u32, max_steps: u32) -> bool {
    max_steps != u32::MAX && steps >= max_steps
}

impl crate::Agent {
    /// Emit one assistant text chunk on the main task stream. `/btw` answers are
    /// handled off-band by `answer_btw_side_questions` and never pass through here.
    pub(crate) fn emit_assistant_text(&mut self, ui: &mut dyn Ui, text: &str) {
        ui.assistant_text(text);
    }

    pub(super) async fn run_model_round(
        &mut self,
        state: &mut ModelRoundState<'_>,
        ui: &mut dyn Ui,
    ) -> Result<ModelRoundControl> {
        let mut steps = *state.steps;
        let mut empty_retries = *state.empty_retries;
        let mut truncation_retries = *state.truncation_retries;
        let mut truncation_total_retries = *state.truncation_total_retries;
        let mut silent_continues = *state.silent_continues;
        let mut generic_completion_retries = *state.generic_completion_retries;
        let mut continue_total_nudges = *state.continue_total_nudges;
        let mut repeat_nudges = *state.repeat_nudges;
        let mut force_tools_next = *state.force_tools_next;
        let mut text_tool_fallback_next = *state.text_tool_fallback_next;
        let mut force_text_answer_next = *state.force_text_answer_next;
        let mut suppress_bookkeeping_tools_next = *state.suppress_bookkeeping_tools_next;
        let made_tool_call = *state.made_tool_call;
        let mut provider_exhausted = *state.provider_exhausted;
        let mut turn_start = *state.turn_start;
        let mut context_generation_seen = *state.context_generation_seen;
        let mut indexed_ledger_revision = *state.indexed_ledger_revision;
        let sched_tool_calls = *state.sched_tool_calls;
        let sched_max_concurrent = *state.sched_max_concurrent;
        let sched_serial_runs = *state.sched_serial_runs;
        let mut tool_schema_tokens = *state.tool_schema_tokens;
        let mut program_fallback_next = *state.program_fallback_next;
        let program_fallback_used = *state.program_fallback_used;
        let ended_at_cap = *state.ended_at_cap;
        let mut cap_wrap_up_requested = *state.cap_wrap_up_requested;
        let mut cap_kind = *state.cap_kind;
        let deepseek_strict_fallback_active = *state.deepseek_strict_fallback_active;
        let mut retry_state = std::mem::take(state.retry_state);
        let mut request_max_tokens_override = std::mem::take(state.request_max_tokens_override);
        let mut compat_fallbacks = std::mem::take(state.compat_fallbacks);
        let mut effective_fallback_route = std::mem::take(state.effective_fallback_route);
        let mut ranked_context_paths = std::mem::take(state.ranked_context_paths);
        let mut progress_tracker = std::mem::take(state.progress_tracker);
        let mut repeat_sampling_rounds = progress_tracker.repeat_sampling_rounds;
        let mut force_no_progress_final_answer_next =
            progress_tracker.force_no_progress_final_answer_next;
        let mut prev_added_no_evidence = progress_tracker.prev_added_no_evidence;
        let mut prev_call_sig = std::mem::take(&mut progress_tracker.prev_call_sig);
        let mut evidence = std::mem::take(state.evidence);
        let mut implementation_tracker = std::mem::take(state.implementation_tracker);
        let mut review_repair = std::mem::take(state.review_repair);
        let last_verify_attributions = std::mem::take(state.last_verify_attributions);
        let tool_timeline = std::mem::take(state.tool_timeline);
        let mut advertised_tool_names = std::mem::take(state.advertised_tool_names);
        let turn_snapshot = std::mem::take(state.turn_snapshot);
        let max_steps = state.max_steps;
        let max_tool_calls = self.config.loop_limits.max_tool_calls;
        let context_task = state.context_task;
        let task_intent = state.task_intent;
        let repository_context_enabled = state.repository_context_enabled;
        let turn_ledger_revision = state.turn_ledger_revision;
        let read_only_intent = state.read_only_intent;
        let implementation_intent = state.implementation_intent;
        let read_only_inspection_cap = state.read_only_inspection_cap;
        let expected_mutation = state.expected_mutation;
        let requested_validation = state.requested_validation;
        let input = state.input;
        let _user_prompt_tokens = state.user_prompt_tokens;
        let inspection_sprawl_intent = state.inspection_sprawl_intent;
        let verifier = state.verifier;

        let result = async {
        self.set_turn_phase(TurnPhase::Model);
        // Reaching the cap grants ONE tool-free wrap-up round so the model can
        // report where it left the work, instead of the turn dying mid-flight
        // with no final answer. The sticky flag makes the second hit terminal.
        let mut request_cap_wrap_up = false;
        let mut request_tool_cap_wrap_up = false;
        let step_cap_reached = model_step_cap_reached(steps, max_steps);
        let tool_cap_reached = self
            .config
            .loop_limits
            .tool_call_cap_reached(sched_tool_calls);
        if step_cap_reached || tool_cap_reached {
            if cap_wrap_up_requested {
                return Ok(ModelRoundControl::BreakInner(true));
            }
            cap_wrap_up_requested = true;
            request_cap_wrap_up = true;
            request_tool_cap_wrap_up = tool_cap_reached && !step_cap_reached;
            cap_kind = Some(match (step_cap_reached, tool_cap_reached) {
                (true, true) => crate::domain::TurnCapKind::Both,
                (true, false) => crate::domain::TurnCapKind::Step,
                (false, true) => crate::domain::TurnCapKind::Tool,
                (false, false) => unreachable!("cap branch requires a reached cap"),
            });
            let (status, nudge) = if request_tool_cap_wrap_up {
                (
                    format!(
                        "reached tool-call limit ({max_tool_calls}); asking for a final wrap-up before stopping"
                    ),
                    TOOL_LIMIT_WRAP_UP_NUDGE,
                )
            } else {
                (
                    format!(
                        "reached step limit ({max_steps}); asking for a final wrap-up before stopping"
                    ),
                    STEP_LIMIT_WRAP_UP_NUDGE,
                )
            };
            ui.nudge(&status);
            // A prior recovery/steering branch may already have left a
            // synthetic user nudge at the tail. Fold the cap instruction into
            // that user turn so the wrap-up request cannot create consecutive
            // user messages and trip the provider-safety invariant.
            self.messages
                .push_nudge_or_fold(NudgeKind::Continue, nudge);
        }
        steps = steps.saturating_add(1);

        // Mid-turn input: `/btw` side questions are answered off-band (bounded
        // read-only tool loop, concurrent with this round). Remaining plain
        // messages are genuine steering, injected at this safe boundary — the
        // prior round's tool calls are all resolved — so the folding nudge push
        // keeps provider alternation valid. Also fold any finished side-job UI.
        self.poll_btw_jobs(ui).await;
        // Keep live snapshot/transcript fresh for immediate BtwDispatcher::ask.
        self.arm_btw_dispatcher();
        let interjected = self.interjections.drain();
        if !interjected.is_empty() {
            let steering = self.answer_btw_side_questions(interjected, ui).await;
            let steer_count = steering.len();
            for message in steering {
                self.messages.push_nudge_or_fold(
                    NudgeKind::Interjection,
                    format!(
                        "The user sent this message while you were working — take it into account now:\n{message}"
                    ),
                );
            }
            if steer_count > 0 {
                ui.status(&format!(
                    "✉ received {steer_count} message(s) from you mid-turn — factoring them in"
                ));
            }
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
            (StallMode::Repeat, self.config.loop_limits.max_repeat_nudges)
        } else if retry_state.protocol_retries > empty_retries {
            (StallMode::Empty, MAX_TOOL_PROTOCOL_RETRIES)
        } else {
            (StallMode::Empty, self.config.loop_limits.max_empty_retries)
        };
        let (temperature, top_p, frequency_penalty) = recovery_sampling(
            sampling_retries,
            self.config.routing.temperature,
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

        let context_safety_window = if read_only_intent.is_some() {
            Some(READ_ONLY_SAFE_CONTEXT_WINDOW)
        } else {
            Some(MUTATION_SAFE_CONTEXT_WINDOW)
        };
        self.elide_in_turn_context_if_needed(ui, context_safety_window);
        evidence.reopen_elided_reads(self.messages.as_slice());

        self.refresh_active_task_context(
            context_task,
            repository_context_enabled,
            false,
            turn_ledger_revision,
            &mut ranked_context_paths,
            &mut context_generation_seen,
            &mut indexed_ledger_revision,
        )
        .await?;

        // The transcript we're about to send must be provider-safe (every
        // tool_use answered, no consecutive user/assistant turns, and visible
        // assistant content). Repair known legacy states once, then fail closed
        // if anything remains instead of sending a request the provider will
        // reject.
        if let Err(err) = self.messages.validate_and_repair_for_provider() {
            anyhow::bail!(
                "Transcript validation failed before provider send; recovery did not resolve it: {err}"
            );
        }

        let request_text_tool_fallback = text_tool_fallback_next;
        text_tool_fallback_next = false;
        let request_text_answer = force_text_answer_next;
        force_text_answer_next = false;
        let request_no_progress_final_answer = force_no_progress_final_answer_next;
        if request_text_answer || request_no_progress_final_answer {
            progress_tracker.record_forced_final_answer_attempt();
        }
        // Keep the no-progress final-answer request sticky until it produces a
        // usable response. A provider may accept the request but return an
        // empty/malformed stream; clearing the flag before that response made
        // the retry tool-capable again and allowed the generic post-tool nudge
        // to contradict the existing "stop using tools" instruction.

        // After a continue-nudge, force this round to call a tool rather
        // than narrate again or come back empty. Only when tools are
        // freely available (Auto): never override an intentional
        // ChatOnly/ReadOnly restriction, and Required already forces.
        // Wrap-up is ChatOnly *policy* (`tool_choice: none`); the catalog
        // stays the working-round set so the tool-prefix cache still hits.
        let wrapping_up = request_text_tool_fallback
            || request_text_answer
            || request_no_progress_final_answer
            || request_cap_wrap_up;
        let tool_mode = if wrapping_up {
            ToolMode::ChatOnly
        } else if force_tools_next && self.config.routing.tool_mode == ToolMode::Auto {
            ToolMode::Required
        } else {
            self.config.routing.tool_mode
        };
        let tool_availability_mode = if read_only_intent.is_some()
            && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
        {
            ToolMode::ReadOnly
        } else {
            self.config.routing.tool_mode
        };
        let requested_request_max_tokens =
            request_max_tokens_override.unwrap_or(self.config.routing.max_tokens);
        let mut request_tools = self.request_tools_for(tool_availability_mode);
        if program_fallback_next {
            program_fallback_next = false;
            request_tools = request_tools
                .iter()
                .filter(|tool| tool.name != "run_program")
                .cloned()
                .collect::<Vec<_>>()
                .into();
            ui.status("retrying with ordinary structured tools after run_program failure");
        }
        if suppress_bookkeeping_tools_next {
            suppress_bookkeeping_tools_next = false;
            request_tools = super::model_request::apply_bookkeeping_suppress(request_tools, true);
        }
        let request_tool_schema_tokens = super::model_request::note_advertised_tools(
            &request_tools,
            &mut advertised_tool_names,
            &mut tool_schema_tokens,
        );
        // Destructive context recovery rebuilds the current turn from `input`.
        // A forced-final instruction normally lives in a later synthetic user
        // nudge, so rebuilding from the raw input alone silently loses the
        // instruction even though the sticky ChatOnly flag survives. Carry the
        // policy in the recovery seed as well; ordinary requests remain
        // byte-for-byte unchanged.
        let forced_final_recovery_input = request_no_progress_final_answer
            .then(|| format!("{input}\n\n{NO_PROGRESS_FINAL_ANSWER_NUDGE}"));
        let recovery_input = forced_final_recovery_input.as_deref().unwrap_or(input);
        let context_preflight = match self.ensure_request_fits_context(
            recovery_input,
            turn_start,
            requested_request_max_tokens,
            request_tool_schema_tokens,
            context_safety_window,
            ui,
        ) {
            Ok(context_preflight) => context_preflight,
            Err(err) => {
                self.reconcile_error_turn_changes(turn_ledger_revision).await?;
                self.truncate_messages(turn_start);
                self.add_error_usage(&err);
                self.emit_usage(ui);
                self.report.last_compat_fallbacks = compat_fallbacks.clone();
                let model_telemetry = self.report.last_turn_telemetry.clone();
                self.report.last_turn_telemetry = build_turn_telemetry(
                    max_steps,
                    verifier.round(),
                    empty_retries,
                    repeat_nudges,
                    continue_total_nudges,
                    truncation_total_retries,
                    &progress_tracker,
                    ended_at_cap
                        && matches!(
                            cap_kind,
                            Some(
                                crate::domain::TurnCapKind::Step
                                    | crate::domain::TurnCapKind::Both
                            )
                        ),
                    ended_at_cap
                        && matches!(
                            cap_kind,
                            Some(
                                crate::domain::TurnCapKind::Tool
                                    | crate::domain::TurnCapKind::Both
                            )
                        ),
                    &last_verify_attributions,
                    verifier.executions(),
                    verifier.executions_dropped(),
                    verifier.execution_count(),
                    verifier.successful_test_stage(),
                    sched_tool_calls,
                    sched_max_concurrent,
                    sched_serial_runs,
                    &tool_timeline,
                    &evidence,
                    &review_repair,
                    &self.prefix_stability,
                );
                self.report
                    .last_turn_telemetry
                    .inherit_model_diagnostics(model_telemetry);
                let _ = self.persist();
                let (kind, guidance) = crate::ui::classify_error(&err);
                ui.turn_error(kind, &err.to_string(), guidance);
                self.report.last_effective_route = effective_model_route(
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
            context_task,
            repository_context_enabled,
            false,
            turn_ledger_revision,
            &mut ranked_context_paths,
            &mut context_generation_seen,
            &mut indexed_ledger_revision,
        )
        .await?;
        let request_max_tokens = context_preflight.max_tokens;
        if request_max_tokens != requested_request_max_tokens {
            request_max_tokens_override = Some(request_max_tokens);
        }
        let advertised_tool_specs = request_tools.clone();
        // Prompt-cache health: measure whether this request extends the
        // previous one append-only (cacheable prefix) or rewrote history.
        self.prefix_stability
            .record_request(self.messages.as_slice(), &request_tools);
        let request = ChatRequest {
            model: self.config.routing.model.clone(),
            request_id: Some(retry_state.request_id()),
            retry_attempt: retry_state.request_attempt(),
            user_turn: true,
            canonical_objective: Some(context_task.to_string()),
            messages: self.messages.arc(),
            tools: request_tools,
            max_tokens: request_max_tokens,
            temperature,
            top_p: self.config.routing.top_p.or(top_p),
            frequency_penalty,
            thinking_budget: self.config.routing.thinking_budget,
            // Repeated verification failure escalates one effort step: the
            // cheap attempt already failed, so spend more thinking on repair.
            reasoning_effort: if self.repair_effort_escalated {
                Some(
                    self.config
                        .routing
                        .reasoning_effort
                        .map_or(hi_ai::ReasoningEffort::High, |effort| effort.next_higher()),
                )
            } else {
                self.config.routing.reasoning_effort
            },
            profile: RequestProfile {
                compat: self.config.routing.compat,
                tool_mode,
                stream_usage: None,
                deepseek_compat: self.config.routing.deepseek_compat,
                deepseek_strict: if deepseek_strict_fallback_active {
                    Some(false)
                } else {
                    None
                },
                // Inspection: thinking off so Flash calls tools instead of
                // writing a CoT essay from preflight. Wrap-up: thinking on.
                deepseek_thinking: deepseek_thinking_for_round(
                    read_only_intent,
                    request_text_answer,
                    request_cap_wrap_up,
                    empty_retries,
                ),
                output_token_parameter: self.config.routing.output_token_parameter,
            },
        };

        self.report
            .last_turn_telemetry
            .record_request_census(crate::census_messages(self.messages.as_slice()));
        self.report.last_turn_telemetry.model_requests = self
            .report
            .last_turn_telemetry
            .model_requests
            .saturating_add(1);
        self.report.last_turn_telemetry.reasoning_requested |= request.thinking_budget.is_some()
            || request.reasoning_effort.is_some()
            || request.profile.deepseek_thinking == Some(true);
        self.report.last_turn_telemetry.reasoning_replayed |= request
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .any(|content| matches!(content, Content::Thinking { .. }));
        self.report.last_turn_telemetry.reasoning_signature_replayed |= request
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .any(|content| {
                matches!(content, Content::Thinking { signature: Some(_), .. })
            });

        let stream_result = self
            .handle_provider_stream(
                request,
                read_only_intent,
                implementation_intent,
                task_intent == crate::TaskIntent::ReadOnly
                    || expected_mutation
                    || requested_validation
                    || crate::task_contract::prompt_is_direct_question(context_task),
                request_max_tokens,
                request_no_progress_final_answer,
                &mut retry_state,
                &mut request_max_tokens_override,
                &mut empty_retries,
                &mut force_tools_next,
                &mut text_tool_fallback_next,
                made_tool_call,
                implementation_tracker.mutation_seen || implementation_tracker.validation_seen,
                &mut provider_exhausted,
                &mut turn_start,
                turn_ledger_revision,
                &turn_snapshot,
                recovery_input,
                max_steps,
                verifier,
                repeat_nudges,
                &mut continue_total_nudges,
                truncation_total_retries,
                &mut progress_tracker,
                ended_at_cap
                    && matches!(
                        cap_kind,
                        Some(
                            crate::domain::TurnCapKind::Step
                                | crate::domain::TurnCapKind::Both
                        )
                    ),
                ended_at_cap
                    && matches!(
                        cap_kind,
                        Some(
                            crate::domain::TurnCapKind::Tool
                                | crate::domain::TurnCapKind::Both
                        )
                    ),
                &last_verify_attributions,
                sched_tool_calls,
                sched_max_concurrent,
                sched_serial_runs,
                &tool_timeline,
                state.speculation_registry,
                &evidence,
                &review_repair,
                &mut compat_fallbacks,
                &mut effective_fallback_route,
                ui,
            )
            .await?;
        let (mut completion, buffered_assistant_text, buffer_read_only_review_text, streamed_assistant_text) =
            match stream_result {
                super::model_retry::ProviderStreamResult::Ready {
                    completion,
                    buffered_assistant_text,
                    buffer_read_only_review_text,
                    streamed_assistant_text,
                } => (
                    completion,
                    buffered_assistant_text,
                    buffer_read_only_review_text,
                    streamed_assistant_text,
                ),
                super::model_retry::ProviderStreamResult::Continue => {
                    return Ok(ModelRoundControl::Continue);
                }
                super::model_retry::ProviderStreamResult::BreakInner(hit) => {
                    return Ok(ModelRoundControl::BreakInner(hit));
                }
            };
        self.report.last_turn_telemetry.accepted_completions = self
            .report
            .last_turn_telemetry
            .accepted_completions
            .saturating_add(1);
        self.report.last_turn_telemetry.last_stop_reason = completion.stop_reason.clone();
        self.report.last_turn_telemetry.reasoning_received |= completion
            .content
            .iter()
            .any(|content| matches!(content, Content::Thinking { .. }));
        self.report.last_turn_telemetry.tool_call_channel = merge_tool_call_channel(
            &self.report.last_turn_telemetry.tool_call_channel,
            completion.tool_call_channel.label(),
        );
        if completion.refusal.is_some() || completion.stop_reason.as_deref() == Some("refusal") {
            self.report.last_turn_telemetry.refusal_source = Some(
                if completion.refusal.is_some() {
                    "structured_provider_signal"
                } else {
                    "finish_reason"
                }
                .to_string(),
            );
        }
        let mut buffered_assistant_text = buffered_assistant_text;
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
        // "picking up where it stopped" on the next prompt). This uses a
        // *dedicated* truncation policy (separate from `empty_retries`). The
        // ordinary policy is unlimited because every truncated response is
        // valid productive output; bounded integrations can still opt into a
        // finite retry count.
        let truncated = matches!(
            completion.stop_reason.as_deref(),
            Some("length" | "max_tokens")
        );
        if truncated
            && self
                .config
                .loop_limits
                .truncation_retry_available(truncation_retries)
        {
            truncation_retries = truncation_retries.saturating_add(1);
            truncation_total_retries = truncation_total_retries.saturating_add(1);
            ui.nudge(&format!(
                "⚠ the model hit the output token limit — continuing ({truncation_retries}/{})",
                crate::config::repair_limit_label(
                    self.config.loop_limits.max_truncation_retries
                )
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
                    || self.goals.plan_incomplete()
                    || self
                        .goals
                        .structured
                        .as_ref()
                        .is_some_and(crate::goal::Goal::should_auto_drive)
                    || looks_like_unfinished_step(&truncated_text));
            if (partial_tool_call || active_tool_work)
                && self.config.routing.tool_mode == ToolMode::Auto
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
            return Ok(ModelRoundControl::Continue);
        }
        // Truncation budget exhausted: the model kept hitting the output
        // token cap through the whole retry budget. Record the truncated
        // output (stripping partial tool calls, as above) and warn the
        // user — the output may be partial. Don't silently end the turn
        // on a half-finished output without surfacing what happened.
        if truncated {
            self.clean_text_tool_calls_from_content(&mut completion.content);
            self.messages
                .push_assistant_text_only(std::mem::take(&mut completion.content));
            ui.status(
                "⚠ output truncated — the model remained incomplete after the retry budget; stopping with the partial response",
            );
            if self.try_no_progress_recovery(
                &mut progress_tracker,
                &mut force_tools_next,
                Some(&mut continue_total_nudges),
                ui,
            ) {
                return Ok(ModelRoundControl::Continue);
            }
            return Ok(ModelRoundControl::BreakInner(false));
        }
        // A public RSI response is terminal, not a local planning round to nudge.
        if completion.stop_reason.as_deref() == Some("rsi_remote_completed") {
            let answer = completion
                .content
                .iter()
                .filter_map(|content| match content {
                    Content::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !answer.trim().is_empty()
                && (buffer_read_only_review_text || !streamed_assistant_text)
            {
                self.emit_assistant_text(ui, &answer);
                ui.assistant_end();
            }
            self.messages
                .push_assistant(std::mem::take(&mut completion.content));
            progress_tracker.record_final_answer();
            return Ok(ModelRoundControl::BreakInner(false));
        }

        let calls: Vec<(String, String, String)> =
            if request_text_answer || request_no_progress_final_answer || request_cap_wrap_up {
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
            && !request_cap_wrap_up
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

        // A plain-text tool fallback is useful only if the model actually
        // emits a parseable call. The old one-shot flag was cleared before the
        // response arrived, so a narrative answer ("Let me read the file")
        // silently abandoned recovery and fell into the generic no-tool path.
        // Keep the same fallback active for one bounded correction.
        let text_fallback_narration = request_text_tool_fallback
            && calls.is_empty()
            && looks_like_unfinished_step(
                &completion
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        if text_fallback_narration {
            const MAX_TEXT_TOOL_FALLBACK_MISSES: u32 = 1;
            if retry_state.text_tool_fallback_misses < MAX_TEXT_TOOL_FALLBACK_MISSES {
                retry_state.text_tool_fallback_misses += 1;
                retry_state.record_recovery_attempt();
                continue_total_nudges = continue_total_nudges.saturating_add(1);
                text_tool_fallback_next = true;
                force_tools_next = false;
                self.messages.push_assistant(vec![Content::Text(
                    "[plain-text tool retry: no executable tool call was emitted]".into(),
                )]);
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    implementation_text_tool_nudge(
                        "The previous fallback response contained no executable tool call. Do not narrate or promise the next action; emit the required call now.",
                    ),
                );
                ui.nudge(
                    "plain-text tool fallback returned narration instead of a call; retrying once",
                );
                return Ok(ModelRoundControl::Continue);
            }
        } else if request_text_tool_fallback && !calls.is_empty() {
            // A successfully promoted call proves the fallback channel works;
            // a later implementation gate gets its own bounded miss recovery.
            retry_state.text_tool_fallback_misses = 0;
        }

        let (calls, duplicate_inspection_calls) =
            collapse_duplicate_inspection_calls(&mut completion.content, calls);
        if duplicate_inspection_calls > 0 {
            ui.nudge(&format!(
                "the model emitted {duplicate_inspection_calls} duplicate read-only tool call{} in one response; executing the first occurrence only",
                if duplicate_inspection_calls == 1 { "" } else { "s" }
            ));
        }

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
        // rather than looping indefinitely.
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
        // can otherwise loop indefinitely without ever re-issuing an
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
        // broader search has surfaced something to re-examine). Extra pages
        // of a complete file are no-new-evidence; a later page of a still-
        // truncated file counts without an arbitrary page limit. Once the turn has made a successful
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
        let repeat_budget_available = repeat_nudges < self.config.loop_limits.max_repeat_nudges;
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
                    } else if name == "read" && evidence.rereads_only_completed_files(&calls) {
                        SKIPPED_COMPLETED_FILE_REREAD_RESULT
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
                let no_progress_reason = if all_plan_reposts {
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
                // Never force a chat-only "final answer" while a mutation
                // turn still has no productive tool evidence. That path exists
                // for inspection stalls where the model already has evidence
                // to summarize; on an edit request it just ends the turn with
                // zero file changes. Once work has landed, a bounded recap is
                // valid again.
                let mutation_evidence_seen = implementation_tracker.mutation_seen
                    || implementation_tracker.validation_seen;
                let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
                    no_progress_reason,
                    no_progress_signature_for_calls(&calls),
                ) && !no_new_after_mutation
                    && implementation_intent.is_none()
                    && (!expected_mutation || mutation_evidence_seen);
                let nudge = if all_bookkeeping_reposts {
                    if all_plan_reposts {
                        ui.nudge(&format!(
                            "the model re-posted an unchanged plan — withholding \
                             bookkeeping tools for a round and nudging it to execute \
                             the next step ({repeat_nudges}/{})",
                            self.config.loop_limits.max_repeat_nudges
                        ));
                    } else {
                        ui.nudge(&format!(
                            "the model repeated bookkeeping calls without real work — \
                             withholding bookkeeping tools for a round \
                             ({repeat_nudges}/{})",
                            self.config.loop_limits.max_repeat_nudges
                        ));
                    }
                    suppress_bookkeeping_tools_next = true;
                    force_tools_next = true;
                    // Cancel any prior force-final from a mixed stall so the
                    // bookkeeping withhold round still has real tools.
                    force_no_progress_final_answer_next = false;
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
                            self.config.loop_limits.max_repeat_nudges
                        ));
                        "The background process handle you just polled is completed, missing, or pruned, so polling it again cannot produce new output. Do not call bash_output for that handle again. Continue from the available output, restart the command if you still need it, or finish with the current result.".to_string()
                    } else {
                        ui.nudge(&format!(
                            "the model kept using stale background process handles — \
                             nudging it to stop using them ({repeat_nudges}/{})",
                            self.config.loop_limits.max_repeat_nudges
                        ));
                        "The background process handle you just used is already killed, already exited, missing, or pruned, so calling bash_kill for it again cannot change anything. Do not call bash_kill for that handle again. Continue from the available output, restart the command if you still need it, or finish with the current result.".to_string()
                    }
                } else if should_nudge_read_after_repeated_search(
                    read_only_intent,
                    &evidence,
                ) {
                    ui.nudge(&format!(
                                "the model re-ran the same search — nudging it to read a matching file ({repeat_nudges}/{})",
                                self.config.loop_limits.max_repeat_nudges
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
                        self.config.loop_limits.max_repeat_nudges
                    ));
                    let paths = inspected_paths_for_prompt(&evidence);
                    let plan_step = self
                        .goals.last_plan
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
                        self.config.loop_limits.max_repeat_nudges
                    ));
                    "The bash command you just called only says stop/quit/done or otherwise does no work. Do not call bash for that. If the task is complete, finish with a text answer; otherwise use a tool that inspects or changes the workspace.".to_string()
                } else if no_new_evidence && !exact_repeat {
                    ui.nudge(&format!(
                        "the model re-read files it already inspected — their contents are \
                         already above; nudging it to act on them ({repeat_nudges}/{})",
                        self.config.loop_limits.max_repeat_nudges
                    ));
                    REREAD_NUDGE.to_string()
                } else {
                    ui.nudge(&format!(
                        "the model re-ran the same command — its output is already above; \
                             nudging it to act on it ({repeat_nudges}/{})",
                        self.config.loop_limits.max_repeat_nudges
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
                return Ok(ModelRoundControl::Continue);
            }
            if stale_background_handle_call {
                if self.try_no_progress_recovery(
                    &mut progress_tracker,
                    &mut force_tools_next,
                    Some(&mut continue_total_nudges),
                    ui,
                ) {
                    prev_call_sig = None;
                    return Ok(ModelRoundControl::Continue);
                }
                return Ok(ModelRoundControl::BreakInner(false));
            }
            if has_no_progress_bash {
                if self.try_no_progress_recovery(
                    &mut progress_tracker,
                    &mut force_tools_next,
                    Some(&mut continue_total_nudges),
                    ui,
                ) {
                    prev_call_sig = None;
                    return Ok(ModelRoundControl::Continue);
                }
                progress_tracker.record(
                    ProgressKind::None,
                    "repeat_no_op_bash",
                    None,
                );
                ui.nudge("model repeated no-op shell commands");
                return Ok(ModelRoundControl::BreakInner(false));
            }
            if read_only_intent.is_some() && evidence.saw_search && !evidence.saw_read {
                if self.try_no_progress_recovery(
                    &mut progress_tracker,
                    &mut force_tools_next,
                    Some(&mut continue_total_nudges),
                    ui,
                ) {
                    prev_call_sig = None;
                    return Ok(ModelRoundControl::Continue);
                }
                progress_tracker.record(
                    ProgressKind::None,
                    "repeat_search_without_read",
                    None,
                );
                ui.nudge("review repeated the same search without reading files");
                return Ok(ModelRoundControl::BreakInner(false));
            }
            if let Some(intent) = read_only_intent
                && (evidence.saw_read || evidence.saw_search)
            {
                // One force-text attempt before settling: if the model already
                // inspected, prefer a chat answer over another tool round.
                if !force_text_answer_next
                    && !request_text_answer
                    && !request_no_progress_final_answer
                {
                    force_text_answer_next = true;
                    force_tools_next = false;
                    repeat_nudges = 0;
                    ui.nudge(
                        "review repeated the same command after inspection; forcing a bounded answer from inspected evidence",
                    );
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        crate::steering::repair_nudge_with_required_next(
                            crate::steering::ReviewRepairMode::SprawlForceAnswer,
                            crate::steering::summarize_inspected_evidence_nudge(intent, &evidence),
                        ),
                    );
                    return Ok(ModelRoundControl::Continue);
                }
                if self.try_no_progress_recovery(
                    &mut progress_tracker,
                    &mut force_tools_next,
                    Some(&mut continue_total_nudges),
                    ui,
                ) {
                    prev_call_sig = None;
                    return Ok(ModelRoundControl::Continue);
                }
                progress_tracker.record(
                    ProgressKind::None,
                    "repeat_after_inspection",
                    None,
                );
                ui.nudge("review repeated the same command after inspection");
                let _ = (intent, &evidence);
                return Ok(ModelRoundControl::BreakInner(false));
            }
            // Implementation / explicit-mutation turns that burned the
            // repeat budget on non-mutating work must not hard-stop yet.
            // Two live failure modes share this path:
            //   1. re-reading already-inspected files without editing
            //   2. pure bookkeeping loops (identical update_plan /
            //      record_decision) that never even inspected the tree
            // Case (2) used to fall through to the generic "kept
            // re-running the same command" stop because the old gate
            // required saw_read/saw_search. That ended turns after two plan
            // re-posts even when the model still had the implementation repair
            // budget — exactly the "I started that fix but didn't land the
            // edit" failure mode. Bookkeeping is zero-progress meta-work, not
            // a dangerous inspection loop; hand it the same edit nudge.
            let bookkeeping_only_no_progress = calls
                .iter()
                .all(|(_, name, _)| hi_tools::is_coordination(name));
            let implementation_needs_mutation = !implementation_tracker.mutation_seen
                && (implementation_intent.is_some() || expected_mutation)
                && ((evidence.saw_read || evidence.saw_search)
                    || bookkeeping_only_no_progress);
            if implementation_needs_mutation {
                if implementation_tracker.no_change_nudges < 2 {
                    implementation_tracker.no_change_nudges += 1;
                    evidence.quality_repair_nudges =
                        evidence.quality_repair_nudges.saturating_add(1);
                    let use_text_fallback = implementation_tracker.no_change_nudges >= 2;
                    force_tools_next = !use_text_fallback;
                    text_tool_fallback_next = use_text_fallback;
                    // Drop the sticky prev signature so the next real
                    // tool call isn't immediately compared against the
                    // bookkeeping-only round that just exhausted the
                    // repeat budget.
                    prev_call_sig = None;
                    prev_added_no_evidence = false;
                    if bookkeeping_only_no_progress {
                        // Keep bookkeeping withheld while we demand real
                        // work — otherwise the model just re-posts the
                        // plan again on the repair round.
                        suppress_bookkeeping_tools_next = true;
                        ui.nudge(
                            "implementation burned the bookkeeping-repeat budget without editing; nudging the model to edit or scaffold",
                        );
                    } else {
                        ui.nudge(
                            "implementation kept repeating without editing; nudging the model to edit or scaffold",
                        );
                    }
                    let nudge = if use_text_fallback {
                        implementation_text_tool_nudge(IMPLEMENTATION_NO_CHANGES_NUDGE)
                    } else {
                        IMPLEMENTATION_NO_CHANGES_NUDGE.to_string()
                    };
                    self.messages.push_nudge(NudgeKind::Continue, nudge);
                    return Ok(ModelRoundControl::Continue);
                }

                if self.try_no_progress_recovery(
                    &mut progress_tracker,
                    &mut force_tools_next,
                    Some(&mut continue_total_nudges),
                    ui,
                ) {
                    prev_call_sig = None;
                    return Ok(ModelRoundControl::Continue);
                }
                progress_tracker.record(
                    ProgressKind::None,
                    "implementation_no_mutation",
                    None,
                );
                ui.nudge(
                    "implementation kept repeating without editing; no file changes were made",
                );
                return Ok(ModelRoundControl::BreakInner(false));
            }
            if self.try_no_progress_recovery(
                &mut progress_tracker,
                &mut force_tools_next,
                Some(&mut continue_total_nudges),
                ui,
            ) {
                prev_call_sig = None;
                return Ok(ModelRoundControl::Continue);
            }
            return Ok(ModelRoundControl::BreakInner(false));
        }
        // A different set of calls (or none) this round — the model moved
        // on. A wait-poll
        // round is not counted as the first wasted round of a cycle:
        // waiting on external state is progress-neutral, not evidence
        // of a loop.
        repeat_sampling_rounds = 0;
        prev_call_sig = Some(call_sig);
        prev_added_no_evidence = no_new_evidence && !has_wait_poll_bash;

        // Inspection-sprawl guard: a read-only review turn that keeps
        // reading *distinct* files (each a new inspection signature, so
        // the repeat/cycle guard above never fires) without ever
        // producing findings. Once enough evidence has accumulated,
        // nudge the model to answer; if it keeps sprawling past the
        // budget, settle without fabricating an answer. This is
        // the only guard that catches the "read 100 files, never
        // answer" failure mode — all review-quality guards fire only
        // on a final text answer, which never comes while the model
        // keeps issuing tool calls.
        //
        // Also bounds ambiguous mutation-capable inspect loops (`/login`,
        // "how does X work") that have neither review sprawl nor
        // implementation discovery. Expected-mutation turns keep the
        // dedicated discovery cap instead.
        let bound_open_ended_inspection = !expected_mutation;
        evidence.record_inspection_round(&calls);
        if inspection_sprawl_exhausted(
            inspection_sprawl_intent,
            bound_open_ended_inspection,
            &evidence,
            &calls,
            read_only_inspection_cap,
        ) {
            // Prefer one force-text recovery when inspection already happened.
            // Do not keep_working here: that re-enabled tools after ChatOnly
            // wrap-up and the live 403 review ran another 8+ inspection rounds.
            if (evidence.saw_read || evidence.saw_search || evidence.saw_listing)
                && !force_text_answer_next
                && !request_text_answer
                && !request_no_progress_final_answer
            {
                force_text_answer_next = true;
                force_tools_next = false;
                if let Some(intent) = inspection_sprawl_intent.or(read_only_intent) {
                    ui.nudge(
                        "review kept inspecting without findings; forcing a bounded answer from inspected evidence",
                    );
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        crate::steering::repair_nudge_with_required_next(
                            crate::steering::ReviewRepairMode::SprawlForceAnswer,
                            crate::steering::summarize_inspected_evidence_nudge(intent, &evidence),
                        ),
                    );
                } else {
                    ui.nudge(
                        "kept inspecting without answering; forcing a bounded answer from inspected evidence",
                    );
                    self.messages.push_nudge(
                        NudgeKind::Continue,
                        inspection_round_cap_nudge(evidence.inspection_only_rounds),
                    );
                }
                return Ok(ModelRoundControl::Continue);
            }
            progress_tracker.record(ProgressKind::None, "inspection_sprawl_exhausted", None);
            ui.nudge("review kept inspecting new files without producing findings");
            return Ok(ModelRoundControl::BreakInner(false));
        }
        if should_nudge_inspection_sprawl(
            inspection_sprawl_intent,
            bound_open_ended_inspection,
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
                evidence.inspection_attempt_count(),
            ));
            self.messages
                .push_assistant_text_only(std::mem::take(&mut completion.content));
            self.messages.push_nudge(
                NudgeKind::Continue,
                inspection_sprawl_nudge(cap, evidence.inspection_attempt_count()),
            );
            return Ok(ModelRoundControl::Continue);
        }

        // This round's assistant text, joined and captured before the
        // content is moved into history. Used both to detect a content-less
        // response (a reasoning model can return only reasoning tokens or
        // whitespace) and by bounded answer-quality/completeness checks.
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

        if request_cap_wrap_up {
            // Whatever the model reports here is accepted as the wrap-up — the
            // turn ends at the cap either way, so no usability gating applies.
            if has_text {
                if buffer_read_only_review_text || !streamed_assistant_text {
                    let text_to_emit = if buffered_assistant_text.is_empty() {
                        assistant_text.as_str()
                    } else {
                        buffered_assistant_text.as_str()
                    };
                    self.emit_assistant_text(ui, text_to_emit);
                    ui.assistant_end();
                }
                progress_tracker.record(
                    ProgressKind::Weak,
                    if request_tool_cap_wrap_up {
                        "tool-limit wrap-up report"
                    } else {
                        "step-limit wrap-up report"
                    },
                    None,
                );
            }
            self.messages
                .push_assistant_text_only(std::mem::take(&mut completion.content));
            return Ok(ModelRoundControl::BreakInner(true));
        }

        if request_no_progress_final_answer {
            // Key on the live flag, not `last_progress_reason`: the reason
            // string is sticky (`ProgressKind::None` rounds never overwrite
            // it), so after any background wait earlier in the turn a stalled
            // final answer would bypass the usability gate and be branded a
            // successful completion.
            let background_status_answer = progress_tracker.awaiting_background;
            let unusable = forced_final_answer_is_unusable(
                &assistant_text,
                self.goals.plan_incomplete() && !background_status_answer,
            ) && !(background_status_answer && has_text);
            if has_text && (buffer_read_only_review_text || !streamed_assistant_text) {
                let text_to_emit = if buffered_assistant_text.is_empty() {
                    assistant_text.as_str()
                } else {
                    buffered_assistant_text.as_str()
                };
                self.emit_assistant_text(ui, text_to_emit);
                ui.assistant_end();
            }
            if unusable {
                // Weak-but-non-empty forced answers still count as a deliverable.
                if has_text && !assistant_text.trim().is_empty() {
                    force_no_progress_final_answer_next = false;
                    self.messages
                        .push_assistant(std::mem::take(&mut completion.content));
                    progress_tracker.no_progress_streak = 0;
                    progress_tracker.last_no_progress_reason.clear();
                    progress_tracker.record_final_answer();
                    ui.status("forced final answer was weak; accepting available text");
                    return Ok(ModelRoundControl::BreakInner(false));
                }
                if empty_retries < self.config.loop_limits.max_empty_retries {
                    empty_retries += 1;
                    ui.nudge(&format!(
                        "the forced final answer was empty; retrying tool-free ({empty_retries}/{})",
                        self.config.loop_limits.max_empty_retries
                    ));
                    return Ok(ModelRoundControl::Continue);
                }
                self.messages
                    .push_assistant_text_only(std::mem::take(&mut completion.content));
                progress_tracker.record(
                    ProgressKind::None,
                    "forced_final_unusable",
                    None,
                );
                return Err(anyhow::anyhow!(
                    "model returned no usable final answer after bounded recovery"
                ));
            }
            force_no_progress_final_answer_next = false;
            self.messages
                .push_assistant(std::mem::take(&mut completion.content));
            progress_tracker.record_final_answer();
            return Ok(ModelRoundControl::BreakInner(false));
        }

        // Auto-recover from a content-less response — no tool calls and no
        // text, i.e. a flaky provider returning only reasoning or an empty
        // message. Silently re-run a few times before giving up, each
        // retry resampling hotter (see the temperature bump above). The
        // dead round isn't recorded, so each retry re-runs with the
        // original context.
        if calls.is_empty() && !has_text {
            if empty_retries < self.config.loop_limits.max_empty_retries {
                empty_retries += 1;
                if made_tool_call {
                    self.nudge_after_post_tool_empty_response(
                        &mut force_tools_next,
                        implementation_intent.is_some(),
                    );
                }
                ui.status(&format!(
                    "⚠ the model returned no response — retrying ({empty_retries}/{})",
                    self.config.loop_limits.max_empty_retries
                ));
                return Ok(ModelRoundControl::Continue);
            }
            // The provider can occasionally return accepted-but-empty streams
            // after every requested tool has already succeeded (observed on the
            // live Pipe route immediately after a write + final update_plan).
            // A completed checklist plus concrete mutation/validation evidence
            // is a stronger terminal signal than a missing prose recap. Preserve
            // that productive outcome with an explicit deterministic message;
            // do not turn finished work into an infrastructure failure merely
            // because the optional summary channel exhausted its fault retries.
            let completed_plan_with_tool_evidence = !self.goals.plan().is_empty()
                && !self.goals.plan_incomplete()
                && (implementation_tracker.mutation_seen
                    || implementation_tracker.validation_seen);
            if completed_plan_with_tool_evidence {
                self.emit_assistant_text(
                    ui,
                    super::model_retry::COMPLETED_PLAN_EMPTY_RECAP_FALLBACK,
                );
                ui.assistant_end();
                self.messages.push_assistant(vec![Content::Text(
                    super::model_retry::COMPLETED_PLAN_EMPTY_RECAP_FALLBACK.into(),
                )]);
                progress_tracker.no_progress_streak = 0;
                progress_tracker.last_no_progress_reason.clear();
                progress_tracker.record_final_answer();
                ui.status(
                    "provider returned no final recap; closing from the completed plan and tool evidence",
                );
                return Ok(ModelRoundControl::BreakInner(false));
            }
            ui.status("⚠ the model returned no response after retrying — ending this bounded turn");
            return Err(anyhow::anyhow!("model returned no response after retrying"));
        }
        // Real output this round — clear the retry counter so the
        // temperature bump is transient: a later, unrelated empty response gets
        // its own budget rather than inheriting this one's elevation.
        empty_retries = 0;
        retry_state.protocol_retries = 0;
        truncation_retries = 0;

        if calls.is_empty() {
            // When a mutation already landed and a deterministic turn-end
            // verifier is configured, let that verifier own the requested
            // validation. Validation-only turns still require an actual model
            // tool call because the workspace verifier skips unchanged turns.
            let validation_gate_required = requested_validation
                && !(implementation_tracker.mutation_seen && verifier.is_on());
            match self.steer_without_tools(
                &assistant_text,
                &mut completion.content,
                read_only_intent,
                implementation_intent,
                expected_mutation,
                validation_gate_required,
                &mut implementation_tracker,
                &mut evidence,
                &mut review_repair,
                &mut progress_tracker,
                &mut silent_continues,
                &mut generic_completion_retries,
                &mut continue_total_nudges,
                &mut force_tools_next,
                &mut force_text_answer_next,
                &mut text_tool_fallback_next,
                &mut buffered_assistant_text,
                buffer_read_only_review_text,
                steps,
                ui,
            )? {
                super::steer::RoundControl::Continue => return Ok(ModelRoundControl::Continue),
                super::steer::RoundControl::BreakInner(hit) => return Ok(ModelRoundControl::BreakInner(hit)),
            }
        }

        // Execution validation uses the complete built-in catalog, plus any
        // dynamically advertised agent/MCP specs. The request may advertise a
        // task-focused subset, but the executor has always safely handled
        // other known calls (including promoted plain-text fallback calls).
        let mut execution_tool_specs = hi_tools::TOOL_SPECS.iter().cloned().collect::<Vec<_>>();
        for tool in advertised_tool_specs.iter() {
            if !execution_tool_specs
                .iter()
                .any(|known| known.name == tool.name)
            {
                execution_tool_specs.push(tool.clone());
            }
        }
        Ok(ModelRoundControl::RunTools {
            calls,
            completion_content: completion.content,
            tool_specs: std::sync::Arc::from(execution_tool_specs),
        })

        }.await;

        *state.steps = steps;
        *state.empty_retries = empty_retries;
        *state.truncation_retries = truncation_retries;
        *state.truncation_total_retries = truncation_total_retries;
        *state.silent_continues = silent_continues;
        *state.generic_completion_retries = generic_completion_retries;
        *state.continue_total_nudges = continue_total_nudges;
        *state.repeat_nudges = repeat_nudges;
        *state.force_tools_next = force_tools_next;
        *state.text_tool_fallback_next = text_tool_fallback_next;
        *state.force_text_answer_next = force_text_answer_next;
        *state.suppress_bookkeeping_tools_next = suppress_bookkeeping_tools_next;
        *state.made_tool_call = made_tool_call;
        *state.provider_exhausted = provider_exhausted;
        *state.turn_start = turn_start;
        *state.context_generation_seen = context_generation_seen;
        *state.indexed_ledger_revision = indexed_ledger_revision;
        *state.sched_tool_calls = sched_tool_calls;
        *state.sched_max_concurrent = sched_max_concurrent;
        *state.sched_serial_runs = sched_serial_runs;
        *state.tool_schema_tokens = tool_schema_tokens;
        *state.program_fallback_next = program_fallback_next;
        *state.program_fallback_used = program_fallback_used;
        *state.ended_at_cap = ended_at_cap;
        *state.cap_wrap_up_requested = cap_wrap_up_requested;
        *state.cap_kind = cap_kind;
        *state.retry_state = retry_state;
        *state.request_max_tokens_override = request_max_tokens_override;
        *state.compat_fallbacks = compat_fallbacks;
        *state.effective_fallback_route = effective_fallback_route;
        *state.ranked_context_paths = ranked_context_paths;
        progress_tracker.repeat_sampling_rounds = repeat_sampling_rounds;
        progress_tracker.force_no_progress_final_answer_next = force_no_progress_final_answer_next;
        progress_tracker.prev_added_no_evidence = prev_added_no_evidence;
        progress_tracker.prev_call_sig = prev_call_sig;
        *state.progress_tracker = progress_tracker;
        *state.evidence = evidence;
        *state.implementation_tracker = implementation_tracker;
        *state.review_repair = review_repair;
        *state.last_verify_attributions = last_verify_attributions;
        *state.tool_timeline = tool_timeline;
        *state.advertised_tool_names = advertised_tool_names;
        *state.turn_snapshot = turn_snapshot;

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_step_sentinel_is_never_a_reached_finite_cap() {
        assert!(!model_step_cap_reached(u32::MAX, u32::MAX));
        assert!(!model_step_cap_reached(6, 7));
        assert!(model_step_cap_reached(7, 7));
        assert!(model_step_cap_reached(8, 7));
    }

    #[test]
    fn empty_completion_retry_disables_deepseek_thinking() {
        assert_eq!(deepseek_thinking_for_round(None, false, false, 0), None);
        assert_eq!(
            deepseek_thinking_for_round(None, false, false, 1),
            Some(false)
        );
        assert_eq!(
            deepseek_thinking_for_round(
                Some(crate::steering::ReviewIntent::Review),
                true,
                false,
                0
            ),
            Some(true)
        );
    }

    #[test]
    fn collapse_duplicate_inspection_calls_keeps_first_and_preserves_mutations() {
        let read_args = r#"{"path":"src/moves.rs","offset":395,"limit":20}"#;
        let mut content = vec![
            Content::Text("inspect the file".into()),
            Content::ToolCall {
                id: "read-1".into(),
                name: "read".into(),
                arguments: read_args.into(),
            },
            Content::ToolCall {
                id: "read-2".into(),
                name: "read".into(),
                arguments: read_args.into(),
            },
            Content::ToolCall {
                id: "bash-1".into(),
                name: "bash".into(),
                arguments: r#"{"command":"touch marker"}"#.into(),
            },
        ];
        let calls = vec![
            ("read-1".into(), "read".into(), read_args.into()),
            ("read-2".into(), "read".into(), read_args.into()),
            (
                "bash-1".into(),
                "bash".into(),
                r#"{"command":"touch marker"}"#.into(),
            ),
        ];

        let (collapsed, duplicate_count) = collapse_duplicate_inspection_calls(&mut content, calls);

        assert_eq!(duplicate_count, 1);
        assert_eq!(collapsed.len(), 2);
        assert_eq!(collapsed[0].0, "read-1");
        assert_eq!(collapsed[1].0, "bash-1");
        let remaining_ids: Vec<_> = content
            .iter()
            .filter_map(|block| match block {
                Content::ToolCall { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(remaining_ids, ["read-1", "bash-1"]);
    }
}
