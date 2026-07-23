//! One Model→(text)Steer iteration of the inner turn loop.

use std::collections::BTreeSet;

use anyhow::Result;
use hi_ai::{ChatRequest, Content, RequestProfile, ToolMode};
use hi_tools::PlanStatus;

use crate::heuristics::{
    RECOVERY_SAMPLING, StallMode, looks_like_unfinished_step, parse_text_tool_calls,
    recovery_sampling, recovery_telemetry, textcall_id_offset,
};
use crate::steering::{
    BOOKKEEPING_REPOST_NUDGE, EvidenceTracker, IMPLEMENTATION_NO_CHANGES_NUDGE,
    ImplementationIntent, ImplementationTracker, PLAN_REPOST_NUDGE, READ_AFTER_SEARCH_NUDGE,
    READ_ONLY_SAFE_CONTEXT_WINDOW, REPEAT_NUDGE, REREAD_NUDGE, ReviewIntent,
    SKIPPED_BOOKKEEPING_REPOST_RESULT, SKIPPED_PLAN_REPOST_RESULT, SKIPPED_REPEATED_CALL_RESULT,
    ToolLoopGuardrail, bash_call_waits, bash_no_progress_signature, implementation_text_tool_nudge,
    inspected_paths_for_prompt, inspection_sprawl_exhausted, inspection_sprawl_nudge,
    should_nudge_inspection_sprawl, should_nudge_read_after_repeated_search,
};
use crate::transcript::NudgeKind;
use crate::verify::WorkspaceRepairVerifier;
use crate::{MAX_TOOL_PROTOCOL_RETRIES, TRUNCATED_TOOL_CALL_NUDGE, TRUNCATION_NUDGE, Ui};

use super::helpers::{build_turn_telemetry, effective_model_route};
use super::phase::TurnPhase;
use super::progress::{
    NO_PROGRESS_FINAL_ANSWER_NUDGE, ProgressKind, ProgressTracker, forced_final_answer_is_unusable,
    no_progress_signature_for_calls,
};
use super::retry::{
    INCOMPLETE_STATUS, ReviewRepairState, TurnRetryState, estimate_tool_schema_tokens,
};

pub(super) enum ModelRoundControl {
    Continue,
    BreakInner(bool),
    RunTools {
        calls: Vec<(String, String, String)>,
        completion_content: Vec<Content>,
        tool_specs: std::sync::Arc<[hi_ai::ToolSpec]>,
    },
}

pub(super) struct ModelRoundState<'a> {
    pub steps: &'a mut u32,
    pub empty_retries: &'a mut u32,
    pub truncation_retries: &'a mut u32,
    pub truncation_total_retries: &'a mut u32,
    pub silent_continues: &'a mut u32,
    pub continue_total_nudges: &'a mut u32,
    pub repeat_nudges: &'a mut u32,
    pub repeat_sampling_rounds: &'a mut u32,
    pub force_tools_next: &'a mut bool,
    pub text_tool_fallback_next: &'a mut bool,
    pub force_text_answer_next: &'a mut bool,
    pub force_no_progress_final_answer_next: &'a mut bool,
    pub suppress_bookkeeping_tools_next: &'a mut bool,
    pub prev_added_no_evidence: &'a mut bool,
    pub made_tool_call: &'a mut bool,
    pub turn_start: &'a mut usize,
    pub stalled_repeating: &'a mut bool,
    pub stalled_unfinished: &'a mut bool,
    pub context_generation_seen: &'a mut u64,
    pub indexed_ledger_revision: &'a mut u64,
    pub sched_tool_calls: &'a mut u32,
    pub sched_max_concurrent: &'a mut u32,
    pub sched_serial_runs: &'a mut u32,
    pub tool_schema_tokens: &'a mut u64,
    pub ended_at_cap: &'a mut bool,
    pub prev_call_sig: &'a mut Option<Vec<(String, String)>>,
    pub retry_state: &'a mut TurnRetryState,
    pub request_max_tokens_override: &'a mut Option<u32>,
    pub compat_fallbacks: &'a mut Vec<String>,
    pub effective_fallback_route: &'a mut Option<String>,
    pub ranked_context_paths: &'a mut BTreeSet<String>,
    pub progress_tracker: &'a mut ProgressTracker,
    pub evidence: &'a mut EvidenceTracker,
    pub implementation_tracker: &'a mut ImplementationTracker,
    pub review_repair: &'a mut ReviewRepairState,
    pub tool_guardrail: &'a mut ToolLoopGuardrail,
    pub last_verify_attributions: &'a mut Vec<hi_tools::Attribution>,
    pub tool_timeline: &'a mut Vec<crate::ToolCallEntry>,
    pub advertised_tool_names: &'a mut BTreeSet<String>,
    pub turn_snapshot: &'a mut Option<crate::verify::Snapshot>,
    pub max_steps: u32,
    pub context_task: &'a str,
    pub repository_context_enabled: bool,
    pub turn_ledger_revision: u64,
    pub read_only_intent: Option<ReviewIntent>,
    pub implementation_intent: Option<ImplementationIntent>,
    pub read_only_inspection_cap: Option<u32>,
    pub expected_mutation: bool,
    pub input: &'a str,
    pub user_prompt_tokens: u64,
    pub inspection_sprawl_intent: Option<ReviewIntent>,
    pub verifier: &'a WorkspaceRepairVerifier,
}

impl crate::Agent {
    /// A compact, model-facing snapshot of the current session, attached to
    /// `/btw` side questions so the model can answer "what's the status / what
    /// are you doing / what changed" without running tools. Kept short — it is
    /// injected into the transcript, so it must not blow up the context budget.
    pub(crate) fn btw_session_snapshot(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("- model: {}", self.model()));
        if let Some(route) = self.provider_route() {
            lines.push(format!("- provider route: {route}"));
        }
        lines.push(format!("- workspace: {}", self.workspace_root().display()));
        let goal = self.goal_summary();
        if goal != "off" {
            lines.push(format!("- goal: {goal}"));
        }
        let plan = self.current_plan();
        if !plan.is_empty() {
            let done = plan
                .iter()
                .filter(|s| s.status == hi_tools::PlanStatus::Done)
                .count();
            lines.push(format!("- plan: {done}/{} steps done", plan.len()));
            for step in plan {
                let mark = match step.status {
                    hi_tools::PlanStatus::Done => "✓",
                    hi_tools::PlanStatus::Active => "→",
                    hi_tools::PlanStatus::Pending => "·",
                };
                lines.push(format!("    {mark} {}", step.title));
            }
        }
        let checkpoints = self.checkpoint_count();
        if checkpoints > 0 {
            lines.push(format!("- checkpoints: {checkpoints}"));
        }
        // Live background jobs (loops, dev servers, training runs the agent
        // spawned). Lets the model answer "is my job still running / did it
        // finish" without polling. Command is truncated to keep the snapshot small.
        let jobs = self.background_snapshot();
        if !jobs.is_empty() {
            lines.push(format!("- background jobs: {}", jobs.len()));
            for (id, command, status) in &jobs {
                let cmd = if command.chars().count() > 60 {
                    let truncated: String = command.chars().take(57).collect();
                    format!("{truncated}…")
                } else {
                    command.clone()
                };
                lines.push(format!("    {id}: {cmd} ({status})"));
            }
        }
        lines.join("\n")
    }

    /// Emit one assistant text chunk, routing it to `btw_answer` when a `/btw`
    /// answer is pending (and clearing the flag on the first chunk) so the
    /// side-answer renders distinctly from main task output.
    pub(crate) fn emit_assistant_text(&mut self, ui: &mut dyn Ui, text: &str) {
        if self.btw_answer_pending {
            self.btw_answer_pending = false;
            ui.btw_answer(text);
        } else {
            ui.assistant_text(text);
        }
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
        let mut continue_total_nudges = *state.continue_total_nudges;
        let mut repeat_nudges = *state.repeat_nudges;
        let mut repeat_sampling_rounds = *state.repeat_sampling_rounds;
        let mut force_tools_next = *state.force_tools_next;
        let mut text_tool_fallback_next = *state.text_tool_fallback_next;
        let mut force_text_answer_next = *state.force_text_answer_next;
        let mut force_no_progress_final_answer_next = *state.force_no_progress_final_answer_next;
        let mut suppress_bookkeeping_tools_next = *state.suppress_bookkeeping_tools_next;
        let mut prev_added_no_evidence = *state.prev_added_no_evidence;
        let made_tool_call = *state.made_tool_call;
        let mut turn_start = *state.turn_start;
        let mut stalled_repeating = *state.stalled_repeating;
        let mut stalled_unfinished = *state.stalled_unfinished;
        let mut context_generation_seen = *state.context_generation_seen;
        let mut indexed_ledger_revision = *state.indexed_ledger_revision;
        let sched_tool_calls = *state.sched_tool_calls;
        let sched_max_concurrent = *state.sched_max_concurrent;
        let sched_serial_runs = *state.sched_serial_runs;
        let mut tool_schema_tokens = *state.tool_schema_tokens;
        let ended_at_cap = *state.ended_at_cap;
        let mut prev_call_sig = std::mem::take(state.prev_call_sig);
        let mut retry_state = std::mem::take(state.retry_state);
        let mut request_max_tokens_override = std::mem::take(state.request_max_tokens_override);
        let mut compat_fallbacks = std::mem::take(state.compat_fallbacks);
        let mut effective_fallback_route = std::mem::take(state.effective_fallback_route);
        let mut ranked_context_paths = std::mem::take(state.ranked_context_paths);
        let mut progress_tracker = std::mem::take(state.progress_tracker);
        let mut evidence = std::mem::take(state.evidence);
        let mut implementation_tracker = std::mem::take(state.implementation_tracker);
        let mut review_repair = std::mem::take(state.review_repair);
        let tool_guardrail = std::mem::take(state.tool_guardrail);
        let last_verify_attributions = std::mem::take(state.last_verify_attributions);
        let tool_timeline = std::mem::take(state.tool_timeline);
        let mut advertised_tool_names = std::mem::take(state.advertised_tool_names);
        let turn_snapshot = std::mem::take(state.turn_snapshot);
        let max_steps = state.max_steps;
        let context_task = state.context_task;
        let repository_context_enabled = state.repository_context_enabled;
        let turn_ledger_revision = state.turn_ledger_revision;
        let read_only_intent = state.read_only_intent;
        let implementation_intent = state.implementation_intent;
        let read_only_inspection_cap = state.read_only_inspection_cap;
        let expected_mutation = state.expected_mutation;
        let input = state.input;
        let _user_prompt_tokens = state.user_prompt_tokens;
        let inspection_sprawl_intent = state.inspection_sprawl_intent;
        let verifier = state.verifier;

        let result = async {
        self.set_turn_phase(TurnPhase::Model);
        if steps >= max_steps {
            return Ok(ModelRoundControl::BreakInner(true));
        }
        steps += 1;

        // Mid-turn steering: inject any messages the user typed while
        // the turn was running, as genuine user messages, before the
        // next model round. This is a safe transcript boundary — the
        // prior round's tool calls are all resolved — so the folding
        // nudge push keeps provider alternation valid. The model
        // decides how to weigh them; we add no deferral directive.
        // `/btw` entries are side *questions*, not steering: frame them as
        // "answer briefly, then continue" and attach a live session snapshot
        // so the model can answer questions about the current session.
        let interjected = self.interjections.drain();
        if !interjected.is_empty() {
            let mut steer_count = 0usize;
            let mut btw_count = 0usize;
            for message in &interjected {
                if let Some(question) = message.strip_prefix(crate::BTW_INTERJECTION_PREFIX) {
                    btw_count += 1;
                    self.messages.push_nudge_or_fold(
                        NudgeKind::Btw,
                        format!(
                            "The user asked a side question while you work. Answer it briefly \
                             (one short paragraph), then continue your current task unchanged. \
                             Do not treat it as a new instruction or change your plan.\n\n\
                             Question: {}\n\nCurrent session snapshot:\n{}",
                            question.trim(),
                            self.btw_session_snapshot()
                        ),
                    );
                } else {
                    steer_count += 1;
                    self.messages.push_nudge_or_fold(
                        NudgeKind::Interjection,
                        format!(
                            "The user sent this message while you were working — take it into account now:\n{message}"
                        ),
                    );
                }
            }
            if steer_count > 0 {
                ui.status(&format!(
                    "✉ received {steer_count} message(s) from you mid-turn — factoring them in"
                ));
            }
            if btw_count > 0 {
                ui.status(&format!(
                    "❓ answering {btw_count} side question(s) — then continuing the task"
                ));
                // The very next assistant text answers the side question; route it
                // to `btw_answer` so the frontend renders it distinctly. The flag
                // lives on the agent (not this round) because the answer may be
                // emitted one or more rounds later, after tool calls.
                self.btw_answer_pending = true;
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
        } else if force_tools_next && self.config.routing.tool_mode == ToolMode::Auto {
            ToolMode::Required
        } else {
            self.config.routing.tool_mode
        };
        let tool_availability_mode = if request_text_tool_fallback
            || request_text_answer
            || request_no_progress_final_answer
        {
            ToolMode::ChatOnly
        } else if read_only_intent.is_some()
            && !matches!(self.config.routing.tool_mode, ToolMode::ChatOnly)
        {
            ToolMode::ReadOnly
        } else {
            self.config.routing.tool_mode
        };
        let requested_request_max_tokens =
            request_max_tokens_override.unwrap_or(self.config.routing.max_tokens);
        let mut request_tools = self.request_tools_for(tool_availability_mode);
        if suppress_bookkeeping_tools_next {
            suppress_bookkeeping_tools_next = false;
            request_tools = super::model_request::apply_bookkeeping_suppress(request_tools, true);
        }
        let request_tool_schema_tokens = super::model_request::note_advertised_tools(
            &request_tools,
            &mut advertised_tool_names,
            &mut tool_schema_tokens,
        );
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
                self.reconcile_error_turn_changes(turn_ledger_revision).await?;
                self.truncate_messages(turn_start);
                self.add_error_usage(&err);
                self.emit_usage(ui);
                self.report.last_compat_fallbacks = compat_fallbacks.clone();
                self.report.last_turn_telemetry = build_turn_telemetry(
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
        let advertised_tool_specs = request_tools.clone();
        let request = ChatRequest {
            model: self.config.routing.model.clone(),
            request_id: Some(retry_state.request_id()),
            user_turn: true,
            canonical_objective: Some(context_task.to_string()),
            messages: self.messages.arc(),
            tools: request_tools,
            max_tokens: request_max_tokens,
            temperature,
            top_p,
            frequency_penalty,
            thinking_budget: self.config.routing.thinking_budget,
            reasoning_effort: self.config.routing.reasoning_effort,
            profile: RequestProfile {
                compat: self.config.routing.compat,
                tool_mode,
                stream_usage: None,
            },
        };

        let stream_result = self
            .handle_provider_stream(
                request,
                read_only_intent,
                implementation_intent,
                request_max_tokens,
                &mut retry_state,
                &mut request_max_tokens_override,
                &mut empty_retries,
                &mut force_tools_next,
                &mut text_tool_fallback_next,
                made_tool_call,
                &mut turn_start,
                turn_ledger_revision,
                &turn_snapshot,
                input,
                max_steps,
                verifier,
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
        // "picking up where it stalled" on the next prompt). Bounded
        // by a *dedicated* truncation budget (separate from
        // `empty_retries`) so a big task that legitimately hits the
        // cap several times can still finish without the user typing
        // "continue".
        let truncated = matches!(
            completion.stop_reason.as_deref(),
            Some("length" | "max_tokens")
        );
        if truncated && truncation_retries < self.config.loop_limits.max_truncation_retries {
            truncation_retries += 1;
            truncation_total_retries += 1;
            ui.nudge(&format!(
                "⚠ the model hit the output token limit — continuing ({truncation_retries}/{})",
                self.config.loop_limits.max_truncation_retries
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
                max = self.config.loop_limits.max_truncation_retries,
            ));
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
                // Never force a chat-only "final answer" after bookkeeping
                // loops on a mutation turn. That path exists for inspection
                // stalls where the model already has evidence to summarize;
                // on an edit request it just ends the turn incomplete with
                // zero file changes (live: "I started the fix but didn't
                // land the edit"). Keep tools required and let the
                // budget-exhausted branch hand off to implementation repair.
                let force_final_after_nudge = progress_tracker.record_no_progress_nudge(
                    stall_reason,
                    no_progress_signature_for_calls(&calls),
                ) && !no_new_after_mutation
                    && implementation_intent.is_none()
                    && !(expected_mutation && all_bookkeeping_reposts);
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
                ui.status(
                    "background process handles were completed, missing, or pruned (or already killed) and the model kept using them — the task may be incomplete. /retry, or send 'continue'.",
                );
                return Ok(ModelRoundControl::BreakInner(false));
            }
            if has_no_progress_bash {
                stalled_unfinished = true;
                ui.nudge("model repeated no-op shell commands; stopping incomplete");
                ui.status(INCOMPLETE_STATUS);
                return Ok(ModelRoundControl::BreakInner(false));
            }
            if read_only_intent.is_some() && evidence.saw_search && !evidence.saw_read {
                stalled_unfinished = true;
                ui.nudge(
                    "review repeated the same search without reading files; stopping incomplete",
                );
                ui.status(INCOMPLETE_STATUS);
                return Ok(ModelRoundControl::BreakInner(false));
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
            // required saw_read/saw_search. That branded turns as
            // `incomplete · stalled` after two plan re-posts even when
            // the model still had the implementation repair budget —
            // exactly the "I started that fix but didn't land the
            // edit" stall. Bookkeeping is zero-progress meta-work, not
            // a dangerous inspection loop; hand it the same edit nudge.
            let bookkeeping_only_stall = calls
                .iter()
                .all(|(_, name, _)| hi_tools::is_coordination(name));
            let implementation_needs_mutation = !implementation_tracker.mutation_seen
                && (implementation_intent.is_some() || expected_mutation)
                && ((evidence.saw_read || evidence.saw_search) || bookkeeping_only_stall);
            if implementation_needs_mutation {
                if implementation_tracker.no_change_nudges < 2 {
                    implementation_tracker.no_change_nudges += 1;
                    evidence.quality_repair_nudges =
                        evidence.quality_repair_nudges.saturating_add(1);
                    let use_text_fallback = implementation_tracker.no_change_nudges >= 2;
                    force_tools_next = !use_text_fallback;
                    text_tool_fallback_next = use_text_fallback;
                    // Clear the sticky repeat stall: we are converting it
                    // into an implementation-repair continue, not ending
                    // the turn as stalled_repeating.
                    stalled_repeating = false;
                    // Drop the sticky prev signature so the next real
                    // tool call isn't immediately compared against the
                    // bookkeeping-only round that just exhausted the
                    // repeat budget.
                    prev_call_sig = None;
                    prev_added_no_evidence = false;
                    if bookkeeping_only_stall {
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

                stalled_unfinished = true;
                ui.nudge(
                    "implementation kept repeating without editing; no file changes were made",
                );
                ui.status(INCOMPLETE_STATUS);
                return Ok(ModelRoundControl::BreakInner(false));
            }
            ui.status(
                "⚠ the model kept re-running the same command without acting on the \
                 result — the task may be incomplete. /retry, or send 'continue'.",
            );
            return Ok(ModelRoundControl::BreakInner(false));
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
            return Ok(ModelRoundControl::BreakInner(false));
        }
        if should_nudge_inspection_sprawl(
            inspection_sprawl_intent,
            &evidence,
            &calls,
            read_only_inspection_cap,
        ) {
            // Before nudging the model to stop, try granting a soft-cap
            // extension. Only grant when the current round uses context-efficient
            // tools (explore, repo_map, find_symbol) — plain read/grep sprawl
            // doesn't justify more budget. The extension is granted in chunks
            // so the model must re-justify if it needs more.
            let base_cap = read_only_inspection_cap
                .unwrap_or_else(|| evidence.inspection_attempt_count());
            let round_uses_context_efficient = calls
                .iter()
                .any(|(_, name, _)| crate::steering::is_context_efficient_tool(name));
            if round_uses_context_efficient && evidence.try_grant_soft_cap_extension() {
                let new_cap = evidence.effective_cap_with_extensions(base_cap);
                let extensions_remaining = crate::steering::MAX_SOFT_CAP_EXTENSIONS
                    .saturating_sub(evidence.soft_cap_extensions);
                ui.nudge(&format!(
                    "review inspected {} files/searches (weighted: {}); granting soft-cap extension to {}",
                    evidence.inspection_attempt_count(),
                    evidence.weighted_inspection_count(),
                    new_cap,
                ));
                self.messages
                    .push_assistant_text_only(std::mem::take(&mut completion.content));
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    crate::steering::soft_cap_extension_nudge(
                        crate::steering::SOFT_CAP_EXTENSION_GRANT,
                        new_cap,
                        extensions_remaining,
                    ),
                );
                return Ok(ModelRoundControl::Continue);
            }
            // No more extensions available — nudge the model to answer.
            evidence.inspection_sprawl_nudges =
                evidence.inspection_sprawl_nudges.saturating_add(1);
            force_text_answer_next = true;
            let cap = evidence.effective_cap_with_extensions(base_cap);
            ui.nudge(&format!(
                "review inspected {} files/searches (weighted: {}) without answering; nudging it to produce findings",
                evidence.inspection_attempt_count(),
                evidence.weighted_inspection_count(),
            ));
            self.messages
                .push_assistant_text_only(std::mem::take(&mut completion.content));
            self.messages.push_nudge(
                NudgeKind::Continue,
                inspection_sprawl_nudge(cap, evidence.weighted_inspection_count()),
            );
            return Ok(ModelRoundControl::Continue);
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
                self.goals.plan_incomplete(),
            );
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
                self.messages
                    .push_assistant_text_only(std::mem::take(&mut completion.content));
                stalled_unfinished = true;
                progress_tracker.record(
                    ProgressKind::None,
                    "forced final-answer attempt was unusable",
                    None,
                );
                ui.status(INCOMPLETE_STATUS);
                return Ok(ModelRoundControl::BreakInner(false));
            }
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
            ui.status("⚠ the model returned no response after retrying — try /retry.");
            return Ok(ModelRoundControl::BreakInner(false));
        }
        // Real output this round — clear the retry counter so the
        // temperature bump is transient: a later, unrelated stall gets
        // its own budget rather than inheriting this one's elevation.
        empty_retries = 0;
        retry_state.protocol_retries = 0;
        truncation_retries = 0;

        if calls.is_empty() {
            match self.steer_without_tools(
                &assistant_text,
                &mut completion.content,
                read_only_intent,
                implementation_intent,
                expected_mutation,
                made_tool_call,
                &mut implementation_tracker,
                &mut evidence,
                &mut review_repair,
                &mut progress_tracker,
                &mut silent_continues,
                &mut continue_total_nudges,
                &mut force_tools_next,
                &mut force_text_answer_next,
                &mut text_tool_fallback_next,
                &mut stalled_unfinished,
                &mut buffered_assistant_text,
                buffer_read_only_review_text,
                steps,
                ui,
            ) {
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
        *state.continue_total_nudges = continue_total_nudges;
        *state.repeat_nudges = repeat_nudges;
        *state.repeat_sampling_rounds = repeat_sampling_rounds;
        *state.force_tools_next = force_tools_next;
        *state.text_tool_fallback_next = text_tool_fallback_next;
        *state.force_text_answer_next = force_text_answer_next;
        *state.force_no_progress_final_answer_next = force_no_progress_final_answer_next;
        *state.suppress_bookkeeping_tools_next = suppress_bookkeeping_tools_next;
        *state.prev_added_no_evidence = prev_added_no_evidence;
        *state.made_tool_call = made_tool_call;
        *state.turn_start = turn_start;
        *state.stalled_repeating = stalled_repeating;
        *state.stalled_unfinished = stalled_unfinished;
        *state.context_generation_seen = context_generation_seen;
        *state.indexed_ledger_revision = indexed_ledger_revision;
        *state.sched_tool_calls = sched_tool_calls;
        *state.sched_max_concurrent = sched_max_concurrent;
        *state.sched_serial_runs = sched_serial_runs;
        *state.tool_schema_tokens = tool_schema_tokens;
        *state.ended_at_cap = ended_at_cap;
        *state.prev_call_sig = prev_call_sig;
        *state.retry_state = retry_state;
        *state.request_max_tokens_override = request_max_tokens_override;
        *state.compat_fallbacks = compat_fallbacks;
        *state.effective_fallback_route = effective_fallback_route;
        *state.ranked_context_paths = ranked_context_paths;
        *state.progress_tracker = progress_tracker;
        *state.evidence = evidence;
        *state.implementation_tracker = implementation_tracker;
        *state.review_repair = review_repair;
        *state.tool_guardrail = tool_guardrail;
        *state.last_verify_attributions = last_verify_attributions;
        *state.tool_timeline = tool_timeline;
        *state.advertised_tool_names = advertised_tool_names;
        *state.turn_snapshot = turn_snapshot;

        result
    }
}
