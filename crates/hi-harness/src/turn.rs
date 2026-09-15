//! Prompt → Pipe stream → tools → repeat until the model stops.

use anyhow::{Result, bail};
use hi_ai::{Content, Message, Usage};
use hi_liveness::{ENV_TURN_INTENT, TurnIntent};
use hi_tools::checkpoint;

use crate::compact::{
    AUTO_COMPACT_THRESHOLD_PERCENT, apply_summary, compact_prompt, emergency_summary,
    estimate_message_tokens, parse_summary,
};
use crate::pipe::{PipeCompletion, PipeError, StreamDelta};
use crate::prompt::SYSTEM_PROMPT;
use crate::tools::{ToolHost, advertised_tools, interrupted_outcome, plan_from_outcome};
use crate::ui::Ui;
use crate::{Harness, TurnCancellation, TurnOutcome, TurnStopReason};

impl Harness {
    pub async fn run_turn_cancellable(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
        cancellation: TurnCancellation,
    ) -> Result<TurnOutcome> {
        self.turn_cancel = Some(cancellation.clone());
        let interrupt = self.tools.interrupt_handle();
        let watch_cancel = cancellation.clone();
        let watcher = tokio::spawn(async move {
            watch_cancel.cancelled().await;
            interrupt.interrupt();
        });
        let result = self.run_turn_inner(input, ui, &cancellation, false).await;
        watcher.abort();
        self.turn_cancel = None;
        result
    }

    /// Continue an unmatched `PendingTurn` without pushing a second user line.
    pub async fn resume_incomplete_turn(
        &mut self,
        ui: &mut dyn Ui,
        cancellation: TurnCancellation,
    ) -> Result<Option<TurnOutcome>> {
        let expected = turn_intent_prompt();
        if !self.can_resume_incomplete(expected.as_deref()) {
            return Ok(None);
        }
        self.turn_cancel = Some(cancellation.clone());
        let interrupt = self.tools.interrupt_handle();
        let watch_cancel = cancellation.clone();
        let watcher = tokio::spawn(async move {
            watch_cancel.cancelled().await;
            interrupt.interrupt();
        });
        let result = self.run_turn_inner("", ui, &cancellation, true).await;
        watcher.abort();
        self.turn_cancel = None;
        Ok(Some(result?))
    }

    async fn run_turn_inner(
        &mut self,
        input: &str,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
        resume: bool,
    ) -> Result<TurnOutcome> {
        let pre = if resume {
            self.pending_turn
                .as_ref()
                .and_then(|pending| pending.pre_checkpoint.clone())
        } else {
            match checkpoint::create_detailed_with_state(&self.workspace_root, &self.state_root)
                .await
            {
                checkpoint::CreateResult::Created(id) => Some(id),
                checkpoint::CreateResult::Unavailable(reason) => {
                    ui.checkpoint_warning(&format!("checkpoint unavailable: {reason}"));
                    None
                }
                checkpoint::CreateResult::Failed(reason) => {
                    ui.checkpoint_warning(&format!("checkpoint failed: {reason}"));
                    None
                }
            }
        };

        self.compact_suppressed = false;
        self.tools.reset_bash_repeats();
        let mut persisted_before = if resume {
            self.turn_open = true;
            if let Some(pending) = &self.pending_turn {
                self.turn_index = pending.turn_index;
                self.liveness.set_turn_index(pending.turn_index);
                self.liveness
                    .set_pre_checkpoint(pending.pre_checkpoint.clone());
            }
            self.liveness
                .set_state(hi_liveness::HarnessState::AwaitingModel);
            self.liveness.note_progress();
            self.messages.len()
        } else {
            self.messages.push(Message::user(input));
            self.begin_persisted_turn(input, pre.as_deref())
        };
        let mut turn_usage = Usage::default();
        let mut changed = Vec::new();
        let mut mutated = false;
        let mut round = 0usize;
        let mut probe_refusals = 0u32;
        let mut empty_after_tools = 0u32;
        let mut continue_hint: Option<&str> = None;

        // Continues until the model stops calling tools, the user cancels, or
        // a request fails. There is no round/step cap. Empty streams are
        // retried in the Pipe client (same request). An empty stop *after
        // tools* is not a finished turn: continue with a hint, then report
        // `EmptyAssistantAfterTools` so Sentinel can auto-repair. A round-0
        // empty stop (no tools) still ends the turn — `/retry` continues.
        loop {
            if cancel.is_cancelled() {
                return self
                    .finish_turn(
                        persisted_before,
                        pre.as_deref(),
                        mutated,
                        turn_usage,
                        changed,
                        TurnStopReason::Cancelled,
                        None,
                        ui,
                    )
                    .await;
            }
            for steered in self.steer.drain() {
                self.messages.push(Message::user(steered));
            }
            if self.should_auto_compact() {
                match self.compact(None, ui, cancel).await {
                    Ok(true) => {
                        persisted_before = self.messages.len();
                        ui.status("compacted conversation");
                    }
                    Ok(false) => {}
                    Err(err) => {
                        ui.status(&format!("compact failed: {err:#}"));
                        if self.occupancy_percent() >= AUTO_COMPACT_THRESHOLD_PERCENT {
                            hi_liveness::report_invariant(
                                &self.liveness,
                                hi_liveness::InvariantCode::CompactFailedOverWindow,
                            );
                            let summary = emergency_summary(self.occupancy_percent());
                            self.messages = apply_summary(&self.messages, &summary);
                            persisted_before = self.messages.len();
                            self.last_context_occupancy = estimate_message_tokens(&self.messages);
                            self.session_usage.context_occupancy = self.last_context_occupancy;
                            self.persist_snapshot();
                            ui.status("emergency compacted conversation");
                            self.compact_suppressed = false;
                        } else {
                            self.compact_suppressed = true;
                        }
                    }
                }
                if self.should_auto_compact() {
                    self.compact_suppressed = true;
                }
            }
            ui.status(&format!("pipe · {}", self.model()));
            self.liveness
                .set_state(hi_liveness::HarnessState::AwaitingModel);
            let messages = self.request_messages(continue_hint);
            let tools = advertised_tools();
            let liveness = self.liveness.clone();
            let mut on_event = |delta: StreamDelta| {
                liveness.note_progress();
                match delta {
                    StreamDelta::Text(text) => ui.assistant_text(&text),
                    StreamDelta::Reasoning(text) => ui.assistant_reasoning(&text),
                }
            };
            let model = self.model();
            let completion = match self
                .client
                .stream(
                    &model,
                    &messages,
                    &tools,
                    self.max_tokens,
                    self.reasoning_effort(),
                    &mut on_event,
                    cancel,
                )
                .await
            {
                Ok(completion) => completion,
                Err(err) => {
                    if let Some(pipe) = err.downcast_ref::<PipeError>() {
                        if pipe.is_cancelled() {
                            return self
                                .finish_turn(
                                    persisted_before,
                                    pre.as_deref(),
                                    mutated,
                                    turn_usage,
                                    changed,
                                    TurnStopReason::Cancelled,
                                    None,
                                    ui,
                                )
                                .await;
                        }
                        let kind = if pipe.is_auth() { "auth" } else { "request" };
                        ui.turn_error(kind, &pipe.to_string(), guidance(pipe));
                        return self
                            .finish_turn(
                                persisted_before,
                                pre.as_deref(),
                                mutated,
                                turn_usage,
                                changed,
                                TurnStopReason::Error,
                                Some(pipe.to_string()),
                                ui,
                            )
                            .await;
                    }
                    ui.turn_error("request", &format!("{err:#}"), "check the Pipe API");
                    self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                    self.close_persisted_turn(persisted_before, TurnStopReason::Error);
                    return Err(err);
                }
            };
            turn_usage.add(completion.usage);
            self.record_context_occupancy(completion.usage);
            ui.usage(
                completion.usage.input_tokens,
                completion.usage.output_tokens,
                completion
                    .usage
                    .context_occupancy
                    .max(completion.usage.input_tokens),
                Some(self.context_window()),
                completion.usage.estimated,
            );
            ui.assistant_end();

            if completion.tool_calls.is_empty() {
                if completion.is_empty() && round > 0 {
                    if empty_after_tools < EMPTY_AFTER_TOOLS_CONTINUATIONS {
                        empty_after_tools = empty_after_tools.saturating_add(1);
                        // Compact may have been suppressed after a prior
                        // failure; try again before the continuation request
                        // so the model is not asked to finish a 3M-token turn.
                        self.compact_suppressed = false;
                        continue_hint = Some(EMPTY_AFTER_TOOLS_HINT);
                        ui.status("empty model stop after tools; continuing");
                        continue;
                    }
                    hi_liveness::report_invariant(
                        &self.liveness,
                        hi_liveness::InvariantCode::EmptyAssistantAfterTools,
                    );
                    const MSG: &str = "model stopped after tool work with no user-visible answer";
                    ui.turn_error(
                        "empty_stop",
                        MSG,
                        "supervised sessions auto-repair; otherwise /retry",
                    );
                    self.session_usage.add(turn_usage);
                    ui.session_usage(self.session_usage);
                    self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                    self.close_persisted_turn(persisted_before, TurnStopReason::Error);
                    ui.changed_files(changed.clone());
                    ui.turn_end("empty stop after tools");
                    return Ok(TurnOutcome {
                        stop_reason: TurnStopReason::Error,
                        usage: turn_usage,
                        changed_files: changed,
                        error: Some(MSG.into()),
                        verification: None,
                    });
                }
                if !completion.is_empty() {
                    self.messages.push(assistant_message(&completion));
                }
                self.session_usage.add(turn_usage);
                ui.session_usage(self.session_usage);
                self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                self.close_persisted_turn(persisted_before, TurnStopReason::Completed);
                let verification = self.run_verify(ui, cancel).await;
                ui.changed_files(changed.clone());
                ui.turn_end(&summary(&completion, round));
                return Ok(TurnOutcome {
                    stop_reason: TurnStopReason::Completed,
                    usage: turn_usage,
                    changed_files: changed,
                    error: None,
                    verification,
                });
            }

            empty_after_tools = 0;
            continue_hint = None;
            self.messages.push(assistant_message(&completion));
            self.liveness
                .set_state(hi_liveness::HarnessState::ExecutingTool);
            for call in &completion.tool_calls {
                if cancel.is_cancelled() {
                    self.messages.push(Message::tool_result(
                        &call.id,
                        interrupted_outcome().content,
                    ));
                    continue;
                }
                let outcome = self
                    .tools
                    .execute(
                        &call.id,
                        &call.name,
                        &call.arguments,
                        || self.permission_mode(),
                        ui,
                    )
                    .await;
                ui.tool_call_id(&call.id, &call.name, &call.arguments);
                if let Some(plan) = plan_from_outcome(&outcome) {
                    self.plan = plan.clone();
                    ui.plan_result_id(
                        &call.id,
                        &call.name,
                        &outcome.content,
                        outcome.status,
                        &plan,
                    );
                } else {
                    ui.tool_result_id(&call.id, &call.name, &outcome.content, outcome.status);
                }
                if outcome.effects.mutation_applied {
                    mutated = true;
                    self.liveness.note_progress();
                }
                for change in &outcome.effects.file_changes {
                    if !changed.contains(&change.path) {
                        changed.push(change.path.clone());
                    }
                }
                self.last_changed_files = changed.clone();
                self.messages
                    .push(Message::tool_result(&call.id, outcome.content.clone()));
                if hi_tools::is_probe_refusal(&outcome.content) {
                    probe_refusals = probe_refusals.saturating_add(1);
                }
                if self.should_stop_for_storm() {
                    const MSG: &str = "identical tool storm; stopping the turn";
                    if self.occupancy_percent() >= AUTO_COMPACT_THRESHOLD_PERCENT
                        || self.compact_suppressed
                    {
                        hi_liveness::report_invariant(
                            &self.liveness,
                            hi_liveness::InvariantCode::CompactFailedOverWindow,
                        );
                    }
                    ui.turn_error(
                        "tool_storm",
                        MSG,
                        "supervised sessions auto-repair when compact has failed; otherwise /retry",
                    );
                    self.session_usage.add(turn_usage);
                    ui.session_usage(self.session_usage);
                    self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                    self.close_persisted_turn(persisted_before, TurnStopReason::Error);
                    ui.changed_files(changed.clone());
                    ui.turn_end("identical tool storm");
                    return Ok(TurnOutcome {
                        stop_reason: TurnStopReason::Error,
                        usage: turn_usage,
                        changed_files: changed,
                        error: Some(MSG.into()),
                        verification: None,
                    });
                }
            }
            if probe_refusals >= hi_tools::STOP_AFTER_PROBE_REFUSALS {
                self.session_usage.add(turn_usage);
                ui.session_usage(self.session_usage);
                self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                self.close_persisted_turn(persisted_before, TurnStopReason::Completed);
                ui.changed_files(changed.clone());
                ui.turn_end("stopped repeating a detached binary probe");
                return Ok(TurnOutcome {
                    stop_reason: TurnStopReason::Completed,
                    usage: turn_usage,
                    changed_files: changed,
                    error: None,
                    verification: None,
                });
            }
            round += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_turn(
        &mut self,
        persisted_before: usize,
        pre: Option<&str>,
        mutated: bool,
        usage: Usage,
        changed_files: Vec<String>,
        stop_reason: TurnStopReason,
        error: Option<String>,
        ui: &mut dyn Ui,
    ) -> Result<TurnOutcome> {
        self.seal_checkpoint(pre, mutated, ui).await;
        self.close_persisted_turn(persisted_before, stop_reason);
        Ok(TurnOutcome {
            stop_reason,
            usage,
            changed_files,
            error,
            verification: None,
        })
    }

    /// Grok-style `/compact`: model summary, then original query + summary.
    pub async fn compact(
        &mut self,
        user_context: Option<&str>,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Result<bool> {
        if self.messages.len() < 4 {
            return Ok(false);
        }
        ui.status("compacting conversation");
        let previous = self.liveness.state();
        self.liveness
            .set_state(hi_liveness::HarnessState::Compacting);
        self.liveness
            .emit(hi_liveness::EventCode::CompactStart, None, None, None);
        let mut request = vec![Message::system(SYSTEM_PROMPT)];
        request.extend(self.messages.iter().cloned());
        request.push(Message::user(compact_prompt(user_context)));
        let model = self.model();
        let liveness = self.liveness.clone();
        let mut on_delta = |_delta: StreamDelta| {
            liveness.note_progress();
        };
        let completion = self
            .client
            .stream(
                &model,
                &request,
                &[],
                self.max_tokens,
                self.reasoning_effort(),
                &mut on_delta,
                cancel,
            )
            .await;
        self.liveness
            .emit(hi_liveness::EventCode::CompactEnd, None, None, None);
        self.liveness.set_state(previous);
        let completion = completion?;
        self.record_context_occupancy(completion.usage);
        if !completion.tool_calls.is_empty() {
            bail!("compaction model called a tool; refusing to replace history");
        }
        let Some(summary) = parse_summary(&completion.text) else {
            bail!("compaction model did not return a <summary> block");
        };
        self.messages = apply_summary(&self.messages, &summary);
        self.last_context_occupancy = estimate_message_tokens(&self.messages);
        self.session_usage.context_occupancy = self.last_context_occupancy;
        self.persist_snapshot();
        Ok(true)
    }

    pub(crate) fn occupancy_percent(&self) -> u64 {
        let window = u64::from(self.context_window().max(1));
        self.current_occupancy().saturating_mul(100) / window
    }

    pub(crate) fn should_auto_compact(&self) -> bool {
        !self.compact_suppressed
            && self.messages.len() >= 4
            && self.occupancy_percent() >= AUTO_COMPACT_THRESHOLD_PERCENT
    }

    fn should_stop_for_storm(&self) -> bool {
        matches!(
            self.liveness.snapshot().invariant.map(|inv| inv.code),
            Some(
                hi_liveness::InvariantCode::IdenticalToolStorm
                    | hi_liveness::InvariantCode::CompactFailedOverWindow
            )
        )
    }

    pub(crate) fn request_messages(&self, continue_hint: Option<&str>) -> Vec<Message> {
        let mut out = vec![Message::system(SYSTEM_PROMPT)];
        out.extend(self.messages.iter().cloned());
        if let Some(hint) = continue_hint.map(str::trim).filter(|text| !text.is_empty()) {
            // Not persisted: the session file stays a record of real user
            // turns. The hint only exists on the next Pipe request.
            out.push(Message::user(hint));
        }
        out
    }

    async fn seal_checkpoint(&mut self, pre: Option<&str>, mutated: bool, ui: &mut dyn Ui) {
        if !mutated {
            return;
        }
        let Some(pre) = pre else {
            return;
        };
        match checkpoint::create_detailed_with_state(&self.workspace_root, &self.state_root).await {
            checkpoint::CreateResult::Created(post) => {
                self.checkpoints
                    .push(checkpoint::sealed_reference(pre, &post));
                if let Some(session) = &mut self.session {
                    let _ = session.record_checkpoints(&self.checkpoints);
                }
            }
            other => ui.checkpoint_warning(&format!("could not seal undo point: {other:?}")),
        }
    }

    async fn run_verify(&self, ui: &mut dyn Ui, cancel: &TurnCancellation) -> Option<String> {
        let command = self.verify_command.as_deref()?;
        if cancel.is_cancelled() {
            return None;
        }
        ui.status(&format!("verify · {command}"));
        self.liveness
            .set_state(hi_liveness::HarnessState::Verifying);
        self.liveness
            .emit(hi_liveness::EventCode::VerifyStart, None, None, None);
        let result =
            match hi_tools::run_check_in_with_runner(self.tools.runner_ref(), command).await {
                Ok(execution) => {
                    let text = execution.display_content();
                    ui.tool_call("verify", command);
                    ui.tool_result("verify", &text);
                    self.liveness.note_progress();
                    Some(if execution.status == hi_tools::ToolStatus::Succeeded {
                        "passed".into()
                    } else {
                        "failed".into()
                    })
                }
                Err(err) => {
                    ui.status(&format!("verify failed to start: {err:#}"));
                    Some("failed".into())
                }
            };
        self.liveness
            .emit(hi_liveness::EventCode::VerifyEnd, None, None, None);
        self.liveness.set_state(hi_liveness::HarnessState::Idle);
        result
    }
}

/// Pipe already retries the same empty stream three times. After that, the
/// harness asks once more with a continuation hint, then once more, then
/// reports the invariant. Two continuations is enough to recover a truncated
/// final answer without looping forever on a model that only returns usage.
const EMPTY_AFTER_TOOLS_CONTINUATIONS: u32 = 2;

const EMPTY_AFTER_TOOLS_HINT: &str = "\
Your last response ended with no user-visible answer after tool work. \
Continue the task and write the complete reply to the user. Do not stop \
without that reply.";

fn turn_intent_prompt() -> Option<String> {
    let path = std::env::var_os(ENV_TURN_INTENT)?;
    let bytes = std::fs::read(path).ok()?;
    let intent: TurnIntent = serde_json::from_slice(&bytes).ok()?;
    Some(intent.prompt)
}

fn assistant_message(completion: &PipeCompletion) -> Message {
    let mut content = Vec::new();
    if !completion.reasoning.is_empty() {
        content.push(Content::Thinking {
            text: completion.reasoning.clone(),
            signature: None,
        });
    }
    if !completion.text.is_empty() {
        content.push(Content::Text(completion.text.clone()));
    }
    for call in &completion.tool_calls {
        content.push(Content::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        });
    }
    Message::assistant(content)
}

fn summary(completion: &PipeCompletion, round: usize) -> String {
    if !completion.text.trim().is_empty() {
        completion.text.trim().chars().take(120).collect()
    } else if round == 0 {
        "done".into()
    } else {
        format!("done after {} tool round(s)", round)
    }
}

fn guidance(error: &PipeError) -> &'static str {
    if error.is_auth() {
        "run `hi auth pipenetwork` or set PIPENETWORK_API_KEY"
    } else if error.retryable {
        "retry in a moment"
    } else {
        "see the Pipe API error"
    }
}

impl ToolHost {
    pub(crate) fn runner_ref(&self) -> &hi_tools::ProcessRunner {
        self.runner_handle()
    }
}
