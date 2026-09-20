//! Prompt → Pipe stream → tools → repeat until the model stops.

use anyhow::{Context, Result, bail};
use hi_ai::{Content, Message, Role, Usage};
use hi_liveness::{ENV_TURN_INTENT, TurnIntent};
use hi_tools::checkpoint;

use crate::compact::{
    AUTO_COMPACT_THRESHOLD_PERCENT, CHEAP_SHRINK_THRESHOLD_PERCENT, apply_summary,
    cheap_shrink_with, compact_prompt, emergency_summary, parse_summary,
};
use crate::completion::{
    self, Decision, Demand, Event, Intent, TurnFacts, coordination_only_round, looks_like_verify,
    read_pagination_only, stall_tool_round,
};
use crate::pipe::{PipeCompletion, PipeError, StreamDelta};
use crate::prompt::SYSTEM_PROMPT;
use crate::tools::{ToolHost, advertised_tools, interrupted_outcome, plan_from_outcome};
use crate::ui::{AutoHint, Ui};
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
        if !resume {
            self.turn_effort = None;
        }
        let managed_turn = self.model() == "pipe/auto";
        if let Some(pending) = &self.pending_turn {
            self.client.validate_pending_workflow(
                &format!("{}:{}", pending.turn_index, pending.started_unix_ms),
                managed_turn,
            )?;
        }
        let effort_join = if managed_turn {
            None
        } else {
            self.spawn_turn_start_effort(resume, input)
        };
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

        if let Some(join) = effort_join
            && let Ok(Some(noul)) = join.await
        {
            self.apply_jev_effort(Some(noul));
        }

        self.compact_suppressed = false;
        self.tools.reset_bash_repeats();
        if !resume && self.dismiss_completed_plan() {
            ui.plan(&[]);
        }
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
        if self.model() == "pipe/auto" {
            let pending = self
                .pending_turn
                .as_ref()
                .context("managed coding requires a persisted pending turn")?;
            self.client.begin_managed_turn(
                format!("{}:{}", pending.turn_index, pending.started_unix_ms),
                resume,
            )?;
            if resume {
                self.client.recover_managed_tools(&mut self.messages)?;
            }
        }
        let mut turn_usage = Usage::default();
        let mut changed = Vec::new();
        let mut mutated = false;
        let mut round = 0usize;
        let mut probe_refusals = 0u32;
        let mut repeat_stop: Option<&'static str> = None;
        let mut truncated_continuations = 0u32;
        let mut promised_continuations = 0u32;
        let mut typesafe_calls = 0u32;
        let intent = Intent::from_prompt(&completion::latest_prompt_text(&self.messages, input));
        let mut facts = TurnFacts::new(intent, completion::plan_is_open(&self.plan));
        if managed_turn && resume {
            let (was_mutated, was_verified, paths) = self.client.recovered_managed_progress()?;
            mutated = was_mutated;
            changed = paths;
            facts.mutated = was_mutated;
            facts.ran_verify = was_verified;
        }
        let mut continue_hint: Option<&str> = None;

        // Continues until the completion policy says Complete/Error, the user
        // cancels, or a request fails. There is no round/step cap. Empty
        // streams are retried in the Pipe client. The harness owns stop:
        // after tools, a turn cannot finish without a user-visible answer;
        // consecutive stall rounds are budgeted; progress is facts (mutation,
        // verify, open plan, visible answer), not prompt-word overlays.
        loop {
            if !ui.dispatch_allowed() {
                cancel.cancel();
            }
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
            self.persist_turn_progress(&mut persisted_before);
            if self.auto_compact && self.apply_cheap_shrink_if_needed(ui) {
                persisted_before = self.messages.len();
            }
            if self.should_auto_compact() {
                match self.reclaim_context(None, ui, cancel, false).await {
                    Ok(true) => {
                        persisted_before = self.messages.len();
                    }
                    Ok(false) => {}
                    Err(err) => {
                        ui.status(&format!("compact failed: {err:#}"));
                        self.compact_suppressed = true;
                    }
                }
                self.compact_suppressed = self.should_auto_compact();
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
                    StreamDelta::Status(text) => ui.status(&text),
                    StreamDelta::Text(text) => ui.assistant_text(&text),
                    StreamDelta::Reasoning(text) => ui.assistant_reasoning(&text),
                }
            };
            let model = self.model();
            anyhow::ensure!(
                (model == "pipe/auto") == managed_turn,
                "managed workflow selection changed during an active turn; finish or cancel this turn before switching"
            );
            let completion = match self
                .client
                .stream(
                    &model,
                    &messages,
                    &tools,
                    self.max_tokens(),
                    self.effective_reasoning_effort(),
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
                let circular = repeating_review_dump(&completion.text);
                if completion.is_truncated() || circular {
                    if !completion.is_empty() {
                        self.messages
                            .push(assistant_message_for_stop(&completion, circular));
                    }
                    if truncated_continuations < TRUNCATED_OUTPUT_CONTINUATIONS {
                        truncated_continuations = truncated_continuations.saturating_add(1);
                        self.compact_suppressed = false;
                        continue_hint = Some(TRUNCATED_OUTPUT_HINT);
                        ui.status(if circular {
                            "repeating review dump; continuing"
                        } else {
                            "model output truncated; continuing"
                        });
                        round += 1;
                        continue;
                    }
                    const MSG: &str = "model output was truncated twice with no complete reply";
                    ui.turn_error(
                        "truncated",
                        MSG,
                        "supervised sessions auto-repair; otherwise /retry",
                    );
                    self.session_usage.add(turn_usage);
                    ui.session_usage(self.session_usage);
                    self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                    self.close_persisted_turn(persisted_before, TurnStopReason::Error);
                    ui.changed_files(changed.clone());
                    ui.turn_end("truncated model output");
                    return Ok(TurnOutcome {
                        stop_reason: TurnStopReason::Error,
                        usage: turn_usage,
                        changed_files: changed,
                        error: Some(MSG.into()),
                        verification: None,
                    });
                }
                truncated_continuations = 0;
                if (promised_unfinished_work(&completion.text)
                    || (!mutated && claimed_unapplied_fixes(&completion.text)))
                    && promised_continuations < PROMISED_WORK_CONTINUATIONS
                {
                    if !completion.is_empty() {
                        self.messages.push(assistant_message(&completion));
                    }
                    promised_continuations = promised_continuations.saturating_add(1);
                    self.compact_suppressed = false;
                    continue_hint =
                        Some(if mutated || !claimed_unapplied_fixes(&completion.text) {
                            PROMISED_WORK_HINT
                        } else {
                            CLAIMED_FIX_HINT
                        });
                    ui.status(if !mutated && claimed_unapplied_fixes(&completion.text) {
                        "claimed fixes with no file changes; continuing"
                    } else {
                        "model described a fix without a tool call; continuing"
                    });
                    round += 1;
                    continue;
                }
                promised_continuations = 0;
                facts.visible_answer = !completion.text.trim().is_empty();
                facts.plan_open = completion::plan_is_open(&self.plan);
                let decision = completion::decide(Event::ModelStop, &mut facts);
                match decision {
                    Decision::Continue { demand, hint } => {
                        if !completion.reasoning.trim().is_empty() || facts.visible_answer {
                            self.messages.push(assistant_message(&completion));
                        }
                        facts.note_continue(demand);
                        self.compact_suppressed = false;
                        continue_hint = Some(hint);
                        ui.status(completion::status_for(demand));
                        round += 1;
                        continue;
                    }
                    Decision::Error { kind, message } => {
                        if !completion.reasoning.trim().is_empty() {
                            self.messages.push(assistant_message(&completion));
                        }
                        if kind == "empty_stop" {
                            hi_liveness::report_invariant(
                                &self.liveness,
                                hi_liveness::InvariantCode::EmptyAssistantAfterTools,
                            );
                        }
                        ui.turn_error(
                            kind,
                            message,
                            if kind == "empty_stop" {
                                "/retry to continue this turn"
                            } else {
                                "/retry to continue, or name the file to edit"
                            },
                        );
                        self.session_usage.add(turn_usage);
                        ui.session_usage(self.session_usage);
                        self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                        self.close_persisted_turn(persisted_before, TurnStopReason::Error);
                        ui.changed_files(changed.clone());
                        ui.turn_end(match kind {
                            "empty_stop" => "empty stop after tools",
                            "plan_stall" => "plan stalled without an edit",
                            _ => "inspect budget exhausted",
                        });
                        return Ok(TurnOutcome {
                            stop_reason: TurnStopReason::Error,
                            usage: turn_usage,
                            changed_files: changed,
                            error: Some(message.into()),
                            verification: None,
                        });
                    }
                    Decision::Proceed | Decision::Complete => {}
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

            truncated_continuations = 0;
            promised_continuations = 0;
            continue_hint = None;
            facts.used_tools = true;
            let mut mutated_this_round = false;
            let mut verify_this_round = false;
            self.messages.push(assistant_message(&completion));
            self.liveness
                .set_state(hi_liveness::HarnessState::ExecutingTool);
            let auto_hints = if managed_turn {
                Default::default()
            } else {
                self.score_tool_autos(&completion.tool_calls).await
            };
            for (i, call) in completion.tool_calls.iter().enumerate() {
                if !ui.dispatch_allowed() {
                    cancel.cancel();
                }
                if cancel.is_cancelled() {
                    if model == "pipe/auto" {
                        self.client
                            .managed_tool_complete(&call.id, &interrupted_outcome())?;
                    }
                    self.messages.push(Message::tool_result(
                        &call.id,
                        interrupted_outcome().content,
                    ));
                    continue;
                }
                let auto = auto_hints
                    .get(&call.id)
                    .copied()
                    .unwrap_or(AutoHint::Heuristic);
                let cached = if model == "pipe/auto" {
                    self.client.managed_tool_start(&call.id)?
                } else {
                    None
                };
                let outcome = if let Some(outcome) = cached {
                    outcome
                } else {
                    self.tools
                        .execute_with_auto(
                            &call.id,
                            &call.name,
                            &call.arguments,
                            || self.permission_mode(),
                            auto,
                            ui,
                        )
                        .await
                };
                if model == "pipe/auto" {
                    self.client.managed_tool_complete(&call.id, &outcome)?;
                }
                ui.tool_call_id(&call.id, &call.name, &call.arguments);
                if looks_like_verify(&call.name, &call.arguments) {
                    facts.ran_verify = true;
                    verify_this_round = true;
                    facts.grant_verify_credit();
                }
                if outcome.effects.mutation_applied {
                    mutated = true;
                    mutated_this_round = true;
                    facts.mutated = true;
                    facts.ran_verify = false;
                    facts.grant_mutation_credit();
                    self.liveness.note_progress();
                }
                if let Some(plan) = plan_from_outcome(&outcome) {
                    let plan = completion::accept_plan(&self.plan, plan, &mut facts.work_credits);
                    self.plan = plan.clone();
                    if let Some(session) = &mut self.session {
                        let _ = session.record_plan(&self.plan);
                    }
                    facts.plan_open = completion::plan_is_open(&self.plan);
                    ui.plan_result_id(
                        &call.id,
                        &call.name,
                        &outcome.content,
                        outcome.status,
                        &plan,
                    );
                } else {
                    ui.tool_result_id(&call.id, &call.name, outcome.ui_text(), outcome.status);
                }
                for change in &outcome.effects.file_changes {
                    if !changed.contains(&change.path) {
                        changed.push(change.path.clone());
                    }
                }
                self.last_changed_files = changed.clone();
                self.messages.push(Message::tool_result(
                    &call.id,
                    if model == "pipe/auto" {
                        crate::managed::reported_outcome(&outcome)
                    } else {
                        outcome.content.clone()
                    },
                ));
                if hi_tools::is_probe_refusal(&outcome.content) {
                    probe_refusals = probe_refusals.saturating_add(1);
                    repeat_stop = Some(repeat_stop_summary(&outcome.content));
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
                    for later in &completion.tool_calls[i + 1..] {
                        if managed_turn {
                            self.client
                                .managed_tool_complete(&later.id, &interrupted_outcome())?;
                        }
                        self.messages.push(Message::tool_result(
                            &later.id,
                            interrupted_outcome().content,
                        ));
                    }
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
            // Persist only after every tool_call has a result. Writing the
            // assistant first leaves a truncated round on disk; resume would
            // replay it and Pipe would reject the request.
            self.persist_turn_progress(&mut persisted_before);
            if mutated_this_round && !ui.mutation_batch_complete().await {
                cancel.cancel();
            }
            if verify_this_round {
                facts.grant_verify_credit();
            }
            facts.plan_open = completion::plan_is_open(&self.plan);
            let stall_only = stall_tool_round(&completion.tool_calls);
            let coordination_only = coordination_only_round(&completion.tool_calls);
            let pagination_only =
                read_pagination_only(&completion.tool_calls, &facts.seen_read_paths);
            facts.observe_read_paths(&completion.tool_calls);
            facts.note_round_kind(
                mutated_this_round,
                stall_only,
                verify_this_round,
                coordination_only,
                pagination_only,
            );
            let inspect_repeat_capped = probe_refusals >= hi_tools::STOP_AFTER_PROBE_REFUSALS;
            let mut decision = completion::decide(
                Event::ToolsFinished {
                    inspect_repeat_capped,
                },
                &mut facts,
            );
            let mut typesafe_label: Option<&'static str> = None;
            if let Decision::Continue { demand, .. } = decision
                && matches!(demand, Demand::Edit | Demand::Verify | Demand::Verdict)
                && stall_only
                && !managed_turn
                && typesafe_calls < 2
            {
                typesafe_calls = typesafe_calls.saturating_add(1);
                let state = crate::typesafe::GateState {
                    asked_to_fix: matches!(facts.intent, Intent::Fix | Intent::Implement),
                    mutated: facts.mutated,
                    ran_verify: facts.ran_verify,
                    inspect_only: stall_only,
                    plan_open: facts.plan_open,
                    last_tools: completion
                        .tool_calls
                        .iter()
                        .map(|call| call.name.clone())
                        .collect(),
                    user_prompt: latest_user_prompt(&self.messages, input),
                };
                if let Some(action) = self.next_action_decision(&state).await {
                    let flavored = completion::flavor_demand(demand, action, &facts);
                    if flavored != demand {
                        decision = Decision::Continue {
                            demand: flavored,
                            hint: completion::hint_for(flavored),
                        };
                    }
                    // Only surface the TypeSafe label when the applied demand
                    // matches what it asked for. A rejected verify/verdict on
                    // an unstarted plan must not look like the harness agreed.
                    if completion::demand_from_typesafe(action) == Some(flavored) {
                        typesafe_label = Some(action.as_str());
                    }
                }
            }
            match decision {
                Decision::Continue { demand, hint } => {
                    if inspect_repeat_capped {
                        probe_refusals = 0;
                    }
                    facts.note_continue(demand);
                    self.compact_suppressed = false;
                    continue_hint = Some(hint);
                    ui.status(completion::status_for(demand));
                    if let Some(label) = typesafe_label {
                        ui.status(&format!("typesafe next-action: {label}"));
                    }
                    round += 1;
                    continue;
                }
                Decision::Error { kind, message } => {
                    if kind == "empty_stop" {
                        hi_liveness::report_invariant(
                            &self.liveness,
                            hi_liveness::InvariantCode::EmptyAssistantAfterTools,
                        );
                    }
                    ui.turn_error(
                        kind,
                        message,
                        if kind == "plan_stall" {
                            "supervised sessions auto-repair; otherwise /retry"
                        } else {
                            "/retry to continue, or name the file to edit"
                        },
                    );
                    self.session_usage.add(turn_usage);
                    ui.session_usage(self.session_usage);
                    self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                    self.close_persisted_turn(persisted_before, TurnStopReason::Error);
                    ui.changed_files(changed.clone());
                    ui.turn_end(match kind {
                        "empty_stop" => "empty stop after tools",
                        "plan_stall" => "plan stalled without an edit",
                        _ => "inspect budget exhausted",
                    });
                    return Ok(TurnOutcome {
                        stop_reason: TurnStopReason::Error,
                        usage: turn_usage,
                        changed_files: changed,
                        error: Some(message.into()),
                        verification: None,
                    });
                }
                Decision::Complete => {
                    self.session_usage.add(turn_usage);
                    ui.session_usage(self.session_usage);
                    self.seal_checkpoint(pre.as_deref(), mutated, ui).await;
                    self.close_persisted_turn(persisted_before, TurnStopReason::Completed);
                    ui.changed_files(changed.clone());
                    ui.turn_end(repeat_stop.unwrap_or("stopped repeating a detached binary probe"));
                    return Ok(TurnOutcome {
                        stop_reason: TurnStopReason::Completed,
                        usage: turn_usage,
                        changed_files: changed,
                        error: None,
                        verification: None,
                    });
                }
                Decision::Proceed => {}
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

    /// `/compact`: Jev prune when enabled, else cheap-shrink, then a model
    /// summary only if occupancy is still ≥85% after prune/shrink (or Jev
    /// did not apply). Auto-reclaim uses [`Self::reclaim_context`] with
    /// `force_model = false` so a successful shrink under 85% skips Pipe.
    pub async fn compact(
        &mut self,
        user_context: Option<&str>,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Result<bool> {
        if self.model() == "pipe/auto" {
            self.client.begin_managed_auxiliary()?;
        }
        self.reclaim_context(user_context, ui, cancel, true).await
    }

    async fn apply_jev_prune(
        &mut self,
        user_context: Option<&str>,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
    ) -> Result<bool, String> {
        let Some(client) = self.typesafe.client() else {
            return Err("no TypeSafe key".into());
        };
        ui.status("jev-compact scoring tools");
        let outcome =
            crate::jev_compact::prune_messages(&self.messages, &client, user_context, cancel)
                .await?;
        if !outcome.changed {
            return Ok(false);
        }
        let forget = crate::compact::inspect_keys_to_forget(&self.messages, &outcome.messages);
        if !self.replace_messages_persisting(outcome.messages) {
            return Err("session rewrite failed".into());
        }
        self.tools.forget_inspect_keys(&forget);
        ui.status(&outcome.stats.status_line());
        Ok(true)
    }

    async fn reclaim_context(
        &mut self,
        user_context: Option<&str>,
        ui: &mut dyn Ui,
        cancel: &TurnCancellation,
        force_model: bool,
    ) -> Result<bool> {
        if self.messages.len() < 4 {
            return Ok(false);
        }
        let mut changed = false;
        let mut jev_applied = false;
        if self.model() != "pipe/auto" && self.jev_compact_ready() {
            match self.apply_jev_prune(user_context, ui, cancel).await {
                Ok(true) => {
                    changed = true;
                    jev_applied = true;
                }
                Ok(false) => {}
                Err(reason) => {
                    ui.status(&format!("jev-compact skipped: {reason}"));
                }
            }
        }
        if !jev_applied {
            changed |= self.apply_cheap_shrink_if_needed(ui);
            if self.occupancy_percent() >= AUTO_COMPACT_THRESHOLD_PERCENT {
                changed |= self.apply_cheap_shrink(ui, false);
            }
        }
        if jev_applied && self.occupancy_percent() < AUTO_COMPACT_THRESHOLD_PERCENT {
            return Ok(changed);
        }
        if !force_model && self.occupancy_percent() < AUTO_COMPACT_THRESHOLD_PERCENT {
            return Ok(changed);
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
        let streamed = self
            .client
            .stream(
                &model,
                &request,
                &[],
                self.max_tokens(),
                self.reasoning_effort(),
                &mut on_delta,
                cancel,
            )
            .await;
        self.liveness
            .emit(hi_liveness::EventCode::CompactEnd, None, None, None);
        self.liveness.set_state(previous);
        let completion = match streamed {
            Ok(completion) => completion,
            Err(err) => {
                if self.emergency_compact_if_over_window(ui) {
                    return Ok(true);
                }
                return Err(err);
            }
        };
        self.record_context_occupancy(completion.usage);
        if !completion.tool_calls.is_empty() {
            if self.emergency_compact_if_over_window(ui) {
                return Ok(true);
            }
            bail!("compaction model called a tool; refusing to replace history");
        }
        let Some(summary) = parse_summary(&completion.text) else {
            if self.emergency_compact_if_over_window(ui) {
                return Ok(true);
            }
            bail!("compaction model did not return a <summary> block");
        };
        let mut next = apply_summary(&self.messages, &summary);
        if model == "pipe/auto" {
            crate::managed::retain_execution_evidence(&self.messages, &mut next);
        }
        if !self.replace_messages_persisting(next) {
            return Ok(changed);
        }
        ui.status("compacted conversation");
        self.compact_suppressed = false;
        Ok(true)
    }

    fn emergency_compact_if_over_window(&mut self, ui: &mut dyn Ui) -> bool {
        if self.occupancy_percent() < AUTO_COMPACT_THRESHOLD_PERCENT {
            return false;
        }
        hi_liveness::report_invariant(
            &self.liveness,
            hi_liveness::InvariantCode::CompactFailedOverWindow,
        );
        let summary = emergency_summary(self.occupancy_percent());
        let mut next = apply_summary(&self.messages, &summary);
        if self.model() == "pipe/auto" {
            crate::managed::retain_execution_evidence(&self.messages, &mut next);
        }
        if !self.replace_messages_persisting(next) {
            return false;
        }
        ui.status("emergency compacted conversation");
        self.compact_suppressed = false;
        true
    }

    pub(crate) fn occupancy_percent(&self) -> u64 {
        let window = u64::from(self.working_context_window());
        self.current_occupancy().saturating_mul(100) / window
    }

    pub(crate) fn should_auto_compact(&self) -> bool {
        self.auto_compact
            && !self.compact_suppressed
            && self.messages.len() >= 4
            && self.occupancy_percent() >= AUTO_COMPACT_THRESHOLD_PERCENT
    }

    pub(crate) fn should_cheap_shrink(&self) -> bool {
        self.occupancy_percent() >= CHEAP_SHRINK_THRESHOLD_PERCENT
    }

    /// Stub old tool bodies when occupancy is high. Returns true only after
    /// the session file is rewritten; the caller then treats `messages.len()`
    /// as the persist cursor. A failed rewrite restores the previous messages.
    pub(crate) fn apply_cheap_shrink_if_needed(&mut self, ui: &mut dyn Ui) -> bool {
        let preserve_current_turn = !self.should_stub_current_turn();
        self.apply_cheap_shrink(ui, preserve_current_turn)
    }

    pub(crate) fn should_stub_current_turn(&self) -> bool {
        self.occupancy_percent() >= AUTO_COMPACT_THRESHOLD_PERCENT
            || crate::compact::current_turn_tool_tokens(&self.messages)
                > crate::compact::CURRENT_TURN_TOOL_BUDGET_TOKENS
    }

    fn apply_cheap_shrink(&mut self, ui: &mut dyn Ui, preserve_current_turn: bool) -> bool {
        if preserve_current_turn && !self.should_cheap_shrink() {
            return false;
        }
        let (mut next, shrunk) = cheap_shrink_with(&self.messages, preserve_current_turn);
        if self.model() == "pipe/auto" {
            crate::managed::preserve_result_receipts(&self.messages, &mut next);
        }
        if !shrunk {
            return false;
        }
        let forget = crate::compact::inspect_keys_to_forget(&self.messages, &next);
        if !self.replace_messages_persisting(next) {
            return false;
        }
        self.tools.forget_inspect_keys(&forget);
        ui.status("shrunk tool results");
        true
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

const TRUNCATED_OUTPUT_CONTINUATIONS: u32 = 2;
const PROMISED_WORK_CONTINUATIONS: u32 = 2;

const TRUNCATED_OUTPUT_HINT: &str = "\
Your previous reply was truncated at the output token limit. Do not repeat \
that monologue. Finish with tool calls if work remains, or a short verdict \
for the user. Stop when the answer is complete.";

const PROMISED_WORK_HINT: &str = "\
You described a fix or further inspection but did not call a tool. Call \
`edit`/`write`/`bash` now to make the change, or if nothing needs changing, \
give a short verdict and stop. Do not only announce the next step.";

const CLAIMED_FIX_HINT: &str = "\
You claimed to have applied fixes but no file was changed this turn. Call \
`edit`/`write` now to apply them, or retract the claim with a short verdict. \
Do not only describe the patches.";

const CIRCULAR_DUMP_STUB: &str = "\
[omitted a truncated repeating review list. Do not continue numbering. \
Call tools to apply remaining fixes, or give a short verdict.]";

fn promised_unfinished_work(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let plans = [
        "let me fix",
        "let me also",
        "i'll fix",
        "i will fix",
        "i am going to fix",
        "going to fix",
        "let me edit",
        "i'll edit",
        "i will edit",
        "let me patch",
        "let me add",
        "i'll add",
        "i will add",
        "let me implement",
        "i'll implement",
        "i will implement",
        "i'll now",
        "next i'll",
    ];
    plans.iter().any(|needle| lower.contains(needle))
}

fn claimed_unapplied_fixes(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "fixes applied",
        "i've fixed",
        "i have fixed",
        "found and fixed",
        "i've applied",
        "i applied the",
        "all fixes applied",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn repeating_review_dump(text: &str) -> bool {
    if text.len() < 4_000 {
        return false;
    }
    let mut bodies = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        let rest = trimmed
            .strip_prefix("### ")
            .or_else(|| trimmed.strip_prefix("## "));
        let Some(rest) = rest else {
            continue;
        };
        let rest = rest.trim_start_matches('*').trim();
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            continue;
        }
        let after = rest[digits..].trim_start_matches('.').trim();
        let body: String = after.chars().take(80).collect();
        if body.len() >= 20 {
            bodies.push(body);
        }
    }
    if bodies.len() >= 12 {
        let mut counts = std::collections::HashMap::<&str, usize>::new();
        for body in &bodies {
            *counts.entry(body.as_str()).or_insert(0) += 1;
        }
        if counts.values().any(|count| *count >= 4) {
            return true;
        }
    }
    let mut seen = std::collections::HashMap::<&str, usize>::new();
    for line in text.lines() {
        let line = line.trim();
        if line.len() < 80 {
            continue;
        }
        *seen.entry(line).or_insert(0) += 1;
    }
    seen.values().any(|count| *count >= 3)
}

fn assistant_message_for_stop(completion: &PipeCompletion, circular: bool) -> Message {
    if circular {
        let mut slim = completion.clone();
        slim.text = CIRCULAR_DUMP_STUB.into();
        slim.reasoning.clear();
        assistant_message(&slim)
    } else {
        assistant_message(completion)
    }
}

fn latest_user_prompt(messages: &[Message], fallback: &str) -> String {
    let text = messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .map(|message| message.text())
        .unwrap_or_else(|| fallback.to_string());
    let trimmed = text.trim();
    const MAX: usize = 240;
    if trimmed.chars().count() <= MAX {
        trimmed.to_string()
    } else {
        trimmed.chars().take(MAX).collect()
    }
}

fn turn_intent_prompt() -> Option<String> {
    let path = std::env::var_os(ENV_TURN_INTENT)?;
    let bytes = std::fs::read(path).ok()?;
    let intent: TurnIntent = serde_json::from_slice(&bytes).ok()?;
    Some(intent.prompt)
}

pub(crate) fn assistant_message(completion: &PipeCompletion) -> Message {
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

fn repeat_stop_summary(content: &str) -> &'static str {
    if content.contains("detached `target/debug") {
        "stopped repeating a detached binary probe"
    } else {
        "stopped repeating the same inspect"
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

#[cfg(test)]
mod helper_tests {
    use super::repeat_stop_summary;
    use super::{claimed_unapplied_fixes, repeating_review_dump};
    use crate::completion::{
        Intent, latest_prompt_text, looks_like_verify, user_asked_to_fix, user_asked_to_implement,
    };
    use hi_ai::Message;

    #[test]
    fn user_asked_to_fix_matches_review_and_repair_prompts() {
        assert!(user_asked_to_fix("review for any major issues and fix"));
        assert!(user_asked_to_fix("cargo check and fix any compile errors"));
        assert!(!user_asked_to_fix(
            "Run cargo test to verify the review change didn't break anything."
        ));
        assert!(!user_asked_to_fix("look at the fixture"));
        assert!(user_asked_to_implement("do all of that"));
        assert!(user_asked_to_implement("build all of that"));
        assert!(user_asked_to_implement(
            "please implement the metrics endpoint"
        ));
        assert!(!user_asked_to_implement("how can we improve this"));
        assert!(!user_asked_to_implement("how do those handlers work"));
        assert!(!user_asked_to_implement(
            "Post a step-by-step plan. Do not implement yet."
        ));
    }

    #[test]
    fn latest_user_prompt_drives_fix_and_implement_intent() {
        let older = vec![Message::user("review for any major issues and fix")];
        assert_eq!(
            Intent::from_prompt(&latest_prompt_text(
                &older,
                "review for any major issues and fix"
            )),
            Intent::Fix
        );
        assert_eq!(
            Intent::from_prompt(&latest_prompt_text(&older, "how can we improve this")),
            Intent::Review
        );
        assert_eq!(
            Intent::from_prompt(&latest_prompt_text(&older, "")),
            Intent::Fix
        );
        assert_eq!(
            Intent::from_prompt(&latest_prompt_text(&older, "do all of that")),
            Intent::Implement
        );
    }

    #[test]
    fn repeat_stop_summary_distinguishes_inspect_from_probes() {
        assert_eq!(
            repeat_stop_summary(
                "This detached `target/debug/… & sleep …` probe already ran this turn (3 times)."
            ),
            "stopped repeating a detached binary probe"
        );
        assert_eq!(
            repeat_stop_summary(
                "This exact `grep` already ran this turn (2 times) and returned:\nno matches"
            ),
            "stopped repeating the same inspect"
        );
    }

    #[test]
    fn tool_looks_like_verify_detects_cargo_test() {
        assert!(looks_like_verify(
            "bash",
            r#"{"command":"cargo test --offline"}"#
        ));
        assert!(!looks_like_verify("read", r#"{"path":"src/main.rs"}"#));
        assert!(!looks_like_verify("bash", r#"{"command":"ls"}"#));
    }

    #[test]
    fn repeating_review_dump_detects_alternating_numbered_items() {
        let mut dump = String::from("I've reviewed the codebase. Issues found:\n\n");
        for i in 1..=40 {
            if i % 2 == 0 {
                dump.push_str(&format!(
                    "### {i}. **`src/ws.rs` — `handle_ws` doesn't check that the channel from the path is valid**\nAlready checked. Fine.\n\n"
                ));
            } else {
                dump.push_str(&format!(
                    "### {i}. **`src/ws.rs` — `handle_ws` doesn't check that the ticket is not empty**\nAlready checked. Fine.\n\n"
                ));
            }
        }
        assert!(repeating_review_dump(&dump));
        assert!(!repeating_review_dump("short recap"));
        assert!(claimed_unapplied_fixes(
            "Fixes applied\n\nI've fixed the two real bugs in ws.rs."
        ));
        assert!(!claimed_unapplied_fixes(
            "I reviewed the tree. No changes were necessary."
        ));
    }
}
