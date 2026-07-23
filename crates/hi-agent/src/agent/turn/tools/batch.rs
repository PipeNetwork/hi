//! Tool-batch execution for one model round (TurnPhase::Tools).
//!
//! Extracted from the main turn loop so orchestration stays thin: run the
//! dep-aware scheduler, record results, attach fast-feedback, then return
//! batch stats for the post-tool Steer phase.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use futures_util::StreamExt;
use hi_tools::protocol::{
    execute_in_runtime, execute_prepared_in_runtime, execute_streaming_in_runtime,
    prepare_mutation_in_with_state,
};

use crate::heuristics::{emit_tool_output, mode_blocks_tool, respects_deps, tool_deps};
use crate::steering::{
    EvidenceTracker, ImplementationTracker, ToolLoopGuardrail, bash_call_waits, bash_command,
    inspection_signature, read_only_blocked_tool_result, read_only_blocks_tool,
};
use crate::transcript::NudgeKind;
use crate::verify::Snapshot;
use crate::{ConfirmationRequest, ConfirmationResult, TaskContract, Ui};
use hi_ai::Content;

use crate::agent::explore_turn::{
    BufferingUi, ExploreJob, MAX_EXPLORE_SUBAGENTS_PER_SESSION, MAX_PARALLEL_EXPLORES,
    explore_tool_outcome, run_explore_job,
};
use crate::agent::delegate_turn::{
    DelegateJob, MAX_DELEGATE_SUBAGENTS_PER_SESSION, MAX_PARALLEL_DELEGATES, file_sets_disjoint,
    run_delegate_job,
};

use crate::apply_plan_to_goal;
use crate::heuristics::plan_has_pending_steps;
use crate::steering::implementation_tool_call_mutates;

use super::super::helpers::{synthetic_tool_outcome, tool_entry, tool_satisfies_validation};
use super::super::phase::TurnPhase;
use super::super::progress::{
    ProgressKind, ProgressTracker, ToolProgressLabel, classify_tool_progress, signature_seen,
};

/// Outcomes and counters produced by one Tools-phase batch.
pub(in crate::agent::turn) struct ToolBatchOutcome {
    pub(in crate::agent::turn) hash_guard_applies: bool,
    pub(in crate::agent::turn) hashable_idempotent_results: usize,
    pub(in crate::agent::turn) repeated_idempotent_results: usize,
    /// How many results in this batch were idle `bash_output` polls
    /// (running, no new output). Used to pick the tight-poll nudge.
    pub(in crate::agent::turn) idle_background_poll_results: usize,
    pub(in crate::agent::turn) tool_progress_labels: Vec<ToolProgressLabel>,
    pub(in crate::agent::turn) plan_changed_this_batch: bool,
    pub(in crate::agent::turn) interrupted_calls: usize,
    pub(in crate::agent::turn) interrupted_coordination_calls: usize,
}

impl crate::Agent {
    /// Execute `calls` for the current round and append assistant+results.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent::turn) async fn execute_tool_batch(
        &mut self,
        calls: &[(String, String, String)],
        completion_content: &mut Vec<Content>,
        tool_specs: &[hi_ai::ToolSpec],
        read_only_intent: Option<crate::steering::ReviewIntent>,
        max_parallel_tools: usize,
        task_contract: &TaskContract,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        tool_guardrail: &mut ToolLoopGuardrail,
        progress_tracker: &mut ProgressTracker,
        tool_timeline: &mut Vec<crate::ToolCallEntry>,
        sched_tool_calls: &mut u32,
        sched_max_concurrent: &mut u32,
        sched_serial_runs: &mut u32,
        plan_updated_goal: &mut bool,
        proposed_goal: &mut Option<crate::Goal>,
        turn_snapshot: &mut Option<Snapshot>,
        turn_checkpoint_allowed: &mut Option<bool>,
        turn_checkpoint_created: &mut bool,
        fast_feedback: &mut super::super::fast_feedback::FastFeedbackState,
        ui: &mut dyn Ui,
    ) -> Result<ToolBatchOutcome> {
        // The batch has not announced any of its tools yet, so an interrupt
        // already present here can only belong to the previously visible tool
        // (most notably a preflight whose result was still queued in the TUI).
        // Never let that stale signal cancel the model's next action.
        self.interrupt
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let hash_guard_applies = calls.iter().all(|(_, name, args)| {
            matches!(
                name.as_str(),
                "read" | "list" | "grep" | "glob" | "bash_output"
            ) || (name == "bash" && bash_call_waits(args))
        });
        let mut hashable_idempotent_results = 0usize;
        let mut repeated_idempotent_results = 0usize;
        let mut idle_background_poll_results = 0usize;
        self.set_turn_phase(TurnPhase::Tools);
        let mut tool_progress_labels: Vec<ToolProgressLabel> = Vec::new();
        let mut plan_changed_this_batch = false;
        let mut interrupted_calls = 0usize;
        let mut interrupted_coordination_calls = 0usize;
        // Infer within-batch dependencies (a read of a file a mutating
        // call earlier in the batch targeted must observe that mutation;
        // mutating calls serialize). The scheduler below runs ready
        // calls concurrently respecting this graph, so independent reads
        // can overlap with an independent later write — while a read
        // whose path matches an earlier write waits for it.
        let deps = tool_deps(calls);
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
        // Reserve the remaining hard tool budget for the model-ordered
        // prefix before any ready batch is dispatched. Calls beyond
        // this prefix receive typed denials and are never executed.
        let permitted_prefix = calls.len().min(
            self.config
                .loop_limits
                .max_tool_calls
                .saturating_sub(*sched_tool_calls) as usize,
        );
        let budget_denied = calls.len().saturating_sub(permitted_prefix);
        for (i, (id, name, arguments)) in calls.iter().enumerate().skip(permitted_prefix) {
            ui.tool_call_id(id, name, arguments);
            let content = serde_json::json!({
                "error": {
                    "kind": "tool_budget_exhausted",
                    "message": "tool call denied: per-turn tool budget exhausted"
                }
            })
            .to_string();
            let output = synthetic_tool_outcome(content.clone(), hi_tools::ToolStatus::Denied);
            emit_tool_output(&mut *ui, id, name, &output);
            let progress_label = ToolProgressLabel::new(
                ProgressKind::None,
                "tool denied by hard budget",
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
        // Pre-pass: resolve calls blocked by read-only intent up front.
        // They produce instant synthetic error results and mutate
        // nothing, so completing them out of dep order is safe.
        // (`explore`/`delegate`/`record_decision` used to run here too,
        // but they *do* have deps that matter — running a subagent
        // before an earlier `write` in the same batch handed it a stale
        // tree — so they now dispatch inside the dep-aware scheduler
        // loop below.)
        for (i, (id, name, arguments)) in calls.iter().enumerate().take(permitted_prefix) {
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
                // Use the session tool_mode, not the per-request mode: text-tool
                // fallback sets request mode to ChatOnly so the provider won't emit
                // structured calls, but promoted prose calls must still execute.
                mode_blocks_tool(self.config.routing.tool_mode, name)
            };
            if let Some(content) = blocked {
                ui.tool_call_id(id, name, arguments);
                let mut output =
                    synthetic_tool_outcome(content.clone(), hi_tools::ToolStatus::Denied);
                output.effects.mutation_attempted =
                    implementation_tool_call_mutates(name, arguments);
                emit_tool_output(&mut *ui, id, name, &output);
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
        // Calls that survived policy/budget denial are about to cross the
        // local execution boundary. Validate the declared Draft
        // 2020-12 schema here: malformed model output receives a typed tool
        // result and can never reach a workspace-mutating executor.
        let batch_validation_error = hi_ai::validate_client_tool_batch_limits(
            calls
                .iter()
                .enumerate()
                .filter(|(index, _)| !completed[*index])
                .map(|(_, (_, _, arguments))| arguments.as_str()),
        )
        .err();
        for (i, (id, name, arguments)) in calls.iter().enumerate().take(permitted_prefix) {
            if completed[i] {
                continue;
            }
            let error = match batch_validation_error.clone() {
                Some(error) => error,
                None => match hi_ai::validate_client_tool_call(id, name, arguments, tool_specs) {
                    Ok(()) => continue,
                    Err(error) => error,
                },
            };
            ui.tool_call_id(id, name, arguments);
            let content = serde_json::json!({
                "error": {
                    "kind": "tool_protocol_error",
                    "message": error.to_string(),
                }
            })
            .to_string();
            let output = synthetic_tool_outcome(content.clone(), hi_tools::ToolStatus::Denied);
            emit_tool_output(&mut *ui, id, name, &output);
            let progress_label = ToolProgressLabel::new(
                ProgressKind::None,
                "tool denied by protocol validation",
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
        let mut done = completion_order.len();
        let initially_executed = done.saturating_sub(budget_denied) as u32;
        if initially_executed > 0 {
            *sched_tool_calls = (*sched_tool_calls).saturating_add(initially_executed);
            *sched_serial_runs = (*sched_serial_runs).saturating_add(initially_executed);
            *sched_max_concurrent = (*sched_max_concurrent).max(1);
        }
        // Proactive per-edit checks: kicked off in the background as
        // mutating calls complete, awaited after the batch so any
        // syntax/lint error surfaces during the turn (before turn-end
        // verify) while the edit is still the model's focus. Each entry
        // is (path, join handle of the check).
        let mut pending_checks: Vec<(String, tokio::task::JoinHandle<(bool, String)>)> = Vec::new();
        // Project-relative paths mutated in this tool batch — drives
        // mid-turn LSP diagnostics + affected cargo check.
        let mut batch_mutated_paths: BTreeSet<String> = BTreeSet::new();
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
                        ui.tool_call_id(id, name, arguments);
                        let msg = "Tool call interrupted by user.".to_string();
                        let mut output =
                            synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Cancelled);
                        output.effects.mutation_attempted =
                            implementation_tool_call_mutates(name, arguments);
                        emit_tool_output(&mut *ui, id, name, &output);
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
                        interrupted_calls = interrupted_calls.saturating_add(1);
                        if hi_tools::is_coordination(name) {
                            interrupted_coordination_calls =
                                interrupted_coordination_calls.saturating_add(1);
                        }
                    }
                }
                *sched_tool_calls = (*sched_tool_calls).saturating_add(interrupted);
                *sched_serial_runs = (*sched_serial_runs).saturating_add(interrupted);
                *sched_max_concurrent = (*sched_max_concurrent).max(1);
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
                let unresolved: Vec<usize> = (0..calls.len()).filter(|&i| !completed[i]).collect();
                scheduler_forced_skip = true;
                ui.status(
                    "⚠ tool scheduler could not make progress; marking unresolved calls as skipped",
                );
                *sched_tool_calls += unresolved.len() as u32;
                for i in unresolved {
                    let (id, name, arguments) = &calls[i];
                    ui.tool_call_id(id, name, arguments);
                    let msg = "Tool scheduler could not make progress; this call was skipped to keep the transcript valid.".to_string();
                    let mut output =
                        synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Cancelled);
                    output.effects.mutation_attempted =
                        implementation_tool_call_mutates(name, arguments);
                    emit_tool_output(&mut *ui, id, name, &output);
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
                if self.config.gates.confirm_edits && bash_mutates {
                    let command = bash_command(arguments).unwrap_or_else(|| arguments.clone());
                    let cwd = self.runtime.root().display().to_string();
                    let decision = ui
                        .confirm(ConfirmationRequest::ShellMutation { command, cwd })
                        .await;
                    if decision != ConfirmationResult::Approved {
                        ui.tool_call_id(id, name, arguments);
                        let msg = if decision == ConfirmationResult::Unavailable {
                    "Shell mutation skipped: confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits."
                } else {
                    "Shell mutation skipped by user (not run)."
                }
                .to_string();
                        let mut output =
                            synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                        output.effects.mutation_attempted = true;
                        emit_tool_output(&mut *ui, id, name, &output);
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
                        *sched_tool_calls += 1;
                        *sched_serial_runs += 1;
                        *sched_max_concurrent = (*sched_max_concurrent).max(1);
                        continue;
                    }
                }
                // Bash is opaque: an apparently read-only script or test
                // can still rewrite files. Capture both the change
                // baseline and undo checkpoint before every shell run;
                // the mutation classifier is only a confirmation hint.
                self.ensure_turn_snapshot(turn_snapshot).await?;
                if !self
                    .ensure_turn_checkpoint(turn_checkpoint_allowed, turn_checkpoint_created, ui)
                    .await
                {
                    ui.tool_call_id(id, name, arguments);
                    let msg = "Shell mutation skipped because strict mode requires an available checkpoint.".to_string();
                    let mut output =
                        synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                    output.effects.mutation_attempted = true;
                    emit_tool_output(&mut *ui, id, name, &output);
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
                    *sched_tool_calls += 1;
                    *sched_serial_runs += 1;
                    *sched_max_concurrent = (*sched_max_concurrent).max(1);
                    continue;
                }
                ui.tool_started_id(id, name, arguments);
                ui.tool_call_id(id, name, arguments);
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
                    self.runtime.repo_map(),
                    name,
                    arguments,
                    &mut |line: &str| ui_ref.tool_stream_id(id, name, line),
                )
                .await;
                let duration_ms = started.elapsed().as_millis() as u64;
                self.record_tool_effects(&output.effects)?;
                self.reconcile_workspace_changes().await?;
                for change in &output.effects.file_changes {
                    batch_mutated_paths.insert(change.path.clone());
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
                let progress = tool_guardrail.record_tool_result(name, arguments, &semantic_output);
                if progress.idle_background_poll {
                    idle_background_poll_results += 1;
                }
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
                emit_tool_output(&mut *ui, id, name, &output);
                results[i] = Some((id.clone(), output.content));
                self.invalidate_snapshot();
                completed[i] = true;
                completion_order.push(i);
                done += 1;
                // Bash runs alone → a serial run and a batch of size 1.
                *sched_tool_calls += 1;
                *sched_serial_runs += 1;
                *sched_max_concurrent = (*sched_max_concurrent).max(1);
                continue;
            }
            // Parallel explore: `explore` is read-only and independent —
            // multiple ready explores can run concurrently. Prepare all jobs
            // (budget check, config extraction) sequentially, run the child
            // turns in parallel via `FuturesUnordered`, then process results
            // sequentially. Each explore buffers its UI output and replays it
            // to the real UI after completion, so `&mut dyn Ui` is never
            // shared across concurrent futures.
            let explore_indices: Vec<usize> = ready
                .iter()
                .copied()
                .filter(|&i| calls[i].1 == "explore")
                .collect();
            if explore_indices.len() > 1 {
                // Prepare jobs for all ready explores (budget permitting).
                let mut prepared: Vec<(usize, ExploreJob)> = Vec::new();
                let mut budget_denied_explores: Vec<usize> = Vec::new();
                for &i in &explore_indices {
                    let (id, _, arguments) = &calls[i];
                    if let Some(job) = self.prepare_explore(arguments) {
                        let summary: String = job.task.chars().take(72).collect();
                        let ellipsis = if job.task.chars().count() > 72 { "…" } else { "" };
                        ui.subagent_note(&format!(
                            "↳ explore subagent {}/{MAX_EXPLORE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}",
                            job.slot,
                        ));
                        ui.tool_call_id(id, "explore", arguments);
                        prepared.push((i, job));
                    } else {
                        budget_denied_explores.push(i);
                    }
                }
                // Complete budget-denied explores immediately.
                for i in budget_denied_explores {
                    let (id, _, arguments) = &calls[i];
                    ui.tool_call_id(id, "explore", arguments);
                    let msg = format!(
                        "explore budget exhausted ({MAX_EXPLORE_SUBAGENTS_PER_SESSION} subagents \
                         this session); investigate directly instead."
                    );
                    let output = explore_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                    emit_tool_output(&mut *ui, id, "explore", &output);
                    let signature = inspection_signature("explore", arguments);
                    let progress_label = ToolProgressLabel::new(
                        ProgressKind::Weak,
                        "explore budget exhausted",
                        signature,
                    );
                    progress_tracker.record_tool(&progress_label);
                    tool_progress_labels.push(progress_label.clone());
                    tool_timeline.push(tool_entry(
                        "explore".to_string(),
                        String::new(),
                        0,
                        &output,
                        &progress_label,
                    ));
                    results[i] = Some((id.clone(), msg));
                    completed[i] = true;
                    completion_order.push(i);
                    done += 1;
                    *sched_tool_calls += 1;
                    *sched_serial_runs += 1;
                }
                // Run all prepared explores concurrently. Cap with a semaphore
                // to avoid spawning too many child Agents at once.
                let max_concurrent = MAX_PARALLEL_EXPLORES.min(prepared.len());
                let started = std::time::Instant::now();
                let mut explore_futures = futures_util::stream::iter(
                    prepared
                        .into_iter()
                        .map(|(i, job)| {
                            let mut buf_ui = BufferingUi::new();
                            // Move buf_ui into the future, return (i, buf_ui, result).
                            async move {
                                let result = run_explore_job(job, &mut buf_ui).await;
                                (i, buf_ui, result)
                            }
                        }),
                )
                .buffer_unordered(max_concurrent)
                .collect::<Vec<_>>()
                .await;
                // Sort by index so results are processed in model order.
                explore_futures.sort_by_key(|(i, _, _)| *i);
                for (i, mut buf_ui, result) in explore_futures {
                    let (id, _, arguments) = &calls[i];
                    // Replay buffered UI events to the real UI.
                    buf_ui.replay_to(&mut *ui);
                    // Fold usage and emit completion callout.
                    let output = self.finish_explore(result);
                    ui.subagent_note(&format!(
                        "↳ explore subagent done"
                    ));
                    let duration_ms = started.elapsed().as_millis() as u64;
                    let error = output.status != hi_tools::ToolStatus::Succeeded;
                    let semantic_output = if error && !output.content.starts_with("Error:") {
                        std::borrow::Cow::Owned(format!("Error: {}", output.content))
                    } else {
                        std::borrow::Cow::Borrowed(output.content.as_str())
                    };
                    let signature = inspection_signature("explore", arguments);
                    let signature_was_seen = signature_seen(&evidence, &signature);
                    let tracker_before = implementation_tracker.clone();
                    let validation_succeeded = tool_satisfies_validation(&output);
                    evidence.record_success("explore", arguments, &semantic_output);
                    implementation_tracker.record_tool_result(
                        "explore",
                        arguments,
                        &semantic_output,
                        validation_succeeded,
                    );
                    let progress = tool_guardrail.record_tool_result("explore", arguments, &semantic_output);
                    if progress.idle_background_poll {
                        idle_background_poll_results += 1;
                    }
                    if progress.hashable_idempotent {
                        hashable_idempotent_results += 1;
                        if progress.repeated_idempotent_result {
                            repeated_idempotent_results += 1;
                        }
                    }
                    let progress_label = classify_tool_progress(
                        "explore",
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
                        "explore".to_string(),
                        String::new(),
                        duration_ms,
                        &output,
                        &progress_label,
                    ));
                    emit_tool_output(&mut *ui, id, "explore", &output);
                    results[i] = Some((id.clone(), output.content));
                    completed[i] = true;
                    completion_order.push(i);
                    done += 1;
                    *sched_tool_calls += 1;
                }
                *sched_max_concurrent =
                    (*sched_max_concurrent).max(max_concurrent as u32);
                continue;
            }
            // Parallel delegate: when 2+ delegate calls are ready AND their
            // task descriptions target disjoint file sets, run them in parallel
            // worktrees. Each `runner.run()` creates its own worktree and child
            // subprocess; the apply-back step is serialized by the global
            // `MERGE_LOCK`. When file sets overlap or can't be determined, fall
            // back to the serial single-dispatch path below.
            let delegate_indices: Vec<usize> = ready
                .iter()
                .copied()
                .filter(|&i| calls[i].1 == "delegate")
                .collect();
            if delegate_indices.len() > 1 {
                // Prepare all delegate jobs (budget, runner, file-set extraction).
                let mut prepared_delegates: Vec<(usize, DelegateJob, u64)> = Vec::new();
                let mut delegate_prep_failed: Vec<usize> = Vec::new();
                for &i in &delegate_indices {
                    let (id, _, arguments) = &calls[i];
                    if let Some((job, ledger_rev)) = self.prepare_delegate(arguments) {
                        let summary: String = job.task.chars().take(72).collect();
                        let ellipsis = if job.task.chars().count() > 72 { "…" } else { "" };
                        ui.subagent_note(&format!(
                            "↳ delegate subagent {}/{MAX_DELEGATE_SUBAGENTS_PER_SESSION}: {summary}{ellipsis}",
                            job.slot,
                        ));
                        ui.tool_call_id(id, "delegate", arguments);
                        prepared_delegates.push((i, job, ledger_rev));
                    } else {
                        delegate_prep_failed.push(i);
                    }
                }
                // Check if all pairs of file sets are disjoint. If any pair
                // overlaps (or has empty file sets), fall back to serial.
                let all_disjoint = prepared_delegates
                    .windows(2)
                    .all(|w| file_sets_disjoint(&w[0].1.file_set, &w[1].1.file_set));
                if all_disjoint && !prepared_delegates.is_empty() {
                    // Complete prep-failed delegates immediately.
                    for i in delegate_prep_failed {
                        let (id, _, arguments) = &calls[i];
                        ui.tool_call_id(id, "delegate", arguments);
                        let msg = format!(
                            "delegate budget exhausted ({MAX_DELEGATE_SUBAGENTS_PER_SESSION} this \
                             session); implement the rest directly instead."
                        );
                        let mut output = synthetic_tool_outcome(
                            msg.clone(),
                            hi_tools::ToolStatus::Denied,
                        );
                        output.effects.mutation_attempted = true;
                        emit_tool_output(&mut *ui, id, "delegate", &output);
                        let signature = inspection_signature("delegate", arguments);
                        let progress_label = ToolProgressLabel::new(
                            ProgressKind::Weak,
                            "delegate budget exhausted",
                            signature,
                        );
                        progress_tracker.record_tool(&progress_label);
                        tool_progress_labels.push(progress_label.clone());
                        tool_timeline.push(tool_entry(
                            "delegate".to_string(),
                            String::new(),
                            0,
                            &output,
                            &progress_label,
                        ));
                        results[i] = Some((id.clone(), msg));
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                        *sched_tool_calls += 1;
                    }
                    // Capture turn snapshot + checkpoint before any delegate
                    // mutates the tree (same as the serial path).
                    self.ensure_turn_snapshot(turn_snapshot).await?;
                    if !self
                        .ensure_turn_checkpoint(
                            turn_checkpoint_allowed,
                            turn_checkpoint_created,
                            ui,
                        )
                        .await
                    {
                        // Checkpoint denied — skip all prepared delegates.
                        for (i, _job, _) in &prepared_delegates {
                            self.release_delegate_slot();
                            let (id, _, arguments) = &calls[*i];
                            let msg = "Delegate skipped because strict mode requires an available \
                                       checkpoint."
                                .to_string();
                            let output = synthetic_tool_outcome(
                                msg.clone(),
                                hi_tools::ToolStatus::Denied,
                            );
                            emit_tool_output(&mut *ui, id, "delegate", &output);
                            let signature = inspection_signature("delegate", arguments);
                            let progress_label = ToolProgressLabel::new(
                                ProgressKind::Weak,
                                "delegate skipped without checkpoint",
                                signature,
                            );
                            progress_tracker.record_tool(&progress_label);
                            tool_progress_labels.push(progress_label.clone());
                            tool_timeline.push(tool_entry(
                                "delegate".to_string(),
                                String::new(),
                                0,
                                &output,
                                &progress_label,
                            ));
                            results[*i] = Some((id.clone(), msg));
                            completed[*i] = true;
                            completion_order.push(*i);
                            done += 1;
                            *sched_tool_calls += 1;
                        }
                        *sched_serial_runs += prepared_delegates.len() as u32;
                        *sched_max_concurrent = (*sched_max_concurrent).max(1);
                        continue;
                    }
                    // Run all prepared delegates concurrently. The
                    // `DelegateRunner` is `Send + Sync`, so multiple
                    // `runner.run()` calls execute in parallel — each creates
                    // its own worktree. The apply-back is serialized by
                    // `MERGE_LOCK`.
                    let max_concurrent =
                        MAX_PARALLEL_DELEGATES.min(prepared_delegates.len());
                    let started = std::time::Instant::now();
                    let delegate_results: Vec<_> =
                        futures_util::stream::iter(
                            prepared_delegates.into_iter().map(|(i, job, ledger_rev)| {
                                async move {
                                    let result = run_delegate_job(job).await;
                                    (i, result, ledger_rev)
                                }
                            }),
                        )
                        .buffer_unordered(max_concurrent)
                        .collect()
                        .await;
                    // Sort by index so results are processed in model order.
                    let mut sorted_results = delegate_results;
                    sorted_results.sort_by_key(|(i, _, _)| *i);
                    for (i, result, ledger_rev) in sorted_results {
                        let (id, _, arguments) = &calls[i];
                        let output = self.finish_delegate(result, ledger_rev, &mut *ui).await;
                        let duration_ms = started.elapsed().as_millis() as u64;
                        let error = output.status != hi_tools::ToolStatus::Succeeded;
                        let semantic_output = if error
                            && !output.content.starts_with("Error:")
                        {
                            std::borrow::Cow::Owned(format!("Error: {}", output.content))
                        } else {
                            std::borrow::Cow::Borrowed(output.content.as_str())
                        };
                        let signature = inspection_signature("delegate", arguments);
                        let signature_was_seen = signature_seen(&evidence, &signature);
                        let tracker_before = implementation_tracker.clone();
                        let validation_succeeded = tool_satisfies_validation(&output);
                        evidence.record_success("delegate", arguments, &semantic_output);
                        implementation_tracker.record_tool_result(
                            "delegate",
                            arguments,
                            &semantic_output,
                            validation_succeeded,
                        );
                        let progress = tool_guardrail.record_tool_result(
                            "delegate",
                            arguments,
                            &semantic_output,
                        );
                        if progress.idle_background_poll {
                            idle_background_poll_results += 1;
                        }
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
                                "delegate",
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
                            "delegate".to_string(),
                            String::new(),
                            duration_ms,
                            &output,
                            &progress_label,
                        ));
                        emit_tool_output(&mut *ui, id, "delegate", &output);
                        results[i] = Some((id.clone(), output.content));
                        completed[i] = true;
                        completion_order.push(i);
                        done += 1;
                        *sched_tool_calls += 1;
                    }
                    *sched_max_concurrent =
                        (*sched_max_concurrent).max(max_concurrent as u32);
                    continue;
                }
                // File sets overlap or are empty — fall back to serial.
                // Release the budget slots we consumed during preparation.
                for (_, _, _) in &prepared_delegates {
                    self.release_delegate_slot();
                }
                // Fall through to the serial self-dispatch path below.
            }
            // Self-dispatched calls: `delegate`/`task` run a child agent turn,
            // `record_decision` mutates agent state, and `get_task_output`/
            // `wait_tasks`/`kill_task` access the agent's background task
            // registry — all need `&mut self` or `&self` and can't join the
            // parallel `execute` stream. Run one alone when it's ready — the
            // dep graph then guarantees earlier mutations in the batch have
            // landed before a subagent sees the tree. (A single ready explore
            // also takes this path — the parallel path above only fires when
            // 2+ explores are ready simultaneously.)
            let self_idx = ready.iter().copied().find(|&i| {
                matches!(
                    calls[i].1.as_str(),
                    "explore"
                        | "delegate"
                        | "record_decision"
                        | "block_step"
                        | "task"
                        | "get_task_output"
                        | "wait_tasks"
                        | "kill_task"
                )
            });
            if let Some(i) = self_idx {
                let (id, name, arguments) = &calls[i];
                if name == "delegate" {
                    if self.config.gates.confirm_edits {
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
                            ui.tool_call_id(id, name, arguments);
                            let msg = if decision == ConfirmationResult::Unavailable {
                        "Delegate skipped: confirmation required, but this frontend cannot answer it."
                    } else {
                        "Delegate skipped by user (no changes applied)."
                    }
                    .to_string();
                            let mut output =
                                synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                            output.effects.mutation_attempted = true;
                            emit_tool_output(&mut *ui, id, name, &output);
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
                            *sched_tool_calls += 1;
                            *sched_serial_runs += 1;
                            *sched_max_concurrent = (*sched_max_concurrent).max(1);
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
                    self.ensure_turn_snapshot(turn_snapshot).await?;
                    if !self
                        .ensure_turn_checkpoint(
                            turn_checkpoint_allowed,
                            turn_checkpoint_created,
                            ui,
                        )
                        .await
                    {
                        ui.tool_call_id(id, name, arguments);
                        let msg = "Delegate skipped because strict mode requires an available checkpoint.".to_string();
                        let output =
                            synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                        emit_tool_output(&mut *ui, id, name, &output);
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
                        *sched_tool_calls += 1;
                        *sched_serial_runs += 1;
                        *sched_max_concurrent = (*sched_max_concurrent).max(1);
                        continue;
                    }
                }
                ui.tool_call_id(id, name, arguments);
                let started = std::time::Instant::now();
                let output = match name.as_str() {
                    "explore" => self.handle_explore(arguments, &mut *ui).await,
                    "delegate" => self.handle_delegate(arguments, &mut *ui).await,
                    "task" => self.handle_task(arguments, &mut *ui).await,
                    "get_task_output" => self.handle_get_task_output(arguments).await,
                    "wait_tasks" => self.handle_wait_tasks(arguments).await,
                    "kill_task" => self.handle_kill_task(arguments).await,
                    "block_step" => self.handle_block_step(arguments),
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
                let progress = tool_guardrail.record_tool_result(name, arguments, &semantic_output);
                if progress.idle_background_poll {
                    idle_background_poll_results += 1;
                }
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
                emit_tool_output(&mut *ui, id, name, &output);
                results[i] = Some((id.clone(), output.content));
                completed[i] = true;
                completion_order.push(i);
                done += 1;
                // Runs alone, like bash.
                *sched_tool_calls += 1;
                *sched_serial_runs += 1;
                *sched_max_concurrent = (*sched_max_concurrent).max(1);
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
                ui.tool_started_id(&calls[i].0, &calls[i].1, &calls[i].2);
            }
            // In --confirm-edits mode, check each mutating call with
            // the UI before executing. Denied calls get a "skipped"
            // result instead of running.
            let mut denied: Vec<usize> = Vec::new();
            let mut checkpoint_denied = BTreeSet::new();
            let mut prepared_mutations = BTreeMap::new();
            let mut preparation_failures = BTreeMap::new();
            if self.config.gates.confirm_edits {
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
                self.ensure_turn_snapshot(turn_snapshot).await?;
                if !self
                    .ensure_turn_checkpoint(turn_checkpoint_allowed, turn_checkpoint_created, ui)
                    .await
                {
                    let blocked: Vec<usize> = approved
                        .iter()
                        .copied()
                        .filter(|&i| {
                            !preparation_failures.contains_key(&i)
                                && implementation_tool_call_mutates(&calls[i].1, &calls[i].2)
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
            let outputs: Vec<_> =
                futures_util::stream::iter(executions.into_iter().map(|(i, prepared, failure)| {
                    let root = &root;
                    let state_root = &state_root;
                    let lsp = &lsp;
                    let background = self.runtime.background();
                    let read_cache = self.runtime.read_cache();
                    let repo_map = self.runtime.repo_map();
                    let calls = &calls;
                    async move {
                        let output = if let Some(failure) = failure {
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
                                repo_map,
                                &calls[i].1,
                                &calls[i].2,
                            )
                            .await
                        };
                        (i, output)
                    }
                }))
                .buffer_unordered(max_parallel_tools)
                .collect()
                .await;
            let batch_duration_ms = batch_started.elapsed().as_millis() as u64;
            // Scheduler telemetry: count every call in the ready batch,
            // but report actual concurrency after the configured cap.
            *sched_tool_calls += batch_size;
            *sched_max_concurrent = (*sched_max_concurrent).max(actual_concurrency);
            if actual_concurrency == 1 {
                *sched_serial_runs += batch_size;
            }
            // Handle denied calls first: emit their headers and "skipped" results.
            for &i in &denied {
                let name = &calls[i].1;
                ui.tool_call_id(&calls[i].0, name, &calls[i].2);
                let skipped_msg = if checkpoint_denied.contains(&i) {
                    "Mutation skipped because strict mode requires an available checkpoint."
                        .to_string()
                } else {
                    "Edit skipped by user (not applied).".to_string()
                };
                let mut output =
                    synthetic_tool_outcome(skipped_msg.clone(), hi_tools::ToolStatus::Denied);
                output.effects.mutation_attempted = true;
                emit_tool_output(&mut *ui, &calls[i].0, name, &output);
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
            for (i, output) in outputs {
                let name = &calls[i].1;
                // Emit the transcript header immediately before its
                // result — in a concurrent batch this pairs each header
                // with its own result in completion order.
                ui.tool_call_id(&calls[i].0, name, &calls[i].2);
                let path = hi_tools::target_path(name, &calls[i].2).unwrap_or_default();
                self.record_tool_effects(&output.effects)?;
                for change in &output.effects.file_changes {
                    batch_mutated_paths.insert(change.path.clone());
                }
                if matches!(name.as_str(), "bash" | "bash_output" | "bash_kill") {
                    self.reconcile_workspace_changes().await?;
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
                        .is_some_and(|plan| self.goals.plan() != plan);
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
                if progress.idle_background_poll {
                    idle_background_poll_results += 1;
                }
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
                emit_tool_output(&mut *ui, &calls[i].0, name, &output);
                results[i] = Some((calls[i].0.clone(), output.content));
                // Track the latest plan state so the continue logic can
                // detect an incomplete plan when the model stops calling
                // tools. The model resubmits the whole list on every
                // call, so the last one is always current.
                if calls[i].1 == "update_plan"
                    && let Some(plan) = output.plan.as_deref()
                {
                    let _ = self.goals.replace_plan(plan);
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
                    if self.config.subagents.long_horizon
                        && let Some(current_goal) = self.goals.structured.as_ref()
                    {
                        let turn_start_active = current_goal.active_index();
                        let goal = proposed_goal.get_or_insert_with(|| current_goal.clone());
                        apply_plan_to_goal(goal, plan, turn_start_active);
                        *plan_updated_goal = true;
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
                    if self.config.gates.proactive_verify
                        && let Some(path) = hi_tools::target_path(&calls[i].1, &calls[i].2)
                        && let Some(cmd) = hi_tools::fast_check_for(&path)
                    {
                        let root = self.runtime.root().to_path_buf();
                        let check = cmd.to_string();
                        let check_path = std::path::PathBuf::from(&path);
                        pending_checks.push((
                            path,
                            tokio::spawn(async move {
                                hi_tools::run_fast_check_in(&root, &check, &check_path).await
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
        let mut results: Vec<(String, String)> = results.into_iter().flatten().collect();
        // Await the proactive per-edit checks kicked off during the
        // batch and surface each as a status line — a syntax/lint error
        // appears here, during the turn, before turn-end verify. A pass
        // is silent (no need to noise a clean edit); a failure names the
        // file and shows the check output so the model can fix it now.
        let mut proactive_failures = Vec::new();
        for (path, handle) in pending_checks {
            if let Ok((passed, output)) = handle.await {
                if passed {
                    continue;
                }
                let msg = format!("⚠ proactive check failed for {path}:\n{output}");
                ui.status(&msg);
                proactive_failures.push(msg);
            }
        }
        // Mid-turn Rust fast path: LSP → affected cargo check → (if
        // test-gated) affected cargo test. Failures append to tool results.
        let mut fast_failures = Vec::new();
        if !batch_mutated_paths.is_empty() {
            let paths = batch_mutated_paths.into_iter().collect::<Vec<_>>();
            let run_tests = task_contract.wants_tests
                || self
                    .task
                    .last_task_contract
                    .as_ref()
                    .is_some_and(|c| c.wants_tests);
            let report = super::super::fast_feedback::run_fast_feedback(
                &self.runtime,
                &paths,
                fast_feedback,
                super::super::fast_feedback::FastFeedbackOptions { run_tests },
                ui,
            )
            .await;
            if let Some(text) = report.combined_failure() {
                fast_failures.push(text);
            }
        }
        // Append failures onto the last mutating tool result so the model
        // sees them in the transcript before the next reasoning step.
        let mut feedback_blocks = proactive_failures;
        feedback_blocks.extend(fast_failures);
        if !feedback_blocks.is_empty() {
            let block = feedback_blocks.join("\n\n");
            if let Some((_, content)) = results.iter_mut().rev().find(|(id, _)| {
                // Prefer a result that came from a filesystem mutation if we can
                // spot one by matching call ids in this batch.
                calls.iter().any(|(call_id, name, _)| {
                    call_id == id && (hi_tools::is_filesystem_mutating(name) || name == "bash")
                })
            }) {
                if !content.ends_with('\n') {
                    content.push('\n');
                }
                content.push('\n');
                content.push_str(&block);
            } else if let Some((_, content)) = results.last_mut() {
                if !content.ends_with('\n') {
                    content.push('\n');
                }
                content.push('\n');
                content.push_str(&block);
            } else {
                // No tool results (shouldn't happen for a mutation batch) —
                // still push a nudge so the model is not blind.
                self.messages.push_nudge(
            NudgeKind::Continue,
            format!(
                "Fast check found problems after your last edits — fix these before continuing:\n{block}"
            ),
        );
            }
        }
        self.messages
            .push_assistant_with_results(std::mem::take(completion_content), results);
        // A fully cancelled batch did not execute discovery or implementation
        // work, so it must not burn the mutation-recovery round budget.
        if interrupted_calls < calls.len() {
            implementation_tracker.record_tool_round();
        }

        Ok(ToolBatchOutcome {
            hash_guard_applies,
            hashable_idempotent_results,
            repeated_idempotent_results,
            idle_background_poll_results,
            tool_progress_labels,
            plan_changed_this_batch,
            interrupted_calls,
            interrupted_coordination_calls,
        })
    }
}
