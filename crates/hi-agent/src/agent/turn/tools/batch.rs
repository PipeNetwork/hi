//! Tool-batch execution for one model round (TurnPhase::Tools).
//!
//! Extracted from the main turn loop so orchestration stays thin: run the
//! dep-aware scheduler, record results, attach fast-feedback, then return
//! batch stats for the post-tool Steer phase.

mod feedback;
mod outcome;
mod policy;

use feedback::{PendingCheck, append_fast_feedback};
pub(in crate::agent::turn) use outcome::ToolBatchOutcome;
use outcome::{append_tool_images, emit_capability_request};
use policy::{
    dry_run_message, parked_or_denied_delegate, parked_or_denied_shell,
    terminal_background_requires_reconciliation, wait_flavored_call, workspace_execution_report,
    workspace_mutation_intent, workspace_operation_requires_settlement,
    workspace_program_execution_report, workspace_program_intent,
};

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context as _, Result};
use futures_util::StreamExt;
use hi_tools::protocol::{
    execute_in_runtime_shared_with_runner, execute_prepared_in_runtime,
    execute_streaming_in_runtime_with_runner, prepare_mutation_in_with_state,
};

use super::super::speculation::{SpeculationKey, SpeculationRegistry};
use crate::heuristics::{emit_tool_output, mode_blocks_tool, respects_deps, tool_deps};
use crate::steering::{
    BashCommandKind, EvidenceTracker, ImplementationTracker, bash_call_waits, bash_command,
    classify_bash_command, inspection_signature, read_only_blocked_tool_result,
    read_only_blocks_tool,
};
use crate::verify::Snapshot;
use crate::{
    ConfirmationRequest, ConfirmationResult, PARKED_TOOL_RESULT, TaskContract, Ui,
    confirmation_for_egress_tool, egress_confirm_required,
};
use hi_ai::Content;
use hi_workflow::{
    ProgramCall, ProgramHostRequest, ProgramOutcome, ProgramRunParams, extract_safe_literal_calls,
};
use tokio_util::sync::CancellationToken;

/// Owns every task launched for one workflow program. Dropping an async
/// `JoinHandle` detaches it, and a running `spawn_blocking` task cannot be
/// aborted. Cancel the cooperative Rhai token before aborting the watcher so a
/// dropped turn cannot leave an unlimited program consuming a worker forever.
struct ProgramRunGuard {
    cancel: CancellationToken,
    _watcher: tokio_util::task::AbortOnDropHandle<()>,
}

impl ProgramRunGuard {
    fn new(cancel: CancellationToken, watcher: tokio::task::JoinHandle<()>) -> Self {
        Self {
            cancel,
            _watcher: tokio_util::task::AbortOnDropHandle::new(watcher),
        }
    }
}

impl Drop for ProgramRunGuard {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[derive(Clone)]
struct ProgramToolRunner {
    root: std::path::PathBuf,
    state_root: std::path::PathBuf,
    process_runner: hi_tools::ProcessRunner,
    lsp: std::sync::Arc<hi_lsp::LspManager>,
    background: std::sync::Arc<hi_tools::BackgroundRegistry>,
    read_cache: std::sync::Arc<std::sync::Mutex<hi_tools::ReadCache>>,
    repo_map: std::sync::Arc<std::sync::Mutex<hi_tools::RepoMapCache>>,
    mcp: Option<std::sync::Arc<dyn hi_tools::McpBackend>>,
    memory: Option<std::sync::Arc<dyn hi_tools::MemoryBackend>>,
}

#[derive(Clone)]
pub(crate) struct ProgramSpeculator {
    runner: ProgramToolRunner,
    allowed_tools: std::sync::Arc<BTreeSet<String>>,
    turn_id: String,
    enabled: bool,
    external_allowed: bool,
    max_calls: usize,
    context_generation: u64,
    ledger_revision: u64,
    external_freshness_epoch: u64,
}

impl ProgramSpeculator {
    pub(crate) fn launch(
        &self,
        speculation_registry: &SpeculationRegistry,
        program_id: &str,
        source: &str,
    ) {
        if !self.enabled {
            return;
        }
        for call in extract_safe_literal_calls(source)
            .into_iter()
            .take(self.max_calls)
        {
            if !self.allowed_tools.contains(&call.name) {
                continue;
            }
            let external = matches!(
                hi_tools::speculation_class(&call.name),
                hi_tools::SpeculationClass::IdempotentExternal
            );
            if external && !self.external_allowed {
                continue;
            }
            if !matches!(
                hi_tools::speculation_class(&call.name),
                hi_tools::SpeculationClass::PureLocal
                    | hi_tools::SpeculationClass::IdempotentExternal
            ) {
                continue;
            }
            let args = serde_json::to_string(&call.arguments).unwrap_or_default();
            let key = SpeculationKey::new(
                &self.turn_id,
                program_id,
                call.occurrence,
                &call.name,
                &args,
                self.context_generation,
                self.ledger_revision,
                if external {
                    self.external_freshness_epoch
                } else {
                    0
                },
            );
            let registry = speculation_registry.clone();
            let runner = self.runner.clone();
            registry.launch(key, external, async move { runner.execute(&call).await.0 });
        }
    }
}

impl ProgramToolRunner {
    async fn execute(
        &self,
        call: &ProgramCall,
    ) -> (
        std::result::Result<hi_workflow::ProgramToolResult, String>,
        hi_tools::ToolOutcome,
    ) {
        let args = serde_json::to_string(&call.arguments).unwrap_or_default();
        let allowed = matches!(
            hi_tools::speculation_class(&call.name),
            hi_tools::SpeculationClass::PureLocal | hi_tools::SpeculationClass::IdempotentExternal
        ) && hi_tools::is_read_only(&call.name)
            && call.name != "run_program"
            && hi_tools::is_known_tool(&call.name);
        if !allowed {
            let message = format!(
                "tool `{}` requires ordinary structured-tool execution; retry without run_program",
                call.name
            );
            let output = synthetic_tool_outcome(message.clone(), hi_tools::ToolStatus::Denied);
            return (Err(message), output);
        }
        let output = execute_in_runtime_shared_with_runner(
            &self.process_runner,
            &self.root,
            &self.state_root,
            &self.lsp,
            &self.background,
            &self.read_cache,
            &self.repo_map,
            self.mcp.as_deref(),
            self.memory.as_deref(),
            &call.name,
            &args,
        )
        .await;
        let status = match output.status {
            hi_tools::ToolStatus::Succeeded => "succeeded",
            hi_tools::ToolStatus::Failed => "failed",
            hi_tools::ToolStatus::Denied => "denied",
            hi_tools::ToolStatus::TimedOut => "timed_out",
            hi_tools::ToolStatus::Cancelled => "cancelled",
        };
        let result = hi_workflow::ProgramToolResult {
            index: call.occurrence,
            name: call.name.clone(),
            status: status.into(),
            output: output.content.clone(),
        };
        (Ok(result), output)
    }
}

use crate::agent::delegate_turn::{
    DelegateJob, delegate_limit_denial, delegate_turn_limit, file_sets_disjoint,
    parallel_delegate_limit, run_delegate_job,
};
use crate::agent::explore_turn::{
    ExploreJob, MAX_PARALLEL_EXPLORES, explore_tool_outcome, run_explore_job,
};

use crate::apply_plan_to_goal;
use crate::heuristics::plan_has_pending_steps;
use crate::steering::implementation_tool_call_mutates;
use hi_tools::PlanStatus;

use super::super::helpers::{
    synthetic_tool_outcome, tool_entry, tool_entry_with_args, tool_satisfies_validation,
};
use super::super::phase::TurnPhase;
use super::super::progress::{
    ProgressKind, ProgressTracker, ToolProgressLabel, classify_tool_progress, signature_seen,
};
use super::super::retention::ToolTimeline;

/// Add a scheduler count without allowing a very long unlimited turn to wrap
/// the telemetry counter back to zero. `usize` inputs are clamped before the
/// addition so this remains correct on 64-bit hosts as well.
fn saturating_add_scheduler_count(total: &mut u32, additional: usize) {
    let additional = u32::try_from(additional).unwrap_or(u32::MAX);
    *total = total.saturating_add(additional);
}

/// Constrain a model-authored checklist while the session is in plan mode.
///
/// A previously completed step may stay completed when the user re-enters
/// planning to revise the remaining work. Every other step is executable work
/// and therefore remains pending until a non-plan turn supplies real evidence.
fn normalize_plan_mode_update(
    current: &[hi_tools::PlanStep],
    proposed: &mut [hi_tools::PlanStep],
) -> usize {
    let mut corrected = 0;
    for (index, step) in proposed.iter_mut().enumerate() {
        let status = if current.get(index).is_some_and(|existing| {
            existing.title == step.title && existing.status == PlanStatus::Done
        }) {
            PlanStatus::Done
        } else {
            PlanStatus::Pending
        };
        if step.status != status {
            step.status = status;
            corrected += 1;
        }
    }
    corrected
}

/// Whether a checklist title describes work that normally needs concrete
/// workspace or validation evidence before it can truthfully become `Done`.
/// Read-only milestones may still be completed from inspection evidence.
fn plan_step_requires_execution_evidence(title: &str) -> bool {
    crate::agent::plan_goal::plan_step_requires_execution_evidence(title)
}

/// Extract a model-supplied reason that this particular implementation step
/// did not require a workspace change. A generic top-level recap cannot
/// self-certify an entire checklist.
fn plan_step_has_no_change_justification(arguments: &str, index: usize) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return false;
    };
    value
        .get("steps")
        .and_then(serde_json::Value::as_array)
        .and_then(|steps| steps.get(index))
        .and_then(|step| step.get("completion_evidence"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .is_some_and(concrete_no_change_justification)
}

fn concrete_no_change_justification(reason: &str) -> bool {
    let normalized = reason
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '/' | '_' | '-') {
                character
            } else {
                ' '
            }
        })
        .collect::<String>();
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if reason.chars().count() < 12 || words.len() < 3 {
        return false;
    }
    !matches!(
        words.join(" ").as_str(),
        "done"
            | "already done"
            | "already complete"
            | "already completed"
            | "no change needed"
            | "no changes needed"
            | "no change required"
            | "no changes required"
            | "not needed"
            | "not required"
    )
}

/// Reject unsupported completion claims on implementation-shaped checklist
/// steps. Successful mutation, successful validation, or an explicit per-step
/// no-change justification is required.
fn normalize_unsupported_plan_completion(
    current: &[hi_tools::PlanStep],
    proposed: &mut [hi_tools::PlanStep],
    arguments: &str,
    execution_evidence: bool,
) -> Vec<usize> {
    // Turn-global mutation/validation proves at most the step that was active
    // when the work began. Letting one unrelated write or an old passing test
    // authorize every `done` entry allowed a weak model to clear an entire
    // durable checklist at once. Additional implementation steps need their
    // own concrete `completion_evidence`.
    let evidenced_index = execution_evidence
        .then(|| {
            current
                .iter()
                .position(|step| step.status == PlanStatus::Active)
                .or_else(|| {
                    current
                        .iter()
                        .position(|step| step.status == PlanStatus::Pending)
                })
                .or_else(|| {
                    proposed.iter().position(|step| {
                        step.status == PlanStatus::Done
                            && plan_step_requires_execution_evidence(&step.title)
                    })
                })
        })
        .flatten();

    let mut corrected = Vec::new();
    for (index, step) in proposed.iter_mut().enumerate() {
        if step.status != PlanStatus::Done || !plan_step_requires_execution_evidence(&step.title) {
            continue;
        }
        let already_done = current.get(index).is_some_and(|existing| {
            existing.title == step.title && existing.status == PlanStatus::Done
        });
        if already_done
            || evidenced_index == Some(index)
            || plan_step_has_no_change_justification(arguments, index)
        {
            continue;
        }

        step.status = current
            .get(index)
            .filter(|existing| existing.title == step.title)
            .map(|existing| existing.status)
            .filter(|status| *status != PlanStatus::Done)
            .unwrap_or(PlanStatus::Pending);
        corrected.push(index);
    }
    corrected
}

impl crate::Agent {
    /// Execute `calls` for the current round and append assistant+results.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::agent::turn) async fn execute_tool_batch(
        &mut self,
        calls: &[(String, String, String)],
        completion_content: &mut Vec<Content>,
        tool_specs: &[hi_ai::ToolSpec],
        tool_envelope: &hi_tools::envelope::ToolEnvelope,
        read_only_intent: Option<crate::steering::ReviewIntent>,
        max_parallel_tools: usize,
        task_contract: &TaskContract,
        implementation_tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        progress_tracker: &mut ProgressTracker,
        tool_timeline: &mut ToolTimeline,
        sched_tool_calls: &mut u32,
        sched_max_concurrent: &mut u32,
        sched_serial_runs: &mut u32,
        speculation_registry: &SpeculationRegistry,
        program_fallback_next: &mut bool,
        program_fallback_used: &mut bool,
        plan_updated_goal: &mut bool,
        proposed_goal: &mut Option<crate::Goal>,
        turn_snapshot: &mut Option<Snapshot>,
        turn_checkpoint_allowed: &mut Option<bool>,
        turn_checkpoint_created: &mut bool,
        fast_feedback: &mut super::super::fast_feedback::FastFeedbackState,
        ui: &mut dyn Ui,
    ) -> Result<ToolBatchOutcome> {
        // The negotiated provider limit is part of the sealed request. Even if
        // the session is configured for wider fan-out, execution may never
        // exceed what the request advertised and audited.
        let max_parallel_tools = max_parallel_tools
            .min(usize::from(tool_envelope.payload.limits.max_parallel_calls))
            .max(1);
        // The batch has not announced any of its tools yet, so an interrupt
        // already present here can only belong to the previously visible tool
        // (most notably a preflight whose result was still queued in the TUI).
        // Never let that stale signal cancel the model's next action.
        self.interrupt
            .store(false, std::sync::atomic::Ordering::Relaxed);
        if calls.iter().any(|(_, name, _)| {
            hi_tools::is_filesystem_mutating(name)
                || name == "bash"
                || (self.pipefs_workspace_active()
                    && matches!(name.as_str(), "bash_output" | "bash_kill" | "use_tool"))
        }) {
            // A real mutation changes the meaning of every shadow read. Drop
            // those entries before dispatching the batch so no stale result
            // can be claimed by a later program call.
            speculation_registry.invalidate_all();
        }
        // A program envelope owns the completion. Reject a mixed batch before
        // announcing any individual tool so ordinary calls cannot look as if
        // they ran and no orphan tool results can enter the transcript.
        if calls.iter().any(|(_, name, _)| name == "run_program") {
            return self
                .execute_program_batch(
                    calls,
                    completion_content,
                    tool_specs,
                    tool_envelope,
                    read_only_intent,
                    progress_tracker,
                    tool_timeline,
                    sched_tool_calls,
                    sched_max_concurrent,
                    sched_serial_runs,
                    speculation_registry,
                    program_fallback_next,
                    program_fallback_used,
                    ui,
                )
                .await;
        }
        // This is deliberately emitted before any scheduler or permission
        // branch. Text-promoted tool calls therefore receive the same typed
        // capability audit as provider-native calls.
        for (id, name, _) in calls {
            emit_capability_request(&mut *ui, id, name);
        }
        let hash_guard_applies = calls.iter().all(|(_, name, args)| {
            matches!(
                name.as_str(),
                "read" | "list" | "grep" | "glob" | "bash_output"
            ) || (name == "bash" && bash_call_waits(args))
        });
        let mut hashable_idempotent_results = 0usize;
        let mut repeated_idempotent_results = 0usize;
        let mut running_background_poll_results = 0usize;
        let mut actionable_poll_results = 0usize;
        let mut wait_flavored_results = 0usize;
        self.set_turn_phase(TurnPhase::Tools);
        let mut tool_progress_labels: Vec<ToolProgressLabel> = Vec::new();
        let mut plan_changed_this_batch = false;
        let mut interrupted_calls = 0usize;
        let mut interrupted_coordination_calls = 0usize;
        let mut protocol_validation_errors = Vec::new();
        let batch_requires_settlement = !self.config.gates.dry_run
            && calls.iter().any(|(_, name, arguments)| {
                workspace_operation_requires_settlement(name, arguments)
            });
        let mut workspace_intent = if batch_requires_settlement {
            let opaque = calls.iter().any(|(_, name, _)| {
                matches!(name.as_str(), "bash" | "bash_output" | "bash_kill")
                    || (self.pipefs_workspace_active() && name == "use_tool")
            });
            let paths = (!opaque).then(|| {
                calls
                    .iter()
                    .filter_map(|(_, name, arguments)| hi_tools::target_path(name, arguments))
                    .collect::<Vec<_>>()
            });
            let intent = workspace_mutation_intent(calls, paths);
            self.begin_classified_workspace_operation(intent.clone())
                .await?;
            Some(intent)
        } else {
            None
        };
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
        let mut vision: Vec<hi_tools::ToolImage> = Vec::new();
        let mut completed = vec![false; calls.len()];
        let mut completion_order: Vec<usize> = Vec::with_capacity(calls.len());
        let mut scheduler_forced_skip = false;
        // Reserve the remaining hard tool budget for the model-ordered
        // prefix before any ready batch is dispatched. Calls beyond
        // this prefix receive typed denials and are never executed.
        let permitted_prefix = calls.len().min(
            self.config
                .loop_limits
                .remaining_tool_calls(*sched_tool_calls) as usize,
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
            if let Some(entry) = tool_timeline.last_mut() {
                entry.completion_index = completion_order.len() as u32;
            }
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
                if let Some(entry) = tool_timeline.last_mut() {
                    entry.completion_index = completion_order.len() as u32;
                }
            }
        }
        // Calls that survived policy/budget denial are about to cross the
        // local execution boundary. Validate the declared Draft
        // 2020-12 schema here: malformed model output receives a typed tool
        // result and can never reach a workspace-mutating executor.
        let envelope_error = if !tool_envelope.digest_is_valid() {
            Some("tool envelope digest does not match its payload".to_string())
        } else if !tool_envelope.matches_specs(tool_specs) {
            Some("execution schemas do not match the sealed tool envelope".to_string())
        } else {
            None
        };
        let batch_validation_error = hi_ai::validate_client_tool_batch_limits_with(
            calls
                .iter()
                .enumerate()
                .filter(|(index, _)| !completed[*index])
                .map(|(_, (_, _, arguments))| arguments.as_str()),
            tool_envelope.payload.limits.max_tool_argument_bytes as usize,
        )
        .err();
        for (i, (id, name, arguments)) in calls.iter().enumerate().take(permitted_prefix) {
            if completed[i] {
                continue;
            }
            let error = if let Some(error) = envelope_error.clone() {
                error
            } else if !tool_envelope.admits(name) {
                format!(
                    "tool `{name}` is outside the model request's sealed envelope {}",
                    tool_envelope.digest
                )
            } else if let Some(error) = batch_validation_error.clone() {
                error.to_string()
            } else {
                match hi_ai::validate_client_tool_call_with_limit(
                    id,
                    name,
                    arguments,
                    tool_specs,
                    tool_envelope.payload.limits.max_tool_argument_bytes as usize,
                ) {
                    Ok(()) => continue,
                    Err(error) => error.to_string(),
                }
            };
            ui.tool_call_id(id, name, arguments);
            let content = serde_json::json!({
                "error": {
                    "kind": "tool_protocol_error",
                    "message": error.to_string(),
                }
            })
            .to_string();
            protocol_validation_errors.push((name.clone(), error.to_string()));
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
            if let Some(entry) = tool_timeline.last_mut() {
                entry.completion_index = completion_order.len() as u32;
            }
        }
        let mut done = completion_order.len();
        // Dry run: every call that survived policy/budget/protocol denial is a
        // planned action. Print what it *would* do and synthesize a result
        // without executing anything — no workspace mutation, no process spawn.
        if self.config.gates.dry_run {
            for (i, (id, name, arguments)) in calls.iter().enumerate().take(permitted_prefix) {
                if completed[i] {
                    continue;
                }
                let mutates = implementation_tool_call_mutates(name, arguments);
                implementation_tracker.record_dry_run_plan(mutates);
                let path = hi_tools::target_path(name, arguments).unwrap_or_default();
                let msg = dry_run_message(name, &path, mutates);
                ui.tool_call_id(id, name, arguments);
                let mut output = synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                output.effects.mutation_attempted = mutates;
                emit_tool_output(&mut *ui, id, name, &output);
                let progress_label = ToolProgressLabel::new(
                    ProgressKind::None,
                    "dry-run: planned action not executed",
                    inspection_signature(name, arguments),
                );
                progress_tracker.record_tool(&progress_label);
                tool_progress_labels.push(progress_label.clone());
                tool_timeline.push(tool_entry(name.clone(), path, 0, &output, &progress_label));
                results[i] = Some((id.clone(), msg));
                completed[i] = true;
                completion_order.push(i);
                if let Some(entry) = tool_timeline.last_mut() {
                    entry.completion_index = completion_order.len() as u32;
                }
                done += 1;
                saturating_add_scheduler_count(sched_tool_calls, 1);
                saturating_add_scheduler_count(sched_serial_runs, 1);
                *sched_max_concurrent = (*sched_max_concurrent).max(1);
            }
        }
        // The dry-run loop above already accounted its calls in the sched
        // counters; only count calls completed by the earlier denial
        // pre-passes here so dry-run stats are not double-counted.
        let initially_executed = if self.config.gates.dry_run {
            0
        } else {
            done.saturating_sub(budget_denied)
        };
        if initially_executed > 0 {
            saturating_add_scheduler_count(sched_tool_calls, initially_executed);
            saturating_add_scheduler_count(sched_serial_runs, initially_executed);
            *sched_max_concurrent = (*sched_max_concurrent).max(1);
        }
        // Proactive per-edit checks: kicked off in the background as
        // mutating calls complete, awaited after the batch so any
        // syntax/lint error surfaces during the turn (before turn-end
        // verify) while the edit is still the model's focus. Each entry
        // is (path, check label, abort-on-drop handle of the check). The
        // ownership is important now that checks have no default deadline:
        // cancelling a turn must not detach an unlimited child process.
        let mut pending_checks: Vec<PendingCheck> = Vec::new();
        // Project-relative paths mutated in this tool batch — drives
        // mid-turn LSP diagnostics + affected cargo check.
        let mut batch_mutated_paths: BTreeSet<String> = BTreeSet::new();
        while done < calls.len() {
            // Check interrupt / whole-turn cancel: Esc skips the current tool
            // batch; turn-level cancel (Ctrl+C) sets the same flag via
            // run_turn_cancellable so in-flight batches synthesize
            // cancelled tool_results instead of only dying on drop.
            let turn_cancelled = self
                .turn_cancellation
                .as_ref()
                .is_some_and(|c| c.is_cancelled());
            if turn_cancelled
                || self
                    .interrupt
                    .swap(false, std::sync::atomic::Ordering::Relaxed)
            {
                let mut interrupted = 0_u32;
                let msg = if turn_cancelled {
                    "Tool call cancelled — turn interrupted by user.".to_string()
                } else {
                    "Tool call interrupted by user.".to_string()
                };
                let progress_detail = if turn_cancelled {
                    "tool cancelled (turn cancel)"
                } else {
                    "tool interrupted by user"
                };
                for i in 0..calls.len() {
                    if !completed[i] {
                        let (id, name, arguments) = &calls[i];
                        ui.tool_call_id(id, name, arguments);
                        let mut output =
                            synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Cancelled);
                        output.effects.mutation_attempted =
                            implementation_tool_call_mutates(name, arguments);
                        emit_tool_output(&mut *ui, id, name, &output);
                        let progress_label = ToolProgressLabel::new(
                            ProgressKind::None,
                            progress_detail,
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
                        results[i] = Some((id.clone(), msg.clone()));
                        completed[i] = true;
                        completion_order.push(i);
                        if let Some(entry) = tool_timeline.last_mut() {
                            entry.completion_index = completion_order.len() as u32;
                        }
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
                if turn_cancelled {
                    ui.status("⚠ turn cancelled — remaining tool calls skipped");
                } else {
                    ui.status("⚠ tool call interrupted by user — the model will adapt");
                }
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
                saturating_add_scheduler_count(sched_tool_calls, unresolved.len());
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
                    if let Some(entry) = tool_timeline.last_mut() {
                        entry.completion_index = completion_order.len() as u32;
                    }
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
            // If any ready call is bash, run it alone (streaming UI) — unless
            // it's a read-only inspection (pwd/ls/find/rg/grep/cat/head/tail/
            // git), which joins the concurrent batch below for parallelism.
            let bash_idx = ready.iter().copied().find(|&i| {
                calls[i].1 == "bash"
                    && !matches!(
                        bash_command(&calls[i].2)
                            .map(|c| classify_bash_command(&c))
                            .unwrap_or(BashCommandKind::Unknown),
                        BashCommandKind::Inspection
                    )
            });
            if let Some(i) = bash_idx {
                let (id, name, arguments) = &calls[i];
                let bash_mutates = implementation_tool_call_mutates(name, arguments);
                if self.config.gates.confirm_edits && bash_mutates {
                    if self.approval_parked {
                        ui.tool_call_id(id, name, arguments);
                        let msg = PARKED_TOOL_RESULT.to_string();
                        let mut output =
                            synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Cancelled);
                        output.effects.mutation_attempted = true;
                        emit_tool_output(&mut *ui, id, name, &output);
                        let progress_label = ToolProgressLabel::new(
                            ProgressKind::Weak,
                            "shell mutation parked for approval",
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
                        if let Some(entry) = tool_timeline.last_mut() {
                            entry.completion_index = completion_order.len() as u32;
                        }
                        done += 1;
                        saturating_add_scheduler_count(sched_tool_calls, 1);
                        saturating_add_scheduler_count(sched_serial_runs, 1);
                        *sched_max_concurrent = (*sched_max_concurrent).max(1);
                        continue;
                    }
                    let command = bash_command(arguments).unwrap_or_else(|| arguments.clone());
                    let cwd = self.runtime.root().display().to_string();
                    let decision = ui
                        .confirm(ConfirmationRequest::ShellMutation { command, cwd })
                        .await;
                    if decision != ConfirmationResult::Approved {
                        if decision == ConfirmationResult::Parked {
                            self.note_approval_parked(ui);
                        }
                        ui.tool_call_id(id, name, arguments);
                        let (msg, status) = parked_or_denied_shell(&decision);
                        let mut output = synthetic_tool_outcome(msg.clone(), status);
                        output.effects.mutation_attempted = true;
                        emit_tool_output(&mut *ui, id, name, &output);
                        let progress_label = ToolProgressLabel::new(
                            ProgressKind::Weak,
                            if decision == ConfirmationResult::Parked {
                                "shell mutation parked for approval"
                            } else {
                                "shell mutation denied by confirmation"
                            },
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
                        if let Some(entry) = tool_timeline.last_mut() {
                            entry.completion_index = completion_order.len() as u32;
                        }
                        done += 1;
                        saturating_add_scheduler_count(sched_tool_calls, 1);
                        saturating_add_scheduler_count(sched_serial_runs, 1);
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
                    if let Some(entry) = tool_timeline.last_mut() {
                        entry.completion_index = completion_order.len() as u32;
                    }
                    done += 1;
                    saturating_add_scheduler_count(sched_tool_calls, 1);
                    saturating_add_scheduler_count(sched_serial_runs, 1);
                    *sched_max_concurrent = (*sched_max_concurrent).max(1);
                    continue;
                }
                ui.tool_started_id(id, name, arguments);
                ui.tool_call_id(id, name, arguments);
                let path = hi_tools::target_path(name, arguments).unwrap_or_default();
                let started = std::time::Instant::now();
                let ui_ref: &mut dyn Ui = &mut *ui;
                let lsp = self.runtime.lsp();
                let output = execute_streaming_in_runtime_with_runner(
                    self.runtime.process_runner(),
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
                if let Some(background) = &output.background {
                    self.observe_durable_background_process(background).await?;
                }
                // Typed mutations already report exact paths. A foreground
                // bash command reports its opaque effects from the before/after
                // snapshot inside the tool; a final turn reconciliation covers
                // long-lived background commands without adding a full walk
                // after every shell call.
                if name != "bash" && !output.effects.file_changes.is_empty() {
                    let paths: Vec<String> = output
                        .effects
                        .file_changes
                        .iter()
                        .map(|change| change.path.clone())
                        .collect();
                    self.runtime.reconcile_dirty_paths_async(paths).await?;
                }
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
                let signature_was_seen = signature_seen(evidence, &signature);
                let tracker_before = implementation_tracker.clone();
                let validation_succeeded = tool_satisfies_validation(&output);
                evidence.record_success(name, arguments, &semantic_output);
                implementation_tracker.record_tool_result(
                    name,
                    arguments,
                    &semantic_output,
                    validation_succeeded,
                    output.effects.mutation_applied,
                );
                let progress = progress_tracker
                    .tool_guardrail
                    .record_tool_result_with_effects(
                        name,
                        arguments,
                        &semantic_output,
                        output.effects.mutation_applied,
                    );
                if progress.running_background_poll {
                    running_background_poll_results += 1;
                }
                if progress.actionable_background_output {
                    actionable_poll_results += 1;
                }
                if wait_flavored_call(name, arguments, &output) {
                    wait_flavored_results += 1;
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
                    self.runtime.root(),
                );
                progress_tracker.record_tool(&progress_label);
                tool_progress_labels.push(progress_label.clone());
                tool_timeline.push(tool_entry_with_args(
                    name.clone(),
                    path,
                    duration_ms,
                    &output,
                    &progress_label,
                    arguments,
                ));
                emit_tool_output(&mut *ui, id, name, &output);
                append_tool_images(&output, &mut vision);
                results[i] = Some((id.clone(), output.content));
                self.invalidate_snapshot();
                completed[i] = true;
                completion_order.push(i);
                if let Some(entry) = tool_timeline.last_mut() {
                    entry.completion_index = completion_order.len() as u32;
                }
                done += 1;
                // Bash runs alone → a serial run and a batch of size 1.
                saturating_add_scheduler_count(sched_tool_calls, 1);
                saturating_add_scheduler_count(sched_serial_runs, 1);
                *sched_max_concurrent = (*sched_max_concurrent).max(1);
                continue;
            }
            // Parallel explore: `explore` is read-only and independent —
            // multiple ready explores can run concurrently. Prepare all jobs
            // (budget check, config extraction) sequentially, run the child
            // turns in parallel via `FuturesUnordered`, then process results
            // sequentially. Each child writes live status through a
            // SubagentSink so `&mut dyn Ui` is never shared across concurrent
            // futures, and child tool calls stay off the parent transcript.
            let explore_indices: Vec<usize> = ready
                .iter()
                .copied()
                .filter(|&i| calls[i].1 == "explore")
                .collect();
            if !explore_indices.is_empty() {
                // Prepare jobs for all ready explores. Their count is
                // unlimited; concurrent fan-out remains bounded below.
                let mut prepared: Vec<(usize, ExploreJob)> = Vec::new();
                let mut unavailable_explores: Vec<usize> = Vec::new();
                for &i in &explore_indices {
                    let (id, _, arguments) = &calls[i];
                    if let Some(job) = self.prepare_explore(arguments) {
                        let summary = crate::clip_subagent_description(&job.task);
                        let id_ui = format!("explore-{}", job.slot);
                        ui.subagent_spawned(&id_ui, "explore", &summary, false);
                        ui.tool_call_id(id, "explore", arguments);
                        prepared.push((i, job));
                    } else {
                        unavailable_explores.push(i);
                    }
                }
                // Complete malformed/unavailable explores immediately.
                for i in unavailable_explores {
                    let (id, _, arguments) = &calls[i];
                    ui.tool_call_id(id, "explore", arguments);
                    let msg = "explore unavailable: could not prepare the requested subagent; \
                               investigate directly."
                        .to_string();
                    let output = explore_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                    emit_tool_output(&mut *ui, id, "explore", &output);
                    let signature = inspection_signature("explore", arguments);
                    let progress_label = ToolProgressLabel::new(
                        ProgressKind::Weak,
                        "explore unavailable",
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
                    if let Some(entry) = tool_timeline.last_mut() {
                        entry.completion_index = completion_order.len() as u32;
                    }
                    done += 1;
                    saturating_add_scheduler_count(sched_tool_calls, 1);
                    saturating_add_scheduler_count(sched_serial_runs, 1);
                }
                // Run prepared explores concurrently and consume each result as
                // soon as it finishes. UI progress no longer waits for the
                // slowest child, while transcript insertion remains ordered at
                // the end of the parent batch.
                // Run prepared explores concurrently. Each child writes live
                // status through a SubagentSink; child tool calls stay off the
                // parent transcript.
                let sink = ui.subagent_sink();
                let max_concurrent = MAX_PARALLEL_EXPLORES.min(prepared.len());
                let mut explore_futures =
                    futures_util::stream::iter(prepared.into_iter().map(|(i, job)| {
                        let sink = sink.clone();
                        let id_ui = format!("explore-{}", job.slot);
                        async move {
                            let started = std::time::Instant::now();
                            let mut child_ui =
                                crate::subagent_progress::SubagentProgressUi { id: id_ui, sink };
                            let result = run_explore_job(job, &mut child_ui).await;
                            (i, result, started.elapsed().as_millis() as u64)
                        }
                    }))
                    .buffer_unordered(max_concurrent);
                while let Some((i, result, duration_ms)) = explore_futures.next().await {
                    let (id, _, arguments) = &calls[i];
                    let slot = result.slot;
                    let output = self.finish_explore(result);
                    let status = crate::subagent_finish_status(output.status);
                    let finish_summary: String = output.content.chars().take(120).collect();
                    ui.subagent_finished(
                        &format!("explore-{slot}"),
                        status,
                        duration_ms,
                        &finish_summary,
                    );
                    let error = output.status != hi_tools::ToolStatus::Succeeded;
                    let semantic_output = if error && !output.content.starts_with("Error:") {
                        std::borrow::Cow::Owned(format!("Error: {}", output.content))
                    } else {
                        std::borrow::Cow::Borrowed(output.content.as_str())
                    };
                    let signature = inspection_signature("explore", arguments);
                    let signature_was_seen = signature_seen(evidence, &signature);
                    let tracker_before = implementation_tracker.clone();
                    let validation_succeeded = tool_satisfies_validation(&output);
                    evidence.record_success("explore", arguments, &semantic_output);
                    implementation_tracker.record_tool_result(
                        "explore",
                        arguments,
                        &semantic_output,
                        validation_succeeded,
                        output.effects.mutation_applied,
                    );
                    let progress = progress_tracker
                        .tool_guardrail
                        .record_tool_result_with_effects(
                            "explore",
                            arguments,
                            &semantic_output,
                            output.effects.mutation_applied,
                        );
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
                        self.runtime.root(),
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
                    append_tool_images(&output, &mut vision);
                    results[i] = Some((id.clone(), output.content));
                    completed[i] = true;
                    completion_order.push(i);
                    if let Some(entry) = tool_timeline.last_mut() {
                        entry.completion_index = completion_order.len() as u32;
                    }
                    done += 1;
                    saturating_add_scheduler_count(sched_tool_calls, 1);
                }
                *sched_max_concurrent = (*sched_max_concurrent).max(max_concurrent as u32);
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
                // Prepare all delegate jobs (optional quota, runner, file-set extraction).
                let mut prepared_delegates: Vec<(usize, DelegateJob, u64)> = Vec::new();
                let mut delegate_prep_failed: Vec<usize> = Vec::new();
                for &i in &delegate_indices {
                    let (_, _, arguments) = &calls[i];
                    if let Some((job, ledger_rev)) = self.prepare_delegate(arguments) {
                        prepared_delegates.push((i, job, ledger_rev));
                    } else {
                        delegate_prep_failed.push(i);
                    }
                }
                // Check every pair of declared workspace scopes. Directory/file
                // containment counts as overlap; unknown scopes stay serial.
                let all_disjoint =
                    prepared_delegates
                        .iter()
                        .enumerate()
                        .all(|(index, (_, job, _))| {
                            prepared_delegates[index + 1..].iter().all(|(_, other, _)| {
                                file_sets_disjoint(&job.file_set, &other.file_set)
                            })
                        });
                if all_disjoint && !prepared_delegates.is_empty() {
                    // Complete prep-failed delegates immediately.
                    for i in delegate_prep_failed {
                        let (id, _, arguments) = &calls[i];
                        ui.tool_call_id(id, "delegate", arguments);
                        let limit = delegate_turn_limit();
                        let (msg, label) =
                            if limit != u32::MAX && self.subagents.delegate_turn_used >= limit {
                                (delegate_limit_denial(limit), "delegate limit reached")
                            } else {
                                (
                                    "delegate unavailable: could not prepare the requested \
                                     subagent; implement it directly."
                                        .to_string(),
                                    "delegate unavailable",
                                )
                            };
                        let mut output =
                            synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
                        output.effects.mutation_attempted = true;
                        emit_tool_output(&mut *ui, id, "delegate", &output);
                        let signature = inspection_signature("delegate", arguments);
                        let progress_label =
                            ToolProgressLabel::new(ProgressKind::Weak, label, signature);
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
                        if let Some(entry) = tool_timeline.last_mut() {
                            entry.completion_index = completion_order.len() as u32;
                        }
                        done += 1;
                        saturating_add_scheduler_count(sched_tool_calls, 1);
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
                            let output =
                                synthetic_tool_outcome(msg.clone(), hi_tools::ToolStatus::Denied);
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
                            saturating_add_scheduler_count(sched_tool_calls, 1);
                        }
                        saturating_add_scheduler_count(sched_serial_runs, prepared_delegates.len());
                        *sched_max_concurrent = (*sched_max_concurrent).max(1);
                        continue;
                    }
                    // Spawn live rows only once we know these jobs will run in
                    // parallel. Preparing then falling back to serial used to
                    // emit a row here and a second row from `handle_delegate`.
                    for (i, job, _) in prepared_delegates.iter_mut() {
                        let (id, _, arguments) = &calls[*i];
                        let summary = crate::clip_subagent_description(&job.task);
                        let delegate_id = format!("delegate-{}", job.slot);
                        ui.subagent_spawned(&delegate_id, "delegate", &summary, false);
                        ui.subagent_progress(&delegate_id, "running");
                        job.progress =
                            crate::subagent_progress::bound_sink_progress(ui, &delegate_id);
                        ui.tool_call_id(id, "delegate", arguments);
                    }
                    // Run prepared delegates concurrently and process each
                    // completion immediately. The runner's destination merge is
                    // transactionally serialized, but fast children no longer
                    // wait for the slowest child before reconciliation/UI output.
                    let max_concurrent = parallel_delegate_limit().min(prepared_delegates.len());
                    let mut delegate_results =
                        futures_util::stream::iter(prepared_delegates.into_iter().map(
                            |(i, job, ledger_rev)| async move {
                                let started = std::time::Instant::now();
                                let result = run_delegate_job(job).await;
                                (i, result, ledger_rev, started.elapsed().as_millis() as u64)
                            },
                        ))
                        .buffer_unordered(max_concurrent);
                    while let Some((i, result, ledger_rev, duration_ms)) =
                        delegate_results.next().await
                    {
                        let (id, _, arguments) = &calls[i];
                        let output = self
                            .finish_delegate(result, ledger_rev, &mut *ui, duration_ms)
                            .await;
                        let error = output.status != hi_tools::ToolStatus::Succeeded;
                        let semantic_output = if error && !output.content.starts_with("Error:") {
                            std::borrow::Cow::Owned(format!("Error: {}", output.content))
                        } else {
                            std::borrow::Cow::Borrowed(output.content.as_str())
                        };
                        let signature = inspection_signature("delegate", arguments);
                        let signature_was_seen = signature_seen(evidence, &signature);
                        let tracker_before = implementation_tracker.clone();
                        let validation_succeeded = tool_satisfies_validation(&output);
                        evidence.record_success("delegate", arguments, &semantic_output);
                        implementation_tracker.record_tool_result(
                            "delegate",
                            arguments,
                            &semantic_output,
                            validation_succeeded,
                            output.effects.mutation_applied,
                        );
                        let progress = progress_tracker
                            .tool_guardrail
                            .record_tool_result_with_effects(
                                "delegate",
                                arguments,
                                &semantic_output,
                                output.effects.mutation_applied,
                            );
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
                                self.runtime.root(),
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
                        append_tool_images(&output, &mut vision);
                        results[i] = Some((id.clone(), output.content));
                        completed[i] = true;
                        completion_order.push(i);
                        if let Some(entry) = tool_timeline.last_mut() {
                            entry.completion_index = completion_order.len() as u32;
                        }
                        done += 1;
                        saturating_add_scheduler_count(sched_tool_calls, 1);
                    }
                    *sched_max_concurrent = (*sched_max_concurrent).max(max_concurrent as u32);
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
                        | "ask_user"
                        | "new_context"
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
                            if decision == ConfirmationResult::Parked {
                                self.note_approval_parked(ui);
                            }
                            ui.tool_call_id(id, name, arguments);
                            let (msg, status) = parked_or_denied_delegate(&decision);
                            let mut output = synthetic_tool_outcome(msg.clone(), status);
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
                            if let Some(entry) = tool_timeline.last_mut() {
                                entry.completion_index = completion_order.len() as u32;
                            }
                            done += 1;
                            saturating_add_scheduler_count(sched_tool_calls, 1);
                            saturating_add_scheduler_count(sched_serial_runs, 1);
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
                        if let Some(entry) = tool_timeline.last_mut() {
                            entry.completion_index = completion_order.len() as u32;
                        }
                        done += 1;
                        saturating_add_scheduler_count(sched_tool_calls, 1);
                        saturating_add_scheduler_count(sched_serial_runs, 1);
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
                    "ask_user" => self.handle_ask_user(arguments, &mut *ui).await,
                    "new_context" => self.handle_new_context(),
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
                let signature_was_seen = signature_seen(evidence, &signature);
                let tracker_before = implementation_tracker.clone();
                let validation_succeeded = tool_satisfies_validation(&output);
                evidence.record_success(name, arguments, &semantic_output);
                implementation_tracker.record_tool_result(
                    name,
                    arguments,
                    &semantic_output,
                    validation_succeeded,
                    output.effects.mutation_applied,
                );
                let progress = progress_tracker
                    .tool_guardrail
                    .record_tool_result_with_effects(
                        name,
                        arguments,
                        &semantic_output,
                        output.effects.mutation_applied,
                    );
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
                        self.runtime.root(),
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
                append_tool_images(&output, &mut vision);
                results[i] = Some((id.clone(), output.content));
                completed[i] = true;
                completion_order.push(i);
                if let Some(entry) = tool_timeline.last_mut() {
                    entry.completion_index = completion_order.len() as u32;
                }
                done += 1;
                // Runs alone, like bash.
                saturating_add_scheduler_count(sched_tool_calls, 1);
                saturating_add_scheduler_count(sched_serial_runs, 1);
                *sched_max_concurrent = (*sched_max_concurrent).max(1);
                continue;
            }
            // Run all ready non-bash calls concurrently. Record the
            // completion order as the ready order (within a concurrent
            // batch, relative order doesn't matter — none depend on
            // each other, or they wouldn't all be ready).
            let batch_size = ready.len();
            // Small ready sets should start immediately; broad cheap-read waves
            // scale to the configured cap. Mutating/coordination-heavy batches
            // stay narrower to preserve foreground responsiveness.
            let read_only_ready = ready
                .iter()
                .filter(|&&i| {
                    hi_tools::is_read_only(&calls[i].1)
                        || (calls[i].1 == "bash"
                            && matches!(
                                bash_command(&calls[i].2)
                                    .map(|c| classify_bash_command(&c))
                                    .unwrap_or(BashCommandKind::Unknown),
                                BashCommandKind::Inspection
                            ))
                })
                .count();
            let dynamic_parallel_tools = if read_only_ready == ready.len() {
                max_parallel_tools.min(ready.len())
            } else {
                max_parallel_tools.min(ready.len()).min(4)
            }
            .max(1);
            let actual_concurrency = dynamic_parallel_tools as u32;
            // Signal each call as started so the live TUI can show a
            // "running {tool}" timer. The transcript header is emitted
            // later, paired with its result, so headers and results
            // never drift apart in a concurrent batch.
            for &i in &ready {
                ui.tool_started_id(&calls[i].0, &calls[i].1, &calls[i].2);
            }
            // A frontend can raise Esc from `tool_started_id` itself. For a
            // concurrent batch, all calls have only been announced and none
            // should run after that signal; let the cancellation branch above
            // synthesize results for the whole pending batch. A single ready
            // call retains the historical boundary: the announced call may
            // finish, and the next scheduler iteration cancels later calls.
            if ready.len() > 1 && self.interrupt.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
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
                    if self.approval_parked {
                        let name = &calls[i].1;
                        if matches!(
                            name.as_str(),
                            "write" | "edit" | "multi_edit" | "apply_patch"
                        ) || egress_confirm_required(
                            self.permission_mode,
                            self.config.gates.confirm_edits,
                            name,
                        ) {
                            denied.push(i);
                        }
                        continue;
                    }
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
                            if decision == ConfirmationResult::Parked {
                                self.note_approval_parked(ui);
                            } else if decision == ConfirmationResult::Unavailable {
                                ui.status("confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits");
                            }
                            denied.push(i);
                        } else {
                            prepared_mutations.insert(i, prepared);
                        }
                    } else if egress_confirm_required(
                        self.permission_mode,
                        self.config.gates.confirm_edits,
                        name,
                    ) {
                        if self.approval_parked {
                            denied.push(i);
                            continue;
                        }
                        let decision = ui
                            .confirm(confirmation_for_egress_tool(name, &calls[i].2))
                            .await;
                        if decision != ConfirmationResult::Approved {
                            if decision == ConfirmationResult::Parked {
                                self.note_approval_parked(ui);
                            } else if decision == ConfirmationResult::Unavailable {
                                ui.status("confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits");
                            }
                            denied.push(i);
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
            let process_runner = self.runtime.process_runner().clone();
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
                    let repo_map = self.runtime.repo_map_arc();
                    let process_runner = process_runner.clone();
                    let mcp = self.mcp.clone();
                    let memory = self.memory.clone();
                    let calls = &calls;
                    async move {
                        let output = if let Some(failure) = failure {
                            failure
                        } else if let Some(prepared) = prepared {
                            execute_prepared_in_runtime(lsp, read_cache, prepared).await
                        } else {
                            execute_in_runtime_shared_with_runner(
                                &process_runner,
                                root,
                                state_root,
                                lsp,
                                background,
                                read_cache,
                                &repo_map,
                                mcp.as_deref(),
                                memory.as_deref(),
                                &calls[i].1,
                                &calls[i].2,
                            )
                            .await
                        };
                        (i, output)
                    }
                }))
                .buffer_unordered(dynamic_parallel_tools)
                .collect()
                .await;
            let batch_duration_ms = batch_started.elapsed().as_millis() as u64;
            // Scheduler telemetry: count every call in the ready batch,
            // but report actual concurrency after the configured cap.
            saturating_add_scheduler_count(sched_tool_calls, batch_size);
            *sched_max_concurrent = (*sched_max_concurrent).max(actual_concurrency);
            if actual_concurrency == 1 {
                saturating_add_scheduler_count(sched_serial_runs, batch_size);
            }
            // Handle denied calls first: emit their headers and "skipped" results.
            for &i in &denied {
                let name = &calls[i].1;
                ui.tool_call_id(&calls[i].0, name, &calls[i].2);
                let skipped_msg = if checkpoint_denied.contains(&i) {
                    "Mutation skipped because strict mode requires an available checkpoint."
                        .to_string()
                } else if self.approval_parked {
                    PARKED_TOOL_RESULT.to_string()
                } else if matches!(
                    name.as_str(),
                    "browser_exec" | "use_tool" | "web_fetch" | "research" | "research_read"
                ) {
                    "External action skipped by user (not run).".to_string()
                } else {
                    "Edit skipped by user (not applied).".to_string()
                };
                let status = if self.approval_parked {
                    hi_tools::ToolStatus::Cancelled
                } else {
                    hi_tools::ToolStatus::Denied
                };
                let mut output = synthetic_tool_outcome(skipped_msg.clone(), status);
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
                if let Some(entry) = tool_timeline.last_mut() {
                    entry.completion_index = completion_order.len() as u32;
                }
                done += 1;
            }
            // A write/check in the same provider batch is valid evidence even
            // when scheduler completion order happens to visit update_plan
            // first. Compute this over the complete successful batch.
            let batch_supplies_plan_completion_evidence = implementation_tracker.mutation_seen
                || implementation_tracker.validation_seen
                || outputs.iter().any(|(index, output)| {
                    output.status == hi_tools::ToolStatus::Succeeded
                        && (output.effects.mutation_applied
                            || (tool_satisfies_validation(output)
                                && crate::steering::implementation_tool_call_validates(
                                    &calls[*index].1,
                                    &calls[*index].2,
                                )))
                });
            for (i, mut output) in outputs {
                let name = &calls[i].1;
                let mut unsupported_completion_claims = Vec::new();
                // In plan mode, `update_plan` describes work to execute after
                // approval; it is not evidence that the work already ran. A
                // weak model can otherwise flip a freshly drafted checklist
                // from 0/N to N/N in one bookkeeping-only call, clearing the
                // durable plan and making the UI look as though the edits and
                // checks happened. Preserve completion established before
                // entering plan mode, but keep all unfinished work pending.
                if name == "update_plan"
                    && self.plan_mode
                    && let Some(plan) = output.plan.as_mut()
                {
                    let corrected = normalize_plan_mode_update(self.goals.plan(), plan);
                    let done = plan
                        .iter()
                        .filter(|step| step.status == PlanStatus::Done)
                        .count();
                    output.content = format!(
                        "Plan recorded: {done}/{} done. Plan mode keeps unexecuted steps pending until approval.",
                        plan.len()
                    );
                    if corrected > 0 {
                        ui.status(&format!(
                            "kept {corrected} unexecuted plan step(s) pending until plan approval"
                        ));
                    }
                }
                if name == "update_plan"
                    && !self.plan_mode
                    && let Some(plan) = output.plan.as_mut()
                {
                    unsupported_completion_claims = normalize_unsupported_plan_completion(
                        self.goals.plan(),
                        plan,
                        &calls[i].2,
                        batch_supplies_plan_completion_evidence,
                    );
                    let corrected = unsupported_completion_claims.len();
                    if corrected > 0 {
                        let done = plan
                            .iter()
                            .filter(|step| step.status == PlanStatus::Done)
                            .count();
                        output.content = format!(
                            "Plan recorded: {done}/{} done. Kept {corrected} implementation step(s) unfinished because this turn lacks mutation or validation evidence scoped to those steps. Do the work, or add a concrete per-step completion_evidence reason when no change is genuinely required.",
                            plan.len()
                        );
                        ui.status(&format!(
                            "kept {corrected} unsupported implementation completion claim(s) unfinished"
                        ));
                    }
                }
                // Emit the transcript header immediately before its
                // result — in a concurrent batch this pairs each header
                // with its own result in completion order.
                ui.tool_call_id(&calls[i].0, name, &calls[i].2);
                let path = hi_tools::target_path(name, &calls[i].2).unwrap_or_default();
                self.record_tool_effects(&output.effects)?;
                // Poll/kill outcomes are authoritative durability notifications.
                if let Some(background) = &output.background {
                    self.observe_durable_background_process(background).await?;
                }
                for change in &output.effects.file_changes {
                    batch_mutated_paths.insert(change.path.clone());
                }
                let error = output.status != hi_tools::ToolStatus::Succeeded;
                let semantic_output = if error && !output.content.starts_with("Error:") {
                    std::borrow::Cow::Owned(format!("Error: {}", output.content))
                } else {
                    std::borrow::Cow::Borrowed(output.content.as_str())
                };
                let signature = inspection_signature(name, &calls[i].2);
                let signature_was_seen = signature_seen(evidence, &signature);
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
                    output.effects.mutation_applied,
                );
                let progress = progress_tracker
                    .tool_guardrail
                    .record_tool_result_with_effects(
                        name,
                        &calls[i].2,
                        &semantic_output,
                        output.effects.mutation_applied,
                    );
                if progress.running_background_poll {
                    running_background_poll_results += 1;
                }
                if progress.actionable_background_output {
                    actionable_poll_results += 1;
                }
                if wait_flavored_call(name, &calls[i].2, &output) {
                    wait_flavored_results += 1;
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
                    self.runtime.root(),
                );
                progress_tracker.record_tool(&progress_label);
                tool_progress_labels.push(progress_label.clone());
                tool_timeline.push(tool_entry_with_args(
                    name.clone(),
                    path,
                    batch_duration_ms,
                    &output,
                    &progress_label,
                    &calls[i].2,
                ));
                emit_tool_output(&mut *ui, &calls[i].0, name, &output);
                append_tool_images(&output, &mut vision);
                results[i] = Some((calls[i].0.clone(), output.content));
                // Track the latest plan state so the continue logic can
                // detect an incomplete plan when the model stops calling
                // tools. The model resubmits the whole list on every
                // call, so the last one is always current.
                if calls[i].1 == "update_plan"
                    && let Some(plan) = output.plan.as_deref()
                {
                    if let Some(session) = self.session.as_mut() {
                        if plan_has_pending_steps(plan) {
                            session.record_plan(plan)?;
                        } else {
                            // Keep the completed checklist visible for this live
                            // turn, but do not resurrect it after a restart.
                            session.clear_plan()?;
                        }
                    }
                    // Publish live state only after its durable write succeeds.
                    // Otherwise an I/O error leaves this process showing a new
                    // plan while restart restores the old one.
                    let _ = self.goals.replace_plan(plan);
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
                        // Normalizing an unsupported `done` claim back to the
                        // current status must not erase the durable warning
                        // that the claim happened. Keep the note separately;
                        // unlike replaying the raw plan, this cannot advance
                        // the active step without execution evidence.
                        goal.record_unsupported_completion_claims(&unsupported_completion_claims);
                        *plan_updated_goal = true;
                    }
                }
                // A filesystem-mutating tool may have changed files —
                // invalidate the snapshot cache so a dependent read
                // (guaranteed to run after by the dep graph) re-walks.
                // `bash` also invalidates but always runs alone (above).
                if hi_tools::is_filesystem_mutating(&calls[i].1)
                    || calls[i].1 == "bash"
                    || (self.pipefs_workspace_active() && calls[i].1 == "use_tool")
                {
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
                        let check_label = check.clone();
                        let check_path = std::path::PathBuf::from(&path);
                        let handle = tokio::spawn(async move {
                            hi_tools::run_fast_check_in(&root, &check, &check_path).await
                        });
                        pending_checks.push((
                            path,
                            check_label,
                            tokio_util::task::AbortOnDropHandle::new(handle),
                        ));
                    }
                }
                completed[i] = true;
                completion_order.push(i);
                if let Some(entry) = tool_timeline.last_mut() {
                    entry.completion_index = completion_order.len() as u32;
                }
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
        if workspace_intent.is_none() {
            let pending = self.runtime.background().pending_job_settlements().await;
            let retained = tool_timeline.as_slice();
            let batch_entries = retained
                .len()
                .checked_sub(calls.len())
                .and_then(|start| retained.get(start..))
                .unwrap_or(&[]);
            if terminal_background_requires_reconciliation(batch_entries, &pending) {
                speculation_registry.invalidate_all();
                let intent = hi_workspace::MutationIntent::reconciliation();
                self.workspace_coordination
                    .begin_intent(self.workspace_durability.clone(), intent.clone())
                    .await?;
                workspace_intent = Some(intent);
            }
        }
        append_fast_feedback(
            self,
            calls,
            pending_checks,
            batch_mutated_paths,
            task_contract,
            fast_feedback,
            implementation_tracker,
            &mut results,
            ui,
        )
        .await;
        // Stage the exact result before remote settlement, then publish it to
        // the live/provider transcript only after the workspace controller has
        // returned a receipt. This prevents both a one-step-behind causal batch
        // and a locally visible false-success result after an ambiguous commit.
        let retained = tool_timeline.as_slice();
        let batch_entries = retained
            .len()
            .checked_sub(calls.len())
            .and_then(|start| retained.get(start..))
            .unwrap_or(&[]);
        if let Some(intent) = workspace_intent.as_ref() {
            let execution = workspace_execution_report(intent, batch_entries, calls.len());
            if let Err(stage_error) = self.stage_active_workspace_execution(
                calls,
                completion_content,
                &results,
                &execution,
            ) {
                let mut indeterminate = execution;
                indeterminate.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
                indeterminate.detail = Some(format!(
                    "workspace effects ran, but their transcript could not be staged: {stage_error:#}"
                ));
                return match self
                    .checkpoint_durable_workspace_with_execution(indeterminate)
                    .await
                {
                    Err(settlement_error) => Err(settlement_error).context(format!(
                        "workspace transcript staging failed before settlement: {stage_error:#}"
                    )),
                    Ok(()) => Err(stage_error).context(
                        "workspace transcript staging failed; execution remains indeterminate",
                    ),
                };
            }
            self.checkpoint_durable_workspace_with_execution(execution)
                .await?;
        }
        self.messages
            .push_assistant_with_results(std::mem::take(completion_content), results);
        if !vision.is_empty() {
            let mut content: Vec<Content> = vision
                .into_iter()
                .map(|image| Content::Image {
                    data: image.data,
                    media_type: image.media_type,
                })
                .collect();
            content.push(Content::Text("Screenshot(s) from browser_exec.".into()));
            self.messages.push_user_or_fold_message(hi_ai::Message {
                role: hi_ai::Role::User,
                content,
            });
        }
        // Collapse older polls of the handles polled this batch to one-line
        // digests: only the newest poll carries information the model still
        // needs, and a long watch otherwise re-sends every stale progress
        // dump on every subsequent request.
        let mut folded_handles: BTreeSet<String> = BTreeSet::new();
        for (_, name, arguments) in calls {
            if name == "bash_output"
                && let Some(handle) = crate::transcript::background_poll_handle(arguments)
                && folded_handles.insert(handle.clone())
            {
                self.messages.fold_superseded_background_polls(&handle);
            }
        }
        // Same for re-reads of the same file region: after each edit models
        // tend to re-read the whole file (one real session read the same
        // source file 21×), and only the newest copy reflects reality.
        for (_, name, arguments) in calls {
            if name == "read"
                && let Some(key) = crate::transcript::read_call_key(arguments)
            {
                self.messages.fold_superseded_file_reads(&key);
            }
        }
        // A fully cancelled batch did not execute discovery or implementation
        // work, so it must not burn the mutation-recovery round budget.
        if interrupted_calls < calls.len() {
            implementation_tracker.record_tool_round();
        }

        Ok(ToolBatchOutcome {
            calls: calls.to_vec(),
            read_only_intent,
            hash_guard_applies,
            hashable_idempotent_results,
            repeated_idempotent_results,
            running_background_poll_results,
            actionable_poll_results,
            wait_flavored_results,
            tool_progress_labels,
            plan_changed_this_batch,
            interrupted_calls,
            interrupted_coordination_calls,
            protocol_validation_errors,
            unknown_background_handles: self.runtime.background().unknown_handles(),
            program_fallback_exhausted: false,
        })
    }

    /// Execute one negotiated program as a single provider-facing tool
    /// result. Nested calls are visible as activity rows only; they never get
    /// appended to the provider transcript independently.
    #[allow(
        clippy::too_many_arguments,
        reason = "program execution must share the turn's existing accounting and UI state"
    )]
    async fn execute_program_batch(
        &mut self,
        calls: &[(String, String, String)],
        completion_content: &mut Vec<Content>,
        tool_specs: &[hi_ai::ToolSpec],
        tool_envelope: &hi_tools::envelope::ToolEnvelope,
        read_only_intent: Option<crate::steering::ReviewIntent>,
        progress_tracker: &mut ProgressTracker,
        tool_timeline: &mut ToolTimeline,
        sched_tool_calls: &mut u32,
        sched_max_concurrent: &mut u32,
        sched_serial_runs: &mut u32,
        speculation_registry: &SpeculationRegistry,
        program_fallback_next: &mut bool,
        program_fallback_used: &mut bool,
        ui: &mut dyn Ui,
    ) -> Result<ToolBatchOutcome> {
        let started = std::time::Instant::now();
        let mut program_fallback_exhausted = false;
        let (program_index, (id, _, arguments)) = calls
            .iter()
            .enumerate()
            .find(|(_, (_, name, _))| name == "run_program")
            .expect("program batch is entered with at least one program call");
        let mut outer_status = hi_tools::ToolStatus::Succeeded;
        let envelope_error = if !tool_envelope.digest_is_valid() {
            Some("tool envelope digest does not match its payload".to_string())
        } else if !tool_envelope.matches_specs(tool_specs) {
            Some("execution schemas do not match the sealed tool envelope".to_string())
        } else if !tool_envelope.admits("run_program") {
            Some(format!(
                "run_program is outside the model request's sealed envelope {}",
                tool_envelope.digest
            ))
        } else {
            hi_ai::validate_client_tool_call(id, "run_program", arguments, tool_specs)
                .err()
                .map(|error| error.to_string())
        };
        let (outcome, program_effect_may_have_occurred, workspace_intent) = if calls.len() != 1 {
            outer_status = hi_tools::ToolStatus::Failed;
            (
                ProgramOutcome::Failed {
                    error: "run_program must be the only tool call in a completion; retry with ordinary structured tools".into(),
                    calls: Vec::new(),
                },
                false,
                None,
            )
        } else if let Some(error) = envelope_error {
            outer_status = hi_tools::ToolStatus::Denied;
            (
                ProgramOutcome::Failed {
                    error,
                    calls: Vec::new(),
                },
                false,
                None,
            )
        } else if !self.config.program.mode_enabled() {
            outer_status = hi_tools::ToolStatus::Denied;
            (
                ProgramOutcome::Failed {
                    error: "run_program is disabled; retry with ordinary structured tools".into(),
                    calls: Vec::new(),
                },
                false,
                None,
            )
        } else {
            match serde_json::from_str::<serde_json::Value>(arguments)
                .ok()
                .and_then(|value| {
                    value
                        .get("source")
                        .and_then(|source| source.as_str())
                        .map(str::to_owned)
                }) {
                Some(source) => {
                    // The nested tool set is selected dynamically. Admit one
                    // conservative outer operation before the program can
                    // dispatch any effect, then settle it once with the exact
                    // provider-facing aggregate result.
                    let intent = workspace_program_intent("run_program", arguments, None);
                    match self
                        .begin_classified_workspace_operation(intent.clone())
                        .await
                    {
                        Ok(()) => {
                            if self.config.program.speculation_enabled() {
                                self.launch_program_speculation(
                                    speculation_registry,
                                    id,
                                    &source,
                                    tool_envelope,
                                );
                            }
                            let remaining_calls = self.max_tool_calls_cap().map(|_| {
                                self.config
                                    .loop_limits
                                    .remaining_tool_calls(*sched_tool_calls)
                                    .saturating_sub(1) as usize
                            });
                            let (outcome, effect_may_have_occurred) = self
                                .run_program_host(
                                    source,
                                    id,
                                    speculation_registry,
                                    remaining_calls,
                                    tool_envelope,
                                    ui,
                                )
                                .await;
                            speculation_registry.cancel_all();
                            let _ = speculation_registry.telemetry();
                            (outcome, effect_may_have_occurred, Some(intent))
                        }
                        Err(error) => (
                            ProgramOutcome::Failed {
                                error: format!("workspace operation blocked: {error:#}"),
                                calls: Vec::new(),
                            },
                            false,
                            None,
                        ),
                    }
                }
                None => {
                    outer_status = hi_tools::ToolStatus::Failed;
                    (
                        ProgramOutcome::Failed {
                            error: "run_program requires a string `source` argument; retry with ordinary structured tools".into(),
                            calls: Vec::new(),
                        },
                        false,
                        None,
                    )
                }
            }
        };
        if matches!(outcome, ProgramOutcome::Cancelled { .. }) {
            outer_status = hi_tools::ToolStatus::Cancelled;
        } else if matches!(outcome, ProgramOutcome::Failed { .. }) {
            if outer_status != hi_tools::ToolStatus::Denied {
                outer_status = hi_tools::ToolStatus::Failed;
            }
            // Give the model one ordinary-tool recovery request. This flag is
            // consumed while shaping that next request, so a repeated model
            // failure cannot create an unbounded fallback loop.
            if *program_fallback_used {
                program_fallback_exhausted = true;
            } else {
                *program_fallback_used = true;
                *program_fallback_next = true;
            }
        }
        let aggregate = match &outcome {
            ProgramOutcome::Succeeded { result, calls } => serde_json::json!({
                "status": "succeeded",
                "result": result,
                "calls": calls,
            }),
            ProgramOutcome::Failed { error, calls } => serde_json::json!({
                "status": "failed",
                "error": error,
                "calls": calls,
            }),
            ProgramOutcome::Cancelled { calls } => serde_json::json!({
                "status": "cancelled",
                "error": "program cancelled",
                "calls": calls,
            }),
        };
        let raw = serde_json::to_string(&aggregate).unwrap_or_else(|_| {
            "{\"status\":\"failed\",\"error\":\"program result was not serializable\"}".into()
        });
        let (content, _) = hi_tools::bound_tool_content(raw);
        if calls.len() != 1 {
            // The provider cannot receive an assistant tool-use without a
            // matching result. Keep only the rejected program envelope in
            // the transcript; ordinary calls are deliberately not executed
            // and will be regenerated on the one-shot fallback request.
            let program_id = &calls[program_index].0;
            completion_content
                .retain(|block| !matches!(block, Content::ToolCall { id, .. } if id != program_id));
        }
        // Nested program calls use the same native tools as ordinary batches,
        // but their effects are hidden behind one provider-facing envelope.
        // Stage that exact envelope before settlement so neither workspace
        // bytes nor a no-head-change external receipt can get one turn ahead
        // of the transcript that explains it.
        if let Some(intent) = workspace_intent.as_ref() {
            let execution = workspace_program_execution_report(
                intent,
                &outcome,
                program_effect_may_have_occurred,
            );
            let results = vec![(id.clone(), content.clone())];
            if let Err(stage_error) = self.stage_active_workspace_execution(
                calls,
                completion_content,
                &results,
                &execution,
            ) {
                let mut indeterminate = execution;
                indeterminate.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
                indeterminate.detail = Some(format!(
                    "program effects ran, but their transcript could not be staged: {stage_error:#}"
                ));
                return match self
                    .checkpoint_durable_workspace_with_execution(indeterminate)
                    .await
                {
                    Err(settlement_error) => Err(settlement_error).context(format!(
                        "program transcript staging failed before settlement: {stage_error:#}"
                    )),
                    Ok(()) => Err(stage_error)
                        .context("program transcript staging failed; execution is indeterminate"),
                };
            }
            self.checkpoint_durable_workspace_with_execution(execution)
                .await?;
        }
        let output = synthetic_tool_outcome(content.clone(), outer_status);
        ui.tool_call_id(id, "run_program", arguments);
        emit_tool_output(&mut *ui, id, "run_program", &output);
        let label = ToolProgressLabel::new(
            if outer_status == hi_tools::ToolStatus::Succeeded {
                ProgressKind::Meaningful
            } else {
                ProgressKind::Weak
            },
            "program execution",
            inspection_signature("run_program", arguments),
        );
        progress_tracker.record_tool(&label);
        tool_timeline.push(tool_entry(
            "run_program".into(),
            String::new(),
            started.elapsed().as_millis() as u64,
            &output,
            &label,
        ));
        let nested_calls = match &outcome {
            ProgramOutcome::Succeeded { calls, .. }
            | ProgramOutcome::Failed { calls, .. }
            | ProgramOutcome::Cancelled { calls } => calls.len(),
        };
        // A program is one provider-facing envelope, but each nested operation
        // consumes the same turn budget as an ordinary structured tool call.
        saturating_add_scheduler_count(sched_tool_calls, 1);
        saturating_add_scheduler_count(sched_tool_calls, nested_calls);
        *sched_serial_runs = sched_serial_runs.saturating_add(1);
        *sched_max_concurrent = (*sched_max_concurrent).max(1);
        self.messages.push_assistant_with_results(
            std::mem::take(completion_content),
            vec![(id.clone(), content)],
        );
        Ok(ToolBatchOutcome {
            calls: calls.to_vec(),
            read_only_intent,
            hash_guard_applies: false,
            hashable_idempotent_results: 0,
            repeated_idempotent_results: 0,
            running_background_poll_results: 0,
            actionable_poll_results: 0,
            wait_flavored_results: 0,
            tool_progress_labels: vec![label],
            plan_changed_this_batch: false,
            interrupted_calls: usize::from(outer_status == hi_tools::ToolStatus::Cancelled),
            interrupted_coordination_calls: 0,
            protocol_validation_errors: Vec::new(),
            unknown_background_handles: self.runtime.background().unknown_handles(),
            program_fallback_exhausted,
        })
    }

    /// Launch only the conservative, explicitly classified prefix of a
    /// streamed program. This method is shared by the provider-delta path and
    /// the final completion path; the registry makes repeated prefixes cheap
    /// and prevents duplicate shadow calls.
    pub(crate) fn launch_program_speculation(
        &self,
        speculation_registry: &SpeculationRegistry,
        program_id: &str,
        source: &str,
        tool_envelope: &hi_tools::envelope::ToolEnvelope,
    ) {
        self.program_speculator(tool_envelope)
            .launch(speculation_registry, program_id, source);
    }

    pub(crate) fn program_speculator(
        &self,
        tool_envelope: &hi_tools::envelope::ToolEnvelope,
    ) -> ProgramSpeculator {
        ProgramSpeculator {
            runner: self.program_tool_runner(),
            allowed_tools: std::sync::Arc::new(
                tool_envelope
                    .payload
                    .program_tools
                    .iter()
                    .map(|tool| tool.name.clone())
                    .collect(),
            ),
            turn_id: format!("turn-{}", self.turn_count),
            enabled: self.config.program.speculation_enabled()
                && self.provider.capabilities().streamed_tool_call_deltas,
            // A speculative external request must not get ahead of an
            // approval prompt that the ordinary real path would show. In Ask
            // mode the egress policy currently requires approval for fetch/
            // research, so disable the whole external shadow class for this
            // turn while retaining local speculation.
            external_allowed: !egress_confirm_required(
                self.permission_mode,
                self.config.gates.confirm_edits,
                "web_fetch",
            ),
            max_calls: self.config.program.max_speculative_calls,
            context_generation: self.runtime.context_generation(),
            ledger_revision: self.runtime.ledger().revision(),
            external_freshness_epoch: external_freshness_epoch(
                self.config.program.external_ttl_seconds,
            ),
        }
    }

    async fn run_program_host(
        &mut self,
        source: String,
        program_call_id: &str,
        speculation_registry: &SpeculationRegistry,
        max_calls: Option<usize>,
        tool_envelope: &hi_tools::envelope::ToolEnvelope,
        ui: &mut dyn Ui,
    ) -> (ProgramOutcome, bool) {
        let tool_specs = tool_envelope.program_specs();
        let cancel = CancellationToken::new();
        let (host_tx, mut host_rx) = tokio::sync::mpsc::unbounded_channel();
        let params = ProgramRunParams {
            source,
            host_tx,
            cancel: cancel.clone(),
            max_ops: ProgramRunParams::DEFAULT_MAX_OPS,
            max_calls,
        };
        let mut task = tokio::task::spawn_blocking(move || hi_workflow::run_program(params));
        let turn_cancellation = self.turn_cancellation.clone();
        let watcher_cancel = cancel.clone();
        let watcher = tokio::spawn(async move {
            loop {
                if turn_cancellation
                    .as_ref()
                    .is_some_and(crate::TurnCancellation::is_cancelled)
                {
                    watcher_cancel.cancel();
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
        });
        let _run_guard = ProgramRunGuard::new(cancel.clone(), watcher);
        let program = async {
            let mut mutated_workspace = false;
            loop {
                tokio::select! {
                    joined = &mut task => {
                        let outcome = match joined {
                            Ok(outcome) => outcome,
                            Err(error) => ProgramOutcome::Failed { error: format!("program worker failed: {error}"), calls: Vec::new() },
                        };
                        break (outcome, mutated_workspace);
                    }
                    request = host_rx.recv() => {
                        let Some(request) = request else {
                            // The worker owns the last sender, so channel
                            // closure normally races with its successful task
                            // settlement. Join it instead of manufacturing a
                            // failure based on which ready select branch won.
                            let outcome = match (&mut task).await {
                                Ok(outcome) => outcome,
                                Err(error) => ProgramOutcome::Failed {
                                    error: format!("program worker failed: {error}"),
                                    calls: Vec::new(),
                                },
                            };
                            break (outcome, mutated_workspace);
                        };
                        match request {
                            ProgramHostRequest::ExecuteTool { call, reply } => {
                                let authorized = self
                                    .authorize_program_call(&call, &tool_specs, tool_envelope, ui)
                                    .await;
                                let mutates = authorized.is_none()
                                    && self.program_call_requires_settlement(&call);
                                mutated_workspace |= mutates;
                                let resolved = if let Some(denied) = authorized {
                                    denied
                                } else {
                                    self.resolve_program_call(
                                        &call,
                                        program_call_id,
                                        speculation_registry,
                                        &cancel,
                                    )
                                    .await
                                };
                                let (result, output) = self
                                    .observe_program_effect(&call, mutates, resolved)
                                    .await;
                                ui.tool_call_id(&format!("program:{}", call.occurrence), &call.name, &serde_json::to_string(&call.arguments).unwrap_or_default());
                                emit_tool_output(ui, &format!("program:{}", call.occurrence), &call.name, &output);
                                let _ = reply.send(result);
                            }
                            ProgramHostRequest::ParallelTools { calls, reply } => {
                                // Confirm externally visible calls before spawning
                                // the parallel batch. This keeps confirmation on
                                // the real path and avoids borrowing the UI from
                                // several concurrent futures.
                                let mut authorized_calls = Vec::with_capacity(calls.len());
                                for call in calls {
                                    let denied = self
                                        .authorize_program_call(
                                            &call,
                                            &tool_specs,
                                            tool_envelope,
                                            ui,
                                        )
                                        .await;
                                    authorized_calls.push((call, denied));
                                }
                                let agent = &*self;
                                let serialize_for_durability = authorized_calls.iter().any(|(call, denied)| {
                                    denied.is_none() && agent.program_call_requires_settlement(call)
                                });
                                let outputs = if serialize_for_durability {
                                    let mut outputs = Vec::with_capacity(authorized_calls.len());
                                    for (call, denied) in authorized_calls {
                                        let (mutates, resolved) = if let Some(denied) = denied {
                                            (false, denied)
                                        } else {
                                            let mutates = agent.program_call_requires_settlement(&call);
                                            mutated_workspace |= mutates;
                                            let resolved = agent
                                                .resolve_program_call(
                                                    &call,
                                                    program_call_id,
                                                    speculation_registry,
                                                    &cancel,
                                                )
                                                .await;
                                            (mutates, resolved)
                                        };
                                        let resolved = agent
                                            .observe_program_effect(&call, mutates, resolved)
                                            .await;
                                        outputs.push((call, resolved));
                                    }
                                    outputs
                                } else {
                                    let parallelism = agent.config.loop_limits.max_parallel_tools.clamp(1, 8);
                                    futures_util::stream::iter(authorized_calls.into_iter().map(|(call, denied)| {
                                        let registry = speculation_registry.clone();
                                        let cancel = cancel.clone();
                                        let program_call_id = program_call_id.to_string();
                                        async move {
                                            let result = match denied {
                                                Some(denied) => denied,
                                                None => agent
                                                    .resolve_program_call(
                                                        &call,
                                                        &program_call_id,
                                                        &registry,
                                                        &cancel,
                                                    )
                                                    .await,
                                            };
                                            (call, result)
                                        }
                                    }))
                                    .buffer_unordered(parallelism)
                                    .collect::<Vec<_>>()
                                    .await
                                };
                                let mut outputs = outputs;
                                outputs.sort_by_key(|(call, _)| call.occurrence);
                                let mut results = Vec::with_capacity(outputs.len());
                                let mut parallel_error = None;
                                for (call, (result, output)) in outputs {
                                    let nested_id = format!("program:{}", call.occurrence);
                                    let args = serde_json::to_string(&call.arguments).unwrap_or_default();
                                    ui.tool_call_id(&nested_id, &call.name, &args);
                                    emit_tool_output(ui, &nested_id, &call.name, &output);
                                    match result {
                                        Ok(value) => results.push(value),
                                        Err(error) => {
                                            parallel_error = Some(error);
                                            break;
                                        }
                                    }
                                }
                                if let Some(error) = parallel_error {
                                    let _ = reply.send(Err(error));
                                } else {
                                    let _ = reply.send(Ok(results));
                                }
                            }
                        }
                    }
                }
            }
        };
        // The program has no independent wall-clock deadline. Cancellation is
        // propagated by the watcher above, while each nested tool retains its
        // own transport/process policy. This avoids aborting productive
        // multi-tool programs solely because 60 seconds elapsed.
        program.await
    }

    async fn resolve_program_call(
        &self,
        call: &ProgramCall,
        program_call_id: &str,
        speculation_registry: &SpeculationRegistry,
        cancel: &CancellationToken,
    ) -> (
        std::result::Result<hi_workflow::ProgramToolResult, String>,
        hi_tools::ToolOutcome,
    ) {
        let args = serde_json::to_string(&call.arguments).unwrap_or_default();
        let key = SpeculationKey::new(
            format!("turn-{}", self.turn_count),
            program_call_id,
            call.occurrence,
            &call.name,
            &args,
            self.runtime.context_generation(),
            self.runtime.ledger().revision(),
            if matches!(
                hi_tools::speculation_class(&call.name),
                hi_tools::SpeculationClass::IdempotentExternal
            ) {
                external_freshness_epoch(self.config.program.external_ttl_seconds)
            } else {
                0
            },
        );
        let claimed = if self.config.program.speculation_enabled() {
            speculation_registry
                .claim_exact_cancelled(&key, Some(cancel))
                .await
        } else {
            None
        };
        match claimed {
            Some(Ok(value)) => {
                let status = match value.status.as_str() {
                    "succeeded" => hi_tools::ToolStatus::Succeeded,
                    "cancelled" => hi_tools::ToolStatus::Cancelled,
                    "denied" => hi_tools::ToolStatus::Denied,
                    _ => hi_tools::ToolStatus::Failed,
                };
                let output = synthetic_tool_outcome(value.output.clone(), status);
                (Ok(value), output)
            }
            Some(Err(error)) => {
                let output = synthetic_tool_outcome(error.clone(), hi_tools::ToolStatus::Failed);
                (Err(error), output)
            }
            None => {
                tokio::select! {
                    _ = cancel.cancelled() => (
                        Err("program cancelled".to_string()),
                        synthetic_tool_outcome(
                            "program cancelled".to_string(),
                            hi_tools::ToolStatus::Cancelled,
                        ),
                    ),
                    result = self.execute_program_tool(call) => result,
                }
            }
        }
    }

    /// Apply the same egress confirmation policy as ordinary tool batches.
    /// This runs only after the program has been selected as the real
    /// completion; shadow calls never reach this method.
    async fn authorize_program_call(
        &mut self,
        call: &ProgramCall,
        tool_specs: &[hi_ai::ToolSpec],
        tool_envelope: &hi_tools::envelope::ToolEnvelope,
        ui: &mut dyn Ui,
    ) -> Option<(
        std::result::Result<hi_workflow::ProgramToolResult, String>,
        hi_tools::ToolOutcome,
    )> {
        let arguments = serde_json::to_string(&call.arguments).unwrap_or_default();
        if !tool_envelope.digest_is_valid()
            || !tool_envelope.matches_program_specs(tool_specs)
            || !tool_envelope.admits_program(&call.name)
        {
            return Some(program_denied_result(
                call,
                format!(
                    "tool `{}` is outside the model request's sealed envelope {}",
                    call.name, tool_envelope.digest
                ),
            ));
        }
        if let Err(error) = hi_ai::validate_client_tool_call(
            &format!("program_{}", call.occurrence),
            &call.name,
            &arguments,
            tool_specs,
        ) {
            return Some(program_denied_result(call, error.to_string()));
        }
        if !egress_confirm_required(
            self.permission_mode,
            self.config.gates.confirm_edits,
            &call.name,
        ) {
            return None;
        }
        if self.approval_parked {
            return Some(program_denied_result(call, PARKED_TOOL_RESULT.to_string()));
        }
        let decision = ui
            .confirm(confirmation_for_egress_tool(&call.name, &arguments))
            .await;
        if decision == ConfirmationResult::Approved {
            return None;
        }
        if decision == ConfirmationResult::Parked {
            self.note_approval_parked(ui);
        } else if decision == ConfirmationResult::Unavailable {
            ui.status("confirmation required, but this frontend cannot answer it; rerun interactively or disable --confirm-edits");
        }
        let message = match decision {
            ConfirmationResult::Parked => PARKED_TOOL_RESULT.to_string(),
            ConfirmationResult::Unavailable => {
                "External tool call skipped because confirmation is unavailable.".to_string()
            }
            _ => "External tool call denied by confirmation.".to_string(),
        };
        Some(program_denied_result(call, message))
    }

    fn program_call_requires_settlement(&self, call: &ProgramCall) -> bool {
        if self.config.gates.dry_run {
            return false;
        }
        let arguments = serde_json::to_string(&call.arguments).unwrap_or_default();
        workspace_operation_requires_settlement(&call.name, &arguments)
    }

    async fn observe_program_effect(
        &self,
        call: &ProgramCall,
        mutates: bool,
        resolved: (
            std::result::Result<hi_workflow::ProgramToolResult, String>,
            hi_tools::ToolOutcome,
        ),
    ) -> (
        std::result::Result<hi_workflow::ProgramToolResult, String>,
        hi_tools::ToolOutcome,
    ) {
        if !mutates {
            return resolved;
        }
        if let Some(background) = &resolved.1.background
            && let Err(error) = self.observe_durable_background_process(background).await
        {
            return program_failed_result(
                call,
                format!(
                    "background process state could not be durably recorded: {error:#}; run /pipefs retry"
                ),
            );
        }
        resolved
    }

    async fn execute_program_tool(
        &self,
        call: &ProgramCall,
    ) -> (
        std::result::Result<hi_workflow::ProgramToolResult, String>,
        hi_tools::ToolOutcome,
    ) {
        self.program_tool_runner().execute(call).await
    }

    fn program_tool_runner(&self) -> ProgramToolRunner {
        ProgramToolRunner {
            root: self.runtime.root().to_path_buf(),
            state_root: self.runtime.state_root().to_path_buf(),
            process_runner: self.runtime.process_runner().clone(),
            lsp: self.runtime.lsp(),
            background: self.runtime.background_arc(),
            read_cache: self.runtime.read_cache_arc(),
            repo_map: self.runtime.repo_map_arc(),
            mcp: self.mcp.clone(),
            memory: self.memory.clone(),
        }
    }
}

fn program_denied_result(
    call: &ProgramCall,
    message: String,
) -> (
    std::result::Result<hi_workflow::ProgramToolResult, String>,
    hi_tools::ToolOutcome,
) {
    let output = synthetic_tool_outcome(message.clone(), hi_tools::ToolStatus::Denied);
    (
        Ok(hi_workflow::ProgramToolResult {
            index: call.occurrence,
            name: call.name.clone(),
            status: "denied".into(),
            output: message,
        }),
        output,
    )
}

fn program_failed_result(
    call: &ProgramCall,
    message: String,
) -> (
    std::result::Result<hi_workflow::ProgramToolResult, String>,
    hi_tools::ToolOutcome,
) {
    let output = synthetic_tool_outcome(message.clone(), hi_tools::ToolStatus::Failed);
    (Err(format!("{}: {message}", call.name)), output)
}

fn external_freshness_epoch(ttl_seconds: u64) -> u64 {
    let ttl_seconds = ttl_seconds.max(1);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() / ttl_seconds)
        .unwrap_or_default()
}

#[cfg(test)]
mod scheduler_count_tests;
