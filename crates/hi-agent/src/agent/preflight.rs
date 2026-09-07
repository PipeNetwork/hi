//! Preflight inspection runs executed before the main turn loop: read-only
//! review preflight (directory listing + targeted grep + extra reads) and
//! implementation preflight (entrypoint detection + validation command).

use anyhow::{Context, Result};
use futures_util::StreamExt;
use hi_ai::Content;
use hi_tools::execute_in_runtime_shared;

use crate::heuristics::emit_tool_output;
use crate::steering::{
    DEFAULT_PREFLIGHT_EXTRA_READ_LIMIT, EvidenceTracker, ImplementationTracker, PreflightCall,
    ReviewIntent, SECURITY_PREFLIGHT_EXTRA_READ_LIMIT, compact_preflight_tool_output,
    implementation_preflight_command, inspection_signature, paths_from_grep_output_in,
    preferred_validation_from_preflight, preflight_path_relevant_for_intent,
    read_only_preflight_initial_calls_for_prompt,
};
use crate::transcript::NudgeKind;
use crate::{ToolCallEntry, Ui};

const PREFLIGHT_INTERRUPTED_NUDGE: &str = "The user skipped the preflight inspection, not the overall task. Continue the original task now with an appropriate tool. Do not stop merely to acknowledge the interruption, and do not retry the same preflight command.";
const PREFLIGHT_CITATION_NUDGE: &str = "The files above are already in this transcript. Cite them by path in your findings. Call a tool only for a file that is not already shown.";

fn cancelled_preflight_outcome() -> hi_tools::ToolOutcome {
    hi_tools::ToolOutcome {
        content: "Preflight tool interrupted by user.".to_string(),
        display: None,
        plan: None,
        status: hi_tools::ToolStatus::Cancelled,
        process: None,
        background: None,
        effects: hi_tools::ToolEffects::default(),
        truncation: hi_tools::TruncationState::Complete,
        images: Vec::new(),
    }
}

fn preflight_progress(output: &hi_tools::ToolOutcome) -> (&'static str, &'static str) {
    match output.status {
        hi_tools::ToolStatus::Cancelled => ("none", "preflight interrupted by user"),
        hi_tools::ToolStatus::Succeeded => ("meaningful", "preflight inspection evidence"),
        _ => ("weak", "preflight inspection failed"),
    }
}

fn implementation_preflight_intent() -> hi_workspace::MutationIntent {
    // This is a fixed harness command, not model-authored shell. Its Git
    // invocations explicitly disable index refresh, fsmonitor, external diff,
    // and textconv, and the remaining commands only inspect local paths. Keep
    // the effect scope conservative so an unexpected byte change is archived,
    // but do not misclassify the inspection as a non-replayable external
    // effect (which would unnecessarily disable protocol-1 PipeFS).
    hi_workspace::MutationIntent {
        effect_scope: hi_workspace::EffectScope::LiveWriter,
        replay_class: hi_workspace::ReplayClass::PureWorkspace,
        dirty_paths: None,
        description: Some("implementation preflight shell inspection".into()),
    }
}

fn implementation_preflight_report(
    output: &hi_tools::ToolOutcome,
    effects_known: bool,
) -> hi_workspace::ExecutionReport {
    let disposition = if !effects_known {
        hi_workspace::ExecutionDisposition::Indeterminate
    } else {
        match output.status {
            hi_tools::ToolStatus::Succeeded => hi_workspace::ExecutionDisposition::Succeeded,
            hi_tools::ToolStatus::Cancelled => hi_workspace::ExecutionDisposition::Cancelled,
            _ => hi_workspace::ExecutionDisposition::Failed,
        }
    };
    let mut changed_paths = output
        .effects
        .file_changes
        .iter()
        .map(|change| change.path.clone().into())
        .collect::<Vec<std::path::PathBuf>>();
    changed_paths.sort();
    changed_paths.dedup();
    hi_workspace::ExecutionReport {
        disposition,
        workspace_may_have_changed: !effects_known
            || !changed_paths.is_empty()
            || output.effects.mutation_applied,
        // The fixed command has no network, credential, or authority-bearing
        // side effect. Its process existence alone is not an external effect.
        external_effect_may_have_occurred: false,
        content_digest: None,
        changed_paths,
        artifacts: Vec::new(),
        detail: if effects_known {
            (output.status != hi_tools::ToolStatus::Succeeded)
                .then(|| format!("implementation preflight was {:?}", output.status))
        } else {
            Some("implementation preflight effects could not be reconciled".into())
        },
    }
}

fn implementation_preflight_publication_failure(
    output: &hi_tools::ToolOutcome,
    detail: impl std::fmt::Display,
) -> hi_tools::ToolOutcome {
    // Put the settlement failure first so bounding a large preflight result
    // cannot hide the reason the visible tool lifecycle failed. Keep the
    // process/effect evidence, but never reuse a rich successful display for
    // this terminal event.
    let combined = if output.content.is_empty() {
        format!("Error: {detail}")
    } else {
        format!("Error: {detail}\n\n{}", output.content)
    };
    let (content, truncation) = hi_tools::bound_tool_content(combined);
    let mut terminal = output.clone();
    terminal.content = content;
    terminal.display = None;
    terminal.status = hi_tools::ToolStatus::Failed;
    terminal.truncation = truncation;
    terminal
}

async fn execute_implementation_preflight_process(
    runner: &hi_tools::ProcessRunner,
    command: &str,
    interrupt: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> (Option<anyhow::Result<hi_tools::ProcessExecution>>, bool) {
    execute_implementation_preflight_process_after_check(runner, command, interrupt, || {}).await
}

async fn execute_implementation_preflight_process_after_check<F>(
    runner: &hi_tools::ProcessRunner,
    command: &str,
    interrupt: std::sync::Arc<std::sync::atomic::AtomicBool>,
    after_prestart_check: F,
) -> (Option<anyhow::Result<hi_tools::ProcessExecution>>, bool)
where
    F: FnOnce(),
{
    // An interrupt raised synchronously by the UI's tool-start callback means
    // the command never starts. This avoids killing an empty registry and then
    // accidentally launching the command while manufacturing Cancelled.
    if interrupt.swap(false, std::sync::atomic::Ordering::AcqRel) {
        return (None, true);
    }
    after_prestart_check();
    let execution =
        runner.run_shell_maybe_timeout(command, Some(std::time::Duration::from_secs(120)));
    tokio::pin!(execution);
    // Poll once before the cancellation select. `run_shell_maybe_timeout`
    // spawns and registers its child on that first poll, closing the race where
    // an interrupt could win, observe an empty registry, and then awaiting the
    // untouched future would launch the supposedly cancelled command.
    let completed = std::future::poll_fn(|context| {
        std::task::Poll::Ready(
            match std::future::Future::poll(execution.as_mut(), context) {
                std::task::Poll::Ready(output) => Some(output),
                std::task::Poll::Pending => None,
            },
        )
    })
    .await;
    if let Some(output) = completed {
        let interrupted = interrupt.swap(false, std::sync::atomic::Ordering::AcqRel);
        return (Some(output), interrupted);
    }
    tokio::select! {
        biased;
        _ = take_tool_interrupt(interrupt.clone()) => {
            // Keep the capture future alive after signalling the group. It is
            // the owner that observes exit, drains the pipes, reaps the direct
            // child, and unregisters it from the foreground inventory.
            runner.foreground_registry().kill_current();
            (Some(execution.await), true)
        }
        output = &mut execution => {
            interrupt.store(false, std::sync::atomic::Ordering::Release);
            (Some(output), false)
        }
    }
}

fn implementation_preflight_outcome(
    process: Option<anyhow::Result<hi_tools::ProcessExecution>>,
    interrupted: bool,
) -> hi_tools::ToolOutcome {
    if process.is_none() {
        return cancelled_preflight_outcome();
    }
    let mut output = match process.expect("checked above") {
        Ok(execution) => {
            let display = execution.display_content();
            let model = execution.model_content();
            let process_outcome = execution.model_outcome();
            let status = execution.status;
            let process_truncation = execution.truncation;
            let (content, boundary_truncation) = hi_tools::bound_tool_content(model);
            hi_tools::ToolOutcome {
                display: (display != content).then_some(display),
                content,
                plan: None,
                status,
                process: Some(process_outcome),
                background: None,
                effects: hi_tools::ToolEffects::default(),
                truncation: if matches!(process_truncation, hi_tools::TruncationState::Complete) {
                    boundary_truncation
                } else {
                    process_truncation
                },
                images: Vec::new(),
            }
        }
        Err(error) => {
            let (content, truncation) = hi_tools::bound_tool_content(format!(
                "Error: implementation preflight process failed: {error:#}"
            ));
            hi_tools::ToolOutcome {
                content,
                display: None,
                plan: None,
                status: hi_tools::ToolStatus::Failed,
                process: None,
                background: None,
                effects: hi_tools::ToolEffects::default(),
                truncation,
                images: Vec::new(),
            }
        }
    };
    if interrupted {
        output.content = "Preflight tool interrupted by user.".into();
        output.display = None;
        output.status = hi_tools::ToolStatus::Cancelled;
    }
    output
}

async fn take_tool_interrupt(interrupt: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    loop {
        if interrupt.swap(false, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

struct PreflightExecution {
    call: PreflightCall,
    id: String,
    output: hi_tools::ToolOutcome,
    duration_ms: u64,
    path: String,
    error: bool,
}

#[derive(Clone)]
struct PreflightRuntime<'a> {
    root: &'a std::path::Path,
    state_root: &'a std::path::Path,
    lsp: &'a std::sync::Arc<hi_lsp::LspManager>,
    background: &'a hi_tools::BackgroundRegistry,
    read_cache: &'a std::sync::Mutex<hi_tools::ReadCache>,
    repo_map: std::sync::Arc<std::sync::Mutex<hi_tools::RepoMapCache>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PreflightSummary {
    pub(crate) executed: u32,
    pub(crate) max_concurrent_batch: u32,
    pub(crate) serial_runs: u32,
    pub(crate) interrupted: bool,
}

async fn execute_preflight_batch(
    runtime: PreflightRuntime<'_>,
    calls: Vec<PreflightCall>,
    id_prefix: &str,
    start_index: u32,
    max_parallel: usize,
    interrupt: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ui: &mut dyn Ui,
) -> Vec<PreflightExecution> {
    if calls.is_empty() {
        return Vec::new();
    }
    // An interrupt belongs only to a tool that was visibly active after this
    // point. Discarding an older signal here prevents Esc during a previous
    // turn/tool from cancelling this preflight before it has even started.
    interrupt.store(false, std::sync::atomic::Ordering::Relaxed);
    for (offset, call) in calls.iter().enumerate() {
        let id = format!("{id_prefix}_{}", start_index.saturating_add(offset as u32));
        ui.tool_started_id(&id, call.name, &call.arguments);
        ui.tool_call_id(&id, call.name, &call.arguments);
    }
    let root = runtime.root.to_path_buf();
    let state_root = runtime.state_root.to_path_buf();
    let cancelled_calls = calls.clone();
    let executions =
        futures_util::stream::iter(calls.into_iter().enumerate().map(|(offset, call)| {
            let root = root.clone();
            let state_root = state_root.clone();
            let lsp = runtime.lsp.clone();
            let repo_map = runtime.repo_map.clone();
            let id = format!("{id_prefix}_{}", start_index.saturating_add(offset as u32));
            async move {
                let started = std::time::Instant::now();
                let output = execute_in_runtime_shared(
                    &root,
                    &state_root,
                    &lsp,
                    runtime.background,
                    runtime.read_cache,
                    &repo_map,
                    call.name,
                    &call.arguments,
                )
                .await;
                let duration_ms = started.elapsed().as_millis() as u64;
                let path = hi_tools::target_path(call.name, &call.arguments).unwrap_or_default();
                let error = output.status != hi_tools::ToolStatus::Succeeded;
                PreflightExecution {
                    call,
                    id,
                    output,
                    duration_ms,
                    path,
                    error,
                }
            }
        }))
        .buffered(max_parallel.max(1))
        .collect::<Vec<_>>();
    tokio::pin!(executions);
    tokio::select! {
        biased;
        _ = take_tool_interrupt(interrupt.clone()) => {
            ui.status("preflight inspection interrupted — continuing the task");
            cancelled_calls
                .into_iter()
                .enumerate()
                .map(|(offset, call)| PreflightExecution {
                    id: format!("{id_prefix}_{}", start_index.saturating_add(offset as u32)),
                    call,
                    output: cancelled_preflight_outcome(),
                    duration_ms: 0,
                    path: String::new(),
                    error: true,
                })
                .collect()
        }
        results = &mut executions => {
            // Do not let an Esc that raced with the final process exit poison
            // the model's next tool call.
            interrupt.store(false, std::sync::atomic::Ordering::Relaxed);
            results
        }
    }
}

fn record_preflight_batch(summary: &mut PreflightSummary, batch_len: usize, max_parallel: usize) {
    if batch_len == 0 {
        return;
    }
    let actual_concurrency = batch_len.min(max_parallel.max(1)) as u32;
    summary.executed = summary.executed.saturating_add(batch_len as u32);
    summary.max_concurrent_batch = summary.max_concurrent_batch.max(actual_concurrency);
    if actual_concurrency == 1 {
        summary.serial_runs = summary.serial_runs.saturating_add(batch_len as u32);
    }
}

impl crate::Agent {
    #[allow(clippy::too_many_arguments)] // preflight threads each turn-scoped dependency explicitly
    pub(crate) async fn run_read_only_preflight(
        &mut self,
        intent: ReviewIntent,
        prompt: &str,
        ui: &mut dyn Ui,
        evidence: &mut EvidenceTracker,
        tool_timeline: &mut crate::agent::turn::retention::ToolTimeline,
        tool_budget: u32,
    ) -> PreflightSummary {
        let calls =
            read_only_preflight_initial_calls_for_prompt(self.runtime.root(), intent, prompt)
                .into_iter()
                .take(tool_budget as usize)
                .collect::<Vec<_>>();
        if calls.is_empty() {
            return PreflightSummary::default();
        }

        ui.status("running read-only preflight inspection");
        let mut summary = PreflightSummary::default();
        let mut content = Vec::new();
        let mut results = Vec::new();
        let mut executed = 0u32;
        let mut extra_reads = Vec::<String>::new();
        let mut seen_read_paths = calls
            .iter()
            .filter(|call| call.name == "read")
            .filter_map(|call| hi_tools::target_path(call.name, &call.arguments))
            .collect::<Vec<_>>();
        let id_prefix = format!("hi_preflight_{}", self.messages.len());

        let initial_batch_len = calls.len();
        let initial_lsp = self.runtime.lsp();
        let initial_results = execute_preflight_batch(
            PreflightRuntime {
                root: self.runtime.root(),
                state_root: self.runtime.state_root(),
                lsp: &initial_lsp,
                background: self.runtime.background(),
                read_cache: self.runtime.read_cache(),
                repo_map: self.runtime.repo_map_arc(),
            },
            calls,
            &id_prefix,
            executed,
            self.config.loop_limits.max_parallel_tools,
            self.interrupt.clone(),
            ui,
        )
        .await;
        record_preflight_batch(
            &mut summary,
            initial_batch_len,
            self.config.loop_limits.max_parallel_tools,
        );
        for result in initial_results {
            summary.interrupted |= result.output.status == hi_tools::ToolStatus::Cancelled;
            if result.output.status == hi_tools::ToolStatus::Succeeded {
                evidence.record_success(
                    result.call.name,
                    &result.call.arguments,
                    &result.output.content,
                );
            }
            let (progress_kind, progress_reason) = preflight_progress(&result.output);
            tool_timeline.push(
                ToolCallEntry {
                    tool: result.call.name.to_string(),
                    path: result.path,
                    duration_ms: result.duration_ms,
                    queue_delay_ms: 0,
                    completion_index: 0,
                    status: result.output.status,
                    background: result.output.background.clone(),
                    process: result.output.process.clone(),
                    effects: result.output.effects.clone(),
                    truncation: result.output.truncation.clone(),
                    error: result.error,
                    progress_kind: progress_kind.to_string(),
                    progress_reason: progress_reason.to_string(),
                    normalized_signature: inspection_signature(
                        result.call.name,
                        &result.call.arguments,
                    ),
                    command: None,
                    arg_chars: 0,
                    result_chars: 0,
                    truncated: false,
                    kind: String::new(),
                }
                .with_tape(&result.call.arguments, &result.output.content),
            );
            if result.call.name == "grep" {
                for path in paths_from_grep_output_in(self.runtime.root(), &result.output.content) {
                    if !preflight_path_relevant_for_intent(intent, &path)
                        || seen_read_paths.iter().any(|existing| existing == &path)
                        || extra_reads.iter().any(|existing| existing == &path)
                    {
                        continue;
                    }
                    extra_reads.push(path.clone());
                    seen_read_paths.push(path.clone());
                }
            }
            let compacted_output =
                compact_preflight_tool_output(result.call.name, &result.output.content);
            let mut display_output = result.output.clone();
            display_output.content = compacted_output.clone();
            display_output.display = None;
            emit_tool_output(ui, &result.id, result.call.name, &display_output);
            content.push(Content::ToolCall {
                id: result.id.clone(),
                name: result.call.name.to_string(),
                arguments: result.call.arguments,
            });
            results.push((result.id, compacted_output));
            executed = executed.saturating_add(1);
        }

        let extra_calls = extra_reads
            .into_iter()
            .map(|path| {
                let limit = if matches!(intent, ReviewIntent::Security) {
                    SECURITY_PREFLIGHT_EXTRA_READ_LIMIT
                } else {
                    DEFAULT_PREFLIGHT_EXTRA_READ_LIMIT
                };
                PreflightCall::read(path, limit)
            })
            .take(tool_budget.saturating_sub(executed) as usize)
            .collect::<Vec<_>>();
        let extra_batch_len = extra_calls.len();
        let extra_lsp = self.runtime.lsp();
        let extra_results = execute_preflight_batch(
            PreflightRuntime {
                root: self.runtime.root(),
                state_root: self.runtime.state_root(),
                lsp: &extra_lsp,
                background: self.runtime.background(),
                read_cache: self.runtime.read_cache(),
                repo_map: self.runtime.repo_map_arc(),
            },
            extra_calls,
            &id_prefix,
            executed,
            self.config.loop_limits.max_parallel_tools,
            self.interrupt.clone(),
            ui,
        )
        .await;
        record_preflight_batch(
            &mut summary,
            extra_batch_len,
            self.config.loop_limits.max_parallel_tools,
        );
        for result in extra_results {
            summary.interrupted |= result.output.status == hi_tools::ToolStatus::Cancelled;
            if result.output.status == hi_tools::ToolStatus::Succeeded {
                evidence.record_success(
                    result.call.name,
                    &result.call.arguments,
                    &result.output.content,
                );
            }
            let (progress_kind, progress_reason) = preflight_progress(&result.output);
            tool_timeline.push(
                ToolCallEntry {
                    tool: result.call.name.to_string(),
                    path: result.path,
                    duration_ms: result.duration_ms,
                    queue_delay_ms: 0,
                    completion_index: 0,
                    status: result.output.status,
                    background: result.output.background.clone(),
                    process: result.output.process.clone(),
                    effects: result.output.effects.clone(),
                    truncation: result.output.truncation.clone(),
                    error: result.error,
                    progress_kind: progress_kind.to_string(),
                    progress_reason: progress_reason.to_string(),
                    normalized_signature: inspection_signature(
                        result.call.name,
                        &result.call.arguments,
                    ),
                    command: None,
                    arg_chars: 0,
                    result_chars: 0,
                    truncated: false,
                    kind: String::new(),
                }
                .with_tape(&result.call.arguments, &result.output.content),
            );
            let compacted_output =
                compact_preflight_tool_output(result.call.name, &result.output.content);
            let mut display_output = result.output.clone();
            display_output.content = compacted_output.clone();
            display_output.display = None;
            emit_tool_output(ui, &result.id, result.call.name, &display_output);
            content.push(Content::ToolCall {
                id: result.id.clone(),
                name: result.call.name.to_string(),
                arguments: result.call.arguments,
            });
            results.push((result.id, compacted_output));
            executed = executed.saturating_add(1);
        }

        if !content.is_empty() {
            self.messages.push_assistant_with_results(content, results);
            self.messages
                .push_nudge(NudgeKind::Continue, PREFLIGHT_CITATION_NUDGE);
        }
        if summary.interrupted {
            self.messages
                .push_nudge(NudgeKind::Continue, PREFLIGHT_INTERRUPTED_NUDGE);
        }
        debug_assert_eq!(summary.executed, executed);
        summary
    }

    pub(crate) async fn run_implementation_preflight(
        &mut self,
        ui: &mut dyn Ui,
        tracker: &mut ImplementationTracker,
        tool_timeline: &mut crate::agent::turn::retention::ToolTimeline,
    ) -> Result<PreflightSummary> {
        let arguments = serde_json::json!({
            "command": implementation_preflight_command(),
            "timeout": 120,
        })
        .to_string();
        let intent = implementation_preflight_intent();
        self.begin_classified_workspace_operation(intent).await?;
        let id = format!("hi_implementation_preflight_{}", self.messages.len());
        ui.status("running implementation preflight inspection");
        self.interrupt
            .store(false, std::sync::atomic::Ordering::Relaxed);
        ui.tool_started_id(&id, "bash", &arguments);
        ui.tool_call_id(&id, "bash", &arguments);
        let started = std::time::Instant::now();
        let ledger_revision = self.runtime.ledger().revision();
        let (process, interrupted) = execute_implementation_preflight_process(
            self.runtime.process_runner(),
            implementation_preflight_command(),
            self.interrupt.clone(),
        )
        .await;
        if interrupted {
            ui.status("implementation preflight interrupted — continuing the task");
        }
        let process_started = process.is_some();
        let mut output = implementation_preflight_outcome(process, interrupted);
        let effects_known = if process_started {
            match self.reconcile_workspace_changes().await {
                Ok(()) => {
                    let file_changes = self.runtime.ledger().changes_since(ledger_revision);
                    if !file_changes.is_empty() && output.status == hi_tools::ToolStatus::Succeeded
                    {
                        output.status = hi_tools::ToolStatus::Failed;
                        let detail = format!(
                            "[implementation preflight unexpectedly changed {} workspace path(s)]",
                            file_changes.len()
                        );
                        let (content, truncation) =
                            hi_tools::bound_tool_content(format!("{}\n{detail}", output.content));
                        output.content = content;
                        output.truncation = truncation;
                    }
                    output.effects = hi_tools::ToolEffects {
                        mutation_attempted: !file_changes.is_empty(),
                        mutation_applied: !file_changes.is_empty(),
                        file_changes,
                    };
                    true
                }
                Err(error) => {
                    output.effects = hi_tools::ToolEffects {
                        mutation_attempted: true,
                        mutation_applied: true,
                        file_changes: Vec::new(),
                    };
                    if output.status == hi_tools::ToolStatus::Succeeded {
                        output.status = hi_tools::ToolStatus::Failed;
                    }
                    let detail = format!(
                        "[infrastructure failure: implementation preflight effects could not be reconciled: {error:#}]"
                    );
                    let combined = if output.content.is_empty() {
                        detail
                    } else {
                        format!("{}\n{detail}", output.content)
                    };
                    let (content, truncation) = hi_tools::bound_tool_content(combined);
                    output.content = content;
                    output.truncation = truncation;
                    false
                }
            }
        } else {
            true
        };
        let execution = implementation_preflight_report(&output, effects_known);
        let calls = vec![(id.clone(), "bash".to_string(), arguments.clone())];
        let assistant_content = vec![Content::ToolCall {
            id: id.clone(),
            name: "bash".to_string(),
            arguments: arguments.clone(),
        }];
        let results = vec![(id.clone(), output.content.clone())];
        if let Err(stage_error) = self
            .stage_active_workspace_execution(&calls, &assistant_content, &results, &execution)
            .await
        {
            let mut indeterminate = execution;
            indeterminate.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            indeterminate.detail = Some(format!(
                "preflight ran, but its transcript could not be staged: {stage_error:#}"
            ));
            let settlement = self
                .checkpoint_durable_workspace_with_execution(indeterminate)
                .await;
            let terminal = implementation_preflight_publication_failure(
                &output,
                format!(
                    "implementation preflight ran, but its transcript could not be staged; execution is indeterminate: {stage_error:#}"
                ),
            );
            emit_tool_output(ui, &id, "bash", &terminal);
            return match settlement {
                Err(settlement) => Err(settlement).context(format!(
                    "preflight transcript staging failed and recovery settlement also failed: {stage_error:#}"
                )),
                Ok(()) => Err(stage_error)
                    .context("preflight transcript staging failed; execution is indeterminate"),
            };
        }
        if let Err(settlement) = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await
        {
            let terminal = implementation_preflight_publication_failure(
                &output,
                format!(
                    "implementation preflight ran, but workspace/transcript settlement is indeterminate: {settlement:#}"
                ),
            );
            emit_tool_output(ui, &id, "bash", &terminal);
            return Err(settlement).context("implementation preflight settlement failed");
        }
        let duration_ms = started.elapsed().as_millis() as u64;
        let error = output.status != hi_tools::ToolStatus::Succeeded;
        tracker.preferred_validation = preferred_validation_from_preflight(&output.content);
        let (progress_kind, progress_reason) = preflight_progress(&output);
        tool_timeline.push(
            ToolCallEntry {
                tool: "bash".to_string(),
                path: String::new(),
                duration_ms,
                queue_delay_ms: 0,
                completion_index: 0,
                status: output.status,
                background: output.background.clone(),
                process: output.process.clone(),
                effects: output.effects.clone(),
                truncation: output.truncation.clone(),
                error,
                progress_kind: progress_kind.to_string(),
                progress_reason: progress_reason.to_string(),
                normalized_signature: None,
                command: crate::steering::bash_command(&arguments),
                arg_chars: 0,
                result_chars: 0,
                truncated: false,
                kind: String::new(),
            }
            .with_tape(&arguments, &output.content),
        );
        emit_tool_output(ui, &id, "bash", &output);
        self.messages
            .push_assistant_with_results(assistant_content, results);
        let interrupted = output.status == hi_tools::ToolStatus::Cancelled;
        if interrupted {
            self.messages
                .push_nudge(NudgeKind::Continue, PREFLIGHT_INTERRUPTED_NUDGE);
        }
        Ok(PreflightSummary {
            executed: 1,
            max_concurrent_batch: 1,
            serial_runs: 1,
            interrupted,
        })
    }
}

#[cfg(test)]
#[path = "preflight_boundary_tests.rs"]
mod boundary_tests;
