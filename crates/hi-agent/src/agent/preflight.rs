//! Preflight inspection runs executed before the main turn loop: read-only
//! review preflight (directory listing + targeted grep + extra reads) and
//! implementation preflight (entrypoint detection + validation command).

use futures_util::StreamExt;
use hi_ai::Content;
use hi_tools::execute_in_runtime;

use crate::heuristics::emit_tool_output;
use crate::steering::{
    DEFAULT_PREFLIGHT_EXTRA_READ_LIMIT, EvidenceKind, EvidenceTracker, ImplementationTracker,
    PreflightCall, READ_ONLY_PREFLIGHT_MAX_EXTRA_READS, ReviewIntent,
    SECURITY_PREFLIGHT_EXTRA_READ_LIMIT, compact_preflight_tool_output, evidence_kind_for_tool,
    implementation_preflight_command, inspection_signature, paths_from_grep_output,
    preferred_validation_from_preflight, preflight_path_relevant_for_intent,
    read_only_preflight_initial_calls,
};
use crate::{ToolCallEntry, Ui};

struct PreflightExecution {
    call: PreflightCall,
    id: String,
    output: hi_tools::ToolOutcome,
    duration_ms: u64,
    path: String,
    error: bool,
}

#[derive(Clone, Copy)]
struct PreflightRuntime<'a> {
    root: &'a std::path::Path,
    state_root: &'a std::path::Path,
    lsp: &'a std::sync::Arc<hi_lsp::LspManager>,
    background: &'a hi_tools::BackgroundRegistry,
    read_cache: &'a std::sync::Mutex<hi_tools::ReadCache>,
    repo_map: &'a std::sync::Mutex<hi_tools::RepoMapCache>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PreflightSummary {
    pub(crate) executed: u32,
    pub(crate) max_concurrent_batch: u32,
    pub(crate) serial_runs: u32,
}

async fn execute_preflight_batch(
    runtime: PreflightRuntime<'_>,
    calls: Vec<PreflightCall>,
    id_prefix: &str,
    start_index: u32,
    max_parallel: usize,
    ui: &mut dyn Ui,
) -> Vec<PreflightExecution> {
    if calls.is_empty() {
        return Vec::new();
    }
    for call in &calls {
        ui.tool_started(call.name, &call.arguments);
        ui.tool_call(call.name, &call.arguments);
    }
    let root = runtime.root.to_path_buf();
    let state_root = runtime.state_root.to_path_buf();
    futures_util::stream::iter(calls.into_iter().enumerate().map(|(offset, call)| {
        let root = root.clone();
        let state_root = state_root.clone();
        let lsp = runtime.lsp.clone();
        let id = format!("{id_prefix}_{}", start_index.saturating_add(offset as u32));
        async move {
            let started = std::time::Instant::now();
            let output = execute_in_runtime(
                &root,
                &state_root,
                &lsp,
                runtime.background,
                runtime.read_cache,
                runtime.repo_map,
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
    .collect()
    .await
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

fn call_counts_against_inspection_cap(call: &PreflightCall) -> bool {
    matches!(
        evidence_kind_for_tool(call.name, &call.arguments),
        Some(EvidenceKind::FileRead | EvidenceKind::TargetedSearch)
    )
}

fn cap_preflight_calls(calls: Vec<PreflightCall>, inspection_cap: u32) -> Vec<PreflightCall> {
    let mut used = 0u32;
    calls
        .into_iter()
        .filter(|call| {
            if !call_counts_against_inspection_cap(call) {
                return true;
            }
            if used >= inspection_cap {
                return false;
            }
            used = used.saturating_add(1);
            true
        })
        .collect()
}

impl crate::Agent {
    pub(crate) async fn run_read_only_preflight(
        &mut self,
        intent: ReviewIntent,
        inspection_cap: u32,
        ui: &mut dyn Ui,
        evidence: &mut EvidenceTracker,
        tool_timeline: &mut Vec<ToolCallEntry>,
        tool_budget: u32,
    ) -> PreflightSummary {
        let calls = cap_preflight_calls(read_only_preflight_initial_calls(intent), inspection_cap)
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
                repo_map: self.runtime.repo_map(),
            },
            calls,
            &id_prefix,
            executed,
            self.config.loop_limits.max_parallel_tools,
            ui,
        )
        .await;
        record_preflight_batch(
            &mut summary,
            initial_batch_len,
            self.config.loop_limits.max_parallel_tools,
        );
        for result in initial_results {
            evidence.record_success(
                result.call.name,
                &result.call.arguments,
                &result.output.content,
            );
            tool_timeline.push(ToolCallEntry {
                tool: result.call.name.to_string(),
                path: result.path,
                duration_ms: result.duration_ms,
                status: result.output.status,
                background: result.output.background.clone(),
                process: result.output.process.clone(),
                effects: result.output.effects.clone(),
                truncation: result.output.truncation.clone(),
                error: result.error,
                progress_kind: "meaningful".to_string(),
                progress_reason: "preflight inspection evidence".to_string(),
                normalized_signature: inspection_signature(
                    result.call.name,
                    &result.call.arguments,
                ),
            });
            if result.call.name == "grep" {
                let remaining_extra_reads =
                    inspection_cap.saturating_sub(evidence.inspection_attempt_count()) as usize;
                for path in paths_from_grep_output(&result.output.content) {
                    if !preflight_path_relevant_for_intent(intent, &path)
                        || seen_read_paths.iter().any(|existing| existing == &path)
                        || extra_reads.iter().any(|existing| existing == &path)
                        || extra_reads.len() >= READ_ONLY_PREFLIGHT_MAX_EXTRA_READS
                        || extra_reads.len() >= remaining_extra_reads
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
            emit_tool_output(ui, result.call.name, &display_output);
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
                repo_map: self.runtime.repo_map(),
            },
            extra_calls,
            &id_prefix,
            executed,
            self.config.loop_limits.max_parallel_tools,
            ui,
        )
        .await;
        record_preflight_batch(
            &mut summary,
            extra_batch_len,
            self.config.loop_limits.max_parallel_tools,
        );
        for result in extra_results {
            evidence.record_success(
                result.call.name,
                &result.call.arguments,
                &result.output.content,
            );
            tool_timeline.push(ToolCallEntry {
                tool: result.call.name.to_string(),
                path: result.path,
                duration_ms: result.duration_ms,
                status: result.output.status,
                background: result.output.background.clone(),
                process: result.output.process.clone(),
                effects: result.output.effects.clone(),
                truncation: result.output.truncation.clone(),
                error: result.error,
                progress_kind: "meaningful".to_string(),
                progress_reason: "preflight inspection evidence".to_string(),
                normalized_signature: inspection_signature(
                    result.call.name,
                    &result.call.arguments,
                ),
            });
            let compacted_output =
                compact_preflight_tool_output(result.call.name, &result.output.content);
            let mut display_output = result.output.clone();
            display_output.content = compacted_output.clone();
            display_output.display = None;
            emit_tool_output(ui, result.call.name, &display_output);
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
        }
        debug_assert_eq!(summary.executed, executed);
        summary
    }

    pub(crate) async fn run_implementation_preflight(
        &mut self,
        ui: &mut dyn Ui,
        tracker: &mut ImplementationTracker,
        tool_timeline: &mut Vec<ToolCallEntry>,
    ) -> u32 {
        let arguments = serde_json::json!({
            "command": implementation_preflight_command(),
            "timeout": 120,
        })
        .to_string();
        let id = format!("hi_implementation_preflight_{}", self.messages.len());
        ui.status("running implementation preflight inspection");
        ui.tool_started("bash", &arguments);
        ui.tool_call("bash", &arguments);
        let started = std::time::Instant::now();
        let lsp = self.runtime.lsp();
        let output = execute_in_runtime(
            self.runtime.root(),
            self.runtime.state_root(),
            &lsp,
            self.runtime.background(),
            self.runtime.read_cache(),
            self.runtime.repo_map(),
            "bash",
            &arguments,
        )
        .await;
        let duration_ms = started.elapsed().as_millis() as u64;
        let error = output.status != hi_tools::ToolStatus::Succeeded;
        tracker.preferred_validation = preferred_validation_from_preflight(&output.content);
        tool_timeline.push(ToolCallEntry {
            tool: "bash".to_string(),
            path: String::new(),
            duration_ms,
            status: output.status,
            background: output.background.clone(),
            process: output.process.clone(),
            effects: output.effects.clone(),
            truncation: output.truncation.clone(),
            error,
            progress_kind: "weak".to_string(),
            progress_reason: "implementation preflight inspection".to_string(),
            normalized_signature: None,
        });
        emit_tool_output(ui, "bash", &output);
        self.messages.push_assistant_with_results(
            vec![Content::ToolCall {
                id: id.clone(),
                name: "bash".to_string(),
                arguments,
            }],
            vec![(id, output.content)],
        );
        1
    }
}
