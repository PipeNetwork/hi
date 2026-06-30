//! Preflight inspection runs executed before the main turn loop: read-only
//! review preflight (directory listing + targeted grep + extra reads) and
//! implementation preflight (entrypoint detection + optional GPU estimator
//! bootstrap + validation command).

use hi_ai::Content;
use hi_tools::{execute, execute_streaming};

use crate::steering::*;
use crate::heuristics::emit_tool_output;
use crate::{ToolCallEntry, Ui};

impl crate::Agent {
    pub(crate) async fn run_read_only_preflight(
        &mut self,
        intent: ReviewIntent,
        ui: &mut dyn Ui,
        evidence: &mut EvidenceTracker,
        tool_timeline: &mut Vec<ToolCallEntry>,
    ) -> u32 {
        let mut calls = read_only_preflight_initial_calls(intent);
        if calls.is_empty() {
            return 0;
        }

        ui.status("running read-only preflight inspection");
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

        while let Some(call) = calls.first().cloned() {
            calls.remove(0);
            let id = format!("{id_prefix}_{executed}");
            ui.tool_started(call.name, &call.arguments);
            ui.tool_call(call.name, &call.arguments);
            let started = std::time::Instant::now();
            let output = execute(call.name, &call.arguments).await;
            let duration_ms = started.elapsed().as_millis() as u64;
            let path = hi_tools::target_path(call.name, &call.arguments).unwrap_or_default();
            let error = output.content.starts_with("Error:");
            evidence.record_success(call.name, &call.arguments, &output.content);
            tool_timeline.push(ToolCallEntry {
                tool: call.name.to_string(),
                path,
                duration_ms,
                error,
            });
            if call.name == "grep" {
                for path in paths_from_grep_output(&output.content) {
                    if !preflight_path_relevant_for_intent(intent, &path)
                        || seen_read_paths.iter().any(|existing| existing == &path)
                        || extra_reads.iter().any(|existing| existing == &path)
                        || extra_reads.len() >= READ_ONLY_PREFLIGHT_MAX_EXTRA_READS
                    {
                        continue;
                    }
                    extra_reads.push(path.clone());
                    seen_read_paths.push(path.clone());
                    let limit = if matches!(intent, ReviewIntent::Security) {
                        SECURITY_PREFLIGHT_EXTRA_READ_LIMIT
                    } else {
                        DEFAULT_PREFLIGHT_EXTRA_READ_LIMIT
                    };
                    calls.push(PreflightCall::read(path, limit));
                }
            }
            let compacted_output = compact_preflight_tool_output(call.name, &output.content);
            let display_output = hi_tools::ToolOutput {
                content: compacted_output.clone(),
                display: None,
                plan: None,
            };
            emit_tool_output(ui, call.name, &display_output);
            content.push(Content::ToolCall {
                id: id.clone(),
                name: call.name.to_string(),
                arguments: call.arguments,
            });
            results.push((id, compacted_output));
            executed = executed.saturating_add(1);
        }

        if !content.is_empty() {
            self.messages.push_assistant_with_results(content, results);
        }
        executed
    }

    pub(crate) async fn run_implementation_preflight(
        &mut self,
        ui: &mut dyn Ui,
        tracker: &mut ImplementationTracker,
        tool_timeline: &mut Vec<ToolCallEntry>,
    ) -> u32 {
        let arguments = serde_json::json!({
            "command": implementation_preflight_command(),
            "timeout": 30,
        })
        .to_string();
        let id = format!("hi_implementation_preflight_{}", self.messages.len());
        ui.status("running implementation preflight inspection");
        ui.tool_started("bash", &arguments);
        ui.tool_call("bash", &arguments);
        let started = std::time::Instant::now();
        let output = execute("bash", &arguments).await;
        let duration_ms = started.elapsed().as_millis() as u64;
        let error = output.content.starts_with("Error:");
        tracker.preferred_validation = preferred_validation_from_preflight(&output.content);
        tool_timeline.push(ToolCallEntry {
            tool: "bash".to_string(),
            path: String::new(),
            duration_ms,
            error,
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

    pub(crate) async fn run_gpu_training_estimator_bootstrap(
        &mut self,
        ui: &mut dyn Ui,
        tracker: &mut ImplementationTracker,
        tool_timeline: &mut Vec<ToolCallEntry>,
        intent: ImplementationIntent,
    ) -> u32 {
        ui.status("bootstrapping GPU training estimator project");
        let mut executed = 0u32;
        let mut content = Vec::new();
        let mut results = Vec::new();

        for (index, (path, file_content)) in gpu_training_estimator_bootstrap_files(intent)
            .into_iter()
            .enumerate()
        {
            let arguments = serde_json::json!({
                "path": path,
                "content": file_content,
            })
            .to_string();
            let id = format!(
                "hi_implementation_bootstrap_{}_{}",
                self.messages.len(),
                index
            );
            ui.tool_started("write", &arguments);
            ui.tool_call("write", &arguments);
            let started = std::time::Instant::now();
            let output = execute("write", &arguments).await;
            let duration_ms = started.elapsed().as_millis() as u64;
            let error = output.content.starts_with("Error:");
            tracker.record_tool_result("write", &arguments, &output.content);
            tool_timeline.push(ToolCallEntry {
                tool: "write".to_string(),
                path: path.to_string(),
                duration_ms,
                error,
            });
            emit_tool_output(ui, "write", &output);
            content.push(Content::ToolCall {
                id: id.clone(),
                name: "write".to_string(),
                arguments,
            });
            results.push((id, output.content));
            self.invalidate_snapshot();
            executed = executed.saturating_add(1);
        }

        tracker.preferred_validation = Some("cargo test".to_string());
        let arguments = serde_json::json!({
            "command": "cargo test",
            "timeout": 180,
        })
        .to_string();
        let id = format!(
            "hi_implementation_bootstrap_validate_{}",
            self.messages.len()
        );
        ui.tool_started("bash", &arguments);
        ui.tool_call("bash", &arguments);
        let started = std::time::Instant::now();
        let output = execute_streaming("bash", &arguments, &mut |line: &str| {
            ui.tool_stream("bash", line);
        })
        .await;
        let duration_ms = started.elapsed().as_millis() as u64;
        let error = output.content.starts_with("Error:");
        tracker.record_tool_result("bash", &arguments, &output.content);
        tool_timeline.push(ToolCallEntry {
            tool: "bash".to_string(),
            path: String::new(),
            duration_ms,
            error,
        });
        emit_tool_output(ui, "bash", &output);
        content.push(Content::ToolCall {
            id: id.clone(),
            name: "bash".to_string(),
            arguments,
        });
        results.push((id, output.content));
        self.invalidate_snapshot();
        executed = executed.saturating_add(1);

        if !content.is_empty() {
            self.messages.push_assistant_with_results(content, results);
        }
        executed
    }

}
