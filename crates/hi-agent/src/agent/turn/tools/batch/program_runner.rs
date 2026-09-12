use super::*;

#[derive(Clone)]
pub(super) struct ProgramToolRunner {
    pub(super) root: std::path::PathBuf,
    pub(super) state_root: std::path::PathBuf,
    pub(super) process_runner: hi_tools::ProcessRunner,
    pub(super) lsp: std::sync::Arc<hi_lsp::LspManager>,
    pub(super) background: std::sync::Arc<hi_tools::BackgroundRegistry>,
    pub(super) read_cache: std::sync::Arc<std::sync::Mutex<hi_tools::ReadCache>>,
    pub(super) repo_map: std::sync::Arc<std::sync::Mutex<hi_tools::RepoMapCache>>,
    pub(super) mcp: Option<std::sync::Arc<dyn hi_tools::McpBackend>>,
    pub(super) memory: Option<std::sync::Arc<dyn hi_tools::MemoryBackend>>,
}

impl ProgramToolRunner {
    pub(super) async fn execute(
        &self,
        call: &ProgramCall,
    ) -> (
        std::result::Result<hi_workflow::ProgramToolResult, String>,
        hi_tools::ToolOutcome,
    ) {
        let args = serde_json::to_string(&call.arguments).unwrap_or_default();
        let allowed = (matches!(
            hi_tools::speculation_class(&call.name),
            hi_tools::SpeculationClass::PureLocal | hi_tools::SpeculationClass::IdempotentExternal
        ) || call.name == "bash_output")
            && hi_tools::is_read_only(&call.name)
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
