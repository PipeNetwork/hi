use super::*;

pub(super) type FusedStep = Pin<Box<dyn Future<Output = hi_tools::ToolOutcome> + Send + 'static>>;

pub(super) struct FusedToolContext {
    pub process_runner: hi_tools::ProcessRunner,
    pub root: std::path::PathBuf,
    pub state_root: std::path::PathBuf,
    pub lsp: std::sync::Arc<hi_lsp::LspManager>,
    pub background: std::sync::Arc<hi_tools::BackgroundRegistry>,
    pub read_cache: std::sync::Arc<std::sync::Mutex<hi_tools::ReadCache>>,
    pub repo_map: std::sync::Arc<std::sync::Mutex<hi_tools::RepoMapCache>>,
    pub mcp: Option<std::sync::Arc<dyn hi_tools::McpBackend>>,
    pub memory: Option<std::sync::Arc<dyn hi_tools::MemoryBackend>>,
}

pub(super) fn fused_tool_step(
    context: FusedToolContext,
    name: String,
    arguments: String,
    prepared: Option<hi_tools::PreparedMutation>,
    failure: Option<hi_tools::ToolOutcome>,
) -> FusedStep {
    Box::pin(async move {
        let FusedToolContext {
            process_runner,
            root,
            state_root,
            lsp,
            background,
            read_cache,
            repo_map,
            mcp,
            memory,
        } = context;
        if let Some(failure) = failure {
            failure
        } else if let Some(prepared) = prepared {
            execute_prepared_in_runtime(&lsp, read_cache.as_ref(), prepared).await
        } else {
            execute_in_runtime_shared_with_runner(
                &process_runner,
                &root,
                &state_root,
                &lsp,
                background.as_ref(),
                read_cache.as_ref(),
                &repo_map,
                mcp.as_deref(),
                memory.as_deref(),
                &name,
                &arguments,
            )
            .await
        }
    })
}

/// Heap-allocate tool futures so the batch state machine does not store two
/// copies of `execute_in_runtime` on the stack. The path lock lives only in
/// [`hi_tools::execute_mutation_then_command`].
pub(super) async fn fuse_mutation_then_command<Mut, Cmd>(
    file_ops: hi_tools::FileOperationLockManager,
    root: std::path::PathBuf,
    path: String,
    mutate: Mut,
    command: Cmd,
) -> (hi_tools::ToolOutcome, hi_tools::ToolOutcome)
where
    Mut: FnOnce() -> FusedStep,
    Cmd: FnOnce() -> FusedStep,
{
    let mutation_slot = std::sync::Arc::new(std::sync::Mutex::new(None::<hi_tools::ToolOutcome>));
    let command_slot = std::sync::Arc::new(std::sync::Mutex::new(None::<hi_tools::ToolOutcome>));
    let mutation_slot_m = mutation_slot.clone();
    let command_slot_c = command_slot.clone();
    let fusion = hi_tools::execute_mutation_then_command(
        &file_ops,
        &root,
        &path,
        || {
            let fut = mutate();
            async move {
                let mutation = fut.await;
                let step = hi_tools::MutationStep {
                    ok: mutation.status == hi_tools::ToolStatus::Succeeded,
                    output: mutation.content.clone(),
                };
                *mutation_slot_m.lock().unwrap() = Some(mutation);
                step
            }
        },
        Some(|| {
            let fut = command();
            async move {
                let command = fut.await;
                let step = hi_tools::CommandStep {
                    ok: command.status == hi_tools::ToolStatus::Succeeded,
                    output: command.content.clone(),
                };
                *command_slot_c.lock().unwrap() = Some(command);
                step
            }
        }),
    )
    .await;
    let mut mutation = mutation_slot
        .lock()
        .unwrap()
        .take()
        .expect("fused mutation ran");
    mutation.content = fusion.combined();
    let command = if fusion.command_ran {
        let mut command = command_slot
            .lock()
            .unwrap()
            .take()
            .expect("fused command ran");
        // Never stub the bash slot. Models look there for cargo check/test
        // output; a "look at the write result" stub is how a failed E0425
        // turned into "check passed" and stalled recovery.
        let marker = if fusion.command_ok == Some(true) {
            hi_tools::FUSED_COMMAND_SUCCEEDED
        } else {
            hi_tools::FUSED_COMMAND_FAILED
        };
        command.content = format!("{marker}\n{}", command.content);
        command
    } else {
        synthetic_tool_outcome(
            format!(
                "{}\nThe file mutation did not complete successfully; the command was not run.",
                hi_tools::FUSED_COMMAND_SKIPPED
            ),
            hi_tools::ToolStatus::Failed,
        )
    };
    (mutation, command)
}
