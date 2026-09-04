//! Workspace MCP inspection and controller-guarded administration.

impl crate::Agent {
    /// Workspace MCP status table, if a host is attached.
    pub async fn mcp_workspace_status(&self) -> Option<String> {
        let backend = self.mcp.as_ref()?;
        Some(backend.workspace_status().await)
    }

    /// Workspace MCP admin (`reconnect` / `enable` / `disable`).
    ///
    /// Admin commands may both write repository configuration and start or
    /// stop a transport. Treat them as non-replayable live-writer operations:
    /// admission must precede the backend call, and the backend's real result
    /// is settled even when it failed after making a partial change.
    pub async fn mcp_workspace_admin(&mut self, args: &str) -> Option<anyhow::Result<String>> {
        let backend = self.mcp.as_ref()?.clone();
        let intent = hi_workspace::MutationIntent {
            effect_scope: hi_workspace::EffectScope::LiveWriter,
            replay_class: hi_workspace::ReplayClass::NonReplayableExternal,
            dirty_paths: None,
            description: Some("workspace MCP administration".into()),
        };
        if let Err(error) = self.begin_classified_workspace_operation(intent).await {
            return Some(Err(error.context(
                "workspace controller refused the MCP configuration mutation",
            )));
        }

        let admin_result = backend.workspace_admin(args).await;
        let reconciliation = self.runtime.reconcile_ledger_async().await;
        let (changes, reconciliation_error) = match reconciliation {
            Ok(changes) => (changes, None),
            Err(error) => (Vec::new(), Some(error)),
        };
        let mut execution = hi_workspace::ExecutionReport {
            disposition: if reconciliation_error.is_some() {
                hi_workspace::ExecutionDisposition::Indeterminate
            } else if admin_result.is_ok() {
                hi_workspace::ExecutionDisposition::Succeeded
            } else {
                hi_workspace::ExecutionDisposition::Failed
            },
            workspace_may_have_changed: reconciliation_error.is_some() || !changes.is_empty(),
            // Even a failed reconnect/enable/disable may have spawned,
            // disconnected, or contacted a transport. It must never be
            // replayed merely because the response or settlement was lost.
            external_effect_may_have_occurred: true,
            content_digest: None,
            changed_paths: changes
                .iter()
                .map(|change| std::path::PathBuf::from(&change.path))
                .collect(),
            artifacts: Vec::new(),
            detail: execution_detail(&admin_result, reconciliation_error.as_ref()),
        };

        let operation_id = self
            .workspace_coordination
            .active_parent_operation()
            .expect("workspace MCP operation was admitted above");
        let call_id = format!("workspace-mcp-admin:{operation_id}");
        let arguments = serde_json::json!({ "command": args }).to_string();
        let calls = [(
            call_id.clone(),
            "workspace_mcp_admin".into(),
            arguments.clone(),
        )];
        let assistant_content = [hi_ai::Content::ToolCall {
            id: call_id.clone(),
            name: "workspace_mcp_admin".into(),
            arguments,
        }];
        let results = [(
            call_id,
            match &admin_result {
                Ok(output) => output.clone(),
                Err(error) => format!("error: {error:#}"),
            },
        )];
        let stage_error = self
            .stage_active_workspace_execution(&calls, &assistant_content, &results, &execution)
            .err();
        if let Some(error) = &stage_error {
            execution.disposition = hi_workspace::ExecutionDisposition::Indeterminate;
            execution.detail = Some(match execution.detail.take() {
                Some(detail) => format!(
                    "{detail}; the exact PipeFS MCP lifecycle record could not be staged: {error:#}"
                ),
                None => {
                    format!("the exact PipeFS MCP lifecycle record could not be staged: {error:#}")
                }
            });
        }

        let settlement = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;
        Some(combine_result(
            admin_result,
            reconciliation_error,
            stage_error,
            settlement,
        ))
    }
}

fn execution_detail(
    admin: &anyhow::Result<String>,
    reconciliation: Option<&anyhow::Error>,
) -> Option<String> {
    match (admin, reconciliation) {
        (Ok(_), None) => None,
        (Err(error), None) => Some(format!("workspace MCP administration failed: {error:#}")),
        (Ok(_), Some(error)) => Some(format!(
            "workspace MCP administration returned success, but its workspace effects could not be reconciled: {error:#}"
        )),
        (Err(admin_error), Some(reconcile_error)) => Some(format!(
            "workspace MCP administration failed ({admin_error:#}), and its workspace effects could not be reconciled ({reconcile_error:#})"
        )),
    }
}

fn combine_result(
    admin: anyhow::Result<String>,
    reconcile: Option<anyhow::Error>,
    stage: Option<anyhow::Error>,
    settle: anyhow::Result<()>,
) -> anyhow::Result<String> {
    if let (Ok(output), None, None, Ok(())) = (&admin, &reconcile, &stage, &settle) {
        return Ok(output.clone());
    }
    let mut failures = Vec::new();
    if let Err(error) = admin {
        failures.push(format!("workspace MCP command failed: {error:#}"));
    }
    if let Some(error) = reconcile {
        failures.push(format!(
            "workspace MCP effects could not be reconciled: {error:#}"
        ));
    }
    if let Some(error) = stage {
        failures.push(format!(
            "workspace MCP transcript could not be staged: {error:#}"
        ));
    }
    if let Err(error) = settle {
        failures.push(format!("workspace MCP settlement failed: {error:#}"));
    }
    Err(anyhow::anyhow!(failures.join("; ")))
}
