//! Durable transcript staging around workspace settlement.

use anyhow::{Context, Result};
use hi_ai::Content;

impl crate::Agent {
    /// Stage an audit-only execution before workspace settlement.
    pub(crate) async fn stage_active_workspace_execution(
        &mut self,
        calls: &[(String, String, String)],
        assistant_content: &[Content],
        results: &[(String, String)],
        execution: &hi_workspace::ExecutionReport,
    ) -> Result<()> {
        self.stage_workspace_execution_inner(calls, assistant_content, results, execution, false)
            .await
    }

    /// Stage a provider-visible tool batch. If the process exits after local
    /// settlement but before ordinary turn persistence, resume reconstructs
    /// this exact assistant/result sequence once from the durable stage.
    pub(crate) async fn stage_visible_workspace_execution(
        &mut self,
        calls: &[(String, String, String)],
        assistant_content: &[Content],
        results: &[(String, String)],
        execution: &hi_workspace::ExecutionReport,
    ) -> Result<()> {
        self.stage_workspace_execution_inner(calls, assistant_content, results, execution, true)
            .await
    }

    async fn stage_workspace_execution_inner(
        &mut self,
        calls: &[(String, String, String)],
        assistant_content: &[Content],
        results: &[(String, String)],
        execution: &hi_workspace::ExecutionReport,
        visible_on_resume: bool,
    ) -> Result<()> {
        let pipefs = matches!(
            self.workspace_controller_binding().authority,
            hi_workspace::WorkspaceAuthority::PipeFs { .. }
        );
        let local_stage_required = !pipefs
            && self
                .session
                .as_ref()
                .is_some_and(|session| session.requires_local_workspace_execution_stage());
        if !pipefs && !local_stage_required {
            return Ok(());
        }
        let operation_id = match self.workspace_coordination.active_parent_operation() {
            Some(operation_id) => operation_id,
            None if !self.config.harness.features.workspace_controller_v2 => return Ok(()),
            None => anyhow::bail!("workspace execution has no admitted operation"),
        };
        anyhow::ensure!(
            calls.len() == results.len(),
            "workspace execution transcript has {} calls but {} results",
            calls.len(),
            results.len()
        );
        let transcript_calls = calls
            .iter()
            .zip(results)
            .map(|((call_id, name, _), (result_id, result))| {
                anyhow::ensure!(
                    call_id == result_id,
                    "workspace execution result order does not match call order"
                );
                Ok(crate::WorkspaceTranscriptCall {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    result: result.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let record = crate::WorkspaceTranscriptExecution {
            schema_version: crate::WorkspaceTranscriptExecution::SCHEMA_VERSION,
            operation_id,
            assistant_content: assistant_content.to_vec(),
            calls: transcript_calls,
            execution: execution.clone(),
        };
        anyhow::ensure!(
            self.session.is_some(),
            "workspace execution requires a durable session sink"
        );
        self.write_session(move |session| {
            if pipefs {
                session
                    .stage_workspace_execution(&record)
                    .context("staging PipeFS workspace execution transcript")
            } else {
                session
                    .stage_local_workspace_execution(&record, visible_on_resume)
                    .context("staging local workspace execution transcript")
            }
        })
        .await
    }

    /// Settle an admitted operation using the executor's real typed result.
    /// Storage success must never rewrite a failed/cancelled/indeterminate
    /// execution into `Succeeded` in the operation journal.
    pub(crate) async fn checkpoint_durable_workspace_with_execution(
        &mut self,
        mut execution: hi_workspace::ExecutionReport,
    ) -> Result<()> {
        if execution.workspace_may_have_changed && execution.content_digest.is_none() {
            execution.content_digest = Some(self.runtime.ledger().workspace_revision());
        }
        let local_operation = if matches!(
            self.workspace_controller_binding().authority,
            hi_workspace::WorkspaceAuthority::Local
        ) && self
            .session
            .as_ref()
            .is_some_and(|session| session.requires_local_workspace_execution_stage())
        {
            self.workspace_coordination.active_parent_operation()
        } else {
            None
        };
        let pending = self.runtime.background().pending_job_settlements().await;
        self.workspace_coordination
            .checkpoint(self.workspace_durability.clone(), execution)
            .await?;
        if let Some(operation_id) = local_operation {
            anyhow::ensure!(
                self.session.is_some(),
                "local workspace settlement lost its durable session sink"
            );
            self.write_session(move |session| {
                session.settle_local_workspace_execution(&operation_id)
            })
            .await
            .context("recording local workspace transcript settlement")?;
        }
        self.runtime
            .background()
            .settle_jobs_after_workspace(&pending)
            .await
    }
}
