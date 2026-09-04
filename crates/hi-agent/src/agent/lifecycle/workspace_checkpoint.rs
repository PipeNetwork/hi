//! Final workspace reconciliation and its PipeFS transcript fence.

use anyhow::{Context, Result, ensure};
use hi_ai::Content;
use hi_workspace::{ExecutionDisposition, ExecutionReport, WorkspaceAuthority};

impl crate::Agent {
    /// Reconcile bytes left by lifecycle work and publish the matching receipt.
    ///
    /// Unlike a typed tool settlement, this boundary may not already own a
    /// permit. Admit its synthetic operation before constructing the evidence
    /// so PipeFS never commits a receipt against an empty transcript batch.
    pub async fn checkpoint_durable_workspace(&self) -> Result<()> {
        if self
            .workspace_coordination
            .active_parent_operation()
            .is_none()
        {
            let pending = self.runtime.background().pending_job_settlements().await;
            let admission = self
                .workspace_coordination
                .begin_intent(
                    self.workspace_durability.clone(),
                    hi_workspace::MutationIntent::reconciliation(),
                )
                .await;
            if let Err(error) = admission {
                let running_writer = error
                    .downcast_ref::<hi_workspace::AdmissionDenied>()
                    .is_some_and(|denied| {
                        denied.reason == hi_workspace::AdmissionDeniedReason::ActiveWriter
                    });
                if running_writer && pending.is_empty() {
                    // A deliberate local service may outlive this turn. Its
                    // spawn operation is already settled; defer its unstable
                    // after-snapshot until the terminal lifecycle callback.
                    return Ok(());
                }
                return Err(error);
            }
        }

        let mut execution =
            ExecutionReport::succeeded(Some(self.runtime.ledger().workspace_revision()));
        let stage_error = self.stage_final_reconciliation(&execution).err();
        if let Some(error) = &stage_error {
            execution.disposition = ExecutionDisposition::Indeterminate;
            execution.content_digest = None;
            execution.detail = Some(format!(
                "final workspace reconciliation transcript could not be staged: {error:#}"
            ));
        }

        let settlement = self
            .checkpoint_durable_workspace_with_execution(execution)
            .await;
        match (stage_error, settlement) {
            (None, Ok(())) => Ok(()),
            (Some(stage), Ok(())) => Err(stage).context(
                "final reconciliation staging failed even though the controller returned a receipt",
            ),
            (None, Err(settlement)) => Err(settlement),
            (Some(stage), Err(settlement)) => Err(settlement).context(format!(
                "final reconciliation staging failed before settlement: {stage:#}"
            )),
        }
    }

    fn stage_final_reconciliation(&self, execution: &ExecutionReport) -> Result<()> {
        let binding = self.workspace_controller_binding();
        if !matches!(binding.authority, WorkspaceAuthority::PipeFs { .. }) {
            return Ok(());
        }

        let permit = self
            .workspace_coordination
            .active_mutation_record()
            .context("PipeFS reconciliation has no admitted workspace operation")?;
        ensure!(
            permit.controller_id == binding.controller_id
                && permit.binding_id == binding.binding_id
                && permit.epoch == binding.epoch
                && permit.base_version == binding.version,
            "PipeFS reconciliation permit no longer matches the exact workspace binding and base version"
        );
        let record = crate::WorkspaceTranscriptExecution {
            schema_version: crate::WorkspaceTranscriptExecution::SCHEMA_VERSION,
            operation_id: permit.operation_id,
            assistant_content: vec![Content::Text(
                "Final workspace reconciliation completed.".into(),
            )],
            calls: Vec::new(),
            execution: execution.clone(),
        };
        self.workspace_durability
            .as_ref()
            .context("PipeFS reconciliation requires a durable transcript stager")?
            .stage_workspace_execution(&record)
            .context("staging final PipeFS workspace reconciliation")
    }
}
