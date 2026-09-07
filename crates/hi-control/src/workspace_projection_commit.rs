//! Atomic binding/publication batches and their durable events.

use super::*;

impl WorkspaceProjectionJournal {
    // This boundary commits the complete binding snapshot and job receipt in
    // one transaction; keep both explicit instead of introducing another DTO.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_job_with_binding(
        &self,
        binding: &WorkspaceBinding,
        status: &WorkspaceStatus,
        capabilities: &WorkspaceCapabilities,
        permit: &JobPermit,
        state: JobState,
        detail: Option<String>,
        candidate_ref: Option<String>,
    ) -> Result<()> {
        let _gate = lock(&self.gate);
        let binding_record = self.prepare_binding(binding, status, capabilities)?;
        let job = self.prepare_job(binding, permit, state, detail, candidate_ref)?;
        self.commit_with_binding(binding_record, ProjectionTransition::Job(job))
    }

    pub(crate) fn record_settlement_with_binding(
        &self,
        binding: &WorkspaceBinding,
        status: &WorkspaceStatus,
        capabilities: &WorkspaceCapabilities,
        permit: &MutationPermitRecord,
        outcome: &SettlementOutcome,
    ) -> Result<()> {
        let _gate = lock(&self.gate);
        let binding_record = self.prepare_binding(binding, status, capabilities)?;
        let operation = self.prepare_operation(
            binding,
            permit,
            OperationProjectionUpdate {
                status: operation_status(outcome.status),
                execution_ref: None,
                settlement_ref: Some(digest_ref(outcome)?),
                result_version: outcome
                    .receipt
                    .as_ref()
                    .map(|receipt| json_string(&receipt.version))
                    .transpose()?,
                error: outcome.detail.clone(),
            },
        )?;
        self.commit_with_binding(
            binding_record,
            ProjectionTransition::WorkspaceOperation(operation),
        )
    }

    pub(crate) fn record_admission_with_binding(
        &self,
        binding: &WorkspaceBinding,
        status: &WorkspaceStatus,
        capabilities: &WorkspaceCapabilities,
        permit: &MutationPermitRecord,
    ) -> Result<()> {
        let _gate = lock(&self.gate);
        let binding_record = self.prepare_binding(binding, status, capabilities)?;
        let operation = self.prepare_operation(
            binding,
            permit,
            OperationProjectionUpdate {
                status: WorkspaceOperationStatus::Admitted,
                execution_ref: None,
                settlement_ref: None,
                result_version: None,
                error: None,
            },
        )?;
        self.commit_with_binding(
            binding_record,
            ProjectionTransition::WorkspaceOperation(operation),
        )
    }

    fn commit_with_binding(
        &self,
        binding: WorkspaceBindingRecord,
        transition: ProjectionTransition,
    ) -> Result<()> {
        let binding_event = projection_event(
            "workspace_binding",
            &binding.binding_id,
            binding.revision,
            binding.updated_at_ms,
            &binding.workspace_id,
            binding.session_id.as_deref(),
            workspace_activity_state(binding.state),
        );
        let event = match &transition {
            ProjectionTransition::Job(job) => projection_event(
                "workspace_job",
                &job.job_id,
                job.revision,
                job.updated_at_ms,
                &binding.workspace_id,
                job.session_id.as_deref(),
                job_activity_state(job.state),
            ),
            ProjectionTransition::WorkspaceOperation(operation) => projection_event(
                "workspace_operation",
                &operation.operation_id,
                operation.revision,
                operation.updated_at_ms,
                &binding.workspace_id,
                operation.session_id.as_deref(),
                operation_activity_state(operation.status),
            ),
            _ => unreachable!("only job and operation publication is batched with a binding"),
        };
        self.store.commit_batch(vec![
            (
                ProjectionTransition::WorkspaceBinding(binding),
                binding_event,
            ),
            (transition, event),
        ])?;
        Ok(())
    }

    pub(crate) fn commit_binding(&self, record: WorkspaceBindingRecord) -> Result<()> {
        let event = projection_event(
            "workspace_binding",
            &record.binding_id,
            record.revision,
            record.updated_at_ms,
            &record.workspace_id,
            record.session_id.as_deref(),
            workspace_activity_state(record.state),
        );
        self.store
            .commit(ProjectionTransition::WorkspaceBinding(record), event)?;
        Ok(())
    }

    pub(crate) fn commit_operation(
        &self,
        record: WorkspaceOperationRecord,
        workspace_id: &str,
    ) -> Result<()> {
        let event = projection_event(
            "workspace_operation",
            &record.operation_id,
            record.revision,
            record.updated_at_ms,
            workspace_id,
            record.session_id.as_deref(),
            operation_activity_state(record.status),
        );
        self.store
            .commit(ProjectionTransition::WorkspaceOperation(record), event)?;
        Ok(())
    }

    pub(crate) fn commit_job(&self, record: ControlJobRecord, workspace_id: &str) -> Result<()> {
        let event = projection_event(
            "workspace_job",
            &record.job_id,
            record.revision,
            record.updated_at_ms,
            workspace_id,
            record.session_id.as_deref(),
            job_activity_state(record.state),
        );
        self.store
            .commit(ProjectionTransition::Job(record), event)?;
        Ok(())
    }

    pub(crate) fn commit_recovery(&self, record: WorkspaceRecoveryRecord) -> Result<()> {
        let event = projection_event(
            "workspace_recovery",
            &record.recovery_id,
            record.revision,
            record.updated_at_ms,
            &record.workspace_id,
            record.session_id.as_deref(),
            recovery_activity_state(record.status),
        );
        self.store
            .commit(ProjectionTransition::WorkspaceRecovery(record), event)?;
        Ok(())
    }
}
