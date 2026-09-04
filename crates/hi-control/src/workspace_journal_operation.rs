use super::*;
use hi_workspace::{MutationPermitRecord, OperationId, restart_operation_recovery_id};

#[derive(Clone, Debug)]
pub(super) enum OperationJournalFence {
    Pending { operation_id: OperationId },
    RecoveryRequired(Box<OperationJournalRecovery>),
}

#[derive(Clone, Debug)]
pub(super) struct OperationJournalRecovery {
    recovery_id: RecoveryId,
    permit: MutationPermitRecord,
    execution: ExecutionReport,
    outcome: SettlementOutcome,
    detail: String,
}

impl OperationJournalFence {
    fn operation_id(&self) -> &OperationId {
        match self {
            Self::Pending { operation_id } => operation_id,
            Self::RecoveryRequired(recovery) => &recovery.permit.operation_id,
        }
    }

    fn recovery(&self) -> Option<(&RecoveryId, &str)> {
        match self {
            Self::Pending { .. } => None,
            Self::RecoveryRequired(recovery) => {
                Some((&recovery.recovery_id, recovery.detail.as_str()))
            }
        }
    }
}

impl JournaledWorkspaceController {
    pub(super) fn begin_operation_publication(&self, permit: &MutationPermitRecord) -> bool {
        if self.journal_health().policy != JournalFailurePolicy::PipeFsFailClosed {
            return false;
        }
        *lock(&self.operation_journal_fence) = Some(OperationJournalFence::Pending {
            operation_id: permit.operation_id.clone(),
        });
        self.publish_status();
        true
    }

    pub(super) fn release_pending_operation(&self, operation_id: &OperationId) {
        let mut fence = lock(&self.operation_journal_fence);
        if matches!(
            fence.as_ref(),
            Some(OperationJournalFence::Pending { operation_id: pending }) if pending == operation_id
        ) {
            *fence = None;
        }
        drop(fence);
        self.publish_status();
    }

    pub(super) fn finish_operation_publication(
        &self,
        permit: MutationPermitRecord,
        execution: ExecutionReport,
        mut outcome: SettlementOutcome,
    ) -> SettlementOutcome {
        let binding = self.inner.binding();
        let proof_error = validate_exact_settlement_proof(&permit, &outcome, &binding).err();

        if proof_error.is_none()
            && self.journal_is_healthy()
            && let Err(error) = self.journal.record_binding(
                &binding,
                &self.inner.status(),
                &self.inner.capabilities(),
            )
        {
            self.note_journal_failure(&error);
        }
        // The operation transition is deliberately the final normal-path
        // write. Any earlier failure leaves the admitted/executing row
        // discoverable after restart instead of persisting a false success.
        if proof_error.is_none()
            && self.journal_is_healthy()
            && let Err(error) = self
                .journal
                .record_operation_settled(&binding, &permit, &outcome)
        {
            self.note_journal_failure(&error);
        }
        if proof_error.is_none() && self.journal_is_healthy() {
            self.release_pending_operation(&permit.operation_id);
            return outcome;
        }

        let recovery_id =
            restart_operation_recovery_id(&permit.binding_id, permit.epoch, &permit.operation_id);
        let detail = proof_error.map_or_else(
            || {
                format!(
                    "PipeFS effects have an authoritative receipt, but control journal publication is incomplete; retry recovery {recovery_id} after repairing the journal"
                )
            },
            |error| {
                format!(
                    "PipeFS settlement lacks exact operation proof for recovery {recovery_id}: {error}"
                )
            },
        );
        *lock(&self.operation_journal_fence) = Some(OperationJournalFence::RecoveryRequired(
            Box::new(OperationJournalRecovery {
                recovery_id: recovery_id.clone(),
                permit: permit.clone(),
                execution,
                outcome: outcome.clone(),
                detail: detail.clone(),
            }),
        ));
        if let Err(error) = self.journal.record_recovery(
            &binding,
            &recovery_id,
            Some(permit.operation_id.to_string()),
            None,
            WorkspaceRecoveryStatus::Required,
            Some(detail.clone()),
        ) {
            self.note_journal_failure(&error);
        }
        self.publish_status();
        outcome.status = SettlementStatus::RecoveryRequired;
        outcome.recovery_id = Some(recovery_id);
        outcome.detail = Some(detail);
        outcome
    }

    pub(super) fn reconcile_operation_publication(
        &self,
        recovery: &RecoveryId,
    ) -> Option<RecoveryOutcome> {
        let fence = lock(&self.operation_journal_fence).clone();
        let OperationJournalFence::RecoveryRequired(fence_recovery) = fence? else {
            return None;
        };
        let OperationJournalRecovery {
            recovery_id,
            permit,
            execution,
            outcome,
            detail,
        } = *fence_recovery;
        if &recovery_id != recovery {
            return None;
        }

        let binding = self.inner.binding();
        let repair = self.repair_operation_publication(
            &recovery_id,
            &permit,
            &execution,
            &outcome,
            &binding,
        );
        if let Err(error) = repair {
            self.note_journal_failure(&error);
            return Some(RecoveryOutcome {
                recovery_id,
                status: RecoveryStatus::Pending,
                binding,
                detail: Some(format!(
                    "{detail}; journal repair/reload is still pending: {error}"
                )),
            });
        }

        let mut active = lock(&self.operation_journal_fence);
        if active
            .as_ref()
            .and_then(OperationJournalFence::recovery)
            .map(|v| v.0)
            == Some(&recovery_id)
        {
            *active = None;
        }
        drop(active);
        if lock(&self.job_journal_fences).is_empty() {
            let mut health = lock(&self.health);
            health.state = JournalHealthState::Healthy;
            health.detail = None;
        }
        self.publish_status();
        Some(RecoveryOutcome {
            recovery_id,
            status: RecoveryStatus::Recovered,
            binding,
            detail: Some(
                "control journal repaired and reloaded from the exact PipeFS operation receipt"
                    .to_owned(),
            ),
        })
    }

    fn repair_operation_publication(
        &self,
        recovery_id: &RecoveryId,
        permit: &MutationPermitRecord,
        execution: &ExecutionReport,
        outcome: &SettlementOutcome,
        binding: &hi_workspace::WorkspaceBinding,
    ) -> Result<()> {
        validate_exact_settlement_proof(permit, outcome, binding)?;
        self.journal.record_recovery(
            binding,
            recovery_id,
            Some(permit.operation_id.to_string()),
            None,
            WorkspaceRecoveryStatus::Required,
            Some("repairing PipeFS operation journal from an exact settlement receipt".into()),
        )?;
        self.journal
            .record_operation_execution(binding, permit, execution)?;
        self.journal
            .record_binding(binding, &self.inner.status(), &self.inner.capabilities())?;
        self.journal
            .record_operation_settled(binding, permit, outcome)?;
        self.verify_reloaded_operation(permit, outcome, binding)?;
        // Resolution is the last write. A crash before it leaves the Required
        // recovery discoverable even though the exact operation row landed.
        self.journal.record_recovery(
            binding,
            recovery_id,
            Some(permit.operation_id.to_string()),
            None,
            WorkspaceRecoveryStatus::Resolved,
            Some("exact remote operation receipt verified; journal repaired".into()),
        )?;
        self.verify_reloaded_recovery(recovery_id, permit, binding)
    }

    fn verify_reloaded_operation(
        &self,
        permit: &MutationPermitRecord,
        outcome: &SettlementOutcome,
        binding: &hi_workspace::WorkspaceBinding,
    ) -> Result<()> {
        let operation = self
            .journal
            .store
            .operation(permit.operation_id.as_str())?
            .ok_or_else(|| ControlError::Invalid("repaired operation did not reload".into()))?;
        let expected_status = match outcome.status {
            SettlementStatus::Durable => crate::WorkspaceOperationStatus::Durable,
            SettlementStatus::NoChange => crate::WorkspaceOperationStatus::NoChange,
            _ => {
                return Err(ControlError::Invalid(
                    "operation proof is not a publishable settlement".into(),
                ));
            }
        };
        let expected_version = serde_json::to_string(&binding.version)?;
        if operation.binding_id != permit.binding_id.as_str()
            || operation.epoch != permit.epoch
            || operation.idempotency_key != permit.idempotency_key.as_str()
            || operation.status != expected_status
            || operation.execution_ref.is_none()
            || operation.settlement_ref.is_none()
            || operation.result_version.as_deref() != Some(expected_version.as_str())
        {
            return Err(ControlError::Invalid(
                "reloaded operation does not match the exact settlement proof".into(),
            ));
        }
        let persisted_binding = self
            .journal
            .store
            .binding(binding.binding_id.as_str())?
            .ok_or_else(|| ControlError::Invalid("repaired binding did not reload".into()))?;
        if persisted_binding.workspace_id != binding.workspace_id.as_str()
            || persisted_binding.epoch != binding.epoch
            || persisted_binding.workspace_version.as_deref() != Some(expected_version.as_str())
        {
            return Err(ControlError::Invalid(
                "reloaded binding does not match the settlement receipt".into(),
            ));
        }
        Ok(())
    }

    fn verify_reloaded_recovery(
        &self,
        recovery_id: &RecoveryId,
        permit: &MutationPermitRecord,
        binding: &hi_workspace::WorkspaceBinding,
    ) -> Result<()> {
        let recovery = self
            .journal
            .store
            .recovery(recovery_id.as_str())?
            .ok_or_else(|| ControlError::Invalid("repaired recovery did not reload".into()))?;
        if recovery.status != WorkspaceRecoveryStatus::Resolved
            || recovery.binding_id.as_deref() != Some(binding.binding_id.as_str())
            || recovery.operation_id.as_deref() != Some(permit.operation_id.as_str())
            || recovery.job_id.is_some()
        {
            return Err(ControlError::Invalid(
                "reloaded recovery does not match its operation fence".into(),
            ));
        }
        Ok(())
    }
}

fn validate_exact_settlement_proof(
    permit: &MutationPermitRecord,
    outcome: &SettlementOutcome,
    binding: &hi_workspace::WorkspaceBinding,
) -> Result<()> {
    if !matches!(
        outcome.status,
        SettlementStatus::Durable | SettlementStatus::NoChange
    ) || outcome.operation_id != permit.operation_id
    {
        return Err(ControlError::Invalid(
            "settlement status or operation identity is not durable".into(),
        ));
    }
    let receipt = outcome.receipt.as_ref().ok_or_else(|| {
        ControlError::Invalid("durable PipeFS settlement omitted its receipt".into())
    })?;
    if receipt.operation_id != permit.operation_id
        || receipt.binding_id != permit.binding_id
        || receipt.epoch != permit.epoch
        || binding.controller_id != permit.controller_id
        || binding.binding_id != permit.binding_id
        || binding.epoch != permit.epoch
        || receipt.version != binding.version
    {
        return Err(ControlError::Invalid(
            "receipt binding, epoch, operation, or version does not match".into(),
        ));
    }
    Ok(())
}

pub(super) fn apply_overlays(
    status: WorkspaceStatus,
    health: &JournalHealth,
    job_fences: &BTreeMap<JobId, JobJournalFence>,
    operation_fence: Option<&OperationJournalFence>,
) -> WorkspaceStatus {
    let mut status = apply_health(status, health);
    if let Some(fence) = operation_fence {
        status.active_operation = Some(fence.operation_id().clone());
    }
    for job in job_fences.keys() {
        if !status.active_jobs.contains(job) {
            status.active_jobs.push(job.clone());
        }
    }
    status.active_jobs.sort();
    status.active_jobs.dedup();

    if let Some((recovery_id, detail)) = operation_fence.and_then(|fence| fence.recovery()) {
        status.state = WorkspaceState::RecoveryRequired;
        status.recovery_id = Some(recovery_id.clone());
        status.detail = Some(append_detail(status.detail, detail.to_owned()));
    } else if let Some((recovery_id, detail)) = job_recovery(job_fences) {
        status.state = WorkspaceState::RecoveryRequired;
        status.recovery_id = Some(recovery_id);
        status.detail = Some(append_detail(status.detail, detail));
    } else if (operation_fence.is_some() || !job_fences.is_empty())
        && status.state == WorkspaceState::Ready
    {
        status.state = WorkspaceState::Settling;
        status.detail = Some(append_detail(
            status.detail,
            "workspace publication is still settling".to_owned(),
        ));
    }
    status
}

pub(super) fn overlay_barrier(
    mut receipt: BarrierReceipt,
    health: &JournalHealth,
    job_fences: &BTreeMap<JobId, JobJournalFence>,
    operation_fence: Option<&OperationJournalFence>,
    deadline: Instant,
) -> BarrierReceipt {
    if let Some(fence) = operation_fence {
        receipt.active_operation = Some(fence.operation_id().clone());
    }
    for job in job_fences.keys() {
        if !receipt.pending_jobs.contains(job) {
            receipt.pending_jobs.push(job.clone());
        }
    }
    if let Some((recovery_id, detail)) = operation_fence.and_then(|fence| fence.recovery()) {
        receipt.status = BarrierStatus::RecoveryRequired;
        receipt.recovery_id = Some(recovery_id.clone());
        receipt.detail = Some(detail.to_owned());
        return receipt;
    }
    if let Some((recovery_id, detail)) = job_recovery(job_fences) {
        receipt.status = BarrierStatus::RecoveryRequired;
        receipt.recovery_id = Some(recovery_id);
        receipt.detail = Some(detail);
        return receipt;
    }
    if (operation_fence.is_some() || !job_fences.is_empty())
        && receipt.status == BarrierStatus::Passed
    {
        receipt.status = if Instant::now() >= deadline {
            BarrierStatus::TimedOut
        } else {
            BarrierStatus::Blocked
        };
        receipt.detail = Some("workspace publication is still settling".to_owned());
    }
    if health.state == JournalHealthState::PipeFsFailClosed {
        receipt.status = BarrierStatus::RecoveryRequired;
        receipt.detail = Some("PipeFS control journal requires reconciliation".to_owned());
    } else if health.state == JournalHealthState::LocalAuditDegraded {
        receipt.detail = Some("local audit journal is degraded".to_owned());
    }
    receipt
}

fn job_recovery(fences: &BTreeMap<JobId, JobJournalFence>) -> Option<(RecoveryId, String)> {
    fences.values().find_map(|fence| match fence {
        JobJournalFence::RecoveryRequired {
            recovery_id,
            detail,
        } => Some((recovery_id.clone(), detail.clone())),
        JobJournalFence::Pending => None,
    })
}

fn apply_health(mut status: WorkspaceStatus, health: &JournalHealth) -> WorkspaceStatus {
    match health.state {
        JournalHealthState::Healthy => status,
        JournalHealthState::LocalAuditDegraded => {
            if status.state == WorkspaceState::Ready {
                status.state = WorkspaceState::LocalAuditDegraded;
            }
            status.detail = health.detail.clone();
            status
        }
        JournalHealthState::PipeFsFailClosed => {
            status.state = WorkspaceState::JournalCorrupt;
            status.detail = health.detail.clone();
            status
        }
    }
}

fn append_detail(existing: Option<String>, addition: String) -> String {
    match existing {
        Some(existing) if existing == addition => existing,
        Some(existing) => format!("{existing}; {addition}"),
        None => addition,
    }
}

#[cfg(test)]
#[path = "workspace_journal_operation_tests.rs"]
mod tests;
