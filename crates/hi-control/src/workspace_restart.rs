//! Crash-safe projection reconciliation for jobs found active after restart.

use hi_workspace::{
    BindingId, JobId, OperationId, RecoveryId, WorkspaceBinding, restart_job_recovery_id,
    restart_operation_recovery_id,
};

use crate::{
    ControlEffectScope, ControlError, ControlJobKind, ControlJobRecord, ControlJobState, Result,
    WorkspaceAuthority, WorkspaceOperationRecord, WorkspaceOperationStatus,
    WorkspaceProjectionJournal, WorkspaceProjectionState, WorkspaceRecoveryRecord,
    WorkspaceRecoveryStatus,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RestartReconciliation {
    pub recovery_required: Vec<JobId>,
    pub operation_recovery_required: Vec<OperationId>,
    pub orphaned: Vec<JobId>,
    pub stale: Vec<JobId>,
    pub recovery_ids: Vec<RecoveryId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalRecoveryDiscardReceipt {
    pub recovery_id: RecoveryId,
    pub operation_id: Option<OperationId>,
    pub job_id: Option<JobId>,
    pub confirmation_digest: String,
    pub lifecycle_marked_failed: bool,
}

impl WorkspaceProjectionJournal {
    pub fn reconcile_jobs_after_restart(
        &self,
        binding: &WorkspaceBinding,
    ) -> Result<RestartReconciliation> {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let unresolved = self
            .store
            .recoveries_for_binding(binding.binding_id.as_str())?
            .into_iter()
            .filter(|record| unresolved_for_binding(record, binding))
            .collect::<Vec<_>>();
        let jobs = self.store.jobs_for_binding(binding.binding_id.as_str())?;
        let mut report = RestartReconciliation::default();
        for mut job in jobs {
            let linked_unresolved = self
                .store
                .recoveries_for_job(&job.job_id)?
                .iter()
                .any(|record| unresolved_for_binding(record, binding));
            let writer = is_writer(&job);
            let recovery_id = (writer || linked_unresolved).then(|| {
                restart_job_recovery_id(
                    &binding.binding_id,
                    binding.epoch,
                    &JobId::new(job.job_id.clone()),
                )
            });
            if job.state.is_terminal() {
                if linked_unresolved {
                    let recovery_id = recovery_id.expect("linked recovery has a stable identity");
                    self.ensure_restart_recovery(binding, &job, &recovery_id)?;
                    push_recovery_id(&mut report, recovery_id);
                }
                continue;
            }
            let existing_recovery = match &recovery_id {
                Some(id) => self.store.recovery(id.as_str())?,
                None => None,
            };
            let recovery_resolved = existing_recovery.is_some_and(|record| {
                matches!(
                    record.status,
                    WorkspaceRecoveryStatus::Resolved | WorkspaceRecoveryStatus::Discarded
                )
            });
            let next = if job.epoch != Some(binding.epoch) && writer {
                report.stale.push(JobId::new(job.job_id.clone()));
                ControlJobState::Stale
            } else if writer && recovery_resolved {
                ControlJobState::Failed
            } else if writer {
                report
                    .recovery_required
                    .push(JobId::new(job.job_id.clone()));
                ControlJobState::RecoveryRequired
            } else {
                report.orphaned.push(JobId::new(job.job_id.clone()));
                ControlJobState::Orphaned
            };

            if job.state != next {
                update_restarted_job(&mut job, next);
                self.commit_job(job.clone(), binding.workspace_id.as_str())?;
            }
            if next == ControlJobState::RecoveryRequired || linked_unresolved {
                let recovery_id = recovery_id.expect("recoverable jobs have stable identities");
                self.ensure_restart_recovery(binding, &job, &recovery_id)?;
                push_recovery_id(&mut report, recovery_id);
            }
        }
        self.reconcile_operations_after_restart(binding, &mut report)?;
        for recovery in unresolved {
            let deterministic = deterministic_recovery_id(binding, &recovery);
            if deterministic
                .as_ref()
                .is_some_and(|id| report.recovery_ids.contains(id))
            {
                continue;
            }
            push_recovery_id(&mut report, RecoveryId::new(recovery.recovery_id));
        }
        if !report.recovery_ids.is_empty() {
            self.fence_binding_for_recovery(binding)?;
        }
        Ok(report)
    }

    /// Accept the current local workspace bytes without interpreting an
    /// interrupted writer's outcome. Callers must bind `confirmation_digest`
    /// to a fresh whole-workspace scan and obtain an explicit external-writer
    /// quiescence acknowledgement before invoking this method.
    pub fn discard_local_restart_recovery(
        &self,
        workspace_id: &str,
        binding_id: &str,
        recovery_id: &RecoveryId,
        operation_id: Option<&str>,
        job_id: Option<&str>,
        confirmation_digest: &str,
    ) -> Result<LocalRecoveryDiscardReceipt> {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if confirmation_digest.trim().is_empty() {
            return Err(ControlError::Invalid(
                "local recovery discard requires a content confirmation digest".into(),
            ));
        }
        if operation_id.is_some() && job_id.is_some() {
            return Err(ControlError::Invalid(
                "local recovery cannot identify both an operation and a writer job".into(),
            ));
        }
        let binding = self.store.binding(binding_id)?.ok_or_else(|| {
            ControlError::Invalid(format!("workspace binding {binding_id} was not found"))
        })?;
        if binding.workspace_id != workspace_id || binding.authority != WorkspaceAuthority::Local {
            return Err(ControlError::Invalid(
                "local recovery target does not belong to this local workspace".into(),
            ));
        }

        let detail = "operator accepted the freshly confirmed current workspace bytes after acknowledging external writers are quiescent; the interrupted lifecycle is Failed, and process reaping, success, cancellation, and rollback were not inferred";
        let now = hi_events::now_ms();
        let (operation, job, kind, artifact_ref) = if let Some(operation_id) = operation_id {
            let operation = self.store.operation(operation_id)?.ok_or_else(|| {
                ControlError::Invalid(format!("workspace operation {operation_id} was not found"))
            })?;
            let expected = restart_operation_recovery_id(
                &BindingId::new(binding.binding_id.clone()),
                binding.epoch,
                &OperationId::new(operation_id),
            );
            if operation.binding_id != binding_id || expected != *recovery_id {
                return Err(ControlError::Invalid(
                    "operation recovery identity does not match its binding and epoch".into(),
                ));
            }
            (Some(operation), None, "discarded_local_operation", None)
        } else if let Some(job_id) = job_id {
            let job = self.store.job(job_id)?.ok_or_else(|| {
                ControlError::Invalid(format!("workspace job {job_id} was not found"))
            })?;
            let expected = restart_job_recovery_id(
                &BindingId::new(binding.binding_id.clone()),
                binding.epoch,
                &JobId::new(job_id),
            );
            if job.binding_id.as_deref() != Some(binding_id)
                || job.epoch != Some(binding.epoch)
                || !is_writer(&job)
                || expected != *recovery_id
            {
                return Err(ControlError::Invalid(
                    "writer recovery identity does not match its binding and epoch".into(),
                ));
            }
            (
                None,
                Some(job.clone()),
                "discarded_local_writer_job",
                job.candidate_ref,
            )
        } else {
            (None, None, "discarded_local_binding_recovery", None)
        };

        let existing = self.store.recovery(recovery_id.as_str())?;
        if let Some(existing) = &existing {
            if existing.workspace_id != workspace_id
                || existing.binding_id.as_deref() != Some(binding_id)
                || existing.operation_id.as_deref() != operation_id
                || existing.job_id.as_deref() != job_id
                || existing.session_id.is_some()
            {
                return Err(ControlError::Invalid(
                    "persisted recovery does not match the requested local lifecycle".into(),
                ));
            }
            if existing.status == WorkspaceRecoveryStatus::Discarded
                && existing.digest.as_deref() != Some(confirmation_digest)
            {
                return Err(ControlError::Invalid(
                    "local recovery was already discarded with a different confirmation".into(),
                ));
            }
            if existing.status == WorkspaceRecoveryStatus::Resolved {
                return Err(ControlError::Invalid(
                    "local recovery was already resolved and cannot be discarded".into(),
                ));
            }
        } else {
            if operation_id.is_none() && job_id.is_none() {
                return Err(ControlError::Invalid(
                    "binding-level local recovery must already exist in the journal".into(),
                ));
            }
            self.commit_recovery(WorkspaceRecoveryRecord {
                recovery_id: recovery_id.to_string(),
                binding_id: Some(binding_id.to_owned()),
                workspace_id: workspace_id.to_owned(),
                session_id: None,
                operation_id: operation_id.map(str::to_owned),
                job_id: job_id.map(str::to_owned),
                kind: kind.to_owned(),
                status: WorkspaceRecoveryStatus::Required,
                digest: None,
                artifact_ref,
                detail: Some("local restart recovery awaiting explicit disposition".into()),
                error: None,
                revision: 1,
                created_at_ms: now,
                updated_at_ms: now,
                resolved_at_ms: None,
            })?;
        }

        // Persist the content decision before terminalizing the lifecycle. If
        // any earlier recovery update fails, the operation/job remains active
        // and restart reconciliation stays fail-closed. Once the stable
        // recovery is Discarded, restart reconciliation may safely finish the
        // lifecycle as Failed without interpreting the external effect.
        self.discard_linked_recoveries(
            recovery_id,
            confirmation_digest,
            detail,
            operation_id,
            job_id,
        )?;
        let mut lifecycle_marked_failed = false;
        if let Some(mut operation) = operation
            && operation.status != WorkspaceOperationStatus::Failed
            && operation_requires_recovery(operation.status)
        {
            update_restarted_operation(&mut operation, WorkspaceOperationStatus::Failed);
            operation.error = Some(detail.into());
            self.commit_operation(operation, workspace_id)?;
            lifecycle_marked_failed = true;
        }
        if let Some(mut job) = job
            && job.state != ControlJobState::Failed
            && !job.state.is_terminal()
        {
            update_restarted_job(&mut job, ControlJobState::Failed);
            job.error = Some(detail.into());
            self.commit_job(job, workspace_id)?;
            lifecycle_marked_failed = true;
        }
        Ok(LocalRecoveryDiscardReceipt {
            recovery_id: recovery_id.clone(),
            operation_id: operation_id.map(OperationId::new),
            job_id: job_id.map(JobId::new),
            confirmation_digest: confirmation_digest.to_owned(),
            lifecycle_marked_failed,
        })
    }

    fn reconcile_operations_after_restart(
        &self,
        binding: &WorkspaceBinding,
        report: &mut RestartReconciliation,
    ) -> Result<()> {
        for mut operation in self
            .store
            .operations_for_binding(binding.binding_id.as_str())?
        {
            let linked_unresolved = self
                .store
                .recoveries_for_operation(&operation.operation_id)?
                .iter()
                .any(|record| unresolved_for_binding(record, binding));
            let recovery_id = restart_operation_recovery_id(
                &binding.binding_id,
                binding.epoch,
                &OperationId::new(operation.operation_id.clone()),
            );
            if !operation_requires_recovery(operation.status) {
                if linked_unresolved {
                    self.ensure_restart_operation_recovery(binding, &operation, &recovery_id)?;
                    push_recovery_id(report, recovery_id);
                }
                continue;
            }
            let existing_recovery = self.store.recovery(recovery_id.as_str())?;
            let recovery_resolved = existing_recovery.as_ref().is_some_and(|record| {
                matches!(
                    record.status,
                    WorkspaceRecoveryStatus::Resolved | WorkspaceRecoveryStatus::Discarded
                )
            });
            let next = if recovery_resolved {
                WorkspaceOperationStatus::Failed
            } else {
                WorkspaceOperationStatus::RecoveryRequired
            };
            if operation.status != next {
                update_restarted_operation(&mut operation, next);
                self.commit_operation(operation.clone(), binding.workspace_id.as_str())?;
            }
            if next == WorkspaceOperationStatus::RecoveryRequired || linked_unresolved {
                self.ensure_restart_operation_recovery(binding, &operation, &recovery_id)?;
                if next == WorkspaceOperationStatus::RecoveryRequired {
                    report
                        .operation_recovery_required
                        .push(OperationId::new(operation.operation_id.clone()));
                }
                push_recovery_id(report, recovery_id);
            }
        }
        Ok(())
    }

    /// End the crashed job lifecycle without inferring success, cancellation,
    /// or rollback after authoritative recovery resolves its workspace effects.
    pub(crate) fn settle_recovered_job(&self, recovery_id: &RecoveryId) -> Result<()> {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(recovery) = self.store.recovery(recovery_id.as_str())? else {
            return Ok(());
        };
        let Some(job_id) = recovery.job_id.as_deref() else {
            return Ok(());
        };
        let Some(mut job) = self.store.job(job_id)? else {
            return Ok(());
        };
        if job.state.is_terminal() {
            return Ok(());
        }
        update_restarted_job(&mut job, ControlJobState::Failed);
        self.commit_job(job, &recovery.workspace_id)
    }

    /// End a crashed foreground operation after authoritative reconciliation.
    /// `Failed` means only that the interrupted lifecycle was not published as
    /// success; the recovery record remains the proof of the byte decision.
    pub(crate) fn settle_recovered_operation(&self, recovery_id: &RecoveryId) -> Result<()> {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(recovery) = self.store.recovery(recovery_id.as_str())? else {
            return Ok(());
        };
        let Some(operation_id) = recovery.operation_id.as_deref() else {
            return Ok(());
        };
        let Some(mut operation) = self.store.operation(operation_id)? else {
            return Ok(());
        };
        if !operation_requires_recovery(operation.status) {
            return Ok(());
        }
        update_restarted_operation(&mut operation, WorkspaceOperationStatus::Failed);
        self.commit_operation(operation, &recovery.workspace_id)
    }

    /// Resolve legacy/random recovery aliases that name the same durable
    /// operation or job as the deterministic restart recovery.
    pub(crate) fn resolve_linked_recoveries(&self, recovery_id: &RecoveryId) -> Result<()> {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(source) = self.store.recovery(recovery_id.as_str())? else {
            return Ok(());
        };
        let mut linked = match source.operation_id.as_deref() {
            Some(operation_id) => self.store.recoveries_for_operation(operation_id)?,
            None => Vec::new(),
        };
        if let Some(job_id) = source.job_id.as_deref() {
            linked.extend(self.store.recoveries_for_job(job_id)?);
        }
        linked.sort_by(|left, right| left.recovery_id.cmp(&right.recovery_id));
        linked.dedup_by(|left, right| left.recovery_id == right.recovery_id);
        let now = hi_events::now_ms();
        for mut record in linked {
            if record.recovery_id == recovery_id.as_str()
                || record.binding_id != source.binding_id
                || record.workspace_id != source.workspace_id
                || record.session_id != source.session_id
                || matches!(
                    record.status,
                    WorkspaceRecoveryStatus::Resolved | WorkspaceRecoveryStatus::Discarded
                )
            {
                continue;
            }
            record.status = WorkspaceRecoveryStatus::Resolved;
            record.revision = record.revision.saturating_add(1);
            record.updated_at_ms = now.max(record.created_at_ms);
            record.resolved_at_ms = Some(record.updated_at_ms);
            record.detail = Some(format!(
                "resolved through matching restart recovery {recovery_id}"
            ));
            self.commit_recovery(record)?;
        }
        Ok(())
    }

    fn discard_linked_recoveries(
        &self,
        recovery_id: &RecoveryId,
        confirmation_digest: &str,
        detail: &str,
        operation_id: Option<&str>,
        job_id: Option<&str>,
    ) -> Result<()> {
        let source = self.store.recovery(recovery_id.as_str())?.ok_or_else(|| {
            ControlError::Invalid(format!("workspace recovery {recovery_id} was not found"))
        })?;
        if source.operation_id.as_deref() != operation_id || source.job_id.as_deref() != job_id {
            return Err(ControlError::Invalid(
                "local recovery identity changed while recording its disposition".into(),
            ));
        }
        let mut linked = match operation_id {
            Some(operation_id) => self.store.recoveries_for_operation(operation_id)?,
            None => Vec::new(),
        };
        if let Some(job_id) = job_id {
            linked.extend(self.store.recoveries_for_job(job_id)?);
        }
        linked.push(source.clone());
        linked.sort_by(|left, right| {
            (left.recovery_id == recovery_id.as_str())
                .cmp(&(right.recovery_id == recovery_id.as_str()))
                .then_with(|| left.recovery_id.cmp(&right.recovery_id))
        });
        linked.dedup_by(|left, right| left.recovery_id == right.recovery_id);
        let now = hi_events::now_ms();
        for mut record in linked {
            if record.binding_id != source.binding_id
                || record.workspace_id != source.workspace_id
                || record.session_id.is_some()
                || record.status == WorkspaceRecoveryStatus::Resolved
            {
                continue;
            }
            if record.status == WorkspaceRecoveryStatus::Discarded {
                if record.digest.as_deref() != Some(confirmation_digest) {
                    return Err(ControlError::Invalid(format!(
                        "linked recovery {} has a different discard confirmation",
                        record.recovery_id
                    )));
                }
                continue;
            }
            record.status = WorkspaceRecoveryStatus::Discarded;
            record.digest = Some(confirmation_digest.to_owned());
            record.detail = Some(detail.to_owned());
            record.error = None;
            record.revision = record.revision.saturating_add(1);
            record.updated_at_ms = now.max(record.created_at_ms);
            record.resolved_at_ms = Some(record.updated_at_ms);
            self.commit_recovery(record)?;
        }
        Ok(())
    }

    fn ensure_restart_recovery(
        &self,
        binding: &WorkspaceBinding,
        job: &ControlJobRecord,
        recovery_id: &RecoveryId,
    ) -> Result<()> {
        if self.store.recovery(recovery_id.as_str())?.is_some() {
            return Ok(());
        }
        let now = hi_events::now_ms();
        self.commit_recovery(WorkspaceRecoveryRecord {
            recovery_id: recovery_id.to_string(),
            binding_id: Some(binding.binding_id.to_string()),
            workspace_id: binding.workspace_id.to_string(),
            session_id: session_id(binding),
            operation_id: None,
            job_id: Some(job.job_id.clone()),
            kind: "crashed_writer_job".to_owned(),
            status: WorkspaceRecoveryStatus::Required,
            digest: None,
            artifact_ref: job.candidate_ref.clone(),
            detail: Some("writer was active across restart; recovery is required".to_owned()),
            error: None,
            revision: 1,
            created_at_ms: now,
            updated_at_ms: now,
            resolved_at_ms: None,
        })
    }

    fn ensure_restart_operation_recovery(
        &self,
        binding: &WorkspaceBinding,
        operation: &WorkspaceOperationRecord,
        recovery_id: &RecoveryId,
    ) -> Result<()> {
        if self.store.recovery(recovery_id.as_str())?.is_some() {
            return Ok(());
        }
        let now = hi_events::now_ms();
        self.commit_recovery(WorkspaceRecoveryRecord {
            recovery_id: recovery_id.to_string(),
            binding_id: Some(binding.binding_id.to_string()),
            workspace_id: binding.workspace_id.to_string(),
            session_id: session_id(binding),
            operation_id: Some(operation.operation_id.clone()),
            job_id: None,
            kind: "crashed_foreground_operation".to_owned(),
            status: WorkspaceRecoveryStatus::Required,
            digest: Some(operation.operation_digest.clone()),
            artifact_ref: None,
            detail: Some(
                "foreground operation was unsettled across restart; reconcile effects without replay"
                    .to_owned(),
            ),
            error: operation.error.clone(),
            revision: 1,
            created_at_ms: now,
            updated_at_ms: now,
            resolved_at_ms: None,
        })
    }

    fn fence_binding_for_recovery(&self, binding: &WorkspaceBinding) -> Result<()> {
        let mut persisted = self
            .store
            .binding(binding.binding_id.as_str())?
            .ok_or_else(|| {
                ControlError::Invalid(format!(
                    "workspace binding {} was not journaled before restart reconciliation",
                    binding.binding_id
                ))
            })?;
        if persisted.state != WorkspaceProjectionState::RecoveryRequired {
            persisted.state = WorkspaceProjectionState::RecoveryRequired;
            persisted.revision = persisted.revision.saturating_add(1);
            persisted.updated_at_ms = hi_events::now_ms().max(persisted.opened_at_ms);
            self.commit_binding(persisted)?;
        }
        Ok(())
    }
}

fn is_writer(job: &ControlJobRecord) -> bool {
    matches!(
        job.effect_scope,
        ControlEffectScope::CandidateOnly | ControlEffectScope::LiveWriter
    ) || job.kind == ControlJobKind::WriteCandidate
}

fn unresolved_for_binding(recovery: &WorkspaceRecoveryRecord, binding: &WorkspaceBinding) -> bool {
    recovery.binding_id.as_deref() == Some(binding.binding_id.as_str())
        && recovery.workspace_id == binding.workspace_id.as_str()
        && recovery.session_id == session_id(binding)
        && !matches!(
            recovery.status,
            WorkspaceRecoveryStatus::Resolved | WorkspaceRecoveryStatus::Discarded
        )
}

fn push_recovery_id(report: &mut RestartReconciliation, recovery_id: RecoveryId) {
    if !report.recovery_ids.contains(&recovery_id) {
        report.recovery_ids.push(recovery_id);
    }
}

fn deterministic_recovery_id(
    binding: &WorkspaceBinding,
    recovery: &WorkspaceRecoveryRecord,
) -> Option<RecoveryId> {
    recovery.operation_id.as_deref().map_or_else(
        || {
            recovery.job_id.as_deref().map(|job_id| {
                restart_job_recovery_id(&binding.binding_id, binding.epoch, &JobId::new(job_id))
            })
        },
        |operation_id| {
            Some(restart_operation_recovery_id(
                &binding.binding_id,
                binding.epoch,
                &OperationId::new(operation_id),
            ))
        },
    )
}

fn update_restarted_job(job: &mut ControlJobRecord, next: ControlJobState) {
    job.state = next;
    job.revision = job.revision.saturating_add(1);
    job.updated_at_ms = hi_events::now_ms().max(job.created_at_ms);
    job.finished_at_ms = next.is_terminal().then_some(job.updated_at_ms);
    job.error = Some(restart_detail(next).to_owned());
}

fn operation_requires_recovery(status: WorkspaceOperationStatus) -> bool {
    !matches!(
        status,
        WorkspaceOperationStatus::Durable
            | WorkspaceOperationStatus::NoChange
            | WorkspaceOperationStatus::LocalAuditDegraded
            | WorkspaceOperationStatus::Failed
    )
}

fn update_restarted_operation(
    operation: &mut WorkspaceOperationRecord,
    next: WorkspaceOperationStatus,
) {
    operation.status = next;
    operation.revision = operation.revision.saturating_add(1);
    operation.updated_at_ms = hi_events::now_ms().max(operation.created_at_ms);
    operation.settled_at_ms =
        (next == WorkspaceOperationStatus::Failed).then_some(operation.updated_at_ms);
    operation.error = Some(
        match next {
            WorkspaceOperationStatus::RecoveryRequired => {
                "operation was unsettled across restart; recovery is required"
            }
            WorkspaceOperationStatus::Failed => {
                "interrupted operation effects were reconciled; success was not inferred"
            }
            _ => "operation was reconciled after restart",
        }
        .to_owned(),
    );
}

fn restart_detail(state: ControlJobState) -> &'static str {
    match state {
        ControlJobState::RecoveryRequired => {
            "writer was active across restart; recovery is required"
        }
        ControlJobState::Stale => "writer belongs to a stale workspace epoch",
        ControlJobState::Failed => {
            "crashed writer effects were reconciled; success was not inferred"
        }
        _ => "read-only job was active when the harness restarted",
    }
}

fn session_id(binding: &WorkspaceBinding) -> Option<String> {
    match &binding.authority {
        hi_workspace::WorkspaceAuthority::Local => None,
        hi_workspace::WorkspaceAuthority::PipeFs { session_id, .. } => Some(session_id.clone()),
    }
}

#[cfg(test)]
#[path = "workspace_local_recovery_tests.rs"]
mod local_recovery_tests;
