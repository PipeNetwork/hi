use super::{
    JobFence, JobRegistryError, JobState, JobTerminal, WorkspaceJobRegistry, WorkspaceJobSnapshot,
    ensure_fence, ensure_job_fence, lock, now_ms, publish,
};
use crate::{JobCompletion, JobPermit, RecoveryId};

pub(super) fn recovery_id_for(permit: &JobPermit) -> RecoveryId {
    let identity = format!(
        "workspace-job-recovery\0{}\0{}\0{}\0{}",
        permit.controller_id, permit.binding_id, permit.epoch, permit.job_id
    );
    RecoveryId::new(uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, identity.as_bytes()).to_string())
}

impl WorkspaceJobRegistry {
    /// Resolve a recovery fence after the authoritative workspace effects have
    /// been inspected or repaired. The interrupted lifecycle is deliberately
    /// finalized as failed; recovery never manufactures a success receipt.
    pub fn reconcile_recovery(
        &self,
        fence: &JobFence,
        recovery_id: &RecoveryId,
        detail: Option<String>,
    ) -> Result<WorkspaceJobSnapshot, JobRegistryError> {
        let mut state = lock(&self.inner.state);
        ensure_fence(&state, fence)?;
        let job_id = state
            .jobs
            .iter()
            .find_map(|(job_id, job)| {
                (job.recovery_id.as_ref() == Some(recovery_id)).then(|| job_id.clone())
            })
            .ok_or_else(|| JobRegistryError::RecoveryNotFound(recovery_id.clone()))?;
        let job = state.jobs.get_mut(&job_id).expect("job was just found");
        ensure_job_fence(job, fence)?;
        if job.state.is_terminal() {
            return Ok(job.clone());
        }
        if job.state != JobState::RecoveryRequired {
            return Err(JobRegistryError::UnexpectedState {
                job_id,
                expected: JobState::RecoveryRequired,
                actual: job.state,
            });
        }

        let detail = detail.unwrap_or_else(|| {
            "workspace recovery resolved; interrupted job was not published as successful".into()
        });
        job.state = JobState::Failed;
        job.revision = job.revision.saturating_add(1);
        job.updated_at_ms = now_ms();
        job.finalized_at_ms = Some(job.updated_at_ms);
        job.detail = Some(detail.clone());
        job.terminal = Some(JobTerminal {
            completion: JobCompletion::Failed,
            detail: Some(detail),
            artifacts: Vec::new(),
        });
        let snapshot = job.clone();
        publish(&self.inner, &mut state);
        Ok(snapshot)
    }
}
