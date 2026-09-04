//! Startup recovery for fsynced detached-candidate artifacts.

use anyhow::Result;
use hi_control::ControlJobState;
use hi_workspace::{RecoveryKind, RecoveryRecord, WorkspaceController};

pub(super) fn reconcile(
    controller: &hi_workspace::InMemoryWorkspaceController,
    journal: &hi_control::WorkspaceProjectionJournal,
    store: &hi_control::ControlStore,
) -> Result<()> {
    let binding = controller.binding();
    for artifact in
        hi_tools::candidate_workspace::PersistedDetachedCandidate::discover(&binding.state_root)?
    {
        let job_id = artifact.detached.candidate.job_id.to_string();
        match store.get_job(&job_id)? {
            Some(job) if candidate_artifact_is_disposable(job.state) => {
                artifact.remove_after_terminal()?
            }
            Some(_) => {
                journal.record_job_artifact(
                    &job_id,
                    &artifact.artifact,
                    binding.workspace_id.as_str(),
                )?;
            }
            None => {
                let uri = artifact.artifact.uri.clone();
                controller.require_recovery(RecoveryRecord {
                    schema_version: hi_workspace::WORKSPACE_CONTRACT_SCHEMA_VERSION,
                    recovery_id: hi_workspace::RecoveryId::new(
                        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, uri.as_bytes()).to_string(),
                    ),
                    kind: RecoveryKind::CrashedWriterJob,
                    binding_id: binding.binding_id.clone(),
                    epoch: binding.epoch,
                    operation_id: None,
                    job_id: Some(artifact.detached.candidate.job_id.clone()),
                    detail: format!(
                        "sealed candidate artifact has no job projection; inspect {uri} before recovery"
                    ),
                    created_at_ms: hi_events::now_ms(),
                    resolved: false,
                })?;
            }
        }
    }
    Ok(())
}

/// A terminal lifecycle is not automatically a disposition of its candidate
/// bytes. `Stale` is intentionally reviewable/rerunnable, while `Orphaned`
/// means the harness cannot prove what happened. Recovery states are likewise
/// retained by the nonterminal arm above. Only outcomes that explicitly say
/// the candidate was applied, rejected, or cancelled may discard the sealed
/// artifact during restart cleanup.
fn candidate_artifact_is_disposable(state: ControlJobState) -> bool {
    matches!(
        state,
        ControlJobState::Succeeded | ControlJobState::Failed | ControlJobState::Cancelled
    )
}

#[cfg(test)]
#[path = "workspace_candidate_recovery_tests.rs"]
mod tests;
