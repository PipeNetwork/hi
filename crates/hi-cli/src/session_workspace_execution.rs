use anyhow::Result;

use super::{JsonlSession, SessionMeta, append_session_records_inner};

impl JsonlSession {
    pub(crate) fn materialize_workspace_execution_recovery(
        &self,
        loaded: &super::LoadedSession,
    ) -> Result<()> {
        if !loaded.workspace_execution_recovered {
            return Ok(());
        }
        let mut payload = serde_json::to_string(&SessionMeta::StateReplacement {
            messages: loaded.messages.clone(),
            goal: loaded.goal.clone(),
            decisions: loaded.decisions.entries().to_vec(),
            plan: loaded.plan.clone(),
        })?;
        payload.push('\n');
        append_session_records_inner(&self.path, &payload, true)?;
        sync_parent(&self.path)
    }

    pub(super) fn stage_workspace_execution_durable(
        &self,
        execution: &hi_agent::WorkspaceTranscriptExecution,
        visible_on_resume: bool,
    ) -> Result<()> {
        let mut payload = serde_json::to_string(&SessionMeta::WorkspaceExecutionStaged {
            visible_on_resume,
            execution: execution.clone(),
        })?;
        payload.push('\n');
        append_session_records_inner(&self.path, &payload, true)?;
        sync_parent(&self.path)
    }

    pub(super) fn settle_workspace_execution_durable(
        &self,
        operation_id: &hi_workspace::OperationId,
    ) -> Result<()> {
        let mut payload = serde_json::to_string(&SessionMeta::WorkspaceExecutionSettled {
            operation_id: operation_id.clone(),
        })?;
        payload.push('\n');
        append_session_records_inner(&self.path, &payload, true)?;
        sync_parent(&self.path)
    }

    pub fn record_remote_session_identity(&mut self, session_id: &str) -> Result<()> {
        crate::sync::validate_session_id(session_id)?;
        self.append_meta(&SessionMeta::RemoteSessionIdentity {
            session_id: session_id.to_string(),
        })
    }

    pub fn record_pipefs_mode(&mut self, enabled: bool) -> Result<()> {
        self.append_meta(&SessionMeta::PipeFsMode { enabled })
    }

    /// Persist checkpoint refs so a resumed session knows where it branched.
    #[allow(dead_code)]
    pub fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        self.append_meta(&SessionMeta::Checkpoints {
            refs: refs.to_vec(),
        })
    }
}

#[cfg(unix)]
fn sync_parent(path: &std::path::Path) -> Result<()> {
    use anyhow::Context;

    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    let result = std::fs::File::open(parent)
        .with_context(|| format!("opening {} for sync", parent.display()))
        .and_then(|directory| {
            directory
                .sync_all()
                .with_context(|| format!("syncing {}", parent.display()))
        });
    #[cfg(target_os = "macos")]
    if result.as_ref().is_err_and(|error| {
        error
            .root_cause()
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::PermissionDenied)
    }) {
        return Ok(());
    }
    result
}

#[cfg(not(unix))]
fn sync_parent(_path: &std::path::Path) -> Result<()> {
    Ok(())
}
