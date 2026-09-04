use anyhow::Result;

/// Host-provided durability fence for a materialized workspace.
///
/// The agent remains unaware of archives or remote storage. It only marks the
/// start of a mutation and waits for the host to durably acknowledge the
/// resulting workspace before the tool batch is committed to the transcript.
#[async_trait::async_trait]
pub trait WorkspaceDurability: Send + Sync {
    /// Refuse when a previous revision is pending or this writer's lease is
    /// stale, then record a recovery marker before local bytes can change.
    async fn mutation_started(&self, dirty_paths: Option<Vec<String>>) -> Result<()>;

    /// Reconcile the materialized tree and durably commit any changed bytes.
    async fn checkpoint(&self) -> Result<()>;

    /// Durably stage the execution record which must be published with the
    /// next remote workspace receipt. Local durability backends never need
    /// this hook. The default is deliberately fail-closed because a remote
    /// controller must not settle bytes against a transcript batch that omits
    /// the native verifier which produced them.
    fn stage_workspace_execution(
        &self,
        _record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        anyhow::bail!("this workspace durability backend cannot stage execution evidence")
    }

    /// Start or stop periodic reconciliation for a native background process.
    /// Implementations that do not need it may ignore the notification.
    async fn background_process_state(&self, _id: &str, _running: bool) -> Result<()> {
        Ok(())
    }
}
