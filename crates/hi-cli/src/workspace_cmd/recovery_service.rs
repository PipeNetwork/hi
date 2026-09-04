use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use super::journal_recovery::{self, JournalRecoveryView, RecoveryTarget};

pub(super) struct RecoveryInventory {
    pub caches: Vec<hi_pipefs::PipeFsRecoveryCache>,
    pub journal_recoveries: Vec<JournalRecoveryView>,
}

pub(super) enum RecoveryInspection {
    Journal(JournalRecoveryView),
    Cache(hi_pipefs::PipeFsRecoveryCache),
}

pub(super) struct RecoveryRetryReceipt {
    pub requested_id: String,
    pub revision_id: Option<uuid::Uuid>,
}

pub(super) fn for_authority<'a>(
    authority: &'a super::CacheAuthority,
    session_id: &'a str,
) -> RecoveryService<'a> {
    RecoveryService::new(
        &authority.client,
        &authority.scope,
        session_id,
        authority.machine_id.as_deref(),
        &authority.sync_config,
    )
}

/// One recovery implementation shared by startup-safe `hi workspace` commands
/// and the interactive `/pipefs recover` compatibility surface.
pub(super) struct RecoveryService<'a> {
    client: &'a hi_pipefs::PipeFsClient,
    scope: &'a hi_pipefs::PipeFsCacheScope,
    session_id: &'a str,
    machine_id: Option<&'a str>,
    sync_config: &'a crate::sync::SyncConfig,
}

impl<'a> RecoveryService<'a> {
    pub fn new(
        client: &'a hi_pipefs::PipeFsClient,
        scope: &'a hi_pipefs::PipeFsCacheScope,
        session_id: &'a str,
        machine_id: Option<&'a str>,
        sync_config: &'a crate::sync::SyncConfig,
    ) -> Self {
        Self {
            client,
            scope,
            session_id,
            machine_id,
            sync_config,
        }
    }

    pub fn inventory(&self) -> Result<RecoveryInventory> {
        let caches = hi_pipefs::list_recovery_caches(self.scope, self.session_id)?;
        let journal_recoveries = journal_recovery::list(self.scope, &caches, self.session_id)?;
        Ok(RecoveryInventory {
            caches,
            journal_recoveries,
        })
    }

    pub fn inspect(&self, requested_id: &str) -> Result<RecoveryInspection> {
        let inventory = self.inventory()?;
        if let Some(recovery) =
            journal_recovery::find_loaded(&inventory.journal_recoveries, requested_id)?
        {
            return Ok(RecoveryInspection::Journal(recovery));
        }
        Ok(RecoveryInspection::Cache(
            hi_pipefs::inspect_recovery_cache(self.scope, self.session_id, requested_id)?,
        ))
    }

    pub async fn retry(&self, requested_id: &str) -> Result<RecoveryRetryReceipt> {
        let target = self.resolve_retry_target(requested_id)?;
        let machine_id = required_machine_id(self.machine_id)?;
        let lease = self
            .client
            .acquire_writer_lease(self.session_id, machine_id, false)
            .await?;
        // Repeat the complete journal-sibling and mixed-cache validation after
        // the network round trip. A target may not change under a held lease.
        let confirmed = self.resolve_retry_target(requested_id)?;
        ensure!(
            confirmed.cache_id == target.cache_id,
            "recovery target changed while acquiring the writer lease; inspect it again"
        );
        let sync = Arc::new(crate::sync::RemoteSessionSink::new_required(
            self.sync_config.clone(),
            self.session_id.to_owned(),
        )?);
        sync.set_pipefs_sync_required(true);
        sync.adopt_preactivation_lease(&lease, machine_id)?;
        let transcript: Arc<dyn hi_pipefs::PipeFsSessionBridge> = Arc::new(
            RecoveryTranscriptBridge::new(lease.lease.clone(), sync.clone()),
        );
        let journal_recovery_id = confirmed
            .journal_recovery_id
            .as_deref()
            .context("recovery retry has no exact journal operation identity")?;
        let receipt = hi_pipefs::retry_recovery_cache(
            self.client,
            lease.lease,
            self.session_id,
            &confirmed.cache_id,
            transcript,
            |operation, revision, cursor| {
                journal_recovery::mark_recovered_before_release(
                    self.scope,
                    self.session_id,
                    &confirmed.cache_id,
                    journal_recovery_id,
                    operation,
                    revision,
                    cursor,
                )
            },
        )
        .await;
        sync.set_pipefs_sync_required(false);
        let receipt = receipt?;
        Ok(RecoveryRetryReceipt {
            requested_id: requested_id.to_owned(),
            revision_id: receipt.revision_id,
        })
    }

    pub async fn export(&self, requested_id: &str, destination: &Path) -> Result<PathBuf> {
        let target = self.resolve(requested_id)?;
        let scope = self.scope.clone();
        let session_id = self.session_id.to_owned();
        let cache_id = target.cache_id;
        let destination = destination.to_owned();
        tokio::task::spawn_blocking(move || {
            hi_pipefs::export_recovery_cache(&scope, &session_id, &cache_id, &destination)
        })
        .await
        .context("PipeFS recovery export task panicked")?
    }

    pub async fn discard(&self, requested_id: &str, confirmation: &str) -> Result<()> {
        let target = self.resolve(requested_id)?;
        let scope = self.scope.clone();
        let session_id = self.session_id.to_owned();
        let cache_id = target.cache_id;
        let confirmation = confirmation.to_owned();
        tokio::task::spawn_blocking(move || {
            hi_pipefs::discard_recovery_cache(&scope, &session_id, &cache_id, &confirmation)
        })
        .await
        .context("PipeFS recovery discard task panicked")?
    }

    fn resolve(&self, requested_id: &str) -> Result<RecoveryTarget> {
        let inventory = self.inventory()?;
        journal_recovery::resolve_loaded(
            &inventory.journal_recoveries,
            requested_id,
            inventory
                .caches
                .iter()
                .any(|cache| cache.id == requested_id),
        )
    }

    fn resolve_retry_target(&self, requested_id: &str) -> Result<RecoveryTarget> {
        let inventory = self.inventory()?;
        resolve_retry_target_loaded(&inventory, requested_id)
    }
}

struct RecoveryTranscriptBridge {
    lease: hi_pipefs::PipeFsLease,
    sync: Arc<crate::sync::RemoteSessionSink>,
    lease_status: tokio::sync::watch::Sender<hi_pipefs::PipeFsLeaseStatus>,
}

impl RecoveryTranscriptBridge {
    fn new(lease: hi_pipefs::PipeFsLease, sync: Arc<crate::sync::RemoteSessionSink>) -> Self {
        let (lease_status, _) = tokio::sync::watch::channel(hi_pipefs::PipeFsLeaseStatus::Valid);
        Self {
            lease,
            sync,
            lease_status,
        }
    }
}

#[async_trait::async_trait]
impl hi_pipefs::PipeFsSessionBridge for RecoveryTranscriptBridge {
    fn subscribe_lease_status(&self) -> tokio::sync::watch::Receiver<hi_pipefs::PipeFsLeaseStatus> {
        self.lease_status.subscribe()
    }

    async fn refresh_lease(&self) -> Result<hi_pipefs::PipeFsLease> {
        if self.sync.writer_lease_is_lost() {
            bail!("lease_lost: preactivation recovery lease was replaced")
        }
        Ok(self.lease.clone())
    }

    async fn prepare_causal_mutation(&self) -> Result<()> {
        bail!("preactivation recovery cannot admit a new mutation")
    }

    async fn causal_transcript_batch(&self) -> Result<hi_pipefs::CausalTranscriptBatch> {
        self.sync.ensure_workspace_execution_staged()?;
        self.sync.causal_pipefs_transcript_batch()
    }

    async fn acknowledge_causal_transcript(
        &self,
        batch: &hi_pipefs::CausalTranscriptBatch,
        cursor: u64,
    ) -> Result<()> {
        self.sync
            .acknowledge_causal_pipefs_transcript(batch, cursor)
    }

    async fn flush_compatibility_transcript(
        &self,
        operation: &hi_pipefs::CausalOperationReceipt,
    ) -> Result<Option<u64>> {
        self.sync.ensure_workspace_execution_staged()?;
        self.sync
            .ensure_compatibility_workspace_execution(operation)?;
        self.sync.flush_required().await?;
        self.sync
            .compatibility_workspace_execution_cursor(operation)
            .map(Some)
    }
}

fn resolve_retry_target_loaded(
    inventory: &RecoveryInventory,
    requested_id: &str,
) -> Result<RecoveryTarget> {
    let mut target = journal_recovery::resolve_loaded(
        &inventory.journal_recoveries,
        requested_id,
        inventory
            .caches
            .iter()
            .any(|cache| cache.id == requested_id),
    )?;
    ensure!(
        target.remote_retry_safe,
        "journal recovery {} has no matching operation proof; export or discard it explicitly",
        target.requested_id
    );
    ensure!(
        inventory.caches.len() == 1 && inventory.caches[0].id == target.cache_id,
        "recovery retry requires the selected owning cache to be the session's only recovery cache"
    );
    let mut exact = inventory
        .journal_recoveries
        .iter()
        .filter(|view| view.recovery_cache_id == target.cache_id && view.remote_retry_safe);
    let proof = exact
        .next()
        .context("recovery retry requires one exact journal operation proof")?;
    ensure!(
        exact.next().is_none(),
        "recovery retry found multiple journal operation proofs in one cache"
    );
    if let Some(expected) = &target.journal_recovery_id {
        ensure!(
            expected == &proof.recovery_id,
            "requested recovery no longer matches its exact journal operation proof"
        );
    }
    target.journal_recovery_id = Some(proof.recovery_id.clone());
    Ok(target)
}

fn required_machine_id(machine_id: Option<&str>) -> Result<&str> {
    machine_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "a stable sync machine identity is required for recovery retry; set HI_SYNC_MACHINE_ID"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(id: &str) -> hi_pipefs::PipeFsRecoveryCache {
        hi_pipefs::PipeFsRecoveryCache {
            id: id.into(),
            confirmation_digest: Some(format!("blake3:{id}")),
            path: PathBuf::from(format!("/cache/{id}")),
            workspace_root: None,
            phase: None,
            logical_size_bytes: 0,
            pending_archive_bytes: 0,
            last_error: None,
        }
    }

    #[test]
    fn legacy_alias_retry_rejects_a_mixed_cache_set() {
        let inventory = RecoveryInventory {
            caches: vec![cache("first"), cache("second")],
            journal_recoveries: Vec::new(),
        };
        let error = resolve_retry_target_loaded(&inventory, "first").unwrap_err();
        assert!(
            error.to_string().contains("only recovery cache"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn recovery_retry_never_invents_a_writer_identity() {
        assert_eq!(
            required_machine_id(Some(" stable-machine ")).unwrap(),
            "stable-machine"
        );
        assert!(required_machine_id(None).is_err());
        assert!(required_machine_id(Some("  ")).is_err());
    }
}
