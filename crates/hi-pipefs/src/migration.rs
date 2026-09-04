use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    ARCHIVE_VERSION, CausalOperationReceipt, PipeFsCacheScope, PipeFsClient, PipeFsLease,
    PipeFsSessionBridge, PipeFsWorkspace, PipeFsWorkspaceConfig, Snapshot, apply_archive_file,
    build_revision_from_snapshot_to_file_bounded, list_recovery_caches, scan_workspace,
};

pub const IMPORT_PREVIEW_SCHEMA_VERSION: u16 = 1;
pub const PIPEFS_SCANNER_VERSION: &str = "pipefs-snapshot-v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportPreview {
    pub schema_version: u16,
    pub scanner_version: String,
    pub source: PathBuf,
    pub confirmation_digest: String,
    pub entry_count: usize,
    pub byte_count: u64,
    pub exclusions: Vec<String>,
    pub unsupported_entries: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportReceipt {
    pub revision_id: Uuid,
    pub preview: ImportPreview,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipeFsRecoveryReceipt {
    pub revision_id: Option<Uuid>,
    pub recovered_cache_id: String,
}

struct TemporaryArchive(PathBuf);

impl Drop for TemporaryArchive {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub fn preview_import(source: &Path) -> Result<ImportPreview> {
    let source = source
        .canonicalize()
        .with_context(|| format!("canonicalizing import source {}", source.display()))?;
    ensure!(source.is_dir(), "PipeFS import source is not a directory");
    let inventory = import_inventory(&source)?;
    if !inventory.unsupported.is_empty() {
        let bytes = serde_json::to_vec(&(
            PIPEFS_SCANNER_VERSION,
            inventory.entry_count,
            inventory.byte_count,
            &inventory.unsupported,
        ))?;
        let mut digest = blake3::Hasher::new_derive_key("hi.pipefs.import-preview.blocked.v1");
        digest.update(&bytes);
        return Ok(ImportPreview {
            schema_version: IMPORT_PREVIEW_SCHEMA_VERSION,
            scanner_version: PIPEFS_SCANNER_VERSION.into(),
            source,
            confirmation_digest: format!("blocked:blake3:{}", digest.finalize().to_hex()),
            entry_count: inventory.entry_count,
            byte_count: inventory.byte_count,
            exclusions: Vec::new(),
            unsupported_entries: inventory.unsupported,
        });
    }
    let snapshot = scan_workspace(&source).context("scanning PipeFS import source")?;
    preview_from_snapshot(source, &snapshot)
}

pub async fn import_workspace(
    client: &PipeFsClient,
    lease: PipeFsLease,
    session_id: &str,
    source: &Path,
    confirmation: &str,
) -> Result<ImportReceipt> {
    ensure!(
        list_recovery_caches(&client.cache_scope(), session_id)?.is_empty(),
        "PipeFS import is blocked while local recovery evidence exists"
    );
    let capabilities = client.capabilities().await?;
    ensure!(
        capabilities.archive_version == ARCHIVE_VERSION,
        "unsupported PipeFS archive version {}",
        capabilities.archive_version
    );
    ensure!(
        capabilities.enrollment_available(),
        "new PipeFS enrollment is disabled"
    );
    let remote = client.state(session_id).await?;
    ensure!(
        remote.current_head.is_none(),
        "PipeFS import is allowed only when the remote workspace has no head"
    );

    let preview = confirmed_import_preview(source, confirmation)?;
    ensure!(
        preview.byte_count <= capabilities.maximum_workspace_bytes,
        "workspace exceeds the negotiated PipeFS workspace limit"
    );

    let workspace = PipeFsWorkspace::new(
        client.clone(),
        lease,
        PipeFsWorkspaceConfig {
            session_id: session_id.to_owned(),
            cache_scope: client.cache_scope(),
            original_workspace_root: preview.source.clone(),
            original_state_root: preview.source.clone(),
            cache_base: None,
        },
    )?;
    let activation = workspace.enable().await?;
    let status = workspace.status().await;
    ensure!(
        status.last_committed_revision.is_none(),
        "remote workspace gained a head during import; source was not applied"
    );

    let archive_path = activation
        .state_root
        .join(format!(".import-{}.tar.zst", Uuid::new_v4().simple()));
    let temporary = TemporaryArchive(archive_path.clone());
    let source_root = preview.source.clone();
    let snapshot = scan_workspace(&source_root)?;
    let artifact = tokio::task::spawn_blocking(move || {
        build_revision_from_snapshot_to_file_bounded(
            &source_root,
            snapshot,
            None,
            true,
            &archive_path,
            capabilities.maximum_revision_bytes,
        )
    })
    .await
    .context("PipeFS import archive task panicked")??;
    let destination = activation.workspace_root.clone();
    let archive = artifact.path.clone();
    tokio::task::spawn_blocking(move || apply_archive_file(&destination, &archive, None))
        .await
        .context("PipeFS import materialization task panicked")??;
    drop(temporary);

    let imported = preview_import(&activation.workspace_root)?;
    ensure!(
        imported.confirmation_digest == confirmation,
        "materialized PipeFS import differs from the confirmed source"
    );
    let revision_id = publish_import_after(
        || {
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::ImportBeforePublish)
                .map_err(anyhow::Error::from)
        },
        async {
            workspace.mutation_started(None).await?;
            workspace
                .checkpoint()
                .await?
                .ok_or_else(|| anyhow::anyhow!("PipeFS import produced no remote revision"))
        },
    )
    .await?;
    Ok(ImportReceipt {
        revision_id,
        preview,
    })
}

async fn publish_import_after<T>(
    before: impl FnOnce() -> Result<()>,
    publish: impl Future<Output = Result<T>>,
) -> Result<T> {
    before()?;
    publish.await
}

pub async fn retry_recovery_cache<F>(
    client: &PipeFsClient,
    lease: PipeFsLease,
    session_id: &str,
    recovery_id: &str,
    transcript: Arc<dyn PipeFsSessionBridge>,
    before_release: F,
) -> Result<PipeFsRecoveryReceipt>
where
    F: FnOnce(&CausalOperationReceipt, Option<Uuid>, u64) -> Result<()>,
{
    let caches = list_recovery_caches(&client.cache_scope(), session_id)?;
    ensure!(
        caches.len() == 1 && caches[0].id == recovery_id,
        "recovery retry requires the named cache to be the session's only recovery cache"
    );
    retry_recovery_cache_from(
        client,
        lease,
        session_id,
        recovery_id,
        caches.into_iter().next().expect("checked one cache"),
        None,
        transcript,
        before_release,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn retry_recovery_cache_from<F>(
    client: &PipeFsClient,
    lease: PipeFsLease,
    session_id: &str,
    recovery_id: &str,
    cache: crate::PipeFsRecoveryCache,
    cache_base: Option<PathBuf>,
    transcript: Arc<dyn PipeFsSessionBridge>,
    before_release: F,
) -> Result<PipeFsRecoveryReceipt>
where
    F: FnOnce(&CausalOperationReceipt, Option<Uuid>, u64) -> Result<()>,
{
    let recovery_root = cache
        .workspace_root
        .clone()
        .ok_or_else(|| anyhow::anyhow!("recovery cache has no verified workspace root"))?;
    let workspace = PipeFsWorkspace::open_recovery_cache(
        client.clone(),
        lease.clone(),
        PipeFsWorkspaceConfig {
            session_id: session_id.to_owned(),
            cache_scope: client.cache_scope(),
            original_workspace_root: recovery_root.clone(),
            original_state_root: recovery_root,
            cache_base,
        },
        recovery_id,
    )?;
    let refreshed = transcript.refresh_lease().await?;
    ensure!(
        refreshed == lease,
        "transcript and workspace recovery leases do not share the exact generation and token"
    );
    let remote = client.state(session_id).await?;
    let activation = if remote.enabled {
        workspace.restore_existing().await?
    } else {
        workspace.enable().await?
    };
    let causal = workspace.persisted_causal_recovery().await;
    let compatibility = workspace.persisted_compatibility_recovery().await;
    ensure!(
        causal.is_some() ^ compatibility.is_some(),
        "recovery cache must contain exactly one pending operation"
    );
    let mut before_release = Some(before_release);
    if let Some(causal) = causal {
        ensure!(
            activation.writer_protocol >= crate::CAUSAL_WRITER_PROTOCOL
                && activation.causal_commit_available
                && activation.writes_available,
            "pending causal recovery requires writer protocol 2 and causal_commit_v1"
        );
        ensure!(
            causal.operation.has_valid_recovery_fence(),
            "pending causal recovery has an invalid operation identity fence"
        );
        let batch = crate::CausalTranscriptBatch {
            records: causal.transcript_records,
        };
        let receipt = workspace
            .causal_checkpoint(causal.operation.clone(), batch.records.clone())
            .await?;
        transcript
            .acknowledge_causal_transcript(&batch, receipt.transcript_cursor)
            .await?;
        before_release
            .take()
            .expect("recovery release callback is called once")(
            &causal.operation,
            receipt.head,
            receipt.transcript_cursor,
        )?;
        workspace
            .finish_causal_checkpoint(&causal.operation.operation_id, receipt.transcript_cursor)
            .await?;
    } else if let Some(compatibility) = compatibility {
        ensure!(
            activation.writer_protocol == 1 && activation.writes_available,
            "pending compatibility recovery requires the protocol-1 transcript fallback"
        );
        ensure!(
            compatibility.operation.has_valid_recovery_fence(),
            "pending compatibility recovery has an invalid operation identity fence"
        );
        workspace
            .checkpoint_for_compatibility_transcript(compatibility.operation.clone())
            .await?;
        let cursor = transcript
            .flush_compatibility_transcript(&compatibility.operation)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("compatibility transcript flush returned no acknowledgement cursor")
            })?;
        let revision = workspace.status().await.last_committed_revision;
        before_release
            .take()
            .expect("recovery release callback is called once")(
            &compatibility.operation,
            revision,
            cursor,
        )?;
        workspace
            .finish_compatibility_checkpoint(&compatibility.operation.operation_id, cursor)
            .await?;
    }
    let status = workspace.status().await;
    match fs::symlink_metadata(cache.path.join("recovery-required")) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => bail!("remote acknowledgement did not clear the selected recovery cache"),
        Err(error) => return Err(error.into()),
    }
    Ok(PipeFsRecoveryReceipt {
        revision_id: status.last_committed_revision,
        recovered_cache_id: recovery_id.to_owned(),
    })
}

pub async fn detach_if_clean(
    client: &PipeFsClient,
    scope: &PipeFsCacheScope,
    session_id: &str,
    lease: &PipeFsLease,
) -> Result<()> {
    ensure!(
        list_recovery_caches(scope, session_id)?.is_empty(),
        "PipeFS detach refused because local recovery evidence exists"
    );
    let before = client.state(session_id).await?;
    ensure!(before.enabled, "PipeFS is already detached");
    client.set_enabled(session_id, lease, false).await?;
    let after = client.state(session_id).await?;
    if after.enabled {
        bail!("PipeFS detach was not acknowledged by the server");
    }
    crate::record_local_mode_hint(scope, session_id, false)
}

fn preview_from_snapshot(source: PathBuf, snapshot: &Snapshot) -> Result<ImportPreview> {
    let bytes = serde_json::to_vec(snapshot).context("encoding PipeFS import preview")?;
    let mut digest = blake3::Hasher::new_derive_key("hi.pipefs.import-preview.v1");
    digest.update(PIPEFS_SCANNER_VERSION.as_bytes());
    digest.update(&[0]);
    digest.update(&bytes);
    Ok(ImportPreview {
        schema_version: IMPORT_PREVIEW_SCHEMA_VERSION,
        scanner_version: PIPEFS_SCANNER_VERSION.into(),
        source,
        confirmation_digest: format!("blake3:{}", digest.finalize().to_hex()),
        entry_count: snapshot.entries.len(),
        byte_count: snapshot.logical_size_bytes,
        exclusions: Vec::new(),
        unsupported_entries: Vec::new(),
    })
}

fn confirmed_import_preview(source: &Path, confirmation: &str) -> Result<ImportPreview> {
    let preview = preview_import(source)?;
    ensure!(
        preview.unsupported_entries.is_empty(),
        "PipeFS import contains unsupported entries: {}",
        preview.unsupported_entries.join(", ")
    );
    ensure!(
        preview.confirmation_digest == confirmation,
        "PipeFS import confirmation digest does not match a fresh preview"
    );
    Ok(preview)
}

#[derive(Default)]
struct ImportInventory {
    entry_count: usize,
    byte_count: u64,
    unsupported: Vec<String>,
}

fn import_inventory(root: &Path) -> Result<ImportInventory> {
    fn visit(root: &Path, directory: &Path, inventory: &mut ImportInventory) -> Result<()> {
        let mut entries = fs::read_dir(directory)
            .with_context(|| format!("reading import directory {}", directory.display()))?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            inventory.entry_count = inventory.entry_count.saturating_add(1);
            let relative = path.strip_prefix(root).expect("walk remains below root");
            let label = relative.to_string_lossy().into_owned();
            if relative.to_str().is_none() {
                inventory
                    .unsupported
                    .push(format!("{label} (non-UTF-8 path)"));
                continue;
            }
            let kind = metadata.file_type();
            if kind.is_file() {
                inventory.byte_count = inventory.byte_count.saturating_add(metadata.len());
            } else if kind.is_dir() {
                visit(root, &path, inventory)?;
            } else if !kind.is_symlink() {
                inventory
                    .unsupported
                    .push(format!("{label} (special file)"));
            }
        }
        Ok(())
    }

    let mut inventory = ImportInventory::default();
    visit(root, root, &mut inventory)?;
    inventory.unsupported.sort();
    Ok(inventory)
}

#[cfg(test)]
#[path = "migration_tests.rs"]
mod tests;
