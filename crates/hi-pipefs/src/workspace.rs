use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    ARCHIVE_VERSION, ArchiveArtifact, PipeFsCapabilities, PipeFsClient, PipeFsError, PipeFsLease,
    PipeFsRemoteState, RevisionKind, Snapshot, apply_archive, build_revision, scan_workspace,
};

#[derive(Clone, Debug)]
pub struct PipeFsWorkspaceConfig {
    pub session_id: String,
    pub original_workspace_root: PathBuf,
    pub original_state_root: PathBuf,
    pub cache_base: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct Activation {
    pub workspace_root: PathBuf,
    pub state_root: PathBuf,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspacePhase {
    Disabled,
    Restoring,
    Clean,
    Dirty,
    Pending,
    LeaseLost,
}

#[derive(Clone, Debug)]
pub struct PipeFsStatus {
    pub phase: WorkspacePhase,
    pub enabled: bool,
    pub workspace_root: Option<PathBuf>,
    pub dirty_paths: Vec<String>,
    pub retry_count: u32,
    pub last_committed_revision: Option<Uuid>,
    pub last_error: Option<String>,
    pub recovery_caches: Vec<PathBuf>,
}

impl std::fmt::Display for PipeFsStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            formatter,
            "PipeFS: {} ({:?})",
            if self.enabled { "on" } else { "off" },
            self.phase
        )?;
        if let Some(root) = &self.workspace_root {
            writeln!(formatter, "workspace: {}", root.display())?;
        }
        writeln!(
            formatter,
            "last committed revision: {}",
            self.last_committed_revision
                .map_or_else(|| "empty".to_string(), |revision| revision.to_string())
        )?;
        if !self.dirty_paths.is_empty() {
            writeln!(formatter, "dirty paths: {}", self.dirty_paths.join(", "))?;
        }
        if self.retry_count > 0 {
            writeln!(formatter, "retry count: {}", self.retry_count)?;
        }
        if let Some(error) = &self.last_error {
            writeln!(formatter, "last persistence error: {error}")?;
            writeln!(formatter, "action: run /pipefs retry")?;
        }
        if !self.recovery_caches.is_empty() {
            writeln!(
                formatter,
                "recovery caches from older lease generations: {}",
                self.recovery_caches
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct PipeFsWorkspace {
    inner: Arc<Inner>,
}

struct Inner {
    client: PipeFsClient,
    session_id: String,
    lease: Mutex<PipeFsLease>,
    original_workspace_root: PathBuf,
    original_state_root: PathBuf,
    session_cache_root: PathBuf,
    mode_hint_file: PathBuf,
    cache_root: PathBuf,
    runtime_state_root: PathBuf,
    state_file: PathBuf,
    recovery_marker: PathBuf,
    pending_archive: PathBuf,
    state: Mutex<ControllerState>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ControllerState {
    phase: WorkspacePhase,
    remote: Option<PipeFsRemoteStateDisk>,
    snapshot: Option<Snapshot>,
    pending: Option<PendingRevision>,
    materialized_root: Option<PathBuf>,
    dirty_paths: BTreeSet<String>,
    #[serde(default)]
    active_background_processes: BTreeSet<String>,
    retry_count: u32,
    last_error: Option<String>,
    capabilities: Option<CapabilitiesDisk>,
}

impl Default for ControllerState {
    fn default() -> Self {
        Self {
            phase: WorkspacePhase::Disabled,
            remote: None,
            snapshot: None,
            pending: None,
            materialized_root: None,
            dirty_paths: BTreeSet::new(),
            active_background_processes: BTreeSet::new(),
            retry_count: 0,
            last_error: None,
            capabilities: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PipeFsRemoteStateDisk {
    enabled: bool,
    current_head: Option<Uuid>,
    sequence: u64,
    manifest_digest: Option<String>,
    logical_size_bytes: u64,
    restore_chain: Vec<RestoreRevisionDisk>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RestoreRevisionDisk {
    revision_id: Uuid,
    base_revision_id: Option<Uuid>,
    revision_type: RevisionKind,
    artifact_size_bytes: u64,
}

impl From<&PipeFsRemoteState> for PipeFsRemoteStateDisk {
    fn from(value: &PipeFsRemoteState) -> Self {
        Self {
            enabled: value.enabled,
            current_head: value.current_head,
            sequence: value.sequence,
            manifest_digest: value.manifest_digest.clone(),
            logical_size_bytes: value.logical_size_bytes,
            restore_chain: value
                .restore_chain
                .iter()
                .map(|revision| RestoreRevisionDisk {
                    revision_id: revision.revision_id,
                    base_revision_id: revision.base_revision_id,
                    revision_type: revision.revision_type,
                    artifact_size_bytes: revision.artifact.size_bytes,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CapabilitiesDisk {
    maximum_revision_bytes: u64,
    maximum_workspace_bytes: u64,
    maximum_delta_chain: u32,
}

impl From<&PipeFsCapabilities> for CapabilitiesDisk {
    fn from(value: &PipeFsCapabilities) -> Self {
        Self {
            maximum_revision_bytes: value.maximum_revision_bytes,
            maximum_workspace_bytes: value.maximum_workspace_bytes,
            maximum_delta_chain: value.maximum_delta_chain,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PendingRevision {
    expected_base_revision_id: Option<Uuid>,
    revision_type: RevisionKind,
    archive_blake3: String,
    manifest_digest: String,
    logical_size_bytes: u64,
    idempotency_key: String,
    snapshot: Snapshot,
}

impl PipeFsWorkspace {
    pub fn new(
        client: PipeFsClient,
        lease: PipeFsLease,
        config: PipeFsWorkspaceConfig,
    ) -> Result<Self> {
        validate_session_id(&config.session_id)?;
        let original_workspace_root =
            config
                .original_workspace_root
                .canonicalize()
                .with_context(|| {
                    format!(
                        "canonicalizing original workspace {}",
                        config.original_workspace_root.display()
                    )
                })?;
        let original_state_root = absolute_path(&config.original_state_root)?;
        let base = config.cache_base.unwrap_or_else(default_cache_base);
        let session_digest = blake3::hash(config.session_id.as_bytes())
            .to_hex()
            .to_string();
        let generation_digest =
            blake3::hash(format!("{}\0{}", config.session_id, lease.generation).as_bytes())
                .to_hex()
                .to_string();
        let session_cache_root = base.join(&session_digest[..32]);
        create_private_dir(&session_cache_root)?;
        let mode_hint_file = session_cache_root.join("remote-mode");
        let generation_cache_root = session_cache_root.join(&generation_digest[..32]);
        let generation_needs_recovery = mark_cache_for_recovery_if_drifted(
            &generation_cache_root,
            &generation_cache_root.join("recovery-required"),
        )?;
        // A crashed process can reacquire the same lease generation (for
        // example a PID-1 container restart). Never reuse and overwrite a
        // marked generation directory: activate from a fresh sibling so the
        // existing bytes are discovered by the normal recovery path.
        let cache_root = if generation_needs_recovery {
            session_cache_root.join(format!(
                "{}-resume-{}",
                &generation_digest[..32],
                Uuid::new_v4().simple()
            ))
        } else {
            generation_cache_root
        };
        create_private_dir(&cache_root)?;
        let runtime_state_root = cache_root.join("runtime-state");
        create_private_dir(&runtime_state_root)?;
        let state_file = cache_root.join("controller.json");
        let recovery_marker = cache_root.join("recovery-required");
        let pending_archive = cache_root.join("pending.tar.zst");
        let state = if state_file.exists() {
            serde_json::from_slice(&fs::read(&state_file)?)
                .context("reading existing PipeFS controller state")?
        } else {
            ControllerState::default()
        };
        Ok(Self {
            inner: Arc::new(Inner {
                client,
                session_id: config.session_id,
                lease: Mutex::new(lease),
                original_workspace_root,
                original_state_root,
                session_cache_root,
                mode_hint_file,
                cache_root,
                runtime_state_root,
                state_file,
                recovery_marker,
                pending_archive,
                state: Mutex::new(state),
            }),
        })
    }

    /// Refresh the shared HI writer lease used by subsequent control-plane
    /// operations. A lease generation can advance when the same client
    /// reacquires an expired lease. Any archive staged under the previous
    /// generation is no longer committable, so discard only that derived
    /// archive and reconcile the still-materialized workspace again.
    pub async fn update_lease(&self, lease: PipeFsLease) -> Result<()> {
        ensure!(
            lease.generation > 0,
            "the PipeFS writer lease has no generation"
        );
        let mut state = self.inner.state.lock().await;
        let mut current = self.inner.lease.lock().await;
        ensure!(
            lease.generation >= current.generation,
            "refusing to replace PipeFS lease generation {} with stale generation {}",
            current.generation,
            lease.generation
        );
        if lease.generation == current.generation {
            *current = lease;
            return Ok(());
        }
        ensure!(
            state.phase != WorkspacePhase::LeaseLost,
            "PipeFS writer lease was already declared lost; reactivate the session before mutating"
        );
        if state.phase == WorkspacePhase::Pending {
            state.pending = None;
            state.phase = WorkspacePhase::Dirty;
            state.last_error = None;
            write_private(
                &self.inner.recovery_marker,
                b"uncommitted workspace changes\n",
            )?;
            self.persist_locked(&state)?;
            let _ = fs::remove_file(&self.inner.pending_archive);
        }
        *current = lease;
        Ok(())
    }

    pub async fn enable(&self) -> Result<Activation> {
        let capabilities = self.inner.client.capabilities().await?;
        ensure!(
            capabilities.enabled,
            "PipeFS is disabled on this IPOP server"
        );
        ensure!(
            capabilities.archive_version == ARCHIVE_VERSION,
            "unsupported PipeFS archive version {} (client supports {})",
            capabilities.archive_version,
            ARCHIVE_VERSION
        );
        ensure!(
            capabilities
                .transfer_modes
                .iter()
                .any(|mode| mode == "proxy" || mode == "presigned"),
            "server advertises no supported PipeFS transfer mode"
        );
        let lease = self.inner.lease.lock().await.clone();
        let mut remote = self
            .inner
            .client
            .set_enabled(&self.inner.session_id, &lease, true)
            .await?;
        self.record_mode_hint(true)?;
        self.cleanup_stale_clean_caches(&remote);

        {
            let mut state = self.inner.state.lock().await;
            state.phase = WorkspacePhase::Restoring;
            state.capabilities = Some((&capabilities).into());
            state.remote = Some((&remote).into());
            state.last_error = None;
            self.persist_locked(&state)?;
        }

        let staging = self
            .inner
            .cache_root
            .join(format!("restore-{}", Uuid::new_v4().simple()));
        create_private_dir(&staging)?;
        let restore_result = self
            .restore_into(&staging, &remote, capabilities.maximum_revision_bytes)
            .await;
        let snapshot = match restore_result {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let _ = fs::remove_dir_all(&staging);
                let mut state = self.inner.state.lock().await;
                state.phase = WorkspacePhase::Disabled;
                state.last_error = Some(format!("restore failed: {error:#}"));
                self.persist_locked(&state)?;
                return Err(error);
            }
        };
        let (staging, snapshot, recovered_cache) = match self
            .recover_local_cache(staging.clone(), snapshot, &mut remote)
            .await
        {
            Ok(recovered) => recovered,
            Err(error) => {
                let _ = remove_cache_directory(&self.inner.cache_root, &staging);
                let mut state = self.inner.state.lock().await;
                state.phase = WorkspacePhase::Disabled;
                state.snapshot = None;
                state.pending = None;
                state.materialized_root = None;
                state.dirty_paths.clear();
                state.last_error = Some(format!("recovery failed: {error:#}"));
                self.persist_locked(&state)?;
                let _ = fs::remove_file(&self.inner.pending_archive);
                let _ = fs::remove_file(&self.inner.recovery_marker);
                return Err(error);
            }
        };
        let materialized = self
            .inner
            .cache_root
            .join(format!("workspace-{}", Uuid::new_v4().simple()));
        fs::rename(&staging, &materialized).with_context(|| {
            format!(
                "atomically activating PipeFS workspace {}",
                materialized.display()
            )
        })?;
        let old_materialized = {
            let mut state = self.inner.state.lock().await;
            let old = state.materialized_root.replace(materialized.clone());
            state.snapshot = Some(snapshot);
            state.pending = None;
            state.phase = WorkspacePhase::Clean;
            state.remote = Some((&remote).into());
            state.dirty_paths.clear();
            state.retry_count = 0;
            state.last_error = None;
            self.persist_locked(&state)?;
            old
        };
        if let Some(old) = old_materialized.filter(|old| *old != materialized) {
            remove_cache_directory(&self.inner.cache_root, &old)?;
        }
        let _ = fs::remove_file(&self.inner.recovery_marker);
        if let Some(recovered_cache) = recovered_cache {
            // The new generation is active and its remote head represents the
            // recovered bytes. The old marker is no longer actionable even if
            // best-effort directory cleanup is interrupted.
            let _ = fs::remove_file(recovered_cache.join("recovery-required"));
            let _ = remove_cache_directory(&self.inner.session_cache_root, &recovered_cache);
        }
        Ok(Activation {
            workspace_root: materialized,
            state_root: self.inner.runtime_state_root.clone(),
        })
    }

    async fn recover_local_cache(
        &self,
        staging: PathBuf,
        restored_snapshot: Snapshot,
        remote: &mut PipeFsRemoteState,
    ) -> Result<(PathBuf, Snapshot, Option<PathBuf>)> {
        let candidates =
            recovery_caches(&self.inner.session_cache_root, Some(&self.inner.cache_root));
        if candidates.is_empty() {
            return Ok((staging, restored_snapshot, None));
        }
        ensure!(
            candidates.len() == 1,
            "recovery_conflict: multiple dirty PipeFS caches exist for this session: {}",
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let candidate = candidates.into_iter().next().expect("checked non-empty");
        let old_state: ControllerState = serde_json::from_slice(
            &fs::read(candidate.join("controller.json"))
                .with_context(|| format!("reading recovery state from {}", candidate.display()))?,
        )
        .with_context(|| format!("parsing recovery state from {}", candidate.display()))?;
        let old_root = old_state
            .materialized_root
            .clone()
            .ok_or_else(|| anyhow!("recovery cache has no materialized workspace"))?;
        ensure!(
            old_root.starts_with(&candidate) && old_root != candidate,
            "recovery cache points outside its generation directory"
        );
        let metadata = fs::symlink_metadata(&old_root)
            .with_context(|| format!("reading recovery workspace {}", old_root.display()))?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "recovery workspace is not a real directory"
        );
        let scan_root = old_root.clone();
        let recovered_snapshot = tokio::task::spawn_blocking(move || scan_workspace(&scan_root))
            .await
            .context("PipeFS recovery scan task panicked")??;

        if recovered_snapshot.entries == restored_snapshot.entries {
            return Ok((staging, restored_snapshot, Some(candidate)));
        }
        let expected_base = old_state
            .pending
            .as_ref()
            .map(|pending| pending.expected_base_revision_id)
            .or_else(|| old_state.remote.as_ref().map(|state| state.current_head))
            .ok_or_else(|| anyhow!("recovery cache has no recorded remote base"))?;
        ensure!(
            expected_base == remote.current_head,
            "recovery_conflict: dirty cache was based on {}, but the remote head is {}; the cache was preserved for manual reconciliation",
            expected_base.map_or_else(|| "empty".to_string(), |id| id.to_string()),
            remote
                .current_head
                .map_or_else(|| "empty".to_string(), |id| id.to_string())
        );

        let force_full = {
            let state = self.inner.state.lock().await;
            must_write_full(&state)
        };
        let build_root = old_root.clone();
        let base = restored_snapshot.clone();
        let mut artifact = tokio::task::spawn_blocking(move || {
            build_revision(&build_root, Some(&base), force_full)
        })
        .await
        .context("PipeFS recovery archive task panicked")??;
        {
            let state = self.inner.state.lock().await;
            if artifact.manifest.revision_type == RevisionKind::Delta
                && cumulative_delta_would_exceed_full(&state, artifact.bytes.len() as u64)
            {
                drop(state);
                let build_root = old_root.clone();
                artifact =
                    tokio::task::spawn_blocking(move || build_revision(&build_root, None, true))
                        .await
                        .context("PipeFS recovery compaction task panicked")??;
            }
        }

        // Materialize the recovered bytes independently from the remote
        // staging tree. A full archive validates the complete local tree and
        // avoids filesystem-dependent recursive copy behavior.
        let recovered_staging = self
            .inner
            .cache_root
            .join(format!("recovery-{}", Uuid::new_v4().simple()));
        create_private_dir(&recovered_staging)?;
        let full_for_restore = if artifact.manifest.revision_type == RevisionKind::Full {
            artifact.clone()
        } else {
            let build_root = old_root;
            tokio::task::spawn_blocking(move || build_revision(&build_root, None, true))
                .await
                .context("PipeFS full recovery archive task panicked")??
        };
        let restore_root = recovered_staging.clone();
        let restore_bytes = full_for_restore.bytes;
        let mut materialized_snapshot =
            tokio::task::spawn_blocking(move || apply_archive(&restore_root, &restore_bytes, None))
                .await
                .context("PipeFS local recovery extraction task panicked")??;

        let commit_result = {
            let mut state = self.inner.state.lock().await;
            state.snapshot = Some(restored_snapshot);
            state.materialized_root = Some(recovered_staging.clone());
            state.phase = WorkspacePhase::Dirty;
            state
                .dirty_paths
                .insert("<recovered interrupted cache>".to_string());
            write_private(
                &self.inner.recovery_marker,
                b"recovering interrupted workspace changes\n",
            )?;
            self.persist_locked(&state)?;
            let generation = self.inner.lease.lock().await.generation;
            self.stage_pending_locked(&mut state, artifact, generation)?;
            self.retry_locked(&mut state).await
        };
        let committed_remote = match commit_result {
            Ok((_, committed_remote)) => committed_remote,
            Err(error) => {
                let mut state = self.inner.state.lock().await;
                state.phase = WorkspacePhase::Disabled;
                state.snapshot = None;
                state.pending = None;
                state.materialized_root = None;
                state.dirty_paths.clear();
                state.last_error = Some(format!("local recovery failed: {error:#}"));
                self.persist_locked(&state)?;
                let _ = fs::remove_file(&self.inner.pending_archive);
                let _ = fs::remove_file(&self.inner.recovery_marker);
                let _ = remove_cache_directory(&self.inner.cache_root, &recovered_staging);
                return Err(error.context(
                    "persisting the interrupted PipeFS cache; the older recovery cache was retained",
                ));
            }
        };
        materialized_snapshot.manifest_digest = committed_remote.manifest_digest.clone();
        *remote = committed_remote;
        let _ = remove_cache_directory(&self.inner.cache_root, &staging);
        Ok((recovered_staging, materialized_snapshot, Some(candidate)))
    }

    async fn restore_into(
        &self,
        staging: &Path,
        remote: &PipeFsRemoteState,
        maximum_revision_bytes: u64,
    ) -> Result<Snapshot> {
        if remote.current_head.is_none() {
            ensure!(
                remote.restore_chain.is_empty(),
                "missing_revision: empty head has a restore chain"
            );
            return scan_workspace(staging);
        }
        ensure!(
            !remote.restore_chain.is_empty(),
            "missing_revision: head has no restore chain"
        );
        let mut prior_id = None;
        let mut prior_sequence = None;
        let mut snapshot = None;
        for (index, revision) in remote.restore_chain.iter().enumerate() {
            ensure!(
                revision.base_revision_id == prior_id,
                "missing_revision: restore chain base mismatch at {}",
                revision.revision_id
            );
            ensure!(
                prior_sequence.is_none_or(|sequence| revision.sequence == sequence + 1),
                "missing_revision: restore chain sequence is not contiguous"
            );
            ensure!(
                (index == 0 && revision.revision_type == RevisionKind::Full)
                    || (index > 0 && revision.revision_type == RevisionKind::Delta),
                "missing_revision: restore chain must be one full revision followed by deltas"
            );
            let bytes = self
                .inner
                .client
                .download_revision(&self.inner.session_id, revision, maximum_revision_bytes)
                .await
                .map_err(classified_restore_error)?;
            let restored = apply_archive(staging, &bytes, snapshot.as_ref())
                .with_context(|| format!("restoring PipeFS revision {}", revision.revision_id))?;
            ensure!(
                restored.manifest_digest.as_deref() == Some(&revision.manifest_digest),
                "corruption_error: manifest digest mismatch for revision {}",
                revision.revision_id
            );
            ensure!(
                restored.logical_size_bytes == revision.logical_size_bytes,
                "corruption_error: logical size mismatch for revision {}",
                revision.revision_id
            );
            snapshot = Some(restored);
            prior_id = Some(revision.revision_id);
            prior_sequence = Some(revision.sequence);
        }
        ensure!(
            prior_id == remote.current_head,
            "missing_revision: restore chain does not end at head"
        );
        ensure!(
            prior_sequence == Some(remote.sequence),
            "missing_revision: restore chain sequence does not match workspace head"
        );
        ensure!(
            snapshot
                .as_ref()
                .and_then(|value| value.manifest_digest.as_ref())
                == remote.manifest_digest.as_ref(),
            "corruption_error: restored manifest does not match workspace head"
        );
        ensure!(
            snapshot.as_ref().map(|value| value.logical_size_bytes)
                == Some(remote.logical_size_bytes),
            "corruption_error: restored logical size does not match workspace head"
        );
        Ok(snapshot.expect("non-empty restore chain has a snapshot"))
    }

    pub async fn mutation_started(&self, paths: Option<Vec<String>>) -> Result<()> {
        let mut state = self.inner.state.lock().await;
        match state.phase {
            WorkspacePhase::Clean => {}
            WorkspacePhase::Dirty if state.last_error.is_none() => {}
            WorkspacePhase::Dirty => bail!(
                "PipeFS persistence failed: {}. Run /pipefs retry before making more changes",
                state.last_error.as_deref().unwrap_or("unknown error")
            ),
            WorkspacePhase::Pending => bail!(
                "PipeFS has an uncommitted revision; run /pipefs retry before making more changes"
            ),
            WorkspacePhase::LeaseLost => {
                bail!("PipeFS writer lease was replaced; this process cannot accept mutations")
            }
            _ => bail!("PipeFS workspace is not active"),
        }
        state.phase = WorkspacePhase::Dirty;
        if let Some(paths) = paths {
            state.dirty_paths.extend(paths);
        } else {
            state
                .dirty_paths
                .insert("<opaque process changes>".to_string());
        }
        write_private(
            &self.inner.recovery_marker,
            b"uncommitted workspace changes\n",
        )?;
        self.persist_locked(&state)
    }

    pub async fn mutation_allowed(&self) -> Result<()> {
        let state = self.inner.state.lock().await;
        match state.phase {
            WorkspacePhase::Clean => Ok(()),
            WorkspacePhase::Dirty if state.last_error.is_none() => Ok(()),
            WorkspacePhase::Dirty => bail!(
                "PipeFS persistence failed after {} attempt(s): {}. Run /pipefs retry",
                state.retry_count,
                state.last_error.as_deref().unwrap_or("unknown error")
            ),
            WorkspacePhase::Pending => bail!(
                "PipeFS persistence is pending after {} attempt(s): {}. Run /pipefs retry",
                state.retry_count,
                state.last_error.as_deref().unwrap_or("unknown error")
            ),
            WorkspacePhase::LeaseLost => bail!("PipeFS writer lease is stale"),
            _ => bail!("PipeFS workspace is not active"),
        }
    }

    pub async fn mark_lease_lost(&self, detail: impl Into<String>) -> Result<()> {
        let mut state = self.inner.state.lock().await;
        if matches!(
            state.phase,
            WorkspacePhase::Clean | WorkspacePhase::Dirty | WorkspacePhase::Pending
        ) {
            state.phase = WorkspacePhase::LeaseLost;
            state.last_error = Some(detail.into());
            self.persist_locked(&state)?;
        }
        Ok(())
    }

    /// Keep the recovery marker present for the entire lifetime of a native
    /// background process. Even immediately after a successful checkpoint the
    /// process can write again, so a forced exit cannot classify this cache as
    /// clean until terminal completion has been observed.
    pub async fn background_process_state(&self, id: &str, running: bool) -> Result<()> {
        let mut state = self.inner.state.lock().await;
        if running {
            state.active_background_processes.insert(id.to_string());
            write_private(
                &self.inner.recovery_marker,
                b"background process may have uncommitted workspace changes\n",
            )?;
        } else {
            state.active_background_processes.remove(id);
        }
        self.persist_locked(&state)?;
        if !running {
            self.clear_recovery_marker_if_safe(&state);
        }
        Ok(())
    }

    pub async fn checkpoint(&self) -> Result<Option<Uuid>> {
        let mut state = self.inner.state.lock().await;
        if state.phase == WorkspacePhase::Pending {
            return self
                .retry_locked(&mut state)
                .await
                .map(|(head, _)| Some(head));
        }
        ensure!(
            state.phase == WorkspacePhase::Dirty || state.phase == WorkspacePhase::Clean,
            "PipeFS workspace is not checkpointable"
        );
        // Always reconcile the materialized tree. An opaque command or a
        // background process can modify it after the last reported mutation,
        // and shutdown must not discard those bytes merely because the
        // in-memory ledger currently says `Clean`.
        let root = state
            .materialized_root
            .clone()
            .ok_or_else(|| anyhow!("PipeFS materialized root is missing"))?;
        let prior = state.snapshot.clone();
        let root_for_scan = root.clone();
        let current =
            match tokio::task::spawn_blocking(move || scan_workspace(&root_for_scan)).await {
                Ok(Ok(current)) => current,
                Ok(Err(error)) => {
                    self.record_checkpoint_failure_locked(&mut state, &error)?;
                    return Err(error);
                }
                Err(join_error) => {
                    let error = anyhow!("PipeFS scan task panicked: {join_error}");
                    self.record_checkpoint_failure_locked(&mut state, &error)?;
                    return Err(error);
                }
            };
        if prior
            .as_ref()
            .is_some_and(|prior| prior.entries == current.entries)
        {
            state.snapshot = Some(current);
            state.phase = WorkspacePhase::Clean;
            state.dirty_paths.clear();
            state.last_error = None;
            self.persist_locked(&state)?;
            self.clear_recovery_marker_if_safe(&state);
            return Ok(state.remote.as_ref().and_then(|remote| remote.current_head));
        }

        let force_full = must_write_full(&state);
        let root_for_build = root;
        let base_for_build = prior.clone();
        let mut artifact = match tokio::task::spawn_blocking(move || {
            build_revision(&root_for_build, base_for_build.as_ref(), force_full)
        })
        .await
        {
            Ok(Ok(artifact)) => artifact,
            Ok(Err(error)) => {
                self.record_checkpoint_failure_locked(&mut state, &error)?;
                return Err(error);
            }
            Err(join_error) => {
                let error = anyhow!("PipeFS archive task panicked: {join_error}");
                self.record_checkpoint_failure_locked(&mut state, &error)?;
                return Err(error);
            }
        };
        if artifact.manifest.revision_type == RevisionKind::Delta
            && cumulative_delta_would_exceed_full(&state, artifact.bytes.len() as u64)
        {
            let root_for_build = state.materialized_root.clone().expect("checked above");
            artifact = match tokio::task::spawn_blocking(move || {
                build_revision(&root_for_build, None, true)
            })
            .await
            {
                Ok(Ok(artifact)) => artifact,
                Ok(Err(error)) => {
                    self.record_checkpoint_failure_locked(&mut state, &error)?;
                    return Err(error);
                }
                Err(join_error) => {
                    let error = anyhow!("PipeFS compaction archive task panicked: {join_error}");
                    self.record_checkpoint_failure_locked(&mut state, &error)?;
                    return Err(error);
                }
            };
        }
        let lease_generation = self.inner.lease.lock().await.generation;
        if let Err(error) = self.stage_pending_locked(&mut state, artifact, lease_generation) {
            self.record_checkpoint_failure_locked(&mut state, &error)?;
            return Err(error);
        }
        self.retry_locked(&mut state)
            .await
            .map(|(head, _)| Some(head))
    }

    fn record_checkpoint_failure_locked(
        &self,
        state: &mut ControllerState,
        error: &anyhow::Error,
    ) -> Result<()> {
        state.phase = WorkspacePhase::Dirty;
        state.retry_count = state.retry_count.saturating_add(1);
        state.last_error = Some(error.to_string());
        write_private(
            &self.inner.recovery_marker,
            b"uncommitted workspace changes\n",
        )?;
        self.persist_locked(state)
    }

    fn stage_pending_locked(
        &self,
        state: &mut ControllerState,
        artifact: ArchiveArtifact,
        lease_generation: u64,
    ) -> Result<()> {
        let capabilities = state
            .capabilities
            .as_ref()
            .ok_or_else(|| anyhow!("PipeFS capabilities are unavailable"))?;
        ensure!(
            artifact.bytes.len() as u64 <= capabilities.maximum_revision_bytes,
            "PipeFS revision is {} bytes, exceeding the server limit of {}",
            artifact.bytes.len(),
            capabilities.maximum_revision_bytes
        );
        ensure!(
            artifact.snapshot.logical_size_bytes <= capabilities.maximum_workspace_bytes,
            "PipeFS workspace is {} bytes, exceeding the server limit of {}",
            artifact.snapshot.logical_size_bytes,
            capabilities.maximum_workspace_bytes
        );
        write_private(&self.inner.pending_archive, &artifact.bytes)?;
        let expected_base_revision_id =
            state.remote.as_ref().and_then(|remote| remote.current_head);
        let idempotency_key = revision_idempotency_key(
            lease_generation,
            expected_base_revision_id,
            artifact.manifest.revision_type,
            &artifact.blake3,
        );
        state.pending = Some(PendingRevision {
            expected_base_revision_id,
            revision_type: artifact.manifest.revision_type,
            archive_blake3: artifact.blake3,
            manifest_digest: artifact.manifest_digest,
            logical_size_bytes: artifact.snapshot.logical_size_bytes,
            idempotency_key,
            snapshot: artifact.snapshot,
        });
        state.phase = WorkspacePhase::Pending;
        state.last_error = None;
        self.persist_locked(state)
    }

    async fn retry_locked(&self, state: &mut ControllerState) -> Result<(Uuid, PipeFsRemoteState)> {
        let pending = state
            .pending
            .clone()
            .ok_or_else(|| anyhow!("PipeFS has no staged revision to retry"))?;
        let bytes = fs::read(&self.inner.pending_archive)
            .context("reading staged PipeFS revision for retry")?;
        ensure!(
            blake3::hash(&bytes).to_hex().as_str() == pending.archive_blake3,
            "local recovery archive failed BLAKE3 verification"
        );
        state.retry_count = state.retry_count.saturating_add(1);
        self.persist_locked(state)?;
        let lease = self.inner.lease.lock().await.clone();
        let result = self
            .inner
            .client
            .commit_archive(
                &self.inner.session_id,
                &lease,
                pending.expected_base_revision_id,
                pending.revision_type,
                &bytes,
                &pending.archive_blake3,
                &pending.manifest_digest,
                pending.logical_size_bytes,
                &pending.idempotency_key,
            )
            .await;
        match result {
            Ok(remote) => {
                let head = remote.current_head.ok_or_else(|| {
                    anyhow!("PipeFS server committed a revision without returning a head")
                })?;
                ensure!(
                    remote.manifest_digest.as_deref() == Some(&pending.manifest_digest),
                    "PipeFS server head manifest does not match committed revision"
                );
                state.remote = Some((&remote).into());
                state.snapshot = Some(pending.snapshot);
                state.pending = None;
                state.phase = WorkspacePhase::Clean;
                state.dirty_paths.clear();
                state.last_error = None;
                self.persist_locked(state)?;
                let _ = fs::remove_file(&self.inner.pending_archive);
                self.clear_recovery_marker_if_safe(state);
                Ok((head, remote))
            }
            Err(error)
                if pending.revision_type == RevisionKind::Delta
                    && is_compaction_required(&error) =>
            {
                // The server has accepted neither the delta nor its artifact
                // as a head candidate. Rebuild a complete archive from the
                // live materialization, not from the prior snapshot: this
                // produces the exact current tree needed to reset a bounded
                // restore chain and replaces the rejected staged delta with a
                // fresh, deterministic idempotency identity.
                match self
                    .restage_full_after_compaction_locked(state, lease.generation)
                    .await
                {
                    // A full revision cannot take this delta-only recovery
                    // branch again. Box the one bounded retry so the async
                    // state machine remains finite.
                    Ok(()) => Box::pin(self.retry_locked(state)).await,
                    Err(restage_error) => {
                        state.phase = WorkspacePhase::Pending;
                        state.last_error = Some(format!(
                            "PipeFS server requires compaction, but building a full revision failed: {restage_error:#}"
                        ));
                        self.persist_locked(state)?;
                        Err(restage_error)
                    }
                }
            }
            Err(error) => {
                if matches!(error, PipeFsError::LeaseLost(_)) {
                    state.phase = WorkspacePhase::LeaseLost;
                } else {
                    state.phase = WorkspacePhase::Pending;
                }
                state.last_error = Some(error.to_string());
                self.persist_locked(state)?;
                Err(error.into())
            }
        }
    }

    async fn restage_full_after_compaction_locked(
        &self,
        state: &mut ControllerState,
        lease_generation: u64,
    ) -> Result<()> {
        let root = state
            .materialized_root
            .clone()
            .ok_or_else(|| anyhow!("PipeFS materialized root is missing for compaction"))?;
        let artifact = tokio::task::spawn_blocking(move || build_revision(&root, None, true))
            .await
            .context("PipeFS server-requested compaction archive task panicked")??;
        ensure!(
            artifact.manifest.revision_type == RevisionKind::Full,
            "PipeFS compaction did not produce a full revision"
        );
        self.stage_pending_locked(state, artifact, lease_generation)
    }

    pub async fn retry(&self) -> Result<Option<Uuid>> {
        let phase = self.inner.state.lock().await.phase;
        match phase {
            WorkspacePhase::Dirty => self.checkpoint().await,
            WorkspacePhase::Pending => {
                let mut state = self.inner.state.lock().await;
                self.retry_locked(&mut state)
                    .await
                    .map(|(head, _)| Some(head))
            }
            WorkspacePhase::Clean => Ok(self
                .inner
                .state
                .lock()
                .await
                .remote
                .as_ref()
                .and_then(|remote| remote.current_head)),
            WorkspacePhase::LeaseLost => bail!(
                "PipeFS writer lease is stale; explicitly take over the session before retrying"
            ),
            _ => bail!("PipeFS workspace is not active"),
        }
    }

    /// Persist the final workspace state, then disable the remote mode. The
    /// cache remains until [`finish_disable`](Self::finish_disable) is called
    /// after the agent runtime has rebound to the original local root.
    pub async fn prepare_disable(&self) -> Result<Activation> {
        let status = self.status().await;
        ensure!(status.enabled, "PipeFS is already off");
        self.checkpoint().await?;
        let lease = self.inner.lease.lock().await.clone();
        let remote = self
            .inner
            .client
            .set_enabled(&self.inner.session_id, &lease, false)
            .await?;
        let _ = self.record_mode_hint(false);
        let mut state = self.inner.state.lock().await;
        state.remote = Some((&remote).into());
        state.phase = WorkspacePhase::Disabled;
        self.persist_locked(&state)?;
        Ok(Activation {
            workspace_root: self.inner.original_workspace_root.clone(),
            state_root: self.inner.original_state_root.clone(),
        })
    }

    pub async fn finish_disable(&self) -> Result<()> {
        let state = self.inner.state.lock().await;
        ensure!(
            state.phase == WorkspacePhase::Disabled,
            "PipeFS has not been disabled"
        );
        drop(state);
        remove_cache_directory(&self.inner.session_cache_root, &self.inner.cache_root)
    }

    /// Restore the enabled controller state when the frontend could not rebind
    /// to its original local root after the remote disable transaction. No
    /// workspace bytes changed during that window, so the existing clean
    /// materialization remains authoritative and can safely stay active.
    pub async fn rollback_disable(&self) -> Result<()> {
        let lease = self.inner.lease.lock().await.clone();
        let remote = self
            .inner
            .client
            .set_enabled(&self.inner.session_id, &lease, true)
            .await?;
        self.record_mode_hint(true)?;
        let mut state = self.inner.state.lock().await;
        ensure!(
            state.phase == WorkspacePhase::Disabled,
            "PipeFS disable is not awaiting a frontend rebind"
        );
        state.remote = Some((&remote).into());
        state.phase = WorkspacePhase::Clean;
        state.last_error = None;
        self.persist_locked(&state)
    }

    /// Clean process exit leaves the remote workspace enabled for resume but
    /// removes local bytes only after the final revision is acknowledged.
    pub async fn clean_exit(&self) -> Result<()> {
        self.checkpoint().await?;
        self.finish_clean_exit().await
    }

    /// Remove a cache only after a caller has completed every other durable
    /// shutdown dependency (notably transcript delivery).  Keeping this split
    /// lets the HI host preserve an otherwise clean workspace when session
    /// synchronization cannot be confirmed.
    pub async fn finish_clean_exit(&self) -> Result<()> {
        let state = self.inner.state.lock().await;
        ensure!(
            state.phase == WorkspacePhase::Clean
                && state.pending.is_none()
                && state.last_error.is_none()
                && state.active_background_processes.is_empty(),
            "PipeFS cache is not safe to remove"
        );
        drop(state);
        self.record_mode_hint(true)?;
        remove_cache_directory(&self.inner.session_cache_root, &self.inner.cache_root)
    }

    /// Record a conservative, session-level mode hint outside generation
    /// caches. It survives a clean cache removal so a later local resume still
    /// knows that it must ask IPOP before activating the launch directory.
    /// This is a safety hint, not authority; IPOP remains the source of truth.
    pub fn record_mode_hint(&self, enabled: bool) -> Result<()> {
        let value: &[u8] = if enabled { b"enabled\n" } else { b"disabled\n" };
        write_private(&self.inner.mode_hint_file, value)
    }

    pub async fn status(&self) -> PipeFsStatus {
        let state = self.inner.state.lock().await;
        let recovery_caches =
            recovery_caches(&self.inner.session_cache_root, Some(&self.inner.cache_root));
        PipeFsStatus {
            phase: state.phase,
            enabled: matches!(
                state.phase,
                WorkspacePhase::Restoring
                    | WorkspacePhase::Clean
                    | WorkspacePhase::Dirty
                    | WorkspacePhase::Pending
                    | WorkspacePhase::LeaseLost
            ),
            workspace_root: state.materialized_root.clone(),
            dirty_paths: state.dirty_paths.iter().cloned().collect(),
            retry_count: state.retry_count,
            last_committed_revision: state.remote.as_ref().and_then(|remote| remote.current_head),
            last_error: state.last_error.clone(),
            recovery_caches,
        }
    }

    pub fn original_activation(&self) -> Activation {
        Activation {
            workspace_root: self.inner.original_workspace_root.clone(),
            state_root: self.inner.original_state_root.clone(),
        }
    }

    fn persist_locked(&self, state: &ControllerState) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(state)?;
        write_private(&self.inner.state_file, &bytes)
    }

    fn clear_recovery_marker_if_safe(&self, state: &ControllerState) {
        if state.active_background_processes.is_empty()
            && matches!(
                state.phase,
                WorkspacePhase::Clean | WorkspacePhase::Disabled
            )
            && state.pending.is_none()
            && state.last_error.is_none()
        {
            let _ = fs::remove_file(&self.inner.recovery_marker);
        }
    }

    /// Remove older generation caches only when their persisted clean
    /// snapshot is already represented by the current remote head. Dirty or
    /// ambiguous caches remain available for explicit recovery.
    fn cleanup_stale_clean_caches(&self, remote: &PipeFsRemoteState) {
        let Ok(entries) = fs::read_dir(&self.inner.session_cache_root) else {
            return;
        };
        for entry in entries.filter_map(std::result::Result::ok) {
            let path = entry.path();
            if path == self.inner.cache_root || path.join("recovery-required").exists() {
                continue;
            }
            let Ok(metadata) = fs::symlink_metadata(&path) else {
                continue;
            };
            if !metadata.is_dir() || metadata.file_type().is_symlink() {
                continue;
            }
            let Ok(bytes) = fs::read(path.join("controller.json")) else {
                continue;
            };
            let Ok(state) = serde_json::from_slice::<ControllerState>(&bytes) else {
                continue;
            };
            let clean = matches!(
                state.phase,
                WorkspacePhase::Clean | WorkspacePhase::Disabled
            ) && state.pending.is_none()
                && state.last_error.is_none();
            let represented = state.remote.as_ref().is_some_and(|stored| {
                stored.current_head == remote.current_head
                    && stored.manifest_digest == remote.manifest_digest
                    && stored.logical_size_bytes == remote.logical_size_bytes
            });
            let materialized_matches = materialized_snapshot_matches(&path, &state);
            if clean && represented && materialized_matches {
                let _ = remove_cache_directory(&self.inner.session_cache_root, &path);
            } else if clean && represented && !materialized_matches {
                // A process outside HI's durability fence may have changed
                // the materialized bytes immediately before a forced exit.
                // Preserve and surface that cache instead of trusting stale
                // controller metadata and deleting uncommitted user work.
                let _ = write_private(
                    &path.join("recovery-required"),
                    b"workspace differs from its last committed snapshot\n",
                );
            }
        }
    }
}

fn mark_cache_for_recovery_if_drifted(cache_root: &Path, marker: &Path) -> Result<bool> {
    if marker.is_file() {
        return Ok(true);
    }
    if !cache_root.exists() {
        return Ok(false);
    }
    let state_file = cache_root.join("controller.json");
    if !state_file.exists() {
        return Ok(false);
    }
    let state = match fs::read(&state_file)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<ControllerState>(&bytes).ok())
    {
        Some(state) => state,
        None => {
            write_private(marker, b"controller state could not be verified\n")?;
            return Ok(true);
        }
    };
    let explicitly_dirty = matches!(state.phase, WorkspacePhase::Dirty | WorkspacePhase::Pending)
        || state.pending.is_some()
        || !state.active_background_processes.is_empty();
    if explicitly_dirty || !materialized_snapshot_matches(cache_root, &state) {
        write_private(
            marker,
            b"workspace differs from its last committed snapshot\n",
        )?;
        return Ok(true);
    }
    Ok(false)
}

fn materialized_snapshot_matches(cache_root: &Path, state: &ControllerState) -> bool {
    let (Some(materialized), Some(expected)) = (&state.materialized_root, &state.snapshot) else {
        return state.materialized_root.is_none() && state.snapshot.is_none();
    };
    if !materialized.starts_with(cache_root) || materialized == cache_root {
        return false;
    }
    let Ok(metadata) = fs::symlink_metadata(materialized) else {
        return false;
    };
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return false;
    }
    scan_workspace(materialized).is_ok_and(|actual| {
        actual.entries == expected.entries
            && actual.logical_size_bytes == expected.logical_size_bytes
    })
}

fn must_write_full(state: &ControllerState) -> bool {
    let Some(remote) = &state.remote else {
        return true;
    };
    if remote.current_head.is_none() || remote.restore_chain.is_empty() {
        return true;
    }
    let deltas = remote
        .restore_chain
        .iter()
        .filter(|revision| revision.revision_type == RevisionKind::Delta)
        .count() as u32;
    state
        .capabilities
        .as_ref()
        .is_none_or(|capabilities| deltas >= capabilities.maximum_delta_chain)
}

fn cumulative_delta_would_exceed_full(state: &ControllerState, next_delta: u64) -> bool {
    let Some(remote) = &state.remote else {
        return true;
    };
    let Some(full) = remote
        .restore_chain
        .iter()
        .find(|revision| revision.revision_type == RevisionKind::Full)
    else {
        return true;
    };
    remote
        .restore_chain
        .iter()
        .filter(|revision| revision.revision_type == RevisionKind::Delta)
        .fold(next_delta, |total, revision| {
            total.saturating_add(revision.artifact_size_bytes)
        })
        > full.artifact_size_bytes
}

fn is_compaction_required(error: &PipeFsError) -> bool {
    matches!(error, PipeFsError::Conflict(detail) if detail.contains("pipefs_compaction_required"))
}

/// Stable per-artifact request identity. Including the revision kind ensures a
/// server-requested full compaction never reuses the rejected delta's upload
/// preparation, even if a future archive encoding happens to share a digest.
fn revision_idempotency_key(
    lease_generation: u64,
    expected_base_revision_id: Option<Uuid>,
    revision_type: RevisionKind,
    artifact_blake3: &str,
) -> String {
    format!(
        "pfs:{lease_generation:x}:{}:{}:{artifact_blake3}",
        expected_base_revision_id
            .map(|id| id.simple().to_string())
            .unwrap_or_else(|| "empty".to_string()),
        match revision_type {
            RevisionKind::Full => "full",
            RevisionKind::Delta => "delta",
        },
    )
}

fn classified_restore_error(error: PipeFsError) -> anyhow::Error {
    anyhow!(match error {
        PipeFsError::Authentication(detail) => format!("authentication_error: {detail}"),
        PipeFsError::Storage(detail) => format!("storage_error: {detail}"),
        PipeFsError::MissingRevision(detail) => format!("missing_revision: {detail}"),
        PipeFsError::Network(detail) => format!("network_error: {detail}"),
        PipeFsError::Corruption(detail) => format!("corruption_error: {detail}"),
        other => other.to_string(),
    })
}

fn validate_session_id(id: &str) -> Result<()> {
    ensure!(
        !id.is_empty()
            && id.len() <= 128
            && !matches!(id, "." | "..")
            && id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')),
        "invalid PipeFS session id"
    );
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn default_cache_base() -> PathBuf {
    if let Some(base) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(base).join("hi/pipefs");
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        #[cfg(target_os = "macos")]
        return home.join("Library/Caches/hi/pipefs");
        #[cfg(not(target_os = "macos"))]
        return home.join(".cache/hi/pipefs");
    }
    std::env::temp_dir().join("hi-pipefs")
}

/// Whether local evidence says startup must resolve the remote PipeFS state
/// before using the launch directory. A dirty recovery cache is always a
/// positive hint; a corrupt/unreadable hint fails closed.
pub fn local_state_requires_remote_probe(session_id: &str) -> bool {
    local_state_requires_remote_probe_at(&default_cache_base(), session_id)
}

/// Update the default-cache mode hint after an authoritative read observes a
/// remote disable performed by another machine.
pub fn record_local_mode_hint(session_id: &str, enabled: bool) -> Result<()> {
    validate_session_id(session_id)?;
    let session_digest = blake3::hash(session_id.as_bytes()).to_hex().to_string();
    let session_root = default_cache_base().join(&session_digest[..32]);
    create_private_dir(&session_root)?;
    let value: &[u8] = if enabled { b"enabled\n" } else { b"disabled\n" };
    write_private(&session_root.join("remote-mode"), value)
}

fn local_state_requires_remote_probe_at(base: &Path, session_id: &str) -> bool {
    if validate_session_id(session_id).is_err() {
        return true;
    }
    if local_recovery_required_at(base, session_id) {
        return true;
    }
    let session_digest = blake3::hash(session_id.as_bytes()).to_hex().to_string();
    let session_root = base.join(&session_digest[..32]);
    match fs::read(session_root.join("remote-mode")) {
        Ok(value) if value == b"enabled\n" => true,
        Ok(value) if value == b"disabled\n" => false,
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

/// A prior process left materialized bytes that were not proven represented by
/// the remote head. Startup uses this stronger signal to re-enter PipeFS even
/// if the last mode transaction reached "disabled" before the process died.
pub fn local_recovery_required(session_id: &str) -> bool {
    local_recovery_required_at(&default_cache_base(), session_id)
}

fn local_recovery_required_at(base: &Path, session_id: &str) -> bool {
    if validate_session_id(session_id).is_err() {
        return true;
    }
    let session_digest = blake3::hash(session_id.as_bytes()).to_hex().to_string();
    let session_root = base.join(&session_digest[..32]);
    match fs::read_dir(&session_root) {
        Ok(entries) => entries
            .filter_map(std::result::Result::ok)
            .any(|entry| entry.path().join("recovery-required").is_file()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "PipeFS private directory is not a real directory: {}",
        path.display()
    );
    set_private_permissions(path)?;
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    #[cfg(unix)]
    {
        write_private_unix(path, bytes)
    }
    #[cfg(not(unix))]
    {
        write_private_portable(path, bytes)
    }
}

#[cfg(unix)]
fn write_private_unix(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::ffi::OsString;

    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("private PipeFS file has no parent: {}", path.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("private PipeFS file has no name: {}", path.display()))?;
    let directory = rustix::fs::open(
        parent,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::DIRECTORY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .map_err(std::io::Error::from)
    .with_context(|| format!("opening private PipeFS directory {}", parent.display()))?;
    let mut temporary = None;
    for _ in 0..32 {
        let name = OsString::from(format!(".pipefs-state-{}", Uuid::new_v4().simple()));
        match rustix::fs::openat(
            &directory,
            &name,
            rustix::fs::OFlags::WRONLY
                | rustix::fs::OFlags::CREATE
                | rustix::fs::OFlags::EXCL
                | rustix::fs::OFlags::NOFOLLOW
                | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::RUSR | rustix::fs::Mode::WUSR,
        ) {
            Ok(file) => {
                temporary = Some((name, fs::File::from(file)));
                break;
            }
            Err(rustix::io::Errno::EXIST) => {}
            Err(error) => return Err(std::io::Error::from(error).into()),
        }
    }
    let (temporary_name, mut file) = temporary
        .ok_or_else(|| anyhow!("could not allocate a collision-free PipeFS state file"))?;
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        rustix::fs::renameat(&directory, &temporary_name, &directory, file_name)
            .map_err(std::io::Error::from)?;
        rustix::fs::fsync(&directory).map_err(std::io::Error::from)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = rustix::fs::unlinkat(&directory, &temporary_name, rustix::fs::AtFlags::empty());
    }
    result.with_context(|| format!("atomically writing private PipeFS file {}", path.display()))
}

#[cfg(not(unix))]
fn write_private_portable(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("private PipeFS file has no parent: {}", path.display()))?;
    let temporary = parent.join(format!(".pipefs-state-{}", Uuid::new_v4().simple()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> Result<()> {
        set_private_permissions(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        if let Ok(metadata) = fs::symlink_metadata(path) {
            ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "refusing to replace non-regular private PipeFS state {}",
                path.display()
            );
            fs::remove_file(path)?;
        }
        fs::rename(&temporary, path)?;
        sync_parent(path);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(unix)]
fn set_private_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = if path.is_dir() { 0o700 } else { 0o600 };
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent()
        && let Ok(directory) = fs::File::open(parent)
    {
        let _ = directory.sync_all();
    }
}

fn remove_cache_directory(scope: &Path, target: &Path) -> Result<()> {
    ensure!(
        target.starts_with(scope) && target != scope,
        "refusing to remove cache outside its scoped session directory"
    );
    match fs::symlink_metadata(target) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            fs::remove_dir_all(target)?;
        }
        Ok(_) => bail!("refusing to recursively remove a non-directory cache target"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn recovery_caches(session_root: &Path, current: Option<&Path>) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(session_root) else {
        return Vec::new();
    };
    entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| current != Some(path.as_path()))
        .filter(|path| path.join("recovery-required").is_file())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remote_with_chain(kinds: &[RevisionKind], sizes: &[u64]) -> ControllerState {
        ControllerState {
            phase: WorkspacePhase::Dirty,
            remote: Some(PipeFsRemoteStateDisk {
                enabled: true,
                current_head: Some(Uuid::new_v4()),
                sequence: kinds.len() as u64,
                manifest_digest: Some("a".repeat(64)),
                logical_size_bytes: 1,
                restore_chain: kinds
                    .iter()
                    .zip(sizes)
                    .map(|(kind, size)| RestoreRevisionDisk {
                        revision_id: Uuid::new_v4(),
                        base_revision_id: None,
                        revision_type: *kind,
                        artifact_size_bytes: *size,
                    })
                    .collect(),
            }),
            capabilities: Some(CapabilitiesDisk {
                maximum_revision_bytes: 1_000_000,
                maximum_workspace_bytes: 1_000_000,
                maximum_delta_chain: 20,
            }),
            ..ControllerState::default()
        }
    }

    #[test]
    fn compaction_after_twenty_deltas_or_cumulative_full_size() {
        let mut kinds = vec![RevisionKind::Full];
        kinds.extend(std::iter::repeat_n(RevisionKind::Delta, 20));
        let sizes = vec![10; kinds.len()];
        assert!(must_write_full(&remote_with_chain(&kinds, &sizes)));

        let state = remote_with_chain(&[RevisionKind::Full, RevisionKind::Delta], &[100, 60]);
        assert!(cumulative_delta_would_exceed_full(&state, 41));
        assert!(!cumulative_delta_would_exceed_full(&state, 40));
    }

    #[test]
    fn compaction_conflict_is_identified_and_gets_a_fresh_full_key() {
        assert!(is_compaction_required(&PipeFsError::Conflict(
            "409: pipefs_compaction_required: next revision must be full".to_string()
        )));
        assert!(!is_compaction_required(&PipeFsError::Conflict(
            "409: pipefs_head_conflict".to_string()
        )));

        let base = Some(Uuid::nil());
        let delta = revision_idempotency_key(7, base, RevisionKind::Delta, "same-artifact");
        let full = revision_idempotency_key(7, base, RevisionKind::Full, "same-artifact");
        assert_ne!(delta, full);
        assert_eq!(
            full,
            revision_idempotency_key(7, base, RevisionKind::Full, "same-artifact")
        );
    }

    #[tokio::test]
    async fn server_requested_compaction_restages_current_tree_as_full() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        fs::write(workspace_root.join("current.txt"), "current tree").unwrap();
        let workspace = PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "http://127.0.0.1:1",
                "test-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: "token".to_string(),
                generation: 1,
            },
            PipeFsWorkspaceConfig {
                session_id: "compaction-restage-test".to_string(),
                original_workspace_root: workspace_root.clone(),
                original_state_root: state_root,
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap();
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Pending;
        state.materialized_root = Some(workspace_root);
        state.remote = Some(
            (&PipeFsRemoteState {
                session_id: "compaction-restage-test".to_string(),
                enabled: true,
                current_head: Some(Uuid::new_v4()),
                sequence: 2,
                manifest_digest: Some("a".repeat(64)),
                logical_size_bytes: 0,
                restore_chain: Vec::new(),
            })
                .into(),
        );
        state.capabilities = Some(CapabilitiesDisk {
            maximum_revision_bytes: 1_000_000,
            maximum_workspace_bytes: 1_000_000,
            maximum_delta_chain: 20,
        });
        let old_key = revision_idempotency_key(
            1,
            state.remote.as_ref().and_then(|remote| remote.current_head),
            RevisionKind::Delta,
            "old-delta",
        );
        state.pending = Some(PendingRevision {
            expected_base_revision_id: state.remote.as_ref().and_then(|remote| remote.current_head),
            revision_type: RevisionKind::Delta,
            archive_blake3: "old-delta".to_string(),
            manifest_digest: "old-manifest".to_string(),
            logical_size_bytes: 0,
            idempotency_key: old_key.clone(),
            snapshot: Snapshot::default(),
        });

        workspace
            .restage_full_after_compaction_locked(&mut state, 1)
            .await
            .unwrap();
        let pending = state.pending.as_ref().unwrap();
        assert_eq!(pending.revision_type, RevisionKind::Full);
        assert_ne!(pending.idempotency_key, old_key);
        assert!(workspace.inner.pending_archive.is_file());
    }

    #[test]
    fn cache_removal_is_scoped() {
        let temporary = tempfile::tempdir().unwrap();
        let session = temporary.path().join("session");
        let generation = session.join("generation");
        fs::create_dir_all(&generation).unwrap();
        remove_cache_directory(&session, &generation).unwrap();
        assert!(!generation.exists());
        assert!(remove_cache_directory(&session, &session).is_err());
        assert!(remove_cache_directory(&session, temporary.path()).is_err());
    }

    #[test]
    fn session_mode_hint_survives_generation_cleanup_and_dirty_cache_fails_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let base = temporary.path().join("cache");
        let session_id = "mode-hint-test";
        let digest = blake3::hash(session_id.as_bytes()).to_hex().to_string();
        let session_root = base.join(&digest[..32]);
        create_private_dir(&session_root).unwrap();

        assert!(!local_state_requires_remote_probe_at(&base, session_id));
        assert!(!local_recovery_required_at(&base, session_id));
        write_private(&session_root.join("remote-mode"), b"enabled\n").unwrap();
        assert!(local_state_requires_remote_probe_at(&base, session_id));
        write_private(&session_root.join("remote-mode"), b"disabled\n").unwrap();
        assert!(!local_state_requires_remote_probe_at(&base, session_id));

        let generation = session_root.join("generation");
        create_private_dir(&generation).unwrap();
        write_private(
            &generation.join("recovery-required"),
            b"uncommitted workspace changes\n",
        )
        .unwrap();
        assert!(local_state_requires_remote_probe_at(&base, session_id));
        assert!(local_recovery_required_at(&base, session_id));
        fs::remove_dir_all(generation).unwrap();

        write_private(&session_root.join("remote-mode"), b"corrupt\n").unwrap();
        assert!(local_state_requires_remote_probe_at(&base, session_id));
    }

    #[cfg(unix)]
    #[test]
    fn private_state_write_replaces_a_symlink_without_following_it() {
        let temporary = tempfile::tempdir().unwrap();
        let victim = temporary.path().join("victim");
        let destination = temporary.path().join("controller.json");
        fs::write(&victim, b"do not overwrite").unwrap();
        std::os::unix::fs::symlink(&victim, &destination).unwrap();

        write_private(&destination, b"safe state").unwrap();

        assert_eq!(fs::read(&victim).unwrap(), b"do not overwrite");
        assert_eq!(fs::read(&destination).unwrap(), b"safe state");
        assert!(
            !fs::symlink_metadata(destination)
                .unwrap()
                .file_type()
                .is_symlink()
        );
    }

    #[tokio::test]
    async fn live_background_process_keeps_recovery_marker_after_clean_checkpoint() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        let workspace = PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "http://127.0.0.1:1",
                "test-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: "token".to_string(),
                generation: 1,
            },
            PipeFsWorkspaceConfig {
                session_id: "background-marker-test".to_string(),
                original_workspace_root: workspace_root,
                original_state_root: state_root,
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap();
        {
            let mut state = workspace.inner.state.lock().await;
            state.phase = WorkspacePhase::Clean;
            workspace.persist_locked(&state).unwrap();
        }

        workspace
            .background_process_state("process-1", true)
            .await
            .unwrap();
        assert!(workspace.inner.recovery_marker.is_file());
        {
            let state = workspace.inner.state.lock().await;
            workspace.clear_recovery_marker_if_safe(&state);
        }
        assert!(workspace.inner.recovery_marker.is_file());

        workspace
            .background_process_state("process-1", false)
            .await
            .unwrap();
        assert!(!workspace.inner.recovery_marker.exists());
    }

    #[tokio::test]
    async fn advancing_lease_generation_restages_a_pending_revision() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        let workspace = PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "http://127.0.0.1:1",
                "test-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: "old-token".to_string(),
                generation: 4,
            },
            PipeFsWorkspaceConfig {
                session_id: "lease-refresh-test".to_string(),
                original_workspace_root: workspace_root,
                original_state_root: state_root,
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap();
        {
            let mut state = workspace.inner.state.lock().await;
            state.phase = WorkspacePhase::Pending;
            state.pending = Some(PendingRevision {
                expected_base_revision_id: None,
                revision_type: RevisionKind::Full,
                archive_blake3: "a".repeat(64),
                manifest_digest: "b".repeat(64),
                logical_size_bytes: 0,
                idempotency_key: "old-generation".to_string(),
                snapshot: Snapshot::default(),
            });
            workspace.persist_locked(&state).unwrap();
        }
        write_private(&workspace.inner.pending_archive, b"staged").unwrap();

        workspace
            .update_lease(PipeFsLease {
                token: "new-token".to_string(),
                generation: 5,
            })
            .await
            .unwrap();

        let state = workspace.inner.state.lock().await;
        assert_eq!(state.phase, WorkspacePhase::Dirty);
        assert!(state.pending.is_none());
        assert!(workspace.inner.recovery_marker.is_file());
        assert!(!workspace.inner.pending_archive.exists());
        drop(state);
        assert_eq!(workspace.inner.lease.lock().await.generation, 5);
    }

    #[tokio::test]
    async fn removes_only_stale_clean_cache_represented_by_remote_head() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        let make_workspace = |generation| {
            PipeFsWorkspace::new(
                PipeFsClient::new(crate::PipeFsClientConfig::new(
                    "http://127.0.0.1:1",
                    "test-key",
                ))
                .unwrap(),
                PipeFsLease {
                    token: format!("token-{generation}"),
                    generation,
                },
                PipeFsWorkspaceConfig {
                    session_id: "clean-cache-test".to_string(),
                    original_workspace_root: workspace_root.clone(),
                    original_state_root: state_root.clone(),
                    cache_base: Some(temporary.path().join("cache")),
                },
            )
            .unwrap()
        };
        let old = make_workspace(1);
        let head = Uuid::new_v4();
        let remote = PipeFsRemoteState {
            session_id: "clean-cache-test".to_string(),
            enabled: true,
            current_head: Some(head),
            sequence: 1,
            manifest_digest: Some("a".repeat(64)),
            logical_size_bytes: 0,
            restore_chain: Vec::new(),
        };
        {
            let mut state = old.inner.state.lock().await;
            let materialized = old.inner.cache_root.join("workspace-clean");
            create_private_dir(&materialized).unwrap();
            state.phase = WorkspacePhase::Clean;
            state.remote = Some((&remote).into());
            state.snapshot = Some(Snapshot {
                manifest_digest: remote.manifest_digest.clone(),
                ..Snapshot::default()
            });
            state.materialized_root = Some(materialized);
            old.persist_locked(&state).unwrap();
        }
        let old_root = old.inner.cache_root.clone();
        let current = make_workspace(2);

        current.cleanup_stale_clean_caches(&remote);

        assert!(!old_root.exists());
        assert!(current.inner.cache_root.exists());
    }

    #[tokio::test]
    async fn same_generation_restart_preserves_clean_controller_with_unscanned_drift() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        let config = PipeFsWorkspaceConfig {
            session_id: "same-generation-drift-test".to_string(),
            original_workspace_root: workspace_root,
            original_state_root: state_root,
            cache_base: Some(temporary.path().join("cache")),
        };
        let make_workspace = || {
            PipeFsWorkspace::new(
                PipeFsClient::new(crate::PipeFsClientConfig::new(
                    "http://127.0.0.1:1",
                    "test-key",
                ))
                .unwrap(),
                PipeFsLease {
                    token: "same-token".to_string(),
                    generation: 7,
                },
                config.clone(),
            )
            .unwrap()
        };
        let old = make_workspace();
        let old_materialized = old.inner.cache_root.join("workspace-old");
        create_private_dir(&old_materialized).unwrap();
        let committed_snapshot = scan_workspace(&old_materialized).unwrap();
        {
            let mut state = old.inner.state.lock().await;
            state.phase = WorkspacePhase::Clean;
            state.snapshot = Some(committed_snapshot);
            state.materialized_root = Some(old_materialized.clone());
            old.persist_locked(&state).unwrap();
        }

        // Simulate a direct editor write followed by SIGKILL, after the
        // controller had most recently persisted `Clean`.
        fs::write(old_materialized.join("unfenced-change"), "keep me").unwrap();
        let old_cache = old.inner.cache_root.clone();
        drop(old);

        let resumed = make_workspace();
        assert_ne!(resumed.inner.cache_root, old_cache);
        assert!(old_cache.join("recovery-required").is_file());
        assert_eq!(
            fs::read_to_string(old_materialized.join("unfenced-change")).unwrap(),
            "keep me"
        );
        assert!(
            recovery_caches(
                &resumed.inner.session_cache_root,
                Some(&resumed.inner.cache_root)
            )
            .contains(&old_cache)
        );
    }

    #[tokio::test]
    async fn stale_clean_cache_with_drift_is_marked_instead_of_deleted() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        let make_workspace = |generation| {
            PipeFsWorkspace::new(
                PipeFsClient::new(crate::PipeFsClientConfig::new(
                    "http://127.0.0.1:1",
                    "test-key",
                ))
                .unwrap(),
                PipeFsLease {
                    token: format!("token-{generation}"),
                    generation,
                },
                PipeFsWorkspaceConfig {
                    session_id: "stale-drift-test".to_string(),
                    original_workspace_root: workspace_root.clone(),
                    original_state_root: state_root.clone(),
                    cache_base: Some(temporary.path().join("cache")),
                },
            )
            .unwrap()
        };
        let old = make_workspace(1);
        let materialized = old.inner.cache_root.join("workspace-old");
        create_private_dir(&materialized).unwrap();
        let snapshot = scan_workspace(&materialized).unwrap();
        let remote = PipeFsRemoteState {
            session_id: "stale-drift-test".to_string(),
            enabled: true,
            current_head: Some(Uuid::new_v4()),
            sequence: 1,
            manifest_digest: Some("a".repeat(64)),
            logical_size_bytes: 0,
            restore_chain: Vec::new(),
        };
        {
            let mut state = old.inner.state.lock().await;
            state.phase = WorkspacePhase::Clean;
            state.remote = Some((&remote).into());
            state.snapshot = Some(snapshot);
            state.materialized_root = Some(materialized.clone());
            old.persist_locked(&state).unwrap();
        }
        fs::write(materialized.join("late-write"), "preserve").unwrap();
        let old_cache = old.inner.cache_root.clone();
        drop(old);
        let current = make_workspace(2);

        current.cleanup_stale_clean_caches(&remote);

        assert!(old_cache.exists());
        assert!(old_cache.join("recovery-required").is_file());
        assert_eq!(
            fs::read_to_string(materialized.join("late-write")).unwrap(),
            "preserve"
        );
    }

    #[tokio::test]
    async fn rejects_workspace_larger_than_server_capability_before_staging() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        fs::write(workspace_root.join("large"), b"0123456789").unwrap();
        let artifact = build_revision(&workspace_root, None, true).unwrap();
        let workspace = PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "http://127.0.0.1:1",
                "test-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: "token".to_string(),
                generation: 1,
            },
            PipeFsWorkspaceConfig {
                session_id: "workspace-limit-test".to_string(),
                original_workspace_root: workspace_root,
                original_state_root: state_root,
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap();
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Dirty;
        state.capabilities = Some(CapabilitiesDisk {
            maximum_revision_bytes: 1_000_000,
            maximum_workspace_bytes: 5,
            maximum_delta_chain: 20,
        });

        let error = workspace
            .stage_pending_locked(&mut state, artifact, 1)
            .unwrap_err();

        assert!(error.to_string().contains("exceeding the server limit"));
        assert!(state.pending.is_none());
        assert!(!workspace.inner.pending_archive.exists());
    }
}
