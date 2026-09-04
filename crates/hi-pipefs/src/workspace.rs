use std::collections::BTreeSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::{
    ARCHIVE_VERSION, PipeFsCacheScope, PipeFsCapabilities, PipeFsClient, PipeFsError, PipeFsLease,
    PipeFsRemoteState, RevisionKind, Snapshot, StagedArchiveArtifact, apply_archive_file,
    build_revision_from_snapshot_to_file_bounded, revision_archive_size_upper_bound,
    scan_workspace,
};

#[derive(Clone, Debug)]
pub struct PipeFsWorkspaceConfig {
    pub session_id: String,
    pub cache_scope: PipeFsCacheScope,
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
    pub materialized_logical_bytes: u64,
    pub pending_archive_bytes: u64,
    pub available_cache_bytes: Option<u64>,
}

/// A crash-recovery cache that has not yet been proven durable remotely.
///
/// `id` is the exact, session-scoped identifier accepted by the recovery
/// export/discard APIs. It is intentionally unrelated to the user or session
/// identity and cannot address paths outside this session's cache directory.
#[derive(Clone, Debug)]
pub struct PipeFsRecoveryCache {
    pub id: String,
    pub path: PathBuf,
    pub workspace_root: Option<PathBuf>,
    pub phase: Option<WorkspacePhase>,
    pub logical_size_bytes: u64,
    pub pending_archive_bytes: u64,
    pub last_error: Option<String>,
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
        writeln!(
            formatter,
            "local cache: {} logical bytes, {} pending archive bytes, {} available",
            self.materialized_logical_bytes,
            self.pending_archive_bytes,
            self.available_cache_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "unknown".to_string())
        )?;
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
    cache_scope: PipeFsCacheScope,
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

/// Owns one UUID-named temporary immediately beneath a private generation
/// cache. Remaining armed across `.await` points makes cancellation and panic
/// cleanup automatic without ever covering a source recovery cache or a
/// controller-acknowledged materialized root.
struct OwnedCacheTemporary {
    scope: PathBuf,
    path: PathBuf,
    recursive: bool,
    armed: bool,
}

impl OwnedCacheTemporary {
    fn archive(scope: &Path, purpose: &str) -> Self {
        Self::new(
            scope,
            format!(".{purpose}-{}.tar.zst", Uuid::new_v4().simple()),
            false,
        )
    }

    fn directory(scope: &Path, purpose: &str) -> Self {
        Self::new(
            scope,
            format!("{purpose}-{}", Uuid::new_v4().simple()),
            true,
        )
    }

    fn new(scope: &Path, file_name: String, recursive: bool) -> Self {
        debug_assert!(
            file_name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-'))
        );
        Self {
            scope: scope.to_path_buf(),
            path: scope.join(file_name),
            recursive,
            armed: true,
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for OwnedCacheTemporary {
    fn drop(&mut self) {
        if !self.armed || self.path.parent() != Some(self.scope.as_path()) {
            return;
        }
        match fs::symlink_metadata(&self.path) {
            Ok(metadata)
                if self.recursive && metadata.is_dir() && !metadata.file_type().is_symlink() =>
            {
                let _ = fs::remove_dir_all(&self.path);
            }
            Ok(metadata) if !metadata.is_dir() || metadata.file_type().is_symlink() => {
                let _ = fs::remove_file(&self.path);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => {}
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ControllerState {
    /// `None` is a legacy, unscoped controller and must never be adopted by an
    /// authenticated automatic-recovery path.
    #[serde(default)]
    cache_authority_scope: Option<String>,
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
            cache_authority_scope: None,
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

impl ControllerState {
    fn for_cache_scope(scope: &PipeFsCacheScope) -> Self {
        Self {
            cache_authority_scope: Some(scope.as_str().to_string()),
            ..Self::default()
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct ModeHint {
    version: u8,
    cache_authority_scope: String,
    enabled: bool,
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
    #[serde(default = "default_true")]
    writes_available: bool,
    #[serde(default = "default_true")]
    restore_available: bool,
}

fn default_true() -> bool {
    true
}

impl From<&PipeFsCapabilities> for CapabilitiesDisk {
    fn from(value: &PipeFsCapabilities) -> Self {
        Self {
            maximum_revision_bytes: value.maximum_revision_bytes,
            maximum_workspace_bytes: value.maximum_workspace_bytes,
            maximum_delta_chain: value.maximum_delta_chain,
            writes_available: value.writes_available(),
            restore_available: value.restore_available(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PendingRevision {
    expected_base_revision_id: Option<Uuid>,
    revision_type: RevisionKind,
    archive_blake3: String,
    #[serde(default)]
    archive_size_bytes: u64,
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
        ensure_supported_platform()?;
        validate_session_id(&config.session_id)?;
        ensure!(
            config.cache_scope == client.cache_scope(),
            "PipeFS cache authority scope does not match the authenticated client"
        );
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
        let authority_cache_root = prepare_authority_cache_root(&base, &config.cache_scope)?;
        let session_digest = blake3::hash(config.session_id.as_bytes())
            .to_hex()
            .to_string();
        let generation_digest = blake3::hash(
            format!(
                "{}\0{}\0{}",
                config.cache_scope.as_str(),
                config.session_id,
                lease.generation
            )
            .as_bytes(),
        )
        .to_hex()
        .to_string();
        let session_cache_root = authority_cache_root.join(&session_digest[..32]);
        create_private_dir(&session_cache_root)?;
        let mode_hint_file = session_cache_root.join("remote-mode");
        let generation_cache_root = session_cache_root.join(&generation_digest[..32]);
        let generation_needs_recovery = mark_cache_for_recovery_if_drifted(
            &generation_cache_root,
            &generation_cache_root.join("recovery-required"),
            &config.cache_scope,
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
            let state: ControllerState = serde_json::from_slice(&fs::read(&state_file)?)
                .context("reading existing PipeFS controller state")?;
            validate_controller_cache_scope(&state, &config.cache_scope)?;
            state
        } else {
            ControllerState::for_cache_scope(&config.cache_scope)
        };
        Ok(Self {
            inner: Arc::new(Inner {
                client,
                session_id: config.session_id,
                cache_scope: config.cache_scope,
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

    fn temporary_archive_path(&self, purpose: &str) -> PathBuf {
        self.inner
            .cache_root
            .join(format!(".{purpose}-{}.tar.zst", Uuid::new_v4().simple()))
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
            capabilities.enrollment_available(),
            "new PipeFS enrollment is disabled on this IPOP server"
        );
        self.validate_capabilities(&capabilities)?;
        let lease = self.inner.lease.lock().await.clone();
        let remote = self
            .inner
            .client
            .set_enabled(&self.inner.session_id, &lease, true)
            .await?;
        self.activate_remote(capabilities, remote).await
    }

    /// Restore an already-enabled remote workspace without issuing a mode
    /// mutation. Rollout may intentionally disable new enrollment/writes while
    /// retaining restore access so users can drain existing sessions safely.
    pub async fn restore_existing(&self) -> Result<Activation> {
        let capabilities = self.inner.client.capabilities().await?;
        ensure!(
            capabilities.restore_available(),
            "PipeFS restore is disabled on this IPOP server"
        );
        self.validate_capabilities(&capabilities)?;
        let remote = self.inner.client.state(&self.inner.session_id).await?;
        ensure!(
            remote.enabled,
            "the remote PipeFS workspace is no longer enabled"
        );
        self.activate_remote(capabilities, remote).await
    }

    fn validate_capabilities(&self, capabilities: &PipeFsCapabilities) -> Result<()> {
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
        Ok(())
    }

    async fn activate_remote(
        &self,
        capabilities: PipeFsCapabilities,
        mut remote: PipeFsRemoteState,
    ) -> Result<Activation> {
        self.record_mode_hint(true)?;
        self.cleanup_stale_clean_caches(&remote);
        ensure_cache_capacity(
            &self.inner.cache_root,
            remote.logical_size_bytes,
            "restoring the remote workspace",
        )?;

        {
            let mut state = self.inner.state.lock().await;
            state.phase = WorkspacePhase::Restoring;
            state.capabilities = Some((&capabilities).into());
            state.remote = Some((&remote).into());
            state.last_error = None;
            self.persist_locked(&state)?;
        }

        let staging = OwnedCacheTemporary::directory(&self.inner.cache_root, "restore");
        create_private_dir(staging.path())?;
        let restore_result = self
            .restore_into(staging.path(), &remote, capabilities.maximum_revision_bytes)
            .await;
        let snapshot = match restore_result {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let mut state = self.inner.state.lock().await;
                state.phase = WorkspacePhase::Disabled;
                state.last_error = Some(format!("restore failed: {error:#}"));
                self.persist_locked(&state)?;
                return Err(error);
            }
        };
        let (staging, snapshot, recovered_cache) = match self
            .recover_local_cache(staging, snapshot, &mut remote)
            .await
        {
            Ok(recovered) => recovered,
            Err(error) => {
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
        let mut materialized = OwnedCacheTemporary::directory(&self.inner.cache_root, "workspace");
        let materialized_path = materialized.path().to_path_buf();
        let old_materialized = {
            let mut state = self.inner.state.lock().await;
            fs::rename(staging.path(), materialized.path()).with_context(|| {
                format!(
                    "atomically activating PipeFS workspace {}",
                    materialized.path().display()
                )
            })?;
            let previous_state = state.clone();
            let old = state.materialized_root.replace(materialized_path.clone());
            state.snapshot = Some(snapshot);
            state.pending = None;
            state.phase = WorkspacePhase::Clean;
            state.remote = Some((&remote).into());
            state.dirty_paths.clear();
            state.retry_count = 0;
            state.last_error = None;
            if let Err(error) = self.persist_locked(&state) {
                *state = previous_state;
                // Restore the previously durable path when possible. The
                // destination guard remains armed as a safe fallback.
                let _ = fs::rename(materialized.path(), staging.path());
                return Err(error);
            }
            old
        };
        // From this point the controller durably owns the activated root.
        materialized.disarm();
        if let Some(old) = old_materialized.filter(|old| *old != materialized_path) {
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
            workspace_root: materialized_path,
            state_root: self.inner.runtime_state_root.clone(),
        })
    }

    async fn recover_local_cache(
        &self,
        staging: OwnedCacheTemporary,
        restored_snapshot: Snapshot,
        remote: &mut PipeFsRemoteState,
    ) -> Result<(OwnedCacheTemporary, Snapshot, Option<PathBuf>)> {
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
        let controller_path = candidate.join("controller.json");
        let controller_metadata = fs::symlink_metadata(&controller_path).with_context(|| {
            format!(
                "reading recovery controller metadata from {}",
                candidate.display()
            )
        })?;
        ensure!(
            controller_metadata.is_file() && !controller_metadata.file_type().is_symlink(),
            "recovery cache controller is not a regular non-symlink file"
        );
        let old_state: ControllerState = serde_json::from_slice(
            &fs::read(&controller_path)
                .with_context(|| format!("reading recovery state from {}", candidate.display()))?,
        )
        .with_context(|| format!("parsing recovery state from {}", candidate.display()))?;
        validate_controller_cache_scope(&old_state, &self.inner.cache_scope).with_context(
            || {
                format!(
                    "recovery cache {} belongs to a different PipeFS authority",
                    candidate.display()
                )
            },
        )?;
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
        // A crash can land after the server advanced its head but before the
        // client persisted that acknowledgement. Matching the pending
        // manifest and logical size proves that revision is now the remote
        // head; treat it as acknowledged and rebase any still-newer local
        // bytes on that head instead of manufacturing a false conflict.
        let pending_landed = old_state
            .pending
            .as_ref()
            .is_some_and(|pending| pending_matches_remote_head(pending, remote));
        let expected_base = if pending_landed {
            remote.current_head
        } else {
            old_state
                .pending
                .as_ref()
                .map(|pending| pending.expected_base_revision_id)
                .or_else(|| old_state.remote.as_ref().map(|state| state.current_head))
                .ok_or_else(|| anyhow!("recovery cache has no recorded remote base"))?
        };
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
        let capabilities = self
            .inner
            .state
            .lock()
            .await
            .capabilities
            .clone()
            .ok_or_else(|| anyhow!("PipeFS capabilities are unavailable during recovery"))?;
        ensure_archive_build_capacity(
            &self.inner.cache_root,
            &recovered_snapshot,
            Some(&restored_snapshot),
            force_full,
            &capabilities,
            "staging interrupted workspace recovery",
        )?;
        let build_root = old_root.clone();
        let base = restored_snapshot.clone();
        let recovered_for_build = recovered_snapshot.clone();
        let mut artifact_temporary =
            OwnedCacheTemporary::archive(&self.inner.cache_root, "recover-delta");
        let build_path_for_task = artifact_temporary.path().to_path_buf();
        let maximum_revision_bytes = capabilities.maximum_revision_bytes;
        let mut artifact = tokio::task::spawn_blocking(move || {
            build_revision_from_snapshot_to_file_bounded(
                &build_root,
                recovered_for_build,
                Some(&base),
                force_full,
                &build_path_for_task,
                maximum_revision_bytes,
            )
        })
        .await
        .context("PipeFS recovery archive task panicked")??;
        {
            let state = self.inner.state.lock().await;
            if artifact.manifest.revision_type == RevisionKind::Delta
                && cumulative_delta_would_exceed_full(&state, artifact.size_bytes)
            {
                drop(state);
                let build_root = old_root.clone();
                let recovered_for_build = recovered_snapshot.clone();
                let replacement_temporary =
                    OwnedCacheTemporary::archive(&self.inner.cache_root, "recover-full");
                ensure_archive_build_capacity(
                    &self.inner.cache_root,
                    &recovered_for_build,
                    None,
                    true,
                    &capabilities,
                    "staging a full interrupted-workspace compaction",
                )?;
                let maximum_revision_bytes = capabilities.maximum_revision_bytes;
                let build_path = replacement_temporary.path().to_path_buf();
                artifact = tokio::task::spawn_blocking(move || {
                    build_revision_from_snapshot_to_file_bounded(
                        &build_root,
                        recovered_for_build,
                        None,
                        true,
                        &build_path,
                        maximum_revision_bytes,
                    )
                })
                .await
                .context("PipeFS recovery compaction task panicked")??;
                artifact_temporary = replacement_temporary;
            }
        }

        // Materialize the recovered bytes independently from the remote
        // staging tree. A full archive validates the complete local tree and
        // avoids filesystem-dependent recursive copy behavior.
        let mut recovered_staging =
            OwnedCacheTemporary::directory(&self.inner.cache_root, "recovery");
        ensure_cache_capacity(
            &self.inner.cache_root,
            recovered_snapshot.logical_size_bytes,
            "materializing interrupted workspace recovery",
        )?;
        create_private_dir(recovered_staging.path())?;
        let materialization_archive = if artifact.manifest.revision_type == RevisionKind::Full {
            None
        } else {
            let build_root = old_root;
            let recovered_for_build = recovered_snapshot;
            let temporary =
                OwnedCacheTemporary::archive(&self.inner.cache_root, "recover-materialize");
            ensure_archive_build_capacity(
                &self.inner.cache_root,
                &recovered_for_build,
                None,
                true,
                &capabilities,
                "building the interrupted-workspace materialization archive",
            )?;
            let maximum_revision_bytes = capabilities.maximum_revision_bytes;
            let build_path = temporary.path().to_path_buf();
            let full = tokio::task::spawn_blocking(move || {
                build_revision_from_snapshot_to_file_bounded(
                    &build_root,
                    recovered_for_build,
                    None,
                    true,
                    &build_path,
                    maximum_revision_bytes,
                )
            })
            .await
            .context("PipeFS full recovery archive task panicked")??;
            debug_assert_eq!(full.path, temporary.path());
            Some(temporary)
        };
        let restore_root = recovered_staging.path().to_path_buf();
        let restore_archive_for_task = materialization_archive
            .as_ref()
            .map_or_else(|| artifact_temporary.path(), OwnedCacheTemporary::path)
            .to_path_buf();
        let materialize_result = tokio::task::spawn_blocking(move || {
            apply_archive_file(&restore_root, &restore_archive_for_task, None)
        })
        .await
        .context("PipeFS local recovery extraction task panicked")?;
        let mut materialized_snapshot = materialize_result?;

        // Acquire the lease generation before publishing a controller which
        // owns the recovered directory. Once the pending revision is durably
        // staged, that directory is recovery state rather than a disposable
        // temporary and must survive cancellation during remote I/O.
        let generation = self.inner.lease.lock().await.generation;
        let commit_result = {
            let mut state = self.inner.state.lock().await;
            state.snapshot = Some(restored_snapshot);
            state.materialized_root = Some(recovered_staging.path().to_path_buf());
            state.phase = WorkspacePhase::Dirty;
            state
                .dirty_paths
                .insert("<recovered interrupted cache>".to_string());
            write_private(
                &self.inner.recovery_marker,
                b"recovering interrupted workspace changes\n",
            )?;
            self.persist_locked(&state)?;
            self.stage_pending_locked(&mut state, artifact, generation)?;
            recovered_staging.disarm();
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
                let _ = remove_cache_directory(&self.inner.cache_root, recovered_staging.path());
                return Err(error.context(
                    "persisting the interrupted PipeFS cache; the older recovery cache was retained",
                ));
            }
        };
        materialized_snapshot.manifest_digest = committed_remote.manifest_digest.clone();
        *remote = committed_remote;
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
            ensure_cache_capacity(
                &self.inner.cache_root,
                revision.artifact.size_bytes,
                "downloading a workspace revision",
            )?;
            let archive = OwnedCacheTemporary::archive(&self.inner.cache_root, "restore");
            let download_result = self
                .inner
                .client
                .download_revision_to_file(
                    &self.inner.session_id,
                    revision,
                    maximum_revision_bytes,
                    archive.path(),
                )
                .await
                .map_err(classified_restore_error);
            download_result?;
            let restore_root = staging.to_path_buf();
            let restore_archive = archive.path().to_path_buf();
            let expected_base = snapshot.clone();
            let restore_result = tokio::task::spawn_blocking(move || {
                apply_archive_file(&restore_root, &restore_archive, expected_base.as_ref())
            })
            .await
            .context("PipeFS restore extraction task panicked");
            let restored = restore_result??;
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
        ensure!(
            state
                .capabilities
                .as_ref()
                .is_some_and(|capabilities| capabilities.writes_available),
            "PipeFS writes are temporarily disabled by the server; this restored workspace is read-only"
        );
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
        let capabilities = state
            .capabilities
            .clone()
            .ok_or_else(|| anyhow!("PipeFS capabilities are unavailable"))?;
        if let Err(error) = ensure_archive_build_capacity(
            &self.inner.cache_root,
            &current,
            prior.as_ref(),
            force_full,
            &capabilities,
            "staging a workspace revision",
        ) {
            self.record_checkpoint_failure_locked(&mut state, &error)?;
            return Err(error);
        }
        let root_for_build = root;
        let base_for_build = prior.clone();
        let build_path = self.temporary_archive_path("checkpoint");
        let build_path_for_task = build_path.clone();
        let maximum_revision_bytes = capabilities.maximum_revision_bytes;
        let mut artifact = match tokio::task::spawn_blocking(move || {
            build_revision_from_snapshot_to_file_bounded(
                &root_for_build,
                current,
                base_for_build.as_ref(),
                force_full,
                &build_path_for_task,
                maximum_revision_bytes,
            )
        })
        .await
        {
            Ok(Ok(artifact)) => artifact,
            Ok(Err(error)) => {
                let _ = fs::remove_file(&build_path);
                self.record_checkpoint_failure_locked(&mut state, &error)?;
                return Err(error);
            }
            Err(join_error) => {
                let _ = fs::remove_file(&build_path);
                let error = anyhow!("PipeFS archive task panicked: {join_error}");
                self.record_checkpoint_failure_locked(&mut state, &error)?;
                return Err(error);
            }
        };
        if artifact.manifest.revision_type == RevisionKind::Delta
            && cumulative_delta_would_exceed_full(&state, artifact.size_bytes)
        {
            let current = artifact.snapshot.clone();
            let _ = fs::remove_file(&artifact.path);
            let root_for_build = state.materialized_root.clone().expect("checked above");
            let build_path = self.temporary_archive_path("checkpoint-full");
            let build_path_for_task = build_path.clone();
            if let Err(error) = ensure_archive_build_capacity(
                &self.inner.cache_root,
                &current,
                None,
                true,
                &capabilities,
                "staging a full workspace compaction",
            ) {
                self.record_checkpoint_failure_locked(&mut state, &error)?;
                return Err(error);
            }
            let maximum_revision_bytes = capabilities.maximum_revision_bytes;
            artifact = match tokio::task::spawn_blocking(move || {
                build_revision_from_snapshot_to_file_bounded(
                    &root_for_build,
                    current,
                    None,
                    true,
                    &build_path_for_task,
                    maximum_revision_bytes,
                )
            })
            .await
            {
                Ok(Ok(artifact)) => artifact,
                Ok(Err(error)) => {
                    let _ = fs::remove_file(&build_path);
                    self.record_checkpoint_failure_locked(&mut state, &error)?;
                    return Err(error);
                }
                Err(join_error) => {
                    let _ = fs::remove_file(&build_path);
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
        artifact: StagedArchiveArtifact,
        lease_generation: u64,
    ) -> Result<()> {
        let staged_path = artifact.path.clone();
        let result = (|| {
            let capabilities = state
                .capabilities
                .as_ref()
                .ok_or_else(|| anyhow!("PipeFS capabilities are unavailable"))?;
            ensure!(
                artifact.size_bytes <= capabilities.maximum_revision_bytes,
                "PipeFS revision is {} bytes, exceeding the server limit of {}",
                artifact.size_bytes,
                capabilities.maximum_revision_bytes
            );
            ensure!(
                artifact.snapshot.logical_size_bytes <= capabilities.maximum_workspace_bytes,
                "PipeFS workspace is {} bytes, exceeding the server limit of {}",
                artifact.snapshot.logical_size_bytes,
                capabilities.maximum_workspace_bytes
            );
            ensure_cache_capacity(
                &self.inner.cache_root,
                artifact.size_bytes,
                "staging a workspace revision",
            )?;
            install_private_archive(&artifact.path, &self.inner.pending_archive)?;
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
                archive_size_bytes: artifact.size_bytes,
                manifest_digest: artifact.manifest_digest,
                logical_size_bytes: artifact.snapshot.logical_size_bytes,
                idempotency_key,
                snapshot: artifact.snapshot,
            });
            state.phase = WorkspacePhase::Pending;
            state.last_error = None;
            self.persist_locked(state)
        })();
        if result.is_err() {
            let _ = fs::remove_file(staged_path);
        }
        result
    }

    async fn retry_locked(&self, state: &mut ControllerState) -> Result<(Uuid, PipeFsRemoteState)> {
        let pending = state
            .pending
            .clone()
            .ok_or_else(|| anyhow!("PipeFS has no staged revision to retry"))?;
        state.retry_count = state.retry_count.saturating_add(1);
        self.persist_locked(state)?;
        let archive_path = self.inner.pending_archive.clone();
        let expected_hash = pending.archive_blake3.clone();
        let expected_size = pending.archive_size_bytes;
        let archive_size_bytes = match tokio::task::spawn_blocking(move || {
            verify_staged_archive(&archive_path, expected_size, &expected_hash)
        })
        .await
        {
            Ok(Ok(size)) => size,
            Ok(Err(error)) => {
                state.phase = WorkspacePhase::Pending;
                state.last_error = Some(error.to_string());
                self.persist_locked(state)?;
                return Err(error);
            }
            Err(join_error) => {
                let error =
                    anyhow!("PipeFS staged archive verification task panicked: {join_error}");
                state.phase = WorkspacePhase::Pending;
                state.last_error = Some(error.to_string());
                self.persist_locked(state)?;
                return Err(error);
            }
        };
        let lease = self.inner.lease.lock().await.clone();
        let result = self
            .inner
            .client
            .commit_archive_file(
                &self.inner.session_id,
                &lease,
                pending.expected_base_revision_id,
                pending.revision_type,
                &self.inner.pending_archive,
                archive_size_bytes,
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
        let capabilities = state
            .capabilities
            .clone()
            .ok_or_else(|| anyhow!("PipeFS capabilities are unavailable for compaction"))?;
        let root_for_scan = root.clone();
        let snapshot = tokio::task::spawn_blocking(move || scan_workspace(&root_for_scan))
            .await
            .context("PipeFS compaction scan task panicked")??;
        ensure_archive_build_capacity(
            &self.inner.cache_root,
            &snapshot,
            None,
            true,
            &capabilities,
            "staging a server-requested full compaction",
        )?;
        let build_path = self.temporary_archive_path("compact");
        let maximum_revision_bytes = capabilities.maximum_revision_bytes;
        let artifact = tokio::task::spawn_blocking(move || {
            build_revision_from_snapshot_to_file_bounded(
                &root,
                snapshot,
                None,
                true,
                &build_path,
                maximum_revision_bytes,
            )
        })
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

    /// Idempotently reconcile remote authority to enabled before an active
    /// runtime retries durability. This is valid even when a disable response
    /// was lost before the local controller reached `Disabled`: the remote
    /// request is authoritative, while dirty and pending local state must be
    /// preserved exactly for the following checkpoint/commit retry.
    pub async fn ensure_remote_enabled(&self) -> Result<()> {
        let mut state = self.inner.state.lock().await;
        let previous_phase = state.phase;
        let enabled_phase = phase_after_remote_enable(previous_phase)?;
        let lease = self.inner.lease.lock().await.clone();
        // Avoid turning an already-enabled restore-only workspace into a
        // write operation during a rollout drain. A mode PUT is required only
        // when the authoritative read proves an ambiguous disable applied.
        let remote = match self.inner.client.state(&self.inner.session_id).await? {
            remote if remote.enabled => remote,
            _ => {
                self.inner
                    .client
                    .set_enabled(&self.inner.session_id, &lease, true)
                    .await?
            }
        };
        self.record_mode_hint(true)?;
        state.remote = Some((&remote).into());
        state.phase = enabled_phase;
        if previous_phase == WorkspacePhase::Disabled {
            state.last_error = None;
        }
        self.persist_locked(&state)
    }

    /// Restore the enabled controller state when the frontend could not rebind
    /// to its original local root after the remote disable transaction. No
    /// workspace bytes changed during that window, so the existing clean
    /// materialization remains authoritative and can safely stay active.
    pub async fn rollback_disable(&self) -> Result<()> {
        self.ensure_remote_enabled().await
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
        write_mode_hint(&self.inner.mode_hint_file, &self.inner.cache_scope, enabled)
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
            materialized_logical_bytes: state
                .snapshot
                .as_ref()
                .map_or(0, |snapshot| snapshot.logical_size_bytes),
            pending_archive_bytes: fs::metadata(&self.inner.pending_archive)
                .map(|metadata| metadata.len())
                .unwrap_or(0),
            available_cache_bytes: available_space_bytes(&self.inner.cache_root),
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
            if validate_controller_cache_scope(&state, &self.inner.cache_scope).is_err() {
                continue;
            }
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

fn validate_controller_cache_scope(
    state: &ControllerState,
    expected: &PipeFsCacheScope,
) -> Result<()> {
    match state.cache_authority_scope.as_deref() {
        Some(actual) if actual == expected.as_str() => Ok(()),
        Some(_) => bail!(
            "PipeFS controller cache authority mismatch; refusing automatic recovery or reuse"
        ),
        None => bail!(
            "legacy unscoped PipeFS controller requires explicit manual recovery; refusing automatic adoption"
        ),
    }
}

fn mark_cache_for_recovery_if_drifted(
    cache_root: &Path,
    marker: &Path,
    cache_scope: &PipeFsCacheScope,
) -> Result<bool> {
    if !cache_root.exists() {
        return Ok(false);
    }
    let state_file = cache_root.join("controller.json");
    if !state_file.exists() {
        return Ok(marker.is_file());
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
    validate_controller_cache_scope(&state, cache_scope).with_context(|| {
        format!(
            "refusing to reuse PipeFS cache with mismatched authority metadata at {}",
            cache_root.display()
        )
    })?;
    if marker.is_file() {
        return Ok(true);
    }
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

fn pending_matches_remote_head(pending: &PendingRevision, remote: &PipeFsRemoteState) -> bool {
    remote.current_head.is_some()
        && remote.manifest_digest.as_deref() == Some(pending.manifest_digest.as_str())
        && remote.logical_size_bytes == pending.logical_size_bytes
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

fn ensure_supported_platform() -> Result<()> {
    #[cfg(unix)]
    {
        Ok(())
    }
    #[cfg(not(unix))]
    {
        bail!("PipeFS v1 requires a Unix host (Linux or macOS); Windows is not supported")
    }
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

fn phase_after_remote_enable(phase: WorkspacePhase) -> Result<WorkspacePhase> {
    match phase {
        WorkspacePhase::Clean | WorkspacePhase::Dirty | WorkspacePhase::Pending => Ok(phase),
        WorkspacePhase::Disabled => Ok(WorkspacePhase::Clean),
        WorkspacePhase::Restoring | WorkspacePhase::LeaseLost => bail!(
            "PipeFS workspace cannot reconcile remote authority from phase {:?}",
            phase
        ),
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

const CACHE_FREE_SPACE_RESERVE_BYTES: u64 = 64 * 1024 * 1024;

fn ensure_archive_build_capacity(
    cache_root: &Path,
    snapshot: &Snapshot,
    base: Option<&Snapshot>,
    force_full: bool,
    capabilities: &CapabilitiesDisk,
    operation: &str,
) -> Result<()> {
    ensure!(
        snapshot.logical_size_bytes <= capabilities.maximum_workspace_bytes,
        "PipeFS workspace is {} bytes, exceeding the server limit of {}",
        snapshot.logical_size_bytes,
        capabilities.maximum_workspace_bytes
    );
    let worst_case = revision_archive_size_upper_bound(snapshot, base, force_full)?;
    // Never reserve more than can be emitted: the encoder below is hard
    // bounded at the negotiated revision limit and removes partial output.
    ensure_cache_capacity(
        cache_root,
        worst_case.min(capabilities.maximum_revision_bytes),
        operation,
    )
}

fn ensure_cache_capacity(path: &Path, operation_bytes: u64, operation: &str) -> Result<()> {
    let Some(available) = available_space_bytes(path) else {
        return Ok(());
    };
    let required = operation_bytes.saturating_add(CACHE_FREE_SPACE_RESERVE_BYTES);
    ensure!(
        available >= required,
        "insufficient PipeFS cache space for {operation}: {} available, {} required (including safety reserve); remove stale recovery caches only after exporting/reconciling them",
        format_bytes(available),
        format_bytes(required)
    );
    Ok(())
}

#[cfg(unix)]
fn available_space_bytes(path: &Path) -> Option<u64> {
    let mut existing = path;
    while !existing.exists() {
        existing = existing.parent()?;
    }
    let stats = rustix::fs::statvfs(existing).ok()?;
    Some(stats.f_bavail.saturating_mul(stats.f_frsize))
}

#[cfg(not(unix))]
fn available_space_bytes(_path: &Path) -> Option<u64> {
    None
}

fn format_bytes(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else {
        format!("{bytes} bytes")
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
pub fn local_state_requires_remote_probe(cache_scope: &PipeFsCacheScope, session_id: &str) -> bool {
    local_state_requires_remote_probe_at(&default_cache_base(), cache_scope, session_id)
}

/// Update the default-cache mode hint after an authoritative read observes a
/// remote disable performed by another machine.
pub fn record_local_mode_hint(
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
    enabled: bool,
) -> Result<()> {
    validate_session_id(session_id)?;
    let base = default_cache_base();
    prepare_authority_cache_root(&base, cache_scope)?;
    let session_root = session_cache_root(&base, cache_scope, session_id)?;
    create_private_dir(&session_root)?;
    write_mode_hint(&session_root.join("remote-mode"), cache_scope, enabled)
}

fn local_state_requires_remote_probe_at(
    base: &Path,
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
) -> bool {
    if validate_session_id(session_id).is_err() {
        return true;
    }
    if local_recovery_required_at(base, cache_scope, session_id) {
        return true;
    }
    let Ok(session_root) = session_cache_root(base, cache_scope, session_id) else {
        return true;
    };
    match fs::read(session_root.join("remote-mode")) {
        Ok(value) => serde_json::from_slice::<ModeHint>(&value).map_or(true, |hint| {
            hint.version != 1 || hint.cache_authority_scope != cache_scope.as_str() || hint.enabled
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

fn write_mode_hint(path: &Path, cache_scope: &PipeFsCacheScope, enabled: bool) -> Result<()> {
    let hint = ModeHint {
        version: 1,
        cache_authority_scope: cache_scope.as_str().to_string(),
        enabled,
    };
    write_private(path, &serde_json::to_vec(&hint)?)
}

/// A prior process left materialized bytes that were not proven represented by
/// the remote head. Startup uses this stronger signal to re-enter PipeFS even
/// if the last mode transaction reached "disabled" before the process died.
pub fn local_recovery_required(cache_scope: &PipeFsCacheScope, session_id: &str) -> bool {
    local_recovery_required_at(&default_cache_base(), cache_scope, session_id)
}

/// List crash-recovery caches for one canonical remote session.
///
/// Only real generation directories containing a real `recovery-required`
/// marker are returned. Corrupt controller metadata is reported on the entry
/// instead of causing the cache to disappear from recovery UX.
pub fn list_recovery_caches(
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
) -> Result<Vec<PipeFsRecoveryCache>> {
    list_recovery_caches_at(&default_cache_base(), cache_scope, session_id)
}

/// Inspect one exact, session-scoped recovery cache identifier.
pub fn inspect_recovery_cache(
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
    cache_id: &str,
) -> Result<PipeFsRecoveryCache> {
    validate_recovery_cache_id(cache_id)?;
    let cache = list_recovery_caches(cache_scope, session_id)?
        .into_iter()
        .find(|cache| cache.id == cache_id)
        .ok_or_else(|| {
            anyhow!(
                "PipeFS recovery cache {cache_id:?} was not found for this authority and session"
            )
        })?;
    validate_recovery_cache_scope(&cache.path, cache_scope)?;
    Ok(cache)
}

/// Export the materialized tree from a recovery cache as a deterministic full
/// PipeFS archive. The source cache is deliberately retained.
pub fn export_recovery_cache(
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
    cache_id: &str,
    destination: &Path,
) -> Result<PathBuf> {
    let cache = inspect_recovery_cache(cache_scope, session_id, cache_id)?;
    let workspace_root = recovery_workspace_root(&cache)?;
    let destination = absolute_path(destination)?;
    let file_name = destination
        .file_name()
        .ok_or_else(|| anyhow!("PipeFS recovery export destination has no file name"))?;
    let parent = destination.parent().ok_or_else(|| {
        anyhow!(
            "PipeFS recovery export destination has no parent: {}",
            destination.display()
        )
    })?;
    let parent = parent.canonicalize().with_context(|| {
        format!(
            "canonicalizing PipeFS recovery export parent {}",
            parent.display()
        )
    })?;
    ensure!(
        parent.is_dir(),
        "PipeFS recovery export parent is not a directory: {}",
        parent.display()
    );
    let destination = parent.join(file_name);
    let session_root = session_cache_root(&default_cache_base(), cache_scope, session_id)?;
    let canonical_session_root = session_root.canonicalize().with_context(|| {
        format!(
            "canonicalizing PipeFS session cache {}",
            session_root.display()
        )
    })?;
    ensure!(
        !destination.starts_with(&canonical_session_root),
        "refusing to export a PipeFS recovery archive inside its managed cache"
    );
    match fs::symlink_metadata(&destination) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => bail!(
            "PipeFS recovery export destination already exists: {}",
            destination.display()
        ),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "checking PipeFS recovery export destination {}",
                    destination.display()
                )
            });
        }
    }
    let snapshot = scan_workspace(&workspace_root)
        .context("scanning the PipeFS recovery workspace for export")?;
    let maximum_archive_bytes = revision_archive_size_upper_bound(&snapshot, None, true)?;
    ensure_cache_capacity(
        &parent,
        maximum_archive_bytes,
        "exporting the recovery workspace",
    )?;
    build_revision_from_snapshot_to_file_bounded(
        &workspace_root,
        snapshot,
        None,
        true,
        &destination,
        maximum_archive_bytes,
    )
    .context("building full PipeFS recovery export")?;
    Ok(destination)
}

/// Permanently discard one recovery cache. The confirmation must exactly
/// equal the cache identifier so a stale path, wildcard, or session ID cannot
/// broaden the deletion target.
pub fn discard_recovery_cache(
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
    cache_id: &str,
    confirmation: &str,
) -> Result<()> {
    ensure!(
        confirmation == cache_id,
        "recovery cache confirmation must exactly match {cache_id:?}"
    );
    let cache = inspect_recovery_cache(cache_scope, session_id, cache_id)?;
    let session_root = session_cache_root(&default_cache_base(), cache_scope, session_id)?;
    remove_cache_directory(&session_root, &cache.path)
}

fn local_recovery_required_at(
    base: &Path,
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
) -> bool {
    if validate_session_id(session_id).is_err() {
        return true;
    }
    let Ok(session_root) = session_cache_root(base, cache_scope, session_id) else {
        return true;
    };
    match fs::read_dir(&session_root) {
        Ok(entries) => entries.filter_map(std::result::Result::ok).any(|entry| {
            entry.file_type().is_ok_and(|kind| kind.is_dir())
                && fs::symlink_metadata(entry.path().join("recovery-required"))
                    .is_ok_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(_) => true,
    }
}

fn session_cache_root(
    base: &Path,
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
) -> Result<PathBuf> {
    validate_session_id(session_id)?;
    let session_digest = blake3::hash(session_id.as_bytes()).to_hex().to_string();
    Ok(base
        .join(cache_scope.directory_name())
        .join(&session_digest[..32]))
}

fn prepare_authority_cache_root(base: &Path, cache_scope: &PipeFsCacheScope) -> Result<PathBuf> {
    // The chosen cache base is itself part of the private-data boundary. In
    // particular, the `/tmp` fallback must not inherit a planted symlink or
    // remain world-readable merely because only its descendants were checked.
    create_private_dir(base)?;
    let authority = base.join(cache_scope.directory_name());
    create_private_dir(&authority)?;
    Ok(authority)
}

fn validate_recovery_cache_id(cache_id: &str) -> Result<()> {
    ensure!(
        !cache_id.is_empty()
            && cache_id.len() <= 128
            && cache_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "invalid PipeFS recovery cache identifier"
    );
    Ok(())
}

fn list_recovery_caches_at(
    base: &Path,
    cache_scope: &PipeFsCacheScope,
    session_id: &str,
) -> Result<Vec<PipeFsRecoveryCache>> {
    let session_root = session_cache_root(base, cache_scope, session_id)?;
    let entries = match fs::read_dir(&session_root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "reading PipeFS recovery cache directory {}",
                    session_root.display()
                )
            });
        }
    };
    let session_metadata = fs::symlink_metadata(&session_root)?;
    ensure!(
        session_metadata.is_dir() && !session_metadata.file_type().is_symlink(),
        "PipeFS session cache is not a real directory: {}",
        session_root.display()
    );

    let mut caches = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            continue;
        }
        let marker = path.join("recovery-required");
        let marker_metadata = match fs::symlink_metadata(&marker) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if !marker_metadata.is_file() || marker_metadata.file_type().is_symlink() {
            continue;
        }
        let Some(id) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if validate_recovery_cache_id(id).is_err() {
            continue;
        }

        let controller = load_scoped_recovery_controller(&path, cache_scope);
        let (workspace_root, phase, logical_size_bytes, last_error) = match controller {
            Ok(state) => {
                let logical_size = state
                    .snapshot
                    .as_ref()
                    .map_or(0, |snapshot| snapshot.logical_size_bytes);
                let mut error = state.last_error;
                let workspace_root = match state.materialized_root {
                    Some(root) => match validate_recovery_workspace_root(&path, &root) {
                        Ok(root) => Some(root),
                        Err(root_error) => {
                            error = Some(match error {
                                Some(error) => format!("{error}; {root_error:#}"),
                                None => format!("{root_error:#}"),
                            });
                            None
                        }
                    },
                    None => None,
                };
                (workspace_root, Some(state.phase), logical_size, error)
            }
            Err(error) => (None, None, 0, Some(format!("{error:#}"))),
        };
        let pending_archive = path.join("pending.tar.zst");
        let pending_archive_bytes = fs::symlink_metadata(pending_archive)
            .ok()
            .filter(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
            .map_or(0, |metadata| metadata.len());
        caches.push(PipeFsRecoveryCache {
            id: id.to_string(),
            path,
            workspace_root,
            phase,
            logical_size_bytes,
            pending_archive_bytes,
            last_error,
        });
    }
    caches.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(caches)
}

fn load_scoped_recovery_controller(
    cache: &Path,
    cache_scope: &PipeFsCacheScope,
) -> Result<ControllerState> {
    let controller_path = cache.join("controller.json");
    let metadata =
        fs::symlink_metadata(&controller_path).context("reading recovery controller metadata")?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "recovery controller is not a regular non-symlink file"
    );
    let state: ControllerState = serde_json::from_slice(
        &fs::read(&controller_path).context("reading recovery controller state")?,
    )
    .context("parsing recovery controller state")?;
    validate_controller_cache_scope(&state, cache_scope)?;
    Ok(state)
}

fn validate_recovery_cache_scope(cache: &Path, cache_scope: &PipeFsCacheScope) -> Result<()> {
    load_scoped_recovery_controller(cache, cache_scope).map(|_| ())
}

fn validate_recovery_workspace_root(cache: &Path, workspace_root: &Path) -> Result<PathBuf> {
    ensure!(
        workspace_root.starts_with(cache) && workspace_root != cache,
        "recovery cache points outside its generation directory"
    );
    let canonical_cache = cache.canonicalize()?;
    let canonical_root = workspace_root.canonicalize().with_context(|| {
        format!(
            "canonicalizing recovery workspace {}",
            workspace_root.display()
        )
    })?;
    ensure!(
        canonical_root.starts_with(&canonical_cache) && canonical_root != canonical_cache,
        "recovery workspace resolves outside its generation directory"
    );
    let metadata = fs::symlink_metadata(&canonical_root)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "recovery workspace is not a real directory"
    );
    Ok(canonical_root)
}

fn recovery_workspace_root(cache: &PipeFsRecoveryCache) -> Result<PathBuf> {
    let workspace_root = cache.workspace_root.as_ref().ok_or_else(|| {
        anyhow!(
            "PipeFS recovery cache {:?} has no safe materialized workspace{}",
            cache.id,
            cache
                .last_error
                .as_deref()
                .map(|error| format!(": {error}"))
                .unwrap_or_default()
        )
    })?;
    validate_recovery_workspace_root(&cache.path, workspace_root)
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

fn install_private_archive(source: &Path, destination: &Path) -> Result<()> {
    ensure!(
        source.parent() == destination.parent(),
        "PipeFS staged archive must remain in its private cache directory"
    );
    let source_metadata = fs::symlink_metadata(source)
        .with_context(|| format!("reading staged PipeFS archive {}", source.display()))?;
    ensure!(
        source_metadata.is_file() && !source_metadata.file_type().is_symlink(),
        "PipeFS staged archive is not a regular file"
    );
    if let Ok(destination_metadata) = fs::symlink_metadata(destination) {
        ensure!(
            destination_metadata.is_file() && !destination_metadata.file_type().is_symlink(),
            "refusing to replace a non-regular PipeFS pending archive"
        );
    }
    fs::rename(source, destination).with_context(|| {
        format!(
            "atomically staging PipeFS archive {}",
            destination.display()
        )
    })?;
    if let Some(parent) = destination.parent() {
        fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn verify_staged_archive(path: &Path, expected_size: u64, expected_hash: &str) -> Result<u64> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading staged PipeFS archive {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "local recovery archive is not a regular file"
    );
    if expected_size != 0 {
        ensure!(
            metadata.len() == expected_size,
            "local recovery archive has size {}, expected {expected_size}",
            metadata.len()
        );
    }
    #[cfg(unix)]
    let mut file = fs::File::from(
        rustix::fs::open(
            path,
            rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        )
        .map_err(std::io::Error::from)?,
    );
    #[cfg(not(unix))]
    let mut file = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 128 * 1024];
    let mut size = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(read as u64)
            .ok_or_else(|| anyhow!("local recovery archive size overflow"))?;
        hasher.update(&buffer[..read]);
    }
    ensure!(
        expected_size == 0 || size == expected_size,
        "local recovery archive changed size while being verified"
    );
    ensure!(
        hasher.finalize().to_hex().as_str() == expected_hash,
        "local recovery archive failed BLAKE3 verification"
    );
    Ok(size)
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
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .filter(|path| current != Some(path.as_path()))
        .filter(|path| {
            fs::symlink_metadata(path.join("recovery-required"))
                .is_ok_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cache_scope() -> PipeFsCacheScope {
        PipeFsClient::new(crate::PipeFsClientConfig::new(
            "http://127.0.0.1:1",
            "test-key",
        ))
        .unwrap()
        .cache_scope()
    }

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
                writes_available: true,
                restore_available: true,
            }),
            ..ControllerState::default()
        }
    }

    #[test]
    fn remote_enable_reconciliation_preserves_uncommitted_phase() {
        assert_eq!(
            phase_after_remote_enable(WorkspacePhase::Clean).unwrap(),
            WorkspacePhase::Clean
        );
        assert_eq!(
            phase_after_remote_enable(WorkspacePhase::Dirty).unwrap(),
            WorkspacePhase::Dirty
        );
        assert_eq!(
            phase_after_remote_enable(WorkspacePhase::Pending).unwrap(),
            WorkspacePhase::Pending
        );
        assert_eq!(
            phase_after_remote_enable(WorkspacePhase::Disabled).unwrap(),
            WorkspacePhase::Clean
        );
        assert!(phase_after_remote_enable(WorkspacePhase::Restoring).is_err());
        assert!(phase_after_remote_enable(WorkspacePhase::LeaseLost).is_err());
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

    #[test]
    fn recovery_recognizes_a_pending_revision_that_already_became_remote_head() {
        let pending = PendingRevision {
            expected_base_revision_id: Some(Uuid::nil()),
            revision_type: RevisionKind::Delta,
            archive_blake3: "b".repeat(64),
            archive_size_bytes: 123,
            manifest_digest: "a".repeat(64),
            logical_size_bytes: 42,
            idempotency_key: "retry-key".to_string(),
            snapshot: Snapshot::default(),
        };
        let remote = PipeFsRemoteState {
            session_id: "recovered-session".to_string(),
            enabled: true,
            current_head: Some(Uuid::new_v4()),
            sequence: 2,
            manifest_digest: Some("a".repeat(64)),
            logical_size_bytes: 42,
            restore_chain: Vec::new(),
        };

        assert!(pending_matches_remote_head(&pending, &remote));
        let mut wrong_size = remote.clone();
        wrong_size.logical_size_bytes += 1;
        assert!(!pending_matches_remote_head(&pending, &wrong_size));
        let mut wrong_manifest = remote;
        wrong_manifest.manifest_digest = Some("c".repeat(64));
        assert!(!pending_matches_remote_head(&pending, &wrong_manifest));
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
                cache_scope: test_cache_scope(),
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
            writes_available: true,
            restore_available: true,
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
            archive_size_bytes: 1,
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

    #[tokio::test]
    async fn restored_drain_mode_workspace_rejects_mutation_before_local_bytes_change() {
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
                session_id: "read-only-drain-test".to_string(),
                cache_scope: test_cache_scope(),
                original_workspace_root: workspace_root.clone(),
                original_state_root: state_root,
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap();
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Clean;
        state.materialized_root = Some(workspace_root);
        state.capabilities = Some(CapabilitiesDisk {
            maximum_revision_bytes: 1_000_000,
            maximum_workspace_bytes: 1_000_000,
            maximum_delta_chain: 20,
            writes_available: false,
            restore_available: true,
        });
        drop(state);

        let error = workspace
            .mutation_started(Some(vec!["blocked.txt".to_string()]))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("read-only"));
        let state = workspace.inner.state.lock().await;
        assert_eq!(state.phase, WorkspacePhase::Clean);
        assert!(state.dirty_paths.is_empty());
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

    #[cfg(unix)]
    #[test]
    fn cache_base_is_private_and_rejects_a_symlink() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let temporary = tempfile::tempdir().unwrap();
        let base = temporary.path().join("cache-base");
        fs::create_dir(&base).unwrap();
        fs::set_permissions(&base, fs::Permissions::from_mode(0o755)).unwrap();
        let scope = test_cache_scope();

        let authority = prepare_authority_cache_root(&base, &scope).unwrap();
        assert!(authority.is_dir());
        assert_eq!(
            fs::metadata(&base).unwrap().permissions().mode() & 0o777,
            0o700
        );

        let target = temporary.path().join("symlink-target");
        fs::create_dir(&target).unwrap();
        let linked_base = temporary.path().join("linked-cache-base");
        symlink(&target, &linked_base).unwrap();
        let error = prepare_authority_cache_root(&linked_base, &scope).unwrap_err();
        assert!(error.to_string().contains("not a real directory"));
        assert!(fs::read_dir(&target).unwrap().next().is_none());
    }

    #[test]
    fn owned_temporary_guard_cleans_panics_and_preserves_disarmed_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let source_recovery = temporary.path().join("source-recovery");
        fs::create_dir(&source_recovery).unwrap();
        fs::write(source_recovery.join("user-data"), b"preserve").unwrap();

        let panic_path = std::sync::Mutex::new(None);
        let result = std::panic::catch_unwind(|| {
            let archive = OwnedCacheTemporary::archive(temporary.path(), "panic");
            *panic_path.lock().unwrap() = Some(archive.path().to_path_buf());
            fs::write(archive.path(), b"partial").unwrap();
            panic!("simulated archive builder panic");
        });
        assert!(result.is_err());
        assert!(!panic_path.lock().unwrap().as_ref().unwrap().exists());

        let mut activated = OwnedCacheTemporary::directory(temporary.path(), "workspace");
        fs::create_dir(activated.path()).unwrap();
        let activated_path = activated.path().to_path_buf();
        activated.disarm();
        drop(activated);
        assert!(activated_path.is_dir());
        assert_eq!(
            fs::read(source_recovery.join("user-data")).unwrap(),
            b"preserve"
        );
    }

    #[tokio::test]
    async fn owned_temporary_guard_cleans_a_cancelled_recovery_task() {
        let temporary = tempfile::tempdir().unwrap();
        let recovery = OwnedCacheTemporary::directory(temporary.path(), "recovery");
        fs::create_dir(recovery.path()).unwrap();
        fs::write(recovery.path().join("partial"), b"partial").unwrap();
        let recovery_path = recovery.path().to_path_buf();
        let task = tokio::spawn(async move {
            let _recovery = recovery;
            std::future::pending::<()>().await;
        });
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;

        assert!(!recovery_path.exists());
    }

    #[test]
    fn cache_and_recovery_are_isolated_by_authenticated_authority() {
        let temporary = tempfile::tempdir().unwrap();
        let cache_base = temporary.path().join("cache");
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        let session_id = "shared-session-id";
        let client_a = PipeFsClient::new(crate::PipeFsClientConfig::new(
            "https://one.example.test/v1",
            "account-a-key",
        ))
        .unwrap();
        let client_b = PipeFsClient::new(crate::PipeFsClientConfig::new(
            "https://one.example.test/v1",
            "account-b-key",
        ))
        .unwrap();
        let client_c = PipeFsClient::new(crate::PipeFsClientConfig::new(
            "https://two.example.test/v1",
            "account-a-key",
        ))
        .unwrap();
        let scope_a = client_a.cache_scope();
        let scope_b = client_b.cache_scope();
        let scope_c = client_c.cache_scope();
        let make_workspace = |client: PipeFsClient, cache_scope: PipeFsCacheScope| {
            PipeFsWorkspace::new(
                client,
                PipeFsLease {
                    token: "token".to_string(),
                    generation: 7,
                },
                PipeFsWorkspaceConfig {
                    session_id: session_id.to_string(),
                    cache_scope,
                    original_workspace_root: workspace_root.clone(),
                    original_state_root: state_root.clone(),
                    cache_base: Some(cache_base.clone()),
                },
            )
            .unwrap()
        };
        let workspace_a = make_workspace(client_a, scope_a.clone());
        let workspace_b = make_workspace(client_b, scope_b.clone());
        let workspace_c = make_workspace(client_c, scope_c.clone());

        assert_ne!(
            workspace_a.inner.session_cache_root,
            workspace_b.inner.session_cache_root
        );
        assert_ne!(
            workspace_a.inner.session_cache_root,
            workspace_c.inner.session_cache_root
        );
        assert_ne!(workspace_a.inner.cache_root, workspace_b.inner.cache_root);
        assert_ne!(workspace_a.inner.cache_root, workspace_c.inner.cache_root);

        let state = workspace_a.inner.state.blocking_lock();
        workspace_a.persist_locked(&state).unwrap();
        drop(state);
        write_private(
            &workspace_a.inner.recovery_marker,
            b"uncommitted account A workspace\n",
        )
        .unwrap();
        workspace_a.record_mode_hint(true).unwrap();

        assert!(local_recovery_required_at(
            &cache_base,
            &scope_a,
            session_id
        ));
        assert!(!local_recovery_required_at(
            &cache_base,
            &scope_b,
            session_id
        ));
        assert!(!local_recovery_required_at(
            &cache_base,
            &scope_c,
            session_id
        ));
        assert!(local_state_requires_remote_probe_at(
            &cache_base,
            &scope_a,
            session_id
        ));
        assert!(!local_state_requires_remote_probe_at(
            &cache_base,
            &scope_b,
            session_id
        ));
        assert_eq!(
            list_recovery_caches_at(&cache_base, &scope_a, session_id)
                .unwrap()
                .len(),
            1
        );
        assert!(
            list_recovery_caches_at(&cache_base, &scope_b, session_id)
                .unwrap()
                .is_empty()
        );

        // Pre-scope cache layouts stay untouched and invisible to automatic
        // recovery. Their bytes remain available for explicit manual salvage.
        let legacy_digest = blake3::hash(session_id.as_bytes()).to_hex().to_string();
        let legacy_generation = cache_base.join(&legacy_digest[..32]).join("generation");
        create_private_dir(&legacy_generation).unwrap();
        write_private(
            &legacy_generation.join("recovery-required"),
            b"legacy unscoped cache; recover manually\n",
        )
        .unwrap();
        assert!(!local_recovery_required_at(
            &cache_base,
            &scope_b,
            session_id
        ));
        assert!(
            list_recovery_caches_at(&cache_base, &scope_b, session_id)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn controller_scope_mismatch_fails_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        let client = PipeFsClient::new(crate::PipeFsClientConfig::new(
            "https://one.example.test/v1",
            "account-a-key",
        ))
        .unwrap();
        let scope_a = client.cache_scope();
        let scope_b = PipeFsClient::new(crate::PipeFsClientConfig::new(
            "https://one.example.test/v1",
            "account-b-key",
        ))
        .unwrap()
        .cache_scope();
        let config = PipeFsWorkspaceConfig {
            session_id: "scope-mismatch".to_string(),
            cache_scope: scope_a.clone(),
            original_workspace_root: workspace_root,
            original_state_root: state_root,
            cache_base: Some(temporary.path().join("cache")),
        };
        let workspace = PipeFsWorkspace::new(
            client.clone(),
            PipeFsLease {
                token: "token".to_string(),
                generation: 1,
            },
            config.clone(),
        )
        .unwrap();
        let mismatched = ControllerState::for_cache_scope(&scope_b);
        write_private(
            &workspace.inner.state_file,
            &serde_json::to_vec(&mismatched).unwrap(),
        )
        .unwrap();
        drop(workspace);

        let error = PipeFsWorkspace::new(
            client,
            PipeFsLease {
                token: "token".to_string(),
                generation: 1,
            },
            config,
        )
        .err()
        .expect("mismatched controller must be rejected");
        assert!(error.to_string().contains("mismatched authority"));

        let wrong_scope_error = PipeFsWorkspace::new(
            PipeFsClient::new(crate::PipeFsClientConfig::new(
                "https://one.example.test/v1",
                "account-a-key",
            ))
            .unwrap(),
            PipeFsLease {
                token: "token".to_string(),
                generation: 2,
            },
            PipeFsWorkspaceConfig {
                session_id: "client-config-scope-mismatch".to_string(),
                cache_scope: scope_b,
                original_workspace_root: temporary.path().join("workspace"),
                original_state_root: temporary.path().join("state"),
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .err()
        .expect("client/config scope mismatch must be rejected");
        assert!(
            wrong_scope_error
                .to_string()
                .contains("authenticated client")
        );
    }

    #[test]
    fn session_mode_hint_survives_generation_cleanup_and_dirty_cache_fails_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let base = temporary.path().join("cache");
        let session_id = "mode-hint-test";
        let cache_scope = test_cache_scope();
        let session_root = session_cache_root(&base, &cache_scope, session_id).unwrap();
        create_private_dir(&session_root).unwrap();

        assert!(!local_state_requires_remote_probe_at(
            &base,
            &cache_scope,
            session_id
        ));
        assert!(!local_recovery_required_at(&base, &cache_scope, session_id));
        write_mode_hint(&session_root.join("remote-mode"), &cache_scope, true).unwrap();
        assert!(local_state_requires_remote_probe_at(
            &base,
            &cache_scope,
            session_id
        ));
        write_mode_hint(&session_root.join("remote-mode"), &cache_scope, false).unwrap();
        assert!(!local_state_requires_remote_probe_at(
            &base,
            &cache_scope,
            session_id
        ));

        let generation = session_root.join("generation");
        create_private_dir(&generation).unwrap();
        write_private(
            &generation.join("recovery-required"),
            b"uncommitted workspace changes\n",
        )
        .unwrap();
        assert!(local_state_requires_remote_probe_at(
            &base,
            &cache_scope,
            session_id
        ));
        assert!(local_recovery_required_at(&base, &cache_scope, session_id));
        fs::remove_dir_all(generation).unwrap();

        write_private(&session_root.join("remote-mode"), b"corrupt\n").unwrap();
        assert!(local_state_requires_remote_probe_at(
            &base,
            &cache_scope,
            session_id
        ));
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
                cache_scope: test_cache_scope(),
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
                cache_scope: test_cache_scope(),
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
                archive_size_bytes: 6,
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
                    cache_scope: test_cache_scope(),
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
            cache_scope: test_cache_scope(),
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
                    cache_scope: test_cache_scope(),
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

    #[cfg(unix)]
    #[test]
    fn recovery_discovery_never_follows_generation_or_marker_symlinks() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let session_root = temporary.path().join("session");
        let outside = temporary.path().join("outside");
        fs::create_dir_all(&session_root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("recovery-required"), b"outside\n").unwrap();

        symlink(&outside, session_root.join("linked-generation")).unwrap();

        let linked_marker = session_root.join("linked-marker");
        fs::create_dir_all(&linked_marker).unwrap();
        symlink(
            outside.join("recovery-required"),
            linked_marker.join("recovery-required"),
        )
        .unwrap();

        let genuine = session_root.join("genuine");
        fs::create_dir_all(&genuine).unwrap();
        fs::write(genuine.join("recovery-required"), b"recover\n").unwrap();

        assert_eq!(recovery_caches(&session_root, None), vec![genuine]);
    }

    #[tokio::test]
    async fn rejects_workspace_larger_than_server_capability_before_staging() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace_root = temporary.path().join("workspace");
        let state_root = temporary.path().join("state");
        fs::create_dir_all(&workspace_root).unwrap();
        fs::create_dir_all(&state_root).unwrap();
        fs::write(workspace_root.join("large"), b"0123456789").unwrap();
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
                cache_scope: test_cache_scope(),
                original_workspace_root: workspace_root.clone(),
                original_state_root: state_root,
                cache_base: Some(temporary.path().join("cache")),
            },
        )
        .unwrap();
        let mut state = workspace.inner.state.lock().await;
        state.phase = WorkspacePhase::Dirty;
        state.materialized_root = Some(workspace_root);
        state.snapshot = Some(Snapshot::default());
        state.capabilities = Some(CapabilitiesDisk {
            maximum_revision_bytes: 1_000_000,
            maximum_workspace_bytes: 5,
            maximum_delta_chain: 20,
            writes_available: true,
            restore_available: true,
        });
        workspace.persist_locked(&state).unwrap();
        drop(state);

        let error = workspace.checkpoint().await.unwrap_err();

        assert!(error.to_string().contains("exceeding the server limit"));
        let state = workspace.inner.state.lock().await;
        assert!(state.pending.is_none());
        assert!(!workspace.inner.pending_archive.exists());
        assert!(
            fs::read_dir(&workspace.inner.cache_root)
                .unwrap()
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().ends_with(".tar.zst"))
        );
    }
}
