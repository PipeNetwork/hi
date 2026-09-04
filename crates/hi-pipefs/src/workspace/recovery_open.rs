use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use super::{
    PipeFsCacheScope, PipeFsClient, PipeFsLease, PipeFsWorkspace, PipeFsWorkspaceConfig,
    cache_has_pending_operation, validate_recovery_cache_id, validate_recovery_cache_scope,
};

impl PipeFsWorkspace {
    pub(crate) fn open_recovery_cache(
        client: PipeFsClient,
        lease: PipeFsLease,
        config: PipeFsWorkspaceConfig,
        cache_id: &str,
    ) -> Result<Self> {
        validate_recovery_cache_id(cache_id)?;
        Self::new_inner(client, lease, config, Some(cache_id))
    }
}

pub(super) fn validated_cache_root(
    session_cache_root: &Path,
    cache_id: &str,
    cache_scope: &PipeFsCacheScope,
) -> Result<PathBuf> {
    let cache_root = session_cache_root.join(cache_id);
    let metadata = fs::symlink_metadata(&cache_root)
        .with_context(|| format!("opening PipeFS recovery cache {}", cache_root.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "PipeFS recovery cache is not a real directory"
    );
    validate_recovery_cache_scope(&cache_root, cache_scope)?;
    let marker = fs::symlink_metadata(cache_root.join("recovery-required"))?;
    ensure!(
        marker.is_file() && !marker.file_type().is_symlink(),
        "PipeFS recovery cache has no regular recovery marker"
    );
    ensure!(
        cache_has_pending_operation(&cache_root, cache_scope)?,
        "PipeFS recovery cache has no exact pending operation evidence"
    );
    Ok(cache_root)
}
