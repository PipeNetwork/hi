//! Read-only remote workspace export into a fresh destination.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use uuid::Uuid;

use crate::{
    ARCHIVE_VERSION, PipeFsClient, RestoreRevision, RevisionKind, Snapshot, apply_archive_file,
    scan_workspace,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteExportReceipt {
    pub destination: PathBuf,
    pub revision_id: Option<Uuid>,
    pub manifest_digest: Option<String>,
    pub entry_count: usize,
    pub logical_size_bytes: u64,
}

/// Materialize a verified remote revision through a private sibling and rename
/// it into a destination that never existed. This is read-only with respect to
/// remote authority and never touches the process launch workspace.
pub async fn export_remote_workspace(
    client: &PipeFsClient,
    session_id: &str,
    requested_revision: Option<Uuid>,
    destination: &Path,
) -> Result<RemoteExportReceipt> {
    validate_session_id(session_id)?;
    let capabilities = client
        .capabilities()
        .await
        .context("reading PipeFS export capabilities")?;
    ensure!(
        capabilities.archive_version == ARCHIVE_VERSION,
        "unsupported PipeFS archive version {}",
        capabilities.archive_version
    );
    ensure!(
        capabilities.restore_available(),
        "PipeFS restore/export is disabled by the server"
    );
    let remote = client
        .state(session_id)
        .await
        .context("reading the remote PipeFS workspace head")?;
    ensure!(
        remote.session_id == session_id,
        "PipeFS server returned state for a different session"
    );
    let (chain, target) = select_restore_chain(
        &remote.restore_chain,
        remote.current_head,
        requested_revision,
    )?;

    let destination = fresh_destination(destination)?;
    let parent = destination
        .parent()
        .expect("fresh_destination always returns a named child");
    let temporary = TemporaryExport::new(parent)?;
    let materialized = temporary.path.join("materialized");
    create_private_directory(&materialized)?;

    let mut snapshot: Option<Snapshot> = None;
    let mut prior_revision = None;
    let mut prior_sequence = None;
    for revision in &chain {
        validate_chain_revision(revision, prior_revision, prior_sequence, snapshot.is_none())?;
        let archive = temporary
            .path
            .join(format!("revision-{}.tar.zst", revision.revision_id));
        client
            .download_revision_to_file(
                session_id,
                revision,
                capabilities.maximum_revision_bytes,
                &archive,
            )
            .await
            .with_context(|| format!("downloading PipeFS revision {}", revision.revision_id))?;
        let root = materialized.clone();
        let archive_for_apply = archive.clone();
        let expected_base = snapshot.clone();
        let restored = tokio::task::spawn_blocking(move || {
            apply_archive_file(&root, &archive_for_apply, expected_base.as_ref())
        })
        .await
        .context("PipeFS export extraction task panicked")??;
        let _ = fs::remove_file(&archive);
        ensure!(
            restored.manifest_digest.as_deref() == Some(&revision.manifest_digest),
            "restored manifest does not match revision {}",
            revision.revision_id
        );
        ensure!(
            restored.logical_size_bytes == revision.logical_size_bytes,
            "restored logical size does not match revision {}",
            revision.revision_id
        );
        snapshot = Some(restored);
        prior_revision = Some(revision.revision_id);
        prior_sequence = Some(revision.sequence);
    }

    let snapshot = match snapshot {
        Some(snapshot) => snapshot,
        None => scan_workspace(&materialized)?,
    };
    ensure!(
        snapshot.logical_size_bytes <= capabilities.maximum_workspace_bytes,
        "restored workspace exceeds the negotiated size limit"
    );
    ensure!(
        prior_revision == target,
        "restore chain does not end at requested revision"
    );
    publish_export(&materialized, &destination)?;
    sync_directory(parent);
    Ok(RemoteExportReceipt {
        destination,
        revision_id: target,
        manifest_digest: snapshot.manifest_digest,
        entry_count: snapshot.entries.len(),
        logical_size_bytes: snapshot.logical_size_bytes,
    })
}

fn publish_export(source: &Path, destination: &Path) -> Result<()> {
    publish_export_after(
        || {
            hi_workspace::hit_harness_failpoint(hi_workspace::HarnessFailpoint::ExportBeforeRename)
                .map_err(anyhow::Error::from)
        },
        source,
        destination,
    )
}

fn publish_export_after(
    before: impl FnOnce() -> Result<()>,
    source: &Path,
    destination: &Path,
) -> Result<()> {
    before()?;
    atomic_rename_fresh(source, destination)
}

fn select_restore_chain(
    restore_chain: &[RestoreRevision],
    current_head: Option<Uuid>,
    requested: Option<Uuid>,
) -> Result<(Vec<RestoreRevision>, Option<Uuid>)> {
    let Some(current_head) = current_head else {
        ensure!(
            restore_chain.is_empty(),
            "empty PipeFS head has a restore chain"
        );
        ensure!(
            requested.is_none(),
            "cannot export an exact revision from an empty PipeFS head"
        );
        return Ok((Vec::new(), None));
    };
    let head_index = restore_chain
        .iter()
        .position(|revision| revision.revision_id == current_head)
        .ok_or_else(|| anyhow!("current PipeFS head {current_head} is not in the restore chain"))?;
    ensure!(
        head_index + 1 == restore_chain.len(),
        "remote restore chain continues past its declared head"
    );

    let target = requested.unwrap_or(current_head);
    let index = restore_chain
        .iter()
        .position(|revision| revision.revision_id == target)
        .ok_or_else(|| {
            anyhow!("requested revision {target} is not in the verified restore chain")
        })?;
    Ok((restore_chain[..=index].to_vec(), Some(target)))
}

fn validate_chain_revision(
    revision: &RestoreRevision,
    prior_revision: Option<Uuid>,
    prior_sequence: Option<u64>,
    first: bool,
) -> Result<()> {
    ensure!(
        revision.base_revision_id == prior_revision,
        "restore chain base mismatch at {}",
        revision.revision_id
    );
    ensure!(
        prior_sequence.is_none_or(|value| revision.sequence == value + 1),
        "restore chain sequence is not contiguous"
    );
    ensure!(
        (first && revision.revision_type == RevisionKind::Full)
            || (!first && revision.revision_type == RevisionKind::Delta),
        "restore chain must start with a full revision followed by deltas"
    );
    Ok(())
}

fn fresh_destination(destination: &Path) -> Result<PathBuf> {
    let absolute = if destination.is_absolute() {
        destination.to_path_buf()
    } else {
        std::env::current_dir()?.join(destination)
    };
    let name = absolute
        .file_name()
        .ok_or_else(|| anyhow!("PipeFS export destination has no file name"))?;
    let parent = absolute
        .parent()
        .ok_or_else(|| anyhow!("PipeFS export destination has no parent"))?
        .canonicalize()
        .context("canonicalizing PipeFS export parent")?;
    ensure!(parent.is_dir(), "PipeFS export parent is not a directory");
    let destination = parent.join(name);
    ensure!(
        fs::symlink_metadata(&destination)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound),
        "PipeFS export destination already exists or cannot be inspected: {}",
        destination.display()
    );
    Ok(destination)
}

struct TemporaryExport {
    path: PathBuf,
}

impl TemporaryExport {
    fn new(parent: &Path) -> Result<Self> {
        for _ in 0..32 {
            let path = parent.join(format!(".hi-pipefs-export-{}", Uuid::new_v4().simple()));
            match fs::create_dir(&path) {
                Ok(()) => {
                    set_private_directory_permissions(&path)?;
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(anyhow!(
            "could not allocate a private PipeFS export directory"
        ))
    }
}

impl Drop for TemporaryExport {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn create_private_directory(path: &Path) -> Result<()> {
    fs::create_dir(path)?;
    set_private_directory_permissions(path)
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
fn atomic_rename_fresh(source: &Path, destination: &Path) -> Result<()> {
    let source_parent = rustix::fs::open(
        source.parent().expect("materialized export has a parent"),
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )?;
    let destination_parent = rustix::fs::open(
        destination
            .parent()
            .expect("export destination has a parent"),
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )?;
    rustix::fs::renameat_with(
        &source_parent,
        source.file_name().expect("materialized export has a name"),
        &destination_parent,
        destination
            .file_name()
            .expect("export destination has a name"),
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(std::io::Error::from)
    .with_context(|| format!("publishing fresh PipeFS export {}", destination.display()))
}

#[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
fn atomic_rename_fresh(source: &Path, destination: &Path) -> Result<()> {
    ensure!(
        !destination.exists(),
        "PipeFS export destination already exists"
    );
    fs::rename(source, destination)?;
    Ok(())
}

fn sync_directory(path: &Path) {
    if let Ok(directory) = fs::File::open(path) {
        let _ = directory.sync_all();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn revision(
        id: Uuid,
        base: Option<Uuid>,
        sequence: u64,
        kind: RevisionKind,
    ) -> RestoreRevision {
        RestoreRevision {
            revision_id: id,
            base_revision_id: base,
            revision_type: kind,
            sequence,
            artifact: crate::ArtifactDescriptor {
                blake3: "a".repeat(64),
                size_bytes: 1,
                media_type: String::new(),
            },
            manifest_digest: "b".repeat(64),
            logical_size_bytes: 1,
        }
    }

    #[test]
    fn selects_an_exact_revision_without_later_deltas() {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let chain = vec![
            revision(first, None, 1, RevisionKind::Full),
            revision(second, Some(first), 2, RevisionKind::Delta),
        ];
        let (selected, target) = select_restore_chain(&chain, Some(second), Some(first)).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(target, Some(first));
    }

    #[test]
    fn exact_export_rejects_revisions_past_the_declared_head() {
        let head = Uuid::new_v4();
        let unacknowledged = Uuid::new_v4();
        let chain = vec![
            revision(head, None, 1, RevisionKind::Full),
            revision(unacknowledged, Some(head), 2, RevisionKind::Delta),
        ];
        let error = select_restore_chain(&chain, Some(head), Some(unacknowledged)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("continues past its declared head")
        );
    }

    #[test]
    fn exact_export_rejects_a_revision_when_the_remote_head_is_empty() {
        let revision_id = Uuid::new_v4();
        let error = select_restore_chain(&[], None, Some(revision_id)).unwrap_err();
        assert!(error.to_string().contains("empty PipeFS head"));
    }

    #[test]
    fn fresh_publish_never_replaces_an_existing_destination() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        let destination = temporary.path().join("destination");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        assert!(atomic_rename_fresh(&source, &destination).is_err());
        assert!(source.is_dir());
        assert!(destination.is_dir());
    }

    #[test]
    fn failure_before_export_rename_does_not_publish_destination() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        let destination = temporary.path().join("destination");
        fs::create_dir(&source).unwrap();
        let result = publish_export_after(
            || Err(anyhow!("injected export boundary failure")),
            &source,
            &destination,
        );
        assert!(result.is_err());
        assert!(source.is_dir());
        assert!(!destination.exists());
    }

    #[cfg(unix)]
    #[test]
    fn fresh_destination_rejects_a_dangling_symlink() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let destination = temporary.path().join("destination");
        symlink(temporary.path().join("missing"), &destination).unwrap();
        let error = fresh_destination(&destination).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("already exists or cannot be inspected")
        );
    }
}
