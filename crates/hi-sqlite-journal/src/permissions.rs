//! Adjust permissions without closing a data descriptor in the SQLite process.
//!
//! POSIX locks belong to a process/inode, so closing an unrelated descriptor
//! can discard SQLite's live DB and WAL-index locks. Linux O_PATH descriptors
//! avoid that close behavior. Other Unix platforms use a small syscall-only
//! child; its descriptor closes cannot release the parent's locks.

mod process;

use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

use anyhow::{Context, Result, ensure};

pub(super) fn prepare_writable_database(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Creating in a separate process also avoids a first-open race:
            // another caller may acquire SQLite locks before the creator closes.
            process::create_owner_only(path)?;
        }
        Err(error) => return Err(error).context("reading sqlite database metadata"),
    }
    set_owner_only(path, false)?;
    tighten_existing_sidecars(path)
}

pub(super) fn tighten_existing_sidecars(path: &Path) -> Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        set_owner_only(&super::sidecar_path(path, suffix), true)?;
    }
    Ok(())
}

fn validate_file(metadata: &Metadata, path: &Path) -> Result<()> {
    ensure!(
        !metadata.file_type().is_symlink(),
        "refusing symbolic link for sqlite file: {}",
        path.display()
    );
    ensure!(
        metadata.is_file(),
        "sqlite path is not a regular file: {}",
        path.display()
    );
    ensure!(
        metadata.nlink() == 1,
        "refusing multiply-linked sqlite file: {}",
        path.display()
    );
    Ok(())
}

pub(super) fn set_owner_only(path: &Path, allow_missing: bool) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if allow_missing && error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(());
        }
        Err(error) => {
            return Err(error).with_context(|| format!("reading mode for {}", path.display()));
        }
    };
    validate_file(&metadata, path)?;
    if metadata.mode() & 0o7777 == 0o600 {
        return Ok(());
    }
    finish_permission_update(path, allow_missing, tighten_verified_file(path, &metadata))
}

fn finish_permission_update(path: &Path, allow_missing: bool, result: Result<()>) -> Result<()> {
    match result {
        Err(error)
            if allow_missing
                && error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound)
                && matches!(std::fs::symlink_metadata(path), Err(error) if error.kind() == std::io::ErrorKind::NotFound) =>
        {
            Ok(())
        }
        result => result,
    }
}

#[cfg(target_os = "linux")]
fn tighten_verified_file(path: &Path, expected: &Metadata) -> Result<()> {
    use std::os::fd::AsRawFd;

    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| {
            format!(
                "opening sqlite metadata without following links {}",
                path.display()
            )
        })?;
    let opened = file.metadata().context("reading opened sqlite metadata")?;
    validate_file(&opened, path)?;
    ensure!(
        expected.dev() == opened.dev() && expected.ino() == opened.ino(),
        "sqlite path changed while securing it: {}",
        path.display()
    );
    // O_PATH cannot use fchmod directly. This procfs reference names the held
    // inode even if its directory entry changes, without reopening a data fd.
    let descriptor_path = format!("/proc/self/fd/{}", file.as_raw_fd());
    std::fs::set_permissions(&descriptor_path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("setting owner-only mode on {}", path.display()))
}

#[cfg(not(target_os = "linux"))]
fn tighten_verified_file(path: &Path, expected: &Metadata) -> Result<()> {
    process::tighten_owner_only(path, expected)
}

#[cfg(test)]
pub(super) fn tighten_in_child_for_test(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    validate_file(&metadata, path)?;
    process::tighten_owner_only(path, &metadata)
}

#[cfg(test)]
pub(super) fn tighten_in_child_with_metadata_for_test(
    path: &Path,
    metadata: &Metadata,
) -> Result<()> {
    process::tighten_owner_only(path, metadata)
}

#[cfg(test)]
pub(super) fn tighten_with_metadata_for_test(path: &Path, metadata: &Metadata) -> Result<()> {
    tighten_verified_file(path, metadata)
}

#[cfg(test)]
mod tests {
    #[test]
    fn absent_procfs_reference_cannot_hide_an_unsecured_present_sidecar() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let sidecar = root.path().join("state-shm");
        std::fs::write(&sidecar, "existing").unwrap();
        std::fs::set_permissions(&sidecar, std::fs::Permissions::from_mode(0o666)).unwrap();
        let missing_descriptor = || {
            Err(
                anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound))
                    .context("chmod /proc/self/fd failed"),
            )
        };
        assert!(super::finish_permission_update(&sidecar, true, missing_descriptor()).is_err());
        std::fs::remove_file(&sidecar).unwrap();
        assert!(super::finish_permission_update(&sidecar, true, missing_descriptor()).is_ok());
        assert!(super::finish_permission_update(&sidecar, false, missing_descriptor()).is_err());
    }
}
