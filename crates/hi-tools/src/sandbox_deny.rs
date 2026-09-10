//! Mode-000 deny placeholders vs sandbox spoof.
//!
//! A chmod-000 inode yields `PermissionDenied` on read. That is a real OS deny,
//! not a bwrap/marker spoof (a process claiming to be inside the sandbox while
//! denied paths stay readable). Walkers and post-apply checks must treat
//! mode-000 `PermissionDenied` as expected denial.

use std::io;
use std::path::Path;
#[cfg(any(test, target_os = "linux"))]
use std::path::PathBuf;

/// True when `err` is `PermissionDenied` and `path` is mode 000 (no rwx).
pub fn permission_denied_is_mode_000(err: &io::Error, path: &Path) -> bool {
    if err.kind() != io::ErrorKind::PermissionDenied {
        return false;
    }
    path_is_mode_000(path)
}

pub fn path_is_mode_000(path: &Path) -> bool {
    mode_of(path).is_some_and(|mode| mode & 0o777 == 0)
}

#[cfg(unix)]
fn mode_of(path: &Path) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .ok()
        .map(|meta| meta.permissions().mode())
}

#[cfg(not(unix))]
fn mode_of(_path: &Path) -> Option<u32> {
    None
}

#[cfg(unix)]
pub fn chmod_000(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o000);
    std::fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
pub fn chmod_000(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// Overlay source bound over a deny-read target. Directories use a private
/// empty mask (chmod 000 when possible); files bind `/dev/null`.
#[cfg(any(test, target_os = "linux"))]
pub fn deny_read_overlay_source(private_temp: Option<&Path>, target: &Path) -> PathBuf {
    if target.is_dir()
        && let Some(mask) = private_temp
            .map(|temp| temp.join(super::PRIVATE_DENY_READ_MASK_DIR))
            .filter(|mask| mask.is_dir())
    {
        let _ = chmod_000(&mask);
        return mask;
    }
    PathBuf::from("/dev/null")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn mode_000_permission_denied_is_not_a_spoof() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("secret");
        fs::write(&path, "x").unwrap();
        chmod_000(&path).unwrap();
        let err = fs::read(&path).expect_err("mode-000 file must not be readable");
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
        assert!(
            permission_denied_is_mode_000(&err, &path),
            "chmod-000 PermissionDenied is a real deny, not a spoof"
        );
        let other = io::Error::new(io::ErrorKind::PermissionDenied, "sandbox");
        let readable = dir.path().join("open");
        fs::write(&readable, "ok").unwrap();
        assert!(
            !permission_denied_is_mode_000(&other, &readable),
            "PermissionDenied on a writable inode is not mode-000"
        );
    }
}
