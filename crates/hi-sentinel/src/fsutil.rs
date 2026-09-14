//! Owner-only dirs/files. Default umask would leave incident bundles readable.

use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use hi_liveness::write_private_file;

static WRITE_NONCE: AtomicU64 = AtomicU64::new(0);

pub fn mkdir_0700(path: &Path) -> io::Result<()> {
    fs::create_dir_all(path)?;
    chmod_0700(path)
}

pub fn chmod_0700(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

pub fn chmod_0600(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

pub fn chmod_0700_file(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

pub fn write_0600(path: &Path, body: &[u8]) -> io::Result<()> {
    write_private_file(path, body)?;
    chmod_0600(path)
}

/// Exclusive `O_NOFOLLOW` temp in the same directory, then rename. A planted
/// `config.toml.tmp` symlink must not receive the payload.
pub fn write_atomic_0600(path: &Path, body: &[u8]) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    for _ in 0..32 {
        let nonce = WRITE_NONCE.fetch_add(1, Ordering::Relaxed);
        let tmp = parent.join(format!(".{name}.{}.{nonce}.tmp", std::process::id()));
        match write_private_nofollow(&tmp, body) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
        match fs::rename(&tmp, path) {
            Ok(()) => {
                chmod_0600(path)?;
                return Ok(());
            }
            Err(error) => {
                if path.exists() {
                    let _ = fs::remove_file(&tmp);
                }
                return Err(error);
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a private temporary file",
    ))
}

#[cfg(unix)]
fn write_private_nofollow(path: &Path, body: &[u8]) -> io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    file.write_all(body)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private_nofollow(path: &Path, body: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(body)?;
    file.sync_all()
}

#[cfg(test)]
pub fn unix_mode(path: &Path) -> io::Result<u32> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        Ok(fs::metadata(path)?.permissions().mode() & 0o777)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(0)
    }
}
