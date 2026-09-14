//! Owner-only dirs/files. Default umask would leave incident bundles readable.

use std::fs;
use std::io;
use std::path::Path;

use hi_liveness::write_private_file;

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
