//! Only fixed libc syscalls execute after fork; no allocation, SQLite, or stdio.

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
#[cfg(any(not(target_os = "linux"), test))]
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

#[derive(Clone, Copy)]
enum Operation {
    Create,
    #[cfg(any(not(target_os = "linux"), test))]
    Tighten {
        device: u64,
        inode: u64,
    },
}

pub(super) fn create_owner_only(path: &Path) -> Result<()> {
    run(path, Operation::Create)
}

#[cfg(any(not(target_os = "linux"), test))]
pub(super) fn tighten_owner_only(path: &Path, expected: &std::fs::Metadata) -> Result<()> {
    run(
        path,
        Operation::Tighten {
            device: expected.dev(),
            inode: expected.ino(),
        },
    )
}

fn run(path: &Path, operation: Operation) -> Result<()> {
    let encoded =
        CString::new(path.as_os_str().as_bytes()).context("sqlite path contains a NUL byte")?;
    // The child never touches inherited SQLite connections or Rust locks. Its
    // data descriptors and their POSIX locks belong to a different process.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error()).context("starting sqlite permission helper");
    }
    if pid == 0 {
        let status = unsafe { child_operation(encoded.as_ptr(), operation) };
        unsafe { libc::_exit(status) };
    }
    let mut status = 0;
    loop {
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        if waited == pid {
            break;
        }
        let error = std::io::Error::last_os_error();
        if waited < 0 && error.kind() == std::io::ErrorKind::Interrupted {
            continue;
        }
        return Err(error).context("reaping sqlite permission helper");
    }
    ensure!(
        libc::WIFEXITED(status),
        "sqlite permission helper terminated unexpectedly for {}",
        path.display()
    );
    match libc::WEXITSTATUS(status) {
        0 => Ok(()),
        // Another opener may have won create_new. The caller validates the
        // winning entry (including links and type) before SQLite sees it.
        1 if matches!(operation, Operation::Create) && std::fs::symlink_metadata(path).is_ok() => {
            Ok(())
        }
        1 => match std::fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(error).with_context(|| {
                    format!("sqlite file disappeared while securing {}", path.display())
                })
            }
            _ => bail!("sqlite permission helper could not open {}", path.display()),
        },
        code => bail!(
            "sqlite permission helper failed for {} (operation status {code})",
            path.display()
        ),
    }
}

// libc::stat field widths differ between Linux and macOS; normalize them to
// MetadataExt's u64 identity without changing signed-device bit patterns.
#[allow(clippy::unnecessary_cast)]
unsafe fn child_operation(path: *const libc::c_char, operation: Operation) -> libc::c_int {
    let flags = libc::O_WRONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC;
    let flags = match operation {
        Operation::Create => flags | libc::O_CREAT | libc::O_EXCL,
        #[cfg(any(not(target_os = "linux"), test))]
        Operation::Tighten { .. } => flags,
    };
    let fd = unsafe { libc::open(path, flags, 0o600 as libc::c_uint) };
    if fd < 0 {
        return 1;
    }
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::uninit();
    let result = if unsafe { libc::fstat(fd, metadata.as_mut_ptr()) } != 0 {
        2
    } else {
        let metadata = unsafe { metadata.assume_init() };
        let regular = metadata.st_mode & libc::S_IFMT == libc::S_IFREG;
        let same_inode = match operation {
            Operation::Create => true,
            #[cfg(any(not(target_os = "linux"), test))]
            Operation::Tighten { device, inode } => {
                metadata.st_dev as u64 == device && metadata.st_ino as u64 == inode
            }
        };
        if !regular || metadata.st_nlink != 1 || !same_inode {
            3
        } else if unsafe { libc::fchmod(fd, 0o600) } != 0 {
            4
        } else {
            0
        }
    };
    unsafe { libc::close(fd) };
    result
}
