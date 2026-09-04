//! Cross-thread and cross-process exclusion for one internal snapshot store.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::Result;

#[cfg(unix)]
use anyhow::Context;
#[cfg(unix)]
use std::fs::{File, OpenOptions};

static PROCESS_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Weak<Mutex<()>>>>> = OnceLock::new();

pub(super) struct StoreLock {
    process: Arc<Mutex<()>>,
    #[cfg(unix)]
    file: File,
}

#[cfg(unix)]
impl Drop for StoreLockGuard<'_> {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd as _;

        // Releasing explicitly keeps the file-lock lifetime identical to the
        // process mutex guard even if StoreLock is retained and reacquired.
        // SAFETY: `_file` remains open for this guard's complete lifetime.
        let _ = unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
    }
}

pub(super) struct StoreLockGuard<'a> {
    _process: std::sync::MutexGuard<'a, ()>,
    #[cfg(unix)]
    _file: &'a File,
}

impl StoreLock {
    pub(super) fn open(workspace_dir: &Path) -> Result<Self> {
        let key = workspace_dir
            .canonicalize()
            .unwrap_or_else(|_| workspace_dir.to_path_buf());
        let registry = PROCESS_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
        let process = {
            let mut registry = registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            registry.retain(|_, lock| lock.strong_count() > 0);
            if let Some(lock) = registry.get(&key).and_then(Weak::upgrade) {
                lock
            } else {
                let lock = Arc::new(Mutex::new(()));
                registry.insert(key, Arc::downgrade(&lock));
                lock
            }
        };
        #[cfg(unix)]
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(workspace_dir.join("store.lock"))
            .with_context(|| {
                format!("opening snapshot store lock in {}", workspace_dir.display())
            })?;
        Ok(Self {
            process,
            #[cfg(unix)]
            file,
        })
    }

    pub(super) fn acquire(&self) -> Result<StoreLockGuard<'_>> {
        let process = self
            .process
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        #[cfg(unix)]
        lock_file(&self.file)?;
        Ok(StoreLockGuard {
            _process: process,
            #[cfg(unix)]
            _file: &self.file,
        })
    }
}

#[cfg(unix)]
fn lock_file(file: &File) -> Result<()> {
    use std::os::fd::AsRawFd as _;

    loop {
        // SAFETY: the descriptor belongs to `file` and remains open for the
        // complete lifetime of the returned store guard.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error).context("locking internal snapshot store");
        }
    }
}
