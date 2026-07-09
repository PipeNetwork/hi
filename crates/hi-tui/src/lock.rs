//! A per-project single-firer lock, so the `/loop` manager runs in exactly one
//! place at a time — either the TUI or the background daemon, never both (which
//! would double-fire every loop). The lock is a PID file next to `loops.json`;
//! it is acquired atomically (`create_new`), released on drop, and reclaimed if
//! the recorded process is gone (a crash leaves a stale file).

use std::io::Write;
use std::path::{Path, PathBuf};

/// Held for as long as the owner is firing loops; removes the PID file on drop.
pub(crate) struct FireLock {
    path: PathBuf,
}

impl Drop for FireLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// The lock path for a project (sibling of its `loops.json`).
pub(crate) fn lock_path(loops_file: &Path) -> PathBuf {
    loops_file.with_file_name("loops.lock")
}

/// The PID currently recorded in the lock file, if any.
pub(crate) fn holder(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| t.trim().parse().ok())
}

/// Whether a process with this pid is alive (`kill -0`, works on macOS + Linux).
fn is_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// The live holder of the lock, if one exists (a stale PID file reads as free).
pub(crate) fn live_holder(path: &Path) -> Option<u32> {
    holder(path).filter(|pid| is_alive(*pid))
}

/// Try to acquire the fire-lock. Returns `None` if a *live* process already
/// holds it; reclaims a stale file (dead holder) and retries.
pub(crate) fn try_acquire(path: &Path) -> Option<FireLock> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
        {
            Ok(mut f) => {
                let _ = write!(f, "{}", std::process::id());
                return Some(FireLock { path: path.into() });
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                match holder(path) {
                    Some(pid) if is_alive(pid) => return None, // a live owner holds it
                    _ => {
                        // Stale (dead holder or unreadable): reclaim and retry.
                        let _ = std::fs::remove_file(path);
                        continue;
                    }
                }
            }
            Err(_) => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_is_exclusive_then_reclaims_stale() {
        let dir = std::env::temp_dir().join(format!("hi-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let loops = dir.join("loops.json");
        let path = lock_path(&loops);

        let a = try_acquire(&path).expect("first acquire succeeds");
        assert_eq!(live_holder(&path), Some(std::process::id()));
        // A second acquire fails while the first is held (live self).
        assert!(try_acquire(&path).is_none(), "second acquire is refused");

        // A stale file (dead pid) is reclaimable.
        drop(a);
        std::fs::write(&path, "999999999").unwrap();
        assert!(live_holder(&path).is_none(), "dead pid reads as free");
        let _b = try_acquire(&path).expect("stale lock reclaimed");
        assert_eq!(live_holder(&path), Some(std::process::id()));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
