//! A per-project single-firer lock, so the `/loop` manager runs in exactly one
//! place at a time — either the TUI or the background daemon, never both (which
//! would double-fire every loop). The lock is a PID file next to `loops.json`.
//!
//! It is acquired atomically by *hard-linking* a fully-written temp file into
//! place: `hard_link` fails if the target exists (so acquisition is exclusive)
//! and, unlike `create_new` followed by a separate write, the lock file is never
//! observed empty — closing the window where a racing acquirer would read an
//! empty holder, call it stale, and steal a lock that is about to become live.
//! The recorded identity is `pid:start-time`, so a recycled pid (after a reboot
//! or pid wraparound) isn't mistaken for the original holder. Released on drop
//! (only if the file is still ours), and reclaimed if the recorded process is
//! gone.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Held for as long as the owner is firing loops; removes the PID file on drop.
pub(crate) struct FireLock {
    path: PathBuf,
    /// The `pid:start` identity we wrote — so drop only removes a lock still ours.
    content: String,
}

impl Drop for FireLock {
    fn drop(&mut self) {
        // Only remove the lock if it's still ours: never delete a lock another
        // process legitimately reclaimed after we were (wrongly or rightly)
        // considered stale.
        if let Ok(cur) = std::fs::read_to_string(&self.path)
            && cur.trim() == self.content
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// The lock path for a project (sibling of its `loops.json`).
pub(crate) fn lock_path(loops_file: &Path) -> PathBuf {
    loops_file.with_file_name("loops.lock")
}

/// A stable-ish process identity: its start time, so a recycled pid isn't taken
/// for the original. Via `ps` (macOS + Linux, like the `kill -0` liveness probe);
/// empty string when unavailable, in which case we fall back to pid-only liveness.
fn proc_start(pid: u32) -> String {
    std::process::Command::new("ps")
        .args(["-o", "lstart=", "-p", &pid.to_string()])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The `(pid, start)` recorded in the lock file, if parseable. `start` may be
/// empty (older format / unavailable). Splits on the first `:` — the pid is
/// digits, and the `lstart` tail can itself contain `:` (e.g. `12:00:00`).
fn holder(path: &Path) -> Option<(u32, String)> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    let (pid_s, start) = text.split_once(':').unwrap_or((text, ""));
    let pid: u32 = pid_s.trim().parse().ok()?;
    Some((pid, start.to_string()))
}

/// Whether the recorded holder is a live process *and* the same one that took the
/// lock. `pid == 0` is never alive (`kill -0 0` signals the whole process group
/// and would falsely read as live). When start times are comparable, a mismatch
/// means the pid was recycled → not our holder → reclaimable.
fn is_alive(pid: u32, recorded_start: &str) -> bool {
    if pid == 0 {
        return false;
    }
    let signalable = std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !signalable {
        return false;
    }
    // Same process? If either start is unknown, fall back to pid-only liveness.
    if recorded_start.is_empty() {
        return true;
    }
    let current = proc_start(pid);
    current.is_empty() || current == recorded_start
}

/// The live holder of the lock, if one exists (a stale PID file reads as free).
pub(crate) fn live_holder(path: &Path) -> Option<u32> {
    let (pid, start) = holder(path)?;
    is_alive(pid, &start).then_some(pid)
}

/// Try to acquire the fire-lock. Returns `None` if a *live* process already holds
/// it; reclaims a stale file (dead/recycled/abandoned holder) and retries.
pub(crate) fn try_acquire(path: &Path) -> Option<FireLock> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let content = format!("{}:{}", std::process::id(), proc_start(std::process::id()));
    let tmp = path.with_extension(format!("lock.{}.tmp", std::process::id()));
    // Bounded so a pathological state (e.g. an un-removable stale file) can't spin.
    for _ in 0..100 {
        let _ = std::fs::remove_file(&tmp);
        if std::fs::write(&tmp, &content).is_err() {
            return None;
        }
        match std::fs::hard_link(&tmp, path) {
            Ok(()) => {
                let _ = std::fs::remove_file(&tmp);
                return Some(FireLock {
                    path: path.into(),
                    content,
                });
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                let _ = std::fs::remove_file(&tmp);
                if live_holder(path).is_some() {
                    return None; // a live owner holds it
                }
                // Stale: reclaim it. A blind `remove_file(path)` here is racy —
                // between the `live_holder` check above (which shells out to
                // `kill`/`ps`, a wide window) and the remove, a peer can reclaim
                // and install its own *live* lock, which we'd then delete, leaving
                // two holders that both double-fire every loop. Instead move the
                // file aside atomically: `rename` of the current file succeeds for
                // exactly one racer (the other gets ENOENT and re-observes the
                // winner's lock on the next iteration). Then re-check the captured
                // file's liveness — if it went live in the meantime, restore it
                // and back off rather than steal a live lock.
                let aside = path.with_extension(format!("lock.reclaim.{}", std::process::id()));
                let _ = std::fs::remove_file(&aside);
                if std::fs::rename(path, &aside).is_ok() {
                    if live_holder(&aside).is_some() {
                        // Became live between the check and the capture — put it
                        // back and let that owner keep the lock.
                        let _ = std::fs::rename(&aside, path);
                        return None;
                    }
                    let _ = std::fs::remove_file(&aside);
                }
                // Whether we captured it or lost the rename race, loop and retry
                // the exclusive hard-link.
            }
            Err(_) => {
                let _ = std::fs::remove_file(&tmp);
                return None;
            }
        }
    }
    let _ = std::fs::remove_file(&tmp);
    None
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

    #[test]
    fn lock_has_no_empty_window_and_guards_pid_zero() {
        let dir = std::env::temp_dir().join(format!("hi-lock2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = lock_path(&dir.join("loops.json"));

        // The lock file always has content (atomic hard-link) — never the empty
        // window a racer could mistake for a stale lock — and a live holder blocks.
        let held = try_acquire(&path).expect("acquire");
        assert!(
            !std::fs::read_to_string(&path).unwrap().trim().is_empty(),
            "lock has content, no empty window"
        );
        assert!(try_acquire(&path).is_none(), "live lock not stolen");
        drop(held);

        // pid 0 is never a live holder (kill -0 0 signals the process group).
        std::fs::write(&path, "0:").unwrap();
        assert!(live_holder(&path).is_none(), "pid 0 is not a live holder");

        // When start times are available, a recycled pid (same number, different
        // start) reads as free rather than a false live lockout.
        if !proc_start(std::process::id()).is_empty() {
            let mypid = std::process::id();
            std::fs::write(&path, format!("{mypid}:definitely-not-the-start")).unwrap();
            assert!(
                live_holder(&path).is_none(),
                "recycled pid (start mismatch) reads free"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
