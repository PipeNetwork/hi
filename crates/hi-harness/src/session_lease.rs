//! Exclusive flock on `<id>.jsonl.lock` plus a pid/hostname record.
//!
//! Reboot releases the flock. A leftover pid file is overwritten once the
//! lock is acquired. SIGSTOP keeps the lock, so a second writer sees a live
//! or stopped holder instead of stealing the JSONL.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rustix::fs::{FlockOperation, flock};
use serde::{Deserialize, Serialize};

/// Another process holds this session's JSONL lock.
#[derive(Debug)]
pub struct SessionBusy {
    pub pid: u32,
    pub stopped: bool,
    pub hostname: String,
}

impl std::fmt::Display for SessionBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.pid == 0 {
            write!(f, "session is already open in another hi process")
        } else if self.stopped {
            write!(f, "session held by pid {} (stopped); kill or fg", self.pid)
        } else {
            write!(
                f,
                "session held by pid {} (live) on {}; kill or wait",
                self.pid, self.hostname
            )
        }
    }
}

impl std::error::Error for SessionBusy {}

/// Lock state observed without taking ownership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionLockStatus {
    Free,
    Held {
        pid: u32,
        stopped: bool,
        hostname: String,
    },
}

impl SessionLockStatus {
    pub fn is_held(&self) -> bool {
        matches!(self, Self::Held { .. })
    }

    pub fn flag_text(&self) -> Option<String> {
        match self {
            Self::Free => None,
            Self::Held {
                pid, stopped: true, ..
            } => Some(format!("held by pid {pid} (stopped)")),
            Self::Held { pid, .. } => Some(format!("held by pid {pid} (live)")),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct HolderRecord {
    pid: u32,
    #[serde(default)]
    hostname: String,
    #[serde(default)]
    started_unix_ms: u64,
}

/// Process-lifetime exclusive lock. Unlock happens on drop (and on reboot).
#[derive(Debug)]
pub struct SessionLease {
    _file: File,
    path: PathBuf,
}

impl SessionLease {
    pub fn acquire(jsonl: &Path) -> Result<Self> {
        let path = lock_path(jsonl);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating session lock dir {}", parent.display()))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("opening session lock {}", path.display()))?;
        match flock(&file, FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => {
                write_holder(&mut file)?;
                Ok(Self { _file: file, path })
            }
            Err(_) => {
                let holder = read_holder(&path).unwrap_or(HolderRecord {
                    pid: 0,
                    hostname: String::new(),
                    started_unix_ms: 0,
                });
                Err(SessionBusy {
                    pid: holder.pid,
                    stopped: process_stopped(holder.pid),
                    hostname: holder.hostname,
                }
                .into())
            }
        }
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        let _ = flock(&self._file, FlockOperation::Unlock);
        let _ = fs::remove_file(&self.path);
    }
}

/// Lock sidecar next to the JSONL (`session.jsonl.lock`).
pub fn lock_path(jsonl: &Path) -> PathBuf {
    let mut path = jsonl.as_os_str().to_os_string();
    path.push(".lock");
    PathBuf::from(path)
}

/// Probe whether another process currently owns the JSONL lock.
pub fn inspect_session_lock(jsonl: &Path) -> SessionLockStatus {
    let path = lock_path(jsonl);
    if !path.exists() {
        return SessionLockStatus::Free;
    }
    let Ok(file) = OpenOptions::new().read(true).write(true).open(&path) else {
        return SessionLockStatus::Free;
    };
    match flock(&file, FlockOperation::NonBlockingLockExclusive) {
        Ok(()) => {
            let _ = flock(&file, FlockOperation::Unlock);
            SessionLockStatus::Free
        }
        Err(_) => {
            let holder = read_holder(&path).unwrap_or(HolderRecord {
                pid: 0,
                hostname: String::new(),
                started_unix_ms: 0,
            });
            SessionLockStatus::Held {
                pid: holder.pid,
                stopped: process_stopped(holder.pid),
                hostname: holder.hostname,
            }
        }
    }
}

fn write_holder(file: &mut File) -> Result<()> {
    let record = HolderRecord {
        pid: std::process::id(),
        hostname: hostname(),
        started_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
    };
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    serde_json::to_writer(&mut *file, &record)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn read_holder(path: &Path) -> Option<HolderRecord> {
    let mut file = File::open(path).ok()?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).ok()?;
    serde_json::from_str(buf.trim()).ok()
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "localhost".into())
}

fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let Some(pid) = rustix::process::Pid::from_raw(pid as i32) else {
        return false;
    };
    rustix::process::test_kill_process(pid).is_ok()
}

fn process_stopped(pid: u32) -> bool {
    if !process_alive(pid) {
        return false;
    }
    let output = std::process::Command::new("ps")
        .args(["-o", "state=", "-p", &pid.to_string()])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let state = String::from_utf8_lossy(&out.stdout);
            let state = state.trim();
            state.starts_with('T') || state.starts_with('t')
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    #[test]
    fn drop_releases_so_a_second_open_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("s.jsonl");
        let first = SessionLease::acquire(&jsonl).unwrap();
        drop(first);
        SessionLease::acquire(&jsonl).expect("lock free after drop");
    }

    #[test]
    fn forged_pid_without_flock_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("s.jsonl");
        let lock = lock_path(&jsonl);
        fs::write(&lock, r#"{"pid":1,"hostname":"ghost","started_unix_ms":1}"#).unwrap();
        SessionLease::acquire(&jsonl).expect("stale pid file must not block");
    }

    #[test]
    fn second_open_fails_while_flock_is_held() {
        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("s.jsonl");
        let lock = lock_path(&jsonl);
        let mut child = match spawn_lock_holder(&lock) {
            Some(child) => child,
            None => return,
        };
        let err = SessionLease::acquire(&jsonl).unwrap_err();
        let busy = err.downcast_ref::<SessionBusy>().expect("SessionBusy");
        assert!(busy.pid > 0, "holder pid recorded");
        let _ = child.kill();
        let _ = child.wait();
    }

    fn spawn_lock_holder(lock: &Path) -> Option<std::process::Child> {
        let script = format!(
            r#"
import fcntl, time, os, json, sys
path = {path:?}
f = open(path, "a+")
fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
f.seek(0)
f.truncate()
f.write(json.dumps({{"pid": os.getpid(), "hostname": "test", "started_unix_ms": 1}}))
f.flush()
print("held", flush=True)
time.sleep(30)
"#,
            path = lock
        );
        let mut child = Command::new("python3")
            .arg("-c")
            .arg(script)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdout = child.stdout.take()?;
        let mut lines = BufReader::new(stdout).lines();
        let ready = lines.next().and_then(|line| line.ok());
        if ready.as_deref() != Some("held") {
            let _ = child.kill();
            return None;
        }
        Some(child)
    }
}
