//! Append-only JSONL event log, 0600, rotated at 2 MiB.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::atomic::{ensure_private_dir, set_file_0600};
use crate::schema::{EVENT_LOG_CAP, LiveEvent};

#[derive(Clone, Debug)]
pub struct EventLog {
    inner: Arc<EventLogInner>,
}

#[derive(Debug)]
struct EventLogInner {
    path: PathBuf,
    cap: u64,
}

impl EventLog {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_cap(path, EVENT_LOG_CAP)
    }

    pub fn with_cap(path: impl Into<PathBuf>, cap: u64) -> Self {
        Self {
            inner: Arc::new(EventLogInner {
                path: path.into(),
                cap: cap.max(1),
            }),
        }
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    pub fn append(&self, event: &LiveEvent) {
        let mut event = event.clone();
        if let Some(detail) = event.detail.take() {
            let redacted = hi_secrets::redact_secrets(&detail).into_owned();
            event.detail = if redacted.is_empty() {
                None
            } else {
                Some(redacted)
            };
        }
        let _ = self.append_inner(&event);
    }

    fn append_inner(&self, event: &LiveEvent) -> std::io::Result<()> {
        if let Some(parent) = self.inner.path.parent() {
            ensure_private_dir(parent)?;
        }
        rotate_if_needed(&self.inner.path, self.inner.cap);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.inner.path)?;
        set_file_0600(&self.inner.path);
        serde_json::to_writer(&mut file, event).map_err(json_err)?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }
}

fn rotate_if_needed(path: &Path, cap: u64) {
    let Ok(meta) = fs::metadata(path) else {
        return;
    };
    if meta.len() < cap {
        return;
    }
    let rotated = {
        let mut p = path.as_os_str().to_os_string();
        p.push(".1");
        PathBuf::from(p)
    };
    let _ = fs::rename(path, rotated);
}

fn json_err(err: serde_json::Error) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, err)
}
