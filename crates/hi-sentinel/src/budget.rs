//! Incident retention and hourly apply cap.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::fsutil;

const HOUR_MS: u64 = 60 * 60 * 1000;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApplyLogLine {
    pub ts_unix_ms: u64,
    pub id: String,
}

pub fn gc_incidents(incidents: &Path, ttl: Duration, max_count: usize) {
    let Ok(entries) = fs::read_dir(incidents) else {
        return;
    };
    let mut dirs: Vec<(SystemTime, std::path::PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let modified = fs::metadata(&p)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            (modified, p)
        })
        .collect();
    dirs.sort_by_key(|(t, _)| *t);
    let now = SystemTime::now();
    for (modified, path) in &dirs {
        let aged = now.duration_since(*modified).unwrap_or_default() > ttl;
        if aged {
            let _ = fs::remove_dir_all(path);
        }
    }
    let Ok(entries) = fs::read_dir(incidents) else {
        return;
    };
    let mut remain: Vec<(SystemTime, std::path::PathBuf)> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .map(|p| {
            let modified = fs::metadata(&p)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            (modified, p)
        })
        .collect();
    remain.sort_by_key(|(t, _)| *t);
    let extra = remain.len().saturating_sub(max_count);
    for (_, path) in remain.into_iter().take(extra) {
        let _ = fs::remove_dir_all(path);
    }
    let _ = fsutil::chmod_0700(incidents);
}

pub fn applies_last_hour(path: &Path, now_ms: u64) -> u32 {
    let Ok(text) = fs::read_to_string(path) else {
        return 0;
    };
    let start = now_ms.saturating_sub(HOUR_MS);
    text.lines()
        .filter(|line| {
            serde_json::from_str::<ApplyLogLine>(line)
                .ok()
                .is_some_and(|l| l.ts_unix_ms > start && l.ts_unix_ms <= now_ms)
        })
        .count() as u32
}

pub fn hourly_cap_reached(path: &Path, cap: u32, now_ms: u64) -> bool {
    applies_last_hour(path, now_ms) >= cap
}

pub fn tail_apply(path: &Path, n: usize) -> Vec<ApplyLogLine> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut entries: Vec<ApplyLogLine> = text
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str(line).ok())
        .take(n)
        .collect();
    entries.reverse();
    entries
}

pub fn append_apply(path: &Path, id: &str, now_ms: u64) -> io::Result<()> {
    let line = ApplyLogLine {
        ts_unix_ms: now_ms,
        id: id.to_string(),
    };
    let mut encoded = serde_json::to_vec(&line).unwrap_or_default();
    encoded.push(b'\n');
    if let Some(parent) = path.parent() {
        fsutil::mkdir_0700(parent)?;
    }
    if path.exists() {
        use std::io::Write;
        let mut file = fs::OpenOptions::new().append(true).open(path)?;
        file.write_all(&encoded)?;
        fsutil::chmod_0600(path)?;
        return Ok(());
    }
    fsutil::write_0600(path, &encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hourly_cap_ignores_entries_older_than_an_hour() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("apply-log.jsonl");
        let now = 10_000_000;
        append_apply(&path, "old", now - HOUR_MS - 1).unwrap();
        append_apply(&path, "a", now - 1_000).unwrap();
        append_apply(&path, "b", now - 2_000).unwrap();
        append_apply(&path, "c", now).unwrap();
        assert_eq!(applies_last_hour(&path, now), 3);
        assert!(hourly_cap_reached(&path, 3, now));
        assert!(!hourly_cap_reached(&path, 4, now));
        assert_eq!(fsutil::unix_mode(&path).unwrap(), 0o600);
    }
}
