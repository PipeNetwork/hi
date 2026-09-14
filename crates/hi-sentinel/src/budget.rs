//! Incident retention: TTL and count cap.

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

use crate::fsutil;

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
