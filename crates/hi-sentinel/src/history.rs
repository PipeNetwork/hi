//! Append-only incident index.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::classify::Class;
use crate::fsutil;
use hi_liveness::unix_ms;

#[derive(Serialize)]
struct HistoryLine<'a> {
    ts_unix_ms: u64,
    id: &'a str,
    class: &'a str,
    kind: &'a str,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HistoryEntry {
    #[allow(dead_code)]
    pub ts_unix_ms: u64,
    pub id: String,
    pub class: String,
    pub kind: String,
}

pub fn append(path: &Path, id: &str, class: &Class) -> std::io::Result<()> {
    append_raw(path, id, class.class_slug(), class.kind_slug())
}

pub fn append_raw(path: &Path, id: &str, class: &str, kind: &str) -> std::io::Result<()> {
    let line = HistoryLine {
        ts_unix_ms: unix_ms(),
        id,
        class,
        kind,
    };
    let mut encoded = serde_json::to_vec(&line).unwrap_or_default();
    encoded.push(b'\n');
    if path.exists() {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
        file.write_all(&encoded)?;
        fsutil::chmod_0600(path)?;
        return Ok(());
    }
    fsutil::write_0600(path, &encoded)
}

pub fn tail(path: &Path, n: usize) -> Vec<HistoryEntry> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let mut entries: Vec<HistoryEntry> = text
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str(line).ok())
        .take(n)
        .collect();
    entries.reverse();
    entries
}
