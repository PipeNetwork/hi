//! Append-only incident index.

use std::path::Path;

use serde::Serialize;

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

pub fn append(path: &Path, id: &str, class: &Class) -> std::io::Result<()> {
    let line = HistoryLine {
        ts_unix_ms: unix_ms(),
        id,
        class: class.class_slug(),
        kind: class.kind_slug(),
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
