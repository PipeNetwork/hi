//! Known-good snapshot and `hi.prev` copies. Does not swap binaries on crash.

use std::fs;
use std::io::{self, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::fsutil;
use crate::paths;
use hi_liveness::unix_ms;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KnownGood {
    pub schema_version: u32,
    pub recorded_unix_ms: u64,
    pub checkout_path: String,
    pub checkout_sha: String,
    pub binary_path: String,
    pub binary_blake3: String,
    #[serde(default)]
    pub prev_binary_path: String,
}

pub struct RecordInput<'a> {
    pub state_dir: &'a Path,
    pub checkout_path: Option<&'a Path>,
    pub checkout_sha: Option<&'a str>,
    pub binary_path: &'a Path,
    pub prev_binary_path: Option<&'a Path>,
}

pub fn hash_file(path: &Path) -> Option<String> {
    let mut file = fs::File::open(path).ok()?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(hasher.finalize().to_hex().to_string())
}

pub fn record(input: &RecordInput<'_>) -> io::Result<KnownGood> {
    let kg = KnownGood {
        schema_version: 1,
        recorded_unix_ms: unix_ms(),
        checkout_path: input
            .checkout_path
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
        checkout_sha: input.checkout_sha.unwrap_or("").to_string(),
        binary_path: input.binary_path.display().to_string(),
        binary_blake3: hash_file(input.binary_path).unwrap_or_default(),
        prev_binary_path: input
            .prev_binary_path
            .map(|p| p.display().to_string())
            .unwrap_or_default(),
    };
    write_known_good(input.state_dir, &kg)?;
    Ok(kg)
}

pub fn load(state_dir: &Path) -> io::Result<KnownGood> {
    let path = paths::known_good_path(state_dir);
    let bytes = fs::read(&path)?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn write_known_good(state_dir: &Path, kg: &KnownGood) -> io::Result<()> {
    fsutil::mkdir_0700(state_dir)?;
    let json =
        serde_json::to_vec_pretty(kg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fsutil::write_0600(&paths::known_good_path(state_dir), &json)
}

pub fn ensure_recorded(input: &RecordInput<'_>) -> io::Result<KnownGood> {
    match load(input.state_dir) {
        Ok(kg) => Ok(kg),
        Err(_) => record(input),
    }
}

pub fn binary_matches(kg: &KnownGood, binary: &Path) -> bool {
    match hash_file(binary) {
        Some(hash) => !hash.is_empty() && hash == kg.binary_blake3,
        None => false,
    }
}

/// Copy `src` to `dest` (0700). No-op if `src` is missing.
pub fn copy_prev(src: &Path, dest: &Path) -> io::Result<()> {
    if !src.is_file() {
        return Ok(());
    }
    if src == dest {
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        fsutil::mkdir_0700(parent)?;
    }
    fs::copy(src, dest)?;
    fsutil::chmod_0700_file(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsutil;

    #[test]
    fn record_round_trips_and_leaves_binaries_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let state = dir.path().join("state");
        fsutil::mkdir_0700(&state).unwrap();
        let bin = dir.path().join("hi");
        let prev = state.join("bin/hi.prev");
        fs::write(&bin, b"running-hi").unwrap();
        fsutil::chmod_0700_file(&bin).unwrap();
        copy_prev(&bin, &prev).unwrap();
        let kg = record(&RecordInput {
            state_dir: &state,
            checkout_path: Some(Path::new("/tmp/hi")),
            checkout_sha: Some("abc123"),
            binary_path: &bin,
            prev_binary_path: Some(&prev),
        })
        .unwrap();
        assert_eq!(kg.schema_version, 1);
        assert_eq!(kg.checkout_sha, "abc123");
        assert_eq!(fs::read_to_string(&bin).unwrap(), "running-hi");
        assert_eq!(fs::read_to_string(&prev).unwrap(), "running-hi");
        let loaded = load(&state).unwrap();
        assert_eq!(loaded.binary_blake3, kg.binary_blake3);
        assert!(binary_matches(&loaded, &bin));
        fs::write(&bin, b"replaced").unwrap();
        assert!(!binary_matches(&loaded, &bin));
        assert_eq!(fs::read_to_string(&prev).unwrap(), "running-hi");
        assert_eq!(
            fsutil::unix_mode(&paths::known_good_path(&state)).unwrap(),
            0o600
        );
    }
}
