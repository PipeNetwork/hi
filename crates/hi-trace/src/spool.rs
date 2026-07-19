use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};

use super::{atomic_private_write, create_private_dir, read_regular, sync_dir};

pub const DEFAULT_UPLOAD_BATCH: usize = 200;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpoolPriority {
    Normal,
    Critical,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpoolRecord {
    pub sequence: u64,
    pub priority: SpoolPriority,
    pub payload_hash: String,
    pub payload: Vec<u8>,
    pub checksum: String,
}

/// Crash-safe upload spool. Each record is an atomic segment so recovery can
/// distinguish a committed record from a torn temporary write.
pub struct DurableSpool {
    directory: PathBuf,
    acknowledged: u64,
    next_sequence: u64,
    bytes: u64,
    maximum_bytes: u64,
}

impl DurableSpool {
    pub fn open(directory: impl AsRef<Path>, maximum_bytes: u64) -> Result<Self> {
        ensure!(maximum_bytes > 0, "spool byte ceiling must be positive");
        let directory = directory.as_ref().to_owned();
        create_private_dir(&directory)?;
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with('.') {
                let _ = fs::remove_file(entry.path());
            }
        }
        let ack_path = directory.join("ack");
        let acknowledged = if ack_path.exists() {
            String::from_utf8(read_regular(&ack_path, 128)?)?
                .trim()
                .parse()?
        } else {
            atomic_private_write(&ack_path, b"0")?;
            0
        };
        let mut sequences = Vec::new();
        let mut bytes = 0_u64;
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if entry.file_name() == "ack" {
                continue;
            }
            let sequence = segment_sequence(&entry.path())?;
            let body = read_regular(&entry.path(), maximum_bytes.saturating_sub(bytes))?;
            let record: SpoolRecord = serde_json::from_slice(&body)?;
            validate_record(&record, sequence)?;
            ensure!(
                sequence > acknowledged,
                "acknowledged spool segment was not removed"
            );
            sequences.push(sequence);
            bytes = bytes
                .checked_add(body.len() as u64)
                .ok_or_else(|| anyhow!("spool size overflow"))?;
        }
        sequences.sort_unstable();
        let expected_first = acknowledged
            .checked_add(1)
            .ok_or_else(|| anyhow!("spool sequence exhausted"))?;
        if let Some(first) = sequences.first() {
            ensure!(*first == expected_first, "spool sequence gap");
        }
        ensure!(
            sequences
                .windows(2)
                .all(|pair| pair[0].checked_add(1) == Some(pair[1])),
            "spool sequence gap"
        );
        ensure!(bytes <= maximum_bytes, "spool exceeds byte ceiling");
        let next_sequence = sequences
            .last()
            .copied()
            .unwrap_or(acknowledged)
            .checked_add(1)
            .ok_or_else(|| anyhow!("spool sequence exhausted"))?;
        Ok(Self {
            directory,
            acknowledged,
            next_sequence,
            bytes,
            maximum_bytes,
        })
    }

    pub fn append(&mut self, priority: SpoolPriority, payload: &[u8]) -> Result<u64> {
        ensure!(!payload.is_empty(), "empty spool payload");
        let sequence = self.next_sequence;
        let payload_hash = blake3::hash(payload).to_hex().to_string();
        let record = SpoolRecord {
            sequence,
            priority,
            checksum: checksum(sequence, priority, &payload_hash, payload)?,
            payload_hash,
            payload: payload.to_vec(),
        };
        let encoded = serde_json::to_vec(&record)?;
        ensure!(
            self.bytes.saturating_add(encoded.len() as u64) <= self.maximum_bytes,
            "trace spool is full; producer backpressure required"
        );
        atomic_private_write(&self.segment_path(sequence), &encoded)?;
        self.bytes += encoded.len() as u64;
        self.next_sequence = sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("spool sequence exhausted"))?;
        Ok(sequence)
    }

    pub fn pending(&self, maximum: usize) -> Result<Vec<SpoolRecord>> {
        ensure!(maximum > 0 && maximum <= 10_000, "invalid spool batch size");
        let mut output = Vec::new();
        for sequence in self.acknowledged + 1..self.next_sequence {
            if output.len() == maximum {
                break;
            }
            let body = read_regular(&self.segment_path(sequence), self.maximum_bytes)?;
            let record: SpoolRecord = serde_json::from_slice(&body)?;
            validate_record(&record, sequence)?;
            output.push(record);
        }
        Ok(output)
    }

    pub fn acknowledge_contiguous(&mut self, highest: u64) -> Result<()> {
        ensure!(
            highest >= self.acknowledged && highest < self.next_sequence,
            "invalid spool acknowledgement"
        );
        for sequence in self.acknowledged + 1..=highest {
            let path = self.segment_path(sequence);
            self.bytes = self.bytes.saturating_sub(fs::metadata(&path)?.len());
            fs::remove_file(path)?;
        }
        atomic_private_write(&self.directory.join("ack"), highest.to_string().as_bytes())?;
        sync_dir(&self.directory)?;
        self.acknowledged = highest;
        Ok(())
    }

    pub fn acknowledged(&self) -> u64 {
        self.acknowledged
    }
    pub fn is_empty(&self) -> bool {
        self.acknowledged + 1 == self.next_sequence
    }
    fn segment_path(&self, sequence: u64) -> PathBuf {
        self.directory.join(format!("{sequence:020}.segment"))
    }
}

pub fn retry_delay(attempt: u32, entropy: u64) -> Duration {
    let base = 100_u64.saturating_mul(1_u64 << attempt.min(8)).min(30_000);
    Duration::from_millis(base.saturating_add(entropy % (base / 2 + 1)))
}

fn segment_sequence(path: &Path) -> Result<u64> {
    ensure!(
        path.extension().and_then(|v| v.to_str()) == Some("segment"),
        "unexpected spool entry"
    );
    Ok(path
        .file_stem()
        .and_then(|v| v.to_str())
        .ok_or_else(|| anyhow!("invalid spool segment"))?
        .parse()?)
}

fn checksum(
    sequence: u64,
    priority: SpoolPriority,
    payload_hash: &str,
    payload: &[u8],
) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(&(
        sequence,
        priority,
        payload_hash,
        payload,
    ))?)
    .to_hex()
    .to_string())
}

fn validate_record(record: &SpoolRecord, expected_sequence: u64) -> Result<()> {
    ensure!(
        record.sequence == expected_sequence,
        "spool record sequence mismatch"
    );
    ensure!(
        blake3::hash(&record.payload).to_hex().as_str() == record.payload_hash,
        "spool payload hash mismatch"
    );
    ensure!(
        checksum(
            record.sequence,
            record.priority,
            &record.payload_hash,
            &record.payload
        )? == record.checksum,
        "spool checksum mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_and_acknowledges_only_contiguous_records() {
        let dir = std::env::temp_dir().join(format!("hi-spool-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut spool = DurableSpool::open(&dir, 1024 * 1024).unwrap();
        assert_eq!(spool.append(SpoolPriority::Normal, b"one").unwrap(), 1);
        assert_eq!(
            spool.append(SpoolPriority::Critical, b"terminal").unwrap(),
            2
        );
        drop(spool);
        let mut recovered = DurableSpool::open(&dir, 1024 * 1024).unwrap();
        assert_eq!(
            recovered
                .pending(10)
                .unwrap()
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        recovered.acknowledge_contiguous(1).unwrap();
        drop(recovered);
        let recovered = DurableSpool::open(&dir, 1024 * 1024).unwrap();
        assert_eq!(recovered.acknowledged(), 1);
        assert_eq!(
            recovered.pending(10).unwrap()[0].priority,
            SpoolPriority::Critical
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
