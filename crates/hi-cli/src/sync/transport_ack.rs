use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct AppendRecordsReceipt {
    record_count: u64,
}

impl AppendRecordsReceipt {
    pub(super) fn validate(self, previous_cursor: u64, submitted_records: usize) -> Result<u64> {
        let submitted = u64::try_from(submitted_records)
            .context("transcript batch record count exceeds u64")?;
        let minimum_cursor = previous_cursor
            .checked_add(submitted)
            .ok_or_else(|| anyhow!("transcript acknowledgement cursor overflow"))?;
        ensure!(
            self.record_count >= minimum_cursor,
            "transcript acknowledgement cursor {} does not cover {} submitted record(s) after cursor {}; outbox retained",
            self.record_count,
            submitted,
            previous_cursor
        );
        Ok(self.record_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acknowledgement_is_typed_and_must_advance_for_the_batch() {
        assert!(serde_json::from_str::<AppendRecordsReceipt>("{}").is_err());
        let stale = AppendRecordsReceipt { record_count: 11 };
        assert!(stale.validate(10, 2).is_err());
        let complete = AppendRecordsReceipt { record_count: 12 };
        assert_eq!(complete.validate(10, 2).unwrap(), 12);
    }
}
