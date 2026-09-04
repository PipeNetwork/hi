use anyhow::{Context, Result, ensure};

use super::{RemoteSessionSink, workspace_execution::project_causal_workspace_execution};

const MAX_CAUSAL_TRANSCRIPT_BYTES: usize = 5_500_000;
const MAX_CAUSAL_TRANSCRIPT_RECORDS: usize = 512;

impl RemoteSessionSink {
    /// Drain the stable transcript prefix before a causal mutation is
    /// admitted. Records produced by the mutation then fit in a fresh causal
    /// batch instead of being stranded behind an unrelated old backlog.
    pub(crate) async fn prepare_causal_pipefs_mutation(&self) -> Result<()> {
        self.ensure_workspace_execution_staged()?;
        self.flush_required().await?;
        let status = self.store.status(Some(&self.session_id))?;
        ensure!(
            status.queue_rows == 0,
            "PipeFS causal admission requires an empty transcript prefix; {} record(s) remain",
            status.queue_rows
        );
        Ok(())
    }

    pub(crate) fn causal_pipefs_transcript_batch(
        &self,
    ) -> Result<hi_pipefs::CausalTranscriptBatch> {
        ensure!(
            self.pipefs_sync_required(),
            "PipeFS transcript durability is not pinned for this session"
        );
        self.store.force_retry_records(&self.session_id)?;
        let status = self.store.status(Some(&self.session_id))?;
        let count = usize::try_from(status.queue_rows)
            .context("PipeFS transcript outbox is too large to address")?;
        ensure!(
            count <= MAX_CAUSAL_TRANSCRIPT_RECORDS,
            "PipeFS causal transcript batch has {count} records; flush it before causal commit"
        );
        let rows = self.store.ready_records(&self.session_id, count.max(1))?;
        ensure!(
            rows.len() == count,
            "PipeFS transcript outbox could not be read as one causal batch"
        );
        let mut wire_bytes = 0usize;
        let records = rows
            .into_iter()
            .map(|row| {
                wire_bytes = wire_bytes.saturating_add(row.payload_json.len() + 256);
                let payload = serde_json::from_str(&row.payload_json)
                    .context("decoding a causal transcript record")?;
                let (record_type, payload) =
                    project_causal_workspace_execution(&row.record_type, payload)?;
                Ok(hi_pipefs::CausalTranscriptRecord {
                    record_id: u64::try_from(row.row_id)
                        .context("PipeFS transcript record ID is invalid")?,
                    client_record_id: row.client_record_id,
                    record_type,
                    payload,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            wire_bytes <= MAX_CAUSAL_TRANSCRIPT_BYTES,
            "PipeFS causal transcript batch is too large; flush it before causal commit"
        );
        Ok(hi_pipefs::CausalTranscriptBatch { records })
    }

    pub(crate) fn acknowledge_causal_pipefs_transcript(
        &self,
        batch: &hi_pipefs::CausalTranscriptBatch,
        cursor: u64,
    ) -> Result<()> {
        let ids = batch
            .records
            .iter()
            .map(|record| {
                i64::try_from(record.record_id)
                    .context("causal transcript record ID exceeds SQLite range")
            })
            .collect::<Result<Vec<_>>>()?;
        let local = self.store.records_by_id(&self.session_id, &ids)?;
        ensure!(
            local.is_empty() || local.len() == ids.len(),
            "causal transcript acknowledgement found only {} of {} local record(s); outbox retained",
            local.len(),
            ids.len()
        );
        for (row, submitted) in local.iter().zip(&batch.records) {
            let payload = serde_json::from_str(&row.payload_json)
                .context("decoding a causal transcript record before acknowledgement")?;
            let (record_type, payload) =
                project_causal_workspace_execution(&row.record_type, payload)?;
            ensure!(
                row.row_id == i64::try_from(submitted.record_id)?
                    && row.client_record_id == submitted.client_record_id
                    && record_type == submitted.record_type
                    && payload == submitted.payload,
                "causal transcript acknowledgement does not exactly match the durable outbox; outbox retained"
            );
        }
        let previous_cursor = self.store.status(Some(&self.session_id))?.server_cursor;
        let submitted =
            u64::try_from(local.len()).context("causal transcript record count exceeds u64")?;
        let minimum_cursor = previous_cursor
            .checked_add(submitted)
            .context("causal transcript acknowledgement cursor overflow")?;
        ensure!(
            cursor >= minimum_cursor,
            "causal transcript cursor {cursor} does not acknowledge {submitted} submitted record(s) after cursor {previous_cursor}; outbox retained"
        );
        self.store
            .acknowledge_records(&self.session_id, &ids, cursor)
    }
}
