use anyhow::{Context, Result, ensure};
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};

use super::SyncStore;

impl SyncStore {
    pub fn records_by_id(&self, session_id: &str, ids: &[i64]) -> Result<Vec<super::OutboxRecord>> {
        let connection = self.connection.lock().unwrap();
        let mut statement = connection.prepare(
            "SELECT id,client_record_id,record_type,payload_json,attempts
             FROM record_outbox WHERE session_id=?1 AND id=?2",
        )?;
        let mut records = Vec::with_capacity(ids.len());
        for id in ids {
            let record = statement
                .query_row(params![session_id, id], |row| {
                    Ok(super::OutboxRecord {
                        row_id: row.get(0)?,
                        client_record_id: row.get(1)?,
                        record_type: row.get(2)?,
                        payload_json: row.get(3)?,
                        attempts: row.get(4)?,
                    })
                })
                .optional()?;
            records.extend(record);
        }
        Ok(records)
    }

    pub fn pending_workspace_execution(
        &self,
        session_id: &str,
        operation_id: &str,
        execution_digest: &str,
    ) -> Result<bool> {
        let connection = self.connection.lock().unwrap();
        let mut statement = connection.prepare(
            "SELECT payload_json FROM record_outbox
             WHERE session_id=?1 AND record_type='usage' ORDER BY id",
        )?;
        let rows = statement.query_map([session_id], |row| row.get::<_, String>(0))?;
        for payload in rows {
            let payload = payload?;
            if execution_identity(&payload)?.is_some_and(|(candidate, digest)| {
                candidate == operation_id && digest == execution_digest
            }) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn workspace_execution_ack_cursor(
        &self,
        session_id: &str,
        operation_id: &str,
        execution_digest: &str,
    ) -> Result<Option<u64>> {
        let cursor = self
            .connection
            .lock()
            .unwrap()
            .query_row(
                "SELECT server_cursor FROM pipefs_workspace_execution_acks
                 WHERE session_id=?1 AND operation_id=?2 AND execution_digest=?3",
                params![session_id, operation_id, execution_digest],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        cursor
            .map(|cursor| {
                u64::try_from(cursor).context("stored workspace execution cursor is invalid")
            })
            .transpose()
    }
}

pub(super) fn acknowledge(
    store: &SyncStore,
    session_id: &str,
    ids: &[i64],
    cursor: u64,
) -> Result<()> {
    let mut connection = store.connection.lock().unwrap();
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    retain_workspace_execution_acks(&transaction, session_id, ids, cursor)?;
    for id in ids {
        transaction.execute("DELETE FROM record_outbox WHERE id=?1", [id])?;
    }
    transaction.execute(
        "INSERT INTO session_sync(session_id,server_cursor,last_success_unix,last_error)
         VALUES(?1,?2,?3,NULL)
         ON CONFLICT(session_id) DO UPDATE SET
           server_cursor=MAX(server_cursor,excluded.server_cursor),
           last_success_unix=excluded.last_success_unix,last_error=NULL",
        params![session_id, cursor, super::now()],
    )?;
    transaction.commit()?;
    Ok(())
}

fn retain_workspace_execution_acks(
    transaction: &Transaction<'_>,
    session_id: &str,
    ids: &[i64],
    cursor: u64,
) -> Result<()> {
    let cursor =
        i64::try_from(cursor).context("workspace execution cursor exceeds SQLite range")?;
    for id in ids {
        let row = transaction
            .query_row(
                "SELECT record_type,payload_json FROM record_outbox
                 WHERE session_id=?1 AND id=?2",
                params![session_id, id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        let Some((record_type, payload)) = row else {
            continue;
        };
        if record_type != "usage" {
            continue;
        }
        let Some((operation_id, execution_digest)) = execution_identity(&payload)? else {
            continue;
        };
        transaction.execute(
            "INSERT INTO pipefs_workspace_execution_acks(
               session_id,operation_id,execution_digest,server_cursor
             ) VALUES(?1,?2,?3,?4)
             ON CONFLICT(session_id,operation_id,execution_digest) DO UPDATE SET
               server_cursor=MAX(server_cursor,excluded.server_cursor)",
            params![session_id, operation_id, execution_digest, cursor],
        )?;
    }
    // The proof is needed only across the narrow transcript-ack/workspace-cleanup
    // crash boundary. Bound retained history without risking the newest retries.
    transaction.execute(
        "DELETE FROM pipefs_workspace_execution_acks
         WHERE session_id=?1 AND rowid NOT IN (
           SELECT rowid FROM pipefs_workspace_execution_acks
           WHERE session_id=?1 ORDER BY server_cursor DESC, rowid DESC LIMIT 1024
         )",
        [session_id],
    )?;
    Ok(())
}

pub(crate) fn execution_digest(
    operation_id: &str,
    execution: &serde_json::Value,
) -> Result<String> {
    ensure!(
        !operation_id.trim().is_empty(),
        "workspace execution has no operation ID"
    );
    let bytes = serde_json::to_vec(&(operation_id, execution))?;
    Ok(super::hex_sha256(&[
        b"hi.pipefs.workspace-execution.v1\0",
        &bytes,
    ]))
}

fn execution_identity(payload: &str) -> Result<Option<(String, String)>> {
    let value: serde_json::Value =
        serde_json::from_str(payload).context("decoding durable workspace execution evidence")?;
    if value.get("type").and_then(serde_json::Value::as_str) != Some("workspace_execution") {
        return Ok(None);
    }
    let operation_id = value
        .get("operation_id")
        .and_then(serde_json::Value::as_str)
        .context("workspace execution evidence has no operation ID")?;
    let execution = value
        .get("execution")
        .context("workspace execution evidence has no execution report")?;
    Ok(Some((
        operation_id.to_owned(),
        execution_digest(operation_id, execution)?,
    )))
}
