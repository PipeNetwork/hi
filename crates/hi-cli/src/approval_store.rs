//! SQLite-backed, one-shot approval records.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use hi_policy::{
    ApprovalDecision, ApprovalRecord, ApprovalState, ApprovalStore, CapabilityRequest,
};
use rusqlite::{Connection, OptionalExtension, params};

#[derive(Clone)]
pub(crate) struct SqliteApprovalStore {
    connection: Arc<Mutex<Connection>>,
}

impl SqliteApprovalStore {
    pub(crate) fn open(path: &std::path::Path) -> Result<Self> {
        let connection = hi_sqlite_journal::JournalMode::for_db_path(path).open(path)?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS approvals (
               approval_id TEXT PRIMARY KEY,
               run_id TEXT,
               request_json TEXT NOT NULL,
               state TEXT NOT NULL,
               created_at_ms INTEGER NOT NULL,
               expires_at_ms INTEGER NOT NULL,
               decided_at_ms INTEGER,
               consumed_at_ms INTEGER
             );
             CREATE INDEX IF NOT EXISTS approvals_pending ON approvals(state, expires_at_ms);
             CREATE INDEX IF NOT EXISTS approvals_run ON approvals(run_id, state);",
        )?;
        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    fn record_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ApprovalRecord> {
        let request_json: String = row.get(0)?;
        let request: CapabilityRequest = serde_json::from_str(&request_json).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                request_json.len(),
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
        let state: String = row.get(1)?;
        let state = match state.as_str() {
            "pending" => ApprovalState::Pending,
            "approved" => ApprovalState::Approved,
            "denied" => ApprovalState::Denied,
            "expired" => ApprovalState::Expired,
            "consumed" => ApprovalState::Consumed,
            "abandoned" => ApprovalState::Abandoned,
            other => {
                return Err(rusqlite::Error::InvalidParameterName(format!(
                    "unknown approval state {other}"
                )));
            }
        };
        Ok(ApprovalRecord {
            request,
            state,
            decided_at_ms: row.get::<_, Option<i64>>(2)?.map(|value| value as u64),
            consumed_at_ms: row.get::<_, Option<i64>>(3)?.map(|value| value as u64),
        })
    }

    fn load_locked(
        connection: &Connection,
        id: &hi_policy::ApprovalId,
    ) -> Result<Option<ApprovalRecord>> {
        connection
            .query_row(
                "SELECT request_json, state, decided_at_ms, consumed_at_ms
                 FROM approvals WHERE approval_id = ?1",
                [&id.0],
                Self::record_from_row,
            )
            .optional()
            .map_err(Into::into)
    }
}

impl ApprovalStore for SqliteApprovalStore {
    fn create(&self, request: CapabilityRequest) -> Result<ApprovalRecord> {
        let connection = self.connection.lock().unwrap();
        let json = serde_json::to_string(&request)?;
        connection.execute(
            "INSERT INTO approvals
             (approval_id, run_id, request_json, state, created_at_ms, expires_at_ms)
             VALUES (?1, ?2, ?3, 'pending', ?4, ?5)",
            params![
                request.approval_id.0,
                request.run_id,
                json,
                request.created_at_ms as i64,
                request.expires_at_ms as i64,
            ],
        )?;
        Ok(ApprovalRecord {
            request,
            state: ApprovalState::Pending,
            decided_at_ms: None,
            consumed_at_ms: None,
        })
    }

    fn get(&self, id: &hi_policy::ApprovalId) -> Result<Option<ApprovalRecord>> {
        let connection = self.connection.lock().unwrap();
        let now = hi_policy::now_ms() as i64;
        connection.execute(
            "UPDATE approvals SET state = 'expired'
             WHERE approval_id = ?1 AND state IN ('pending', 'approved') AND expires_at_ms <= ?2",
            params![&id.0, now],
        )?;
        Self::load_locked(&connection, id)
    }

    fn decide(
        &self,
        id: &hi_policy::ApprovalId,
        decision: ApprovalDecision,
    ) -> Result<ApprovalRecord> {
        let connection = self.connection.lock().unwrap();
        let tx = connection.unchecked_transaction()?;
        let mut current = tx
            .query_row(
                "SELECT request_json, state, decided_at_ms, consumed_at_ms
                 FROM approvals WHERE approval_id = ?1",
                [&id.0],
                Self::record_from_row,
            )
            .optional()?
            .context("approval request not found")?;
        if current.state != ApprovalState::Pending {
            bail!("approval is not pending: {:?}", current.state);
        }
        let now = hi_policy::now_ms();
        let next = if now >= current.request.expires_at_ms {
            ApprovalState::Expired
        } else {
            match decision {
                ApprovalDecision::Approved => ApprovalState::Approved,
                ApprovalDecision::Denied
                | ApprovalDecision::Cancelled
                | ApprovalDecision::Unavailable => ApprovalState::Denied,
            }
        };
        tx.execute(
            "UPDATE approvals SET state = ?1, decided_at_ms = ?2 WHERE approval_id = ?3",
            params![state_name(&next), now as i64, id.0],
        )?;
        tx.commit()?;
        current.state = next;
        current.decided_at_ms = Some(now);
        Ok(current)
    }

    fn claim(
        &self,
        id: &hi_policy::ApprovalId,
        digest: &hi_policy::OperationDigest,
    ) -> Result<ApprovalRecord> {
        let connection = self.connection.lock().unwrap();
        let tx = connection.unchecked_transaction()?;
        let mut current = tx
            .query_row(
                "SELECT request_json, state, decided_at_ms, consumed_at_ms
                 FROM approvals WHERE approval_id = ?1",
                [&id.0],
                Self::record_from_row,
            )
            .optional()?
            .context("approval request not found")?;
        if current.request.operation_digest != *digest {
            bail!("approval operation digest mismatch");
        }
        if current.state != ApprovalState::Approved {
            bail!("approval is not approved: {:?}", current.state);
        }
        let now = hi_policy::now_ms();
        if now >= current.request.expires_at_ms {
            tx.execute(
                "UPDATE approvals SET state = 'expired' WHERE approval_id = ?1",
                [&id.0],
            )?;
            tx.commit()?;
            bail!("approval expired");
        }
        let changed = tx.execute(
            "UPDATE approvals SET state = 'consumed', consumed_at_ms = ?1
             WHERE approval_id = ?2 AND state = 'approved'",
            params![now as i64, id.0],
        )?;
        if changed != 1 {
            bail!("approval was consumed concurrently");
        }
        tx.commit()?;
        current.state = ApprovalState::Consumed;
        current.consumed_at_ms = Some(now);
        Ok(current)
    }

    fn abandon_run(&self, run_id: &str) -> Result<u64> {
        let connection = self.connection.lock().unwrap();
        Ok(connection.execute(
            "UPDATE approvals SET state = 'abandoned'
             WHERE run_id = ?1 AND state IN ('pending', 'approved')",
            [run_id],
        )? as u64)
    }

    fn abandon_interactive(&self) -> Result<u64> {
        let connection = self.connection.lock().unwrap();
        Ok(connection.execute(
            "UPDATE approvals SET state = 'abandoned'
             WHERE run_id IS NULL AND state IN ('pending', 'approved')",
            [],
        )? as u64)
    }

    fn pending(&self) -> Result<Vec<ApprovalRecord>> {
        let connection = self.connection.lock().unwrap();
        connection.execute(
            "UPDATE approvals SET state = 'expired'
             WHERE state IN ('pending', 'approved') AND expires_at_ms <= ?1",
            [hi_policy::now_ms() as i64],
        )?;
        let mut statement = connection.prepare(
            "SELECT request_json, state, decided_at_ms, consumed_at_ms
             FROM approvals WHERE state IN ('pending', 'approved')
             ORDER BY created_at_ms ASC",
        )?;
        let rows = statement.query_map([], Self::record_from_row)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
}

fn state_name(state: &ApprovalState) -> &'static str {
    match state {
        ApprovalState::Pending => "pending",
        ApprovalState::Approved => "approved",
        ApprovalState::Denied => "denied",
        ApprovalState::Expired => "expired",
        ApprovalState::Consumed => "consumed",
        ApprovalState::Abandoned => "abandoned",
    }
}

pub(crate) fn open_for_state(state_root: &std::path::Path) -> Result<SqliteApprovalStore> {
    SqliteApprovalStore::open(&state_root.join("events.sqlite3"))
        .with_context(|| format!("opening approval store under {}", state_root.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_policy::{CapabilityKind, OperationDigest, ResourceScope, approval_request};

    fn request() -> CapabilityRequest {
        approval_request(
            CapabilityKind::WorkspaceWrite,
            ResourceScope::Operation {
                workspace_id: "w".into(),
                label: "file".into(),
            },
            OperationDigest("digest".into()),
            "edit",
            Some("run".into()),
            Some("session".into()),
            "edit file",
            "safe preview",
        )
    }

    #[test]
    fn approval_is_one_shot_and_digest_bound() {
        let dir = tempfile::tempdir().unwrap();
        let store = SqliteApprovalStore::open(&dir.path().join("events.sqlite3")).unwrap();
        let record = store.create(request()).unwrap();
        store
            .decide(&record.request.approval_id, ApprovalDecision::Approved)
            .unwrap();
        store
            .claim(
                &record.request.approval_id,
                &OperationDigest("digest".into()),
            )
            .unwrap();
        assert!(
            store
                .claim(
                    &record.request.approval_id,
                    &OperationDigest("digest".into()),
                )
                .is_err()
        );
    }
}
