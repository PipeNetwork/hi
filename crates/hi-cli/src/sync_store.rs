//! Crash-safe local state for portal synchronization.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MAX_EVENTS: i64 = 10_000;
const MAX_EVENT_BYTES: i64 = 25 * 1024 * 1024;
const MAX_EVENT_AGE_SECS: i64 = 15 * 60;
static PROCESS_MODE_OVERRIDE: OnceLock<SyncMode> = OnceLock::new();

pub fn set_process_mode_override(mode: SyncMode) {
    let _ = PROCESS_MODE_OVERRIDE.set(mode);
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncMode {
    On,
    Paused,
    #[default]
    Off,
}

impl SyncMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Paused => "paused",
            Self::Off => "off",
        }
    }

    fn parse(value: &str) -> Self {
        match value {
            "on" => Self::On,
            "paused" => Self::Paused,
            _ => Self::Off,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SyncStatus {
    pub mode: SyncMode,
    pub queue_rows: u64,
    pub queue_bytes: u64,
    pub oldest_item_unix: Option<u64>,
    pub last_success_unix: Option<u64>,
    pub last_error: Option<String>,
    pub next_retry_unix: Option<u64>,
    pub quarantined_records: u64,
    pub server_cursor: u64,
    pub lease_generation: u64,
    pub lease_owner: Option<String>,
    pub lease_expiry_unix: u64,
    pub event_drops: u64,
}

#[derive(Clone, Debug)]
pub struct OutboxRecord {
    pub row_id: i64,
    pub client_record_id: String,
    pub record_type: String,
    pub payload_json: String,
    pub attempts: u32,
}

pub struct SyncStore {
    connection: Mutex<Connection>,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or_default()
}

fn live_event_drop_delta(before: i64, after: i64) -> i64 {
    before.saturating_sub(after).max(0)
}

fn hex_sha256(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}

impl SyncStore {
    pub fn open() -> Result<Self> {
        let root = crate::session::data_root().context("could not determine hi data root")?;
        std::fs::create_dir_all(&root)
            .with_context(|| format!("creating sync data root {}", root.display()))?;
        Self::open_at(root.join("portal-sync.sqlite3"))
    }

    pub fn open_at(path: PathBuf) -> Result<Self> {
        let connection = Connection::open(&path)
            .with_context(|| format!("opening portal sync database {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        connection.execute_batch(
            "BEGIN IMMEDIATE;
             CREATE TABLE IF NOT EXISTS sync_settings (
               key TEXT PRIMARY KEY, value TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS record_outbox (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT NOT NULL,
               client_record_id TEXT NOT NULL UNIQUE,
               record_type TEXT NOT NULL,
               payload_json TEXT NOT NULL,
               created_at_unix INTEGER NOT NULL,
               attempts INTEGER NOT NULL DEFAULT 0,
               next_retry_unix INTEGER NOT NULL DEFAULT 0,
               last_error TEXT,
               quarantined INTEGER NOT NULL DEFAULT 0
             );
             CREATE INDEX IF NOT EXISTS record_outbox_ready
               ON record_outbox(session_id, quarantined, next_retry_unix, id);
             CREATE TABLE IF NOT EXISTS live_event_queue (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id TEXT NOT NULL,
               event_json TEXT NOT NULL,
               created_at_unix INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS session_sync (
               session_id TEXT PRIMARY KEY,
               jsonl_path TEXT,
               jsonl_offset INTEGER NOT NULL DEFAULT 0,
               server_cursor INTEGER NOT NULL DEFAULT 0,
               last_success_unix INTEGER,
               last_error TEXT,
               lease_token TEXT,
               lease_generation INTEGER NOT NULL DEFAULT 0,
               lease_owner TEXT,
               lease_expiry_unix INTEGER NOT NULL DEFAULT 0,
               event_drops INTEGER NOT NULL DEFAULT 0
             );
             -- Every pre-v2 drop count included one false drop per successful
             -- enqueue, so none of those counters are trustworthy. Reset them
             -- once, then seal the migration before using the corrected delta.
             UPDATE session_sync SET event_drops=0
               WHERE NOT EXISTS (
                 SELECT 1 FROM sync_settings
                  WHERE key='live_event_drop_formula' AND value='2'
               );
             INSERT INTO sync_settings(key,value) VALUES('live_event_drop_formula','2')
               ON CONFLICT(key) DO UPDATE SET value=excluded.value;
             PRAGMA user_version = 2;
             COMMIT;",
        )?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    /// Initialize the persisted mode once. Legacy `enabled = true` migrates
    /// to `on`; every other fresh install defaults to `off`.
    pub fn initialize_mode(&self, legacy_enabled: bool) -> Result<SyncMode> {
        let connection = self.connection.lock().unwrap();
        let existing: Option<String> = connection
            .query_row(
                "SELECT value FROM sync_settings WHERE key='mode'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(value) = existing {
            return Ok(SyncMode::parse(&value));
        }
        let mode = if legacy_enabled {
            SyncMode::On
        } else {
            SyncMode::Off
        };
        connection.execute(
            "INSERT INTO sync_settings(key,value) VALUES('mode',?1)",
            [mode.as_str()],
        )?;
        Ok(mode)
    }

    pub fn mode(&self) -> Result<SyncMode> {
        let connection = self.connection.lock().unwrap();
        let value: Option<String> = connection
            .query_row(
                "SELECT value FROM sync_settings WHERE key='mode'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value.as_deref().map(SyncMode::parse).unwrap_or_default())
    }

    pub fn effective_mode(&self) -> Result<SyncMode> {
        PROCESS_MODE_OVERRIDE
            .get()
            .copied()
            .map(Ok)
            .unwrap_or_else(|| self.mode())
    }

    pub fn set_mode(&self, mode: SyncMode) -> Result<()> {
        self.connection.lock().unwrap().execute(
            "INSERT INTO sync_settings(key,value) VALUES('mode',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [mode.as_str()],
        )?;
        Ok(())
    }

    pub fn enqueue_record(
        &self,
        session_id: &str,
        record_type: &str,
        payload_json: &str,
    ) -> Result<()> {
        if self.effective_mode()? == SyncMode::Off {
            return Ok(());
        }
        let mut connection = self.connection.lock().unwrap();
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO record_outbox(session_id,client_record_id,record_type,payload_json,created_at_unix)
             VALUES(?1,'pending',?2,?3,?4)",
            params![session_id, record_type, payload_json, now()],
        )?;
        let row_id = transaction.last_insert_rowid();
        let id = hex_sha256(&[
            session_id.as_bytes(),
            b"\0",
            row_id.to_string().as_bytes(),
            b"\0",
            record_type.as_bytes(),
            b"\0",
            payload_json.as_bytes(),
        ]);
        transaction.execute(
            "UPDATE record_outbox SET client_record_id=?1 WHERE id=?2",
            params![id, row_id],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn enqueue_record_with_id(
        &self,
        session_id: &str,
        client_record_id: &str,
        record_type: &str,
        payload_json: &str,
    ) -> Result<()> {
        if self.effective_mode()? == SyncMode::Off {
            return Ok(());
        }
        self.connection.lock().unwrap().execute(
            "INSERT OR IGNORE INTO record_outbox(session_id,client_record_id,record_type,payload_json,created_at_unix)
             VALUES(?1,?2,?3,?4,?5)",
            params![session_id, client_record_id, record_type, payload_json, now()],
        )?;
        Ok(())
    }

    pub fn track_jsonl(&self, session_id: &str, path: &std::path::Path) -> Result<u64> {
        let initial = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        let path = path.to_string_lossy();
        let connection = self.connection.lock().unwrap();
        connection.execute(
            "INSERT OR IGNORE INTO session_sync(session_id,jsonl_path,jsonl_offset) VALUES(?1,?2,?3)",
            params![session_id, path.as_ref(), initial as i64],
        )?;
        let (tracked_path, mut offset) = connection.query_row(
            "SELECT jsonl_path,jsonl_offset FROM session_sync WHERE session_id=?1",
            [session_id],
            |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?)),
        )?;
        if tracked_path.as_deref() != Some(path.as_ref()) {
            offset = initial as i64;
            connection.execute(
                "UPDATE session_sync SET jsonl_path=?1,jsonl_offset=?2 WHERE session_id=?3",
                params![path.as_ref(), offset, session_id],
            )?;
        }
        Ok(offset.max(0) as u64)
    }

    pub fn set_jsonl_offset(&self, session_id: &str, offset: u64) -> Result<()> {
        self.connection.lock().unwrap().execute(
            "UPDATE session_sync SET jsonl_offset=?1 WHERE session_id=?2",
            params![offset as i64, session_id],
        )?;
        Ok(())
    }

    pub fn ready_records(&self, session_id: &str, limit: usize) -> Result<Vec<OutboxRecord>> {
        let connection = self.connection.lock().unwrap();
        let mut statement = connection.prepare(
            "SELECT id,client_record_id,record_type,payload_json,attempts
             FROM record_outbox
             WHERE session_id=?1 AND quarantined=0 AND next_retry_unix<=?2
             ORDER BY id LIMIT ?3",
        )?;
        let rows = statement.query_map(params![session_id, now(), limit as i64], |row| {
            Ok(OutboxRecord {
                row_id: row.get(0)?,
                client_record_id: row.get(1)?,
                record_type: row.get(2)?,
                payload_json: row.get(3)?,
                attempts: row.get(4)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn acknowledge_records(&self, session_id: &str, ids: &[i64], cursor: u64) -> Result<()> {
        let mut connection = self.connection.lock().unwrap();
        let transaction = connection.transaction()?;
        for id in ids {
            transaction.execute("DELETE FROM record_outbox WHERE id=?1", [id])?;
        }
        transaction.execute(
            "INSERT INTO session_sync(session_id,server_cursor,last_success_unix,last_error)
             VALUES(?1,?2,?3,NULL)
             ON CONFLICT(session_id) DO UPDATE SET
               server_cursor=MAX(server_cursor,excluded.server_cursor),
               last_success_unix=excluded.last_success_unix,last_error=NULL",
            params![session_id, cursor, now()],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn fail_records(
        &self,
        session_id: &str,
        records: &[OutboxRecord],
        error: &str,
        retry_after_secs: Option<u64>,
        permanent: bool,
    ) -> Result<()> {
        let connection = self.connection.lock().unwrap();
        for record in records {
            let exponential = 2u64.saturating_pow(record.attempts.min(10) + 1).min(900);
            let jitter = record.row_id.unsigned_abs() % (exponential / 4 + 1);
            let retry = retry_after_secs.unwrap_or(exponential + jitter);
            connection.execute(
                "UPDATE record_outbox SET attempts=attempts+1,next_retry_unix=?1,last_error=?2,quarantined=?3 WHERE id=?4",
                params![now().saturating_add(retry as i64), error, permanent, record.row_id],
            )?;
        }
        connection.execute(
            "INSERT INTO session_sync(session_id,last_error) VALUES(?1,?2)
             ON CONFLICT(session_id) DO UPDATE SET last_error=excluded.last_error",
            params![session_id, error],
        )?;
        Ok(())
    }

    pub fn enqueue_event(&self, session_id: &str, event_json: &str) -> Result<()> {
        if self.effective_mode()? == SyncMode::Off {
            return Ok(());
        }
        let mut connection = self.connection.lock().unwrap();
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO live_event_queue(session_id,event_json,created_at_unix) VALUES(?1,?2,?3)",
            params![session_id, event_json, now()],
        )?;
        let cutoff = now() - MAX_EVENT_AGE_SECS;
        let before: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM live_event_queue WHERE session_id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        transaction.execute(
            "DELETE FROM live_event_queue WHERE session_id=?1 AND created_at_unix<?2",
            params![session_id, cutoff],
        )?;
        while transaction.query_row(
            "SELECT COUNT(*) FROM live_event_queue WHERE session_id=?1",
            [session_id], |row| row.get::<_, i64>(0))? > MAX_EVENTS
            || transaction.query_row(
                "SELECT COALESCE(SUM(LENGTH(event_json)),0) FROM live_event_queue WHERE session_id=?1",
                [session_id], |row| row.get::<_, i64>(0))? > MAX_EVENT_BYTES
        {
            transaction.execute(
                "DELETE FROM live_event_queue WHERE id=(SELECT id FROM live_event_queue WHERE session_id=?1 ORDER BY id LIMIT 1)",
                [session_id],
            )?;
        }
        let after: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM live_event_queue WHERE session_id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        // `before` is measured after inserting the new event. Only rows removed
        // by age/count/byte enforcement are drops.
        let dropped = live_event_drop_delta(before, after);
        transaction.execute(
            "INSERT INTO session_sync(session_id,event_drops) VALUES(?1,?2)
             ON CONFLICT(session_id) DO UPDATE SET event_drops=event_drops+excluded.event_drops",
            params![session_id, dropped],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn ready_events(&self, session_id: &str, limit: usize) -> Result<Vec<(i64, String)>> {
        let connection = self.connection.lock().unwrap();
        let mut statement = connection.prepare(
            "SELECT id,event_json FROM live_event_queue WHERE session_id=?1 ORDER BY id LIMIT ?2",
        )?;
        let rows = statement.query_map(params![session_id, limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn acknowledge_events(&self, ids: &[i64]) -> Result<()> {
        let connection = self.connection.lock().unwrap();
        for id in ids {
            connection.execute("DELETE FROM live_event_queue WHERE id=?1", [id])?;
        }
        Ok(())
    }

    pub fn store_lease(
        &self,
        session_id: &str,
        token: &str,
        generation: u64,
        owner: &str,
        expiry: u64,
    ) -> Result<()> {
        self.connection.lock().unwrap().execute(
            "INSERT INTO session_sync(session_id,lease_token,lease_generation,lease_owner,lease_expiry_unix)
             VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(session_id) DO UPDATE SET lease_token=excluded.lease_token,
               lease_generation=excluded.lease_generation,lease_owner=excluded.lease_owner,
               lease_expiry_unix=excluded.lease_expiry_unix",
            params![session_id, token, generation, owner, expiry],
        )?;
        Ok(())
    }

    pub fn lease_token(&self, session_id: &str) -> Result<Option<String>> {
        Ok(self
            .connection
            .lock()
            .unwrap()
            .query_row(
                "SELECT lease_token FROM session_sync WHERE session_id=?1",
                [session_id],
                |row| row.get(0),
            )
            .optional()?
            .flatten())
    }

    pub fn purge(&self) -> Result<()> {
        self.connection.lock().unwrap().execute_batch(
            "DELETE FROM record_outbox; DELETE FROM live_event_queue; DELETE FROM session_sync;",
        )?;
        Ok(())
    }

    pub fn status(&self, session_id: Option<&str>) -> Result<SyncStatus> {
        let mode = self.effective_mode()?;
        let connection = self.connection.lock().unwrap();
        let where_clause = if session_id.is_some() {
            " WHERE session_id=?1"
        } else {
            ""
        };
        let sql = format!(
            "SELECT COUNT(*),COALESCE(SUM(LENGTH(payload_json)),0),MIN(created_at_unix),SUM(quarantined),MIN(NULLIF(next_retry_unix,0)) FROM record_outbox{where_clause}"
        );
        let query = |row: &rusqlite::Row<'_>| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<i64>>(4)?,
            ))
        };
        let (rows, bytes, oldest, quarantined, next_retry) = if let Some(id) = session_id {
            connection.query_row(&sql, [id], query)?
        } else {
            connection.query_row(&sql, [], query)?
        };
        let metadata = session_id.and_then(|id| connection.query_row(
            "SELECT server_cursor,last_success_unix,last_error,lease_generation,lease_owner,lease_expiry_unix,event_drops FROM session_sync WHERE session_id=?1",
            [id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?, row.get::<_, Option<String>>(2)?, row.get::<_, i64>(3)?, row.get::<_, Option<String>>(4)?, row.get::<_, i64>(5)?, row.get::<_, i64>(6)?)),
        ).optional().ok().flatten()).unwrap_or_default();
        Ok(SyncStatus {
            mode,
            queue_rows: rows as u64,
            queue_bytes: bytes as u64,
            oldest_item_unix: oldest.map(|v| v as u64),
            last_success_unix: metadata.1.map(|v| v as u64),
            last_error: metadata.2,
            next_retry_unix: next_retry.map(|v| v as u64),
            quarantined_records: quarantined.unwrap_or(0) as u64,
            server_cursor: metadata.0 as u64,
            lease_generation: metadata.3 as u64,
            lease_owner: metadata.4,
            lease_expiry_unix: metadata.5 as u64,
            event_drops: metadata.6 as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store_path(tag: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hi-sync-{tag}-{}-{nonce}.sqlite",
            std::process::id()
        ))
    }

    #[test]
    fn mode_outbox_and_purge_survive_reopen() {
        let path = temp_store_path("mode-outbox");
        let store = SyncStore::open_at(path.clone()).unwrap();
        assert_eq!(store.initialize_mode(false).unwrap(), SyncMode::Off);
        store.enqueue_record("s", "message", "{}").unwrap();
        assert!(store.ready_records("s", 10).unwrap().is_empty());
        store.set_mode(SyncMode::Paused).unwrap();
        store.enqueue_record("s", "message", "{}").unwrap();
        drop(store);
        let store = SyncStore::open_at(path.clone()).unwrap();
        assert_eq!(store.mode().unwrap(), SyncMode::Paused);
        assert_eq!(store.ready_records("s", 10).unwrap().len(), 1);
        store.purge().unwrap();
        assert!(store.ready_records("s", 10).unwrap().is_empty());
        drop(store);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn live_event_drop_delta_counts_only_removed_rows() {
        assert_eq!(live_event_drop_delta(2, 2), 0);
        assert_eq!(live_event_drop_delta(5, 2), 3);
        assert_eq!(live_event_drop_delta(2, 3), 0);
    }

    #[test]
    fn opening_store_resets_polluted_legacy_drop_counts_once() {
        let path = temp_store_path("event-drop-migration");
        let store = SyncStore::open_at(path.clone()).unwrap();
        {
            let connection = store.connection.lock().unwrap();
            connection
                .execute(
                    "DELETE FROM sync_settings WHERE key='live_event_drop_formula'",
                    [],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO session_sync(session_id,event_drops) VALUES('legacy',633)",
                    [],
                )
                .unwrap();
        }
        drop(store);

        let store = SyncStore::open_at(path.clone()).unwrap();
        assert_eq!(store.status(Some("legacy")).unwrap().event_drops, 0);
        store
            .connection
            .lock()
            .unwrap()
            .execute(
                "UPDATE session_sync SET event_drops=7 WHERE session_id='legacy'",
                [],
            )
            .unwrap();
        drop(store);

        let reopened = SyncStore::open_at(path.clone()).unwrap();
        assert_eq!(reopened.status(Some("legacy")).unwrap().event_drops, 7);
        drop(reopened);
        let _ = std::fs::remove_file(path);
    }
}
