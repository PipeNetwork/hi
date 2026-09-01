//! Crash-safe local state for portal synchronization.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MAX_EVENTS: i64 = 10_000;
const MAX_EVENT_BYTES: i64 = 25 * 1024 * 1024;
const MAX_EVENT_AGE_SECS: i64 = 15 * 60;
static PROCESS_MODE_OVERRIDE: OnceLock<SyncMode> = OnceLock::new();
static PROCESS_MODE_DEFAULT: OnceLock<SyncMode> = OnceLock::new();

pub fn set_process_mode_override(mode: SyncMode) {
    let _ = PROCESS_MODE_OVERRIDE.set(mode);
}

/// Mode used when the user has never chosen one (no persisted row). Set once
/// at startup after the provider is known: a pipenetwork pairing syncs by
/// default — the records go to the user's own account on the provider that
/// already serves every prompt — while everything else stays off. Unlike the
/// override, an explicit `/sync on|off` (which persists a row) beats this.
pub fn set_process_mode_default(mode: SyncMode) {
    let _ = PROCESS_MODE_DEFAULT.set(mode);
}

/// Precedence: `--sync`-style process override, then the user's persisted
/// choice, then the provider default, then off.
fn resolve_mode(
    process_override: Option<SyncMode>,
    stored: Option<SyncMode>,
    default: Option<SyncMode>,
) -> SyncMode {
    process_override.or(stored).or(default).unwrap_or_default()
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

#[derive(Clone, Debug, Default, Serialize)]
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

/// True when any cause in the chain is SQLite's "database is locked"
/// (SQLITE_BUSY / SQLITE_LOCKED) — transient peer contention, safe to retry.
fn is_busy_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<rusqlite::Error>(),
            Some(rusqlite::Error::SqliteFailure(failure, _)) if matches!(
                failure.code,
                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
            )
        )
    })
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

    pub(crate) fn in_memory() -> Self {
        let connection = Connection::open_in_memory().expect("in-memory portal sync database");
        // Minimal schema so enqueue/status calls don't fail; full migrations
        // run via open_at for durable stores, but in-memory fallback just
        // needs to be non-panicking and best-effort.
        let _ = connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS sync_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS record_outbox (
               id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
               client_record_id TEXT NOT NULL UNIQUE, record_type TEXT NOT NULL,
               payload_json TEXT NOT NULL, created_at_unix INTEGER NOT NULL,
               attempts INTEGER NOT NULL DEFAULT 0, next_retry_unix INTEGER NOT NULL DEFAULT 0,
               last_error TEXT, quarantined INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS live_event_queue (
               id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
               event_json TEXT NOT NULL, created_at_unix INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS session_sync (
               session_id TEXT PRIMARY KEY, jsonl_path TEXT,
               jsonl_offset INTEGER NOT NULL DEFAULT 0, server_cursor INTEGER NOT NULL DEFAULT 0,
               last_success_unix INTEGER, last_error TEXT, lease_token TEXT,
               lease_generation INTEGER NOT NULL DEFAULT 0, lease_owner TEXT,
               lease_expiry_unix INTEGER NOT NULL DEFAULT 0, event_drops INTEGER NOT NULL DEFAULT 0
             );",
        );
        Self {
            connection: Mutex::new(connection),
        }
    }

    /// Read existing local sync state without creating a database or running migrations.
    /// Doctor uses this path so an absent store remains an inexpensive, valid state.
    pub fn status_if_available(session_id: Option<&str>) -> Result<Option<SyncStatus>> {
        let root = crate::session::data_root().context("could not determine hi data root")?;
        let path = root.join("portal-sync.sqlite3");
        if !path.is_file() {
            return Ok(None);
        }
        let connection = Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening local sync state {}", path.display()))?;
        connection.busy_timeout(std::time::Duration::from_millis(75))?;
        match status_from_connection(&connection, session_id) {
            Ok(status) => Ok(Some(status)),
            // Recoverable states must not fail a doctor run: a pre-migration
            // database (missing table/column) migrates on the next full open,
            // and after a crash a read-only connection cannot run WAL recovery
            // or wait out a writer ("unable to open", "locked", "readonly") —
            // the next normal hi run heals those too. Only genuine corruption
            // should propagate.
            Err(error)
                if {
                    let text = format!("{error:#}");
                    [
                        "no such table",
                        "no such column",
                        "unable to open database",
                        "database is locked",
                        "readonly database",
                    ]
                    .iter()
                    .any(|needle| text.contains(needle))
                } =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    pub fn open_at(path: PathBuf) -> Result<Self> {
        // Peer hi processes (TUI, daemon, background jobs) share this file
        // and hold short write locks. busy_timeout in try_open_at absorbs
        // ordinary contention; this bounded retry absorbs a peer that holds
        // the lock across an entire timeout window.
        let mut attempts = 0u64;
        loop {
            match Self::try_open_at(&path) {
                Err(error) if attempts < 3 && is_busy_error(&error) => {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(100 * attempts));
                }
                result => return result,
            }
        }
    }

    fn try_open_at(path: &std::path::Path) -> Result<Self> {
        // hi-sqlite-journal owns the lock-safe open: busy_timeout before any
        // lock-taking statement, a poll loop for the journal-mode switch
        // (which SQLite refuses to apply busy_timeout to), and a rollback
        // journal instead of WAL on network filesystems.
        let connection = hi_sqlite_journal::JournalMode::for_db_path(path)
            .open(path)
            .with_context(|| format!("opening portal sync database {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        connection.pragma_update(None, "synchronous", "FULL")?;
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
    /// to `on`; otherwise nothing is written — an absent row means "the user
    /// never chose", which the provider default fills at read time. (An
    /// earlier version persisted an implicit `off` here, which silently
    /// killed sync for every pipenetwork install; see
    /// [`Self::heal_implicit_off`].)
    pub fn initialize_mode(&self, legacy_enabled: bool) -> Result<Option<SyncMode>> {
        let connection = self.connection.lock().unwrap();
        let existing: Option<String> = connection
            .query_row(
                "SELECT value FROM sync_settings WHERE key='mode'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(value) = existing {
            return Ok(Some(SyncMode::parse(&value)));
        }
        if legacy_enabled {
            connection.execute(
                "INSERT INTO sync_settings(key,value) VALUES('mode','on')",
                [],
            )?;
            return Ok(Some(SyncMode::On));
        }
        Ok(None)
    }

    /// One-time repair for stores poisoned by the old implicit-`off`
    /// initialize: an `off` row that no user action wrote (no `mode_source`
    /// marker — [`Self::set_mode`] stamps one) is deleted so the provider
    /// default applies again. An explicit `/sync off` from before the marker
    /// existed is indistinguishable and gets re-defaulted once; it sticks
    /// again the moment the user repeats it.
    pub fn heal_implicit_off(&self) -> Result<bool> {
        let connection = self.connection.lock().unwrap();
        let healed = connection.execute(
            "DELETE FROM sync_settings
             WHERE key='mode' AND value='off'
               AND NOT EXISTS (
                 SELECT 1 FROM sync_settings WHERE key='mode_source'
               )",
            [],
        )?;
        Ok(healed > 0)
    }

    pub fn stored_mode(&self) -> Result<Option<SyncMode>> {
        let connection = self.connection.lock().unwrap();
        let value: Option<String> = connection
            .query_row(
                "SELECT value FROM sync_settings WHERE key='mode'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value.as_deref().map(SyncMode::parse))
    }

    pub fn effective_mode(&self) -> Result<SyncMode> {
        Ok(resolve_mode(
            PROCESS_MODE_OVERRIDE.get().copied(),
            self.stored_mode()?,
            PROCESS_MODE_DEFAULT.get().copied(),
        ))
    }

    pub fn set_mode(&self, mode: SyncMode) -> Result<()> {
        // The marker records that a person (or their config file) chose this
        // mode, which is what exempts the row from heal_implicit_off.
        self.connection.lock().unwrap().execute(
            "INSERT INTO sync_settings(key,value) VALUES('mode',?1), ('mode_source','user')
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [mode.as_str()],
        )?;
        Ok(())
    }

    /// Endpoint circuit breaker, shared by every hi process through this
    /// database. When the portal is unreachable, one process pays a short
    /// connect timeout, trips the breaker, and every sync path in every
    /// process skips network work until the cooldown passes — records and
    /// events stay queued in the outbox for later. Without this, a dead
    /// endpoint stacked 10s timeouts onto startups, turn ends, one-shot
    /// exits, and session switches all day.
    pub fn breaker_open_until(&self) -> Result<Option<i64>> {
        let connection = self.connection.lock().unwrap();
        let value: Option<String> = connection
            .query_row(
                "SELECT value FROM sync_settings WHERE key='endpoint_down_until'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(value.and_then(|v| v.parse().ok()))
    }

    /// Record an endpoint connect/timeout failure at `now_unix`. The cooldown
    /// doubles per consecutive trip (60s → 900s cap) and the breaker opens
    /// until `now + cooldown`. Returns the open-until timestamp.
    pub fn trip_breaker(&self, now_unix: i64) -> Result<i64> {
        const BASE_SECS: i64 = 60;
        const CAP_SECS: i64 = 900;
        let connection = self.connection.lock().unwrap();
        let previous: Option<String> = connection
            .query_row(
                "SELECT value FROM sync_settings WHERE key='endpoint_backoff_secs'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let backoff = previous
            .and_then(|v| v.parse::<i64>().ok())
            .map(|b| (b * 2).min(CAP_SECS))
            .unwrap_or(BASE_SECS);
        let until = now_unix + backoff;
        connection.execute(
            "INSERT INTO sync_settings(key,value) VALUES('endpoint_backoff_secs',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [backoff.to_string()],
        )?;
        connection.execute(
            "INSERT INTO sync_settings(key,value) VALUES('endpoint_down_until',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [until.to_string()],
        )?;
        Ok(until)
    }

    /// Clear the breaker after any successful endpoint round-trip.
    pub fn reset_breaker(&self) -> Result<()> {
        let connection = self.connection.lock().unwrap();
        connection.execute(
            "DELETE FROM sync_settings WHERE key IN (
               'endpoint_down_until','endpoint_backoff_secs','endpoint_timeout_streak'
             )",
            [],
        )?;
        Ok(())
    }

    /// Count a request timeout. Unlike a refused connection, a timeout is
    /// ambiguous — a live server under a slow persist looks the same as a
    /// dead one — so one of them must not blackhole every sync path for a
    /// minute. Returns the consecutive streak; the caller trips the breaker
    /// once the streak is long enough to mean "down".
    pub fn note_timeout(&self) -> Result<u32> {
        let connection = self.connection.lock().unwrap();
        let previous: Option<String> = connection
            .query_row(
                "SELECT value FROM sync_settings WHERE key='endpoint_timeout_streak'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let streak = previous
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0)
            .saturating_add(1);
        connection.execute(
            "INSERT INTO sync_settings(key,value) VALUES('endpoint_timeout_streak',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [streak.to_string()],
        )?;
        Ok(streak)
    }

    /// Remember why the breaker last tripped, for `/sync status` and doctor.
    pub fn note_endpoint_failure(&self, detail: &str) -> Result<()> {
        let detail: String = detail.chars().take(240).collect();
        self.connection.lock().unwrap().execute(
            "INSERT INTO sync_settings(key,value) VALUES('endpoint_last_failure',?1)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            [detail],
        )?;
        Ok(())
    }

    /// Sessions other than `current` that still have records to send:
    /// ready rows, or quarantined rows that have not been retried to death.
    /// A one-shot that exited before its flush (breaker open, interrupted)
    /// leaves exactly this behind, and only the session's own process ever
    /// drained it — so those sessions sat in the console with zero records.
    pub fn sessions_with_pending_records(
        &self,
        current: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let connection = self.connection.lock().unwrap();
        let mut statement = connection.prepare(
            "SELECT session_id FROM record_outbox
             WHERE session_id<>?1
               AND ((quarantined=0 AND next_retry_unix<=?2) OR (quarantined=1 AND attempts<8))
             GROUP BY session_id
             ORDER BY MIN(id)
             LIMIT ?3",
        )?;
        let rows = statement.query_map(params![current, now(), limit as i64], |row| {
            row.get::<_, String>(0)
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
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
        // Immediate (here and below): take the write lock at BEGIN, where
        // busy_timeout applies. A deferred BEGIN that reads before writing
        // can hit SQLITE_BUSY on lock upgrade with no timeout at all.
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
        // Identify the file canonically: session ids derive from the file
        // stem, so `--session-file session.json` in two directories collides
        // on session_id — the offset must not carry across distinct files
        // that merely share a relative path string.
        let path = Self::canonical_jsonl_identity(path);
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

    fn canonical_jsonl_identity(path: &std::path::Path) -> std::path::PathBuf {
        if let Ok(canonical) = std::fs::canonicalize(path) {
            return canonical;
        }
        // The file may not exist yet (tracking starts before the first
        // append). Canonicalize the parent so the identity is the same
        // before and after creation — otherwise the first real reconcile
        // would look like a path change and reset the offset to EOF,
        // silently skipping the session's first records.
        if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
            let parent = if parent.as_os_str().is_empty() {
                std::path::Path::new(".")
            } else {
                parent
            };
            if let Ok(parent) = std::fs::canonicalize(parent) {
                return parent.join(name);
            }
        }
        path.to_path_buf()
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
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
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

    /// Give quarantined records another chance. Called once per process (per
    /// sink) before draining: a fixed or upgraded server may now accept what
    /// an older one rejected, and earlier releases quarantined whole batches
    /// for a single bad record. Truly poisonous records simply re-quarantine
    /// after one attempt.
    pub fn requeue_quarantined(&self, session_id: &str) -> Result<usize> {
        let connection = self.connection.lock().unwrap();
        let requeued = connection.execute(
            "UPDATE record_outbox SET quarantined=0, next_retry_unix=0
             WHERE session_id=?1 AND quarantined=1",
            [session_id],
        )?;
        Ok(requeued)
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
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
        let mut status = status_from_connection(&connection, session_id)?;
        status.mode = mode;
        Ok(status)
    }
}

fn status_from_connection(connection: &Connection, session_id: Option<&str>) -> Result<SyncStatus> {
    let mode = connection
        .query_row(
            "SELECT value FROM sync_settings WHERE key='mode'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .as_deref()
        .map(SyncMode::parse)
        .unwrap_or_default();
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
    let metadata_row = |row: &rusqlite::Row<'_>| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, Option<i64>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, i64>(5)?,
            row.get::<_, i64>(6)?,
        ))
    };
    const METADATA_COLUMNS: &str = "server_cursor,last_success_unix,last_error,lease_generation,lease_owner,lease_expiry_unix,event_drops";
    let metadata = match session_id {
        Some(id) => connection
            .query_row(
                &format!("SELECT {METADATA_COLUMNS} FROM session_sync WHERE session_id=?1"),
                [id],
                metadata_row,
            )
            .optional()
            .ok()
            .flatten(),
        // No session named (doctor's overall check): report the most recently
        // active session — a live lease first, then the latest success — so
        // lease and error state stay observable instead of defaulting to
        // "no lease, no error" and hiding real failures.
        None => connection
            .query_row(
                &format!(
                    "SELECT {METADATA_COLUMNS} FROM session_sync \
                     ORDER BY MAX(COALESCE(last_success_unix,0), lease_expiry_unix) DESC LIMIT 1"
                ),
                [],
                metadata_row,
            )
            .optional()
            .ok()
            .flatten(),
    }
    .unwrap_or_default();
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

#[cfg(test)]
#[path = "sync_store_tests.rs"]
mod tests;
