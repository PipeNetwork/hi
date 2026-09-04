//! Remote session sync: pushes hi session records to an ipop API endpoint so
//! the session can be viewed (and later resumed) from another machine.
//!
//! Phase 1 is sync-only: the local `hi` process still owns the agent and the
//! filesystem. This module provides a [`RemoteSessionSink`] that mirrors the
//! JSONL records to ipop alongside the local file. The sink is best-effort —
//! if the network is down, the local session continues uninterrupted and the
//! failed records are queued for the next flush.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Lock a `std::sync::Mutex`, recovering the guard if a panic poisoned it. A
/// poisoned lock here means a producer panicked mid-update; the sync layer is
/// best-effort and must not take the whole session down over it.
fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

use anyhow::{Context, Result, anyhow};
use hi_agent::SessionSink;
use hi_ai::{Message, Role, Usage};
use serde::Deserialize;

/// Session IDs are used in URL paths and local token filenames. Keep them to
/// one safe path segment so a caller cannot redirect either operation.
pub fn validate_session_id(id: &str) -> Result<()> {
    if id.is_empty()
        || id.len() > 128
        || matches!(id, "." | "..")
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        anyhow::bail!("invalid session id: use 1-128 ASCII letters, digits, '.', '_' or '-'");
    }
    Ok(())
}

/// The record types that hi writes to a session JSONL file. Each variant
/// matches one `SessionMeta` tag, plus `message` for a bare `Message` line.
/// The server uses this to discriminate records without parsing the payload.
const RECORD_TYPE_MESSAGE: &str = "message";
const RECORD_TYPE_USAGE: &str = "usage";
const RECORD_TYPE_CHECKPOINTS: &str = "checkpoints";
const RECORD_TYPE_STATE_REPLACEMENT: &str = "state_replacement";
const RECORD_TYPE_PLAN_DRIVE: &str = "plan_drive";
const RECORD_TYPE_PLAN_APPROVAL: &str = "plan_approval";
const RECORD_TYPE_GOAL_DRIVE: &str = "goal_drive";
const MAX_RECORD_WIRE_BYTES: usize = 5_000_000;
// Leave room for JSON escaping and chunk metadata so each encoded chunk_part
// remains below the 1 MiB wire contract.
const CHUNK_PART_BYTES: usize = 450 * 1024;
const LEASE_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const LEASE_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(250);
const LEASE_MAX_ATTEMPTS: usize = 2;

/// Configuration for syncing a session to ipop.
#[derive(Clone, Debug)]
pub struct SyncConfig {
    /// The ipop API base URL, e.g. `https://api.pipenetwork.ai/v1`.
    pub base_url: String,
    /// The project API key for authentication.
    pub api_key: String,
    /// A stable identifier for this machine (so a remote viewer knows where
    /// the coding work runs). If `None`, the server omits it.
    pub machine_id: Option<String>,
    /// The hi cwd digest (16 hex chars) — groups sessions by project.
    pub cwd_digest: Option<String>,
}

/// A [`SessionSink`] that mirrors session records to an ipop API endpoint.
///
/// Records are buffered in memory and flushed in batches. If a flush fails,
/// the records stay buffered and are retried on the next flush. This keeps
/// the local session uninterrupted — sync is best-effort, never blocking.
///
/// The sink is **not** responsible for local file persistence; that stays
/// with [`crate::session::JsonlSession`]. Use [`SyncSession`] to multiplex
/// both.
/// What the host reports to the control plane on heartbeat, so a remote
/// viewer can show the model and context spend without guessing. `None`
/// fields are omitted server-side ("`Some` updates, `None` leaves").
#[derive(Clone, Default)]
pub struct HeartbeatTelemetry {
    pub model: Option<String>,
    pub context_used_tokens: Option<u64>,
    pub context_max_tokens: Option<u64>,
}

pub struct RemoteSessionSink {
    config: SyncConfig,
    session_id: String,
    /// The HTTP client. Reused across flushes for connection pooling.
    client: reqwest::Client,
    /// Buffered records waiting for the next flush. Protected by a mutex so
    /// the flush task can run concurrently with record() calls (though in
    /// practice the agent is single-threaded for turn execution).
    store: std::sync::Arc<crate::sync_store::SyncStore>,
    /// Whether the session has been registered with ipop yet.
    registered: Mutex<bool>,
    /// The per-session input token returned by ipop at registration.
    input_token: Mutex<Option<String>>,
    lease_lost: std::sync::Arc<AtomicBool>,
    heartbeat_started: std::sync::Arc<AtomicBool>,
    /// Shared with the heartbeat task; written by the agent via the session
    /// sink, read on every heartbeat tick.
    telemetry: std::sync::Arc<std::sync::Mutex<HeartbeatTelemetry>>,
    /// Whether this process collects remote prompts for the session (`--daemon`). Advertised at
    /// registration so a remote viewer can tell a steerable session from one that merely mirrors
    /// its transcript — input sent to the latter would queue with nobody polling for it.
    accepts_input: AtomicBool,
    /// Display title discovered from a custom name or first user message.
    title: Mutex<Option<String>>,
    /// Last title confirmed by the server, used to avoid redundant renames.
    registered_title: Mutex<Option<String>>,
    /// Serializes flushes. Waiting (rather than skipping a concurrent flush)
    /// is important at shutdown, when there may not be another retry.
    flush_lock: tokio::sync::Mutex<()>,
    /// Optional handoff barrier used during an in-process session switch. The
    /// replacement waits to register until the previous session has flushed
    /// and ended, but the interactive UI does not wait for that network work.
    activation: tokio::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    next_record_id: AtomicU64,
    /// Poison isolation: after a permanent rejection of a multi-record batch,
    /// flush one record at a time so only the actual offender is quarantined
    /// — batch validation otherwise strands every record queued with it.
    flush_singly: AtomicBool,
    /// Quarantined records get exactly one more chance per process: earlier
    /// releases quarantined whole batches for one bad record (and old servers
    /// rejected record kinds they had not learned yet), so a fixed server
    /// deserves a retry — while a truly poisonous record re-quarantines after
    /// a single attempt instead of thrashing on every flush.
    requeued_quarantined: AtomicBool,
    /// Whether registration may take the writer lease from another holder.
    /// The session's own process does (it is the writer); a stranded-record
    /// drain must not, or it would steal the lease from a live hi in another
    /// terminal that merely has not flushed yet.
    lease_takeover: AtomicBool,
    /// PipeFS turns transcript synchronization into a durability dependency:
    /// while it is set, this one live sink keeps queuing and transporting even
    /// if another process persists `/sync off` in the shared SQLite store.
    /// This is deliberately session-local rather than a SyncStore setting so
    /// disabling PipeFS restores the user's normal global sync preference.
    pipefs_sync_required: std::sync::Arc<AtomicBool>,
}

impl RemoteSessionSink {
    pub fn new(config: SyncConfig, session_id: String) -> Self {
        let cfg = config.clone();
        let sid = session_id.clone();
        Self::with_activation(config, session_id, None).unwrap_or_else(|err| {
            eprintln!("warning: portal sync unavailable ({err:#}); continuing without sync");
            Self::with_store(
                cfg,
                sid,
                None,
                remote_session_http_client(),
                std::sync::Arc::new(crate::sync_store::SyncStore::in_memory()),
            )
        })
    }

    #[cfg(test)]
    pub fn new_for_test(config: SyncConfig, session_id: String) -> Self {
        Self::with_store(
            config,
            session_id,
            None,
            remote_session_http_client(),
            unique_test_sync_store(),
        )
    }

    #[cfg(test)]
    pub fn new_after_drain(
        config: SyncConfig,
        session_id: String,
        activation: tokio::sync::oneshot::Receiver<()>,
    ) -> Self {
        Self::with_store(
            config,
            session_id,
            Some(activation),
            remote_session_http_client(),
            unique_test_sync_store(),
        )
    }

    fn with_activation(
        config: SyncConfig,
        session_id: String,
        activation: Option<tokio::sync::oneshot::Receiver<()>>,
    ) -> Result<Self> {
        let client = remote_session_http_client();
        let store = std::sync::Arc::new(
            crate::sync_store::SyncStore::open().context("opening durable portal sync database")?,
        );
        Ok(Self::with_store(
            config, session_id, activation, client, store,
        ))
    }

    fn with_store(
        config: SyncConfig,
        session_id: String,
        activation: Option<tokio::sync::oneshot::Receiver<()>>,
        client: reqwest::Client,
        store: std::sync::Arc<crate::sync_store::SyncStore>,
    ) -> Self {
        Self {
            config,
            session_id,
            client,
            store,
            registered: Mutex::new(false),
            input_token: Mutex::new(None),
            lease_lost: std::sync::Arc::new(AtomicBool::new(false)),
            heartbeat_started: std::sync::Arc::new(AtomicBool::new(false)),
            telemetry: std::sync::Arc::new(std::sync::Mutex::new(HeartbeatTelemetry::default())),
            accepts_input: AtomicBool::new(false),
            title: Mutex::new(None),
            registered_title: Mutex::new(None),
            flush_lock: tokio::sync::Mutex::new(()),
            activation: tokio::sync::Mutex::new(activation),
            next_record_id: AtomicU64::new(0),
            flush_singly: AtomicBool::new(false),
            requeued_quarantined: AtomicBool::new(false),
            lease_takeover: AtomicBool::new(true),
            pipefs_sync_required: std::sync::Arc::new(AtomicBool::new(false)),
        }
    }

    /// Pin this sink's transport while PipeFS is restoring or active.  The
    /// caller must clear it only after the remote workspace has been durably
    /// disabled (or its cache is retained on a failed disable).
    pub fn set_pipefs_sync_required(&self, required: bool) {
        self.pipefs_sync_required.store(required, Ordering::Release);
    }

    pub fn pipefs_sync_required(&self) -> bool {
        self.pipefs_sync_required.load(Ordering::Acquire)
    }

    fn transport_enabled(&self) -> Result<bool> {
        Ok(self.pipefs_sync_required()
            || self.store.effective_mode()? == crate::sync_store::SyncMode::On)
    }

    /// Push a record to the pending buffer. `&self` because it uses interior
    /// mutability — this lets `SyncSession` call it via an `Arc` handle.
    pub fn push(&self, record_type: &str, payload_json: &str) {
        if self.lease_lost.load(Ordering::Acquire) {
            return;
        }
        let wire_bytes = serde_json::to_string(payload_json)
            .map(|wire| wire.len())
            .unwrap_or(usize::MAX);
        if wire_bytes <= MAX_RECORD_WIRE_BYTES {
            let _ = self.store.enqueue_record_with_sync_pin(
                &self.session_id,
                record_type,
                payload_json,
                self.pipefs_sync_required(),
            );
            return;
        }

        // Oversized logical records are never omitted. Parts remain valid JSON
        // and are followed by a hash-bearing commit; readers apply only a
        // complete, verified set. The chunked write is all-or-nothing: if any
        // part fails to enqueue, the commit is never emitted, so durable
        // history can never reference missing parts.
        if let Err(error) = self.push_chunked(record_type, payload_json) {
            eprintln!(
                "\x1b[33msync: failed to enqueue chunked record; no chunk_commit was written: {error:#}\x1b[0m"
            );
        }
    }

    /// Enqueue an oversized record as chunk_part records followed by a
    /// hash-bearing chunk_commit. All-or-nothing: returns an error before
    /// writing the commit if any part fails to enqueue.
    fn push_chunked(&self, record_type: &str, payload_json: &str) -> Result<()> {
        use sha2::{Digest, Sha256};
        let nonce = self.next_record_id.fetch_add(1, Ordering::Relaxed);
        let logical_id = format!(
            "{:x}",
            Sha256::digest(format!(
                "{}\0{}\0{}\0{}",
                self.session_id, record_type, nonce, payload_json
            ))
        );
        let mut parts = Vec::new();
        let mut start = 0;
        while start < payload_json.len() {
            let mut end = (start + CHUNK_PART_BYTES).min(payload_json.len());
            while !payload_json.is_char_boundary(end) {
                end -= 1;
            }
            parts.push(&payload_json[start..end]);
            start = end;
        }
        // All-or-nothing: if any chunk_part fails to enqueue, never emit the
        // chunk_commit. A commit referencing missing parts would make every
        // future resume hard-fail with "chunk_commit is incomplete".
        for (index, data) in parts.iter().enumerate() {
            let part = serde_json::json!({
                "logical_id": logical_id,
                "index": index,
                "parts": parts.len(),
                "data": data,
            });
            self.store.enqueue_record_with_sync_pin(
                &self.session_id,
                "chunk_part",
                &part.to_string(),
                self.pipefs_sync_required(),
            )?;
        }
        let commit = serde_json::json!({
            "logical_id": logical_id,
            "record_type": record_type,
            "parts": parts.len(),
            "sha256": format!("{:x}", Sha256::digest(payload_json.as_bytes())),
            "bytes": payload_json.len(),
        });
        self.store.enqueue_record_with_sync_pin(
            &self.session_id,
            "chunk_commit",
            &commit.to_string(),
            self.pipefs_sync_required(),
        )?;
        Ok(())
    }

    /// Reconcile complete JSONL lines after the last committed local offset.
    /// The offset-derived ids make replay deterministic across crashes, and
    /// `INSERT OR IGNORE` on the record id makes replay idempotent — so a
    /// suspect offset can always be reset to 0 rather than trusted.
    pub fn reconcile_jsonl(&self, path: &std::path::Path) -> Result<()> {
        use sha2::{Digest, Sha256};
        use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
        let mut offset = self.store.track_jsonl(&self.session_id, path)?;
        let mut file = match std::fs::File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        // The tracked offset can go stale: `--session-file` session ids come
        // from the file stem, so distinct sessions ("session.json" in two
        // directories, or a recreated file) share one offset row. A stale
        // offset points past EOF or mid-record; reading from it used to fail
        // every reconcile forever — and, via the session sink, poison whole
        // turns as infrastructure errors. Validate and reset instead.
        let len = file.metadata()?.len();
        if offset > len || !Self::offset_on_record_boundary(&mut file, offset)? {
            offset = 0;
            self.store.set_jsonl_offset(&self.session_id, offset)?;
        }
        file.seek(SeekFrom::Start(offset))?;
        // Read only the file length observed above: a concurrently appending
        // session must not make reconciliation chase a moving EOF forever.
        // Retain at most one JSONL record instead of the entire unsynced tail.
        let mut reader = BufReader::new(file.take(len.saturating_sub(offset)));
        let mut line = Vec::new();
        let mut consumed_any = false;
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            if !line.ends_with(b"\n") {
                // Leave an incomplete crash/concurrent-write tail for the next
                // reconciliation rather than publishing a partial record.
                break;
            }
            let payload = std::str::from_utf8(&line[..line.len() - 1])?;
            if payload.is_empty() {
                consumed_any = true;
                offset = offset.saturating_add(line.len() as u64);
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(payload)
                .with_context(|| format!("invalid JSONL record at byte {offset}"))?;
            let record_type = value
                .get("type")
                .and_then(|kind| kind.as_str())
                .unwrap_or(RECORD_TYPE_MESSAGE);
            if record_type != "name" {
                let base_id = format!(
                    "{:x}",
                    Sha256::digest(
                        format!("{}\0{}\0{}", self.session_id, path.display(), offset).as_bytes()
                    )
                );
                self.enqueue_reconciled(&base_id, record_type, payload)?;
            }
            consumed_any = true;
            offset = offset.saturating_add(line.len() as u64);
        }
        if consumed_any {
            self.store.set_jsonl_offset(&self.session_id, offset)?;
        }
        Ok(())
    }

    fn enqueue_reconciled(&self, base_id: &str, record_type: &str, payload: &str) -> Result<()> {
        use sha2::{Digest, Sha256};
        if serde_json::to_string(payload)?.len() <= MAX_RECORD_WIRE_BYTES {
            return self.store.enqueue_record_with_id_and_sync_pin(
                &self.session_id,
                base_id,
                record_type,
                payload,
                self.pipefs_sync_required(),
            );
        }
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < payload.len() {
            let mut end = (start + CHUNK_PART_BYTES).min(payload.len());
            while !payload.is_char_boundary(end) {
                end -= 1;
            }
            chunks.push(&payload[start..end]);
            start = end;
        }
        for (index, data) in chunks.iter().enumerate() {
            let part = serde_json::json!({
                "logical_id": base_id, "index": index, "parts": chunks.len(), "data": data,
            });
            self.store.enqueue_record_with_id_and_sync_pin(
                &self.session_id,
                &format!("{base_id}.p{index}"),
                "chunk_part",
                &part.to_string(),
                self.pipefs_sync_required(),
            )?;
        }
        let commit = serde_json::json!({
            "logical_id": base_id, "record_type": record_type, "parts": chunks.len(),
            "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())), "bytes": payload.len(),
        });
        self.store.enqueue_record_with_id_and_sync_pin(
            &self.session_id,
            &format!("{base_id}.commit"),
            "chunk_commit",
            &commit.to_string(),
            self.pipefs_sync_required(),
        )
    }

    fn set_title(&self, title: Option<String>) {
        let title = title
            .map(|title| title.trim().to_string())
            .filter(|title| !title.is_empty());
        if title.is_some() {
            *lock_recover(&self.title) = title;
        }
    }

    /// True when `offset` is 0 or immediately follows a `\n` in this file —
    /// i.e. sits on a JSONL record boundary.
    fn offset_on_record_boundary(file: &mut std::fs::File, offset: u64) -> Result<bool> {
        use std::io::{Read, Seek, SeekFrom};
        if offset == 0 {
            return Ok(true);
        }
        file.seek(SeekFrom::Start(offset - 1))?;
        let mut byte = [0u8; 1];
        let read = file.read(&mut byte)?;
        Ok(read == 1 && byte[0] == b'\n')
    }

    /// Update the desired portal title. If the immediate rename request fails,
    /// the next record flush retries it before sending more records.
    pub fn update_title(&self, title: &str) {
        self.set_title(Some(title.to_string()));
    }

    fn observe_messages(&self, messages: &[Message]) {
        if lock_recover(&self.title).is_some() {
            return;
        }
        let title = messages.iter().find_map(|message| {
            if message.role != Role::User {
                return None;
            }
            let title = hi_agent::ui::user_prompt_title(&message.text(), 72);
            (!title.is_empty()).then_some(title)
        });
        self.set_title(title);
    }

    /// Queue one authoritative state snapshot when adopting a session. This
    /// backfills its existing history instead of syncing only future turns.
    pub fn seed_snapshot(&self, loaded: &crate::session::LoadedSession) -> Result<()> {
        self.set_title(loaded.name.clone());
        self.observe_messages(&loaded.messages);
        let payload = serde_json::to_string(&serde_json::json!({
            "type": "state_replacement",
            "messages": loaded.messages,
            "goal": loaded.goal,
            "decisions": loaded.decisions.entries(),
            "plan": loaded.plan,
        }))?;
        self.push(RECORD_TYPE_STATE_REPLACEMENT, &payload);
        if !loaded.usage.is_zero() {
            self.push(
                RECORD_TYPE_USAGE,
                &serde_json::to_string(&serde_json::json!({
                    "type": "usage",
                    "input_tokens": loaded.usage.input_tokens,
                    "output_tokens": loaded.usage.output_tokens,
                    "cache_read_tokens": loaded.usage.cache_read_tokens,
                    "cache_creation_tokens": loaded.usage.cache_creation_tokens,
                    "estimated": loaded.usage.estimated,
                }))?,
            );
        }
        if !loaded.checkpoint_refs.is_empty() {
            self.push(
                RECORD_TYPE_CHECKPOINTS,
                &serde_json::to_string(&serde_json::json!({
                    "type": "checkpoints",
                    "refs": loaded.checkpoint_refs,
                }))?,
            );
        }
        // These are last-write-wins records, so seed even the running/default
        // values. The destination session may already contain an older pause,
        // park, stall, or evidence ledger that this snapshot must clear.
        self.push(
            RECORD_TYPE_PLAN_DRIVE,
            &serde_json::to_string(&serde_json::json!({
                "type": "plan_drive",
                "paused": loaded.plan_drive_paused,
                "resume_on_user_input": loaded.plan_drive_resume_on_user_input,
                "stall": loaded.plan_drive_stall,
                "evidence_reset": true,
                "evidence_add": loaded.plan_drive_evidence,
            }))?,
        );
        self.push(
            RECORD_TYPE_PLAN_APPROVAL,
            &serde_json::to_string(&serde_json::json!({
                "type": "plan_approval",
                "parked": loaded.plan_approval_parked,
            }))?,
        );
        self.push(
            RECORD_TYPE_GOAL_DRIVE,
            &serde_json::to_string(&serde_json::json!({
                "type": "goal_drive",
                "stall": loaded.goal_drive_stall,
                "evidence_reset": true,
                "evidence_add": loaded.goal_drive_evidence,
            }))?,
        );
        Ok(())
    }

    /// The per-session input token, if the server returned one at registration.
    /// Used by the daemon to write a local token file so `hi --attach` on the
    /// same machine can submit inputs.
    pub fn input_token(&self) -> Option<String> {
        lock_recover(&self.input_token).clone()
    }

    /// Force registration now (normally deferred to the first flush). The
    /// daemon calls this at startup so the input token is available
    /// immediately.
    ///
    /// Retries with backoff rather than failing on the first error. Registration
    /// is a write, and control-plane writes can stall well past this client's
    /// request timeout under load. Giving up immediately takes the whole daemon
    /// down over a transient stall — and because the next start derives a fresh
    /// session id from a fresh session file, every such death also strands an
    /// empty session in the user's catalog.
    pub async fn ensure_registered_now(&self) -> Result<()> {
        self.ensure_registered_now_with_announcements(true).await
    }

    /// [`ensure_registered_now`](Self::ensure_registered_now) with retry
    /// chatter suppressed — for best-effort callers (session switches) where
    /// portal trouble must stay invisible to the user. The daemon keeps
    /// announcing, since it cannot function without registration.
    pub async fn ensure_registered_now_quiet(&self) -> Result<()> {
        self.ensure_registered_now_with_announcements(false).await
    }

    async fn ensure_registered_now_with_announcements(&self, announce: bool) -> Result<()> {
        const MAX_ATTEMPTS: u32 = 5;
        const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(30);

        let mut backoff = std::time::Duration::from_secs(2);
        let mut last_error = None;
        for attempt in 1..=MAX_ATTEMPTS {
            // The retry ladder is for transient control-plane stalls. A
            // tripped circuit breaker means the endpoint is not reachable at
            // all (connect/timeout failures) — burning the full ladder adds
            // up to a minute of startup hang for the same outcome, so fail
            // fast and let the next flush retry after the cooldown.
            if attempt > 1 && sync_breaker_open(&self.store) {
                return Err(last_error.unwrap_or_else(|| {
                    anyhow!("sync endpoint unreachable (circuit breaker open)")
                }));
            }
            match self.ensure_registered().await {
                Ok(()) => {
                    if announce && attempt > 1 {
                        eprintln!(
                            "\x1b[33mdaemon: session registered after {attempt} attempts\x1b[0m"
                        );
                    }
                    return Ok(());
                }
                Err(error) => {
                    if attempt < MAX_ATTEMPTS {
                        if announce {
                            eprintln!(
                                "\x1b[33mdaemon: registration attempt {attempt}/{MAX_ATTEMPTS} failed ({error:#}); retrying in {}s\x1b[0m",
                                backoff.as_secs()
                            );
                        }
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                    }
                    last_error = Some(error);
                }
            }
        }
        Err(last_error
            .unwrap_or_else(|| anyhow!("session registration failed with no reported error")))
    }

    /// Register the session with ipop if not already done. Called before the
    /// first flush. A failed registration is retried on the next flush; marking
    /// it successful after a network error permanently strands the session.
    async fn ensure_registered(&self) -> Result<()> {
        if !self.transport_enabled()? {
            return Ok(());
        }
        // Endpoint cooling down after connect failures: fail fast instead of
        // paying another timeout. Every caller treats registration as
        // retryable-later (flushes) or best-effort (host mode, switches).
        if sync_breaker_open(&self.store) {
            anyhow::bail!(
                "sync endpoint cooling down after recent connect failures; retrying later"
            );
        }
        if self.lease_lost.load(Ordering::Acquire) {
            anyhow::bail!("lease_lost: select another session before accepting new turns");
        }
        let activation = self.activation.lock().await.take();
        if let Some(activation) = activation {
            // A dropped sender means the predecessor task was aborted; allow
            // this session to proceed rather than deadlocking sync forever.
            let _ = activation.await;
        }
        if *lock_recover(&self.registered) {
            return self.sync_title().await;
        }
        let url = format!("{}/hi/sessions", self.config.base_url);
        let title = lock_recover(&self.title).clone();
        let body = serde_json::json!({
            "session_id": self.session_id,
            "machine_id": self.config.machine_id,
            // The id above is a stable random identity; the label is what a
            // person reads in a remote session list instead of hex.
            "machine_label": crate::tickets::hostname(),
            "cwd_digest": self.config.cwd_digest,
            "project_fingerprint": crate::session::project_fingerprint(),
            "title": title,
            "accepts_input": self.accepts_input.load(Ordering::Acquire),
        });
        // Registration is an idempotent metadata write, and on the control
        // plane it queues behind the aggregate persist lane — so it is the
        // request that intermittently outlives the client's 10s budget even
        // when the server committed it. It gets its own budget; a miss is
        // retried by the next flush (a one-shot's exit flush included), so
        // no in-call retry is needed to keep a turn end bounded.
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .json(&body)
            .timeout(REGISTER_REQUEST_TIMEOUT)
            .send()
            .await;
        note_endpoint_outcome(&self.store, response.as_ref().err());
        let response = response.with_context(|| format!("registering session at {url}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("ipop session registration failed: {status} {body}"));
        }
        if let Ok(json) = response.json::<serde_json::Value>().await
            && let Some(token) = json.get("input_token").and_then(|v| v.as_str())
        {
            *lock_recover(&self.input_token) = Some(token.to_string());
        }
        *lock_recover(&self.registered_title) = title;
        self.acquire_lease(self.lease_takeover.load(Ordering::Acquire))
            .await?;
        self.start_lease_heartbeat();
        *lock_recover(&self.registered) = true;
        Ok(())
    }

    /// Upload records that earlier processes left behind for *other*
    /// sessions — a one-shot that exited while the breaker was open, or was
    /// interrupted before its flush — and mark those sessions ended. Without
    /// this, only a session's own process ever drained its outbox, so the
    /// console filled with sessions that registered and never got a record.
    ///
    /// Bounded and polite: a handful of sessions per run, nothing while the
    /// breaker is open, and never a lease takeover — a session whose owner is
    /// alive in another terminal refuses the lease and is skipped.
    pub async fn drain_stranded_sessions(&self) -> usize {
        const MAX_SESSIONS_PER_RUN: usize = 8;
        if self.store.effective_mode().ok() != Some(crate::sync_store::SyncMode::On) {
            return 0;
        }
        let Ok(stranded) = self
            .store
            .sessions_with_pending_records(&self.session_id, MAX_SESSIONS_PER_RUN)
        else {
            return 0;
        };
        let mut drained = 0;
        for session_id in stranded {
            if sync_breaker_open(&self.store) {
                break;
            }
            let sink = Self::with_store(
                self.config.clone(),
                session_id,
                None,
                self.client.clone(),
                self.store.clone(),
            );
            sink.lease_takeover.store(false, Ordering::Release);
            if sink.flush().await.is_err() {
                continue;
            }
            // `flush` also returns Ok when it skipped the network (breaker
            // open, mode flipped). Only an empty outbox proves the records
            // went; ending the session on anything less marks it finished
            // server-side with its records still sitting here.
            let uploaded = sink
                .store
                .ready_records(&sink.session_id, 1)
                .map(|left| left.is_empty())
                .unwrap_or(false);
            if !uploaded {
                continue;
            }
            // Nothing more will come from a process that is gone; leave the
            // session in its truthful terminal state rather than "active"
            // under a lease this drain just took.
            sink.end_session().await;
            drained += 1;
        }
        drained
    }

    async fn acquire_lease(&self, takeover: bool) -> Result<()> {
        self.acquire_lease_with_policy(
            takeover,
            LEASE_REQUEST_TIMEOUT,
            LEASE_RETRY_DELAY,
            LEASE_MAX_ATTEMPTS,
        )
        .await
    }

    async fn acquire_lease_with_policy(
        &self,
        takeover: bool,
        request_timeout: std::time::Duration,
        retry_delay: std::time::Duration,
        max_attempts: usize,
    ) -> Result<()> {
        let url = format!(
            "{}/hi/sessions/{}/lease",
            self.config.base_url, self.session_id
        );
        let machine_id = self
            .config
            .machine_id
            .clone()
            .unwrap_or_else(|| "unknown-machine".to_string());
        let client_instance_id = format!("{}-{}", machine_id, std::process::id());
        // A stable client-generated token makes a retry safe when the API committed the lease but
        // its response was lost. The patched server stores only its hash; older servers ignore it,
        // so the API side of this contract must roll out before this client.
        let requested_lease_token = format!("hl_{}", uuid::Uuid::new_v4().simple());
        let body = serde_json::json!({
            "client_instance_id": &client_instance_id,
            "machine_id": &machine_id,
            "takeover": takeover,
            "lease_token": &requested_lease_token,
        });
        let max_attempts = max_attempts.max(1);
        let mut attempt = 0;
        let response = loop {
            attempt += 1;
            match self
                .client
                .post(&url)
                .header("x-api-key", &self.config.api_key)
                .json(&body)
                .timeout(request_timeout)
                .send()
                .await
            {
                Ok(response) => break response,
                Err(error)
                    if attempt < max_attempts && (error.is_timeout() || error.is_connect()) =>
                {
                    tokio::time::sleep(retry_delay).await;
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("acquiring session lease at {url}"));
                }
            }
        };
        if matches!(
            response.status(),
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::METHOD_NOT_ALLOWED
        ) {
            // Client-first rollout: legacy servers remain usable until lease
            // enforcement is deployed.
            return Ok(());
        }
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("session lease failed: {status} {body}"));
        }
        let body: serde_json::Value = response.json().await.context("parsing session lease")?;
        let Some(token) = body.get("lease_token").and_then(|value| value.as_str()) else {
            // Some legacy test/proxy deployments answer unknown POST routes
            // with a generic success body. Absence of the capability field is
            // treated the same as a missing lease endpoint during rollout.
            return Ok(());
        };
        let generation = body
            .get("generation")
            .and_then(|value| value.as_u64())
            .unwrap_or_default();
        let expiry = body
            .get("expires_at_unix")
            .and_then(|value| value.as_u64())
            .unwrap_or_default();
        self.store.store_lease(
            &self.session_id,
            token,
            generation,
            &client_instance_id,
            expiry,
        )?;
        self.lease_lost.store(false, Ordering::Release);
        Ok(())
    }

    /// Declare that this process is polling for remote input. Must be called before the session
    /// registers, since the flag is sent in the registration body.
    pub fn set_accepts_input(&self, value: bool) {
        self.accepts_input.store(value, Ordering::Release);
    }

    /// Flip `accepts_input` and re-register so the control plane advertises the
    /// new capability (registration is otherwise a one-shot). Used by the TUI
    /// `/sessions host` path so a live session can start accepting remote
    /// prompts without restarting as `--daemon`.
    pub async fn publish_accepts_input(&self, value: bool) -> Result<()> {
        self.set_accepts_input(value);
        // Force a fresh POST /hi/sessions so `accepts_input` is updated server-side.
        *lock_recover(&self.registered) = false;
        self.ensure_registered().await
    }

    /// Writer lease token for authenticated long-polls (GET input / heartbeat).
    pub fn writer_lease_token(&self) -> Option<String> {
        self.lease_token()
    }

    /// Session identity shared by transcript sync and PipeFS. PipeFS checks
    /// this before using the lease so a live session switch cannot persist
    /// workspace bytes under the previously active transcript.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Generation paired with [`writer_lease_token`](Self::writer_lease_token).
    /// PipeFS uses the same writer identity rather than inventing a parallel
    /// session or lock namespace.
    pub fn writer_lease_generation(&self) -> u64 {
        self.store
            .status(Some(&self.session_id))
            .map(|status| status.lease_generation)
            .unwrap_or_default()
    }

    /// True once any lease-authenticated sync operation learns that another
    /// machine took over this session. PipeFS consults the same flag before
    /// admitting a native filesystem mutation.
    pub fn writer_lease_is_lost(&self) -> bool {
        self.lease_lost.load(Ordering::Acquire)
    }

    pub fn lease_token(&self) -> Option<String> {
        self.store.lease_token(&self.session_id).ok().flatten()
    }

    /// Record the model this session runs, for the heartbeat body. Called via
    /// the session sink whenever the agent attaches or switches models.
    pub fn set_model_context(&self, model: &str, context_window: Option<u32>) {
        let mut telemetry = lock_recover(&self.telemetry);
        let model = model.trim();
        telemetry.model = (!model.is_empty()).then(|| model.to_string());
        telemetry.context_max_tokens = context_window.map(u64::from);
    }

    /// Record the context spend of the latest request. `context_occupancy` is
    /// computed at the provider adapter, so it is already the right number.
    pub fn observe_context_used(&self, context_occupancy: u64) {
        if context_occupancy == 0 {
            return;
        }
        lock_recover(&self.telemetry).context_used_tokens = Some(context_occupancy);
    }

    fn start_lease_heartbeat(&self) {
        if self.lease_token().is_none() || self.heartbeat_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let client = self.client.clone();
        let url = format!(
            "{}/hi/sessions/{}/heartbeat",
            self.config.base_url, self.session_id
        );
        let api_key = self.config.api_key.clone();
        let session_id = self.session_id.clone();
        let store = self.store.clone();
        let pipefs_sync_required = self.pipefs_sync_required.clone();
        let heartbeat_started = self.heartbeat_started.clone();
        let lease_lost = self.lease_lost.clone();
        let telemetry = self.telemetry.clone();
        tokio::spawn(async move {
            struct Reset(std::sync::Arc<AtomicBool>);
            impl Drop for Reset {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Release);
                }
            }
            let _reset = Reset(heartbeat_started);
            let mut consecutive_failures = 0_u8;
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                if !pipefs_sync_required.load(Ordering::Acquire)
                    && store.effective_mode().ok() != Some(crate::sync_store::SyncMode::On)
                {
                    continue;
                }
                let Some(token) = store.lease_token(&session_id).ok().flatten() else {
                    break;
                };
                // The body carries host telemetry when known. `None` fields
                // serialise to null, which the server reads as "leave as is" —
                // identical to the empty body older builds send.
                let body = {
                    let telemetry = lock_recover(&telemetry).clone();
                    serde_json::json!({
                        "model": telemetry.model,
                        "context_used_tokens": telemetry.context_used_tokens,
                        "context_max_tokens": telemetry.context_max_tokens,
                    })
                };
                let response = client
                    .post(&url)
                    .header("x-api-key", &api_key)
                    .header("x-hi-lease-token", token)
                    .json(&body)
                    .send()
                    .await;
                match response {
                    Ok(response) if response.status() == reqwest::StatusCode::CONFLICT => {
                        lease_lost.store(true, Ordering::Release);
                        break;
                    }
                    Ok(response) if response.status().is_success() => consecutive_failures = 0,
                    Ok(_) | Err(_) => {
                        consecutive_failures = consecutive_failures.saturating_add(1);
                        if consecutive_failures >= 5 {
                            break;
                        }
                    }
                }
            }
        });
    }

    async fn sync_title(&self) -> Result<()> {
        let title = lock_recover(&self.title).clone();
        if title.is_none() || title == *lock_recover(&self.registered_title) {
            return Ok(());
        }
        let url = format!(
            "{}/hi/sessions/{}/rename",
            self.config.base_url, self.session_id
        );
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .json(&serde_json::json!({ "title": title }))
            .send()
            .await
            .with_context(|| format!("updating session title at {url}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("ipop session rename failed: {status} {body}"));
        }
        *lock_recover(&self.registered_title) = title;
        Ok(())
    }

    /// Flush all pending records to ipop. Called after each turn. Best-effort:
    /// on failure, records stay buffered for the next attempt.
    pub async fn flush(&self) -> Result<()> {
        self.flush_with_requirement(false).await
    }

    /// Flush every transcript record as a required PipeFS durability barrier.
    /// Unlike ordinary best-effort sync, this performs a real attempt through
    /// an open circuit breaker and errors while any delayed/quarantined row
    /// remains, so callers cannot delete a recovery cache prematurely.
    pub async fn flush_required(&self) -> Result<()> {
        if !self.pipefs_sync_required() {
            anyhow::bail!("PipeFS transcript durability is not pinned for this session");
        }
        self.flush_with_requirement(true).await
    }

    async fn flush_with_requirement(&self, require_complete: bool) -> Result<()> {
        let _flush = self.flush_lock.lock().await;
        if !self.transport_enabled()? {
            if require_complete {
                anyhow::bail!("transcript synchronization is disabled");
            }
            return Ok(());
        }
        // Endpoint cooling down after connect failures: skip silently —
        // records stay queued in the durable outbox for a later flush. This
        // keeps a dead portal from stacking timeouts onto startups, turn
        // ends, and exits (observed: 36s added to a sub-second one-shot).
        if sync_breaker_open(&self.store) {
            if !require_complete {
                return Ok(());
            }
            // A required flush is itself the explicit retry. Clear only the
            // local cooldown; the attempted request will trip it again if the
            // endpoint is still unreachable.
            self.store.reset_breaker()?;
        }
        if require_complete {
            let forced = self.store.force_retry_records(&self.session_id)?;
            if forced > 0 {
                self.flush_singly.store(true, Ordering::Release);
            }
            self.requeued_quarantined.store(true, Ordering::Release);
        }
        self.ensure_registered().await?;
        // The heartbeat task gives up after sustained failures (e.g. a portal
        // deploy window) and registration only starts it once per process;
        // re-arming here revives it on the next turn so the lease is not
        // silently forfeited for the rest of the session. Idempotent while a
        // heartbeat is already running.
        self.start_lease_heartbeat();
        if !self.requeued_quarantined.swap(true, Ordering::AcqRel) {
            let requeued = self.store.requeue_quarantined(&self.session_id)?;
            if requeued > 0 {
                self.flush_singly.store(true, Ordering::Release);
            }
        }
        loop {
            let limit = if self.flush_singly.load(Ordering::Acquire) {
                1
            } else {
                512
            };
            let mut records = self.store.ready_records(&self.session_id, limit)?;
            if records.is_empty() {
                // The queue drained clean; the next flush may batch again.
                self.flush_singly.store(false, Ordering::Release);
                if require_complete {
                    let status = self.store.status(Some(&self.session_id))?;
                    if status.queue_rows != 0 {
                        anyhow::bail!(
                            "transcript flush left {} buffered record(s), including {} quarantined; recovery cache retained",
                            status.queue_rows,
                            status.quarantined_records
                        );
                    }
                }
                return Ok(());
            }
            let mut bytes = 0usize;
            records.retain(|record| {
                let next = bytes.saturating_add(record.payload_json.len() + 256);
                if bytes > 0 && next > 5_500_000 {
                    false
                } else {
                    bytes = next;
                    true
                }
            });

            let url = format!(
                "{}/hi/sessions/{}/records",
                self.config.base_url, self.session_id
            );
            let append_request = serde_json::json!({
                "records": records.iter().map(|r| {
                        serde_json::json!({
                            "client_record_id": r.client_record_id,
                            "record_type": r.record_type,
                        "payload_json": r.payload_json,
                    })
                }).collect::<Vec<_>>(),
            });

            let mut request = self
                .client
                .post(&url)
                .header("x-api-key", &self.config.api_key)
                .json(&append_request);
            if let Some(token) = self.lease_token() {
                request = request.header("x-hi-lease-token", token);
            }
            let response = request.send().await;
            note_endpoint_outcome(&self.store, response.as_ref().err());
            let response = match response {
                Ok(response) => response,
                Err(err) => {
                    self.store.fail_records(
                        &self.session_id,
                        &records,
                        &err.to_string(),
                        None,
                        false,
                    )?;
                    return Err(err).with_context(|| format!("flushing session records to {url}"));
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok());
                let body = response.text().await.unwrap_or_default();
                if status == reqwest::StatusCode::CONFLICT && body.contains("lease_lost") {
                    self.lease_lost.store(true, Ordering::Release);
                }
                let permanent = status.is_client_error()
                    && !matches!(
                        status,
                        reqwest::StatusCode::REQUEST_TIMEOUT
                            | reqwest::StatusCode::CONFLICT
                            | reqwest::StatusCode::TOO_MANY_REQUESTS
                    );
                if permanent && records.len() > 1 {
                    // Batch validation rejects everything for one bad record.
                    // Leave the batch queued and go one-at-a-time: good
                    // records flow, only the offender gets quarantined below.
                    self.flush_singly.store(true, Ordering::Release);
                    continue;
                }
                self.store.fail_records(
                    &self.session_id,
                    &records,
                    &format!("HTTP {status}: {body}"),
                    retry_after,
                    permanent,
                )?;
                if permanent {
                    // The lone offender is quarantined; keep flushing what
                    // is queued behind it rather than abandoning the turn.
                    continue;
                }
                return Err(anyhow!("ipop sync flush failed: {status} {body}"));
            }
            let cursor = response
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|body| body.get("record_count").and_then(|value| value.as_u64()))
                .unwrap_or_default();
            self.store.acknowledge_records(
                &self.session_id,
                &records
                    .iter()
                    .map(|record| record.row_id)
                    .collect::<Vec<_>>(),
                cursor,
            )?;
        }
    }

    /// Mark the session as ended on ipop. Called when the hi process exits
    /// cleanly. Best-effort.
    pub async fn end_session(&self) {
        self.flush().await.ok();
        if sync_breaker_open(&self.store) {
            return;
        }
        let url = format!(
            "{}/hi/sessions/{}/end",
            self.config.base_url, self.session_id
        );
        let mut request = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .json(&serde_json::json!({}));
        if let Some(token) = self.lease_token() {
            request = request.header("x-hi-lease-token", token);
        }
        let result = request.send().await;
        note_endpoint_outcome(&self.store, result.as_ref().err());
    }
}

/// Builder for portal-sync HTTP clients. Every request carries `x-api-key`,
/// so redirects are pinned to the configured origin (same policy as the
/// agent LLM client). Callers add timeouts; [`hi_ai::timed_http_client_fallback`]
/// is the last-resort build path and uses the same policy.
fn sync_http_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .redirect(hi_ai::credential_redirect_policy())
        .http1_only()
}

fn remote_session_http_client() -> reqwest::Client {
    sync_http_builder()
        // A dead endpoint must cost seconds, not tens of seconds: the connect
        // phase gets its own short budget, and the first failure trips the
        // shared circuit breaker so later calls skip the network entirely.
        .connect_timeout(std::time::Duration::from_secs(3))
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(5, 10))
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Whether the shared endpoint circuit breaker is open (recent connect
/// failures; cooling down). Sync paths skip network work while open — queued
/// records/events simply wait for a later flush.
fn sync_breaker_open(store: &crate::sync_store::SyncStore) -> bool {
    matches!(store.breaker_open_until(), Ok(Some(until)) if until > unix_now())
}

/// Record the outcome of an endpoint round-trip attempt: connect/timeout
/// failures trip the shared breaker (doubling cooldown), anything that
/// reached the server resets it.
/// Registration's own request budget (see `ensure_registered`).
const REGISTER_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Consecutive request timeouts before the breaker treats the endpoint as
/// down. A refused connection trips it at once; a single slow write does not
/// — the control plane can take seconds under a persist, and one such
/// timeout used to blackhole the next minute of runs (records stranded, the
/// console empty) for a server that was merely busy.
const TIMEOUT_STREAK_TO_TRIP: u32 = 3;

fn note_endpoint_outcome(store: &crate::sync_store::SyncStore, error: Option<&reqwest::Error>) {
    match error {
        Some(err) if err.is_connect() => {
            let _ = store.note_endpoint_failure(&format!("connect: {err}"));
            let _ = store.trip_breaker(unix_now());
        }
        Some(err) if err.is_timeout() => {
            let _ = store.note_endpoint_failure(&format!("timeout: {err}"));
            if store.note_timeout().unwrap_or(0) >= TIMEOUT_STREAK_TO_TRIP {
                let _ = store.trip_breaker(unix_now());
            }
        }
        Some(_) => {}
        None => {
            let _ = store.reset_breaker();
        }
    }
}

/// A multiplexing [`SessionSink`] that writes to both a local JSONL file and
/// a remote ipop endpoint. The local write is synchronous (must succeed for
/// the turn to continue); the remote write is buffered and flushed
/// asynchronously after each turn.
///
/// The remote sink is wrapped in an `Arc` so a handle can be retained outside
/// the agent (which owns the sink as `Box<dyn SessionSink>`) for flushing
/// after each turn and ending the session on exit.
pub struct SyncSession {
    local: crate::session::JsonlSession,
    remote: std::sync::Arc<RemoteSessionSink>,
}

impl SyncSession {
    pub fn new(local: crate::session::JsonlSession, remote: RemoteSessionSink) -> Self {
        let session = Self {
            local,
            remote: std::sync::Arc::new(remote),
        };
        session.reconcile_best_effort();
        session
    }

    /// Get a handle to the remote sink for flushing / ending the session.
    /// Call this before boxing the `SyncSession` for the agent.
    pub fn remote_handle(&self) -> std::sync::Arc<RemoteSessionSink> {
        self.remote.clone()
    }

    /// Mirror new JSONL lines into the durable outbox, swallowing failures.
    /// The mirror is offset-tracked and idempotent, so an error (a peer
    /// holding the SQLite write lock past its busy_timeout, a broken store)
    /// only defers those lines to the next reconcile. The local JSONL is the
    /// session's source of truth and has already been written — failing the
    /// caller's turn over the mirror would surface an error the user can't
    /// act on.
    fn reconcile_best_effort(&self) {
        let _ = self.remote.reconcile_jsonl(self.local.path());
    }
}

impl SessionSink for SyncSession {
    fn id(&self) -> Option<String> {
        self.local.id()
    }

    fn record(&mut self, messages: &[Message], usage: Usage) -> Result<()> {
        self.local.record(messages, usage)?;
        self.remote.observe_messages(messages);
        self.remote.observe_context_used(usage.context_occupancy);
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_model_context(&mut self, model: &str, context_window: Option<u32>) {
        self.remote.set_model_context(model, context_window);
    }

    fn record_compaction(&mut self, messages: &[Message]) -> Result<()> {
        self.local.record_compaction(messages)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&hi_agent::Goal>,
        decisions: &hi_agent::DecisionLog,
        plan: &[hi_agent::PlanStep],
    ) -> Result<()> {
        self.local
            .record_state_replacement(messages, goal, decisions, plan)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        self.local.record_checkpoints(refs)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_goal(&mut self, goal: &hi_agent::Goal) -> Result<()> {
        self.local.record_goal(goal)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn clear_goal(&mut self) -> Result<()> {
        self.local.clear_goal()?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan(&mut self, plan: &[hi_agent::PlanStep]) -> Result<()> {
        self.local.record_plan(plan)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn clear_plan(&mut self) -> Result<()> {
        self.local.clear_plan()?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan_drive(&mut self, paused: bool, stall: u32) -> Result<()> {
        self.record_plan_drive_state(paused, stall, false, &[])
    }

    fn record_plan_drive_state(
        &mut self,
        paused: bool,
        stall: u32,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.local
            .record_plan_drive_state(paused, stall, evidence_reset, evidence_add)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        stall: u32,
        resume_on_user_input: bool,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.local.record_plan_drive_state_with_policy(
            paused,
            stall,
            resume_on_user_input,
            evidence_reset,
            evidence_add,
        )?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan_approval_parked(&mut self, parked: bool) -> Result<()> {
        self.local.record_plan_approval_parked(parked)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_goal_drive(&mut self, stall: u32) -> Result<()> {
        self.record_goal_drive_state(stall, false, &[])
    }

    fn record_goal_drive_state(
        &mut self,
        stall: u32,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.local
            .record_goal_drive_state(stall, evidence_reset, evidence_add)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_decisions(&mut self, decisions: &hi_agent::DecisionLog) -> Result<()> {
        self.local.record_decisions(decisions)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_turn_outcome(
        &mut self,
        outcome: &hi_agent::TurnOutcome,
        review_unavailable_reason: Option<&str>,
    ) -> Result<()> {
        self.local
            .record_turn_outcome(outcome, review_unavailable_reason)?;
        self.reconcile_best_effort();
        Ok(())
    }
}

// ─── Live event streaming (Phase 2) ─────────────────────────────────────────

/// A [`hi_agent::Ui`] that serializes each callback as a [`hi_tui::event::UiEvent`]
/// and buffers it for flushing to ipop's live event endpoint. The flush is
/// async (HTTP) so it can't happen inside the sync `Ui` methods — call
/// [`RemoteUi::flush`] after each turn (or mid-turn from a timer).
///
/// Best-effort: if the flush fails, events are retained for the next attempt.
/// The local UI is unaffected — sync never blocks the turn.
pub struct RemoteUi {
    config: SyncConfig,
    session_id: String,
    client: reqwest::Client,
    store: std::sync::Arc<crate::sync_store::SyncStore>,
    /// Serializes flushes to preserve ordering and make the final shutdown
    /// flush wait for any in-flight background flush.
    flush_lock: tokio::sync::Mutex<()>,
}

impl RemoteUi {
    /// Fallible: a disk/permission failure opening the store must surface to
    /// the caller (e.g. as a `/sessions switch` error line), not panic inside
    /// the alternate-screen TUI.
    pub fn new(config: SyncConfig, session_id: String) -> Result<Self> {
        let client = sync_http_builder()
            .connect_timeout(std::time::Duration::from_secs(3))
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(5, 10));
        let store = std::sync::Arc::new(
            crate::sync_store::SyncStore::open()
                .context("opening durable portal event database")?,
        );
        Ok(Self::with_store(config, session_id, client, store))
    }

    #[cfg(test)]
    pub fn new_for_test(config: SyncConfig, session_id: String) -> Self {
        let client = sync_http_builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(5, 10));
        Self::with_store(config, session_id, client, unique_test_sync_store())
    }

    fn with_store(
        config: SyncConfig,
        session_id: String,
        client: reqwest::Client,
        store: std::sync::Arc<crate::sync_store::SyncStore>,
    ) -> Self {
        Self {
            config,
            session_id,
            client,
            store,
            flush_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Serialize and buffer a UiEvent for the next flush. `&self` because it
    /// uses interior mutability — this lets `MultiplexUi` call it via an `Arc`.
    pub fn push_event(&self, event: hi_tui::event::UiEvent) {
        if let Ok(json) = serde_json::to_string(&event) {
            let json = if json.len() <= 256_000 {
                json
            } else {
                serde_json::to_string(&hi_tui::event::UiEvent::Status {
                    text: "(oversized live event omitted; durable session record is unchanged)"
                        .to_string(),
                })
                .unwrap_or_default()
            };
            let _ = self.store.enqueue_event(&self.session_id, &json);
        }
    }

    /// Flush all buffered events to ipop's live event endpoint. Best-effort:
    /// on failure, events stay buffered for retry.
    pub async fn flush(&self) -> Result<()> {
        let _flush = self.flush_lock.lock().await;
        if self.store.effective_mode()? != crate::sync_store::SyncMode::On {
            return Ok(());
        }
        // Endpoint cooling down: skip silently, events stay queued.
        if sync_breaker_open(&self.store) {
            return Ok(());
        }
        loop {
            let mut rows = self.store.ready_events(&self.session_id, 256)?;
            if rows.is_empty() {
                return Ok(());
            }
            let mut bytes = 0usize;
            rows.retain(|(_, event)| {
                let next = bytes.saturating_add(event.len());
                if bytes > 0 && next > 1_800_000 {
                    false
                } else {
                    bytes = next;
                    true
                }
            });

            let url = format!(
                "{}/hi/sessions/{}/events",
                self.config.base_url, self.session_id
            );
            let body = serde_json::json!({
                "events": rows.iter().map(|(_, e)| {
                    serde_json::json!({ "event_json": e })
                }).collect::<Vec<_>>(),
            });

            let mut request = self
                .client
                .post(&url)
                .header("x-api-key", &self.config.api_key)
                .json(&body);
            if let Some(token) = self.store.lease_token(&self.session_id)? {
                request = request.header("x-hi-lease-token", token);
            }
            let response = request.send().await;
            note_endpoint_outcome(&self.store, response.as_ref().err());
            let response = match response {
                Ok(response) => response,
                Err(err) => {
                    return Err(err).with_context(|| format!("flushing live events to {url}"));
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(anyhow!("live event flush failed: {status} {body}"));
            }
            self.store
                .acknowledge_events(&rows.iter().map(|(id, _)| *id).collect::<Vec<_>>())?;
        }
    }
}

#[cfg(test)]
fn unique_test_sync_store() -> std::sync::Arc<crate::sync_store::SyncStore> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "hi-sync-test-{}-{nonce}.sqlite3",
        std::process::id()
    ));
    let store =
        crate::sync_store::SyncStore::open_at(path).expect("opening isolated sync test database");
    store
        .set_mode(crate::sync_store::SyncMode::On)
        .expect("enabling isolated sync test database");
    std::sync::Arc::new(store)
}

/// A [`hi_agent::Ui`] that forwards every call to both a primary (local) UI
/// and a secondary (remote) UI. The local UI renders normally; the remote UI
/// buffers events for network sync. This lets a single `run_turn` call
/// simultaneously render locally and stream to remote viewers.
///
/// The `RemoteUi` is wrapped in an `Arc` so it can be flushed after the turn
/// (the `Ui` trait methods use `&mut self`, but `RemoteUi` uses interior
/// mutability via `Mutex`, so sharing is safe).
pub struct MultiplexUi {
    pub primary: Box<dyn hi_agent::Ui>,
    pub remote: std::sync::Arc<RemoteUi>,
}

struct MultiplexSubagentSink {
    primary: Option<Arc<dyn hi_agent::SubagentSink>>,
    remote: Arc<RemoteUi>,
}

impl hi_agent::SubagentSink for MultiplexSubagentSink {
    fn spawned(&self, id: &str, kind: &str, description: &str, background: bool) {
        if let Some(primary) = &self.primary {
            primary.spawned(id, kind, description, background);
        }
        self.remote
            .push_event(hi_tui::event::UiEvent::SubagentSpawned {
                id: id.to_string(),
                subagent_kind: kind.to_string(),
                description: description.to_string(),
                background,
            });
    }
    fn progress(&self, id: &str, activity: &str, line: Option<&str>) {
        if let Some(primary) = &self.primary {
            primary.progress(id, activity, line);
        }
        self.remote
            .push_event(hi_tui::event::UiEvent::SubagentProgress {
                id: id.to_string(),
                activity: activity.to_string(),
                line: line.map(str::to_string),
            });
    }
    fn finished(&self, id: &str, status: &str, elapsed_ms: u64, summary: &str) {
        if let Some(primary) = &self.primary {
            primary.finished(id, status, elapsed_ms, summary);
        }
        self.remote
            .push_event(hi_tui::event::UiEvent::SubagentFinished {
                id: id.to_string(),
                status: status.to_string(),
                elapsed_ms,
                summary: summary.to_string(),
            });
    }
}

impl hi_agent::Ui for MultiplexUi {
    fn assistant_text(&mut self, text: &str) {
        self.primary.assistant_text(text);
        self.remote.push_event(hi_tui::event::UiEvent::Text {
            text: text.to_string(),
        });
    }
    fn assistant_reasoning(&mut self, text: &str) {
        self.primary.assistant_reasoning(text);
        self.remote.push_event(hi_tui::event::UiEvent::Reasoning {
            text: text.to_string(),
        });
    }
    fn assistant_end(&mut self) {
        self.primary.assistant_end();
        self.remote.push_event(hi_tui::event::UiEvent::AssistantEnd);
    }
    fn tool_started(&mut self, name: &str, arguments: &str) {
        self.primary.tool_started(name, arguments);
        self.remote.push_event(hi_tui::event::UiEvent::ToolStarted {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_stream(&mut self, name: &str, line: &str) {
        self.primary.tool_stream(name, line);
        self.remote.push_event(hi_tui::event::UiEvent::ToolStream {
            name: name.to_string(),
            line: line.to_string(),
        });
    }
    fn confirm(
        &mut self,
        request: hi_agent::ConfirmationRequest,
    ) -> hi_agent::ConfirmationFuture<'_> {
        // Only the primary UI confirms edits; the remote viewer is read-only.
        self.primary.confirm(request)
    }
    fn ask_user(&mut self, question: &str, options: &[String]) -> hi_agent::AskUserFuture<'_> {
        self.primary.ask_user(question, options)
    }
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.primary.tool_call(name, arguments);
        self.remote.push_event(hi_tui::event::UiEvent::ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.primary.tool_result(name, result);
        let display_result = hi_agent::ui::user_visible_tool_result(result);
        self.remote.push_event(hi_tui::event::UiEvent::ToolResult {
            name: name.to_string(),
            result: display_result,
        });
    }
    fn plan_result_id(
        &mut self,
        id: &str,
        name: &str,
        result: &str,
        status: hi_tools::ToolStatus,
        steps: &[hi_agent::PlanStep],
    ) {
        self.primary.plan_result_id(id, name, result, status, steps);
        self.remote.push_event(hi_tui::event::UiEvent::Plan {
            steps: steps.to_vec(),
        });
    }
    fn status(&mut self, text: &str) {
        let Some(text) = hi_agent::ui::user_facing_status(text) else {
            return;
        };
        self.primary.status(&text);
        {
            self.remote
                .push_event(hi_tui::event::UiEvent::Status { text });
        }
    }
    fn checkpoint_warning(&mut self, text: &str) {
        self.primary.checkpoint_warning(text);
        self.remote
            .push_event(hi_tui::event::UiEvent::CheckpointWarning {
                text: text.to_string(),
            });
    }
    fn subagent_note(&mut self, text: &str) {
        self.primary.subagent_note(text);
        self.remote.push_event(hi_tui::event::UiEvent::Status {
            text: text.to_string(),
        });
    }
    fn subagent_sink(&self) -> Option<Arc<dyn hi_agent::SubagentSink>> {
        Some(Arc::new(MultiplexSubagentSink {
            primary: self.primary.subagent_sink(),
            remote: self.remote.clone(),
        }))
    }
    fn subagent_spawned(&mut self, id: &str, kind: &str, description: &str, background: bool) {
        self.primary
            .subagent_spawned(id, kind, description, background);
        self.remote
            .push_event(hi_tui::event::UiEvent::SubagentSpawned {
                id: id.to_string(),
                subagent_kind: kind.to_string(),
                description: description.to_string(),
                background,
            });
    }
    fn subagent_progress(&mut self, id: &str, activity: &str) {
        self.primary.subagent_progress(id, activity);
        self.remote
            .push_event(hi_tui::event::UiEvent::SubagentProgress {
                id: id.to_string(),
                activity: activity.to_string(),
                line: None,
            });
    }
    fn subagent_finished(&mut self, id: &str, status: &str, elapsed_ms: u64, summary: &str) {
        self.primary
            .subagent_finished(id, status, elapsed_ms, summary);
        self.remote
            .push_event(hi_tui::event::UiEvent::SubagentFinished {
                id: id.to_string(),
                status: status.to_string(),
                elapsed_ms,
                summary: summary.to_string(),
            });
    }
    fn plan(&mut self, steps: &[hi_agent::PlanStep]) {
        self.primary.plan(steps);
        self.remote.push_event(hi_tui::event::UiEvent::Plan {
            steps: steps.to_vec(),
        });
    }
    fn usage(
        &mut self,
        prompt_tokens: u64,
        generated_tokens: u64,
        context_used: u64,
        context_window: Option<u32>,
        usage_estimated: bool,
    ) {
        self.primary.usage(
            prompt_tokens,
            generated_tokens,
            context_used,
            context_window,
            usage_estimated,
        );
        self.remote.push_event(hi_tui::event::UiEvent::Usage {
            prompt: prompt_tokens,
            generated: generated_tokens,
            ctx_used: context_used,
            ctx_window: context_window,
            estimated: usage_estimated,
        });
    }
    fn session_usage(&mut self, usage: &hi_ai::Usage) {
        self.primary.session_usage(usage);
        self.remote
            .push_event(hi_tui::event::UiEvent::SessionUsage { usage: *usage });
    }
    fn rate_limits(&mut self, rate_limits: Option<hi_ai::RateLimitState>) {
        self.primary.rate_limits(rate_limits);
        self.remote
            .push_event(hi_tui::event::UiEvent::RateLimits { rate_limits });
    }
    fn turn_end(&mut self, summary: &str) {
        self.primary.turn_end(summary);
        self.remote.push_event(hi_tui::event::UiEvent::TurnEnd {
            summary: summary.to_string(),
        });
    }
    fn changed_files(&mut self, files: &[String]) {
        self.primary.changed_files(files);
        self.remote
            .push_event(hi_tui::event::UiEvent::ChangedFiles {
                files: files.to_vec(),
            });
    }
    fn suggested_prompt(&mut self, text: &str) {
        self.primary.suggested_prompt(text);
        self.remote
            .push_event(hi_tui::event::UiEvent::SuggestedPrompt {
                text: text.to_string(),
            });
    }
    fn turn_error(&mut self, kind: &str, message: &str, guidance: &str) {
        self.primary.turn_error(kind, message, guidance);
        self.remote.push_event(hi_tui::event::UiEvent::TurnError {
            error_kind: kind.to_string(),
            message: message.to_string(),
            guidance: guidance.to_string(),
        });
    }
    fn nudge(&mut self, text: &str) {
        self.primary.nudge(text);
    }
}

// ─── Daemon mode (Phase 3) ──────────────────────────────────────────────────

/// A pending input fetched from ipop's input queue.
#[derive(Deserialize)]
struct QueuedInput {
    prompt: String,
    input_seq: u64,
}

/// Response from `GET /v1/hi/sessions/{id}/input`.
#[derive(Deserialize)]
struct InputListResponse {
    inputs: Vec<QueuedInput>,
}

fn remote_input_poll_status_is_terminal(status: reqwest::StatusCode) -> bool {
    matches!(
        status,
        reqwest::StatusCode::UNAUTHORIZED
            | reqwest::StatusCode::FORBIDDEN
            | reqwest::StatusCode::NOT_FOUND
            | reqwest::StatusCode::CONFLICT
            | reqwest::StatusCode::GONE
    )
}

/// Spawn a background long-poll that delivers remote session prompts into
/// `tx`. Stops when `tx` is dropped or the task is aborted. Used by the TUI
/// host mode so remote attach clients can steer a live interactive session.
pub fn spawn_remote_input_poller(
    sync_config: SyncConfig,
    session_id: String,
    lease_token: Option<String>,
    tx: tokio::sync::mpsc::UnboundedSender<String>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let client = sync_http_builder()
            .timeout(std::time::Duration::from_secs(35))
            .build()
            .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(10, 35));
        let input_url = format!("{}/hi/sessions/{session_id}/input", sync_config.base_url);
        let ack_url = format!("{}/hi/sessions/{session_id}/ack", sync_config.base_url);
        let api_key = sync_config.api_key;
        let mut last_acked = 0u64;
        loop {
            if tx.is_closed() {
                break;
            }
            let mut request = client.get(&input_url).header("x-api-key", &api_key);
            if let Some(token) = &lease_token {
                request = request.header("x-hi-lease-token", token);
            }
            let response = match request.send().await {
                Ok(response) => response,
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    continue;
                }
            };
            if !response.status().is_success() {
                if remote_input_poll_status_is_terminal(response.status()) {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                continue;
            }
            let Ok(body) = response.json::<InputListResponse>().await else {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            };
            let mut max_seq = last_acked;
            for input in body.inputs {
                if input.input_seq <= last_acked {
                    continue;
                }
                max_seq = max_seq.max(input.input_seq);
                let prompt = input.prompt.trim().to_string();
                if prompt.is_empty() {
                    continue;
                }
                if tx.send(prompt).is_err() {
                    return;
                }
            }
            if max_seq > last_acked {
                last_acked = max_seq;
                let mut ack = client
                    .post(&ack_url)
                    .header("x-api-key", &api_key)
                    .json(&serde_json::json!({ "up_to_seq": last_acked }));
                if let Some(token) = &lease_token {
                    ack = ack.header("x-hi-lease-token", token);
                }
                let _ = ack.send().await;
            }
        }
    })
}

/// Run the daemon service loop: long-poll ipop for queued inputs, run each as
/// a turn, flush sync records + live events after each turn. Runs until
/// Ctrl-C or a fatal error.
///
/// The `agent` and `sync_handle`/`remote_ui` are the already-configured
/// objects from main.rs — the daemon reuses the same setup as a normal
/// one-shot run, just with a different turn loop.
pub async fn run_daemon_loop(
    mut agent: hi_agent::Agent,
    sync_config: SyncConfig,
    session_id: String,
    sync_handle: Option<std::sync::Arc<RemoteSessionSink>>,
    remote_ui: Option<std::sync::Arc<RemoteUi>>,
) -> Result<()> {
    let client = sync_http_builder()
        .timeout(std::time::Duration::from_secs(35))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(10, 35));

    let base_url = sync_config.base_url.clone();
    let api_key = sync_config.api_key.clone();
    let input_url = format!("{base_url}/hi/sessions/{session_id}/input");
    let heartbeat_url = format!("{base_url}/hi/sessions/{session_id}/heartbeat");
    let ack_url = format!("{base_url}/hi/sessions/{session_id}/ack");

    println!(
        "⟳ hi daemon (pid {}) — session {session_id}; Ctrl-C to stop",
        std::process::id()
    );

    // Trigger early registration so the input token is available immediately.
    if let Some(handle) = &sync_handle {
        handle.ensure_registered_now().await?;
        if let Some(token) = handle.input_token() {
            // Write the token to a local file so `hi --attach` on the same
            // machine can read it automatically.
            if let Some(path) = crate::session::sessions_dir() {
                let token_path = path.join(format!("{session_id}.token"));
                // Ensure the directory exists (it may not if --no-save was used).
                let _ = std::fs::create_dir_all(&path);
                if let Err(err) = write_private_token(&token_path, &token) {
                    eprintln!("\x1b[33mdaemon: couldn't save input token: {err:#}\x1b[0m");
                }
            }
            println!("  input token saved to the private local session token file");
        }
    }
    let writer_lease = sync_handle.as_ref().and_then(|handle| handle.lease_token());

    // Spawn a periodic heartbeat task so ipop knows the daemon is alive.
    let hb_client = client.clone();
    let hb_url = heartbeat_url.clone();
    let hb_key = api_key.clone();
    let hb_lease = writer_lease.clone();
    tokio::spawn(async move {
        let mut consecutive_failures = 0_u8;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            let mut request = hb_client
                .post(&hb_url)
                .header("x-api-key", &hb_key)
                .json(&serde_json::json!({}));
            if let Some(token) = &hb_lease {
                request = request.header("x-hi-lease-token", token);
            }
            match request.send().await {
                Ok(_) => consecutive_failures = 0,
                Err(_) => {
                    consecutive_failures = consecutive_failures.saturating_add(1);
                    if consecutive_failures >= 5 {
                        break;
                    }
                }
            }
        }
    });

    loop {
        // Long-poll for pending inputs, but also watch for shutdown.
        let mut poll_request = client.get(&input_url).header("x-api-key", &api_key);
        if let Some(token) = &writer_lease {
            poll_request = poll_request.header("x-hi-lease-token", token);
        }
        let poll_future = poll_request.send();
        let inputs: Vec<QueuedInput> = tokio::select! {
            result = poll_future => {
                match result {
                    Ok(response) if response.status().is_success() => {
                        match response.json::<InputListResponse>().await {
                            Ok(resp) => resp.inputs,
                            Err(_) => Vec::new(),
                        }
                    }
                    Ok(response) => {
                        let status = response.status();
                        if status == reqwest::StatusCode::CONFLICT {
                            let body = response.text().await.unwrap_or_default();
                            if body.contains("lease_lost") {
                                return Err(anyhow!("lease_lost: this daemon was replaced by another writer"));
                            }
                            return Err(anyhow!("the remote session writer lease is no longer valid"));
                        }
                        if remote_input_poll_status_is_terminal(status) {
                            return Err(anyhow!(
                                "remote session input polling stopped after terminal HTTP status {status}"
                            ));
                        }
                        eprintln!("\x1b[33mdaemon: input poll returned {status}\x1b[0m");
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                    Err(err) => {
                        eprintln!("\x1b[33mdaemon: input poll failed: {err:#}\x1b[0m");
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        continue;
                    }
                }
            }
            _ = hi_daemon_shutdown_signal() => {
                println!("\x1b[2m⟳ daemon stopping — flushing sync and ending session\x1b[0m");
                // Flush any pending sync records + live events.
                if let Some(handle) = &sync_handle {
                    if let Err(err) = handle.flush().await {
                        eprintln!("\x1b[33msync: {err:#}\x1b[0m");
                    }
                    handle.end_session().await;
                }
                if let Some(rui) = &remote_ui
                    && let Err(err) = rui.flush().await {
                        eprintln!("\x1b[33msync events: {err:#}\x1b[0m");
                    }
                // Clean up the local token file so it doesn't persist after the
                // session ends. Best-effort — if removal fails, the token is
                // stale but harmless (the session is ended on the server).
                if let Some(dir) = crate::session::sessions_dir() {
                    let token_path = dir.join(format!("{session_id}.token"));
                    let _ = std::fs::remove_file(&token_path);
                }
                return Ok(());
            }
        };

        if inputs.is_empty() {
            continue;
        }

        // Process each queued input as a turn.
        let max_input_seq = inputs.iter().map(|i| i.input_seq).max();
        for input in inputs {
            let prompt = input.prompt;
            println!("› {prompt}");

            // Build the view: plain stdout + optional remote streamer.
            let result = if let Some(ref rui) = remote_ui {
                let mut multi = MultiplexUi {
                    primary: Box::new(crate::ui::PlainUi::new()),
                    remote: rui.clone(),
                };
                agent.run_turn(&prompt, &mut multi).await
            } else {
                let mut plain = crate::ui::PlainUi::new();
                agent.run_turn(&prompt, &mut plain).await
            };
            if let Err(err) = &result {
                let (kind, guidance) = hi_agent::classify_error(err);
                eprintln!("\x1b[31m{kind}: {err:#} — {guidance}\x1b[0m");
            }
            if result.is_err() {
                let _ = agent.cleanup_turn(hi_agent::TurnCleanupKind::Fail).await;
            }

            // Flush sync records + live events to ipop.
            if let Some(handle) = &sync_handle
                && let Err(err) = handle.flush().await
            {
                eprintln!("\x1b[33msync: {err:#}\x1b[0m");
            }
            if let Some(rui) = &remote_ui
                && let Err(err) = rui.flush().await
            {
                eprintln!("\x1b[33msync events: {err:#}\x1b[0m");
            }
        }

        // Ack the highest processed input_seq so clients know their inputs
        // were received and processed.
        if let Some(last_seq) = max_input_seq {
            let mut request = client
                .post(&ack_url)
                .header("x-api-key", &api_key)
                .json(&serde_json::json!({ "input_seq": last_seq }));
            if let Some(token) = &writer_lease {
                request = request.header("x-hi-lease-token", token);
            }
            let _ = request.send().await;
        }
    }
}

// ─── Attach mode (Phase 3) ──────────────────────────────────────────────────

/// A live event received from the SSE stream.
#[derive(Deserialize)]
struct StreamedEvent {
    event_json: String,
    #[serde(default)]
    event_seq: u64,
}

#[derive(Debug, PartialEq)]
enum AttachStreamMessage {
    Event(String),
    Reconnecting {
        attempt: u32,
        delay: std::time::Duration,
    },
    Restored {
        cursor: u64,
    },
}

fn attach_retry_delay(attempt: u32) -> std::time::Duration {
    std::time::Duration::from_millis(500_u64.saturating_mul(1_u64 << attempt.min(4)))
}

fn accept_streamed_event(last_seq: u64, event: StreamedEvent) -> Option<(u64, String)> {
    if event.event_seq > 0 && event.event_seq <= last_seq {
        return None;
    }
    Some((event.event_seq.max(last_seq), event.event_json))
}

/// How a client should join a remote session over the ipop API (no SSH).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionJoinMode {
    /// Host is alive and accepting input — steer that runtime (tmux-like).
    SteerHost,
    /// No live host — continue the conversation with a local agent (portable).
    ContinueHere,
}

/// Inspect session metadata and choose steer-host vs continue-here.
pub fn classify_session_join(detail: &serde_json::Value) -> SessionJoinMode {
    let host_alive = detail
        .get("host_alive")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let accepts_input = detail
        .get("accepts_input")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let status = detail.get("status").and_then(|v| v.as_str()).unwrap_or("");
    let lease_fresh = detail
        .get("lease_expires_at_unix")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        > std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    // Prefer the server's host_alive bit when present; fall back to local
    // inference for older control planes that omit it.
    if host_alive || (accepts_input && status == "active" && lease_fresh) {
        SessionJoinMode::SteerHost
    } else {
        SessionJoinMode::ContinueHere
    }
}

/// Fetch `GET /hi/sessions/{id}` metadata.
pub async fn fetch_session_detail(
    sync_config: &SyncConfig,
    session_id: &str,
) -> Result<serde_json::Value> {
    validate_session_id(session_id)?;
    let client = sync_http_builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(5, 15));
    let url = format!("{}/hi/sessions/{session_id}", sync_config.base_url);
    let response = client
        .get(&url)
        .header("x-api-key", &sync_config.api_key)
        .send()
        .await
        .context("fetching session metadata")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("session metadata failed: HTTP {status} {body}");
    }
    response.json().await.context("parsing session metadata")
}

/// Smart attach: if a remote host is alive, open a steer session over the API;
/// otherwise resume the conversation locally (portable session).
pub async fn run_smart_attach(
    sync_config: SyncConfig,
    session_id: String,
    input_token: Option<String>,
    settings: &crate::config::Settings,
    cli: &crate::config::Cli,
    agent: &mut hi_agent::Agent,
) -> Result<()> {
    let detail = fetch_session_detail(&sync_config, &session_id).await?;
    match classify_session_join(&detail) {
        SessionJoinMode::SteerHost => {
            let host = detail
                .get("machine_id")
                .and_then(|v| v.as_str())
                .unwrap_or("remote-host");
            println!("\x1b[2m⟳ host alive on {host} — steering over API (no SSH)\x1b[0m");
            run_attach_client(sync_config, session_id, input_token).await
        }
        SessionJoinMode::ContinueHere => {
            println!("\x1b[2m⟳ no live host — continuing conversation on this machine\x1b[0m");
            run_resume_local(sync_config, session_id, settings, cli, agent).await
        }
    }
}

/// Run the attach client: fetch session history, subscribe to the live event
/// stream, and forward typed prompts to the hosting daemon via ipop.
///
/// This is a read-only viewer + input sender. The actual coding work happens
/// on the machine running the daemon.
pub async fn run_attach_client(
    sync_config: SyncConfig,
    session_id: String,
    mut input_token: Option<String>,
) -> Result<()> {
    // If no token was passed via --input-token, try reading it from the local
    // token file (written by the daemon on the same machine).
    if input_token.is_none()
        && let Some(dir) = crate::session::sessions_dir()
    {
        let token_path = dir.join(format!("{session_id}.token"));
        if let Ok(token) = std::fs::read_to_string(&token_path) {
            let token = token.trim().to_string();
            if !token.is_empty() {
                input_token = Some(token);
            }
        }
    }

    let client = sync_http_builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        // The long-lived SSE stream overrides the total timeout per request;
        // these bound it instead: keepalive catches dead hosts, and the
        // read timeout catches wedged applications (a SIGSTOPped server or a
        // proxy that stops forwarding ACKs keepalive probes forever) at the
        // cost of one clean resumed reconnect per 5 idle minutes.
        .read_timeout(std::time::Duration::from_secs(300))
        .tcp_keepalive(Some(std::time::Duration::from_secs(60)))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(10, 300));

    let base_url = sync_config.base_url.clone();
    let api_key = sync_config.api_key.clone();

    // 1. Fetch session metadata.
    let detail_url = format!("{base_url}/hi/sessions/{session_id}");
    let detail: serde_json::Value = client
        .get(&detail_url)
        .header("x-api-key", &api_key)
        .send()
        .await
        .context("fetching session metadata")?
        .error_for_status()
        .context("session metadata request failed")?
        .json()
        .await
        .context("parsing session metadata")?;

    let status = detail
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let record_count = detail
        .get("record_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let title = detail.get("title").and_then(|v| v.as_str()).unwrap_or("");

    let accepts_input = detail
        .get("accepts_input")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    println!(
        "⟳ hi attach — session {session_id} ({status}, {record_count} records){}",
        if title.is_empty() {
            String::new()
        } else {
            format!(": {title}")
        }
    );
    if !accepts_input {
        println!(
            "\x1b[33m  note: this session is not accepting remote input \
             (host with `hi --daemon --sync` to steer it)\x1b[0m"
        );
    }

    // 2. Fetch the full durable history (paginated — never truncate at page size).
    let history_records = fetch_remote_records(&client, &sync_config, &session_id)
        .await
        .context("fetching session records")?;
    for record in &history_records {
        // Render the record: if it's a message, show the role + text;
        // otherwise surface usage tags.
        if record.record_type == "message" {
            if let Ok(msg) = serde_json::from_str::<hi_ai::Message>(&record.payload_json) {
                let role = match msg.role {
                    hi_ai::Role::User => "you",
                    hi_ai::Role::Assistant => "hi",
                    hi_ai::Role::System => "sys",
                    hi_ai::Role::Tool => "tool",
                };
                let text = msg.text();
                if !text.trim().is_empty() {
                    println!("\x1b[36m{role}\x1b[0m: {text}");
                }
            }
        } else if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&record.payload_json)
            && let Some(meta_type) = meta.get("type").and_then(|v| v.as_str())
            && meta_type == "usage"
        {
            let input = meta
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output = meta
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            println!("\x1b[2m  [{input} in · {output} out]\x1b[0m");
        }
    }

    println!("\x1b[2m  — live stream follows (type to send input, Ctrl-C to exit) —\x1b[0m");

    // 3. Spawn the SSE event stream subscriber.
    let stream_url = format!("{base_url}/hi/sessions/{session_id}/events/stream");
    let stream_client = client.clone();
    let stream_api_key = api_key.clone();
    let (stream_tx, mut stream_rx) = tokio::sync::mpsc::unbounded_channel::<AttachStreamMessage>();

    tokio::spawn(async move {
        let mut last_seq: u64 = 0;
        let mut retry_attempt = 0_u32;
        let mut reconnecting = false;
        loop {
            // On reconnect, include from_seq so the server backfills missed
            // durable records before the live stream resumes.
            let url = if last_seq > 0 {
                format!("{stream_url}?from_seq={}", last_seq + 1)
            } else {
                stream_url.clone()
            };
            // Per-request timeout override: the client-wide 60s total timeout
            // covers body streaming, which would kill the healthy live stream
            // every minute — and each forced reconnect silently drops events
            // on servers that don't stamp event_seq. Dead peers are detected
            // by TCP keepalive instead.
            let response = match stream_client
                .get(&url)
                .timeout(std::time::Duration::from_secs(24 * 60 * 60))
                .header("x-api-key", &stream_api_key)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(_) => {
                    let delay = attach_retry_delay(retry_attempt);
                    retry_attempt = retry_attempt.saturating_add(1);
                    reconnecting = true;
                    let _ = stream_tx.send(AttachStreamMessage::Reconnecting {
                        attempt: retry_attempt,
                        delay,
                    });
                    tokio::time::sleep(delay).await;
                    continue;
                }
            };
            if !response.status().is_success() {
                let delay = attach_retry_delay(retry_attempt);
                retry_attempt = retry_attempt.saturating_add(1);
                reconnecting = true;
                let _ = stream_tx.send(AttachStreamMessage::Reconnecting {
                    attempt: retry_attempt,
                    delay,
                });
                tokio::time::sleep(delay).await;
                continue;
            }
            if reconnecting {
                let _ = stream_tx.send(AttachStreamMessage::Restored { cursor: last_seq });
            }
            // The backoff resets only once an event actually arrives (below):
            // a server that returns 200 and instantly closes the body would
            // otherwise reconnect at the minimum delay forever.

            use futures_util::StreamExt;
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(_) => break,
                };
                buffer.push_str(&String::from_utf8_lossy(&chunk));
                // SSE permits CRLF as well as LF. Normalize after each chunk;
                // this also handles a CR/LF pair split across two chunks.
                if buffer.contains('\r') {
                    buffer = buffer.replace("\r\n", "\n");
                }

                // Process complete SSE events (separated by "\n\n").
                while let Some(pos) = buffer.find("\n\n") {
                    let event_text = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    // Parse "data: <json>" lines.
                    for line in event_text.lines() {
                        if let Some(data) = line.strip_prefix("data: ")
                            && let Ok(event) = serde_json::from_str::<StreamedEvent>(data)
                            && let Some((next_seq, event_json)) =
                                accept_streamed_event(last_seq, event)
                            && stream_tx
                                .send(AttachStreamMessage::Event(event_json))
                                .is_ok()
                        {
                            last_seq = next_seq;
                            retry_attempt = 0;
                        }
                    }
                }
            }

            let delay = attach_retry_delay(retry_attempt);
            retry_attempt = retry_attempt.saturating_add(1);
            reconnecting = true;
            let _ = stream_tx.send(AttachStreamMessage::Reconnecting {
                attempt: retry_attempt,
                delay,
            });
            tokio::time::sleep(delay).await;
        }
    });

    // 4. Spawn the input reader (stdin → ipop).
    let input_url = format!("{base_url}/hi/sessions/{session_id}/input");
    let input_client = client.clone();
    let input_api_key = api_key.clone();
    let input_token_clone = input_token.clone();
    let (input_tx, mut input_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    tokio::task::spawn_blocking(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let trimmed = line.trim().to_string();
            if !trimmed.is_empty() {
                let _ = input_tx.send(trimmed);
            }
        }
    });

    // 5. Main loop: select between live events and user input.
    let mut live_status = None;
    loop {
        tokio::select! {
            Some(message) = stream_rx.recv() => {
                match message {
                    AttachStreamMessage::Event(event_json) => {
                        if let Ok(event) = serde_json::from_str::<hi_tui::event::UiEvent>(&event_json) {
                            render_live_event(&event, &mut live_status);
                        }
                    }
                    AttachStreamMessage::Reconnecting { attempt, delay } => {
                        clear_live_status(&mut live_status);
                        eprintln!(
                            "\x1b[33m  ⟳ live stream disconnected — reconnecting (attempt {attempt}, {:.1}s)\x1b[0m",
                            delay.as_secs_f32()
                        );
                    }
                    AttachStreamMessage::Restored { cursor } => {
                        clear_live_status(&mut live_status);
                        eprintln!(
                            "\x1b[32m  ✓ live stream restored from cursor {cursor}\x1b[0m"
                        );
                    }
                }
            }
            Some(prompt) = input_rx.recv() => {
                clear_live_status(&mut live_status);
                // Send the prompt to ipop's input queue.
                let body = serde_json::json!({ "prompt": prompt });
                let mut req = input_client
                    .post(&input_url)
                    .header("x-api-key", &input_api_key);
                if let Some(token) = &input_token_clone {
                    req = req.header("x-hi-input-token", token);
                }
                let resp = req
                    .json(&body)
                    .send()
                    .await;
                match resp {
                    Ok(r) if r.status().is_success() => {
                        println!("\x1b[2m  → sent to daemon\x1b[0m");
                    }
                    Ok(r) => {
                        eprintln!("\x1b[33m  → failed: HTTP {}\x1b[0m", r.status());
                    }
                    Err(err) => {
                        eprintln!("\x1b[33m  → failed: {err:#}\x1b[0m");
                    }
                }
            }
            _ = hi_daemon_shutdown_signal() => {
                clear_live_status(&mut live_status);
                println!("\x1b[2m  — detaching —\x1b[0m");
                break;
            }
        }
    }

    Ok(())
}

/// Render a live UiEvent to stdout (a simplified version of the TUI transcript).
fn render_live_event(event: &hi_tui::event::UiEvent, live_status: &mut Option<String>) {
    use hi_tui::event::UiEvent;

    // Keep one replaceable line for retry/progress chatter. Usage and other
    // event-only updates should not erase it; any event that emits durable
    // output clears it first so the next line starts cleanly.
    if let UiEvent::Status { text } = event
        && let Some(text) = hi_agent::ui::user_facing_status(text)
        && hi_agent::ui::is_live_progress_status(&text)
    {
        if live_status.as_deref() == Some(text.as_str()) {
            return;
        }
        clear_live_status(live_status);
        eprint!("\r\x1b[K\x1b[34m{text}\x1b[0m");
        use std::io::Write;
        let _ = std::io::stderr().flush();
        *live_status = Some(text);
        return;
    }
    let keep_live_status = match event {
        UiEvent::BtwEnd
        | UiEvent::ProviderRequest { .. }
        | UiEvent::SessionUsage { .. }
        | UiEvent::RateLimits { .. } => true,
        UiEvent::Status { text } => hi_agent::ui::user_facing_status(text).is_none(),
        UiEvent::SubagentProgress { activity, .. } => activity.is_empty(),
        _ => false,
    };
    if !keep_live_status {
        clear_live_status(live_status);
    }

    match event {
        UiEvent::Text { text } => {
            print!("{text}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        UiEvent::BtwQuestion { question } => {
            eprintln!("\x1b[2m❓ btw: {question}\x1b[0m");
        }
        UiEvent::BtwAnswer { text } => {
            // Side-question answer: dim it so it reads as an aside from task output.
            eprintln!("\x1b[2m↳ btw: {text}\x1b[0m");
        }
        UiEvent::BtwToolStarted { name, arguments } => {
            eprintln!("\x1b[2m  · btw {name} {arguments}\x1b[0m");
        }
        UiEvent::BtwToolResult { name, result } => {
            let clipped = clip_chars(result, 120);
            eprintln!("\x1b[2m  ← btw {name}: {clipped}\x1b[0m");
        }
        UiEvent::BtwEnd => {}
        UiEvent::ProviderRequest { .. } => {}
        UiEvent::Reasoning { text } => {
            eprintln!("\x1b[2m{text}\x1b[0m");
        }
        UiEvent::AssistantEnd => {
            println!();
        }
        UiEvent::ToolStarted { name, arguments } => {
            eprintln!("\x1b[36m  ⏺ {name} {arguments}\x1b[0m");
        }
        UiEvent::ToolCall { name, arguments } => {
            eprintln!("\x1b[36m  ⏺ {name} {arguments}\x1b[0m");
        }
        UiEvent::ToolResult { name, result } => {
            let display_result = hi_agent::ui::user_visible_tool_result(result);
            let clipped = clip_chars(&display_result, 200);
            eprintln!("\x1b[2m  ← {name}: {clipped}\x1b[0m");
        }
        UiEvent::ToolStream { name, line } => {
            eprintln!("\x1b[2m  │ {name}: {line}\x1b[0m");
        }
        UiEvent::Status { text } => {
            if let Some(text) = hi_agent::ui::user_facing_status(text) {
                eprintln!("\x1b[2m  {text}\x1b[0m");
            }
        }
        UiEvent::TopStatus { text } => {
            if let Some(text) = hi_agent::ui::user_facing_status(text) {
                let text = hi_agent::ui::without_leading_warning_marker(&text);
                eprintln!("\x1b[33m  ⚠ {text}\x1b[0m");
            }
        }
        UiEvent::CheckpointWarning { text } => {
            let text = hi_agent::ui::without_leading_warning_marker(text);
            eprintln!("\x1b[33m  ⚠ {text}\x1b[0m");
        }
        UiEvent::Plan { steps } => {
            eprintln!("\x1b[2m  plan: {} step(s)\x1b[0m", steps.len());
        }
        UiEvent::Usage {
            prompt,
            generated,
            estimated,
            ..
        } => {
            let approx = if *estimated { "~" } else { "" };
            eprintln!(
                "\x1b[2m  [user prompt estimate {prompt} · output across all model calls {approx}{generated}]\x1b[0m"
            );
        }
        UiEvent::SessionUsage { .. } => {}
        UiEvent::RateLimits { .. } => {}
        UiEvent::TurnEnd { summary } => {
            println!("\x1b[2m  ✓ {summary}\x1b[0m");
        }
        UiEvent::TurnError {
            error_kind,
            message,
            guidance,
        } => {
            eprintln!("\x1b[31m  ✗ {error_kind}: {message} — {guidance}\x1b[0m");
        }
        UiEvent::ChangedFiles { files } => {
            eprintln!(
                "\x1b[32m  ✎ {} file(s) changed: {}\x1b[0m",
                files.len(),
                files.join(", ")
            );
        }
        UiEvent::SuggestedPrompt { text } => {
            eprintln!("\x1b[2m  hint: {text}\x1b[0m");
        }
        UiEvent::WorkflowUpdated { snapshot } => {
            eprintln!(
                "\x1b[2m  ⚙ workflow {}: {:?}\x1b[0m",
                snapshot.run_id, snapshot.status
            );
        }
        UiEvent::DiffRunUpdated { snapshot } => {
            eprintln!(
                "\x1b[2m  ⇄ diff run {}: {:?} · {}/{} cases · {} mismatches\x1b[0m",
                snapshot.run_id,
                snapshot.status,
                snapshot.cases_completed,
                snapshot.cases_total,
                snapshot.mismatches
            );
        }
        UiEvent::SubagentSpawned {
            subagent_kind,
            description,
            background,
            ..
        } => {
            let bg = if *background { " background" } else { "" };
            eprintln!("\x1b[36m  ↳ {subagent_kind}{bg} subagent: {description}\x1b[0m");
        }
        UiEvent::SubagentProgress { activity, .. } => {
            if !activity.is_empty() {
                eprintln!("\x1b[2m  ↳ {activity}\x1b[0m");
            }
        }
        UiEvent::SubagentFinished {
            status, summary, ..
        } => {
            eprintln!("\x1b[2m  ↳ subagent {status}: {summary}\x1b[0m");
        }
    }
}

fn clear_live_status(live_status: &mut Option<String>) {
    if live_status.take().is_some() {
        eprint!("\r\x1b[K");
        use std::io::Write;
        let _ = std::io::stderr().flush();
    }
}

fn clip_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let clipped: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{clipped}…")
    } else {
        clipped
    }
}

fn write_private_token(path: &std::path::Path, token: &str) -> Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("securing {}", path.display()))?;
    }
    use std::io::Write;
    file.write_all(token.as_bytes())
        .with_context(|| format!("writing {}", path.display()))
}

/// Resolves on Ctrl-C or SIGTERM.
async fn hi_daemon_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Response from `GET /v1/hi/sessions/{id}/records`.
#[derive(Deserialize)]
struct RecordsResponse {
    records: Vec<RemoteRecordResponse>,
    #[serde(default)]
    has_more: bool,
    #[serde(default)]
    next_seq: Option<u64>,
}

/// One record in the records response.
#[derive(Deserialize)]
struct RemoteRecordResponse {
    record_type: String,
    payload_json: String,
    /// 1-based sequence number, used for pagination. May be absent in older
    /// server responses — defaults to None.
    #[serde(default)]
    record_seq: Option<u64>,
}

/// Page through `GET /hi/sessions/{id}/records` until every durable record is
/// local. Shared by resume-local, TUI `/sessions switch`, and attach history so
/// a long session is never truncated at the server's default page size (500).
async fn fetch_remote_records(
    client: &reqwest::Client,
    sync_config: &SyncConfig,
    session_id: &str,
) -> Result<Vec<crate::session::RemoteRecord>> {
    validate_session_id(session_id)?;
    let records_url = format!("{}/hi/sessions/{session_id}/records", sync_config.base_url);
    let mut all_records: Vec<RemoteRecordResponse> = Vec::new();
    let mut from_seq: Option<u64> = Some(1);
    let mut expected_seq = 1u64;
    let mut pages = 0usize;
    loop {
        let mut request = client
            .get(&records_url)
            .header("x-api-key", &sync_config.api_key);
        request = request.query(&[("from_seq", from_seq.unwrap_or(1)), ("limit", 1000)]);
        let response = request
            .send()
            .await
            .context("fetching session records from ipop")?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("failed to fetch records: HTTP {status} {body}");
        }

        pages = pages.saturating_add(1);
        if pages > 10_000 {
            // Catches non-progress patterns the cursor guard below can't see,
            // e.g. a broken server oscillating next_seq between two values
            // while returning already-seen records.
            anyhow::bail!("session record pagination exceeded 10000 pages");
        }
        let batch: RecordsResponse = response.json().await.context("parsing session records")?;
        let batch_len = batch.records.len();
        for record in batch.records {
            if let Some(sequence) = record.record_seq {
                if sequence < expected_seq {
                    continue;
                }
                if sequence != expected_seq {
                    anyhow::bail!(
                        "session record gap: expected sequence {expected_seq}, received {sequence}"
                    );
                }
                expected_seq = expected_seq.saturating_add(1);
            } else {
                expected_seq = expected_seq.saturating_add(1);
            }
            all_records.push(record);
        }
        if !batch.has_more && batch_len < 5_000 {
            break;
        }
        let next_from = batch.next_seq.or(Some(expected_seq));
        // A page must move the cursor: a server that keeps returning
        // already-seen sequences with has_more=true would otherwise loop
        // forever (duplicates are `continue`d above without advancing).
        if batch_len == 0 || next_from == from_seq {
            anyhow::bail!("session record pagination stalled at sequence {expected_seq}");
        }
        from_seq = next_from;
    }
    reassemble_remote_records(all_records)
}

/// Fetch and reconstruct a synced session's durable state. Shared by startup
/// resume and in-TUI `/sessions switch`, so a session has the same behavior
/// whether or not this machine already has its JSONL cache.
pub struct FetchedSessionHistory {
    pub loaded: crate::session::LoadedSession,
    pub pipefs: Option<RemotePipeFsSummary>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RemotePipeFsSummary {
    pub enabled: bool,
    pub restoration_required: bool,
}

#[derive(Deserialize)]
struct RemoteSessionDetail {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    pipefs: Option<RemotePipeFsSummary>,
}

pub async fn fetch_session_history(
    sync_config: &SyncConfig,
    session_id: &str,
) -> Result<FetchedSessionHistory> {
    validate_session_id(session_id)?;
    let client = sync_http_builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| hi_ai::timed_http_client_fallback(10, 30));
    let records = fetch_remote_records(&client, sync_config, session_id).await?;
    let mut loaded = crate::session::load_history_from_records(&records)?;
    // The rename endpoint updates session metadata without appending a durable
    // record, so fetch the current title separately when restoring a session.
    let detail_url = format!("{}/hi/sessions/{session_id}", sync_config.base_url);
    let response = client
        .get(detail_url)
        .header("x-api-key", &sync_config.api_key)
        .send()
        .await
        .context("fetching session detail from ipop")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("failed to fetch session detail: HTTP {status} {body}");
    }
    let detail: RemoteSessionDetail = response
        .json()
        .await
        .context("parsing session detail from ipop")?;
    if let Some(title) = detail.title.filter(|title| !title.trim().is_empty()) {
        loaded.name = Some(title.trim().to_string());
    }
    Ok(FetchedSessionHistory {
        loaded,
        pipefs: detail.pipefs,
    })
}

fn reassemble_remote_records(
    records: Vec<RemoteRecordResponse>,
) -> Result<Vec<crate::session::RemoteRecord>> {
    use sha2::{Digest, Sha256};
    let mut parts: std::collections::HashMap<String, Vec<Option<String>>> =
        std::collections::HashMap::new();
    let mut output = Vec::new();
    for record in records {
        match record.record_type.as_str() {
            "chunk_part" => {
                // Tolerate a malformed chunk_part: skip it with a warning and
                // leave the slot absent. The matching chunk_commit will then
                // find an incomplete set and skip itself, so a single corrupt
                // part never makes the entire session unresumable.
                let parsed = (|| -> Result<()> {
                    let value: serde_json::Value = serde_json::from_str(&record.payload_json)
                        .context("invalid chunk_part payload")?;
                    let id = value["logical_id"]
                        .as_str()
                        .context("chunk_part omitted logical_id")?
                        .to_string();
                    let index = value["index"]
                        .as_u64()
                        .context("chunk_part omitted index")?
                        as usize;
                    let count = value["parts"]
                        .as_u64()
                        .context("chunk_part omitted parts")?
                        as usize;
                    let data = value["data"].as_str().context("chunk_part omitted data")?;
                    if count == 0 || count > 65_536 || index >= count {
                        anyhow::bail!("invalid chunk_part bounds");
                    }
                    let entry = parts.entry(id).or_insert_with(|| vec![None; count]);
                    if entry.len() != count {
                        anyhow::bail!("chunk_part count changed within logical record");
                    }
                    if entry[index]
                        .as_deref()
                        .is_some_and(|existing| existing != data)
                    {
                        anyhow::bail!("conflicting duplicate chunk_part");
                    }
                    entry[index] = Some(data.to_string());
                    Ok(())
                })();
                if let Err(error) = parsed {
                    eprintln!("\x1b[33msync: skipping malformed chunk_part: {error:#}\x1b[0m");
                }
            }
            "chunk_commit" => {
                // Tolerate any corruption in a chunk_commit — missing fields,
                // incomplete parts, hash mismatch, or invalid reassembled JSON.
                // The writer contract states "readers apply only a complete,
                // verified set", so a single corrupt oversized record must not
                // make the entire session unresumable. Drop it with a warning
                // and continue processing the rest of the history.
                let parsed = (|| -> Result<serde_json::Value> {
                    let value: serde_json::Value = serde_json::from_str(&record.payload_json)
                        .context("invalid chunk_commit payload")?;
                    let id = value["logical_id"]
                        .as_str()
                        .context("chunk_commit omitted logical_id")?;
                    let record_type = value["record_type"]
                        .as_str()
                        .context("chunk_commit omitted record_type")?;
                    let expected_hash = value["sha256"]
                        .as_str()
                        .context("chunk_commit omitted sha256")?;
                    let expected_parts = value["parts"]
                        .as_u64()
                        .context("chunk_commit omitted parts")?
                        as usize;
                    let Some(chunks) = parts.remove(id) else {
                        eprintln!(
                            "\x1b[33msync: skipping incomplete chunk_commit {id} — no chunk_part records found\x1b[0m"
                        );
                        return Ok(serde_json::Value::Null);
                    };
                    if chunks.len() != expected_parts || chunks.iter().any(Option::is_none) {
                        eprintln!(
                            "\x1b[33msync: skipping incomplete chunk_commit {id} — expected {expected_parts} parts, got {}\x1b[0m",
                            chunks.iter().filter(|c| c.is_some()).count()
                        );
                        return Ok(serde_json::Value::Null);
                    }
                    let payload_json = chunks.into_iter().flatten().collect::<String>();
                    let actual_hash = format!("{:x}", Sha256::digest(payload_json.as_bytes()));
                    if actual_hash != expected_hash {
                        eprintln!(
                            "\x1b[33msync: skipping chunk_commit {id} — hash mismatch\x1b[0m"
                        );
                        return Ok(serde_json::Value::Null);
                    }
                    if serde_json::from_str::<serde_json::Value>(&payload_json).is_err() {
                        eprintln!(
                            "\x1b[33msync: skipping chunk_commit {id} — reassembled payload is not valid JSON\x1b[0m"
                        );
                        return Ok(serde_json::Value::Null);
                    }
                    output.push(crate::session::RemoteRecord {
                        record_type: record_type.to_string(),
                        payload_json,
                    });
                    Ok(serde_json::Value::Null)
                })();
                if let Err(error) = parsed {
                    eprintln!("\x1b[33msync: skipping malformed chunk_commit: {error:#}\x1b[0m");
                }
            }
            _ => output.push(crate::session::RemoteRecord {
                record_type: record.record_type,
                payload_json: record.payload_json,
            }),
        }
    }
    if !parts.is_empty() {
        // Orphaned chunk_part records without a matching chunk_commit are
        // tolerated: the writer may have failed before emitting the commit,
        // or the commit may have been lost. Skip them with a warning rather
        // than making the session unresumable.
        for id in parts.keys() {
            eprintln!(
                "\x1b[33msync: skipping orphaned chunk_part records for {id} — no chunk_commit found\x1b[0m"
            );
        }
    }
    Ok(output)
}

/// Resume a remote session locally: fetch the durable record history from ipop,
/// reconstruct the conversation via `load_history_from_records`, apply it to
/// the agent, and run a local interactive REPL that continues from there.
///
/// This is the "daemon is down, keep working" path. The local agent picks up
/// the remote session's transcript, goal, and decisions, and continues as if
/// the session had been resumed from a local JSONL file.
pub async fn run_resume_local(
    sync_config: SyncConfig,
    session_id: String,
    settings: &crate::config::Settings,
    cli: &crate::config::Cli,
    agent: &mut hi_agent::Agent,
) -> Result<()> {
    // 1. Fetch and reconstruct the synced session.
    let fetched = fetch_session_history(&sync_config, &session_id).await?;
    let remote_requires_pipefs = fetched
        .pipefs
        .as_ref()
        .is_some_and(|pipefs| pipefs.enabled || pipefs.restoration_required);
    let loaded = fetched.loaded;

    let n_messages = loaded.messages.len();
    let has_goal = loaded.goal.is_some();
    println!(
        "\x1b[2m⟳ resume-local — session {session_id}: {n_messages} messages{} from ipop\x1b[0m",
        if has_goal { " + goal" } else { "" },
    );

    // 3. Seed a new local continuation file with the complete remote state.
    //    Merely detaching the startup sink would make every continued turn
    //    ephemeral and leave a partial local file that cannot be resumed.
    let local_path = crate::session::new_session_path()?;
    let local_id = local_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("continuation")
        .to_string();
    crate::session::cache_loaded_session(&local_path, &loaded)?;
    let local = crate::session::JsonlSession::new(local_path.clone());
    let remote = RemoteSessionSink::new(sync_config.clone(), session_id.clone());
    remote.seed_snapshot(&loaded)?;
    // Take the writer lease (and re-register) *before* any continued turn can
    // flush. Without this, a remote machine's first append races a still-live
    // host lease and fails with lease_lost — the TUI `/sessions switch` path
    // already does ensure_registered_now for the same reason.
    remote
        .ensure_registered_now()
        .await
        .context("claiming writer lease for resume-local")?;

    // Detach the startup sink (which points at an unrelated empty session),
    // apply the remote state, then attach the seeded local continuation plus a
    // remote sink for subsequent portal records.
    agent.detach_session();
    crate::session::apply_loaded_session(agent, loaded)?;
    let sync_session = SyncSession::new(local, remote);
    let sync_handle = sync_session.remote_handle();
    let pipefs_sync_handle: crate::pipefs::SharedSyncHandle =
        std::sync::Arc::new(std::sync::Mutex::new(Some(sync_handle.clone())));
    agent.set_session(Box::new(sync_session));
    println!("\x1b[2m  local continuation: {local_id} (writer lease claimed)\x1b[0m");

    // The remote session is the authority for PipeFS mode when resuming on a
    // different machine. Restore and verify the workspace before accepting a
    // prompt; a failed restore must leave the launch directory inactive rather
    // than silently continuing against the wrong files.
    let pipefs_host = std::sync::Arc::new(crate::pipefs::PipeFsHost::new(
        sync_config.clone(),
        session_id.clone(),
        local_path,
        pipefs_sync_handle,
        agent.workspace_root().to_path_buf(),
        agent.state_root().to_path_buf(),
        crate::pipefs::PipeFsMcpConfig::resolve(
            settings,
            &crate::config::load_config(cli.config.as_deref()).unwrap_or_default(),
        ),
    )?);
    let pipefs_activated = pipefs_host
        .activate_for_startup(agent, false, remote_requires_pipefs)
        .await
        .context("restoring PipeFS workspace for resume-local")?;
    anyhow::ensure!(
        !remote_requires_pipefs || pipefs_activated,
        "the session summary requires PipeFS restoration, but the workspace service did not return an enabled workspace; refusing to continue in the launch directory"
    );

    // 4. Run a local interactive REPL (plain mode — no TUI since we're in
    //    attach context). The user continues the conversation locally.
    if let Some(prompt) = &cli.prompt {
        // One-shot mode: run a single turn and exit.
        let mut plain = crate::ui::PlainUi::new();
        let result = agent.run_turn(prompt, &mut plain).await;
        if let Err(err) = &result {
            let (kind, guidance) = hi_agent::classify_error(err);
            eprintln!("\x1b[31m{kind}: {err:#} — {guidance}\x1b[0m");
        }
        if result.is_err() {
            let _ = agent.cleanup_turn(hi_agent::TurnCleanupKind::Fail).await;
        }
        agent.kill_background_processes();
        agent.background_task_registry().kill_all().await;
        if let Err(err) = sync_handle.flush().await {
            eprintln!("\x1b[33msync: {err:#}\x1b[0m");
        }
        pipefs_host
            .clean_exit(agent)
            .await
            .context("persisting PipeFS workspace during resume-local shutdown")?;
        sync_handle.end_session().await;
        return result.map(|_| ());
    }

    // Interactive: delegate to the plain REPL.
    // Load a fresh config for the REPL — it needs a mutable Config for profile
    // lookups, but doesn't persist changes (no save_config calls in repl.rs).
    let mut file = crate::config::load_config(cli.config.as_deref()).unwrap_or_default();
    let auto_memory = !cli.no_memory && !cli.no_save;
    let active_profile = cli.profile.clone().or_else(|| file.default_profile.clone());
    let after_turn: std::sync::Arc<dyn Fn() + Send + Sync> = {
        let sync_handle = sync_handle.clone();
        std::sync::Arc::new(move || {
            let sync_handle = sync_handle.clone();
            tokio::spawn(async move {
                let _ = sync_handle.flush().await;
            });
        })
    };
    let result = crate::repl::repl(
        agent,
        settings,
        &mut file,
        auto_memory,
        active_profile,
        cli.config.clone(),
        Some(after_turn),
        None,
        Some(pipefs_host.clone()),
    )
    .await;
    if let Err(err) = sync_handle.flush().await {
        eprintln!("\x1b[33msync: {err:#}\x1b[0m");
    }
    agent.kill_background_processes();
    agent.background_task_registry().kill_all().await;
    pipefs_host
        .clean_exit(agent)
        .await
        .context("persisting PipeFS workspace during resume-local shutdown")?;
    sync_handle.end_session().await;
    result
}

#[cfg(test)]
#[path = "sync_tests/mod.rs"]
mod tests;
