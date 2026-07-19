//! Remote session sync: pushes hi session records to an ipop API endpoint so
//! the session can be viewed (and later resumed) from another machine.
//!
//! Phase 1 is sync-only: the local `hi` process still owns the agent and the
//! filesystem. This module provides a [`RemoteSessionSink`] that mirrors the
//! JSONL records to ipop alongside the local file. The sink is best-effort —
//! if the network is down, the local session continues uninterrupted and the
//! failed records are queued for the next flush.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
const MAX_RECORD_WIRE_BYTES: usize = 5_000_000;
// Leave room for JSON escaping and chunk metadata so each encoded chunk_part
// remains below the 1 MiB wire contract.
const CHUNK_PART_BYTES: usize = 450 * 1024;

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
    lease_lost: AtomicBool,
    heartbeat_started: AtomicBool,
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
}

impl RemoteSessionSink {
    pub fn new(config: SyncConfig, session_id: String) -> Self {
        Self::with_activation(config, session_id, None)
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
    ) -> Self {
        let client = remote_session_http_client();
        let store = std::sync::Arc::new(
            crate::sync_store::SyncStore::open().expect("opening durable portal sync database"),
        );
        Self::with_store(config, session_id, activation, client, store)
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
            lease_lost: AtomicBool::new(false),
            heartbeat_started: AtomicBool::new(false),
            accepts_input: AtomicBool::new(false),
            title: Mutex::new(None),
            registered_title: Mutex::new(None),
            flush_lock: tokio::sync::Mutex::new(()),
            activation: tokio::sync::Mutex::new(activation),
            next_record_id: AtomicU64::new(0),
        }
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
            let _ = self
                .store
                .enqueue_record(&self.session_id, record_type, payload_json);
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
            self.store
                .enqueue_record(&self.session_id, "chunk_part", &part.to_string())?;
        }
        let commit = serde_json::json!({
            "logical_id": logical_id,
            "record_type": record_type,
            "parts": parts.len(),
            "sha256": format!("{:x}", Sha256::digest(payload_json.as_bytes())),
            "bytes": payload_json.len(),
        });
        self.store
            .enqueue_record(&self.session_id, "chunk_commit", &commit.to_string())?;
        Ok(())
    }

    /// Reconcile complete JSONL lines after the last committed local offset.
    /// The offset-derived ids make replay deterministic across crashes, and
    /// `INSERT OR IGNORE` on the record id makes replay idempotent — so a
    /// suspect offset can always be reset to 0 rather than trusted.
    pub fn reconcile_jsonl(&self, path: &std::path::Path) -> Result<()> {
        use sha2::{Digest, Sha256};
        use std::io::{Read, Seek, SeekFrom};
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
        let mut remaining = Vec::new();
        file.read_to_end(&mut remaining)?;
        let mut consumed = 0usize;
        for line in remaining.split_inclusive(|byte| *byte == b'\n') {
            if !line.ends_with(b"\n") {
                break;
            }
            let payload = std::str::from_utf8(&line[..line.len() - 1])?;
            if payload.is_empty() {
                consumed += line.len();
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
            consumed += line.len();
            offset = offset.saturating_add(line.len() as u64);
        }
        if consumed > 0 {
            self.store.set_jsonl_offset(&self.session_id, offset)?;
        }
        Ok(())
    }

    fn enqueue_reconciled(&self, base_id: &str, record_type: &str, payload: &str) -> Result<()> {
        use sha2::{Digest, Sha256};
        if serde_json::to_string(payload)?.len() <= MAX_RECORD_WIRE_BYTES {
            return self.store.enqueue_record_with_id(
                &self.session_id,
                base_id,
                record_type,
                payload,
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
            self.store.enqueue_record_with_id(
                &self.session_id,
                &format!("{base_id}.p{index}"),
                "chunk_part",
                &part.to_string(),
            )?;
        }
        let commit = serde_json::json!({
            "logical_id": base_id, "record_type": record_type, "parts": chunks.len(),
            "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())), "bytes": payload.len(),
        });
        self.store.enqueue_record_with_id(
            &self.session_id,
            &format!("{base_id}.commit"),
            "chunk_commit",
            &commit.to_string(),
        )
    }

    fn set_title(&self, title: Option<String>) {
        let title = title
            .map(|title| title.trim().to_string())
            .filter(|title| !title.is_empty());
        if title.is_some() {
            *self.title.lock().unwrap() = title;
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
        if self.title.lock().unwrap().is_some() {
            return;
        }
        let title = messages
            .iter()
            .find(|message| message.role == Role::User)
            .map(|message| {
                let text = message.text();
                let title = text.split_whitespace().collect::<Vec<_>>().join(" ");
                hi_agent::ui::clip(&title, 120)
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
        Ok(())
    }

    /// The per-session input token, if the server returned one at registration.
    /// Used by the daemon to write a local token file so `hi --attach` on the
    /// same machine can submit inputs.
    pub fn input_token(&self) -> Option<String> {
        self.input_token.lock().unwrap().clone()
    }

    /// Force registration now (normally deferred to the first flush). The
    /// daemon calls this at startup so the input token is available
    /// immediately.
    pub async fn ensure_registered_now(&self) -> Result<()> {
        self.ensure_registered().await
    }

    /// Register the session with ipop if not already done. Called before the
    /// first flush. A failed registration is retried on the next flush; marking
    /// it successful after a network error permanently strands the session.
    async fn ensure_registered(&self) -> Result<()> {
        if self.store.effective_mode()? != crate::sync_store::SyncMode::On {
            return Ok(());
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
        if *self.registered.lock().unwrap() {
            return self.sync_title().await;
        }
        let url = format!("{}/hi/sessions", self.config.base_url);
        let title = self.title.lock().unwrap().clone();
        let body = serde_json::json!({
            "session_id": self.session_id,
            "machine_id": self.config.machine_id,
            "cwd_digest": self.config.cwd_digest,
            "project_fingerprint": crate::session::project_fingerprint(),
            "title": title,
            "accepts_input": self.accepts_input.load(Ordering::Acquire),
        });
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("registering session at {url}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("ipop session registration failed: {status} {body}"));
        }
        if let Ok(json) = response.json::<serde_json::Value>().await
            && let Some(token) = json.get("input_token").and_then(|v| v.as_str())
        {
            *self.input_token.lock().unwrap() = Some(token.to_string());
        }
        *self.registered_title.lock().unwrap() = title;
        self.acquire_lease(true).await?;
        self.start_lease_heartbeat();
        *self.registered.lock().unwrap() = true;
        Ok(())
    }

    async fn acquire_lease(&self, takeover: bool) -> Result<()> {
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
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.config.api_key)
            .json(&serde_json::json!({
                "client_instance_id": client_instance_id,
                "machine_id": machine_id,
                "takeover": takeover,
            }))
            .send()
            .await
            .with_context(|| format!("acquiring session lease at {url}"))?;
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
        Ok(())
    }

    /// Declare that this process is polling for remote input. Must be called before the session
    /// registers, since the flag is sent in the registration body.
    pub fn set_accepts_input(&self, value: bool) {
        self.accepts_input.store(value, Ordering::Release);
    }

    pub fn lease_token(&self) -> Option<String> {
        self.store.lease_token(&self.session_id).ok().flatten()
    }

    fn start_lease_heartbeat(&self) {
        if self.heartbeat_started.swap(true, Ordering::AcqRel) || self.lease_token().is_none() {
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
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                if store.effective_mode().ok() != Some(crate::sync_store::SyncMode::On) {
                    continue;
                }
                let Some(token) = store.lease_token(&session_id).ok().flatten() else {
                    break;
                };
                let response = client
                    .post(&url)
                    .header("x-api-key", &api_key)
                    .header("x-hi-lease-token", token)
                    .json(&serde_json::json!({}))
                    .send()
                    .await;
                if response
                    .as_ref()
                    .is_ok_and(|response| response.status() == reqwest::StatusCode::CONFLICT)
                {
                    break;
                }
            }
        });
    }

    async fn sync_title(&self) -> Result<()> {
        let title = self.title.lock().unwrap().clone();
        if title.is_none() || title == *self.registered_title.lock().unwrap() {
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
        *self.registered_title.lock().unwrap() = title;
        Ok(())
    }

    /// Flush all pending records to ipop. Called after each turn. Best-effort:
    /// on failure, records stay buffered for the next attempt.
    pub async fn flush(&self) -> Result<()> {
        let _flush = self.flush_lock.lock().await;
        if self.store.effective_mode()? != crate::sync_store::SyncMode::On {
            return Ok(());
        }
        self.ensure_registered().await?;
        loop {
            let mut records = self.store.ready_records(&self.session_id, 512)?;
            if records.is_empty() {
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
            let response = match request.send().await {
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
                self.store.fail_records(
                    &self.session_id,
                    &records,
                    &format!("HTTP {status}: {body}"),
                    retry_after,
                    permanent,
                )?;
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
        let _ = request.send().await;
    }
}

fn remote_session_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .http1_only()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
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
        remote
            .reconcile_jsonl(local.path())
            .expect("reconciling durable portal outbox");
        Self {
            local,
            remote: std::sync::Arc::new(remote),
        }
    }

    /// Get a handle to the remote sink for flushing / ending the session.
    /// Call this before boxing the `SyncSession` for the agent.
    pub fn remote_handle(&self) -> std::sync::Arc<RemoteSessionSink> {
        self.remote.clone()
    }
}

impl SessionSink for SyncSession {
    fn record(&mut self, messages: &[Message], usage: Usage) -> Result<()> {
        self.local.record(messages, usage)?;
        self.remote.observe_messages(messages);
        self.remote.reconcile_jsonl(self.local.path())
    }

    fn record_compaction(&mut self, messages: &[Message]) -> Result<()> {
        self.local.record_compaction(messages)?;
        self.remote.reconcile_jsonl(self.local.path())
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
        self.remote.reconcile_jsonl(self.local.path())
    }

    fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        self.local.record_checkpoints(refs)?;
        self.remote.reconcile_jsonl(self.local.path())
    }

    fn record_goal(&mut self, goal: &hi_agent::Goal) -> Result<()> {
        self.local.record_goal(goal)?;
        self.remote.reconcile_jsonl(self.local.path())
    }

    fn clear_goal(&mut self) -> Result<()> {
        self.local.clear_goal()?;
        self.remote.reconcile_jsonl(self.local.path())
    }

    fn record_plan(&mut self, plan: &[hi_agent::PlanStep]) -> Result<()> {
        self.local.record_plan(plan)?;
        self.remote.reconcile_jsonl(self.local.path())
    }

    fn clear_plan(&mut self) -> Result<()> {
        self.local.clear_plan()?;
        self.remote.reconcile_jsonl(self.local.path())
    }

    fn record_decisions(&mut self, decisions: &hi_agent::DecisionLog) -> Result<()> {
        self.local.record_decisions(decisions)?;
        self.remote.reconcile_jsonl(self.local.path())
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
    pub fn new(config: SyncConfig, session_id: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .http1_only()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let store = std::sync::Arc::new(
            crate::sync_store::SyncStore::open().expect("opening durable portal event database"),
        );
        Self::with_store(config, session_id, client, store)
    }

    #[cfg(test)]
    pub fn new_for_test(config: SyncConfig, session_id: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .http1_only()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
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
            let response = match request.send().await {
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
    fn tool_call(&mut self, name: &str, arguments: &str) {
        self.primary.tool_call(name, arguments);
        self.remote.push_event(hi_tui::event::UiEvent::ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        });
    }
    fn tool_result(&mut self, name: &str, result: &str) {
        self.primary.tool_result(name, result);
        self.remote.push_event(hi_tui::event::UiEvent::ToolResult {
            name: name.to_string(),
            result: result.to_string(),
        });
    }
    fn status(&mut self, text: &str) {
        self.primary.status(text);
        self.remote.push_event(hi_tui::event::UiEvent::Status {
            text: text.to_string(),
        });
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
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(35))
        .http1_only()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

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
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            let mut request = hb_client
                .post(&hb_url)
                .header("x-api-key", &hb_key)
                .json(&serde_json::json!({}));
            if let Some(token) = &hb_lease {
                request = request.header("x-hi-lease-token", token);
            }
            let _ = request.send().await;
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
                agent.finalize_failed_turn();
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

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .http1_only()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

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

    println!(
        "⟳ hi attach — session {session_id} ({status}, {record_count} records){}",
        if title.is_empty() {
            String::new()
        } else {
            format!(": {title}")
        }
    );

    // 2. Fetch session records (the durable history).
    let records_url = format!("{base_url}/hi/sessions/{session_id}/records");
    let records: serde_json::Value = client
        .get(&records_url)
        .header("x-api-key", &api_key)
        .send()
        .await
        .context("fetching session records")?
        .error_for_status()
        .context("session records request failed")?
        .json()
        .await
        .context("parsing session records")?;

    if let Some(records_arr) = records.get("records").and_then(|v| v.as_array()) {
        for record in records_arr {
            if let Some(payload) = record.get("payload_json").and_then(|v| v.as_str()) {
                // Render the record: if it's a message, show the role + text;
                // otherwise show the type tag.
                if let Ok(msg) = serde_json::from_str::<hi_ai::Message>(payload) {
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
                } else if let Ok(meta) = serde_json::from_str::<serde_json::Value>(payload)
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
        }
    }

    println!("\x1b[2m  — live stream follows (type to send input, Ctrl-C to exit) —\x1b[0m");

    // 3. Spawn the SSE event stream subscriber.
    let stream_url = format!("{base_url}/hi/sessions/{session_id}/events/stream");
    let stream_client = client.clone();
    let stream_api_key = api_key.clone();
    let (stream_tx, mut stream_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let mut last_seq: u64 = 0;
        loop {
            // On reconnect, include from_seq so the server backfills missed
            // durable records before the live stream resumes.
            let url = if last_seq > 0 {
                format!("{stream_url}?from_seq={}", last_seq + 1)
            } else {
                stream_url.clone()
            };
            let response = match stream_client
                .get(&url)
                .header("x-api-key", &stream_api_key)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(_) => {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    continue;
                }
            };

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
                        {
                            if event.event_seq > 0 {
                                last_seq = last_seq.max(event.event_seq);
                            }
                            let _ = stream_tx.send(event.event_json);
                        }
                    }
                }
            }

            // Reconnect after a short delay.
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
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
    loop {
        tokio::select! {
            Some(event_json) = stream_rx.recv() => {
                // Render the live event.
                if let Ok(event) = serde_json::from_str::<hi_tui::event::UiEvent>(&event_json) {
                    render_live_event(&event);
                }
            }
            Some(prompt) = input_rx.recv() => {
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
                println!("\x1b[2m  — detaching —\x1b[0m");
                break;
            }
        }
    }

    Ok(())
}

/// Render a live UiEvent to stdout (a simplified version of the TUI transcript).
fn render_live_event(event: &hi_tui::event::UiEvent) {
    use hi_tui::event::UiEvent;
    match event {
        UiEvent::Text { text } => {
            print!("{text}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
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
            let clipped = clip_chars(result, 200);
            eprintln!("\x1b[2m  ← {name}: {clipped}\x1b[0m");
        }
        UiEvent::ToolStream { name, line } => {
            eprintln!("\x1b[2m  │ {name}: {line}\x1b[0m");
        }
        UiEvent::Status { text } => {
            eprintln!("\x1b[2m  {text}\x1b[0m");
        }
        UiEvent::CheckpointWarning { text } => {
            eprintln!("\x1b[33m  {text}\x1b[0m");
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

/// Fetch and reconstruct a synced session's durable state. Shared by startup
/// resume and in-TUI `/sessions switch`, so a session has the same behavior
/// whether or not this machine already has its JSONL cache.
pub async fn fetch_session_history(
    sync_config: &SyncConfig,
    session_id: &str,
) -> Result<crate::session::LoadedSession> {
    validate_session_id(session_id)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .http1_only()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let records_url = format!("{}/hi/sessions/{session_id}/records", sync_config.base_url);
    let mut all_records: Vec<RemoteRecordResponse> = Vec::new();
    let mut from_seq: Option<u64> = Some(1);
    let mut expected_seq = 1u64;
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
        from_seq = batch.next_seq.or(Some(expected_seq));
        if batch_len == 0 {
            anyhow::bail!("session record pagination stalled at sequence {expected_seq}");
        }
    }

    let records = reassemble_remote_records(all_records)?;
    let mut loaded = crate::session::load_history_from_records(&records)?;
    // The rename endpoint updates session metadata without appending a durable
    // record, so fetch the current title separately when restoring a session.
    let detail_url = format!("{}/hi/sessions/{session_id}", sync_config.base_url);
    if let Ok(response) = client
        .get(detail_url)
        .header("x-api-key", &sync_config.api_key)
        .send()
        .await
        && response.status().is_success()
        && let Ok(detail) = response.json::<serde_json::Value>().await
        && let Some(title) = detail.get("title").and_then(|value| value.as_str())
        && !title.trim().is_empty()
    {
        loaded.name = Some(title.trim().to_string());
    }
    Ok(loaded)
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
                        .context("chunk_part omitted index")? as usize;
                    let count = value["parts"]
                        .as_u64()
                        .context("chunk_part omitted parts")? as usize;
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
                    eprintln!(
                        "\x1b[33msync: skipping malformed chunk_part: {error:#}\x1b[0m"
                    );
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
                    eprintln!(
                        "\x1b[33msync: skipping malformed chunk_commit: {error:#}\x1b[0m"
                    );
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
    let loaded = fetch_session_history(&sync_config, &session_id).await?;

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
    let local = crate::session::JsonlSession::new(local_path);
    let remote = RemoteSessionSink::new(sync_config.clone(), session_id.clone());
    remote.seed_snapshot(&loaded)?;

    // Detach the startup sink (which points at an unrelated empty session),
    // apply the remote state, then attach the seeded local continuation plus a
    // remote sink for subsequent portal records.
    agent.detach_session();
    agent.apply_loaded_session(
        loaded.messages,
        loaded.usage,
        loaded.checkpoint_refs,
        loaded.goal,
        loaded.decisions,
        loaded.plan,
    );
    let sync_session = SyncSession::new(local, remote);
    let sync_handle = sync_session.remote_handle();
    agent.set_session(Box::new(sync_session));
    println!("\x1b[2m  local continuation: {local_id}\x1b[0m");

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
            agent.finalize_failed_turn();
        }
        if let Err(err) = sync_handle.flush().await {
            eprintln!("\x1b[33msync: {err:#}\x1b[0m");
        }
        sync_handle.end_session().await;
        agent.kill_background_processes();
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
    )
    .await;
    if let Err(err) = sync_handle.flush().await {
        eprintln!("\x1b[33msync: {err:#}\x1b[0m");
    }
    sync_handle.end_session().await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::Message;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn read_mock_http_request(
        socket: &mut tokio::net::TcpStream,
    ) -> std::io::Result<Vec<u8>> {
        const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;
        let mut request = Vec::new();
        let mut chunk = [0_u8; 16 * 1024];
        loop {
            let read = socket.read(&mut chunk).await?;
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
            if request.len() > MAX_REQUEST_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "mock request exceeds test limit",
                ));
            }
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let header_bytes = &request[..header_end];
            let headers = String::from_utf8_lossy(header_bytes);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                break;
            }
        }
        Ok(request)
    }

    /// A minimal mock HTTP server that records received requests.
    /// Returns 200 OK for every request and counts POSTs.
    struct MockServer {
        base_url: String,
        post_count: Arc<AtomicUsize>,
        _handle: tokio::task::JoinHandle<()>,
    }

    impl MockServer {
        async fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let post_count = Arc::new(AtomicUsize::new(0));
            let count_clone = post_count.clone();
            let handle = tokio::spawn(async move {
                loop {
                    let (mut sock, _) = match listener.accept().await {
                        Ok(s) => s,
                        Err(_) => break,
                    };
                    let count = count_clone.clone();
                    tokio::spawn(async move {
                        let Ok(request) = read_mock_http_request(&mut sock).await else {
                            return;
                        };
                        let request = String::from_utf8_lossy(&request);
                        if request.starts_with("POST") {
                            count.fetch_add(1, Ordering::SeqCst);
                        }
                        // This mock handles exactly one request per accepted
                        // socket. Advertise that lifecycle explicitly so the
                        // pooled reqwest client never races a follow-up request
                        // against a socket the task has already dropped.
                        let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
                        let _ = sock.write_all(response.as_bytes()).await;
                        let _ = sock.shutdown().await;
                    });
                }
            });
            Self {
                base_url: format!("http://{addr}"),
                post_count,
                _handle: handle,
            }
        }

        fn post_count(&self) -> usize {
            self.post_count.load(Ordering::SeqCst)
        }
    }

    #[tokio::test]
    async fn remote_session_sink_flushes_records() {
        let server = MockServer::start().await;
        let config = SyncConfig {
            base_url: server.base_url.clone(),
            api_key: "test-key".to_string(),
            machine_id: Some("test-machine".to_string()),
            cwd_digest: Some("0123456789abcdef".to_string()),
        };
        let sink = RemoteSessionSink::new_for_test(config, "test-session-1".to_string());

        // Push a message record via SyncSession (which delegates to the remote sink).
        let local = crate::session::JsonlSession::new(
            std::env::temp_dir().join(format!("hi-sync-test-{}.jsonl", std::process::id())),
        );
        let mut sync = SyncSession::new(local, sink);
        let messages = vec![Message::user("hello world")];
        sync.record(&messages, Usage::default()).unwrap();

        // Flush — should send a POST to the server.
        sync.remote_handle().flush().await.unwrap();

        // The server should have received at least one POST (registration + records).
        assert!(
            server.post_count() >= 1,
            "expected at least 1 POST, got {}",
            server.post_count()
        );

        // Clean up.
        let _ = std::fs::remove_file(
            std::env::temp_dir().join(format!("hi-sync-test-{}.jsonl", std::process::id())),
        );
    }

    /// The `--session-file` collision bug: session ids derive from the file
    /// stem, so a second session at a same-named path inherits the first
    /// session's byte offset into a different file. Reconcile must reset the
    /// stale offset and proceed — the old behavior failed with "invalid JSONL
    /// record at byte N" on every reconcile forever, poisoning whole turns as
    /// infrastructure errors.
    #[test]
    fn reconcile_jsonl_recovers_from_stale_offset_of_a_previous_session_file() {
        let sink =
            RemoteSessionSink::new_for_test(unreachable_config(), "stale-offset".to_string());
        let dir = std::env::temp_dir().join(format!(
            "hi-sync-stale-offset-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.jsonl");

        // Session A: tracking begins before the file exists (as in the real
        // flow), then content appends and reconcile commits the EOF offset.
        sink.reconcile_jsonl(&path).unwrap();
        std::fs::write(
            &path,
            "{\"type\":\"usage\",\"input_tokens\":1,\"output_tokens\":1,\"padding\":\"a fairly long first-session record\"}\n",
        )
        .unwrap();
        sink.reconcile_jsonl(&path).unwrap();
        assert!(
            !sink
                .store
                .ready_records("stale-offset", 32)
                .unwrap()
                .is_empty(),
            "session A records enqueued"
        );

        // Session B replaces the file with a shorter transcript: the stored
        // offset now points past the new EOF.
        std::fs::write(
            &path,
            "{\"type\":\"usage\",\"input_tokens\":2,\"output_tokens\":2}\n",
        )
        .unwrap();
        sink.reconcile_jsonl(&path)
            .expect("past-EOF offset must reset, not fail");

        // A stale offset can also land mid-record (byte before it is not a
        // newline). That, too, must reset instead of failing.
        sink.store.set_jsonl_offset("stale-offset", 5).unwrap();
        sink.reconcile_jsonl(&path)
            .expect("mid-record offset must reset, not fail");

        // After recovery the committed offset is the current file's EOF.
        let len = std::fs::metadata(&path).unwrap().len();
        assert_eq!(sink.store.track_jsonl("stale-offset", &path).unwrap(), len);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn session_snapshot_backfills_state_and_title() {
        let sink = RemoteSessionSink::new_for_test(unreachable_config(), "snapshot".to_string());
        let loaded = crate::session::LoadedSession {
            messages: vec![Message::user("first portal prompt")],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 2,
                ..Usage::default()
            },
            checkpoint_refs: vec!["checkpoint-1".into()],
            name: Some("Named portal session".into()),
            goal: None,
            decisions: hi_agent::DecisionLog::default(),
            plan: Vec::new(),
        };

        sink.seed_snapshot(&loaded).unwrap();

        assert_eq!(
            sink.title.lock().unwrap().as_deref(),
            Some("Named portal session")
        );
        let record_types = sink
            .store
            .ready_records("snapshot", 32)
            .unwrap()
            .iter()
            .map(|record| record.record_type.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            record_types,
            vec![
                RECORD_TYPE_STATE_REPLACEMENT.to_string(),
                RECORD_TYPE_USAGE.to_string(),
                RECORD_TYPE_CHECKPOINTS.to_string()
            ]
        );
    }

    #[test]
    fn oversized_record_cannot_jam_later_sync() {
        let sink = RemoteSessionSink::new_for_test(unreachable_config(), "oversized".to_string());
        let huge =
            serde_json::to_string(&Message::user("x".repeat(MAX_RECORD_WIRE_BYTES))).unwrap();
        sink.push(RECORD_TYPE_MESSAGE, &huge);
        sink.push(RECORD_TYPE_STATE_REPLACEMENT, &huge);
        sink.push(
            RECORD_TYPE_MESSAGE,
            &serde_json::to_string(&Message::user("next turn")).unwrap(),
        );

        let pending = sink.store.ready_records("oversized", 64).unwrap();
        assert!(
            pending
                .iter()
                .any(|record| record.record_type == "chunk_part")
        );
        assert_eq!(
            pending
                .iter()
                .filter(|record| record.record_type == "chunk_commit")
                .count(),
            2,
            "both oversized logical records must be committed"
        );
        assert!(
            pending
                .iter()
                .any(|record| record.payload_json.contains("next turn"))
        );
    }

    #[tokio::test]
    async fn title_discovered_after_registration_is_synced() {
        let server = MockServer::start().await;
        let sink = RemoteSessionSink::new_for_test(
            SyncConfig {
                base_url: server.base_url.clone(),
                api_key: "test-key".into(),
                machine_id: None,
                cwd_digest: None,
            },
            "title-sync".into(),
        );
        sink.ensure_registered_now().await.unwrap();
        assert_eq!(
            server.post_count(),
            2,
            "registration plus lease capability probe"
        );

        sink.update_title("Portal work");
        sink.flush().await.unwrap();

        assert_eq!(
            server.post_count(),
            3,
            "registration, lease, and title update"
        );
        assert_eq!(
            sink.registered_title.lock().unwrap().as_deref(),
            Some("Portal work")
        );
    }

    #[tokio::test]
    async fn replacement_session_waits_for_background_handoff() {
        let server = MockServer::start().await;
        let (handoff_tx, handoff_rx) = tokio::sync::oneshot::channel();
        let sink = std::sync::Arc::new(RemoteSessionSink::new_after_drain(
            SyncConfig {
                base_url: server.base_url.clone(),
                api_key: "test-key".into(),
                machine_id: None,
                cwd_digest: None,
            },
            "replacement".into(),
            handoff_rx,
        ));
        sink.push(
            RECORD_TYPE_MESSAGE,
            &serde_json::to_string(&Message::user("after switch")).unwrap(),
        );
        let flushing = {
            let sink = sink.clone();
            tokio::spawn(async move { sink.flush().await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(
            server.post_count(),
            0,
            "registered before predecessor drained"
        );

        handoff_tx.send(()).unwrap();
        flushing.await.unwrap().unwrap();
        assert_eq!(
            server.post_count(),
            3,
            "registration, lease, and record append"
        );
    }

    #[tokio::test]
    async fn remote_ui_flushes_events() {
        let server = MockServer::start().await;
        let config = SyncConfig {
            base_url: server.base_url.clone(),
            api_key: "test-key".to_string(),
            machine_id: None,
            cwd_digest: None,
        };
        let rui = RemoteUi::new_for_test(config, "test-session-2".to_string());

        // Push some events via the MultiplexUi (which calls push_event).
        let mut multi = MultiplexUi {
            primary: Box::new(crate::ui::QuietUi),
            remote: std::sync::Arc::new(rui),
        };
        use hi_agent::Ui;
        multi.assistant_text("hello");
        multi.assistant_end();
        multi.turn_end("[10 in · 5 out]");

        // Flush — should send a POST to the events endpoint.
        // We need to get the RemoteUi back to flush. Since it's behind Arc in
        // the MultiplexUi, we can access it via the remote field.
        multi.remote.flush().await.unwrap();

        assert!(
            server.post_count() >= 1,
            "expected at least 1 POST, got {}",
            server.post_count()
        );
    }

    #[tokio::test]
    async fn uievent_roundtrips_through_sync() {
        // Verify that a UiEvent can be serialized, sent as event_json, and
        // deserialized back — the core of the live streaming protocol.
        let original = hi_tui::event::UiEvent::Text {
            text: "hello from the agent".to_string(),
        };
        let json = serde_json::to_string(&original).unwrap();
        // Simulate what the server receives: an event_json string inside a POST body.
        let body = serde_json::json!({
            "events": [{"event_json": json}]
        });
        let body_str = serde_json::to_string(&body).unwrap();
        // Parse it back.
        let parsed: serde_json::Value = serde_json::from_str(&body_str).unwrap();
        let event_json = parsed["events"][0]["event_json"].as_str().unwrap();
        let decoded: hi_tui::event::UiEvent = serde_json::from_str(event_json).unwrap();
        match decoded {
            hi_tui::event::UiEvent::Text { text } => {
                assert_eq!(text, "hello from the agent");
            }
            _ => panic!("expected Text event"),
        }
    }

    fn unreachable_config() -> SyncConfig {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        SyncConfig {
            base_url: format!("http://{addr}"),
            api_key: "test-key".to_string(),
            machine_id: None,
            cwd_digest: None,
        }
    }

    #[tokio::test]
    async fn failed_record_flush_keeps_records_and_retries_registration() {
        let sink = RemoteSessionSink::new_for_test(unreachable_config(), "safe-id".to_string());
        sink.push(RECORD_TYPE_MESSAGE, r#"{"role":"user","content":[]}"#);

        assert!(sink.flush().await.is_err());
        assert_eq!(sink.store.ready_records("safe-id", 10).unwrap().len(), 1);
        assert!(!*sink.registered.lock().unwrap());
    }

    #[tokio::test]
    async fn failed_event_flush_keeps_events() {
        let ui = RemoteUi::new_for_test(unreachable_config(), "safe-id".to_string());
        ui.push_event(hi_tui::event::UiEvent::Text {
            text: "keep me".to_string(),
        });

        assert!(ui.flush().await.is_err());
        assert_eq!(ui.store.ready_events("safe-id", 10).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn flush_chunks_batches_to_server_contract_limits() {
        let server = MockServer::start().await;
        let config = SyncConfig {
            base_url: server.base_url.clone(),
            api_key: "test-key".to_string(),
            machine_id: None,
            cwd_digest: None,
        };
        let records = RemoteSessionSink::new_for_test(config.clone(), "record-chunks".to_string());
        for _ in 0..513 {
            records.push(RECORD_TYPE_MESSAGE, r#"{"role":"user","content":[]}"#);
        }
        records.flush().await.unwrap();
        // Registration, lease capability probe, plus two record batches.
        assert_eq!(server.post_count(), 4);

        let events = RemoteUi::new_for_test(config, "event-chunks".to_string());
        for _ in 0..257 {
            events.push_event(hi_tui::event::UiEvent::AssistantEnd);
        }
        events.flush().await.unwrap();
        // Two more event batches (256 + 1).
        assert_eq!(server.post_count(), 6);
    }

    #[test]
    fn session_ids_are_safe_single_path_segments() {
        for valid in ["session-123", "abc_DEF", "2026.07.12"] {
            validate_session_id(valid).unwrap();
        }
        for invalid in ["", "../escape", "with/slash", "contains space", "é"] {
            assert!(
                validate_session_id(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn unicode_tool_output_clipping_stays_on_char_boundaries() {
        let input = "🦀".repeat(201);
        let clipped = clip_chars(&input, 200);
        assert_eq!(clipped.chars().count(), 201);
        assert!(clipped.ends_with('…'));
    }

    /// Helper: build a RemoteRecordResponse with a given type and payload.
    fn record(record_type: &str, payload_json: &str, seq: u64) -> RemoteRecordResponse {
        RemoteRecordResponse {
            record_type: record_type.to_string(),
            payload_json: payload_json.to_string(),
            record_seq: Some(seq),
        }
    }

    /// Helper: build chunk_part + chunk_commit records for a logical payload.
    fn chunked_records(
        logical_id: &str,
        record_type: &str,
        payload: &str,
        start_seq: u64,
    ) -> Vec<RemoteRecordResponse> {
        use sha2::{Digest, Sha256};
        let mut parts = Vec::new();
        let mut start = 0;
        while start < payload.len() {
            let mut end = (start + CHUNK_PART_BYTES).min(payload.len());
            while !payload.is_char_boundary(end) {
                end -= 1;
            }
            parts.push(&payload[start..end]);
            start = end;
        }
        let mut out = Vec::new();
        let mut seq = start_seq;
        for (index, data) in parts.iter().enumerate() {
            out.push(record(
                "chunk_part",
                &serde_json::json!({
                    "logical_id": logical_id,
                    "index": index,
                    "parts": parts.len(),
                    "data": data,
                })
                .to_string(),
                seq,
            ));
            seq += 1;
        }
        out.push(record(
            "chunk_commit",
            &serde_json::json!({
                "logical_id": logical_id,
                "record_type": record_type,
                "parts": parts.len(),
                "sha256": format!("{:x}", Sha256::digest(payload.as_bytes())),
                "bytes": payload.len(),
            })
            .to_string(),
            seq,
        ));
        out
    }

    #[test]
    fn reassemble_skips_incomplete_chunk_commit_with_missing_parts() {
        // A chunk_commit arrives but its chunk_part records were never
        // persisted (writer failed mid-way but the commit slipped through).
        // The reader must skip it with a warning, not bail.
        let commit = record(
            "chunk_commit",
            &serde_json::json!({
                "logical_id": "abc123",
                "record_type": "message",
                "parts": 2,
                "sha256": "deadbeef",
                "bytes": 100,
            })
            .to_string(),
            1,
        );
        let normal = record(
            "message",
            &serde_json::to_string(&Message::user("survives")).unwrap(),
            2,
        );
        let records = vec![commit, normal];
        let output = reassemble_remote_records(records).expect("must not bail on incomplete commit");
        assert_eq!(
            output.len(),
            1,
            "incomplete chunk_commit is skipped, normal record survives"
        );
        assert!(output[0].payload_json.contains("survives"));
    }

    #[test]
    fn reassemble_skips_chunk_commit_with_partial_parts() {
        // Two parts expected, only one arrived. Must skip, not bail.
        let part = record(
            "chunk_part",
            &serde_json::json!({
                "logical_id": "xyz789",
                "index": 0,
                "parts": 2,
                "data": "partial",
            })
            .to_string(),
            1,
        );
        let commit = record(
            "chunk_commit",
            &serde_json::json!({
                "logical_id": "xyz789",
                "record_type": "message",
                "parts": 2,
                "sha256": "deadbeef",
                "bytes": 100,
            })
            .to_string(),
            2,
        );
        let records = vec![part, commit];
        let output =
            reassemble_remote_records(records).expect("partial parts must not bail the reader");
        assert!(
            output.is_empty(),
            "incomplete chunk_commit with partial parts is skipped"
        );
    }

    #[test]
    fn reassemble_skips_chunk_commit_with_hash_mismatch() {
        // All parts present but the reassembled hash doesn't match the commit.
        let payload = "hello world";
        let mut records = chunked_records("hashbad", "message", payload, 1);
        // Corrupt the sha256 in the commit record (last record).
        let commit_idx = records.len() - 1;
        let mut commit_value: serde_json::Value =
            serde_json::from_str(&records[commit_idx].payload_json).unwrap();
        commit_value["sha256"] = serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
        records[commit_idx].payload_json = commit_value.to_string();

        let output =
            reassemble_remote_records(records).expect("hash mismatch must not bail the reader");
        assert!(
            output.is_empty(),
            "chunk_commit with hash mismatch is skipped"
        );
    }

    #[test]
    fn reassemble_tolerates_orphaned_chunk_parts_without_commit() {
        // chunk_part records exist but no chunk_commit ever arrived. The reader
        // must skip them with a warning, not bail.
        let part = record(
            "chunk_part",
            &serde_json::json!({
                "logical_id": "orphan1",
                "index": 0,
                "parts": 2,
                "data": "orphaned",
            })
            .to_string(),
            1,
        );
        let normal = record(
            "message",
            &serde_json::to_string(&Message::user("survives")).unwrap(),
            2,
        );
        let records = vec![part, normal];
        let output = reassemble_remote_records(records)
            .expect("orphaned parts must not bail the reader");
        assert_eq!(output.len(), 1, "orphaned parts skipped, normal record kept");
        assert!(output[0].payload_json.contains("survives"));
    }

    #[test]
    fn reassemble_complete_chunked_record_round_trips() {
        // Sanity: a well-formed chunked record still reassembles correctly.
        let payload = serde_json::to_string(&Message::user("x".repeat(MAX_RECORD_WIRE_BYTES))).unwrap();
        let records = chunked_records("good1", "message", &payload, 1);
        let output = reassemble_remote_records(records).expect("complete chunked record reassembles");
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].record_type, "message");
        assert_eq!(output[0].payload_json, payload);
    }

    #[test]
    fn reassemble_skips_malformed_chunk_part_payload() {
        // A chunk_part with invalid JSON payload must be skipped, not bailed.
        let bad_part = record("chunk_part", "{not valid json", 1);
        let normal = record(
            "message",
            &serde_json::to_string(&Message::user("survives")).unwrap(),
            2,
        );
        let records = vec![bad_part, normal];
        let output = reassemble_remote_records(records)
            .expect("malformed chunk_part must not bail the reader");
        assert_eq!(output.len(), 1, "malformed part skipped, normal record kept");
        assert!(output[0].payload_json.contains("survives"));
    }

    #[test]
    fn reassemble_skips_malformed_chunk_commit_payload() {
        // A chunk_commit with invalid JSON payload must be skipped, not bailed.
        let bad_commit = record("chunk_commit", "{not valid json", 1);
        let normal = record(
            "message",
            &serde_json::to_string(&Message::user("survives")).unwrap(),
            2,
        );
        let records = vec![bad_commit, normal];
        let output = reassemble_remote_records(records)
            .expect("malformed chunk_commit must not bail the reader");
        assert_eq!(output.len(), 1, "malformed commit skipped, normal record kept");
        assert!(output[0].payload_json.contains("survives"));
    }

    #[test]
    fn reassemble_skips_chunk_commit_missing_required_fields() {
        // A chunk_commit that is valid JSON but missing required fields
        // (e.g. no "parts") must be skipped, not bailed.
        let bad_commit = record(
            "chunk_commit",
            &serde_json::json!({
                "logical_id": "nofields",
                "record_type": "message",
                "sha256": "abc",
            })
            .to_string(),
            1,
        );
        let normal = record(
            "message",
            &serde_json::to_string(&Message::user("survives")).unwrap(),
            2,
        );
        let records = vec![bad_commit, normal];
        let output = reassemble_remote_records(records)
            .expect("chunk_commit missing fields must not bail the reader");
        assert_eq!(output.len(), 1, "incomplete commit skipped, normal record kept");
        assert!(output[0].payload_json.contains("survives"));
    }
}
