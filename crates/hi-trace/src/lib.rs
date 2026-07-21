//! Private, append-only candidate evidence for RSI-observed `hi` turns.
//!
//! Event records contain only metadata and content-addressed references. Full
//! prompts, model messages, tool payloads, patches, and verification output are
//! stored in the BLAKE3 CAS below the trace directory.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

mod spool;
pub use spool::{DEFAULT_UPLOAD_BATCH, DurableSpool, SpoolPriority, SpoolRecord, retry_delay};

pub const TRACE_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_RUN_MAX_BYTES: u64 = 512 * 1024 * 1024;
pub const DEFAULT_GLOBAL_MAX_BYTES: u64 = 5 * 1024 * 1024 * 1024;
pub const DEFAULT_RETENTION_DAYS: u64 = 30;
const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
static TRACE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceMode {
    Local,
    Managed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BlobRef {
    pub hash: String,
    pub size_bytes: u64,
    pub media_type: String,
}

/// Worker-provided provenance that binds a candidate-authored trace to the
/// authoritative control-plane run. It is repeated in the report summary and
/// trace manifest; events are transitively bound through `trace_id` and the
/// manifest's hash-chain root.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TraceIdentity {
    pub run_id: String,
    pub task_id: String,
    pub candidate_id: String,
    pub manifest_hash: String,
    pub agent_artifact_hash: String,
    pub repository_snapshot_hash: String,
    pub runtime_descriptor_hash: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TraceSummary {
    pub mode: TraceMode,
    pub trace_schema: u16,
    pub trace_id: String,
    pub event_count: u64,
    pub root_hash: String,
    pub complete: bool,
    pub fully_observed: bool,
    pub candidate_evidence: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<TraceIdentity>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TraceManifest {
    pub trace_schema: u16,
    pub trace_id: String,
    pub mode: TraceMode,
    pub event_count: u64,
    pub root_hash: String,
    pub complete: bool,
    pub fully_observed: bool,
    pub total_bytes: u64,
    pub blobs: Vec<BlobRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity: Option<TraceIdentity>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Event {
    pub trace_schema: u16,
    pub trace_id: String,
    pub sequence: u64,
    pub timestamp_unix_ms: u64,
    pub elapsed_ms: u64,
    pub kind: String,
    pub stage: String,
    pub attempt: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub causation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation: Option<String>,
    pub previous_hash: String,
    pub data: Value,
    pub event_hash: String,
}

#[derive(Serialize)]
struct EventHashMaterial<'a> {
    trace_schema: u16,
    trace_id: &'a str,
    sequence: u64,
    timestamp_unix_ms: u64,
    elapsed_ms: u64,
    kind: &'a str,
    stage: &'a str,
    attempt: u32,
    causation: &'a Option<String>,
    correlation: &'a Option<String>,
    previous_hash: &'a str,
    data: &'a Value,
}

pub struct TraceWriter {
    dir: PathBuf,
    trace_id: String,
    mode: TraceMode,
    events: File,
    started: Instant,
    sequence: u64,
    root_hash: String,
    total_bytes: u64,
    max_bytes: u64,
    blobs: BTreeMap<String, BlobRef>,
    fully_observed: bool,
    complete: bool,
    identity: Option<TraceIdentity>,
}

impl TraceWriter {
    /// Create a trace at an exact directory. The caller is responsible for
    /// choosing a fresh directory; existing content is rejected.
    pub fn create(dir: impl AsRef<Path>, mode: TraceMode, max_bytes: u64) -> Result<Self> {
        Self::create_with_identity(dir, mode, max_bytes, None)
    }

    /// Create a trace bound to worker-provided run and candidate provenance.
    /// Managed traces require this constructor; local traces may remain
    /// anonymous for backwards compatibility.
    pub fn create_bound(
        dir: impl AsRef<Path>,
        mode: TraceMode,
        max_bytes: u64,
        identity: TraceIdentity,
    ) -> Result<Self> {
        Self::create_with_identity(dir, mode, max_bytes, Some(identity))
    }

    fn create_with_identity(
        dir: impl AsRef<Path>,
        mode: TraceMode,
        max_bytes: u64,
        identity: Option<TraceIdentity>,
    ) -> Result<Self> {
        ensure!(max_bytes > 0, "RSI trace byte limit must be positive");
        if mode == TraceMode::Managed {
            ensure!(
                identity.is_some(),
                "managed RSI trace requires run provenance"
            );
        }
        if let Some(identity) = &identity {
            validate_identity(identity)?;
        }
        let dir = dir.as_ref().to_path_buf();
        if dir.exists() {
            ensure!(
                fs::read_dir(&dir)?.next().is_none(),
                "RSI trace directory is not empty"
            );
        } else {
            create_private_dir(&dir)?;
        }
        create_private_dir(&dir.join("blobs"))?;
        let trace_id = dir
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| {
                name.len() == 32
                    && name
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            })
            .map(str::to_owned)
            .unwrap_or_else(new_trace_id);
        let events_path = dir.join("events.jsonl");
        let events = private_new_file(&events_path)?;
        let mut writer = Self {
            dir,
            trace_id,
            mode,
            events,
            started: Instant::now(),
            sequence: 0,
            root_hash: ZERO_HASH.into(),
            total_bytes: 0,
            max_bytes,
            blobs: BTreeMap::new(),
            fully_observed: true,
            complete: false,
            identity,
        };
        writer.write_manifest(false)?;
        Ok(writer)
    }

    /// Create a fresh local trace beneath `$XDG_STATE_HOME/hi/rsi`.
    pub fn create_local(state_home: &Path, max_bytes: u64) -> Result<Self> {
        let root = state_home.join("hi/rsi");
        create_private_dir(&root)?;
        prune(&root, DEFAULT_RETENTION_DAYS, DEFAULT_GLOBAL_MAX_BYTES)?;
        let id = new_trace_id();
        Self::create(root.join(id), TraceMode::Local, max_bytes)
    }

    pub fn directory(&self) -> &Path {
        &self.dir
    }
    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }

    pub fn put_blob(&mut self, body: &[u8], media_type: impl Into<String>) -> Result<BlobRef> {
        // Sanitize secrets from blob content. Text blobs (JSON, prompts, tool
        // output) are scanned and replaced; binary blobs pass through unchanged
        // when they aren't valid UTF-8.
        let sanitized: Vec<u8>;
        let body: &[u8] = if let Ok(text) = std::str::from_utf8(body) {
            let redacted = hi_secrets::redact_secrets(text);
            sanitized = redacted.into_owned().into_bytes();
            &sanitized
        } else {
            body
        };
        let hash = blake3::hash(body).to_hex().to_string();
        if let Some(existing) = self.blobs.get(&hash) {
            return Ok(existing.clone());
        }
        let projected = self.total_bytes.saturating_add(body.len() as u64);
        ensure!(
            projected <= self.max_bytes,
            "RSI trace exceeded its per-run byte limit"
        );
        let path = self.dir.join("blobs").join(&hash);
        atomic_private_write(&path, body)?;
        let blob = BlobRef {
            hash: hash.clone(),
            size_bytes: body.len() as u64,
            media_type: media_type.into(),
        };
        self.total_bytes = projected;
        self.blobs.insert(hash, blob.clone());
        Ok(blob)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        kind: impl Into<String>,
        stage: impl Into<String>,
        attempt: u32,
        causation: Option<String>,
        correlation: Option<String>,
        mut data: Value,
    ) -> Result<String> {
        ensure!(!self.complete, "cannot append to a completed RSI trace");
        // Sanitize secrets from trace event data before it is persisted to disk
        // or uploaded — tool output, prompts, and model messages can carry API
        // keys, tokens, and credentials that must never leave the local machine.
        hi_secrets::redact_json_string_values(&mut data);
        let kind = kind.into();
        let stage = stage.into();
        validate_label(&kind, "event kind")?;
        validate_label(&stage, "event stage")?;
        let causation = Some(causation.unwrap_or_else(|| self.root_hash.clone()));
        let correlation =
            Some(correlation.unwrap_or_else(|| format!("event-{}", self.sequence + 1)));
        let sequence = self.sequence + 1;
        let timestamp_unix_ms = unix_ms()?;
        let elapsed_ms = self.started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let material = EventHashMaterial {
            trace_schema: TRACE_SCHEMA_VERSION,
            trace_id: &self.trace_id,
            sequence,
            timestamp_unix_ms,
            elapsed_ms,
            kind: &kind,
            stage: &stage,
            attempt,
            causation: &causation,
            correlation: &correlation,
            previous_hash: &self.root_hash,
            data: &data,
        };
        let event_hash = blake3::hash(&serde_json::to_vec(&material)?)
            .to_hex()
            .to_string();
        let event = Event {
            trace_schema: TRACE_SCHEMA_VERSION,
            trace_id: self.trace_id.clone(),
            sequence,
            timestamp_unix_ms,
            elapsed_ms,
            kind,
            stage,
            attempt,
            causation,
            correlation,
            previous_hash: self.root_hash.clone(),
            data,
            event_hash: event_hash.clone(),
        };
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');
        ensure!(
            self.total_bytes.saturating_add(line.len() as u64) <= self.max_bytes,
            "RSI trace exceeded its per-run byte limit"
        );
        self.events.write_all(&line)?;
        self.events.sync_data()?;
        self.total_bytes += line.len() as u64;
        self.sequence = sequence;
        self.root_hash = event_hash.clone();
        Ok(event_hash)
    }

    pub fn mark_unobserved(&mut self) {
        self.fully_observed = false;
    }

    /// Persist the latest incomplete state after observation has been disabled.
    /// This is best-effort recovery metadata, never a successful trace.
    pub fn abandon(&mut self) -> Result<()> {
        self.fully_observed = false;
        self.complete = false;
        self.events.sync_all()?;
        self.write_manifest(false)?;
        sync_dir(&self.dir)
    }

    pub fn finalize(mut self) -> Result<TraceSummary> {
        self.complete = true;
        self.events.sync_all()?;
        self.write_manifest(true)?;
        sync_dir(&self.dir)?;
        Ok(self.summary())
    }

    pub fn summary(&self) -> TraceSummary {
        TraceSummary {
            mode: self.mode,
            trace_schema: TRACE_SCHEMA_VERSION,
            trace_id: self.trace_id.clone(),
            event_count: self.sequence,
            root_hash: self.root_hash.clone(),
            complete: self.complete,
            fully_observed: self.fully_observed,
            candidate_evidence: true,
            identity: self.identity.clone(),
        }
    }

    fn write_manifest(&mut self, complete: bool) -> Result<()> {
        let manifest = TraceManifest {
            trace_schema: TRACE_SCHEMA_VERSION,
            trace_id: self.trace_id.clone(),
            mode: self.mode,
            event_count: self.sequence,
            root_hash: self.root_hash.clone(),
            complete,
            fully_observed: self.fully_observed,
            total_bytes: self.total_bytes,
            blobs: self.blobs.values().cloned().collect(),
            identity: self.identity.clone(),
        };
        atomic_private_write(
            &self.dir.join("manifest.json"),
            &serde_json::to_vec_pretty(&manifest)?,
        )
    }
}

pub fn validate_trace(dir: &Path, max_bytes: u64, max_entries: usize) -> Result<TraceManifest> {
    let dir_meta = fs::symlink_metadata(dir)?;
    ensure!(
        dir_meta.is_dir() && !dir_meta.file_type().is_symlink(),
        "trace root is not a directory"
    );
    let manifest_path = dir.join("manifest.json");
    let manifest: TraceManifest =
        serde_json::from_slice(&read_regular(&manifest_path, max_bytes)?)?;
    ensure!(
        manifest.trace_schema == TRACE_SCHEMA_VERSION,
        "unsupported trace schema"
    );
    if manifest.mode == TraceMode::Managed {
        validate_identity(
            manifest
                .identity
                .as_ref()
                .ok_or_else(|| anyhow!("managed trace has no run provenance"))?,
        )?;
    } else if let Some(identity) = &manifest.identity {
        validate_identity(identity)?;
    }
    ensure!(manifest.complete, "trace journal is incomplete");
    let allowed = BTreeSet::from(["manifest.json", "events.jsonl", "blobs"]);
    let mut entries = 0usize;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 trace path"))?;
        ensure!(allowed.contains(name), "unexpected trace entry {name}");
        entries += 1;
    }
    ensure!(entries <= max_entries, "trace has too many entries");
    let blobs_dir = dir.join("blobs");
    let blobs_meta = fs::symlink_metadata(&blobs_dir)?;
    ensure!(
        blobs_meta.is_dir() && !blobs_meta.file_type().is_symlink(),
        "trace blobs path is not a directory"
    );
    let expected = manifest
        .blobs
        .iter()
        .map(|blob| (blob.hash.clone(), blob))
        .collect::<BTreeMap<_, _>>();
    ensure!(
        expected.len() == manifest.blobs.len(),
        "trace manifest contains duplicate blobs"
    );
    let mut seen = BTreeSet::new();
    let mut actual_total = 0u64;
    for entry in fs::read_dir(&blobs_dir)? {
        let entry = entry?;
        entries += 1;
        ensure!(entries <= max_entries, "trace has too many entries");
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow!("non-UTF-8 blob path"))?;
        ensure!(is_hash(&name), "invalid trace blob name");
        let expected_blob = expected
            .get(&name)
            .ok_or_else(|| anyhow!("unreferenced trace blob"))?;
        let body = read_regular(&entry.path(), max_bytes.saturating_sub(actual_total))?;
        ensure!(
            body.len() as u64 == expected_blob.size_bytes,
            "trace blob size mismatch"
        );
        ensure!(
            blake3::hash(&body).to_hex().as_str() == name,
            "trace blob hash mismatch"
        );
        actual_total = actual_total
            .checked_add(body.len() as u64)
            .ok_or_else(|| anyhow!("trace size overflow"))?;
        ensure!(actual_total <= max_bytes, "trace exceeds byte limit");
        seen.insert(name);
    }
    ensure!(
        seen.len() == expected.len(),
        "trace manifest blob set mismatch"
    );

    let events_path = dir.join("events.jsonl");
    let event_bytes = read_regular(&events_path, max_bytes.saturating_sub(actual_total))?;
    ensure!(
        event_bytes.is_empty() || event_bytes.ends_with(b"\n"),
        "partial trace journal"
    );
    actual_total = actual_total.saturating_add(event_bytes.len() as u64);
    ensure!(actual_total <= max_bytes, "trace exceeds byte limit");
    ensure!(
        actual_total == manifest.total_bytes,
        "trace manifest byte count mismatch"
    );
    let mut previous = ZERO_HASH.to_string();
    let mut count = 0u64;
    for line in BufReader::new(event_bytes.as_slice()).lines() {
        let line = line?;
        ensure!(!line.is_empty(), "empty trace journal record");
        let event: Event = serde_json::from_str(&line)?;
        count += 1;
        ensure!(
            event.trace_schema == TRACE_SCHEMA_VERSION && event.trace_id == manifest.trace_id,
            "trace event identity mismatch"
        );
        validate_label(&event.kind, "event kind")?;
        validate_label(&event.stage, "event stage")?;
        ensure!(
            event
                .correlation
                .as_ref()
                .is_some_and(|value| !value.is_empty())
                && event.causation.as_ref().is_some_and(|value| is_hash(value)),
            "trace event metadata is incomplete"
        );
        ensure!(
            event.sequence == count && event.previous_hash == previous,
            "trace hash chain discontinuity"
        );
        let material = EventHashMaterial {
            trace_schema: event.trace_schema,
            trace_id: &event.trace_id,
            sequence: event.sequence,
            timestamp_unix_ms: event.timestamp_unix_ms,
            elapsed_ms: event.elapsed_ms,
            kind: &event.kind,
            stage: &event.stage,
            attempt: event.attempt,
            causation: &event.causation,
            correlation: &event.correlation,
            previous_hash: &event.previous_hash,
            data: &event.data,
        };
        let hash = blake3::hash(&serde_json::to_vec(&material)?)
            .to_hex()
            .to_string();
        ensure!(hash == event.event_hash, "trace event hash mismatch");
        previous = hash;
    }
    ensure!(
        count == manifest.event_count && previous == manifest.root_hash,
        "trace manifest does not match its journal"
    );
    Ok(manifest)
}

/// Remove only completed local traces, oldest first. Active/incomplete traces
/// are never selected, even when the global cap is exceeded.
pub fn prune(root: &Path, retention_days: u64, global_cap: u64) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(retention_days.saturating_mul(86_400)))
        .unwrap_or(UNIX_EPOCH);
    let mut completed = Vec::new();
    let mut total = 0u64;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let meta = fs::symlink_metadata(entry.path())?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            continue;
        }
        let manifest_path = entry.path().join("manifest.json");
        let Ok(raw) = read_regular(&manifest_path, 4 * 1024 * 1024) else {
            continue;
        };
        let Ok(manifest) = serde_json::from_slice::<TraceManifest>(&raw) else {
            continue;
        };
        if !manifest.complete {
            continue;
        }
        let modified = meta.modified().unwrap_or(UNIX_EPOCH);
        total = total.saturating_add(manifest.total_bytes);
        completed.push((modified, entry.path(), manifest.total_bytes));
    }
    completed.sort_by_key(|item| item.0);
    for (modified, path, bytes) in completed {
        if modified < cutoff || total > global_cap {
            fs::remove_dir_all(path)?;
            total = total.saturating_sub(bytes);
        }
    }
    Ok(())
}

fn read_regular(path: &Path, max: u64) -> Result<Vec<u8>> {
    let path_meta =
        fs::symlink_metadata(path).with_context(|| format!("inspecting {}", path.display()))?;
    ensure!(
        path_meta.is_file() && !path_meta.file_type().is_symlink(),
        "trace entry is not a regular file"
    );
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let meta = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(
            path_meta.dev() == meta.dev() && path_meta.ino() == meta.ino(),
            "trace entry changed before open"
        );
        ensure!(meta.nlink() == 1, "linked trace file rejected");
    }
    ensure!(meta.len() <= max, "trace entry exceeds byte limit");
    let mut body = Vec::with_capacity(meta.len() as usize);
    use std::io::Read as _;
    file.take(max + 1).read_to_end(&mut body)?;
    ensure!(
        body.len() as u64 == meta.len(),
        "trace entry changed while reading"
    );
    Ok(body)
}

fn validate_label(value: &str, label: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && value.len() <= 128 && !value.contains(['\0', '\r', '\n']),
        "invalid {label}"
    );
    Ok(())
}
fn validate_identity(identity: &TraceIdentity) -> Result<()> {
    for (label, value) in [
        ("run id", identity.run_id.as_str()),
        ("task id", identity.task_id.as_str()),
        ("candidate id", identity.candidate_id.as_str()),
    ] {
        validate_label(value, label)?;
    }
    for (label, value) in [
        ("manifest hash", &identity.manifest_hash),
        ("agent artifact hash", &identity.agent_artifact_hash),
        (
            "repository snapshot hash",
            &identity.repository_snapshot_hash,
        ),
        ("runtime descriptor hash", &identity.runtime_descriptor_hash),
    ] {
        ensure!(is_hash(value), "invalid trace {label}");
    }
    Ok(())
}
fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn unix_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}
fn new_trace_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = TRACE_COUNTER.fetch_add(1, Ordering::Relaxed);
    blake3::hash(format!("{}:{now}:{counter}", std::process::id()).as_bytes()).to_hex()[..32]
        .to_string()
}
fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
fn private_new_file(path: &Path) -> Result<File> {
    let mut opts = OpenOptions::new();
    opts.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    Ok(opts.open(path)?)
}
fn atomic_private_write(path: &Path, body: &[u8]) -> Result<()> {
    let name = path
        .file_name()
        .ok_or_else(|| anyhow!("trace path has no file name"))?
        .to_string_lossy();
    let tmp = path.with_file_name(format!(
        ".{name}.tmp-{}",
        TRACE_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = private_new_file(&tmp)?;
    file.write_all(body)?;
    file.sync_all()?;
    match fs::rename(&tmp, path) {
        Ok(()) => {}
        Err(error) if path.exists() => {
            let _ = fs::remove_file(&tmp);
            if fs::read(path)? != body {
                return Err(error.into());
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&tmp);
            return Err(error.into());
        }
    }
    if let Some(parent) = path.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}
fn sync_dir(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn temp() -> PathBuf {
        std::env::temp_dir().join(format!("hi-trace-{}", new_trace_id()))
    }

    #[test]
    fn round_trip_hash_chain_and_cas() {
        let dir = temp();
        let mut trace =
            TraceWriter::create_bound(&dir, TraceMode::Managed, 1024 * 1024, identity()).unwrap();
        let body = trace.put_blob(b"secret prompt", "text/plain").unwrap();
        trace
            .record(
                "model_request",
                "model",
                1,
                None,
                Some("turn-1".into()),
                serde_json::json!({"body": body}),
            )
            .unwrap();
        let summary = trace.finalize().unwrap();
        let manifest = validate_trace(&dir, 1024 * 1024, 20).unwrap();
        assert_eq!(manifest.root_hash, summary.root_hash);
        assert!(
            !std::fs::read_to_string(dir.join("events.jsonl"))
                .unwrap()
                .contains("secret prompt")
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(dir.join("events.jsonl"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn detects_tampering_and_partial_journal() {
        let dir = temp();
        let mut trace =
            TraceWriter::create_bound(&dir, TraceMode::Managed, 1024 * 1024, identity()).unwrap();
        trace
            .record("terminal", "terminal", 1, None, None, serde_json::json!({}))
            .unwrap();
        trace.finalize().unwrap();
        let mut raw = fs::read(dir.join("events.jsonl")).unwrap();
        raw.pop();
        fs::write(dir.join("events.jsonl"), raw).unwrap();
        assert!(validate_trace(&dir, 1024 * 1024, 20).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn pruning_never_removes_an_active_trace() {
        let root = temp();
        create_private_dir(&root).unwrap();
        let active_dir = root.join(new_trace_id());
        let active = TraceWriter::create(&active_dir, TraceMode::Local, 1024 * 1024).unwrap();
        let completed_dir = root.join(new_trace_id());
        let mut completed =
            TraceWriter::create(&completed_dir, TraceMode::Local, 1024 * 1024).unwrap();
        completed
            .record("terminal", "terminal", 1, None, None, serde_json::json!({}))
            .unwrap();
        completed.finalize().unwrap();
        prune(&root, 30, 0).unwrap();
        assert!(active_dir.exists());
        assert!(!completed_dir.exists());
        drop(active);
        fs::remove_dir_all(root).unwrap();
    }

    fn identity() -> TraceIdentity {
        TraceIdentity {
            run_id: "run-1".into(),
            task_id: "task-1".into(),
            candidate_id: "candidate-1".into(),
            manifest_hash: "1".repeat(64),
            agent_artifact_hash: "2".repeat(64),
            repository_snapshot_hash: "3".repeat(64),
            runtime_descriptor_hash: "4".repeat(64),
        }
    }

    #[test]
    fn managed_trace_requires_and_preserves_provenance() {
        let dir = temp();
        assert!(TraceWriter::create(&dir, TraceMode::Managed, 1024).is_err());
        let dir = temp();
        let expected = identity();
        let trace =
            TraceWriter::create_bound(&dir, TraceMode::Managed, 1024 * 1024, expected.clone())
                .unwrap();
        let summary = trace.finalize().unwrap();
        assert_eq!(summary.identity, Some(expected.clone()));
        assert_eq!(
            validate_trace(&dir, 1024 * 1024, 20).unwrap().identity,
            Some(expected)
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
