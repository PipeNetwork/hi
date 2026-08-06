//! Shared announcement types, persistence, and formatting for `hi` CLI apps.
//!
//! Provides the wire types for remote announcements (e.g. release notes, tips,
//! important changes surfaced to users on startup), plus persistence for
//! dismissed/hidden announcement IDs and filtering of expired entries.
//!
//! Inspired by grok-build's `xai-grok-announcements` crate.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn run() -> anyhow::Result<()> {
//! use hi_announcements::{RemoteAnnouncement, read_hidden_announcement_ids};
//!
//! let hidden = read_hidden_announcement_ids("~/.hi").await?;
//! let ann = RemoteAnnouncement {
//!     id: Some("release-0.3".to_string()),
//!     message: Some("hi 0.3 is out!".to_string()),
//!     ..Default::default()
//! };
//! assert!(!hi_announcements::visible_announcements(&[ann]).is_empty());
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use reqwest::header::{ETAG, IF_NONE_MATCH};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

static HIDDEN_IDS_LOCK: Mutex<()> = Mutex::const_new(());
static CACHE_LOCK: Mutex<()> = Mutex::const_new(());
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub const MAX_ANNOUNCEMENTS: usize = 32;
pub const MAX_RESPONSE_BYTES: usize = 256 * 1024;
pub const MAX_ID_CHARS: usize = 128;
pub const MAX_TITLE_CHARS: usize = 256;
pub const MAX_MESSAGE_CHARS: usize = 8 * 1024;
pub const MAX_CTA_LABEL_CHARS: usize = 128;
pub const MAX_CTA_CAPTION_CHARS: usize = 512;
pub const MAX_CTA_URL_CHARS: usize = 2048;

/// Trusted transport configuration. Plain HTTP is rejected unless explicitly
/// enabled by a caller, which is intended only for local tests.
#[derive(Debug, Clone)]
pub struct AnnouncementEndpointConfig {
    pub endpoint: String,
    pub timeout: Duration,
    pub allow_http: bool,
}

impl AnnouncementEndpointConfig {
    pub fn https(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            timeout: Duration::from_secs(2),
            allow_http: false,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AnnouncementCache {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    etag: Option<String>,
    #[serde(default)]
    generation: u64,
    #[serde(default)]
    announcements: Vec<RemoteAnnouncement>,
}

/// A call-to-action link attached to an announcement.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnnouncementCta {
    /// The display label for the link (e.g. `"Read more"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// The URL to open when the user activates the CTA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Optional caption/tooltip text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
}

/// Severity level for an announcement.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnnouncementSeverity {
    /// Low-priority informational message.
    #[default]
    Info,
    /// Something the user should pay attention to.
    Warning,
    /// Critical: action required.
    Critical,
}

/// A remote announcement fetched from a server or override.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteAnnouncement {
    /// Unique identifier. If absent, a content-based key is derived.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The announcement body text. Entries with empty/absent messages are
    /// filtered out by [`visible_announcements`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Severity level.
    #[serde(default)]
    pub severity: AnnouncementSeverity,
    /// Optional title/header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Optional call-to-action link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cta: Option<AnnouncementCta>,
    /// Unix timestamp (seconds) of last update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<u64>,
    /// Unix timestamp (seconds) after which the announcement is expired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    /// Whether the user can dismiss the announcement.
    #[serde(default = "default_true")]
    pub dismissible: bool,
    /// Whether the announcement persists across sessions (not auto-dismissed).
    #[serde(default)]
    pub persistent: bool,
}

fn default_true() -> bool {
    true
}

// Manual impl so `..Default::default()` matches the wire defaults: the derive
// would give `dismissible: false`, silently making every default-constructed
// announcement undismissable while deserialized ones default to dismissible.
impl Default for RemoteAnnouncement {
    fn default() -> Self {
        Self {
            id: None,
            message: None,
            severity: AnnouncementSeverity::default(),
            title: None,
            cta: None,
            updated_at: None,
            expires_at: None,
            dismissible: true,
            persistent: false,
        }
    }
}

/// Notification that announcements have been refreshed from a remote source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnnouncementsRefreshed {
    /// Monotonically increasing generation number.
    #[serde(rename = "gen")]
    pub r#gen: u64,
    /// The full set of announcements from the refresh.
    pub announcements: Vec<RemoteAnnouncement>,
}

fn validate_text(value: &Option<String>, name: &str, max: usize) -> Result<()> {
    let Some(value) = value else { return Ok(()) };
    if value.chars().count() > max {
        bail!("announcement {name} exceeds {max} characters");
    }
    // This text reaches raw terminals: reject escape/control sequences (OSC
    // clipboard writes, CSI cursor moves, carriage-return line rewrites) from
    // the untrusted feed. `is_control` covers C0, DEL, and C1.
    if value
        .chars()
        .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        bail!("announcement {name} contains control characters");
    }
    Ok(())
}

/// Cap the blast radius of a compromised (or sloppy) feed: a non-dismissible
/// banner with no expiry would have no in-product removal path at all, so it
/// is downgraded to dismissible rather than rejected — rejecting would fail
/// the whole feed (and discard the cache) over one authoring slip.
pub fn normalize_announcements(
    mut announcements: Vec<RemoteAnnouncement>,
) -> Vec<RemoteAnnouncement> {
    for announcement in &mut announcements {
        if !announcement.dismissible && announcement.expires_at.is_none() {
            announcement.dismissible = true;
        }
    }
    announcements
}

/// Validate all untrusted wire fields before rendering or storage.
pub fn validate_announcements(announcements: &[RemoteAnnouncement]) -> Result<()> {
    let result = validate_announcements_inner(announcements);
    if result.is_err() {
        hi_observability::record(hi_observability::ReliabilityEvent::AnnouncementValidationFailure);
    }
    result
}

fn validate_announcements_inner(announcements: &[RemoteAnnouncement]) -> Result<()> {
    if announcements.len() > MAX_ANNOUNCEMENTS {
        bail!("announcement count exceeds {MAX_ANNOUNCEMENTS}");
    }
    for announcement in announcements {
        validate_text(&announcement.id, "id", MAX_ID_CHARS)?;
        validate_text(&announcement.title, "title", MAX_TITLE_CHARS)?;
        validate_text(&announcement.message, "message", MAX_MESSAGE_CHARS)?;
        if let Some(cta) = &announcement.cta {
            validate_text(&cta.label, "CTA label", MAX_CTA_LABEL_CHARS)?;
            validate_text(&cta.caption, "CTA caption", MAX_CTA_CAPTION_CHARS)?;
            validate_text(&cta.url, "CTA URL", MAX_CTA_URL_CHARS)?;
            if let Some(raw) = &cta.url {
                let url = url::Url::parse(raw).context("invalid announcement CTA URL")?;
                if url.scheme() != "https" || url.host_str().is_none() {
                    bail!("announcement CTA URL must be HTTPS");
                }
            }
        }
    }
    Ok(())
}

fn cache_file_path(hi_home: impl AsRef<Path>) -> PathBuf {
    hi_home.as_ref().join("announcements-cache.json")
}

async fn read_cache(hi_home: &Path) -> AnnouncementCache {
    match tokio::fs::read(cache_file_path(hi_home)).await {
        Ok(bytes) => match serde_json::from_slice::<AnnouncementCache>(&bytes) {
            Ok(mut cache) if validate_announcements(&cache.announcements).is_ok() => {
                // A tampered cache holding a non-header-safe etag would poison
                // the conditional request builder on every fetch — and only a
                // successful fetch replaces the etag. Drop it instead.
                if cache
                    .etag
                    .as_deref()
                    .is_some_and(|etag| reqwest::header::HeaderValue::from_str(etag).is_err())
                {
                    cache.etag = None;
                }
                cache.announcements = normalize_announcements(cache.announcements);
                cache
            }
            _ => {
                hi_observability::record(
                    hi_observability::ReliabilityEvent::AnnouncementCacheFailure,
                );
                AnnouncementCache::default()
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => AnnouncementCache::default(),
        Err(_) => {
            hi_observability::record(hi_observability::ReliabilityEvent::AnnouncementCacheFailure);
            AnnouncementCache::default()
        }
    }
}

async fn write_cache(hi_home: &Path, cache: &AnnouncementCache) -> Result<()> {
    let path = cache_file_path(hi_home);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let temp = hidden_temp_file_path(&path);
    let data = serde_json::to_vec(cache)?;
    if let Err(error) = write_private_temp_file(&temp, &data).await {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(error);
    }
    if let Err(error) = tokio::fs::rename(&temp, &path).await {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(error.into());
    }
    sync_parent_directory(&path).await
}

/// Fetch, validate, and atomically cache announcements. A 304 reuses the cache;
/// a changed successful response increments the local generation.
pub async fn fetch_announcements(
    hi_home: impl AsRef<Path>,
    config: &AnnouncementEndpointConfig,
) -> Result<AnnouncementsRefreshed> {
    let result = fetch_announcements_inner(hi_home.as_ref(), config).await;
    if result.is_err() {
        hi_observability::record(hi_observability::ReliabilityEvent::AnnouncementFetchFailure);
    }
    result
}

async fn fetch_announcements_inner(
    hi_home: &Path,
    config: &AnnouncementEndpointConfig,
) -> Result<AnnouncementsRefreshed> {
    let endpoint = url::Url::parse(&config.endpoint).context("invalid announcement endpoint")?;
    if endpoint.host_str().is_none()
        || (endpoint.scheme() != "https" && !(config.allow_http && endpoint.scheme() == "http"))
    {
        bail!("announcement endpoint must be HTTPS");
    }
    let home = hi_home;
    let _guard = CACHE_LOCK.lock().await;
    let mut cache = read_cache(home).await;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let mut request = client.get(endpoint);
    if let Some(etag) = &cache.etag {
        request = request.header(IF_NONE_MATCH, etag);
    }
    let response = tokio::time::timeout(config.timeout, async {
        let response = request.send().await?;
        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            return Ok::<_, anyhow::Error>((None, None));
        }
        if !response.status().is_success() {
            bail!("announcement endpoint returned {}", response.status());
        }
        let etag = response
            .headers()
            .get(ETAG)
            .map(|value| value.to_str().map(str::to_owned))
            .transpose()?;
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                bail!("announcement response exceeds {MAX_RESPONSE_BYTES} bytes");
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok((Some(bytes), etag))
    })
    .await
    .map_err(|_| anyhow!("announcement request timed out"))??;

    if let (Some(bytes), etag) = response {
        let announcements: Vec<RemoteAnnouncement> = serde_json::from_slice(&bytes)?;
        validate_announcements(&announcements)?;
        let announcements = normalize_announcements(announcements);
        cache.announcements = announcements;
        cache.etag = etag;
        cache.generation = cache.generation.saturating_add(1);
        write_cache(home, &cache).await?;
    }
    Ok(AnnouncementsRefreshed {
        r#gen: cache.generation,
        announcements: cache.announcements,
    })
}

// ---------------------------------------------------------------------------
// Hidden/dismissed ID persistence
// ---------------------------------------------------------------------------

/// Derive the hide key for an announcement: the `id` if present, otherwise a
/// content-based fallback using the message text.
pub fn announcement_hide_key(a: &RemoteAnnouncement) -> String {
    if let Some(id) = &a.id
        && !id.is_empty()
    {
        return id.clone();
    }
    let msg = a.message.as_deref().unwrap_or("");
    format!("content-sha256:{:x}", Sha256::digest(msg.as_bytes()))
}

/// Parse persisted hidden IDs. JSON arrays are the canonical format; legacy
/// comma-separated contents remain readable for migration compatibility.
pub fn parse_hidden_announcement_ids(s: &str) -> BTreeSet<String> {
    if let Ok(ids) = serde_json::from_str::<Vec<String>>(s) {
        return ids
            .into_iter()
            .filter(|id| !id.is_empty())
            .map(migrate_legacy_hide_key)
            .collect();
    }
    s.split(',')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .map(migrate_legacy_hide_key)
        .collect()
}

/// Earlier releases stored content-based hide keys as `content\x1f<message>`;
/// map them onto the current hash form so upgrading keeps previously dismissed
/// announcements hidden.
fn migrate_legacy_hide_key(key: String) -> String {
    match key.strip_prefix("content\x1f") {
        Some(message) => format!("content-sha256:{:x}", Sha256::digest(message.as_bytes())),
        None => key,
    }
}

/// Serialize hidden IDs as a JSON array. Returns `None` if the set is empty.
pub fn serialize_hidden_announcement_ids(ids: &BTreeSet<String>) -> Option<String> {
    (!ids.is_empty()).then(|| serde_json::to_string(ids).expect("serializing strings cannot fail"))
}

/// Remove hidden IDs that no longer appear in the active set. Returns `true`
/// if any IDs were pruned.
pub fn prune_hidden_announcement_ids(
    hidden: &mut BTreeSet<String>,
    active: &[&RemoteAnnouncement],
) -> bool {
    let active_keys: BTreeSet<String> = active.iter().map(|a| announcement_hide_key(a)).collect();
    let before = hidden.len();
    hidden.retain(|k| active_keys.contains(k));
    hidden.len() != before
}

/// Path to the hidden-announcements file within a hi home directory.
fn hidden_file_path(hi_home: impl AsRef<Path>) -> PathBuf {
    hi_home.as_ref().join("announcements.json")
}

fn hidden_temp_file_path(path: &Path) -> PathBuf {
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("json.tmp-{}-{nonce}", std::process::id()))
}

/// Read the set of hidden/dismissed announcement IDs from disk.
/// Returns an empty set if the file doesn't exist.
pub async fn read_hidden_announcement_ids(hi_home: impl AsRef<Path>) -> Result<BTreeSet<String>> {
    let path = hidden_file_path(hi_home);
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let data = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(parse_hidden_announcement_ids(&data))
}

/// Write the set of hidden/dismissed announcement IDs to disk.
/// If the set is empty, the file is removed.
pub async fn write_hidden_announcement_ids(
    hi_home: impl AsRef<Path>,
    ids: &BTreeSet<String>,
) -> Result<()> {
    let path = hidden_file_path(hi_home);
    if ids.is_empty() {
        let _ = tokio::fs::remove_file(&path).await;
        return Ok(());
    }
    let data = serialize_hidden_announcement_ids(ids).unwrap_or_default();
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("creating {}", parent.display()))?;
    }

    let temp = hidden_temp_file_path(&path);
    let write_result = write_private_temp_file(&temp, data.as_bytes()).await;
    if let Err(error) = write_result {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(error);
    }
    if let Err(error) = tokio::fs::rename(&temp, &path).await {
        let _ = tokio::fs::remove_file(&temp).await;
        return Err(error).with_context(|| format!("replacing {}", path.display()));
    }
    sync_parent_directory(&path).await?;
    Ok(())
}

async fn sync_parent_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        tokio::fs::File::open(parent)
            .await
            .with_context(|| format!("opening {}", parent.display()))?
            .sync_all()
            .await
            .with_context(|| format!("syncing {}", parent.display()))?;
    }
    Ok(())
}

/// Atomically read, mutate, and durably persist hidden IDs, serialized against
/// concurrent tasks in this process (mutex) and other `hi` processes (advisory
/// file lock). All dismiss/prune operations should use this API: the atomic
/// rename in the writer prevents corruption but not lost updates.
pub async fn mutate_hidden_announcement_ids<T>(
    hi_home: impl AsRef<Path>,
    mutate: impl FnOnce(&mut BTreeSet<String>) -> T,
) -> Result<T> {
    let _guard = HIDDEN_IDS_LOCK.lock().await;
    let home = hi_home.as_ref();
    let _file_lock = lock_hidden_ids_file(home.to_path_buf()).await?;
    let mut ids = read_hidden_announcement_ids(home).await?;
    let result = mutate(&mut ids);
    write_hidden_announcement_ids(home, &ids).await?;
    Ok(result)
}

/// Take the cross-process lock for the hidden-IDs read-modify-write. Blocking
/// acquisition runs on the blocking pool; the OS releases the lock when the
/// returned handle drops (or the process dies).
async fn lock_hidden_ids_file(hi_home: PathBuf) -> Result<std::fs::File> {
    tokio::task::spawn_blocking(move || {
        let path = hidden_file_path(&hi_home).with_extension("lock");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        // Bounded acquisition: the OS only auto-releases the lock when the
        // holder dies, so a wedged (e.g. SIGSTOPped) hi process must not hang
        // another one's exit path or dismissal forever.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(file),
                Err(std::fs::TryLockError::WouldBlock) => {
                    if std::time::Instant::now() >= deadline {
                        bail!(
                            "timed out waiting for {} (held by another hi process)",
                            path.display()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(std::fs::TryLockError::Error(error)) => {
                    return Err(error).with_context(|| format!("locking {}", path.display()));
                }
            }
        }
    })
    .await
    .context("acquiring hidden-ids file lock")?
}

async fn write_private_temp_file(path: &Path, data: &[u8]) -> Result<()> {
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .await
        .with_context(|| format!("creating {}", path.display()))?;
    use tokio::io::AsyncWriteExt;
    file.write_all(data)
        .await
        .with_context(|| format!("writing {}", path.display()))?;
    file.sync_all()
        .await
        .with_context(|| format!("syncing {}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

/// Filter to announcements that have a non-empty message.
pub fn visible_announcements(announcements: &[RemoteAnnouncement]) -> Vec<&RemoteAnnouncement> {
    announcements
        .iter()
        .filter(|a| a.message.as_deref().is_some_and(|m| !m.trim().is_empty()))
        .collect()
}

/// Whether an announcement is expired at the given Unix timestamp.
pub fn is_expired_at(a: &RemoteAnnouncement, now: u64) -> bool {
    a.expires_at.is_some_and(|exp| exp <= now)
}

/// Filter out expired announcements, using the current system time.
pub fn filter_expired(
    announcements: impl IntoIterator<Item = RemoteAnnouncement>,
) -> Vec<RemoteAnnouncement> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    filter_expired_at(announcements, now)
}

/// Filter out expired announcements at a specific timestamp.
pub fn filter_expired_at(
    announcements: impl IntoIterator<Item = RemoteAnnouncement>,
    now: u64,
) -> Vec<RemoteAnnouncement> {
    announcements
        .into_iter()
        .filter(|a| !is_expired_at(a, now))
        .collect()
}

/// Resolve startup announcements from a remote fetch result.
///
/// Honors the `HI_ANNOUNCEMENTS_OVERRIDE` environment variable: if set to
/// valid JSON, it replaces the remote result entirely (for testing/dev).
pub fn resolve_startup_announcements(
    remote: Result<Vec<RemoteAnnouncement>>,
) -> Option<Vec<RemoteAnnouncement>> {
    let override_str = std::env::var("HI_ANNOUNCEMENTS_OVERRIDE").ok();
    resolve_startup_announcements_with_override(remote, override_str.as_deref())
}

/// Like [`resolve_startup_announcements`] but takes the override string
/// explicitly, for testability without env-var races.
pub fn resolve_startup_announcements_with_override(
    remote: Result<Vec<RemoteAnnouncement>>,
    override_str: Option<&str>,
) -> Option<Vec<RemoteAnnouncement>> {
    if let Some(s) = override_str
        && let Ok(parsed) = serde_json::from_str::<Vec<RemoteAnnouncement>>(s)
    {
        return Some(normalize_announcements(parsed));
    }
    match remote {
        Ok(anns) if !anns.is_empty() => Some(anns),
        Ok(_) => None,
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_failure_snapshot_does_not_capture_message() {
        let secret = "FORBIDDEN_ANNOUNCEMENT_TEXT";
        let before = hi_observability::snapshot().announcement_validation_failures;
        let item = RemoteAnnouncement {
            message: Some(format!("{secret}{}", "x".repeat(MAX_MESSAGE_CHARS))),
            ..Default::default()
        };
        let error = validate_announcements(&[item]).unwrap_err().to_string();
        assert_eq!(
            hi_observability::snapshot().announcement_validation_failures,
            before + 1
        );
        assert!(!error.contains(secret));
        assert!(!format!("{:?}", hi_observability::snapshot()).contains(secret));
    }

    fn ann(id: &str, msg: &str) -> RemoteAnnouncement {
        RemoteAnnouncement {
            id: Some(id.to_string()),
            message: Some(msg.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn hide_key_uses_id_when_present() {
        let a = ann("release-1", "hi 1.0 is out");
        assert_eq!(announcement_hide_key(&a), "release-1");
    }

    #[test]
    fn hide_key_falls_back_to_content() {
        let a = RemoteAnnouncement {
            id: None,
            message: Some("hello".to_string()),
            ..Default::default()
        };
        assert!(announcement_hide_key(&a).starts_with("content-sha256:"));
        assert_eq!(
            announcement_hide_key(&a).len(),
            "content-sha256:".len() + 64
        );
    }

    #[test]
    fn hide_key_falls_back_when_id_empty() {
        let a = RemoteAnnouncement {
            id: Some("".to_string()),
            message: Some("hello".to_string()),
            ..Default::default()
        };
        assert!(announcement_hide_key(&a).starts_with("content-sha256:"));
        assert_eq!(
            announcement_hide_key(&a).len(),
            "content-sha256:".len() + 64
        );
    }

    #[test]
    fn parse_and_serialize_roundtrip() {
        let mut ids = BTreeSet::new();
        ids.insert("a".to_string());
        ids.insert("b".to_string());
        ids.insert("c".to_string());
        let s = serialize_hidden_announcement_ids(&ids).unwrap();
        let back = parse_hidden_announcement_ids(&s);
        assert_eq!(ids, back);
    }

    #[test]
    fn json_roundtrip_preserves_commas() {
        let ids = BTreeSet::from(["a,b".to_string(), "plain".to_string()]);
        let serialized = serialize_hidden_announcement_ids(&ids).unwrap();
        assert!(serialized.starts_with('['));
        assert_eq!(parse_hidden_announcement_ids(&serialized), ids);
    }

    #[test]
    fn legacy_comma_separated_ids_are_migrated() {
        assert_eq!(
            parse_hidden_announcement_ids("old, keep"),
            BTreeSet::from(["keep".to_string(), "old".to_string()])
        );
    }

    #[test]
    fn legacy_content_keys_migrate_to_hash_form() {
        let announcement = RemoteAnnouncement {
            id: None,
            message: Some("hello".to_string()),
            ..Default::default()
        };
        let parsed = parse_hidden_announcement_ids("[\"content\\u001fhello\"]");
        assert!(parsed.contains(&announcement_hide_key(&announcement)));
    }

    #[test]
    fn non_dismissible_without_expiry_is_downgraded_to_dismissible() {
        let pinned = RemoteAnnouncement {
            id: Some("pinned".into()),
            message: Some("m".into()),
            dismissible: false,
            ..Default::default()
        };
        let normalized = normalize_announcements(vec![pinned.clone()]);
        assert!(normalized[0].dismissible, "no expiry → must be dismissible");
        let bounded = RemoteAnnouncement {
            expires_at: Some(u64::MAX),
            ..pinned
        };
        let normalized = normalize_announcements(vec![bounded]);
        assert!(!normalized[0].dismissible, "bounded banner may stay pinned");
    }

    // Uses the non-recording inner function: the observability counters are
    // process-global, and the snapshot test above asserts an exact delta.
    #[test]
    fn validation_rejects_control_characters_but_allows_newlines() {
        let escape = RemoteAnnouncement {
            id: Some("x".into()),
            message: Some("\u{1b}]52;c;payload\u{7}".into()),
            ..Default::default()
        };
        assert!(validate_announcements_inner(&[escape]).is_err());
        let carriage_return = RemoteAnnouncement {
            title: Some("spoof\rannouncement:".into()),
            message: Some("m".into()),
            ..Default::default()
        };
        assert!(validate_announcements_inner(&[carriage_return]).is_err());
        let multiline = RemoteAnnouncement {
            id: Some("x".into()),
            message: Some("line one\nline two\ttabbed".into()),
            ..Default::default()
        };
        assert!(validate_announcements_inner(&[multiline]).is_ok());
    }

    #[test]
    fn serialize_empty_returns_none() {
        let ids = BTreeSet::new();
        assert!(serialize_hidden_announcement_ids(&ids).is_none());
    }

    #[test]
    fn prune_removes_stale_ids() {
        let mut hidden = BTreeSet::from(["old".to_string(), "keep".to_string()]);
        let active = [ann("keep", "msg")];
        let active_refs: Vec<&RemoteAnnouncement> = active.iter().collect();
        assert!(prune_hidden_announcement_ids(&mut hidden, &active_refs));
        assert!(hidden.contains("keep"));
        assert!(!hidden.contains("old"));
    }

    #[test]
    fn prune_noop_when_all_active() {
        let mut hidden = BTreeSet::from(["a".to_string()]);
        let active = [ann("a", "msg")];
        let active_refs: Vec<&RemoteAnnouncement> = active.iter().collect();
        assert!(!prune_hidden_announcement_ids(&mut hidden, &active_refs));
    }

    #[test]
    fn visible_filters_empty_messages() {
        let anns = vec![
            ann("1", "hello"),
            RemoteAnnouncement {
                id: Some("2".to_string()),
                message: Some("".to_string()),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("3".to_string()),
                message: None,
                ..Default::default()
            },
            ann("4", "  "), // whitespace-only
        ];
        let visible = visible_announcements(&anns);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].id.as_deref(), Some("1"));
    }

    #[test]
    fn filter_expired_at_removes_past_expiry() {
        let anns = vec![
            RemoteAnnouncement {
                id: Some("expired".to_string()),
                message: Some("old".to_string()),
                expires_at: Some(100),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("active".to_string()),
                message: Some("new".to_string()),
                expires_at: Some(200),
                ..Default::default()
            },
            RemoteAnnouncement {
                id: Some("noexpiry".to_string()),
                message: Some("forever".to_string()),
                ..Default::default()
            },
        ];
        let filtered = filter_expired_at(anns, 150);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|a| a.id.as_deref() != Some("expired")));
    }

    #[test]
    fn resolve_startup_with_override() {
        let result = resolve_startup_announcements_with_override(
            Ok(vec![]),
            Some(r#"[{"id":"test","message":"override"}]"#),
        );
        assert!(result.is_some());
        let anns = result.unwrap();
        assert_eq!(anns.len(), 1);
        assert_eq!(anns[0].id.as_deref(), Some("test"));
    }

    #[test]
    fn resolve_startup_empty_remote_returns_none() {
        assert!(resolve_startup_announcements_with_override(Ok(vec![]), None).is_none());
    }

    #[test]
    fn resolve_startup_error_returns_none() {
        assert!(
            resolve_startup_announcements_with_override(Err(anyhow::anyhow!("network")), None)
                .is_none()
        );
    }

    #[test]
    fn resolve_startup_override_with_nonempty_remote_uses_override() {
        let remote = vec![ann("remote", "remote msg")];
        let result = resolve_startup_announcements_with_override(
            Ok(remote),
            Some(r#"[{"id":"override","message":"ov"}]"#),
        );
        // Override takes precedence.
        assert_eq!(result.unwrap()[0].id.as_deref(), Some("override"));
    }

    #[test]
    fn resolve_startup_invalid_override_falls_back_to_remote() {
        let remote = vec![ann("remote", "remote msg")];
        let result = resolve_startup_announcements_with_override(Ok(remote), Some("not json"));
        assert_eq!(result.unwrap()[0].id.as_deref(), Some("remote"));
    }

    #[test]
    fn validation_rejects_untrusted_fields() {
        let mut announcement = ann("safe", "hello");
        announcement.cta = Some(AnnouncementCta {
            url: Some("http://example.com".into()),
            ..Default::default()
        });
        assert!(validate_announcements(&[announcement]).is_err());

        let oversized = RemoteAnnouncement {
            message: Some("x".repeat(MAX_MESSAGE_CHARS + 1)),
            ..Default::default()
        };
        assert!(validate_announcements(&[oversized]).is_err());
        assert!(
            validate_announcements(&vec![RemoteAnnouncement::default(); MAX_ANNOUNCEMENTS + 1])
                .is_err()
        );
    }

    #[tokio::test]
    async fn fetch_uses_etag_cache_and_http_requires_explicit_test_opt_in() {
        use axum::{Json, Router, extract::Request, http::StatusCode, routing::get};
        use std::sync::{Arc, Mutex as StdMutex};

        let seen = Arc::new(StdMutex::new(Vec::<Option<String>>::new()));
        let app = Router::new().route(
            "/announcements",
            get({
                let seen = Arc::clone(&seen);
                move |request: Request| {
                    let seen = Arc::clone(&seen);
                    async move {
                        let header = request
                            .headers()
                            .get(IF_NONE_MATCH)
                            .and_then(|value| value.to_str().ok())
                            .map(str::to_owned);
                        seen.lock().unwrap().push(header.clone());
                        if header.as_deref() == Some("\"v1\"") {
                            return (
                                StatusCode::NOT_MODIFIED,
                                [(ETAG, "\"v1\"")],
                                Json(Vec::new()),
                            );
                        }
                        (
                            StatusCode::OK,
                            [(ETAG, "\"v1\"")],
                            Json(vec![ann("cached", "hello")]),
                        )
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let tmp = tempfile::tempdir().unwrap();
        let mut config = AnnouncementEndpointConfig {
            endpoint: format!("http://{address}/announcements"),
            timeout: Duration::from_secs(1),
            allow_http: false,
        };
        assert!(fetch_announcements(tmp.path(), &config).await.is_err());
        config.allow_http = true;
        let first = fetch_announcements(tmp.path(), &config).await.unwrap();
        let second = fetch_announcements(tmp.path(), &config).await.unwrap();
        assert_eq!(first.r#gen, 1);
        assert_eq!(second, first);
        assert_eq!(*seen.lock().unwrap(), vec![None, Some("\"v1\"".into())]);
        server.abort();
    }

    #[tokio::test]
    async fn read_write_hidden_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let ids = BTreeSet::from(["a".to_string(), "b".to_string()]);
        write_hidden_announcement_ids(tmp.path(), &ids)
            .await
            .unwrap();
        let back = read_hidden_announcement_ids(tmp.path()).await.unwrap();
        assert_eq!(ids, back);
        assert!(!hidden_temp_file_path(&hidden_file_path(tmp.path())).exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn hidden_file_is_owner_only_on_unix() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        write_hidden_announcement_ids(tmp.path(), &BTreeSet::from(["private".to_string()]))
            .await
            .unwrap();
        let mode = std::fs::metadata(hidden_file_path(tmp.path()))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[tokio::test]
    async fn write_empty_removes_file() {
        let tmp = tempfile::tempdir().unwrap();
        let ids = BTreeSet::from(["a".to_string()]);
        write_hidden_announcement_ids(tmp.path(), &ids)
            .await
            .unwrap();
        assert!(hidden_file_path(tmp.path()).exists());
        let empty = BTreeSet::new();
        write_hidden_announcement_ids(tmp.path(), &empty)
            .await
            .unwrap();
        assert!(!hidden_file_path(tmp.path()).exists());
    }

    #[tokio::test]
    async fn concurrent_mutations_do_not_lose_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_path_buf();
        let mut tasks = Vec::new();
        for index in 0..32 {
            let home = home.clone();
            tasks.push(tokio::spawn(async move {
                mutate_hidden_announcement_ids(home, |ids| ids.insert(format!("id-{index}")))
                    .await
                    .unwrap();
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        let ids = read_hidden_announcement_ids(tmp.path()).await.unwrap();
        assert_eq!(ids.len(), 32);
    }

    #[tokio::test]
    async fn legacy_file_is_rewritten_as_json_on_mutation() {
        let tmp = tempfile::tempdir().unwrap();
        tokio::fs::create_dir_all(tmp.path()).await.unwrap();
        tokio::fs::write(hidden_file_path(tmp.path()), "old,keep")
            .await
            .unwrap();
        mutate_hidden_announcement_ids(tmp.path(), |ids| ids.insert("new".into()))
            .await
            .unwrap();
        let data = tokio::fs::read_to_string(hidden_file_path(tmp.path()))
            .await
            .unwrap();
        assert_eq!(serde_json::from_str::<Vec<String>>(&data).unwrap().len(), 3);
    }

    #[tokio::test]
    async fn read_nonexistent_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let ids = read_hidden_announcement_ids(tmp.path()).await.unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn announcements_refreshed_serde() {
        let ar = AnnouncementsRefreshed {
            r#gen: 5,
            announcements: vec![ann("1", "hi")],
        };
        let json = serde_json::to_string(&ar).unwrap();
        assert!(json.contains("\"gen\":5"));
        let back: AnnouncementsRefreshed = serde_json::from_str(&json).unwrap();
        assert_eq!(ar, back);
    }
}
