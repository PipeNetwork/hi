//! Shared HTTP send-with-retry used by every adapter.
//!
//! Retries the *initial* request (before streaming begins) on transient
//! failures — connection/timeout errors and 429 / 5xx responses — with capped
//! exponential backoff. The budget is wide enough to ride out a brief backend
//! blip silently (e.g. a 502 while the provider rolls out an update) rather than
//! surfacing it as a turn failure. Mid-stream failures are not retried (they'd
//! duplicate already-emitted output).

use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures_util::{Stream, StreamExt};
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::Deserialize;

use crate::provider::ServedModel;

/// Retry budget for transient connection/timeout errors: brief, then surface.
const MAX_RETRIES: u32 = 3;
/// Wider budget for a transient *server outage* (5xx): ~6 attempts with the
/// capped backoff below span ~12s — enough to silently ride out a quick provider
/// restart/deploy (a 502 while it rolls out an update) instead of failing the
/// turn. Scoped to the status path; a 502 arrives as a fast HTTP response, so the
/// cost is just the backoff sleeps, not stacked connection timeouts.
const OUTAGE_RETRIES: u32 = 6;
const BASE_DELAY_MS: u64 = 250;
/// Cap on a single backoff so the wider budget stays bounded (a few seconds per
/// wait) instead of exploding exponentially.
const MAX_DELAY_MS: u64 = 4_000;
const DEFAULT_STREAM_IDLE_TIMEOUT_SECS: u64 = 300;
const DEFAULT_HTTP_TIMEOUT_SECS: u64 = 900;

#[derive(Deserialize)]
struct ModelsList {
    data: Vec<ModelEntry>,
}
/// One `/models` entry. Only `id` is standard; the rest are pipenetwork-style
/// extensions that other endpoints simply omit (hence all optional).
#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    max_context_tokens: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    input_token_rate: Option<f64>,
    #[serde(default)]
    output_token_rate: Option<f64>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    available: Option<bool>,
}

impl ModelEntry {
    fn into_served(self) -> ServedModel {
        ServedModel {
            id: self.id,
            context_window: self.max_context_tokens,
            max_output_tokens: self.max_output_tokens.or(self.max_completion_tokens),
            // Reported rates are per token; the rest of the app uses per-1M.
            price: match (self.input_token_rate, self.output_token_rate) {
                (Some(i), Some(o)) => Some((i * 1_000_000.0, o * 1_000_000.0)),
                _ => None,
            },
            provider_label: None,
            status: self.status,
            available: self.available.unwrap_or(true),
            availability_reason: None,
            capabilities: Vec::new(),
        }
    }
}

/// GET an OpenAI/Anthropic-style `/models` list from an already-authenticated
/// request and return the served models — what the *current endpoint* actually
/// offers (with any live window/price/health it reports), as opposed to the
/// static models.dev catalog. Bounded by a short timeout so a hung endpoint
/// can't wedge the caller; on timeout the caller falls back to the catalog.
pub async fn fetch_models(builder: RequestBuilder) -> Result<Vec<ServedModel>> {
    let fetch = async {
        let resp = send_with_retry(builder).await?;
        if !resp.status().is_success() {
            bail!("models endpoint returned {}", resp.status());
        }
        let list: ModelsList = resp.json().await.context("parsing models list")?;
        Ok(list.data.into_iter().map(ModelEntry::into_served).collect())
    };
    match tokio::time::timeout(Duration::from_secs(6), fetch).await {
        Ok(result) => result,
        Err(_) => bail!("models request timed out after 6s"),
    }
}

// --- On-disk startup cache for /models results ---
//
// A successful `/models` fetch is cached locally so the next startup applies
// model metadata (window/price/health) instantly, without blocking on the
// network. The live fetch still runs in the background and refreshes the cache;
// the cache just covers the cold-start gap so the UI never looks stalled.

/// The cache file lives in the hi config dir alongside `config.toml`.
fn cache_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| std::path::PathBuf::from(home).join(".config"))
        })?;
    Some(base.join("hi").join("models-cache.json"))
}

/// A stable key for a provider endpoint so pipenetwork@v1 and ollama@localhost
/// don't collide. Includes the base_url so two OpenAI-compatible endpoints with
/// different URLs get separate entries.
pub fn cache_key(provider: &str, base_url: &str) -> String {
    format!("{provider}@{base_url}")
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    /// Unix timestamp (seconds) of the fetch that produced this entry.
    ts: u64,
    models: Vec<ServedModel>,
}

/// Load the cached `/models` result for `key`, if present and not stale.
/// Entries older than 24h are ignored (model metadata drifts: windows expand,
/// prices change, models are added/removed).
pub async fn load_cache(key: &str) -> Option<Vec<ServedModel>> {
    let path = cache_path()?;
    let text = tokio::fs::read_to_string(&path).await.ok()?;
    let map: std::collections::HashMap<String, CacheEntry> = serde_json::from_str(&text).ok()?;
    let entry = map.get(key)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    if now.saturating_sub(entry.ts) > 24 * 60 * 60 {
        return None; // stale
    }
    Some(entry.models.clone())
}

/// Persist a fresh `/models` result for `key`, merging with any other providers'
/// entries already in the cache file. Best-effort: errors are silently dropped
/// (the cache is an optimization, not a source of truth).
pub async fn save_cache(key: &str, models: &[ServedModel]) {
    let Some(path) = cache_path() else { return };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Load existing entries (other providers) so we don't clobber them.
    let mut map: std::collections::HashMap<String, CacheEntry> = tokio::fs::read_to_string(&path)
        .await
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    map.insert(
        key.to_string(),
        CacheEntry {
            ts: now,
            models: models.to_vec(),
        },
    );
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    let _ = tokio::fs::write(&path, serde_json::to_string(&map).unwrap_or_default()).await;
}

/// Give up on a stream if the model produces no output — content, reasoning, or
/// tool tokens — and no stream activity arrives for this long (default 300s,
/// override with `HI_STREAM_TIMEOUT` in seconds). Before the first token,
/// heartbeat/data frames keep the cold-start wait alive; after output starts,
/// adapters use the shorter stall timeout and require real tokens.
pub fn stream_idle_timeout() -> Duration {
    let configured = std::env::var("HI_STREAM_TIMEOUT").ok();
    timeout_from_env_value(configured.as_deref(), DEFAULT_STREAM_IDLE_TIMEOUT_SECS)
}

/// Once output *has* started flowing, end the stream this long after tokens stop
/// if no completion signal (`finish_reason`/`[DONE]`/socket close) arrives —
/// covers a provider that streams a full answer then holds the connection open
/// without terminating it (default 15s, override with `HI_STREAM_STALL`). Much
/// shorter than the cold-start [`stream_idle_timeout`] because a multi-second
/// gap *between* tokens means the stream has effectively ended, whereas a slow
/// time-to-first-token can be a legitimately queued request.
pub fn stream_stall_timeout() -> Duration {
    let configured = std::env::var("HI_STREAM_STALL").ok();
    timeout_from_env_value(configured.as_deref(), 15)
}

fn agent_http_timeout() -> Duration {
    let configured = std::env::var("HI_HTTP_TIMEOUT").ok();
    timeout_from_env_value(configured.as_deref(), DEFAULT_HTTP_TIMEOUT_SECS)
}

fn timeout_from_env_value(value: Option<&str>, default_seconds: u64) -> Duration {
    value
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(default_seconds))
}

/// When `HI_DEBUG_STREAM` is set, echo every raw byte chunk (escaped, so SSE
/// comment heartbeats and data lines are both visible) to stderr — a wire-level
/// view for diagnosing a provider that returns nothing. A no-op otherwise.
pub fn debug_tap<B, S>(stream: S) -> impl Stream<Item = Result<B, reqwest::Error>>
where
    S: Stream<Item = Result<B, reqwest::Error>>,
    B: AsRef<[u8]>,
{
    let on = std::env::var_os("HI_DEBUG_STREAM").is_some();
    stream.inspect(move |item| {
        if on && let Ok(bytes) = item {
            let raw = bytes.as_ref();
            eprintln!(
                "\x1b[2m[sse {}b] {}\x1b[0m",
                raw.len(),
                String::from_utf8_lossy(raw).escape_debug()
            );
        }
    })
}

/// Build a `reqwest::Client` with connection-pool and keep-alive tuned for an
/// agent loop that makes many sequential requests to the same endpoint.
/// Reusing connections avoids a TLS handshake on every model call — the
/// default `Client::new()` does pool internally, but this sets explicit
/// limits and keep-alive so long sessions reuse connections reliably.
pub fn agent_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_idle_timeout(Some(Duration::from_secs(90)))
        .pool_max_idle_per_host(4)
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .connect_timeout(Duration::from_secs(15))
        .timeout(agent_http_timeout())
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Send `builder`, retrying transient failures with exponential backoff.
pub async fn send_with_retry(builder: RequestBuilder) -> Result<Response> {
    let mut attempt = 0;
    loop {
        // Clone so the body survives a retry; fall back to a single send if the
        // body isn't cloneable (not the case for our JSON bodies).
        let Some(attempt_builder) = builder.try_clone() else {
            return Ok(builder.send().await?);
        };

        match attempt_builder.send().await {
            Ok(response) => {
                if attempt < retry_limit(response.status()) {
                    attempt += 1;
                    backoff(attempt).await;
                    continue;
                }
                return Ok(response);
            }
            Err(err) => {
                if attempt < MAX_RETRIES && is_retryable_error(&err) {
                    attempt += 1;
                    backoff(attempt).await;
                    continue;
                }
                bail!("request failed: {err}");
            }
        }
    }
}

/// How many times a given response status is worth retrying: a wide budget for a
/// transient 5xx outage (ride out a deploy), none otherwise. A 429 throttle (and
/// every other 4xx) surfaces immediately — retrying a rate limit just stalls the
/// turn and can deepen the throttle; the caller backs off deliberately instead.
fn retry_limit(status: StatusCode) -> u32 {
    if status.is_server_error() {
        OUTAGE_RETRIES
    } else {
        0
    }
}

fn is_retryable_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request()
}

async fn backoff(attempt: u32) {
    tokio::time::sleep(Duration::from_millis(backoff_delay(attempt))).await;
}

/// Backoff for `attempt` (1-based): exponential from [`BASE_DELAY_MS`], capped at
/// [`MAX_DELAY_MS`]. Split out from the sleep so it's unit-testable.
fn backoff_delay(attempt: u32) -> u64 {
    let exp = BASE_DELAY_MS.saturating_mul(2u64.saturating_pow(attempt - 1));
    exp.min(MAX_DELAY_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_server_outages_are_retried() {
        // 5xx (transient outage / deploy) rides out the wider budget…
        assert_eq!(retry_limit(StatusCode::BAD_GATEWAY), OUTAGE_RETRIES);
        assert_eq!(retry_limit(StatusCode::SERVICE_UNAVAILABLE), OUTAGE_RETRIES);
        assert_eq!(retry_limit(StatusCode::GATEWAY_TIMEOUT), OUTAGE_RETRIES);
        // The outage budget is non-zero — a compile-time invariant, asserted in a
        // const block so clippy doesn't flag it as a constant-condition assertion.
        const _: () = assert!(OUTAGE_RETRIES > 0);
        // …everything else surfaces immediately: a 429 throttle, auth, client errors.
        assert_eq!(retry_limit(StatusCode::TOO_MANY_REQUESTS), 0);
        assert_eq!(retry_limit(StatusCode::BAD_REQUEST), 0);
        assert_eq!(retry_limit(StatusCode::UNAUTHORIZED), 0);
        assert_eq!(retry_limit(StatusCode::OK), 0);
    }

    #[test]
    fn backoff_is_exponential_then_capped() {
        assert_eq!(backoff_delay(1), 250);
        assert_eq!(backoff_delay(2), 500);
        assert_eq!(backoff_delay(3), 1000);
        assert_eq!(backoff_delay(4), 2000);
        // 250 * 2^4 = 4000 hits the cap; later attempts stay there (no overflow).
        assert_eq!(backoff_delay(5), MAX_DELAY_MS);
        assert_eq!(backoff_delay(6), MAX_DELAY_MS);
        assert_eq!(backoff_delay(64), MAX_DELAY_MS);
    }

    #[test]
    fn timeout_defaults_are_long_enough_for_backend_retrying() {
        assert_eq!(
            timeout_from_env_value(None, DEFAULT_STREAM_IDLE_TIMEOUT_SECS),
            Duration::from_secs(300)
        );
        assert_eq!(
            timeout_from_env_value(None, DEFAULT_HTTP_TIMEOUT_SECS),
            Duration::from_secs(900)
        );
        assert_eq!(
            timeout_from_env_value(Some("42"), DEFAULT_STREAM_IDLE_TIMEOUT_SECS),
            Duration::from_secs(42)
        );
        assert_eq!(
            timeout_from_env_value(Some("0"), DEFAULT_STREAM_IDLE_TIMEOUT_SECS),
            Duration::from_secs(300)
        );
        assert_eq!(
            timeout_from_env_value(Some("not-a-number"), DEFAULT_HTTP_TIMEOUT_SECS),
            Duration::from_secs(900)
        );
    }

    #[test]
    fn parses_openai_style_models_list() {
        // Extra fields (object, created, …) are ignored; only `data[].id` matters.
        let json = r#"{"object":"list","data":[
            {"id":"ipop/coder-balanced","object":"model","created":1},
            {"id":"another-model"}
        ]}"#;
        let list: ModelsList = serde_json::from_str(json).unwrap();
        let ids: Vec<String> = list.data.into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["ipop/coder-balanced", "another-model"]);
    }

    #[test]
    fn parses_pipenetwork_model_metadata() {
        // pipenetwork.ai extends /models with window, per-token rates, and health.
        let json = r#"{"data":[
            {"id":"ipop/coder-balanced","max_context_tokens":1000000,
             "max_output_tokens":131072,
             "input_token_rate":0.000001,"output_token_rate":0.000002,
             "status":"available","available":true},
            {"id":"pipe/auto-code","max_completion_tokens":16384},
            {"id":"grok","status":"degraded","available":true},
            {"id":"down","available":false}
        ]}"#;
        let list: ModelsList = serde_json::from_str(json).unwrap();
        let served: Vec<ServedModel> = list.data.into_iter().map(ModelEntry::into_served).collect();

        assert_eq!(served[0].context_window, Some(1_000_000));
        assert_eq!(served[0].max_output_tokens, Some(131_072));
        assert_eq!(served[0].price, Some((1.0, 2.0))); // per-token → per-1M
        assert_eq!(served[0].health(), None, "available is healthy");

        assert_eq!(served[1].max_output_tokens, Some(16_384));

        assert_eq!(served[2].context_window, None);
        assert_eq!(served[2].health(), Some("degraded"));

        assert_eq!(
            served[3].health(),
            Some("unavailable"),
            "available:false flagged"
        );
    }

    #[test]
    fn cache_key_distinguishes_providers_and_urls() {
        assert_ne!(
            cache_key("pipenetwork", "https://api.pipenetwork.ai/v1"),
            cache_key("ollama", "http://localhost:11434/v1"),
        );
        // Same provider, different base URLs → different keys.
        assert_ne!(
            cache_key("openai", "https://a.com/v1"),
            cache_key("openai", "https://b.com/v1"),
        );
        // Same inputs → same key.
        assert_eq!(
            cache_key("pipenetwork", "https://api.pipenetwork.ai/v1"),
            cache_key("pipenetwork", "https://api.pipenetwork.ai/v1"),
        );
    }

    #[test]
    fn cache_entry_round_trips_through_json() {
        // The on-disk cache serializes Vec<ServedModel> + a timestamp. A
        // round-trip must preserve every field so metadata (window/price/health)
        // survives across startups.
        let entry = CacheEntry {
            ts: 1_700_000_000,
            models: vec![
                ServedModel {
                    id: "ipop/coder-balanced".into(),
                    context_window: Some(1_000_000),
                    max_output_tokens: Some(131_072),
                    price: Some((1.0, 2.0)),
                    provider_label: None,
                    status: Some("available".into()),
                    available: true,
                    availability_reason: None,
                    capabilities: Vec::new(),
                },
                ServedModel {
                    id: "grok".into(),
                    context_window: None,
                    max_output_tokens: None,
                    price: None,
                    provider_label: None,
                    status: Some("degraded".into()),
                    available: false,
                    availability_reason: None,
                    capabilities: Vec::new(),
                },
            ],
        };
        let json = serde_json::to_string(&entry).unwrap();
        let back: CacheEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ts, entry.ts);
        assert_eq!(back.models.len(), 2);
        assert_eq!(back.models[0].context_window, Some(1_000_000));
        assert_eq!(back.models[0].max_output_tokens, Some(131_072));
        assert_eq!(back.models[0].price, Some((1.0, 2.0)));
        assert_eq!(back.models[1].status, Some("degraded".into()));
        assert!(!back.models[1].available);
    }

    #[tokio::test]
    async fn cache_disk_round_trip_uses_temp_home() {
        // Verify the load/save path through the real filesystem, isolated via a
        // temp HOME. Runs serially (no other test in this module touches HOME).
        let dir = std::env::temp_dir().join(format!(
            "hi-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: this test runs in a single task; no other code in this test
        // reads HOME/XDG_CONFIG_HOME concurrently. Other tests in this crate
        // don't touch these vars.
        unsafe {
            std::env::set_var("HOME", &dir);
            std::env::remove_var("XDG_CONFIG_HOME");
        }

        let key = cache_key("pipenetwork", "https://api.pipenetwork.ai/v1");
        let models = vec![ServedModel {
            id: "m1".into(),
            context_window: Some(128_000),
            max_output_tokens: Some(16_384),
            price: None,
            provider_label: None,
            status: None,
            available: true,
            availability_reason: None,
            capabilities: Vec::new(),
        }];

        assert!(load_cache(&key).await.is_none(), "empty before save");
        save_cache(&key, &models).await;
        let loaded = load_cache(&key).await.expect("hit after save");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "m1");
        assert_eq!(loaded[0].context_window, Some(128_000));
        assert_eq!(loaded[0].max_output_tokens, Some(16_384));

        // A second key doesn't clobber the first.
        save_cache(&cache_key("ollama", "http://x/v1"), &[]).await;
        assert!(load_cache(&key).await.is_some(), "first entry preserved");

        // Stale (>24h) entry is ignored.
        let path = dir.join(".config/hi/models-cache.json");
        let text = std::fs::read_to_string(&path).unwrap();
        let mut map: std::collections::HashMap<String, CacheEntry> =
            serde_json::from_str(&text).unwrap();
        if let Some(e) = map.get_mut(&key) {
            e.ts = e.ts.saturating_sub(25 * 60 * 60 + 1);
        }
        std::fs::write(&path, serde_json::to_string(&map).unwrap()).unwrap();
        assert!(load_cache(&key).await.is_none(), "stale entry ignored");

        std::fs::remove_dir_all(&dir).ok();
    }
}
