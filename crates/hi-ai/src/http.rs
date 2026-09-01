//! Shared HTTP send-with-retry used by every adapter.
//!
//! Retries the *initial* request (before streaming begins) on transient
//! failures — connection/timeout errors only — with capped exponential
//! backoff. HTTP responses are returned to the provider adapter so its typed
//! `code`/`retryable` contract decides whether another logical attempt is safe.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures_util::{Stream, StreamExt};
#[cfg(test)]
use reqwest::StatusCode;
use reqwest::{RequestBuilder, Response};
use serde::Deserialize;
use tokio::time::Instant;

use crate::provider::ServedModel;

/// The agent identifier sent over HTTP, mirroring the `AI_AGENT=hi` env var the
/// shell path sets (`hi-tools::tools::mark_agent_harness`). HuggingFace-side
/// infrastructure that detects `hi` agent harnesses keys off this token; sending
/// it as a header makes the identification consistent across the subprocess and
/// in-process HTTP surfaces.
const HF_AGENT_HEADER_NAME: &str = "AI_AGENT";
const HF_AGENT_ID: &str = "hi";

/// Retry budget for transient connection/timeout errors: brief, then surface.
const MAX_RETRIES: u32 = 3;
const BASE_DELAY_MS: u64 = 250;
/// Cap on a single backoff so the wider budget stays bounded (a few seconds per
/// wait) instead of exploding exponentially.
const MAX_DELAY_MS: u64 = 4_000;
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 120;
const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = 16;
const DEFAULT_TCP_KEEPALIVE_SECS: u64 = 30;
/// Idle read timeout for streaming LLM responses (chunks can be sparse during
/// long generations). Non-stream metadata/tool calls use
/// [`agent_http_client_quick`] instead.
const DEFAULT_READ_TIMEOUT_SECS: u64 = 360;
/// xAI grok-4.6 reasoning can sit silent before the first token; their SDK
/// examples use a 3600s timeout. Used only by [`agent_http_client_xai`].
const DEFAULT_XAI_READ_TIMEOUT_SECS: u64 = 3_600;
/// Idle window for the xAI Responses stream. Longer than the shared 240s
/// default so a thinking gap is not mistaken for a dead connection.
const DEFAULT_XAI_STREAM_IDLE_SECS: u64 = 3_600;
/// Connect/read budget for non-streaming agent HTTP (auth, /models, MCP, search).
const DEFAULT_QUICK_CONNECT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_QUICK_READ_TIMEOUT_SECS: u64 = 60;
const MAX_HTTP_TIMEOUT_SECS: u64 = 3_600;
const DEFAULT_MODEL_DISCOVERY_DEADLINE_SECS: u64 = 30;
const DEFAULT_AUTH_REFRESH_DEADLINE_SECS: u64 = 30;

/// One absolute deadline shared by every phase of an HTTP operation.
#[derive(Clone, Copy, Debug)]
pub struct OperationBudget {
    deadline: Instant,
}

impl OperationBudget {
    pub fn new(duration: Duration) -> Self {
        Self {
            deadline: Instant::now() + duration,
        }
    }

    pub async fn run<T>(
        self,
        context: impl FnOnce() -> String,
        future: impl std::future::Future<Output = T>,
    ) -> Result<T> {
        match tokio::time::timeout_at(self.deadline, future).await {
            Ok(value) => Ok(value),
            Err(error) => {
                hi_observability::record(hi_observability::ReliabilityEvent::HttpDeadline);
                Err(error).with_context(context)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpRetryEvent {
    pub retry: u32,
    pub fresh_connection: bool,
    pub error_kind: &'static str,
    pub delay: Duration,
}

pub trait HttpRetryObserver: Send + Sync {
    fn on_retry(&self, event: &HttpRetryEvent);
}

static HTTP_RETRY_OBSERVER: OnceLock<Arc<dyn HttpRetryObserver>> = OnceLock::new();

pub fn set_http_retry_observer(
    observer: Arc<dyn HttpRetryObserver>,
) -> Result<(), Arc<dyn HttpRetryObserver>> {
    HTTP_RETRY_OBSERVER.set(observer)
}

fn observe_retry(event: HttpRetryEvent) {
    hi_observability::record(if event.fresh_connection {
        hi_observability::ReliabilityEvent::HttpFreshPoolEscape
    } else {
        hi_observability::ReliabilityEvent::HttpRetry
    });
    tracing::info!(
        target: "hi::reliability",
        event_kind = "http_retry",
        retry = event.retry,
        fresh_connection = event.fresh_connection,
        error_kind = event.error_kind,
        delay_ms = event.delay.as_millis() as u64,
    );
    if let Some(observer) = HTTP_RETRY_OBSERVER.get() {
        observer.on_retry(&event);
    }
}

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
/// offers (with any live window/price/health it reports).
pub async fn fetch_models(builder: RequestBuilder) -> Result<Vec<ServedModel>> {
    let duration = operation_deadline(
        "HI_MODEL_DISCOVERY_DEADLINE_SECS",
        DEFAULT_MODEL_DISCOVERY_DEADLINE_SECS,
    );
    let budget = OperationBudget::new(duration);
    budget
        .run(
            || format!("model discovery exceeded {}s deadline", duration.as_secs()),
            fetch_models_inner(builder, budget),
        )
        .await?
}

async fn fetch_models_inner(
    builder: RequestBuilder,
    budget: OperationBudget,
) -> Result<Vec<ServedModel>> {
    let resp = send_with_retry_deadline(builder, budget).await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if crate::is_billing_or_quota_text(&text) {
            bail!(
                "models endpoint returned {status}: this provider is out of credits — \
                 /login pipenetwork then /provider pipenetwork"
            );
        }
        bail!("models endpoint returned {status}");
    }
    let list: ModelsList = resp.json().await.context("parsing models list")?;
    Ok(list.data.into_iter().map(ModelEntry::into_served).collect())
}

// --- On-disk startup cache for /models results ---
//
// A successful `/models` fetch is cached locally so the next startup applies
// model metadata (window/price/health) instantly, without blocking on the
// network. The live fetch still runs in the background and refreshes the cache.

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
    /// Model ids from the previous successful fetch for this key. Used to mark
    /// newly advertised ids as `(new)` in the picker. Empty on the first save.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    previous_ids: Vec<String>,
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
    let previous_ids = map
        .get(key)
        .map(|entry| entry.models.iter().map(|m| m.id.clone()).collect())
        .unwrap_or_else(|| models.iter().map(|m| m.id.clone()).collect());
    map.insert(
        key.to_string(),
        CacheEntry {
            ts: now,
            models: models.to_vec(),
            previous_ids,
        },
    );
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    let _ = tokio::fs::write(&path, serde_json::to_string(&map).unwrap_or_default()).await;
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
///
/// Prefer [`agent_http_client_quick`] for non-streaming calls so a stuck peer
/// cannot sit on the 360s streaming read timeout.
pub fn agent_http_client() -> reqwest::Client {
    agent_http_client_for_socket(None)
}

/// Streaming client for the xAI Responses adapter. Read timeout defaults to
/// 3600s (xAI's published reasoning-model budget) instead of the shared 360s.
pub(crate) fn agent_http_client_xai() -> reqwest::Client {
    build_agent_http_client(
        None,
        http_timeout_secs("HI_HTTP_CONNECT_TIMEOUT_SECS", DEFAULT_CONNECT_TIMEOUT_SECS),
        http_timeout_secs("HI_HTTP_READ_TIMEOUT_SECS", DEFAULT_XAI_READ_TIMEOUT_SECS),
    )
}

/// Like [`agent_http_client`] but with short connect/read timeouts for
/// metadata, auth, MCP, and other non-streaming requests.
pub fn agent_http_client_quick() -> reqwest::Client {
    build_agent_http_client(
        None,
        http_timeout_secs(
            "HI_HTTP_QUICK_CONNECT_TIMEOUT_SECS",
            DEFAULT_QUICK_CONNECT_TIMEOUT_SECS,
        ),
        http_timeout_secs(
            "HI_HTTP_QUICK_READ_TIMEOUT_SECS",
            DEFAULT_QUICK_READ_TIMEOUT_SECS,
        ),
    )
}

/// Build the normal agent client while pinning all HTTP transport to one Unix
/// socket. The URL still supplies HTTP paths and Host semantics; no TCP or DNS
/// connection can be made by this client.
pub fn agent_http_client_for_socket(socket: Option<&std::path::Path>) -> reqwest::Client {
    build_agent_http_client(
        socket,
        http_timeout_secs("HI_HTTP_CONNECT_TIMEOUT_SECS", DEFAULT_CONNECT_TIMEOUT_SECS),
        http_timeout_secs("HI_HTTP_READ_TIMEOUT_SECS", DEFAULT_READ_TIMEOUT_SECS),
    )
}

fn build_agent_http_client(
    socket: Option<&std::path::Path>,
    connect_timeout_secs: u64,
    read_timeout_secs: u64,
) -> reqwest::Client {
    // Identify hi to upstream HTTP services. `User-Agent` is the standard
    // channel; the `AI_AGENT` header mirrors the env-var convention the shell
    // path already uses, so HuggingFace infra sees a consistent `hi` marker on
    // both the subprocess and in-process HTTP surfaces. Additive for existing
    // providers — unknown headers are ignored.
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(value) = reqwest::header::HeaderValue::from_str(HF_AGENT_ID) {
        headers.insert(HF_AGENT_HEADER_NAME, value);
    }
    let mut builder = reqwest::Client::builder()
        .user_agent(format!("hi/{}", env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        // Credentials are attached to requests against a configured base host.
        // reqwest strips `Authorization` on cross-host redirects but NOT custom
        // headers like Anthropic's `x-api-key`, so a same-scheme redirect to a
        // different host would forward the key. A same-host https→http hop
        // would also send the key in the clear. Refuse any origin change
        // (host, port, or scheme); same-origin path/version redirects still work.
        .redirect(credential_redirect_policy())
        .connect_timeout(Duration::from_secs(connect_timeout_secs))
        .read_timeout(Duration::from_secs(read_timeout_secs))
        .pool_idle_timeout(Some(Duration::from_secs(http_timeout_secs(
            "HI_HTTP_POOL_IDLE_TIMEOUT_SECS",
            DEFAULT_POOL_IDLE_TIMEOUT_SECS,
        ))))
        .pool_max_idle_per_host(http_env_usize(
            "HI_HTTP_POOL_MAX_IDLE_PER_HOST",
            DEFAULT_POOL_MAX_IDLE_PER_HOST,
            1,
            128,
        ))
        .tcp_keepalive(Some(Duration::from_secs(http_timeout_secs(
            "HI_HTTP_TCP_KEEPALIVE_SECS",
            DEFAULT_TCP_KEEPALIVE_SECS,
        ))));
    #[cfg(unix)]
    if let Some(socket) = socket {
        builder = builder.unix_socket(socket);
    }
    builder
        .build()
        .unwrap_or_else(|_| timed_http_client_fallback(connect_timeout_secs, read_timeout_secs))
}

/// Last-resort client that still carries timeouts — never fall back to an
/// unbounded `Client::new()`. Keeps the same credential redirect policy as
/// the primary agent client so a builder failure cannot silently start
/// forwarding `x-api-key` across hosts or onto http.
pub fn timed_http_client_fallback(
    connect_timeout_secs: u64,
    read_timeout_secs: u64,
) -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(credential_redirect_policy())
        .connect_timeout(Duration::from_secs(connect_timeout_secs.max(1)))
        .read_timeout(Duration::from_secs(read_timeout_secs.max(1)))
        .build()
        .unwrap_or_else(|_| {
            reqwest::Client::builder()
                .redirect(credential_redirect_policy())
                .timeout(Duration::from_secs(
                    read_timeout_secs.max(connect_timeout_secs).max(1),
                ))
                .build()
                .expect("failed to build timed reqwest Client")
        })
}

/// Redirect policy for HTTP clients that attach credentials (`Authorization`,
/// Anthropic `x-api-key`, portal `x-api-key`). Follows only same-origin hops
/// (host, port, and scheme). reqwest strips `Authorization` on a cross-host
/// redirect but not custom headers, and a same-host https→http hop would
/// send the key in the clear.
pub fn credential_redirect_policy() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(credential_redirect_action)
}

/// Follow only same-origin redirects. Credentials (including Anthropic's
/// `x-api-key`, which reqwest does not strip) stay on the configured host.
fn credential_redirect_action(
    attempt: reqwest::redirect::Attempt<'_>,
) -> reqwest::redirect::Action {
    let Some(prev) = attempt.previous().last() else {
        return attempt.error("refusing redirect with empty previous chain");
    };
    if redirect_leaves_origin(prev, attempt.url()) {
        attempt.error("refusing cross-origin redirect with credentials attached")
    } else {
        attempt.follow()
    }
}

/// True when a redirect would leave the request origin: different host, port,
/// or scheme. Same-host https→http is included so a key is never sent in the
/// clear after an https start.
fn redirect_leaves_origin(from: &reqwest::Url, to: &reqwest::Url) -> bool {
    from.host_str() != to.host_str()
        || from.port_or_known_default() != to.port_or_known_default()
        || from.scheme() != to.scheme()
}

pub fn auth_refresh_deadline() -> Duration {
    operation_deadline(
        "HI_AUTH_REFRESH_DEADLINE_SECS",
        DEFAULT_AUTH_REFRESH_DEADLINE_SECS,
    )
}

fn operation_deadline(var_name: &str, default_secs: u64) -> Duration {
    Duration::from_secs(http_timeout_secs(var_name, default_secs))
}

fn http_timeout_secs(var_name: &str, default_secs: u64) -> u64 {
    std::env::var(var_name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(|seconds| seconds.min(MAX_HTTP_TIMEOUT_SECS))
        .unwrap_or(default_secs)
}

fn http_env_usize(var_name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(var_name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

/// Send `builder`, retrying transient failures with exponential backoff, then
/// one final attempt without backoff. Every attempt runs on the request's own
/// client — a substitute client would discard its transport configuration
/// (Unix-socket pinning, default headers, timeout profile).
pub async fn send_with_retry(builder: RequestBuilder) -> Result<Response> {
    send_with_retry_deadline(
        builder,
        OperationBudget::new(Duration::from_secs(MAX_HTTP_TIMEOUT_SECS)),
    )
    .await
}

/// Send with retries while charging attempts, backoffs, and the fresh HTTP/1
/// escape attempt to one absolute operation deadline.
pub async fn send_with_retry_deadline(
    builder: RequestBuilder,
    budget: OperationBudget,
) -> Result<Response> {
    let mut attempt = 0;
    loop {
        let Some(attempt_builder) = builder.try_clone() else {
            return budget
                .run(
                    || "one-shot HTTP request exceeded its operation deadline".to_string(),
                    builder.send(),
                )
                .await?
                .context("one-shot non-cloneable HTTP request failed");
        };

        match budget
            .run(
                || {
                    format!(
                        "HTTP operation deadline exceeded during attempt {}",
                        attempt + 1
                    )
                },
                attempt_builder.send(),
            )
            .await?
        {
            Ok(response) => return Ok(response),
            Err(err) if attempt < MAX_RETRIES && is_retryable_error(&err) => {
                attempt += 1;
                let delay = Duration::from_millis(backoff_delay(attempt));
                observe_retry(HttpRetryEvent {
                    retry: attempt,
                    fresh_connection: false,
                    error_kind: retry_error_kind(&err),
                    delay,
                });
                budget
                    .run(
                        || "HTTP operation deadline exceeded during retry backoff".to_string(),
                        tokio::time::sleep(delay),
                    )
                    .await?;
            }
            Err(err) if is_retryable_error(&err) => {
                // Final attempt on the request's OWN client. A stock client
                // here would silently escape the originating client's transport
                // configuration — most critically `unix_socket()` pinning,
                // whose invariant is that no TCP or DNS connection can ever be
                // made — and would also drop its default headers and timeout
                // profile. The tradeoff: the pool evicts only the connection
                // that just failed, so with several dead idle connections this
                // attempt may still draw a dead one — accepted, since retries
                // have already evicted one per attempt and transport pinning
                // is a security boundary while pool freshness is not.
                let Some((client, Ok(request))) =
                    builder.try_clone().map(|retry| retry.build_split())
                else {
                    bail!("request failed after {attempt} retries: {err}");
                };
                observe_retry(HttpRetryEvent {
                    retry: attempt + 1,
                    fresh_connection: true,
                    error_kind: retry_error_kind(&err),
                    delay: Duration::ZERO,
                });
                return budget
                    .run(
                        || "HTTP operation deadline exceeded on final connection".to_string(),
                        client.execute(request),
                    )
                    .await?
                    .context("request failed on final fresh connection");
            }
            Err(err) => bail!("request failed: {err}"),
        }
    }
}

fn retry_error_kind(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connect"
    } else {
        "request"
    }
}

fn is_retryable_error(err: &reqwest::Error) -> bool {
    // `is_request` covers mid-request transport failures — canonically a reused
    // keep-alive connection dying under us (ECONNRESET/IncompleteMessage), the
    // stale-pool class the final fresh-connection attempt exists for. Without
    // it, the first request after an idle period fails outright.
    err.is_timeout() || err.is_connect() || err.is_request()
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
    fn response_statuses_are_classified_by_the_adapter() {
        for status in [
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::BAD_REQUEST,
        ] {
            assert!(status.is_client_error() || status.is_server_error());
        }
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
    fn retry_event_is_structured() {
        let event = HttpRetryEvent {
            retry: MAX_RETRIES + 1,
            fresh_connection: true,
            error_kind: "connect",
            delay: Duration::ZERO,
        };
        assert_eq!(event.retry, 4);
        assert!(event.fresh_connection);
        assert_eq!(event.error_kind, "connect");
        assert_eq!(event.delay, Duration::ZERO);
    }

    #[test]
    fn operation_deadlines_are_positive_and_bounded() {
        assert_eq!(
            operation_deadline("HI_MISSING_DEADLINE_TEST", 30),
            Duration::from_secs(30)
        );
        assert!(auth_refresh_deadline() <= Duration::from_secs(MAX_HTTP_TIMEOUT_SECS));
    }

    #[tokio::test(start_paused = true)]
    async fn operation_budget_bounds_retry_backoff() {
        let budget = OperationBudget::new(Duration::from_millis(100));
        let started = Instant::now();
        let error = budget
            .run(
                || "budget expired during backoff".to_string(),
                tokio::time::sleep(Duration::from_secs(10)),
            )
            .await
            .unwrap_err();
        assert_eq!(Instant::now() - started, Duration::from_millis(100));
        assert!(error.to_string().contains("budget expired during backoff"));
    }

    #[tokio::test(start_paused = true)]
    async fn absolute_budget_is_shared_across_operation_phases() {
        let budget = OperationBudget::new(Duration::from_secs(5));
        budget
            .run(
                || "first phase".to_string(),
                tokio::time::sleep(Duration::from_secs(4)),
            )
            .await
            .unwrap();
        let started = Instant::now();
        let error = budget
            .run(
                || "polling expired".to_string(),
                tokio::time::sleep(Duration::from_secs(4)),
            )
            .await
            .unwrap_err();
        assert_eq!(Instant::now() - started, Duration::from_secs(1));
        assert!(error.to_string().contains("polling expired"));
    }

    #[test]
    fn http_timeout_env_is_bounded() {
        unsafe {
            std::env::remove_var("HI_HTTP_TIMEOUT_TEST");
        }
        assert_eq!(http_timeout_secs("HI_HTTP_TIMEOUT_TEST", 123), 123);

        unsafe {
            std::env::set_var("HI_HTTP_TIMEOUT_TEST", "0");
        }
        assert_eq!(http_timeout_secs("HI_HTTP_TIMEOUT_TEST", 123), 123);

        unsafe {
            std::env::set_var("HI_HTTP_TIMEOUT_TEST", "42");
        }
        assert_eq!(http_timeout_secs("HI_HTTP_TIMEOUT_TEST", 123), 42);

        unsafe {
            std::env::set_var("HI_HTTP_TIMEOUT_TEST", "999999");
        }
        assert_eq!(
            http_timeout_secs("HI_HTTP_TIMEOUT_TEST", 123),
            MAX_HTTP_TIMEOUT_SECS
        );

        unsafe {
            std::env::remove_var("HI_HTTP_TIMEOUT_TEST");
        }
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
            {"id":"pipe/auto-coder","max_completion_tokens":16384},
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
            previous_ids: vec!["grok".into()],
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
        assert_eq!(back.previous_ids, vec!["grok".to_string()]);
    }

    #[tokio::test]
    async fn cache_disk_round_trip_uses_temp_home() {
        // Verify the load/save path through the real filesystem, isolated via a
        // temp HOME. The credential store's tests redirect HOME too, so this
        // takes the crate-wide lock rather than assuming it is alone.
        let _home_guard = crate::ENV_HOME_LOCK.lock().await;
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

        let path = dir.join(".config/hi/models-cache.json");
        let text = std::fs::read_to_string(&path).unwrap();
        let map: std::collections::HashMap<String, CacheEntry> =
            serde_json::from_str(&text).unwrap();
        assert_eq!(
            map.get(&key).map(|e| e.previous_ids.clone()),
            Some(vec!["m1".into()]),
            "first save records the current ids so nothing is (new)"
        );

        let models2 = vec![
            models[0].clone(),
            ServedModel {
                id: "m2".into(),
                context_window: None,
                max_output_tokens: None,
                price: None,
                provider_label: None,
                status: None,
                available: true,
                availability_reason: None,
                capabilities: Vec::new(),
            },
        ];
        save_cache(&key, &models2).await;
        let text = std::fs::read_to_string(&path).unwrap();
        let map: std::collections::HashMap<String, CacheEntry> =
            serde_json::from_str(&text).unwrap();
        assert_eq!(
            map.get(&key).map(|e| e.previous_ids.clone()),
            Some(vec!["m1".into()]),
            "second save keeps the previous id set"
        );

        // A second key doesn't clobber the first.
        save_cache(&cache_key("ollama", "http://x/v1"), &[]).await;
        assert!(load_cache(&key).await.is_some(), "first entry preserved");

        // Stale (>24h) entry is ignored.
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

    /// Spin up a minimal HTTP/1.1 server on loopback that answers every
    /// request with `response`. Returns the bound port.
    async fn one_shot_server(response: String) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            // Accept a couple of connections in case the client retries.
            for _ in 0..2 {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let response = response.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = [0u8; 4096];
                    let _ = socket.read(&mut buf).await;
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.flush().await;
                });
            }
        });
        port
    }

    #[tokio::test]
    async fn agent_client_refuses_cross_host_redirect() {
        // The target server would happily receive the request, but reaching it
        // requires a cross-host (different-port) redirect, which the client
        // must refuse so attached credentials are never forwarded off-host.
        let target_port = one_shot_server(
            "HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok".to_string(),
        )
        .await;
        let redirect_port = one_shot_server(format!(
            "HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:{target_port}/\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
        ))
        .await;

        let client = agent_http_client_quick();
        let result = client
            .get(format!("http://127.0.0.1:{redirect_port}/"))
            .send()
            .await;
        let err = result.expect_err("cross-host redirect must be rejected");
        assert!(
            err.is_redirect(),
            "expected a redirect-policy rejection, got: {err}"
        );
    }

    #[tokio::test]
    async fn agent_client_follows_same_host_redirect() {
        // A redirect to the same host:port (only the path changes) must still
        // be followed — the policy blocks cross-host hops, not all redirects.
        // One server issues the redirect, then serves the final 200 on the
        // follow-up request.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            // First request: redirect to /v2 on the same port.
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let redirect = format!(
                "HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:{port}/v2\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            );
            let _ = socket.write_all(redirect.as_bytes()).await;
            let _ = socket.flush().await;
            drop(socket);
            // Second request (the follow): serve 200.
            let (mut socket, _) = listener.accept().await.unwrap();
            let _ = socket.read(&mut buf).await;
            let _ = socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
                .await;
            let _ = socket.flush().await;
        });

        let client = agent_http_client_quick();
        let body = client
            .get(format!("http://127.0.0.1:{port}/v1"))
            .send()
            .await
            .expect("same-host redirect should be followed")
            .text()
            .await
            .unwrap();
        assert_eq!(body, "ok");
    }

    #[test]
    fn redirect_leaves_origin_on_host_port_or_scheme_change() {
        let http_a = reqwest::Url::parse("http://127.0.0.1:8080/v1").unwrap();
        let http_a_path = reqwest::Url::parse("http://127.0.0.1:8080/v2").unwrap();
        let http_b_port = reqwest::Url::parse("http://127.0.0.1:8081/v1").unwrap();
        let http_other_host = reqwest::Url::parse("http://example.test/v1").unwrap();
        let https_a = reqwest::Url::parse("https://127.0.0.1:8080/v1").unwrap();

        assert!(
            !redirect_leaves_origin(&http_a, &http_a_path),
            "same origin path change must still be followable"
        );
        assert!(redirect_leaves_origin(&http_a, &http_b_port));
        assert!(redirect_leaves_origin(&http_a, &http_other_host));
        assert!(
            redirect_leaves_origin(&https_a, &http_a),
            "https→http on the same host must not forward credentials"
        );
    }
}

/// Error from [`idle_guard`]: either the underlying transport failed, or the
/// stream went silent past the idle budget.
#[derive(Debug)]
pub enum StreamGuardError {
    /// No bytes (data or SSE keepalive) for the configured idle window.
    Idle(std::time::Duration),
    Transport(reqwest::Error),
}

impl std::fmt::Display for StreamGuardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle(window) => write!(
                f,
                "provider stream went silent for {}s (connection likely died without close — \
                 e.g. system sleep or NAT timeout); treating as a transient transport failure",
                window.as_secs()
            ),
            Self::Transport(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for StreamGuardError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Idle(_) => None,
            Self::Transport(error) => Some(error),
        }
    }
}

/// Abort a provider byte stream that has gone silent instead of blocking on it
/// forever. A healthy stream always has bytes flowing (data or SSE keepalive
/// comments); a connection that died without a FIN — a laptop sleeping
/// mid-request, a NAT timeout, a half-open TCP — goes quiet indefinitely, and
/// the consumer otherwise blocks on the next chunk forever (observed: a
/// 13-hour zombie one-shot after macOS slept mid-stream). The idle error
/// surfaces through the normal stream-error path, which the retry layer
/// already treats as transient.
///
/// Sleep needs no special clock-jump detection: macOS/Linux monotonic clocks
/// pause during sleep, so the idle window simply resumes counting on wake and
/// fires within one window of the machine waking.
pub fn idle_guard<B, S>(
    stream: S,
    idle_window: std::time::Duration,
) -> impl Stream<Item = Result<B, StreamGuardError>>
where
    S: Stream<Item = Result<B, reqwest::Error>> + Unpin,
{
    futures_util::stream::unfold(Some(stream), move |state| async move {
        let mut stream = state?;
        match tokio::time::timeout(idle_window, stream.next()).await {
            Ok(Some(item)) => Some((item.map_err(StreamGuardError::Transport), Some(stream))),
            Ok(None) => None,
            // One idle error, then end: the connection is presumed dead, so
            // polling it again would only block for another window.
            Err(_) => Some((Err(StreamGuardError::Idle(idle_window)), None)),
        }
    })
}

/// The idle window for [`idle_guard`]: `HI_STREAM_IDLE_TIMEOUT_SECS`, default
/// 240s — generous enough for providers with long silent thinking gaps, small
/// enough that a dead connection is abandoned in minutes rather than forever.
pub fn stream_idle_window() -> std::time::Duration {
    stream_idle_window_or(240)
}

/// Idle window for the xAI Responses adapter. Defaults to 3600s so grok-4.6
/// thinking is not cut off; `HI_STREAM_IDLE_TIMEOUT_SECS` still overrides.
pub(crate) fn xai_stream_idle_window() -> std::time::Duration {
    stream_idle_window_or(DEFAULT_XAI_STREAM_IDLE_SECS)
}

fn stream_idle_window_or(default_secs: u64) -> std::time::Duration {
    let secs = std::env::var("HI_STREAM_IDLE_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value >= 30)
        .unwrap_or(default_secs);
    std::time::Duration::from_secs(secs)
}

#[cfg(test)]
mod idle_guard_tests {
    use super::*;
    use futures_util::StreamExt;

    #[tokio::test(start_paused = true)]
    async fn silent_stream_yields_idle_error_then_ends() {
        let silent = futures_util::stream::pending::<Result<Vec<u8>, reqwest::Error>>();
        let mut guarded = Box::pin(idle_guard(silent, std::time::Duration::from_secs(240)));
        let first = guarded.next().await;
        assert!(
            matches!(first, Some(Err(StreamGuardError::Idle(_)))),
            "expected idle error, got {first:?}"
        );
        assert!(guarded.next().await.is_none(), "guard must end after idle");
    }

    #[tokio::test]
    async fn flowing_stream_passes_through_and_ends_cleanly() {
        let items = futures_util::stream::iter(vec![
            Ok::<Vec<u8>, reqwest::Error>(b"a".to_vec()),
            Ok(b"b".to_vec()),
        ]);
        let guarded = idle_guard(items, std::time::Duration::from_secs(240));
        let collected: Vec<_> = guarded.collect().await;
        assert_eq!(collected.len(), 2);
        assert!(collected.iter().all(Result::is_ok));
    }
}
