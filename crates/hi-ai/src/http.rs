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

#[derive(Deserialize)]
struct ModelsList {
    data: Vec<ModelEntry>,
}
/// One `/models` entry. Only `id` is standard; the rest are terminaili-style
/// extensions that other endpoints simply omit (hence all optional).
#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    max_context_tokens: Option<u32>,
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
            // Reported rates are per token; the rest of the app uses per-1M.
            price: match (self.input_token_rate, self.output_token_rate) {
                (Some(i), Some(o)) => Some((i * 1_000_000.0, o * 1_000_000.0)),
                _ => None,
            },
            status: self.status,
            available: self.available.unwrap_or(true),
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

/// Give up on a stream if the model produces no output — content, reasoning, or
/// tool tokens — for this long (default 120s, override with `HI_STREAM_TIMEOUT`
/// in seconds). Keep-alive heartbeats do NOT count as progress: a provider that
/// only sends heartbeats is stalled, not working. (Adapters reset their deadline
/// whenever they emit a real token.)
pub fn stream_idle_timeout() -> Duration {
    std::env::var("HI_STREAM_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(120))
}

/// Once output *has* started flowing, end the stream this long after tokens stop
/// if no completion signal (`finish_reason`/`[DONE]`/socket close) arrives —
/// covers a provider that streams a full answer then holds the connection open
/// without terminating it (default 15s, override with `HI_STREAM_STALL`). Much
/// shorter than the cold-start [`stream_idle_timeout`] because a multi-second
/// gap *between* tokens means the stream has effectively ended, whereas a slow
/// time-to-first-token can be a legitimately queued request.
pub fn stream_stall_timeout() -> Duration {
    std::env::var("HI_STREAM_STALL")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(15))
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
    fn parses_terminaili_model_metadata() {
        // terminaili extends /models with window, per-token rates, and health.
        let json = r#"{"data":[
            {"id":"ipop/coder-balanced","max_context_tokens":1000000,
             "input_token_rate":0.000001,"output_token_rate":0.000002,
             "status":"available","available":true},
            {"id":"grok","status":"degraded","available":true},
            {"id":"down","available":false}
        ]}"#;
        let list: ModelsList = serde_json::from_str(json).unwrap();
        let served: Vec<ServedModel> = list.data.into_iter().map(ModelEntry::into_served).collect();

        assert_eq!(served[0].context_window, Some(1_000_000));
        assert_eq!(served[0].price, Some((1.0, 2.0))); // per-token → per-1M
        assert_eq!(served[0].health(), None, "available is healthy");

        assert_eq!(served[1].context_window, None);
        assert_eq!(served[1].health(), Some("degraded"));

        assert_eq!(
            served[2].health(),
            Some("unavailable"),
            "available:false flagged"
        );
    }
}
