//! Shared HTTP send-with-retry used by every adapter.
//!
//! Retries the *initial* request (before streaming begins) on transient
//! failures — connection/timeout errors and 429 / 5xx responses — with
//! exponential backoff. Mid-stream failures are not retried (they'd duplicate
//! already-emitted output).

use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures_util::{Stream, StreamExt};
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::Deserialize;

use crate::provider::ServedModel;

const MAX_RETRIES: u32 = 3;
const BASE_DELAY_MS: u64 = 250;

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
                if attempt < MAX_RETRIES && is_retryable_status(response.status()) {
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

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn is_retryable_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect() || err.is_request()
}

async fn backoff(attempt: u32) {
    let delay = BASE_DELAY_MS * 2u64.pow(attempt - 1);
    tokio::time::sleep(Duration::from_millis(delay)).await;
}

#[cfg(test)]
mod tests {
    use super::*;

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
