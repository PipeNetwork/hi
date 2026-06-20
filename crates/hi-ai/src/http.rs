//! Shared HTTP send-with-retry used by every adapter.
//!
//! Retries the *initial* request (before streaming begins) on transient
//! failures — connection/timeout errors and 429 / 5xx responses — with
//! exponential backoff. Mid-stream failures are not retried (they'd duplicate
//! already-emitted output).

use std::time::Duration;

use anyhow::{Result, bail};
use futures_util::{Stream, StreamExt};
use reqwest::{RequestBuilder, Response, StatusCode};

const MAX_RETRIES: u32 = 3;
const BASE_DELAY_MS: u64 = 250;

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
