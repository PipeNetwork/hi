//! OpenAI Chat Completions adapter.
//!
//! Covers OpenRouter, pipenetwork.ai, and local servers (Ollama, llama.cpp,
//! LM Studio, vLLM) — they differ only by base URL and API key.
//!
//! Request translation lives in [`request`], and SSE stream parsing in
//! [`stream`]; this module holds the [`OpenAiProvider`] struct and its
//! [`Provider`] impl, which wires the two together.

mod request;
mod stream;

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::header;
use serde_json::{Value, json};

use crate::provider::{Provider, ProviderError, ProviderErrorKind};
use crate::token::{StaticToken, TokenSource};
use crate::types::{
    ChatRequest, CompatMode, Completion, RateLimitBucket, RateLimitState, StreamEvent, ToolMode,
    Usage, estimate_messages_tokens,
};

// Capacity (429) retries: a local single-slot server (`--max-active-requests 1`)
// 429s whenever a second request overlaps, so the budget is sized to ride out a
// busy local sidecar rather than a brief cloud throttle.
const MAX_CAPACITY_HTTP_RETRIES: u32 = 5;
const DEFAULT_CAPACITY_RETRY_SECS: u64 = 2;
const MAX_CAPACITY_RETRY_SECS: u64 = 30;

pub struct OpenAiProvider {
    http: reqwest::Client,
    base_url: String,
    auth: Arc<dyn TokenSource>,
    pipe_metadata: bool,
}

impl OpenAiProvider {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self::with_token_source(base_url, Arc::new(StaticToken(api_key)))
    }

    /// Build against a credential that can change mid-session (OAuth). The
    /// token is re-read per request, and an auth rejection triggers one
    /// refresh-and-retry instead of failing the turn.
    pub fn with_token_source(base_url: String, auth: Arc<dyn TokenSource>) -> Self {
        Self {
            http: crate::http::agent_http_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth,
            pipe_metadata: false,
        }
    }

    pub fn new_pipenetwork(base_url: String, api_key: String) -> Self {
        let mut provider = Self::new(base_url, api_key);
        provider.pipe_metadata = true;
        provider
    }

    pub fn new_unix(base_url: String, api_key: String, socket: &std::path::Path) -> Self {
        Self {
            http: crate::http::agent_http_client_for_socket(Some(socket)),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth: Arc::new(StaticToken(api_key)),
            pipe_metadata: false,
        }
    }

    fn request_metadata(&self, request: &ChatRequest) -> Option<Value> {
        if !self.pipe_metadata {
            return None;
        }
        let uses_tools =
            !request.tools.is_empty() && request.profile.tool_mode != ToolMode::ChatOnly;
        let mut metadata = json!({
            "endpoint_name": "pipenetworkai",
            "request_type": if uses_tools {
                "agent_tool_invocation"
            } else {
                "code_generation"
            },
            "selected_agent_model": request.model,
            "max_output_tokens": request.max_tokens,
        });
        if uses_tools {
            metadata["agent_turn_kind"] = json!("root_agent_turn");
        }
        Some(metadata)
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let url = format!("{}/chat/completions", self.base_url);
        let attempts = request::request_attempts(&request);
        let mut last_error: Option<ProviderError> = None;
        let mut resp = None;
        let mut idx = 0;
        let mut capacity_retries = 0;
        let mut auth_refreshed = false;
        while idx < attempts.len() {
            let attempt = attempts[idx];
            let request_metadata = self.request_metadata(&request);
            let body = request::build_body(&request, attempt, request_metadata.as_ref());
            // Read the token per attempt: a refresh below replaces it in place.
            let token = self.auth.token().await;
            let response =
                crate::http::send_with_retry(self.http.post(&url).bearer_auth(&token).json(&body))
                    .await
                    .context("request to model endpoint failed")?;

            if response.status().is_success() {
                if let Some(status) = attempt.status {
                    sink(StreamEvent::Status(status.into()));
                }
                resp = Some(response);
                break;
            }

            let status = response.status();
            let retry_after = retry_after_header_seconds(&response);
            let rate_limits = rate_limits_from_headers(response.headers());
            let text = response.text().await.unwrap_or_default();
            let kind = request::classify_http_error(status, &text);
            // An expiring credential (OAuth) can die mid-session. Re-mint it and
            // replay the same attempt once. Guarded by `auth_refreshed` so a
            // source that refreshes to an equally-rejected token can't loop, and
            // skipped entirely for API keys, whose `refresh` returns false.
            if kind == ProviderErrorKind::Auth && !auth_refreshed && self.auth.refresh().await {
                auth_refreshed = true;
                sink(StreamEvent::Status(
                    "credential expired; refreshed it — retrying".to_string(),
                ));
                continue;
            }
            if kind == ProviderErrorKind::CapacityUnavailable
                && capacity_retries < MAX_CAPACITY_HTTP_RETRIES
            {
                capacity_retries += 1;
                let delay_secs = retry_after
                    .or_else(|| request::capacity_retry_after_seconds(&text))
                    .unwrap_or(DEFAULT_CAPACITY_RETRY_SECS)
                    .min(MAX_CAPACITY_RETRY_SECS);
                sink(StreamEvent::Status(format!(
                    "capacity temporarily unavailable; retrying in {delay_secs}s ({capacity_retries}/{MAX_CAPACITY_HTTP_RETRIES})"
                )));
                if delay_secs > 0 {
                    tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                }
                continue;
            }
            let mut error = ProviderError::new(kind, format!("API error {status}: {text}"));
            if let Some(rate_limits) = rate_limits {
                error = error.with_usage(Usage {
                    rate_limits: Some(rate_limits),
                    ..Default::default()
                });
            }
            last_error = Some(error);
            if request.profile.compat == CompatMode::Strict {
                break;
            }
            // Degrade toward the attempt that actually addresses this error.
            // Tool rejection is surfaced: an agent turn that advertised tools
            // cannot safely continue chat-only because it would be unable to
            // inspect or modify the workspace.
            match request::next_degraded_attempt(&attempts, idx, kind, &text) {
                Some(next) => idx = next,
                None => break,
            }
        }
        let Some(resp) = resp else {
            return Err(last_error
                .unwrap_or_else(|| {
                    ProviderError::new(ProviderErrorKind::Other, "request failed before streaming")
                })
                .into());
        };
        let rate_limits = rate_limits_from_headers(resp.headers());

        // `debug_tap` optionally echoes the raw wire bytes when HI_DEBUG_STREAM
        // is set. Reduce the stream to its SSE `data` strings so the collection
        // loop is provider-agnostic and unit-testable.
        let stream = crate::http::debug_tap(resp.bytes_stream())
            .eventsource()
            .map(|res| res.map(|event| event.data).context("error reading stream"));
        let estimated_input_tokens = estimate_messages_tokens(&request.messages);
        let mut completion = stream::collect_completion(Box::pin(stream), sink)
            .await
            .map_err(|err| {
                stream::classify_stream_error(err).with_usage(Usage {
                    input_tokens: estimated_input_tokens,
                    output_tokens: 0,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 0,
                    input_includes_cache: true,
                    context_occupancy: estimated_input_tokens,
                    rate_limits,
                    estimated: true,
                })
            })?;
        stream::backfill_missing_usage(&mut completion, &request);
        completion.usage.rate_limits = completion.usage.rate_limits.or(rate_limits);
        if completion.content.is_empty() {
            return Err(ProviderError::new(
                ProviderErrorKind::EmptyCompletion,
                "model returned an empty completion",
            )
            .with_usage(completion.usage)
            .into());
        }
        Ok(completion)
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ServedModel>> {
        let url = format!("{}/models", self.base_url);
        let token = self.auth.token().await;
        crate::http::fetch_models(self.http.get(&url).bearer_auth(&token)).await
    }
}

fn retry_after_header_seconds(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn rate_limits_from_headers(headers: &header::HeaderMap) -> Option<RateLimitState> {
    if !headers
        .keys()
        .any(|name| name.as_str().starts_with("x-ratelimit-"))
    {
        return None;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let state = RateLimitState {
        requests_min: rate_limit_bucket(headers, "requests", ""),
        requests_hour: rate_limit_bucket(headers, "requests", "-1h"),
        tokens_min: rate_limit_bucket(headers, "tokens", ""),
        tokens_hour: rate_limit_bucket(headers, "tokens", "-1h"),
        captured_at_unix_seconds: now,
    };
    state.has_data().then_some(state)
}

fn rate_limit_bucket(
    headers: &header::HeaderMap,
    resource: &'static str,
    suffix: &'static str,
) -> RateLimitBucket {
    RateLimitBucket {
        limit: header_number(headers, &format!("x-ratelimit-limit-{resource}{suffix}")),
        remaining: header_number(
            headers,
            &format!("x-ratelimit-remaining-{resource}{suffix}"),
        ),
        reset_seconds: header_number(headers, &format!("x-ratelimit-reset-{resource}{suffix}")),
    }
}

fn header_number(headers: &header::HeaderMap, name: &str) -> u64 {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| value as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{OpenAiProvider, rate_limits_from_headers};
    use crate::provider::{Provider, ProviderErrorKind, provider_error_kind, provider_error_usage};
    use crate::test_support::{FakeOpenAiServer, Response, sse_text};
    use crate::types::{
        ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode, ToolSpec,
    };
    use reqwest::header::{HeaderMap, HeaderValue};

    /// A `TokenSource` whose token changes exactly once, on the first refresh —
    /// standing in for an OAuth credential that expires mid-session.
    struct RotatingToken {
        current: std::sync::Mutex<String>,
        refreshes: std::sync::atomic::AtomicUsize,
    }

    impl RotatingToken {
        fn new() -> std::sync::Arc<Self> {
            std::sync::Arc::new(Self {
                current: std::sync::Mutex::new("stale-token".to_string()),
                refreshes: std::sync::atomic::AtomicUsize::new(0),
            })
        }
    }

    #[async_trait::async_trait]
    impl crate::token::TokenSource for RotatingToken {
        async fn token(&self) -> String {
            self.current.lock().unwrap().clone()
        }
        async fn refresh(&self) -> bool {
            self.refreshes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            *self.current.lock().unwrap() = "fresh-token".to_string();
            true
        }
    }

    /// An expired OAuth token must not kill the turn: the provider re-mints it
    /// and replays the same request with the new credential.
    #[tokio::test]
    async fn expired_token_is_refreshed_and_the_request_is_replayed() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(401, r#"{"error":"invalid access token"}"#),
            Response::sse(sse_text("ok")),
        ]) else {
            return;
        };
        let auth = RotatingToken::new();
        let provider = OpenAiProvider::with_token_source(server.url().to_string(), auth.clone());
        let mut sink = |_event| {};
        let completion = provider
            .stream(request(vec![], Default::default()), &mut sink)
            .await
            .unwrap();

        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "ok"));
        assert_eq!(
            auth.refreshes.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "exactly one refresh should be attempted"
        );
        let sent = server.authorizations();
        assert_eq!(sent.len(), 2, "the request should be replayed once");
        assert_eq!(sent[0].as_deref(), Some("bearer stale-token"));
        assert_eq!(
            sent[1].as_deref(),
            Some("bearer fresh-token"),
            "the replay must carry the refreshed credential, not the stale one"
        );
    }

    /// A key that is simply wrong must fail fast. `StaticToken::refresh` returns
    /// false, so there is nothing to retry and the user hears about their key.
    #[tokio::test]
    async fn a_static_api_key_is_not_retried_on_auth_failure() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(401, r#"{"error":"invalid api key"}"#),
            Response::sse(sse_text("should never be reached")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "sk-wrong".into());
        let mut sink = |_event| {};
        let error = provider
            .stream(request(vec![], Default::default()), &mut sink)
            .await
            .unwrap_err();

        assert_eq!(provider_error_kind(&error), Some(ProviderErrorKind::Auth));
        assert_eq!(
            server.authorizations().len(),
            1,
            "a static key has no second credential to try"
        );
    }

    #[tokio::test]
    async fn fake_server_rejects_stream_options_then_succeeds() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(400, r#"{"error":"stream_options unsupported"}"#),
            Response::sse(sse_text("ok")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let request = request(vec![], Default::default());
        let mut statuses = Vec::new();
        let mut sink = |event| {
            if let StreamEvent::Status(status) = event {
                statuses.push(status);
            }
        };
        let completion = provider.stream(request, &mut sink).await.unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "ok"));
        assert!(
            completion.usage.input_tokens > 0,
            "fallback request gets estimated input usage: {:?}",
            completion.usage
        );
        assert!(
            completion.usage.output_tokens > 0,
            "fallback request gets estimated output usage: {:?}",
            completion.usage
        );
        assert!(
            statuses.iter().any(|s| s.contains("stream_options")),
            "{statuses:?}"
        );
        let bodies = server.bodies();
        assert!(bodies[0].contains("stream_options"));
        assert!(!bodies[1].contains("stream_options"));
    }

    #[tokio::test]
    async fn success_captures_rate_limit_headers() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::sse(sse_text("ok"))
                .with_header("x-ratelimit-limit-requests", "60")
                .with_header("x-ratelimit-remaining-requests", "58")
                .with_header("x-ratelimit-reset-requests", "12")
                .with_header("x-ratelimit-limit-tokens", "100000")
                .with_header("x-ratelimit-remaining-tokens", "88000")
                .with_header("x-ratelimit-reset-tokens", "42"),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap();
        let limits = completion
            .usage
            .rate_limits
            .expect("rate limit headers parsed");
        assert_eq!(limits.requests_min.limit, 60);
        assert_eq!(limits.requests_min.remaining, 58);
        assert_eq!(limits.requests_min.reset_seconds, 12);
        assert_eq!(limits.tokens_min.limit, 100000);
        assert_eq!(limits.tokens_min.remaining, 88000);
        assert_eq!(limits.tokens_min.reset_seconds, 42);
    }

    #[tokio::test]
    async fn http_errors_carry_rate_limit_headers_in_usage() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(429, r#"{"error":"too many requests"}"#)
                .with_header("x-ratelimit-limit-requests", "60")
                .with_header("x-ratelimit-remaining-requests", "0")
                .with_header("x-ratelimit-reset-requests", "55"),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::RateLimit)
        );
        let usage = provider_error_usage(&err);
        let limits = usage.rate_limits.expect("rate limit headers parsed");
        assert_eq!(limits.requests_min.limit, 60);
        assert_eq!(limits.requests_min.remaining, 0);
        assert_eq!(limits.requests_min.reset_seconds, 55);
    }

    #[test]
    fn parses_hourly_rate_limit_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-ratelimit-limit-requests-1h",
            HeaderValue::from_static("1200"),
        );
        headers.insert(
            "x-ratelimit-remaining-requests-1h",
            HeaderValue::from_static("1197"),
        );
        headers.insert(
            "x-ratelimit-reset-requests-1h",
            HeaderValue::from_static("3580"),
        );
        let limits = rate_limits_from_headers(&headers).expect("headers parsed");
        assert_eq!(limits.requests_hour.limit, 1200);
        assert_eq!(limits.requests_hour.remaining, 1197);
        assert_eq!(limits.requests_hour.reset_seconds, 3580);
        assert!(limits.captured_at_unix_seconds > 0);
    }

    #[tokio::test]
    async fn fake_server_rejects_tools_fails_fast() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"tools unsupported"}"#,
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![tool()], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::UnsupportedTools)
        );
        let bodies = server.bodies();
        assert_eq!(bodies.len(), 1);
        assert!(bodies[0].contains("\"tools\""));
    }

    #[tokio::test]
    async fn pipenetwork_provider_sends_agent_metadata_for_tool_requests() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_text("ok"))]) else {
            return;
        };
        let provider = OpenAiProvider::new_pipenetwork(server.url().to_string(), "test".into());

        provider
            .stream(request(vec![tool()], Default::default()), &mut |_| {})
            .await
            .unwrap();

        let bodies = server.bodies();
        let body: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
        assert_eq!(body["metadata"]["endpoint_name"], "pipenetworkai");
        assert_eq!(body["metadata"]["request_type"], "agent_tool_invocation");
        assert_eq!(body["metadata"]["agent_turn_kind"], "root_agent_turn");
        assert_eq!(body["metadata"]["selected_agent_model"], "m");
        assert_eq!(body["metadata"]["max_output_tokens"], 16);
    }

    #[tokio::test]
    async fn generic_openai_provider_does_not_send_pipe_metadata() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_text("ok"))]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());

        provider
            .stream(request(vec![tool()], Default::default()), &mut |_| {})
            .await
            .unwrap();

        let bodies = server.bodies();
        let body: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
        assert!(body.get("metadata").is_none());
    }

    #[tokio::test]
    async fn required_tool_mode_does_not_degrade() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"tools unsupported"}"#,
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let profile = RequestProfile {
            tool_mode: ToolMode::Required,
            ..Default::default()
        };
        let err = provider
            .stream(request(vec![tool()], profile), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::UnsupportedTools)
        );
        assert_eq!(server.bodies().len(), 1);
    }

    #[tokio::test]
    async fn auth_rate_limit_and_malformed_stream_are_classified() {
        for (status, kind) in [
            (401, ProviderErrorKind::Auth),
            (403, ProviderErrorKind::Auth),
            (429, ProviderErrorKind::RateLimit),
        ] {
            let Some(server) =
                FakeOpenAiServer::new(vec![Response::json(status, r#"{"error":"nope"}"#)])
            else {
                return;
            };
            let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
            let err = provider
                .stream(request(vec![], Default::default()), &mut |_| {})
                .await
                .unwrap_err();
            assert_eq!(provider_error_kind(&err), Some(kind), "status {status}");
        }

        let Some(server) = FakeOpenAiServer::new(vec![Response::sse("data: {not-json}\n\n")])
        else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::MalformedStream)
        );
    }

    #[tokio::test]
    async fn request_too_large_400_is_classified() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"chat input exceeds the maximum allowed size of 131072 bytes","error_type":"invalid_request_error"}"#,
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::RequestTooLarge)
        );
    }

    #[tokio::test]
    async fn model_temporarily_unavailable_is_not_capacity() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            409,
            r#"{"error":"model temporarily unavailable","code":"capacity_unavailable"}"#,
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::ModelUnavailable)
        );
    }

    #[tokio::test]
    async fn soft_protocol_http_errors_are_classified() {
        for (body, expected) in [
            (
                r#"{"error":"model output did not satisfy the tool protocol"}"#,
                ProviderErrorKind::ToolProtocol,
            ),
            (
                r#"{"error":"quality_rejected: provider quality check failed"}"#,
                ProviderErrorKind::QualityRejected,
            ),
            (
                r#"{"error":"request not found"}"#,
                ProviderErrorKind::MalformedStream,
            ),
        ] {
            let Some(server) = FakeOpenAiServer::new(vec![Response::json(400, body)]) else {
                return;
            };
            let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
            let err = provider
                .stream(request(vec![], Default::default()), &mut |_| {})
                .await
                .unwrap_err();
            assert_eq!(provider_error_kind(&err), Some(expected), "{body}");
        }
    }

    #[tokio::test]
    async fn server_error_retries_then_succeeds() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(500, r#"{"error":"temporary server error"}"#),
            Response::sse(sse_text("recovered")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "recovered"));
        assert_eq!(server.bodies().len(), 2);
    }

    #[tokio::test]
    async fn capacity_unavailable_conflict_retries_then_succeeds() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(
                409,
                r#"{"error":"capacity temporarily unavailable","code":"capacity_unavailable","retry_after_seconds":0}"#,
            ),
            Response::sse(sse_text("recovered")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let mut statuses = Vec::new();
        let mut sink = |event| {
            if let StreamEvent::Status(status) = event {
                statuses.push(status);
            }
        };

        let completion = provider
            .stream(request(vec![], Default::default()), &mut sink)
            .await
            .unwrap();

        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "recovered"));
        assert_eq!(server.bodies().len(), 2);
        assert!(
            statuses
                .iter()
                .any(|status| status.contains("capacity temporarily unavailable")),
            "{statuses:?}"
        );
    }

    #[tokio::test]
    async fn empty_completion_error_carries_usage() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":3}}\n\n",
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::EmptyCompletion)
        );
        assert_eq!(crate::provider::provider_error_usage(&err).input_tokens, 42);
        assert_eq!(crate::provider::provider_error_usage(&err).output_tokens, 3);
    }

    #[tokio::test]
    async fn streamed_error_payload_is_not_reported_as_empty_completion() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(
            "data: {\"error\":{\"message\":\"capacity temporarily unavailable\"}}\n\ndata: [DONE]\n\n",
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::CapacityUnavailable)
        );
        assert!(
            err.to_string().contains("capacity temporarily unavailable"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn malformed_stream_error_does_not_charge_full_output_budget() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse("data: {malformed-json}\n\n")])
        else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        let usage = crate::provider::provider_error_usage(&err);

        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::MalformedStream)
        );
        assert!(usage.input_tokens > 0, "input estimate should be retained");
        assert_eq!(
            usage.output_tokens, 0,
            "failed stream should not bill the full max_tokens output budget"
        );
    }

    #[tokio::test]
    async fn fake_server_stream_can_finish_without_done() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n",
        )]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "done"));
    }

    fn request(tools: Vec<ToolSpec>, profile: RequestProfile) -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: tools.into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile,
        }
    }

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run shell command".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        }
    }
}
