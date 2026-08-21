//! Native xAI Responses API adapter.
//!
//! `--provider xai` posts to `/v1/responses` with `store: false`, flat function
//! tools, and encrypted-reasoning replay. Chat Completions is not used.

mod request;
mod stream;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::header;
use serde_json::Value;

use crate::provider::{Provider, ProviderError, ProviderErrorKind};
use crate::token::{StaticToken, TokenSource};
use crate::types::{
    ChatRequest, Completion, RateLimitBucket, RateLimitState, StreamEvent, Usage,
    estimate_request_input_tokens,
};

pub struct XaiProvider {
    http: reqwest::Client,
    base_url: String,
    auth: Arc<dyn TokenSource>,
}

impl XaiProvider {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self::with_token_source(base_url, Arc::new(StaticToken(api_key)))
    }

    /// Build against a credential that can change mid-session (OAuth). The
    /// token is re-read per request, and an auth rejection triggers one
    /// refresh-and-retry instead of failing the turn.
    pub fn with_token_source(base_url: String, auth: Arc<dyn TokenSource>) -> Self {
        Self {
            http: crate::http::agent_http_client_xai(),
            base_url: base_url.trim_end_matches('/').to_string(),
            auth,
        }
    }
}

#[async_trait]
impl Provider for XaiProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let body = request::build_body(&request);
        let url = format!("{}/responses", self.base_url);
        let correlation_id = canonical_request_id(request.request_id.as_deref());
        let idempotency_key = request_idempotency_key(&correlation_id, &body);
        let mut auth_refreshed = false;

        loop {
            let token = self.auth.token().await;
            let response = crate::http::send_with_retry(
                self.http
                    .post(&url)
                    .bearer_auth(&token)
                    .header("x-request-id", &correlation_id)
                    .header("x-request-attempt", request.retry_attempt.to_string())
                    .header("idempotency-key", &idempotency_key)
                    .json(&body),
            )
            .await
            .map_err(|error| {
                ProviderError::new(
                    ProviderErrorKind::Outage,
                    format!("request to xAI Responses endpoint failed: {error}"),
                )
                .with_api_contract(None, Some(true), None)
            })?;

            if response.status().is_success() {
                sink(StreamEvent::WireAudit(Box::new(request::wire_audit(
                    &request,
                    &self.base_url,
                    &body,
                    true,
                    Some(response.status().as_u16()),
                ))));
                let rate_limits = rate_limits_from_headers(response.headers());
                let stream = crate::http::idle_guard(
                    crate::http::debug_tap(response.bytes_stream()),
                    crate::http::xai_stream_idle_window(),
                )
                .eventsource()
                .map(|res| {
                    res.map_err(|err| anyhow::anyhow!(err))
                        .context("error reading stream")
                });
                let estimated_input_tokens =
                    estimate_request_input_tokens(&request.messages, &request.tools);
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
                if completion.content.is_empty()
                    && completion.refusal.is_none()
                    && completion.stop_reason.as_deref() != Some("refusal")
                {
                    return Err(ProviderError::new(
                        ProviderErrorKind::EmptyCompletion,
                        "model returned an empty completion",
                    )
                    .with_usage(completion.usage)
                    .into());
                }
                return Ok(completion);
            }

            let status = response.status();
            sink(StreamEvent::WireAudit(Box::new(request::wire_audit(
                &request,
                &self.base_url,
                &body,
                false,
                Some(status.as_u16()),
            ))));
            let retry_after = retry_after_header_seconds(&response);
            let rate_limits = rate_limits_from_headers(response.headers());
            let text = response.text().await.unwrap_or_default();
            let kind = request::classify_http_error(status, &text);
            if kind == ProviderErrorKind::Auth
                && !crate::is_billing_or_quota_text(&text)
                && !auth_refreshed
                && self.auth.refresh().await
            {
                auth_refreshed = true;
                sink(StreamEvent::Status(
                    "credential expired; refreshed it — retrying".to_string(),
                ));
                continue;
            }
            let mut error = request::provider_error_from_http(status, &text);
            if error.retry_after_seconds.is_none() {
                error.retry_after_seconds = retry_after;
            }
            if let Some(rate_limits) = rate_limits {
                error = error.with_usage(Usage {
                    rate_limits: Some(rate_limits),
                    ..Default::default()
                });
            }
            return Err(error.into());
        }
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ServedModel>> {
        let url = format!("{}/models", self.base_url);
        let token = self.auth.token().await;
        crate::http::fetch_models(self.http.get(&url).bearer_auth(&token)).await
    }
}

fn canonical_request_id(request_id: Option<&str>) -> String {
    let request_id = request_id.unwrap_or_default().trim();
    if !request_id.is_empty()
        && request_id.len() <= 96
        && request_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        request_id.to_string()
    } else {
        format!("hi_{}", uuid::Uuid::new_v4().simple())
    }
}

fn request_idempotency_key(correlation_id: &str, body: &Value) -> String {
    let encoded = serde_json::to_vec(body).unwrap_or_default();
    let digest = blake3::hash(&encoded).to_hex();
    format!("{correlation_id}:{}", &digest[..24])
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
    use super::XaiProvider;
    use crate::provider::{Provider, ProviderErrorKind, provider_error_kind};
    use crate::test_support::{FakeOpenAiServer, Response};
    use crate::types::{
        ChatRequest, Content, Message, ReasoningEffort, RequestProfile, StreamEvent, ToolMode,
        ToolSpec,
    };

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

    fn request(tools: Vec<ToolSpec>, profile: RequestProfile) -> ChatRequest {
        ChatRequest {
            model: "grok-4.6".into(),
            request_id: Some("req_test".into()),
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: tools.into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: Some(0.4),
            thinking_budget: None,
            reasoning_effort: None,
            profile,
        }
    }

    fn bash_tool() -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        }
    }

    fn sse_event(event: &str, data: &str) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }

    fn sse_text(text: &str) -> String {
        let encoded = serde_json::to_string(text).unwrap();
        format!(
            "{}{}",
            sse_event(
                "response.output_text.delta",
                &format!(r#"{{"type":"response.output_text.delta","delta":{encoded}}}"#)
            ),
            sse_event(
                "response.completed",
                r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":2}}}"#
            )
        )
    }

    fn sse_function_call() -> String {
        format!(
            "{}{}",
            sse_event(
                "response.output_item.done",
                r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"bash","arguments":"{\"command\":\"ls\"}"}}"#
            ),
            sse_event(
                "response.completed",
                r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":12,"output_tokens":8},"output":[{"type":"reasoning","encrypted_content":"enc-blob","summary":[{"type":"summary_text","text":"list files"}]}]}}"#
            )
        )
    }

    #[tokio::test]
    async fn posts_responses_with_flat_tools_and_store_false() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_text("ok"))]) else {
            return;
        };
        let provider = XaiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(
                request(
                    vec![bash_tool()],
                    RequestProfile {
                        tool_mode: ToolMode::Auto,
                        ..Default::default()
                    },
                ),
                &mut |_| {},
            )
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "ok"));
        let body: serde_json::Value = serde_json::from_str(&server.bodies()[0]).unwrap();
        assert_eq!(body["store"], false);
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert_eq!(body["tools"][0]["name"], "bash");
        assert!(body["tools"][0].get("function").is_none());
        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("messages").is_none());
        assert!(body.get("input").is_some());
    }

    #[tokio::test]
    async fn whole_chunk_function_call_becomes_native_tool_call() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_function_call())]) else {
            return;
        };
        let provider = XaiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(
                request(
                    vec![bash_tool()],
                    RequestProfile {
                        tool_mode: ToolMode::Required,
                        ..Default::default()
                    },
                ),
                &mut |_| {},
            )
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_str(&server.bodies()[0]).unwrap();
        assert_eq!(body["tool_choice"], "required");
        assert!(matches!(
            completion.content.iter().find(|c| matches!(c, Content::ToolCall { .. })),
            Some(Content::ToolCall { id, name, arguments })
                if id == "call_1" && name == "bash" && arguments.contains("ls")
        ));
        assert!(matches!(
            completion.content.iter().find(|c| matches!(c, Content::Thinking { .. })),
            Some(Content::Thinking { text, signature: Some(sig) })
                if text == "list files" && sig == "enc-blob"
        ));
        assert_eq!(
            completion.tool_call_channel,
            crate::types::ToolCallChannel::Native
        );
    }

    #[tokio::test]
    async fn follow_up_replays_function_output_and_encrypted_reasoning() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_text("hi-ai"))]) else {
            return;
        };
        let provider = XaiProvider::new(server.url().to_string(), "test".into());
        let mut req = request(vec![bash_tool()], Default::default());
        req.messages = vec![
            Message::user("read the crate name"),
            Message::assistant(vec![
                Content::Thinking {
                    text: "read cargo".into(),
                    signature: Some("enc-2".into()),
                },
                Content::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"crates/hi-ai/Cargo.toml"}"#.into(),
                },
            ]),
            Message::tool_result("call_1", "name = \"hi-ai\""),
        ]
        .into();
        provider.stream(req, &mut |_| {}).await.unwrap();
        let body: serde_json::Value = serde_json::from_str(&server.bodies()[0]).unwrap();
        let input = body["input"].as_array().unwrap();
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "reasoning" && item["encrypted_content"] == "enc-2")
        );
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "function_call" && item["call_id"] == "call_1")
        );
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "function_call_output"
                    && item["output"] == "name = \"hi-ai\"")
        );
    }

    #[tokio::test]
    async fn minimal_effort_is_low_on_the_wire() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_text("ok"))]) else {
            return;
        };
        let provider = XaiProvider::new(server.url().to_string(), "test".into());
        let mut req = request(vec![], Default::default());
        req.reasoning_effort = Some(ReasoningEffort::Minimal);
        provider.stream(req, &mut |_| {}).await.unwrap();
        let body: serde_json::Value = serde_json::from_str(&server.bodies()[0]).unwrap();
        assert_eq!(body["reasoning"]["effort"], "low");
    }

    #[tokio::test]
    async fn expired_token_is_refreshed_and_the_request_is_replayed() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(401, r#"{"error":"invalid access token"}"#),
            Response::sse(sse_text("ok")),
        ]) else {
            return;
        };
        let auth = RotatingToken::new();
        let provider = XaiProvider::with_token_source(server.url().to_string(), auth.clone());
        let mut sink = |_event: StreamEvent| {};
        let completion = provider
            .stream(request(vec![], Default::default()), &mut sink)
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "ok"));
        assert_eq!(auth.refreshes.load(std::sync::atomic::Ordering::SeqCst), 1);
        let sent = server.authorizations();
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[0].as_deref(), Some("bearer stale-token"));
        assert_eq!(sent[1].as_deref(), Some("bearer fresh-token"));
        assert_eq!(server.request_ids()[0], server.request_ids()[1]);
        assert_eq!(server.idempotency_keys()[0], server.idempotency_keys()[1]);
    }

    #[tokio::test]
    async fn a_static_api_key_is_not_retried_on_auth_failure() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(401, r#"{"error":"invalid api key"}"#),
            Response::sse(sse_text("should never be reached")),
        ]) else {
            return;
        };
        let provider = XaiProvider::new(server.url().to_string(), "sk-wrong".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(provider_error_kind(&err), Some(ProviderErrorKind::Auth));
        assert_eq!(server.bodies().len(), 1);
    }
}
