//! OpenAI Chat Completions adapter.
//!
//! Covers OpenRouter, pipenetwork.ai, and local servers (Ollama, llama.cpp,
//! LM Studio, vLLM) — they differ only by base URL and API key.
//!
//! Request translation lives in [`request`], and SSE stream parsing in
//! [`stream`]; this module holds the [`OpenAiProvider`] struct and its
//! [`Provider`] impl, which wires the two together.

mod deepseek;
mod request;
mod stream;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest::header;
use serde_json::{Value, json};

use crate::provider::{Provider, ProviderError, ProviderErrorKind};
use crate::token::{StaticToken, TokenSource};
use crate::types::{
    ChatRequest, CompatMode, Completion, Content, OutputTokenParameter, RateLimitBucket,
    RateLimitState, StreamEvent, ToolMode, Usage, WireAudit, estimate_request_input_tokens,
};

pub struct OpenAiProvider {
    http: reqwest::Client,
    base_url: String,
    auth: Arc<dyn TokenSource>,
    pipe_metadata: bool,
    /// Endpoint/model pairs that have explicitly reported that strict tool
    /// schemas are unsupported. Gateways often proxy models with different
    /// capabilities, so this is learned from the response rather than baked
    /// into model-name checks.
    deepseek_strict_cache: Arc<Mutex<HashMap<String, bool>>>,
    /// Gateway/model pairs that cannot preserve DeepSeek reasoning fields.
    /// `false` means later requests start with thinking disabled.
    deepseek_thinking_cache: Arc<Mutex<HashMap<String, bool>>>,
    /// Endpoint/model output-token spelling learned from a successful
    /// compatibility retry. This is intentionally process-local: a stale
    /// persisted capability must never suppress the bounded probe.
    output_token_cache: Arc<Mutex<HashMap<String, OutputTokenParameter>>>,
    #[cfg(test)]
    capability_base_url: Option<String>,
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
            deepseek_strict_cache: Arc::new(Mutex::new(HashMap::new())),
            deepseek_thinking_cache: Arc::new(Mutex::new(HashMap::new())),
            output_token_cache: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(test)]
            capability_base_url: None,
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
            deepseek_strict_cache: Arc::new(Mutex::new(HashMap::new())),
            deepseek_thinking_cache: Arc::new(Mutex::new(HashMap::new())),
            output_token_cache: Arc::new(Mutex::new(HashMap::new())),
            #[cfg(test)]
            capability_base_url: None,
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

    #[cfg(test)]
    fn with_capability_base_url(mut self, base_url: &str) -> Self {
        self.capability_base_url = Some(base_url.to_string());
        self
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn stream(
        &self,
        mut request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        #[cfg(test)]
        let capability_base_url = self
            .capability_base_url
            .as_deref()
            .unwrap_or(&self.base_url);
        #[cfg(not(test))]
        let capability_base_url = self.base_url.as_str();
        let auto_output_parameter =
            request.profile.output_token_parameter == OutputTokenParameter::Auto;
        let output_cache_key = format!("{}|{}", self.base_url, request.model);
        if auto_output_parameter
            && let Ok(cache) = self.output_token_cache.lock()
            && let Some(parameter) = cache.get(&output_cache_key).copied()
        {
            request.profile.output_token_parameter = parameter;
        }
        let detected_capabilities = deepseek::ProviderCapabilities::detect(
            capability_base_url,
            &request.model,
            request.profile.deepseek_compat,
        );
        let strict_cache_key = if detected_capabilities.deepseek {
            Some(deepseek::strict_cache_key(&self.base_url, &request.model))
        } else {
            None
        };
        let cached_thinking_enabled = strict_cache_key
            .as_ref()
            .and_then(|key| self.deepseek_thinking_cache.lock().ok()?.get(key).copied());
        let cached_strict_tools = strict_cache_key
            .as_ref()
            .and_then(|key| self.deepseek_strict_cache.lock().ok()?.get(key).copied());
        let capabilities = deepseek::apply_cached_strict_capability(
            deepseek::apply_cached_thinking_capability(
                detected_capabilities,
                request.profile.deepseek_compat,
                cached_thinking_enabled,
            ),
            request.profile.deepseek_compat,
            cached_strict_tools,
        );
        let attempts = request::request_attempts_for(&request, &capabilities);
        if capabilities.deepseek {
            // This is wire-level diagnostics, not progress for the user.  Sending
            // it through `StreamEvent::Status` puts provider internals in the
            // transcript (and makes every DeepSeek turn start with a compat line).
            // Keep it available to debug logs while reserving Status for events
            // that need to be acted on or understood by the user.
            tracing::debug!(
                target: "hi::provider",
                wire_profile = %capabilities.diagnostic_status(attempts[0].strict_tools),
                "detected provider wire profile"
            );
        }
        let mut last_error: Option<ProviderError> = None;
        let mut idx = 0;
        let mut auth_refreshed = false;
        let correlation_id = canonical_request_id(request.request_id.as_deref());
        while idx < attempts.len() {
            let attempt = attempts[idx];
            let request_metadata = self.request_metadata(&request);
            let body = request::build_body_with_capabilities(
                &request,
                attempt,
                request_metadata.as_ref(),
                &capabilities,
            );
            let url = capabilities.completion_url(&self.base_url, attempt.strict_tools);
            // A payload-changing compatibility repair is a new provider
            // request identity. Credential refreshes keep the same `idx` and
            // therefore intentionally retain the original identity.
            let wire_request_id = if idx == 0 {
                correlation_id.clone()
            } else {
                format!("{correlation_id}-wire{}", idx + 1)
            };
            let idempotency_key = request_idempotency_key(&wire_request_id, &body);
            // Read the token per attempt: a refresh below replaces it in place.
            let token = self.auth.token().await;
            let response = crate::http::send_with_retry(
                self.http
                    .post(&url)
                    .bearer_auth(&token)
                    .header("x-request-id", &wire_request_id)
                    .header("x-request-attempt", request.retry_attempt.to_string())
                    .header("idempotency-key", &idempotency_key)
                    .json(&body),
            )
            .await
            .map_err(|error| {
                ProviderError::new(
                    ProviderErrorKind::Outage,
                    format!("request to model endpoint failed: {error}"),
                )
                .with_api_contract(None, Some(true), None)
            })?;

            if response.status().is_success() {
                sink(StreamEvent::WireAudit(wire_audit(
                    &request,
                    &self.base_url,
                    attempt,
                    idx,
                    &body,
                    true,
                    Some(response.status().as_u16()),
                )));
                let rate_limits = rate_limits_from_headers(response.headers());
                // `debug_tap` optionally echoes the raw wire bytes when
                // HI_DEBUG_STREAM is set; `idle_guard` aborts a connection
                // that went silent instead of blocking forever. Reduce the
                // stream to provider-agnostic SSE data strings.
                let stream = crate::http::idle_guard(
                    crate::http::debug_tap(response.bytes_stream()),
                    crate::http::stream_idle_window(),
                )
                .eventsource()
                .map(|res| res.map(|event| event.data).context("error reading stream"));
                let estimated_input_tokens =
                    estimate_request_input_tokens(&request.messages, &request.tools);
                let mut completion = stream::collect_completion_with_protocol(
                    Box::pin(stream),
                    sink,
                    capabilities.tool_protocol,
                )
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
                if auto_output_parameter && let Ok(mut cache) = self.output_token_cache.lock() {
                    cache.insert(output_cache_key.clone(), attempt.output_token_parameter);
                }
                stream::backfill_missing_usage(&mut completion, &request);
                completion.usage.rate_limits = completion.usage.rate_limits.or(rate_limits);

                let thinking_was_enabled = capabilities.deepseek
                    && request
                        .profile
                        .deepseek_thinking
                        .unwrap_or(capabilities.default_thinking_enabled)
                    && attempt.deepseek_thinking.unwrap_or(true);
                let has_thinking = completion
                    .content
                    .iter()
                    .any(|content| matches!(content, Content::Thinking { .. }));
                let has_visible_text = completion.content.iter().any(
                    |content| matches!(content, Content::Text(text) if !text.trim().is_empty()),
                );
                let stripped_reasoning = capabilities.deepseek
                    && !capabilities.official
                    && !capabilities.local_native_dsml
                    && thinking_was_enabled
                    && !completion.tool_calls().is_empty()
                    && !has_thinking
                    && !has_visible_text;
                if capabilities.deepseek
                    && capabilities.official
                    && thinking_was_enabled
                    && !completion.tool_calls().is_empty()
                    && !has_thinking
                {
                    return Err(ProviderError::new(
                        ProviderErrorKind::ToolProtocol,
                        "DeepSeek returned a tool call without required reasoning_content",
                    )
                    .with_usage(completion.usage)
                    .into());
                }
                if stripped_reasoning
                    && let Some(next) = request::next_deepseek_reasoning_attempt(
                        &attempts,
                        idx,
                        "gateway stripped reasoning_content",
                    )
                {
                    if let Some(key) = strict_cache_key.as_ref()
                        && let Ok(mut cache) = self.deepseek_thinking_cache.lock()
                    {
                        cache.insert(key.clone(), false);
                    }
                    tracing::debug!(
                        target: "hi::provider",
                        request_id = %correlation_id,
                        "gateway omitted DeepSeek reasoning_content; retrying with thinking disabled"
                    );
                    idx = next;
                    continue;
                }

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
            sink(StreamEvent::WireAudit(wire_audit(
                &request,
                &self.base_url,
                attempt,
                idx,
                &body,
                false,
                Some(status.as_u16()),
            )));
            let retry_after = retry_after_header_seconds(&response);
            let rate_limits = rate_limits_from_headers(response.headers());
            let text = response.text().await.unwrap_or_default();
            if attempt.strict_tools
                && request::is_deepseek_strict_schema_unsupported(&text)
                && let Some(key) = strict_cache_key.as_ref()
                && let Ok(mut cache) = self.deepseek_strict_cache.lock()
            {
                cache.insert(key.clone(), false);
            }
            if capabilities.deepseek
                && let Some(hint) = request::deepseek_compatibility_hint(&text)
            {
                tracing::debug!(
                    target: "hi::provider",
                    request_id = %correlation_id,
                    hint,
                    "provider compatibility response"
                );
            }
            let parsed = request::parse_api_error(Some(status), &text);
            let kind = parsed.kind;
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
            let mut error = parsed.into_provider_error(Some(status));
            if error.retry_after_seconds.is_none() {
                error.retry_after_seconds = retry_after;
            }
            if let Some(rate_limits) = rate_limits {
                error = error.with_usage(Usage {
                    rate_limits: Some(rate_limits),
                    ..Default::default()
                });
            }
            last_error = Some(error);
            if let Some(next) = request::next_deepseek_reasoning_attempt(&attempts, idx, &text) {
                if let Some(key) = strict_cache_key.as_ref()
                    && let Ok(mut cache) = self.deepseek_thinking_cache.lock()
                {
                    cache.insert(key.clone(), false);
                }
                tracing::debug!(
                    target: "hi::provider",
                    request_id = %correlation_id,
                    "gateway rejected DeepSeek reasoning_content; retrying with thinking disabled"
                );
                idx = next;
                continue;
            }
            if request.profile.compat == CompatMode::Strict {
                // DeepSeek's strict-schema fallback is a provider wire
                // compatibility requirement, not part of the generic retry
                // ladder. Allow that one shape change even when the caller
                // selected the generic strict retry policy.
                if let Some(next) = request::next_degraded_attempt(&attempts, idx, kind, &text) {
                    idx = next;
                    continue;
                }
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
        Err(last_error
            .unwrap_or_else(|| {
                ProviderError::new(ProviderErrorKind::Other, "request failed before streaming")
            })
            .into())
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ServedModel>> {
        let url = format!("{}/models", self.base_url);
        let token = self.auth.token().await;
        crate::http::fetch_models(self.http.get(&url).bearer_auth(&token)).await
    }
}

fn wire_audit(
    request: &ChatRequest,
    route: &str,
    attempt: request::RequestAttempt,
    index: usize,
    body: &Value,
    accepted: bool,
    response_status: Option<u16>,
) -> WireAudit {
    let reasoning_replay = request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .find_map(|content| match content {
            Content::Thinking {
                signature: Some(_), ..
            } => Some("signed_thinking"),
            Content::Thinking { .. } => Some("thinking_blocks"),
            _ => None,
        })
        .map(str::to_string);
    let reasoning_request = body
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| body.get("thinking").map(|_| "thinking".to_string()))
        .or_else(|| {
            request
                .thinking_budget
                .map(|_| "thinking_budget".to_string())
        });
    WireAudit {
        provider: "openai_compatible".to_string(),
        route: route.to_string(),
        model: request.model.clone(),
        output_token_parameter: attempt.output_token_parameter.label().to_string(),
        max_output_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        reasoning_request,
        reasoning_replay,
        native_tools_enabled: attempt.include_tools,
        tool_count: body
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        strict_schema: attempt.strict_tools,
        tool_choice: body
            .get("tool_choice")
            .and_then(Value::as_str)
            .map(str::to_string),
        request_attempt: index as u32 + 1,
        compatibility_fallback: compatibility_fallback(attempt, index),
        accepted,
        request_body: Some(body.clone()),
        response_status,
    }
}

fn compatibility_fallback(attempt: request::RequestAttempt, index: usize) -> Option<String> {
    if attempt.output_token_fallback {
        Some("output_token_parameter".to_string())
    } else if attempt.reasoning_fallback {
        Some("reasoning".to_string())
    } else if attempt.strict_fallback {
        Some("strict_schema".to_string())
    } else if index > 0 && !attempt.include_usage {
        Some("stream_usage".to_string())
    } else if index > 0 && !attempt.include_frequency_penalty {
        Some("frequency_penalty".to_string())
    } else {
        None
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
        let request_ids = server.request_ids();
        let idempotency_keys = server.idempotency_keys();
        assert_eq!(request_ids[0], request_ids[1]);
        assert_eq!(idempotency_keys[0], idempotency_keys[1]);
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
            statuses.is_empty(),
            "provider wire-shape retries must not appear as user status: {statuses:?}"
        );
        let bodies = server.bodies();
        assert!(bodies[0].contains("stream_options"));
        assert!(!bodies[1].contains("stream_options"));
        let request_ids = server.request_ids();
        let idempotency_keys = server.idempotency_keys();
        assert_ne!(request_ids[0], request_ids[1]);
        assert_ne!(idempotency_keys[0], idempotency_keys[1]);
    }

    #[tokio::test]
    async fn wire_audit_records_each_shape_attempt_and_acceptance() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(400, r#"{"error":"stream_options unsupported"}"#),
            Response::sse(sse_text("ok")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let mut audits = Vec::new();
        let mut sink = |event| {
            if let StreamEvent::WireAudit(audit) = event {
                audits.push(audit);
            }
        };
        provider
            .stream(request(vec![], Default::default()), &mut sink)
            .await
            .unwrap();
        assert_eq!(audits.len(), 2);
        assert!(!audits[0].accepted);
        assert_eq!(audits[0].request_attempt, 1);
        assert_eq!(audits[0].compatibility_fallback, None);
        assert!(audits[1].accepted);
        assert_eq!(audits[1].request_attempt, 2);
        assert_eq!(
            audits[1].compatibility_fallback.as_deref(),
            Some("stream_usage")
        );
        assert_eq!(audits[1].response_status, Some(200));
        assert!(
            audits[0]
                .request_body
                .as_ref()
                .is_some_and(|body| body["max_tokens"] == 16)
        );
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
    async fn request_identity_is_sent_on_every_model_call() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(sse_text("ok"))]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let mut req = request(vec![], Default::default());
        req.request_id = Some("hi_turn_123".to_string());
        req.retry_attempt = 1;

        provider.stream(req, &mut |_| {}).await.unwrap();

        assert_eq!(server.request_ids(), vec![Some("hi_turn_123".to_string())]);
        assert_eq!(server.request_attempts(), vec![Some("1".to_string())]);
        let keys = server.idempotency_keys();
        assert!(
            keys[0]
                .as_deref()
                .is_some_and(|key| key.starts_with("hi_turn_123:"))
        );
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
            r#"{"error":"model temporarily unavailable"}"#,
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
    async fn server_error_is_not_retried_inside_the_adapter() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(500, r#"{"error":"temporary server error"}"#),
            Response::sse(sse_text("recovered")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(provider_error_kind(&err), Some(ProviderErrorKind::Outage));
        assert_eq!(server.bodies().len(), 1);
    }

    #[tokio::test]
    async fn capacity_unavailable_is_not_retried_inside_the_adapter() {
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
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::CapacityUnavailable)
        );
        assert_eq!(server.bodies().len(), 1);
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

    #[tokio::test]
    async fn deepseek_two_tool_rounds_replay_reasoning_and_tool_history() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::sse(deepseek_tool_sse(
                "inspect the first file",
                "call_1",
                "read",
                r#"{"path":"README.md"}"#,
            )),
            Response::sse(deepseek_tool_sse(
                "inspect the second file",
                "call_2",
                "read",
                r#"{"path":"Cargo.toml"}"#,
            )),
            Response::sse(sse_text("complete")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
        let profile = RequestProfile {
            tool_mode: ToolMode::Required,
            deepseek_compat: crate::types::DeepSeekCompat::On,
            ..Default::default()
        };
        let tools = vec![ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }];
        let mut first_request = request(tools.clone(), profile);
        first_request.model = "DeepSeek-V4-Flash-0731".into();
        first_request.user_turn = true;
        let first = provider
            .stream(first_request.clone(), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(
            &first.content[0],
            Content::Thinking { text, .. } if text == "inspect the first file"
        ));
        assert!(matches!(
            &first.content[1],
            Content::ToolCall { id, name, .. } if id == "call_1" && name == "read"
        ));

        let mut second_messages = (*first_request.messages).clone();
        second_messages.push(Message::assistant(first.content.clone()));
        second_messages.push(Message::tool_result("call_1", "README contents"));
        let mut second_request = first_request.clone();
        second_request.messages = second_messages.into();
        let second = provider
            .stream(second_request.clone(), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(
            &second.content[0],
            Content::Thinking { text, .. } if text == "inspect the second file"
        ));
        assert!(matches!(
            &second.content[1],
            Content::ToolCall { id, name, .. } if id == "call_2" && name == "read"
        ));

        let mut third_messages = (*second_request.messages).clone();
        third_messages.push(Message::assistant(second.content.clone()));
        third_messages.push(Message::tool_result("call_2", "Cargo manifest"));
        let mut third_request = second_request;
        third_request.messages = third_messages.into();
        let final_completion = provider.stream(third_request, &mut |_| {}).await.unwrap();
        assert!(matches!(
            final_completion.content.first(),
            Some(Content::Text(text)) if text == "complete"
        ));

        let bodies = server
            .bodies()
            .into_iter()
            .map(|body| serde_json::from_str::<serde_json::Value>(&body).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(bodies.len(), 3);
        assert_eq!(bodies[0]["thinking"]["type"], "enabled");
        assert!(bodies[0].get("tool_choice").is_none());
        assert_eq!(
            bodies[1]["messages"][1]["reasoning_content"],
            "inspect the first file"
        );
        assert_eq!(bodies[1]["messages"][2]["content"], "README contents");
        assert_eq!(
            bodies[2]["messages"][3]["reasoning_content"],
            "inspect the second file"
        );
        assert_eq!(bodies[2]["messages"][4]["content"], "Cargo manifest");
    }

    #[tokio::test]
    async fn deepseek_gateway_reasoning_rejection_is_cached_after_one_retry() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(400, r#"{"error":"reasoning_content unsupported"}"#),
            Response::sse(deepseek_tool_sse_without_reasoning(
                "call_gateway_1",
                "read",
                r#"{"path":"README.md"}"#,
            )),
            Response::sse(sse_text("cached")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into())
            .with_capability_base_url("https://gateway.example/v1");
        let profile = RequestProfile {
            tool_mode: ToolMode::Required,
            deepseek_compat: crate::types::DeepSeekCompat::Auto,
            ..Default::default()
        };
        let mut first_request = request(vec![tool()], profile);
        first_request.model = "deepseek-v4-flash".into();
        provider
            .stream(first_request.clone(), &mut |_| {})
            .await
            .unwrap();
        let second = provider.stream(first_request, &mut |_| {}).await.unwrap();
        assert!(matches!(
            second.content.first(),
            Some(Content::Text(text)) if text == "cached"
        ));

        let bodies = server
            .bodies()
            .into_iter()
            .map(|body| serde_json::from_str::<serde_json::Value>(&body).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(bodies.len(), 3);
        assert_eq!(bodies[0]["thinking"]["type"], "enabled");
        assert_eq!(bodies[1]["thinking"]["type"], "disabled");
        assert_eq!(bodies[2]["thinking"]["type"], "disabled");
        assert_ne!(
            server.idempotency_keys()[0],
            server.idempotency_keys()[1],
            "thinking fallback must have a new payload identity"
        );
    }

    #[tokio::test]
    async fn deepseek_gateway_stripped_reasoning_retries_before_tool_execution() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::sse(deepseek_tool_sse_without_reasoning(
                "call_gateway_2",
                "read",
                r#"{"path":"README.md"}"#,
            )),
            Response::sse(deepseek_tool_sse(
                "fallback reasoning",
                "call_gateway_3",
                "read",
                r#"{"path":"README.md"}"#,
            )),
            Response::sse(sse_text("cached")),
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into())
            .with_capability_base_url("https://gateway.example/v1");
        let profile = RequestProfile {
            tool_mode: ToolMode::Required,
            deepseek_compat: crate::types::DeepSeekCompat::Auto,
            ..Default::default()
        };
        let mut first_request = request(vec![tool()], profile);
        first_request.model = "deepseek-v4-flash".into();
        let first = provider
            .stream(first_request.clone(), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(
            first.content.first(),
            Some(Content::Thinking { text, .. }) if text == "fallback reasoning"
        ));
        let second = provider.stream(first_request, &mut |_| {}).await.unwrap();
        assert!(matches!(
            second.content.first(),
            Some(Content::Text(text)) if text == "cached"
        ));
        let bodies = server.bodies();
        let first_body: serde_json::Value = serde_json::from_str(&bodies[0]).unwrap();
        let fallback_body: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
        let cached_body: serde_json::Value = serde_json::from_str(&bodies[2]).unwrap();
        assert_eq!(first_body["thinking"]["type"], "enabled");
        assert_eq!(fallback_body["thinking"]["type"], "disabled");
        assert_eq!(cached_body["thinking"]["type"], "disabled");
    }

    #[tokio::test]
    async fn official_deepseek_missing_reasoning_is_not_silently_replayed() {
        let Some(server) =
            FakeOpenAiServer::new(vec![Response::sse(deepseek_tool_sse_without_reasoning(
                "call_official_1",
                "read",
                r#"{"path":"README.md"}"#,
            ))])
        else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().to_string(), "test".into())
            .with_capability_base_url("https://api.deepseek.com/v1");
        let mut req = request(
            vec![tool()],
            RequestProfile {
                tool_mode: ToolMode::Required,
                deepseek_compat: crate::types::DeepSeekCompat::Auto,
                ..Default::default()
            },
        );
        req.model = "deepseek-v4-flash".into();
        let error = provider.stream(req, &mut |_| {}).await.unwrap_err();
        assert_eq!(
            provider_error_kind(&error),
            Some(ProviderErrorKind::ToolProtocol)
        );
        assert_eq!(server.bodies().len(), 1);
    }

    fn request(tools: Vec<ToolSpec>, profile: RequestProfile) -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            request_id: None,
            retry_attempt: 0,
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

    fn deepseek_tool_sse(reasoning: &str, id: &str, name: &str, arguments: &str) -> String {
        let reasoning = serde_json::to_string(reasoning).unwrap();
        let id = serde_json::to_string(id).unwrap();
        let name = serde_json::to_string(name).unwrap();
        let arguments = serde_json::to_string(arguments).unwrap();
        format!(
            "data: {{\"choices\":[{{\"delta\":{{\"reasoning_content\":{reasoning},\"tool_calls\":[{{\"index\":0,\"id\":{id},\"function\":{{\"name\":{name},\"arguments\":{arguments}}}}}]}}}}]}}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\ndata: [DONE]\n\n"
        )
    }

    fn deepseek_tool_sse_without_reasoning(id: &str, name: &str, arguments: &str) -> String {
        let id = serde_json::to_string(id).unwrap();
        let name = serde_json::to_string(name).unwrap();
        let arguments = serde_json::to_string(arguments).unwrap();
        format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":{id},\"function\":{{\"name\":{name},\"arguments\":{arguments}}}}}]}}}}]}}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\ndata: [DONE]\n\n"
        )
    }
}
