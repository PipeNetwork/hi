//! Request translation: `hi_ai` messages → OpenAI Chat Completions wire JSON,
//! plus the degraded-retry attempt ladder and HTTP error classification.

use reqwest::StatusCode;
use serde_json::{Value, json};

use super::deepseek::ProviderCapabilities;
use crate::provider::{ProviderError, ProviderErrorKind};
use crate::types::{ChatRequest, CompatMode, Message, Role, ToolMode};

/// One shape the request is sent in. The provider tries the most capable shape
/// first and degrades through this list when the server rejects a compatible
/// optional feature.
#[derive(Clone, Copy)]
pub(crate) struct RequestAttempt {
    pub(crate) include_usage: bool,
    pub(crate) include_tools: bool,
    /// OpenAI-style `frequency_penalty`. Some models (e.g. xAI grok-4.5) reject
    /// the parameter entirely; the compat ladder drops it on that 400.
    pub(crate) include_frequency_penalty: bool,
    pub(crate) strict_tools: bool,
    /// True only for the one controlled DeepSeek strict-schema fallback. It
    /// must not re-enter the generic request-shape retry ladder if it fails.
    pub(crate) strict_fallback: bool,
    pub(crate) status: Option<&'static str>,
}

/// Given the attempt that just failed (at `current`) and its error, the index of
/// the next attempt to try — the one whose degradation actually addresses this
/// error — or `None` to stop and surface the error. Tool rejection is never
/// downgraded to chat-only: a coding-agent turn that advertised tools cannot
/// reliably complete after losing workspace access.
pub(crate) fn next_degraded_attempt(
    attempts: &[RequestAttempt],
    current: usize,
    kind: ProviderErrorKind,
    text: &str,
) -> Option<usize> {
    let cur = attempts[current];
    let after = current + 1;
    if cur.strict_fallback {
        return None;
    }
    if cur.strict_tools && is_deepseek_strict_schema_text(text) {
        return attempts[after..]
            .iter()
            // Preserve earlier degradations. For example, if the provider
            // already rejected `stream_options`, the strict-schema fallback
            // must not re-add it and pay for another avoidable 400.
            .position(|a| {
                !a.strict_tools
                    && a.include_tools
                    && a.include_usage == cur.include_usage
                    && a.include_frequency_penalty == cur.include_frequency_penalty
            })
            .map(|i| after + i);
    }
    // Usage streaming rejected → retry without it (keeping tools).
    if cur.include_usage && mentions(text, &["stream_options", "include_usage"]) {
        return attempts[after..]
            .iter()
            .position(|a| !a.include_usage)
            .map(|i| after + i);
    }
    // frequency_penalty rejected (xAI: "does not support parameter frequencyPenalty")
    // → retry without it. Keep tools and stream_options.
    if cur.include_frequency_penalty && is_unsupported_frequency_penalty_text(text) {
        return attempts[after..]
            .iter()
            .position(|a| !a.include_frequency_penalty)
            .map(|i| after + i);
    }
    // Tool schema rejected → fail fast. Use `--tool-mode chat-only` for an
    // explicit no-tools request.
    if cur.include_tools
        && matches!(
            kind,
            ProviderErrorKind::UnsupportedTools | ProviderErrorKind::UnsupportedRequestShape
        )
    {
        return None;
    }
    // Provider/transport failures never justify mutating and replaying the
    // payload against the same route. The outer route/fallback policy may move
    // to another compatible backend when the typed error permits it.
    None
}

#[cfg(test)]
pub(crate) fn request_attempts(request: &ChatRequest) -> Vec<RequestAttempt> {
    request_attempts_for(request, &ProviderCapabilities::generic())
}

pub(crate) fn request_attempts_for(
    request: &ChatRequest,
    capabilities: &ProviderCapabilities,
) -> Vec<RequestAttempt> {
    let include_usage = request.profile.stream_usage.unwrap_or(true);
    let include_tools =
        !request.tools.is_empty() && request.profile.tool_mode != ToolMode::ChatOnly;
    let include_frequency_penalty = request.frequency_penalty.is_some();
    let strict_capability =
        capabilities.strict_tools && request.profile.deepseek_strict != Some(false);
    let mut attempts = vec![RequestAttempt {
        include_usage,
        include_tools,
        include_frequency_penalty,
        strict_tools: include_tools && strict_capability,
        strict_fallback: false,
        status: None,
    }];
    if request.profile.compat == CompatMode::Strict {
        if strict_capability {
            let mut fallback = attempts[0];
            fallback.strict_tools = false;
            fallback.strict_fallback = true;
            fallback.status =
                Some("compat: DeepSeek rejected strict tool schemas; retried without strict mode");
            attempts.push(fallback);
        }
        return attempts;
    }
    if include_usage {
        attempts.push(RequestAttempt {
            include_usage: false,
            include_tools,
            include_frequency_penalty,
            strict_tools: include_tools && strict_capability,
            strict_fallback: false,
            status: Some(
                "compat: provider rejected stream_options; retried without usage streaming",
            ),
        });
    }
    // Recovery sampling sets frequency_penalty; several OpenAI-compatible hosts
    // (notably xAI grok-4.5) reject the field. Offer a same-shape retry without it.
    if include_frequency_penalty {
        attempts.push(RequestAttempt {
            include_usage,
            include_tools,
            include_frequency_penalty: false,
            strict_tools: include_tools && strict_capability,
            strict_fallback: false,
            status: Some("compat: provider rejected frequency_penalty; retried without it"),
        });
        if include_usage {
            attempts.push(RequestAttempt {
                include_usage: false,
                include_tools,
                include_frequency_penalty: false,
                strict_tools: include_tools && strict_capability,
                strict_fallback: false,
                status: Some(
                    "compat: provider rejected stream_options/frequency_penalty; retried without both",
                ),
            });
        }
    }
    if strict_capability {
        let strict_attempts = attempts.clone();
        for mut attempt in strict_attempts {
            attempt.strict_tools = false;
            attempt.strict_fallback = true;
            attempt.status =
                Some("compat: DeepSeek rejected strict tool schemas; retried without strict mode");
            attempts.push(attempt);
        }
    }
    attempts
}

pub(crate) fn is_deepseek_strict_schema_text(text: &str) -> bool {
    mentions(
        text,
        &[
            "strict",
            "schema",
            "additionalproperties",
            "additional properties",
            "required",
            "unsupported keyword",
        ],
    ) && mentions(
        text,
        &[
            "invalid",
            "unsupported",
            "not support",
            "must be",
            "rejected",
            "beta",
        ],
    )
}

/// A gateway has positively identified that this model cannot consume strict
/// tool schemas. Unlike a generic schema validation error, this is safe to
/// remember for the endpoint/model pair and avoids paying the failed strict
/// request on every later turn.
pub(crate) fn is_deepseek_strict_schema_unsupported(text: &str) -> bool {
    mentions(
        text,
        &[
            "model_does_not_support_strict_tools",
            "does not support strict tool",
            "doesn't support strict tool",
            "unsupported strict tool",
        ],
    )
}

pub(crate) fn deepseek_compatibility_hint(text: &str) -> Option<&'static str> {
    if mentions(text, &["tool_choice", "tool choice"]) {
        Some("endpoint rejected tool_choice; DeepSeek uses client-side required-tool validation")
    } else if mentions(text, &["reasoning_content", "reasoning content"]) {
        Some(
            "endpoint rejected reasoning_content; verify the gateway preserves DeepSeek thinking fields",
        )
    } else if mentions(text, &["developer role", "developer message", "developer"]) {
        Some("endpoint rejected the developer role; DeepSeek requests use system messages")
    } else if is_deepseek_strict_schema_text(text) {
        Some("strict tool schema was rejected; retrying with the normalized non-strict schema")
    } else {
        None
    }
}

/// xAI returns camelCase (`frequencyPenalty`); OpenAI-style wording uses snake_case.
fn is_unsupported_frequency_penalty_text(text: &str) -> bool {
    mentions(
        text,
        &["frequency_penalty", "frequencypenalty", "frequency penalty"],
    ) && mentions(
        text,
        &["does not support", "unsupported", "unknown", "invalid"],
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedApiError {
    pub(crate) kind: ProviderErrorKind,
    pub(crate) message: String,
    pub(crate) code: Option<String>,
    pub(crate) retryable: Option<bool>,
    pub(crate) retry_after_seconds: Option<u64>,
}

impl ParsedApiError {
    pub(crate) fn into_provider_error(self, status: Option<StatusCode>) -> ProviderError {
        let message = match status {
            Some(status) => format!("API error {}: {}", api_status_label(status), self.message),
            None => self.message,
        };
        ProviderError::new(self.kind, message).with_api_contract(
            self.code,
            self.retryable,
            self.retry_after_seconds,
        )
    }
}

fn api_status_label(status: StatusCode) -> String {
    match status.as_u16() {
        // Cloudflare's origin timeout is intentionally outside the standard
        // HTTP status registry, so `http::StatusCode` otherwise displays it as
        // `<unknown status code>`.
        524 => "524 Gateway Timeout".to_string(),
        _ => status.to_string(),
    }
}

pub(crate) fn parse_api_error(status: Option<StatusCode>, text: &str) -> ParsedApiError {
    let structured = structured_api_error_fields(text);
    let mut kind = structured
        .as_ref()
        .and_then(|fields| {
            structured_error_kind(fields.code.as_deref(), fields.error_type.as_deref())
        })
        .unwrap_or_else(|| classify_http_error_fallback(status, text));
    if kind == ProviderErrorKind::Other
        && status.is_none()
        && structured
            .as_ref()
            .and_then(|fields| fields.error_type.as_deref())
            == Some("invalid_request_error")
    {
        kind = ProviderErrorKind::UnsupportedRequestShape;
    }
    let message = structured
        .as_ref()
        .and_then(|fields| fields.message.clone())
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| text.to_string());
    let retryable = structured
        .as_ref()
        .and_then(|fields| fields.retryable)
        .or_else(|| inferred_retryable(status, kind, text));
    ParsedApiError {
        kind,
        code: structured.as_ref().and_then(|fields| fields.code.clone()),
        retryable,
        retry_after_seconds: structured
            .as_ref()
            .and_then(|fields| fields.retry_after_seconds),
        message,
    }
}

fn inferred_retryable(
    status: Option<StatusCode>,
    kind: ProviderErrorKind,
    text: &str,
) -> Option<bool> {
    match kind {
        ProviderErrorKind::RateLimit
        | ProviderErrorKind::CapacityUnavailable
        | ProviderErrorKind::MalformedStream
        | ProviderErrorKind::ToolProtocol => Some(true),
        ProviderErrorKind::ModelUnavailable => Some(is_model_unavailable_text(text)),
        ProviderErrorKind::Outage => Some(status.is_none_or(|status| status.is_server_error())),
        ProviderErrorKind::Auth
        | ProviderErrorKind::UnsupportedRequestShape
        | ProviderErrorKind::UnsupportedTools
        | ProviderErrorKind::RequestTooLarge
        | ProviderErrorKind::QualityRejected => Some(false),
        ProviderErrorKind::EmptyCompletion | ProviderErrorKind::Other => None,
    }
}

#[cfg(test)]
pub(crate) fn classify_http_error(status: StatusCode, text: &str) -> ProviderErrorKind {
    parse_api_error(Some(status), text).kind
}

fn classify_http_error_fallback(status: Option<StatusCode>, text: &str) -> ProviderErrorKind {
    let Some(status) = status else {
        return classify_message_fallback(text);
    };
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ProviderErrorKind::Auth,
        // Not every backend uses 401 for a bad credential: xAI answers a wrong
        // or expired key with 400 `invalid-argument`. Without this the body
        // falls through to the 400 arm and is reported as an unsupported
        // request shape, so the compat ladder retries a request that can never
        // succeed and the user is told to fix their request, not their key.
        _ if is_auth_text(text) => ProviderErrorKind::Auth,
        StatusCode::TOO_MANY_REQUESTS => ProviderErrorKind::RateLimit,
        _ if mentions(text, &["request not found"]) => ProviderErrorKind::MalformedStream,
        StatusCode::NOT_FOUND => ProviderErrorKind::ModelUnavailable,
        _ if is_model_unavailable_text(text) => ProviderErrorKind::ModelUnavailable,
        StatusCode::CONFLICT | StatusCode::SERVICE_UNAVAILABLE
            if is_capacity_unavailable_text(text) =>
        {
            ProviderErrorKind::CapacityUnavailable
        }
        _ if is_quality_rejected_text(text) => ProviderErrorKind::QualityRejected,
        _ if is_tool_protocol_text(text) => ProviderErrorKind::ToolProtocol,
        s if s.is_server_error() => ProviderErrorKind::Outage,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
            if mentions(
                text,
                &[
                    "maximum allowed size",
                    "input exceeds",
                    "context length",
                    "context_length_exceeded",
                    "resident model context",
                    "too many tokens",
                    "request too large",
                    // Provider-specific wording (e.g. "maximum prompt length is
                    // 500000 but the request contains 500547 tokens") — must not
                    // fall through as UnsupportedRequestShape / fake compat tip.
                    "maximum prompt length",
                    "prompt length",
                    "maximum context length",
                    "exceeds the context",
                    "exceed context",
                ],
            ) {
                ProviderErrorKind::RequestTooLarge
            } else if mentions(text, &["tool", "function_call", "function"]) {
                ProviderErrorKind::UnsupportedTools
            } else {
                ProviderErrorKind::UnsupportedRequestShape
            }
        }
        _ => ProviderErrorKind::Other,
    }
}

fn classify_message_fallback(text: &str) -> ProviderErrorKind {
    let lower = text.to_ascii_lowercase();
    if mentions(&lower, &["rate limit", "too many requests", "429"]) {
        ProviderErrorKind::RateLimit
    } else if mentions(
        &lower,
        &[
            "resident model context",
            "context_length_exceeded",
            "context length",
            "request too large",
            "too many tokens",
            "maximum prompt length",
            "prompt length",
            "maximum context length",
            "exceeds the context",
            "exceed context",
        ],
    ) {
        ProviderErrorKind::RequestTooLarge
    } else if is_model_unavailable_text(text) {
        ProviderErrorKind::ModelUnavailable
    } else if is_capacity_unavailable_text(text) {
        ProviderErrorKind::CapacityUnavailable
    } else if is_quality_rejected_text(text) {
        ProviderErrorKind::QualityRejected
    } else if is_tool_protocol_text(text) {
        ProviderErrorKind::ToolProtocol
    } else if mentions(
        &lower,
        &[
            "service unavailable",
            "no route",
            "overloaded",
            "cooling down",
            "first_token_stall",
            "first token",
        ],
    ) {
        ProviderErrorKind::Outage
    } else if lower.contains("request not found") {
        ProviderErrorKind::MalformedStream
    } else {
        ProviderErrorKind::Other
    }
}

#[derive(Default)]
struct StructuredApiErrorFields {
    message: Option<String>,
    code: Option<String>,
    error_type: Option<String>,
    retryable: Option<bool>,
    retry_after_seconds: Option<u64>,
}

fn structured_api_error_fields(text: &str) -> Option<StructuredApiErrorFields> {
    let root = serde_json::from_str::<Value>(text).ok()?;
    let error = root.get("error").unwrap_or(&root);
    let message = match error {
        Value::String(message) => Some(message.clone()),
        Value::Object(object) => object
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| object.get("error").and_then(Value::as_str))
            .map(str::to_string),
        _ => None,
    }
    .or_else(|| {
        root.get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    let code = value_string_or_number(error.get("code").or_else(|| root.get("code")));
    let error_type = error
        .get("type")
        .or_else(|| root.get("type"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let retryable = error
        .get("retryable")
        .or_else(|| root.get("retryable"))
        .and_then(Value::as_bool);
    let retry_after_seconds = error
        .get("retry_after_seconds")
        .or_else(|| root.get("retry_after_seconds"))
        .and_then(Value::as_u64);
    (message.is_some()
        || code.is_some()
        || error_type.is_some()
        || retryable.is_some()
        || retry_after_seconds.is_some())
    .then_some(StructuredApiErrorFields {
        message,
        code,
        error_type,
        retryable,
        retry_after_seconds,
    })
}

fn value_string_or_number(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn structured_error_kind(
    code: Option<&str>,
    error_type: Option<&str>,
) -> Option<ProviderErrorKind> {
    match code.unwrap_or_default() {
        "capacity_unavailable" => Some(ProviderErrorKind::CapacityUnavailable),
        "tool_protocol_error" => Some(ProviderErrorKind::ToolProtocol),
        "service_unavailable" => Some(ProviderErrorKind::Outage),
        "context_length_exceeded" | "output_token_limit" => {
            Some(ProviderErrorKind::RequestTooLarge)
        }
        "quality_rejected" => Some(ProviderErrorKind::QualityRejected),
        "model_unavailable" => Some(ProviderErrorKind::ModelUnavailable),
        "rate_limit" | "rate_limit_exceeded" => Some(ProviderErrorKind::RateLimit),
        "bad_request" | "policy_violation" => Some(ProviderErrorKind::UnsupportedRequestShape),
        _ => match error_type.unwrap_or_default() {
            "rate_limit_error" => Some(ProviderErrorKind::RateLimit),
            "service_unavailable_error" => Some(ProviderErrorKind::Outage),
            "authentication_error" | "permission_error" => Some(ProviderErrorKind::Auth),
            _ => None,
        },
    }
}

pub(crate) fn is_capacity_unavailable_text(text: &str) -> bool {
    mentions(
        text,
        &["capacity_unavailable", "capacity temporarily unavailable"],
    )
}

pub(crate) fn is_model_unavailable_text(text: &str) -> bool {
    mentions(
        text,
        &[
            "model_unavailable",
            "model temporarily unavailable",
            "requested model is unavailable",
            "model not available",
            "model not enabled",
            "model not supported",
            "unknown model",
        ],
    )
}

pub(crate) fn is_quality_rejected_text(text: &str) -> bool {
    if is_review_evidence_repair_text(text) {
        return false;
    }
    mentions(
        text,
        &["quality_rejected", "quality rejected", "quality check"],
    )
}

fn is_review_evidence_repair_text(text: &str) -> bool {
    mentions(
        text,
        &[
            "insufficient evidence",
            "inspected evidence",
            "review evidence",
        ],
    )
}

pub(crate) fn is_tool_protocol_text(text: &str) -> bool {
    mentions(
        text,
        &[
            "tool protocol",
            "did not satisfy the tool protocol",
            "did not match the tool protocol",
            "tool-enabled chat output must be valid json",
        ],
    )
}

/// Does the body name a credential problem, whatever the status code says?
///
/// Deliberately narrow: every phrase here names a key/token/credential, so a
/// request-shape error that merely happens to mention "token" (as token-limit
/// errors do) is not swept up. Token-*limit* wording is handled by the
/// request-too-large branch and must stay there.
fn is_auth_text(text: &str) -> bool {
    mentions(
        text,
        &[
            "incorrect api key",
            "invalid api key",
            "api key is missing",
            "invalid_api_key",
            "unauthenticated",
            "invalid access token",
            "expired token",
            "token has expired",
            "token is expired",
            "invalid_grant",
        ],
    )
}

fn mentions(text: &str, needles: &[&str]) -> bool {
    let lower = text.to_ascii_lowercase();
    needles.iter().any(|needle| lower.contains(needle))
}

#[cfg(test)]
pub(crate) fn build_body(
    request: &ChatRequest,
    attempt: RequestAttempt,
    metadata: Option<&Value>,
) -> Value {
    build_body_with_capabilities(request, attempt, metadata, &ProviderCapabilities::generic())
}

pub(crate) fn build_body_with_capabilities(
    request: &ChatRequest,
    attempt: RequestAttempt,
    metadata: Option<&Value>,
    capabilities: &ProviderCapabilities,
) -> Value {
    let messages = to_openai_messages_with_capabilities(&request.messages, capabilities);
    let mut body = json!({
        "model": capabilities.model_for_request(&request.model),
        "messages": messages,
        "stream": true,
        "max_tokens": request.max_tokens,
    });
    if let Some(metadata) = metadata {
        body["metadata"] = metadata.clone();
    }
    if attempt.include_usage {
        body["stream_options"] = json!({ "include_usage": true });
    }
    if attempt.include_tools {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": if attempt.strict_tools {
                            super::deepseek::normalize_strict_schema(&t.parameters)
                        } else {
                            t.parameters.clone()
                        },
                    }
                })
            })
            .collect();
        body["tools"] = json!(tools);
        if request.profile.tool_mode == ToolMode::Required && capabilities.supports_tool_choice {
            body["tool_choice"] = json!("required");
        }
        if attempt.strict_tools {
            if let Some(tools) = body["tools"].as_array_mut() {
                for tool in tools {
                    tool["function"]["strict"] = json!(true);
                }
            }
        }
    }
    if capabilities.deepseek {
        let thinking_enabled = request.profile.deepseek_thinking.unwrap_or(true);
        body["thinking"] = json!({
            "type": if thinking_enabled { "enabled" } else { "disabled" }
        });
    }
    if capabilities.supports_sampling_params {
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
        if attempt.include_frequency_penalty
            && let Some(frequency_penalty) = request.frequency_penalty
        {
            body["frequency_penalty"] = json!(frequency_penalty);
        }
    }
    // Abstract reasoning level (GPT-5/o-series style). Endpoints that don't
    // support it validate the value and 400 on an unknown one, so we only send
    // it when explicitly requested. The Anthropic adapter ignores this field
    // and uses `thinking_budget` instead.
    if (!capabilities.deepseek || request.profile.deepseek_thinking != Some(false))
        && let Some(effort) = request.reasoning_effort
    {
        body["reasoning_effort"] = json!(capabilities.reasoning_wire_value(effort));
    }
    body
}

/// Flatten neutral messages into OpenAI's wire shape. The generic path keeps
/// thinking as inline tags for cross-provider handoff; DeepSeek uses its native
/// reasoning_content field through the capability-aware path.
#[cfg(test)]
pub(crate) fn to_openai_messages(messages: &[Message]) -> Vec<Value> {
    to_openai_messages_with_capabilities(messages, &ProviderCapabilities::generic())
}

pub(crate) fn to_openai_messages_with_capabilities(
    messages: &[Message],
    capabilities: &ProviderCapabilities,
) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message.role {
            Role::System => out.push(json!({ "role": "system", "content": message.text() })),
            Role::User => {
                // If the message carries any image blocks, emit OpenAI's
                // multipart `content` array (text + image_url). Otherwise fall
                // back to the plain string form, which is cheaper and more
                // broadly compatible.
                let has_image = message
                    .content
                    .iter()
                    .any(|b| matches!(b, crate::types::Content::Image { .. }));
                if has_image {
                    let mut parts = Vec::new();
                    for block in &message.content {
                        match block {
                            crate::types::Content::Image { data, media_type } => {
                                parts.push(json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!("data:{media_type};base64,{data}"),
                                    },
                                }))
                            }
                            crate::types::Content::Text(t) if !t.is_empty() => {
                                parts.push(json!({ "type": "text", "text": t }));
                            }
                            _ => {}
                        }
                    }
                    if parts.is_empty() {
                        parts.push(json!({ "type": "text", "text": message.text() }));
                    }
                    out.push(json!({ "role": "user", "content": parts }));
                } else {
                    out.push(json!({ "role": "user", "content": message.text() }));
                }
            }
            Role::Assistant => {
                let mut thinking = String::new();
                let mut text = String::new();
                let mut tool_calls = Vec::new();
                for block in &message.content {
                    match block {
                        crate::types::Content::Text(t) => text.push_str(t),
                        // Cross-provider handoff: the Chat Completions API has no
                        // reasoning field, so preserve Anthropic-style thinking as
                        // inline tags rather than dropping it.
                        crate::types::Content::Thinking { text: t, .. } => thinking.push_str(t),
                        crate::types::Content::ToolCall {
                            id,
                            name,
                            arguments,
                        } => tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": arguments },
                        })),
                        crate::types::Content::ToolResult { .. } => {}
                        // Images don't appear in assistant turns; ignore them.
                        crate::types::Content::Image { .. } => {}
                    }
                }
                let mut msg = json!({ "role": "assistant" });
                if capabilities.requires_assistant_content {
                    msg["content"] = json!(text);
                    if capabilities.requires_reasoning_content && !thinking.is_empty() {
                        msg["reasoning_content"] = json!(thinking);
                    }
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = json!(tool_calls);
                    }
                } else {
                    let mut content = String::new();
                    if !thinking.is_empty() {
                        content.push_str(&format!("<thinking>\n{thinking}\n</thinking>\n"));
                    }
                    content.push_str(&text);
                    if tool_calls.is_empty() {
                        msg["content"] = json!(content);
                    } else {
                        msg["tool_calls"] = json!(tool_calls);
                        if !content.is_empty() {
                            msg["content"] = json!(content);
                        }
                    }
                }
                out.push(msg);
            }
            Role::Tool => {
                for block in &message.content {
                    if let crate::types::Content::ToolResult { call_id, output } = block {
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output,
                        }));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        build_body, build_body_with_capabilities, classify_http_error,
        is_deepseek_strict_schema_unsupported, is_quality_rejected_text,
        is_unsupported_frequency_penalty_text, next_degraded_attempt, parse_api_error,
        request_attempts, request_attempts_for, to_openai_messages,
        to_openai_messages_with_capabilities,
    };
    use reqwest::StatusCode;

    use super::super::deepseek::ProviderCapabilities;
    use crate::provider::ProviderErrorKind;
    use crate::types::{
        CompatMode, Content, DeepSeekCompat, Message, RequestProfile, Role, ToolMode, ToolSpec,
    };

    /// Verbatim body from api.x.ai when recovery sampling sends frequency_penalty
    /// to grok-4.5 (the model rejects the parameter entirely).
    #[test]
    fn xai_frequency_penalty_rejection_is_detected() {
        let body = r#"{"code":"invalid-argument","error":"Model grok-4.5 does not support parameter frequencyPenalty."}"#;
        assert!(
            is_unsupported_frequency_penalty_text(body),
            "xAI camelCase wording must match"
        );
        assert!(is_unsupported_frequency_penalty_text(
            "does not support parameter frequency_penalty"
        ));
        // An unrelated 400 that merely mentions "invalid" must not match.
        assert!(!is_unsupported_frequency_penalty_text(
            r#"{"error":"invalid temperature"}"#
        ));
    }

    #[test]
    fn frequency_penalty_rejection_retries_without_the_field() {
        let mut req = crate::types::ChatRequest {
            model: "grok-4.5".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: Some(0.8),
            top_p: Some(0.95),
            frequency_penalty: Some(0.4),
            thinking_budget: None,
            reasoning_effort: None,
            profile: Default::default(),
        };
        let attempts = request_attempts(&req);
        assert!(
            attempts.len() >= 2,
            "compat auto must offer a no-frequency_penalty shape"
        );
        let first = attempts[0];
        assert!(first.include_frequency_penalty);
        let body = build_body(&req, first, None);
        assert!(body.get("frequency_penalty").is_some());

        let body_text = r#"{"code":"invalid-argument","error":"Model grok-4.5 does not support parameter frequencyPenalty."}"#;
        let next = next_degraded_attempt(
            &attempts,
            0,
            ProviderErrorKind::UnsupportedRequestShape,
            body_text,
        )
        .expect("must degrade");
        assert!(!attempts[next].include_frequency_penalty);
        let retry = build_body(&req, attempts[next], None);
        assert!(
            retry.get("frequency_penalty").is_none(),
            "retry must omit frequency_penalty"
        );
        // top_p/temperature stay — only the rejected field is stripped.
        assert!(retry.get("top_p").is_some());
        assert!(retry.get("temperature").is_some());

        // Strict compat does not offer the degradation.
        req.profile.compat = crate::types::CompatMode::Strict;
        let strict = request_attempts(&req);
        assert_eq!(strict.len(), 1);
        assert!(strict[0].include_frequency_penalty);
    }

    /// Verbatim body from api.x.ai for a wrong key. xAI answers 400, not 401,
    /// so classifying on status alone reports a request-shape problem and the
    /// compat ladder retries a doomed request.
    #[test]
    fn xai_bad_key_400_is_an_auth_error_not_a_request_shape_error() {
        let body = r#"{"code":"invalid-argument","error":"Incorrect API key provided. You can obtain an API key from https://console.x.ai."}"#;
        assert_eq!(
            classify_http_error(StatusCode::BAD_REQUEST, body),
            ProviderErrorKind::Auth
        );
    }

    /// Verbatim body from api.x.ai when the Authorization header is absent.
    #[test]
    fn xai_missing_key_is_an_auth_error() {
        let body = r#"{"code":"unauthenticated","error":"API key is missing."}"#;
        assert_eq!(
            classify_http_error(StatusCode::UNAUTHORIZED, body),
            ProviderErrorKind::Auth
        );
    }

    /// The auth guard runs before the 400 branch, so it must not capture the
    /// context-length errors that branch exists to classify.
    #[test]
    fn token_limit_errors_are_not_mistaken_for_auth_errors() {
        for body in [
            r#"{"error":"This model's maximum context length is 8192 tokens"}"#,
            r#"{"error":"too many tokens in request"}"#,
            r#"{"error":"context_length_exceeded"}"#,
            // Verbatim host wording that previously misclassified as compat shape.
            r#"{"code":"invalid-argument","error":"This model's maximum prompt length is 500000 but the request contains 500547 tokens."}"#,
            r#"{"error":"prompt length 128 exceeds context"}"#,
        ] {
            assert_eq!(
                classify_http_error(StatusCode::BAD_REQUEST, body),
                ProviderErrorKind::RequestTooLarge,
                "token-limit wording must stay a size error: {body}"
            );
        }
    }

    #[test]
    fn generic_invalid_request_type_does_not_hide_context_failures() {
        let body = r#"{"error":{"type":"invalid_request_error","message":"maximum context length is 8192 tokens"}}"#;
        let parsed = parse_api_error(Some(StatusCode::BAD_REQUEST), body);
        assert_eq!(parsed.kind, ProviderErrorKind::RequestTooLarge);
        assert_eq!(parsed.retryable, Some(false));
    }

    #[test]
    fn structured_429_codes_override_the_http_status() {
        for (body, expected, retryable) in [
            (
                r#"{"error":{"message":"no route","code":"capacity_unavailable","retryable":true}}"#,
                ProviderErrorKind::CapacityUnavailable,
                Some(true),
            ),
            (
                r#"{"error":{"message":"bad tool JSON","code":"tool_protocol_error","retryable":true}}"#,
                ProviderErrorKind::ToolProtocol,
                Some(true),
            ),
            (
                r#"{"error":{"message":"payload rejected","code":"service_unavailable","retryable":false}}"#,
                ProviderErrorKind::Outage,
                Some(false),
            ),
        ] {
            let parsed = parse_api_error(Some(StatusCode::TOO_MANY_REQUESTS), body);
            assert_eq!(parsed.kind, expected, "{body}");
            assert_eq!(parsed.retryable, retryable, "{body}");
        }
    }

    #[test]
    fn top_level_pipe_service_error_is_not_capacity() {
        let body = r#"{"error":"external model service unavailable","message":"external model service unavailable","type":"service_unavailable_error","code":"service_unavailable","retryable":true,"retry_after_seconds":1}"#;
        let parsed = parse_api_error(Some(StatusCode::TOO_MANY_REQUESTS), body);

        assert_eq!(parsed.kind, ProviderErrorKind::Outage);
        assert_eq!(parsed.code.as_deref(), Some("service_unavailable"));
        assert_eq!(parsed.retryable, Some(true));
    }

    #[test]
    fn legacy_http_statuses_get_a_bounded_retry_contract() {
        assert_eq!(
            parse_api_error(Some(StatusCode::BAD_GATEWAY), "upstream failed").retryable,
            Some(true)
        );
        assert_eq!(
            parse_api_error(Some(StatusCode::BAD_REQUEST), "invalid payload").retryable,
            Some(false)
        );
    }

    #[test]
    fn cloudflare_origin_timeout_has_a_useful_retryable_error() {
        let status = StatusCode::from_u16(524).expect("Cloudflare timeout status");
        let provider =
            parse_api_error(Some(status), "error code: 524").into_provider_error(Some(status));

        assert_eq!(provider.kind, ProviderErrorKind::Outage);
        assert_eq!(provider.retryable, Some(true));
        assert_eq!(
            provider.message,
            "API error 524 Gateway Timeout: error code: 524"
        );
    }

    #[test]
    fn provider_failures_do_not_trigger_same_route_shape_mutation() {
        let req = crate::types::ChatRequest {
            model: "m".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: Default::default(),
        };
        let attempts = request_attempts(&req);
        assert_eq!(
            next_degraded_attempt(&attempts, 0, ProviderErrorKind::Outage, "server error"),
            None
        );
    }

    #[test]
    fn system_and_user_become_text_messages() {
        let out = to_openai_messages(&[Message::system("sys"), Message::user("hi")]);
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[0]["content"], "sys");
        assert_eq!(out[1]["role"], "user");
        assert_eq!(out[1]["content"], "hi");
    }

    #[test]
    fn assistant_tool_call_omits_content_rather_than_null() {
        let out =
            to_openai_messages(&[Message::assistant(vec![crate::types::Content::ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            }])]);
        // Ollama rejects null content; we omit the key entirely.
        assert!(out[0].get("content").is_none());
        assert!(out[0]["tool_calls"].is_array());
        assert_eq!(out[0]["tool_calls"][0]["function"]["name"], "bash");
    }

    #[test]
    fn thinking_is_preserved_as_inline_tag() {
        let out = to_openai_messages(&[Message::assistant(vec![
            crate::types::Content::Thinking {
                text: "my reasoning".into(),
                signature: None,
            },
            crate::types::Content::Text("the answer".into()),
        ])]);
        let content = out[0]["content"].as_str().unwrap();
        assert!(content.contains("<thinking>"));
        assert!(content.contains("my reasoning"));
        assert!(content.contains("the answer"));
    }

    #[test]
    fn tool_result_maps_to_tool_role() {
        let out = to_openai_messages(&[Message::tool_result("call_1", "the output")]);
        assert_eq!(out[0]["role"], "tool");
        assert_eq!(out[0]["tool_call_id"], "call_1");
        assert_eq!(out[0]["content"], "the output");
    }

    #[test]
    fn request_body_can_omit_stream_options() {
        let req = crate::types::ChatRequest {
            model: "m".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: Default::default(),
        };

        let normal = build_body(&req, request_attempts(&req)[0], None);
        assert_eq!(normal["stream_options"]["include_usage"], true);

        let fallback = build_body(&req, request_attempts(&req)[1], None);
        assert!(fallback.get("stream_options").is_none());
        assert_eq!(fallback["stream"], true);
    }

    #[test]
    fn request_body_can_carry_provider_metadata() {
        let req = crate::types::ChatRequest {
            model: "m".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: Default::default(),
        };
        let metadata = serde_json::json!({
            "endpoint_name": "pipenetworkai",
            "request_type": "code_generation"
        });

        let body = build_body(&req, request_attempts(&req)[0], Some(&metadata));

        assert_eq!(body["metadata"], metadata);
    }

    #[test]
    fn request_body_carries_recovery_sampling() {
        // top_p/frequency_penalty (set by recovery sampling on a retry) reach the
        // wire; absent fields stay absent so the provider default applies.
        let mut req = crate::types::ChatRequest {
            model: "m".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: Default::default(),
        };
        let plain = build_body(&req, request_attempts(&req)[0], None);
        assert!(plain.get("top_p").is_none(), "omitted when unset");
        assert!(plain.get("frequency_penalty").is_none());

        req.temperature = Some(0.9);
        req.top_p = Some(0.95);
        req.frequency_penalty = Some(0.4);
        let hot = build_body(&req, request_attempts(&req)[0], None);
        // f32 → JSON f64 isn't exact (0.9f32 ≈ 0.89999996), so compare with tolerance.
        let near = |v: &serde_json::Value, want: f64| (v.as_f64().unwrap() - want).abs() < 1e-6;
        assert!(
            near(&hot["temperature"], 0.9),
            "temperature: {}",
            hot["temperature"]
        );
        assert!(near(&hot["top_p"], 0.95), "top_p: {}", hot["top_p"]);
        assert!(
            near(&hot["frequency_penalty"], 0.4),
            "frequency_penalty: {}",
            hot["frequency_penalty"]
        );
    }

    #[test]
    fn request_body_emits_reasoning_effort_only_when_set() {
        let mut req = crate::types::ChatRequest {
            model: "m".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: Default::default(),
        };
        // Absent by default so the endpoint's own default applies.
        let plain = build_body(&req, request_attempts(&req)[0], None);
        assert!(plain.get("reasoning_effort").is_none());

        // Each level reaches the wire as its lowercase string.
        req.reasoning_effort = Some(crate::types::ReasoningEffort::High);
        let body = build_body(&req, request_attempts(&req)[0], None);
        assert_eq!(body["reasoning_effort"], "high");

        req.reasoning_effort = Some(crate::types::ReasoningEffort::Minimal);
        let body = build_body(&req, request_attempts(&req)[0], None);
        assert_eq!(body["reasoning_effort"], "minimal");
    }

    #[test]
    fn deepseek_v4_request_uses_its_thinking_tool_contract() {
        let req = crate::types::ChatRequest {
            model: "DeepSeek-V4-Flash-0731".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: true,
            canonical_objective: None,
            messages: vec![Message::user("read README")].into(),
            tools: vec![ToolSpec {
                name: "read".into(),
                description: "read a file".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            }]
            .into(),
            max_tokens: 512,
            temperature: Some(0.2),
            top_p: Some(0.9),
            frequency_penalty: Some(0.1),
            thinking_budget: None,
            reasoning_effort: Some(crate::types::ReasoningEffort::Xhigh),
            profile: RequestProfile {
                tool_mode: ToolMode::Required,
                deepseek_compat: DeepSeekCompat::On,
                ..RequestProfile::default()
            },
        };
        let caps = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            &req.model,
            req.profile.deepseek_compat,
        );
        let attempts = request_attempts_for(&req, &caps);
        let body = build_body_with_capabilities(&req, attempts[0], None, &caps);
        assert_eq!(body["model"], "deepseek-v4-flash");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("frequency_penalty").is_none());
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["tools"][0]["function"]["strict"], true);
        assert!(
            body["tools"][0]["function"]["parameters"]["properties"]["path"]
                .get("minLength")
                .is_none()
        );

        let mut final_req = req.clone();
        final_req.profile.deepseek_thinking = Some(false);
        let final_caps = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            &final_req.model,
            final_req.profile.deepseek_compat,
        );
        let final_attempts = request_attempts_for(&final_req, &final_caps);
        let final_body =
            build_body_with_capabilities(&final_req, final_attempts[0], None, &final_caps);
        assert_eq!(final_body["thinking"]["type"], "disabled");
        assert!(final_body.get("reasoning_effort").is_none());
    }

    #[test]
    fn deepseek_replays_reasoning_and_non_null_tool_content() {
        let caps = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            "deepseek-v4-flash",
            DeepSeekCompat::Auto,
        );
        let messages = to_openai_messages_with_capabilities(
            &[Message {
                role: Role::Assistant,
                content: vec![
                    Content::Thinking {
                        text: "inspect the file".into(),
                        signature: None,
                    },
                    Content::ToolCall {
                        id: "call_1".into(),
                        name: "read".into(),
                        arguments: "{\"path\":\"README.md\"}".into(),
                    },
                ],
            }],
            &caps,
        );
        assert_eq!(messages[0]["content"], "");
        assert_eq!(messages[0]["reasoning_content"], "inspect the file");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
    }

    #[test]
    fn deepseek_strict_schema_failure_has_one_non_strict_retry() {
        let req = crate::types::ChatRequest {
            model: "deepseek-v4-flash".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: true,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![ToolSpec {
                name: "read".into(),
                description: String::new(),
                parameters: serde_json::json!({"type": "object"}),
            }]
            .into(),
            max_tokens: 32,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                deepseek_compat: DeepSeekCompat::On,
                ..RequestProfile::default()
            },
        };
        let caps = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            &req.model,
            req.profile.deepseek_compat,
        );
        let attempts = request_attempts_for(&req, &caps);
        assert!(attempts[0].strict_tools);
        let next = next_degraded_attempt(
            &attempts,
            0,
            ProviderErrorKind::UnsupportedTools,
            "strict schema rejected: unsupported keyword",
        )
        .expect("strict mode should have one fallback");
        assert!(!attempts[next].strict_tools);

        let mut strict_request = req.clone();
        strict_request.profile.compat = CompatMode::Strict;
        let strict_attempts = request_attempts_for(&strict_request, &caps);
        assert_eq!(strict_attempts.len(), 2);
        assert!(strict_attempts[0].strict_tools);
        assert!(!strict_attempts[1].strict_tools);
        assert!(strict_attempts[1].strict_fallback);
        assert_eq!(
            next_degraded_attempt(
                &strict_attempts,
                1,
                ProviderErrorKind::UnsupportedRequestShape,
                "stream_options unsupported"
            ),
            None
        );

        let mut client_fallback = req;
        client_fallback.profile.deepseek_strict = Some(false);
        let fallback_attempts = request_attempts_for(&client_fallback, &caps);
        assert!(
            fallback_attempts
                .iter()
                .all(|attempt| !attempt.strict_tools)
        );
        assert!(
            fallback_attempts
                .iter()
                .all(|attempt| !attempt.strict_fallback)
        );
    }

    #[test]
    fn deepseek_strict_fallback_keeps_prior_usage_degradation() {
        let req = crate::types::ChatRequest {
            model: "deepseek-v4-flash".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: true,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![ToolSpec {
                name: "read".into(),
                description: String::new(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            }]
            .into(),
            max_tokens: 32,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                deepseek_compat: DeepSeekCompat::On,
                ..RequestProfile::default()
            },
        };
        let caps = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            &req.model,
            req.profile.deepseek_compat,
        );
        let attempts = request_attempts_for(&req, &caps);

        let without_usage = next_degraded_attempt(
            &attempts,
            0,
            ProviderErrorKind::UnsupportedRequestShape,
            "stream_options unsupported",
        )
        .expect("usage rejection should have a degraded attempt");
        assert!(!attempts[without_usage].include_usage);

        let fallback = next_degraded_attempt(
            &attempts,
            without_usage,
            ProviderErrorKind::UnsupportedRequestShape,
            "strict schema rejected: unsupported keyword",
        )
        .expect("strict schema rejection should have a matching fallback");
        assert!(!attempts[fallback].strict_tools);
        assert!(
            !attempts[fallback].include_usage,
            "strict fallback must not re-add stream_options"
        );
    }

    #[test]
    fn deepseek_model_capability_error_is_safe_to_cache() {
        assert!(is_deepseek_strict_schema_unsupported(
            r#"{"error":"model `fireworks/pipe/deepseek-v4-flash-0731` does not support strict tool schemas (code: model_does_not_support_strict_tools)"} "#
        ));
        assert!(!is_deepseek_strict_schema_unsupported(
            "invalid strict schema: additionalProperties must be false"
        ));
    }

    #[test]
    fn review_evidence_repair_text_is_not_quality_rejected() {
        for text in [
            "insufficient evidence after review repair",
            "model needs inspected evidence before answering",
            "review evidence repair exhausted",
            "quality_rejected: review evidence repair exhausted",
        ] {
            assert!(
                !is_quality_rejected_text(text),
                "review repair text should not be quality_rejected: {text}"
            );
        }
    }

    #[test]
    fn non_review_quality_rejected_text_still_classifies() {
        for text in [
            "quality_rejected: provider quality check failed",
            r#"{"error":"quality_rejected: provider quality check failed"}"#,
            r#"{"error":{"message":"quality_rejected: provider quality check failed"}}"#,
        ] {
            assert!(is_quality_rejected_text(text), "{text}");
            assert_eq!(
                classify_http_error(StatusCode::BAD_REQUEST, text),
                ProviderErrorKind::QualityRejected,
                "{text}"
            );
        }
    }
}
