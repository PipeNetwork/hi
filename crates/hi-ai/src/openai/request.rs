//! Request translation: `hi_ai` messages → OpenAI Chat Completions wire JSON,
//! plus the degraded-retry attempt ladder and HTTP error classification.

use reqwest::StatusCode;
use serde_json::{Value, json};

use crate::provider::ProviderErrorKind;
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
    // A persistent 5xx that already survived transport retries: try the next
    // shape in case the response was really a request-shape problem.
    if matches!(kind, ProviderErrorKind::Outage) && after < attempts.len() {
        return Some(after);
    }
    None
}

pub(crate) fn request_attempts(request: &ChatRequest) -> Vec<RequestAttempt> {
    let include_usage = request.profile.stream_usage.unwrap_or(true);
    let include_tools =
        !request.tools.is_empty() && request.profile.tool_mode != ToolMode::ChatOnly;
    let include_frequency_penalty = request.frequency_penalty.is_some();
    let mut attempts = vec![RequestAttempt {
        include_usage,
        include_tools,
        include_frequency_penalty,
        status: None,
    }];
    if request.profile.compat == CompatMode::Strict {
        return attempts;
    }
    if include_usage {
        attempts.push(RequestAttempt {
            include_usage: false,
            include_tools,
            include_frequency_penalty,
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
            status: Some(
                "compat: provider rejected frequency_penalty; retried without it",
            ),
        });
        if include_usage {
            attempts.push(RequestAttempt {
                include_usage: false,
                include_tools,
                include_frequency_penalty: false,
                status: Some(
                    "compat: provider rejected stream_options/frequency_penalty; retried without both",
                ),
            });
        }
    }
    attempts
}

/// xAI returns camelCase (`frequencyPenalty`); OpenAI-style wording uses snake_case.
fn is_unsupported_frequency_penalty_text(text: &str) -> bool {
    mentions(
        text,
        &[
            "frequency_penalty",
            "frequencypenalty",
            "frequency penalty",
        ],
    ) && mentions(text, &["does not support", "unsupported", "unknown", "invalid"])
}

pub(crate) fn classify_http_error(status: StatusCode, text: &str) -> ProviderErrorKind {
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

pub(crate) fn capacity_retry_after_seconds(text: &str) -> Option<u64> {
    serde_json::from_str::<Value>(text).ok().and_then(|value| {
        value
            .get("retry_after_seconds")
            .and_then(Value::as_u64)
            .or_else(|| {
                value
                    .get("error")
                    .and_then(|error| error.get("retry_after_seconds"))
                    .and_then(Value::as_u64)
            })
    })
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

pub(crate) fn build_body(
    request: &ChatRequest,
    attempt: RequestAttempt,
    metadata: Option<&Value>,
) -> Value {
    let messages = to_openai_messages(&request.messages);
    let mut body = json!({
        "model": request.model,
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
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = json!(tools);
        if request.profile.tool_mode == ToolMode::Required {
            body["tool_choice"] = json!("required");
        }
    }
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
    // Abstract reasoning level (GPT-5/o-series style). Endpoints that don't
    // support it validate the value and 400 on an unknown one, so we only send
    // it when explicitly requested. The Anthropic adapter ignores this field
    // and uses `thinking_budget` instead.
    if let Some(effort) = request.reasoning_effort {
        body["reasoning_effort"] = json!(effort.as_str());
    }
    body
}

/// Flatten neutral messages into OpenAI's wire shape. Thinking blocks are
/// dropped (the Chat Completions API has no place to put them).
pub(crate) fn to_openai_messages(messages: &[Message]) -> Vec<Value> {
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
                let mut content = String::new();
                if !thinking.is_empty() {
                    content.push_str(&format!("<thinking>\n{thinking}\n</thinking>\n"));
                }
                content.push_str(&text);

                let mut msg = json!({ "role": "assistant" });
                if tool_calls.is_empty() {
                    msg["content"] = json!(content);
                } else {
                    msg["tool_calls"] = json!(tool_calls);
                    // Omit content when empty; OpenAI allows it and some servers
                    // (e.g. Ollama) reject an explicit null.
                    if !content.is_empty() {
                        msg["content"] = json!(content);
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
        build_body, classify_http_error, is_quality_rejected_text, is_unsupported_frequency_penalty_text,
        next_degraded_attempt, request_attempts, to_openai_messages,
    };
    use reqwest::StatusCode;

    use crate::provider::ProviderErrorKind;
    use crate::types::Message;

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
        ] {
            assert_eq!(
                classify_http_error(StatusCode::BAD_REQUEST, body),
                ProviderErrorKind::RequestTooLarge,
                "token-limit wording must stay a size error: {body}"
            );
        }
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
