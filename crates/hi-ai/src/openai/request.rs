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
    // A persistent outage that already survived transport retries: try the next
    // (degraded) shape in case the 5xx was really a request-shape problem.
    if matches!(kind, ProviderErrorKind::Outage) && after < attempts.len() {
        return Some(after);
    }
    None
}

pub(crate) fn request_attempts(request: &ChatRequest) -> Vec<RequestAttempt> {
    let include_usage = request.profile.stream_usage.unwrap_or(true);
    let include_tools =
        !request.tools.is_empty() && request.profile.tool_mode != ToolMode::ChatOnly;
    let mut attempts = vec![RequestAttempt {
        include_usage,
        include_tools,
        status: None,
    }];
    if request.profile.compat == CompatMode::Strict {
        return attempts;
    }
    if include_usage {
        attempts.push(RequestAttempt {
            include_usage: false,
            include_tools,
            status: Some(
                "compat: provider rejected stream_options; retried without usage streaming",
            ),
        });
    }
    attempts
}

pub(crate) fn classify_http_error(status: StatusCode, text: &str) -> ProviderErrorKind {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ProviderErrorKind::Auth,
        StatusCode::TOO_MANY_REQUESTS => ProviderErrorKind::RateLimit,
        StatusCode::CONFLICT | StatusCode::SERVICE_UNAVAILABLE
            if is_capacity_unavailable_text(text) =>
        {
            ProviderErrorKind::CapacityUnavailable
        }
        s if s.is_server_error() => ProviderErrorKind::Outage,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
            if mentions(
                text,
                &[
                    "maximum allowed size",
                    "input exceeds",
                    "context length",
                    "context_length_exceeded",
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
        &[
            "capacity_unavailable",
            "capacity temporarily unavailable",
            "temporarily unavailable",
        ],
    )
}

fn mentions(text: &str, needles: &[&str]) -> bool {
    let lower = text.to_ascii_lowercase();
    needles.iter().any(|needle| lower.contains(needle))
}

pub(crate) fn build_body(request: &ChatRequest, attempt: RequestAttempt) -> Value {
    let messages = to_openai_messages(&request.messages);
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
        "max_tokens": request.max_tokens,
    });
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
    if let Some(frequency_penalty) = request.frequency_penalty {
        body["frequency_penalty"] = json!(frequency_penalty);
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
    use super::{build_body, request_attempts, to_openai_messages};
    use crate::types::Message;

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
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: Default::default(),
        };

        let normal = build_body(&req, request_attempts(&req)[0]);
        assert_eq!(normal["stream_options"]["include_usage"], true);

        let fallback = build_body(&req, request_attempts(&req)[1]);
        assert!(fallback.get("stream_options").is_none());
        assert_eq!(fallback["stream"], true);
    }

    #[test]
    fn request_body_carries_recovery_sampling() {
        // top_p/frequency_penalty (set by recovery sampling on a retry) reach the
        // wire; absent fields stay absent so the provider default applies.
        let mut req = crate::types::ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: Default::default(),
        };
        let plain = build_body(&req, request_attempts(&req)[0]);
        assert!(plain.get("top_p").is_none(), "omitted when unset");
        assert!(plain.get("frequency_penalty").is_none());

        req.temperature = Some(0.9);
        req.top_p = Some(0.95);
        req.frequency_penalty = Some(0.4);
        let hot = build_body(&req, request_attempts(&req)[0]);
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
}
