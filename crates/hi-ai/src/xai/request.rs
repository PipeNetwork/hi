//! Request translation: `hi_ai` messages → xAI Responses API wire JSON.

use reqwest::StatusCode;
use serde_json::{Value, json};

use crate::provider::{ProviderError, ProviderErrorKind};
use crate::types::{ChatRequest, Content, Message, ReasoningEffort, Role, ToolMode, WireAudit};

/// Build the Responses API body. Always `store: false` so hi keeps the
/// transcript; always ask for encrypted reasoning so the next round can replay it.
pub(crate) fn build_body(request: &ChatRequest) -> Value {
    let include_tools = !request.tools.is_empty();
    let mut body = json!({
        "model": request.model,
        "input": to_responses_input(&request.messages),
        "max_output_tokens": request.max_tokens,
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"],
    });
    if include_tools {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": xai_tool_parameters(&tool.parameters),
                })
            })
            .collect();
        body["tools"] = json!(tools);
        match request.profile.tool_mode {
            ToolMode::Required => body["tool_choice"] = json!("required"),
            ToolMode::ChatOnly => body["tool_choice"] = json!("none"),
            _ => {}
        }
    }
    if let Some(effort) = request.reasoning_effort {
        body["reasoning"] = json!({ "effort": xai_reasoning_effort(effort) });
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }
    body
}

/// xAI grok-4.6 accepts low/medium/high/xhigh. `minimal` is not a valid wire
/// value and would 400; collapse it to `low`.
pub(crate) fn xai_reasoning_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
    }
}

/// xAI compiles each tool schema into a grammar. The root must be
/// `"type": "object"` or a union of objects; a sibling `oneOf` of
/// `{ "required": ["path"] }` fragments (hi's `read` tool) is rejected as a
/// non-object branch. Expand those into typed object variants; leave a plain
/// object schema alone.
fn xai_tool_parameters(schema: &Value) -> Value {
    let Some(root) = schema.as_object() else {
        return json!({ "type": "object", "properties": {} });
    };
    let union_key = if root.contains_key("oneOf") {
        Some("oneOf")
    } else if root.contains_key("anyOf") {
        Some("anyOf")
    } else {
        None
    };
    if let Some(key) = union_key {
        return expand_root_union(root, key);
    }
    let mut out = schema.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.entry("type").or_insert(json!("object"));
        if obj.get("type").and_then(Value::as_str) == Some("object") {
            obj.entry("properties").or_insert(json!({}));
        }
    }
    out
}

fn expand_root_union(root: &serde_json::Map<String, Value>, key: &'static str) -> Value {
    let branches = root
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let sibling_properties = root.get("properties");
    let sibling_additional = root.get("additionalProperties");
    let sibling_required = root.get("required").and_then(Value::as_array);
    let expanded: Vec<Value> = branches
        .iter()
        .map(|branch| {
            let mut object = match branch {
                Value::Object(map) => Value::Object(map.clone()),
                _ => json!({}),
            };
            let map = object
                .as_object_mut()
                .expect("object variant is always an object");
            let has_properties = map.contains_key("properties");
            map.insert("type".into(), json!("object"));
            if !has_properties {
                map.insert(
                    "properties".into(),
                    sibling_properties.cloned().unwrap_or_else(|| json!({})),
                );
            }
            if !map.contains_key("additionalProperties")
                && let Some(additional) = sibling_additional
            {
                map.insert("additionalProperties".into(), additional.clone());
            }
            if let Some(required) = sibling_required {
                let mut merged = map
                    .get("required")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for item in required {
                    if !merged.contains(item) {
                        merged.push(item.clone());
                    }
                }
                if !merged.is_empty() {
                    map.insert("required".into(), Value::Array(merged));
                }
            }
            object
        })
        .collect();
    let mut out = json!({ key: expanded });
    if let Some(description) = root.get("description") {
        out["description"] = description.clone();
    }
    out
}

/// Flatten hi messages into Responses `input` items. Tool calls and results
/// become typed items; signed thinking becomes a `reasoning` item with
/// encrypted content. Unsigned thinking is dropped, matching Anthropic.
pub(crate) fn to_responses_input(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message.role {
            Role::System => {
                let text = message.text();
                if !text.is_empty() {
                    out.push(json!({ "role": "system", "content": text }));
                }
            }
            Role::User => out.push(user_input_item(message)),
            Role::Assistant => out.extend(assistant_input_items(message)),
            Role::Tool => {
                for block in &message.content {
                    if let Content::ToolResult { call_id, output } = block {
                        out.push(json!({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": output,
                        }));
                    }
                }
            }
        }
    }
    out
}

fn user_input_item(message: &Message) -> Value {
    let has_image = message
        .content
        .iter()
        .any(|block| matches!(block, Content::Image { .. }));
    if !has_image {
        return json!({ "role": "user", "content": message.text() });
    }
    let mut parts = Vec::new();
    for block in &message.content {
        match block {
            Content::Image { data, media_type } => {
                parts.push(json!({
                    "type": "input_image",
                    "image_url": format!("data:{media_type};base64,{data}"),
                }));
            }
            Content::Text(text) if !text.is_empty() => {
                parts.push(json!({ "type": "input_text", "text": text }));
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        parts.push(json!({ "type": "input_text", "text": message.text() }));
    }
    json!({ "role": "user", "content": parts })
}

fn assistant_input_items(message: &Message) -> Vec<Value> {
    let mut items = Vec::new();
    let mut text = String::new();
    for block in &message.content {
        match block {
            Content::Thinking {
                signature: Some(signature),
                text: summary,
            } if !signature.is_empty() => {
                let mut item = json!({
                    "type": "reasoning",
                    "encrypted_content": signature,
                });
                if !summary.is_empty() {
                    item["summary"] = json!([{ "type": "summary_text", "text": summary }]);
                }
                items.push(item);
            }
            Content::Text(chunk) => text.push_str(chunk),
            Content::ToolCall {
                id,
                name,
                arguments,
            } => {
                items.push(json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": name,
                    "arguments": arguments,
                }));
            }
            Content::Thinking { .. } | Content::ToolResult { .. } | Content::Image { .. } => {}
        }
    }
    if !text.is_empty() {
        items.push(json!({ "role": "assistant", "content": text }));
    }
    items
}

pub(crate) fn wire_audit(
    request: &ChatRequest,
    route: &str,
    body: &Value,
    accepted: bool,
    response_status: Option<u16>,
) -> WireAudit {
    let include_tools = body.get("tools").and_then(Value::as_array).is_some();
    WireAudit {
        provider: "xai".to_string(),
        route: route.to_string(),
        model: request.model.clone(),
        output_token_parameter: "max_output_tokens".to_string(),
        max_output_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        reasoning_request: body
            .get("reasoning")
            .and_then(|value| value.get("effort"))
            .and_then(Value::as_str)
            .map(str::to_string),
        reasoning_replay: request
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .find_map(|content| match content {
                Content::Thinking {
                    signature: Some(_), ..
                } => Some("encrypted_reasoning".to_string()),
                _ => None,
            }),
        native_tools_enabled: include_tools,
        tool_count: body
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        strict_schema: false,
        tool_choice: body
            .get("tool_choice")
            .and_then(Value::as_str)
            .map(str::to_string),
        request_attempt: 1,
        compatibility_fallback: None,
        accepted,
        request_body: Some(body.clone()),
        response_status,
    }
}

pub(crate) fn classify_http_error(status: StatusCode, text: &str) -> ProviderErrorKind {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ProviderErrorKind::Auth,
        _ if is_auth_text(text) => ProviderErrorKind::Auth,
        StatusCode::TOO_MANY_REQUESTS => ProviderErrorKind::RateLimit,
        StatusCode::PAYLOAD_TOO_LARGE => ProviderErrorKind::RequestTooLarge,
        StatusCode::NOT_FOUND => ProviderErrorKind::ModelUnavailable,
        s if s.is_server_error() => ProviderErrorKind::Outage,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
            if mentions(
                text,
                &[
                    "context length",
                    "context_length_exceeded",
                    "maximum prompt length",
                    "prompt length",
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

pub(crate) fn provider_error_from_http(status: StatusCode, text: &str) -> ProviderError {
    let kind = classify_http_error(status, text);
    let message = api_error_message(text).unwrap_or_else(|| text.to_string());
    ProviderError::new(kind, format!("API error {status}: {message}"))
        .with_http_status(Some(status.as_u16()))
        .with_api_contract(None, Some(kind_retryable(kind)), None)
}

fn kind_retryable(kind: ProviderErrorKind) -> bool {
    matches!(
        kind,
        ProviderErrorKind::RateLimit
            | ProviderErrorKind::CapacityUnavailable
            | ProviderErrorKind::Outage
            | ProviderErrorKind::MalformedStream
            | ProviderErrorKind::ToolProtocol
    )
}

fn api_error_message(text: &str) -> Option<String> {
    let value: Value = serde_json::from_str(text).ok()?;
    value
        .get("error")
        .and_then(|error| {
            error
                .as_str()
                .map(str::to_string)
                .or_else(|| {
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .or_else(|| {
                    error
                        .get("error")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
        })
        .or_else(|| {
            value
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

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
mod tests {
    use super::*;
    use crate::types::{Message, RequestProfile, ToolSpec};

    fn request(tools: Vec<ToolSpec>, profile: RequestProfile) -> ChatRequest {
        ChatRequest {
            model: "grok-4.6".into(),
            request_id: None,
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
            parameters: json!({
                "type": "object",
                "properties": { "command": { "type": "string" } },
                "required": ["command"]
            }),
        }
    }

    #[test]
    fn body_uses_responses_shape_and_omits_frequency_penalty() {
        let body = build_body(&request(
            vec![bash_tool()],
            RequestProfile {
                tool_mode: ToolMode::Auto,
                ..Default::default()
            },
        ));
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_output_tokens"], 16);
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert!(body.get("messages").is_none());
        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
        assert!(body.get("stop").is_none());
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "bash");
        assert!(body["tools"][0].get("function").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn required_tool_mode_sets_tool_choice() {
        let body = build_body(&request(
            vec![bash_tool()],
            RequestProfile {
                tool_mode: ToolMode::Required,
                ..Default::default()
            },
        ));
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn chat_only_keeps_tools_with_none() {
        let body = build_body(&request(
            vec![bash_tool()],
            RequestProfile {
                tool_mode: ToolMode::ChatOnly,
                ..Default::default()
            },
        ));
        assert!(body.get("tools").is_some());
        assert_eq!(body["tool_choice"], "none");
    }

    #[test]
    fn minimal_effort_is_sent_as_low() {
        let mut req = request(vec![], Default::default());
        req.reasoning_effort = Some(ReasoningEffort::Minimal);
        let body = build_body(&req);
        assert_eq!(body["reasoning"]["effort"], "low");
    }

    #[test]
    fn xhigh_effort_is_preserved() {
        let mut req = request(vec![], Default::default());
        req.reasoning_effort = Some(ReasoningEffort::Xhigh);
        let body = build_body(&req);
        assert_eq!(body["reasoning"]["effort"], "xhigh");
    }

    #[test]
    fn tool_result_and_call_replay_as_typed_items() {
        let input = to_responses_input(&[
            Message::user("list files"),
            Message::assistant(vec![
                Content::Thinking {
                    text: "need ls".into(),
                    signature: Some("enc-1".into()),
                },
                Content::ToolCall {
                    id: "call_1".into(),
                    name: "bash".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                },
            ]),
            Message::tool_result("call_1", "Cargo.toml"),
        ]);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["encrypted_content"], "enc-1");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["name"], "bash");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_1");
        assert_eq!(input[3]["output"], "Cargo.toml");
    }

    #[test]
    fn read_oneof_required_branches_become_typed_objects() {
        let mut req = request(
            vec![ToolSpec {
                name: "read".into(),
                description: "Read a file".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "paths": { "type": "array", "items": { "type": "string" } }
                    },
                    "oneOf": [
                        { "required": ["path"] },
                        { "required": ["paths"] }
                    ],
                    "additionalProperties": false
                }),
            }],
            RequestProfile {
                tool_mode: ToolMode::Auto,
                ..Default::default()
            },
        );
        let body = build_body(&req);
        let parameters = &body["tools"][0]["parameters"];
        assert!(parameters.get("type").is_none());
        let branches = parameters["oneOf"].as_array().unwrap();
        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0]["type"], "object");
        assert_eq!(branches[0]["required"], json!(["path"]));
        assert_eq!(branches[0]["properties"]["path"]["type"], "string");
        assert_eq!(branches[1]["required"], json!(["paths"]));
        assert_eq!(branches[0]["additionalProperties"], false);
        req.tools = vec![bash_tool()].into();
        let plain = build_body(&req);
        assert_eq!(plain["tools"][0]["parameters"]["type"], "object");
        assert!(plain["tools"][0]["parameters"].get("oneOf").is_none());
    }

    #[test]
    fn unsigned_thinking_is_dropped() {
        let input = to_responses_input(&[Message::assistant(vec![
            Content::Thinking {
                text: "scratch".into(),
                signature: None,
            },
            Content::Text("done".into()),
        ])]);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["role"], "assistant");
        assert_eq!(input[0]["content"], "done");
    }

    #[test]
    fn xai_400_bad_key_is_auth() {
        let body = r#"{"code":"invalid-argument","error":"invalid access token"}"#;
        assert_eq!(
            classify_http_error(StatusCode::BAD_REQUEST, body),
            ProviderErrorKind::Auth
        );
    }
}
