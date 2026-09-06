//! Astra request translation, preserving complete Responses output history.

use crate::{ChatRequest, Content, Message, ReasoningEffort, Role, ToolMode, WireAudit};
use serde_json::{Value, json};

pub(super) fn build_body(request: &ChatRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "input": to_input(&request.messages),
        "max_output_tokens": request.max_tokens,
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_options": {"ttl": "30m"},
    });
    if !request.tools.is_empty() {
        body["tools"] = tool_definitions(request);
        body["tool_choice"] = json!(match request.profile.tool_mode {
            ToolMode::Required => "required",
            ToolMode::ChatOnly => "none",
            ToolMode::Auto | ToolMode::ReadOnly => "auto",
        });
    }
    if let Some(effort) = request.reasoning_effort {
        body["reasoning"] = json!({"effort": match effort {
            ReasoningEffort::Minimal => "low",
            effort => effort.as_str(),
        }});
    }
    // Astra rejects temperature/top_p/logprobs. Recovery sampling and legacy
    // Chat Completions token/penalty fields must not leak into Responses.
    body
}

fn tool_definitions(request: &ChatRequest) -> Value {
    json!(
        request
            .tools
            .iter()
            .map(|tool| json!({
                "type": "function", "name": tool.name, "description": tool.description,
                "parameters": tool.parameters,
                // Preserve optional fields and local validation semantics. Responses
                // otherwise attempts to normalize schemas into strict mode.
                "strict": false,
            }))
            .collect::<Vec<_>>()
    )
}

fn to_input(messages: &[Message]) -> Vec<Value> {
    let mut input = Vec::new();
    for message in messages {
        if message.role == Role::Assistant
            && let Some(items) = message.provider_replay("openai-responses")
        {
            input.extend_from_slice(items);
            continue;
        }
        match message.role {
            Role::System => {
                if !message.text().is_empty() {
                    input.push(json!({"role": "developer", "content": message.text()}));
                }
            }
            Role::User => {
                let parts: Vec<_> = message.content.iter().filter_map(|content| match content {
                    Content::Text(text) => Some(json!({"type": "input_text", "text": text})),
                    Content::Image {data, media_type} => Some(json!({
                        "type": "input_image", "image_url": format!("data:{media_type};base64,{data}"),
                    })),
                    _ => None,
                }).collect();
                input.push(json!({"role": "user", "content": parts}));
            }
            Role::Assistant => {
                for content in &message.content {
                    match content {
                        Content::Text(text) if !text.is_empty() => input.push(json!({
                            "role": "assistant", "content": text,
                        })),
                        Content::ToolCall {id, name, arguments} => input.push(json!({
                            "type": "function_call", "call_id": id, "name": name, "arguments": arguments,
                        })),
                        // Provider-specific signatures are not transferable.
                        _ => {},
                    }
                }
            }
            Role::Tool => {
                for content in &message.content {
                    if let Content::ToolResult { call_id, output } = content {
                        input.push(json!({"type": "function_call_output", "call_id": call_id, "output": output}));
                    }
                }
            }
        }
    }
    input
}

pub(super) fn wire_audit(
    request: &ChatRequest,
    route: &str,
    body: &Value,
    accepted: bool,
    response_status: Option<u16>,
) -> WireAudit {
    WireAudit {
        provider: "openai".into(),
        route: crate::endpoint_capability_route("openai", route),
        model: request.model.clone(),
        output_token_parameter: "max_output_tokens".into(),
        max_output_tokens: request.max_tokens,
        temperature: None,
        top_p: None,
        reasoning_request: body
            .pointer("/reasoning/effort")
            .and_then(Value::as_str)
            .map(str::to_string),
        reasoning_replay: request
            .messages
            .iter()
            .any(|message| {
                message
                    .provider_replay("openai-responses")
                    .is_some_and(|items| items.iter().any(|item| item["type"] == "reasoning"))
            })
            .then(|| "encrypted_reasoning".into()),
        native_tools_enabled: body.get("tools").is_some(),
        tool_count: body
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        strict_schema: false,
        tool_choice: body
            .get("tool_choice")
            .and_then(Value::as_str)
            .map(str::to_string),
        request_attempt: request.retry_attempt.saturating_add(1),
        compatibility_fallback: None,
        accepted,
        response_status,
        request_body: Some(body.clone()),
        tool_envelope_digest: request
            .tool_envelope
            .as_ref()
            .map(|envelope| envelope.digest.clone()),
        tool_envelope: request
            .tool_envelope
            .as_ref()
            .map(|envelope| envelope.payload.clone()),
        tool_schema: crate::wire_audit::tool_schema_audit(
            (!request.tools.is_empty()).then(|| tool_definitions(request)),
            body.get("tools"),
            "openai_responses_v1",
        ),
    }
}
