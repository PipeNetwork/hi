use serde_json::Value;

use crate::{ChatRequest, Content, WireAudit};

pub(super) fn emit(
    sink: &mut (dyn FnMut(crate::StreamEvent) + Send),
    request: &ChatRequest,
    route: &str,
    body: &Value,
    status: reqwest::StatusCode,
) {
    sink(crate::StreamEvent::WireAudit(Box::new(build(
        request,
        route,
        body,
        status.is_success(),
        Some(status.as_u16()),
    ))));
}

pub(super) fn build(
    request: &ChatRequest,
    route: &str,
    body: &Value,
    accepted: bool,
    response_status: Option<u16>,
) -> WireAudit {
    WireAudit {
        provider: "anthropic".to_string(),
        route: route.to_string(),
        model: request.model.clone(),
        output_token_parameter: "max_tokens".to_string(),
        max_output_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        reasoning_request: request
            .thinking_budget
            .map(|_| "thinking_budget".to_string()),
        reasoning_replay: request
            .messages
            .iter()
            .flat_map(|message| message.content.iter())
            .find_map(|content| match content {
                Content::Thinking {
                    signature: Some(_), ..
                } => Some("signed_thinking".to_string()),
                Content::Thinking { .. } => Some("thinking_blocks".to_string()),
                _ => None,
            }),
        native_tools_enabled: body.get("tools").and_then(Value::as_array).is_some(),
        tool_count: body
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        strict_schema: false,
        tool_choice: body
            .get("tool_choice")
            .and_then(|choice| choice.get("type"))
            .and_then(Value::as_str)
            .map(str::to_string),
        request_attempt: request.retry_attempt.saturating_add(1),
        compatibility_fallback: None,
        accepted,
        request_body: Some(body.clone()),
        response_status,
        tool_envelope_digest: request
            .tool_envelope
            .as_ref()
            .map(|envelope| envelope.digest.clone()),
        tool_envelope: request
            .tool_envelope
            .as_ref()
            .map(|envelope| envelope.payload.clone()),
    }
}
