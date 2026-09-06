use serde_json::{Value, json};

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
    let tool_schema = crate::wire_audit::tool_schema_audit(
        request_tool_definitions(request),
        body.get("tools"),
        "anthropic_cache_control_v1",
    );
    WireAudit {
        provider: "anthropic".to_string(),
        route: crate::endpoint_capability_route("anthropic", route),
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
        tool_schema,
    }
}

/// Shape the locally validated specs like Anthropic definitions before the
/// adapter adds provider-only cache metadata. This is evidence only; the
/// original request schemas remain the execution-validation authority.
fn request_tool_definitions(request: &ChatRequest) -> Option<Value> {
    if request.tools.is_empty() {
        return None;
    }
    Some(Value::Array(
        request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                })
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, RequestProfile, ToolMode, ToolSpec};

    fn request(tool_mode: ToolMode) -> ChatRequest {
        ChatRequest {
            model: "claude-test".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![ToolSpec {
                name: "read".into(),
                description: "Read a workspace file".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }),
            }]
            .into(),
            tool_envelope: None,
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile {
                tool_mode,
                ..RequestProfile::default()
            },
        }
    }

    #[test]
    fn route_does_not_retain_endpoint_credentials() {
        let request = ChatRequest {
            model: "claude-test".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: false,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![].into(),
            tool_envelope: None,
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile::default(),
        };
        let route = "https://wire-user:wire-pass@example.invalid/v1?api_key=query-secret";
        let audit = build(&request, route, &serde_json::json!({}), true, None);
        assert_eq!(
            audit.route,
            crate::endpoint_capability_route("anthropic", route)
        );
        assert!(!audit.route.contains("wire-pass"));
        assert!(!audit.route.contains("query-secret"));
    }

    #[test]
    fn exact_wire_identity_records_anthropic_cache_metadata() {
        let request = request(ToolMode::Auto);
        let body = super::super::build_body(&request);
        let audit = build(&request, "https://example.invalid", &body, true, Some(200));
        let schema = audit.tool_schema.as_ref().unwrap();

        assert_ne!(schema.request_digest, schema.wire_digest);
        assert_eq!(
            schema.transform.as_deref(),
            Some("anthropic_cache_control_v1")
        );
        assert_eq!(
            schema.wire_digest,
            Some(crate::wire_audit::canonical_value_digest(&body["tools"]))
        );
        assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
        let mut redacted = serde_json::to_value(&audit).unwrap();
        redacted.as_object_mut().unwrap().remove("request_body");
        assert_eq!(
            redacted["tool_schema"]["wire_digest"],
            schema.wire_digest.as_deref().unwrap()
        );
    }

    #[test]
    fn chat_only_keeps_audited_tools_and_omission_is_visible() {
        let request = request(ToolMode::ChatOnly);
        let mut body = super::super::build_body(&request);
        let audit = build(&request, "https://example.invalid", &body, true, Some(200));

        assert_eq!(body["tool_choice"], json!({"type": "none"}));
        assert!(audit.native_tools_enabled);
        assert!(audit.tool_schema.as_ref().unwrap().wire_digest.is_some());

        body.as_object_mut().unwrap().remove("tools");
        let omitted = build(&request, "https://example.invalid", &body, false, Some(400));
        let schema = omitted.tool_schema.as_ref().unwrap();
        assert!(schema.request_digest.is_some());
        assert_eq!(schema.wire_digest, None);
        assert_eq!(schema.transform.as_deref(), Some("tools_omitted_v1"));
    }
}
