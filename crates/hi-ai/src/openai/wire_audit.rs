//! Audit identity for concrete OpenAI-compatible request shapes.
//!
//! The agent seals the original tool schemas because those remain the local
//! execution-validation authority. OpenAI-compatible adapters may then apply
//! bounded schema normalization for a particular wire attempt. Both identities
//! are retained so a redacted trace can still prove which contract the model
//! actually saw without pretending the provider transform changed admission.

use serde_json::{Value, json};

use super::request::RequestAttempt;
use crate::wire_audit::canonical_value_digest;
use crate::{ChatRequest, Content, WireAudit, WireToolSchemaAudit};

pub(super) fn build(
    request: &ChatRequest,
    route: &str,
    attempt: RequestAttempt,
    index: usize,
    body: &Value,
    accepted: bool,
    response_status: Option<u16>,
) -> WireAudit {
    let request_digest = request_tool_definitions(request)
        .as_ref()
        .map(canonical_value_digest);
    let wire_digest = body
        .get("tools")
        .filter(|tools| tools.is_array())
        .map(canonical_value_digest);
    let transform = schema_transform(request_digest.as_deref(), wire_digest.as_deref(), attempt);
    let tool_schema =
        (request_digest.is_some() || wire_digest.is_some()).then_some(WireToolSchemaAudit {
            request_digest,
            wire_digest,
            transform,
        });
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
        route: crate::endpoint_capability_route("openai_compatible", route),
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
        // `ChatRequest::retry_attempt` is zero-based; the audit is a
        // human-facing one-based ordinal for the logical request replay. A
        // compatibility-shape fallback is still the same logical replay.
        request_attempt: request.retry_attempt.saturating_add(1),
        compatibility_fallback: compatibility_fallback(attempt, index),
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

/// Shape the locally validated specs exactly like an unmodified OpenAI tools
/// array. This lets its digest be compared directly with `body["tools"]`.
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
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect(),
    ))
}

fn schema_transform(
    request_digest: Option<&str>,
    wire_digest: Option<&str>,
    attempt: RequestAttempt,
) -> Option<String> {
    match (request_digest, wire_digest) {
        (None, None) => None,
        (Some(request), Some(wire)) if request == wire => None,
        (_, Some(_)) if attempt.strict_tools => Some("deepseek_strict_v1".to_string()),
        (Some(_), Some(_)) => Some("openai_compat_v1".to_string()),
        (Some(_), None) => Some("tools_omitted_v1".to_string()),
        (None, Some(_)) => Some("provider_injected_tools_v1".to_string()),
    }
}

fn compatibility_fallback(attempt: RequestAttempt, index: usize) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::openai::deepseek::ProviderCapabilities;
    use crate::openai::request::{build_body_with_capabilities, request_attempts_for};
    use crate::{DeepSeekCompat, Message, RequestProfile, ToolSpec};

    #[test]
    fn unchanged_schema_has_one_identity_and_no_transform() {
        let request = request_with_schema(json!({
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"]
        }));
        let capabilities = ProviderCapabilities::generic();
        let attempt = request_attempts_for(&request, &capabilities)[0];
        let body = build_body_with_capabilities(&request, attempt, None, &capabilities);
        let audit = build(
            &request,
            "https://example.invalid/v1",
            attempt,
            0,
            &body,
            true,
            Some(200),
        );

        let schema = audit.tool_schema.as_ref().unwrap();
        assert_eq!(schema.request_digest, schema.wire_digest);
        assert_eq!(schema.transform, None);
        assert_eq!(
            schema.wire_digest,
            Some(canonical_value_digest(&body["tools"]))
        );
        let mut redacted = serde_json::to_value(&audit).unwrap();
        redacted.as_object_mut().unwrap().remove("request_body");
        assert_eq!(
            redacted["tool_schema"]["wire_digest"],
            schema.wire_digest.as_deref().unwrap()
        );
    }

    #[test]
    fn openai_compat_normalization_gets_a_distinct_wire_identity() {
        let request = request_with_schema(json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}}
            },
            "oneOf": [
                {"required": ["path"]},
                {"required": ["paths"]}
            ]
        }));
        let capabilities = ProviderCapabilities::generic();
        let attempt = request_attempts_for(&request, &capabilities)[0];
        let body = build_body_with_capabilities(&request, attempt, None, &capabilities);
        let audit = build(
            &request,
            "https://example.invalid/v1",
            attempt,
            0,
            &body,
            true,
            Some(200),
        );

        let schema = audit.tool_schema.as_ref().unwrap();
        assert_ne!(schema.request_digest, schema.wire_digest);
        assert_eq!(schema.transform.as_deref(), Some("openai_compat_v1"));
        assert!(
            body["tools"][0]["function"]["parameters"]
                .get("oneOf")
                .is_none()
        );
        assert_eq!(
            schema.wire_digest,
            Some(canonical_value_digest(&body["tools"]))
        );
    }

    #[test]
    fn deepseek_strict_and_non_strict_retries_have_exact_distinct_identities() {
        let mut request = request_with_schema(json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"}
            },
            "required": ["path"]
        }));
        request.model = "deepseek-v4-flash".to_string();
        request.profile.deepseek_compat = DeepSeekCompat::On;
        let capabilities = ProviderCapabilities::detect(
            "https://api.deepseek.com",
            &request.model,
            request.profile.deepseek_compat,
        );
        let attempts = request_attempts_for(&request, &capabilities);
        let strict_index = attempts
            .iter()
            .position(|attempt| attempt.strict_tools && !attempt.strict_fallback)
            .unwrap();
        let fallback_index = attempts
            .iter()
            .position(|attempt| attempt.strict_fallback)
            .unwrap();
        let strict_body =
            build_body_with_capabilities(&request, attempts[strict_index], None, &capabilities);
        let fallback_body =
            build_body_with_capabilities(&request, attempts[fallback_index], None, &capabilities);
        let strict = build(
            &request,
            "https://api.deepseek.com",
            attempts[strict_index],
            strict_index,
            &strict_body,
            false,
            Some(400),
        );
        let fallback = build(
            &request,
            "https://api.deepseek.com",
            attempts[fallback_index],
            fallback_index,
            &fallback_body,
            true,
            Some(200),
        );

        let strict_schema = strict.tool_schema.as_ref().unwrap();
        let fallback_schema = fallback.tool_schema.as_ref().unwrap();
        assert_eq!(strict_schema.request_digest, fallback_schema.request_digest);
        assert_ne!(strict_schema.wire_digest, fallback_schema.wire_digest);
        assert_eq!(
            strict_schema.transform.as_deref(),
            Some("deepseek_strict_v1")
        );
        assert_eq!(
            fallback.compatibility_fallback.as_deref(),
            Some("strict_schema")
        );
        assert_eq!(fallback_schema.transform, None);
        assert_eq!(
            strict_schema.wire_digest,
            Some(canonical_value_digest(&strict_body["tools"]))
        );
        assert_eq!(
            fallback_schema.wire_digest,
            Some(canonical_value_digest(&fallback_body["tools"]))
        );
    }

    fn request_with_schema(parameters: Value) -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            request_id: None,
            retry_attempt: 0,
            user_turn: true,
            canonical_objective: None,
            messages: vec![Message::user("hi")].into(),
            tools: vec![ToolSpec {
                name: "read".into(),
                description: "Read workspace files".into(),
                parameters,
            }]
            .into(),
            tool_envelope: None,
            max_tokens: 32,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            reasoning_effort: None,
            profile: RequestProfile::default(),
        }
    }
}
