use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock},
};

use serde_json::Value;

use crate::{Completion, Content, ProviderError, ProviderErrorKind, ToolMode, ToolSpec};

const MAX_TOOL_CALLS: usize = 128;
const MAX_TOOL_ARGUMENT_BYTES: usize = 4 * 1024 * 1024;
const MAX_TOTAL_TOOL_ARGUMENT_BYTES: usize = 8 * 1024 * 1024;

static VALIDATORS: OnceLock<Mutex<HashMap<String, Arc<jsonschema::Validator>>>> = OnceLock::new();

fn validators() -> &'static Mutex<HashMap<String, Arc<jsonschema::Validator>>> {
    VALIDATORS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Enforce the schema contract at the client-executor boundary. Routed APIs
/// intentionally treat public schemas as shadow-only, so invalid arguments
/// must be rejected here before any workspace tool can run.
pub fn validate_client_tool_calls(
    completion: &Completion,
    tools: &[ToolSpec],
    tool_mode: ToolMode,
) -> Result<(), ProviderError> {
    let calls = completion
        .content
        .iter()
        .filter_map(|block| match block {
            Content::ToolCall {
                id,
                name,
                arguments,
            } => Some((id, name, arguments)),
            _ => None,
        })
        .collect::<Vec<_>>();

    if calls.is_empty() {
        if tool_mode == ToolMode::Required {
            return Err(tool_protocol_error(
                "model did not emit a tool call when tool_choice was required",
            ));
        }
        return Ok(());
    }
    if tool_mode == ToolMode::ChatOnly || tools.is_empty() {
        return Err(tool_protocol_error(
            "model emitted tool calls when tools were disabled",
        ));
    }
    if calls.len() > MAX_TOOL_CALLS {
        return Err(tool_protocol_error(
            "model exceeded the client tool-call count limit",
        ));
    }

    validate_client_tool_batch_limits(calls.iter().map(|(_, _, arguments)| arguments.as_str()))?;

    let mut ids = HashSet::new();
    for (id, name, arguments) in calls {
        if !ids.insert(id.as_str()) {
            return Err(tool_protocol_error(
                "model emitted an invalid or duplicate tool-call id",
            ));
        }
        validate_client_tool_call(id, name, arguments, tools)?;
    }
    Ok(())
}

/// Enforce aggregate count/size limits before any call in a batch executes.
pub fn validate_client_tool_batch_limits<'a>(
    arguments: impl IntoIterator<Item = &'a str>,
) -> Result<(), ProviderError> {
    let mut count = 0usize;
    let mut total_bytes = 0usize;
    for argument in arguments {
        count = count.saturating_add(1);
        total_bytes = total_bytes.saturating_add(argument.len());
        if count > MAX_TOOL_CALLS {
            return Err(tool_protocol_error(
                "model exceeded the client tool-call count limit",
            ));
        }
        if total_bytes > MAX_TOTAL_TOOL_ARGUMENT_BYTES {
            return Err(tool_protocol_error(
                "model exceeded the total client tool-argument size limit",
            ));
        }
    }
    Ok(())
}

/// Validate one call immediately before the local executor receives it.
pub fn validate_client_tool_call(
    id: &str,
    name: &str,
    arguments: &str,
    tools: &[ToolSpec],
) -> Result<(), ProviderError> {
    if !valid_tool_call_id(id) {
        return Err(tool_protocol_error("model emitted an invalid tool-call id"));
    }
    let Some(tool) = tools.iter().find(|tool| tool.name == name) else {
        return Err(tool_protocol_error("model emitted an unknown tool name"));
    };
    if arguments.len() > MAX_TOOL_ARGUMENT_BYTES {
        return Err(tool_protocol_error(
            "model exceeded the client tool-argument size limit",
        ));
    }
    let value = serde_json::from_str::<Value>(arguments)
        .map_err(|_| tool_protocol_error("invalid tool arguments: incomplete JSON object"))?;
    if !value.is_object() {
        return Err(tool_protocol_error(
            "model tool arguments were not a JSON object",
        ));
    }
    validate_schema(&tool.parameters, &value)
}

fn valid_tool_call_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
}

fn validate_schema(schema: &Value, value: &Value) -> Result<(), ProviderError> {
    let key = blake3::hash(&serde_json::to_vec(schema).unwrap_or_default())
        .to_hex()
        .to_string();
    let validator = {
        let mut cache = validators()
            .lock()
            .map_err(|_| tool_protocol_error("client tool validator is unavailable"))?;
        if let Some(validator) = cache.get(&key) {
            Arc::clone(validator)
        } else {
            // Built without file/HTTP resolver features: local JSON Pointer
            // references work, while remote schema retrieval is impossible.
            let validator = Arc::new(
                jsonschema::draft202012::options()
                    .build(schema)
                    .map_err(|_| tool_protocol_error("client tool schema is invalid"))?,
            );
            cache.insert(key, Arc::clone(&validator));
            validator
        }
    };
    if validator.is_valid(value) {
        Ok(())
    } else {
        Err(tool_protocol_error(
            "invalid tool arguments: did not match the declared schema",
        ))
    }
}

fn tool_protocol_error(message: &str) -> ProviderError {
    ProviderError::new(ProviderErrorKind::ToolProtocol, message).with_api_contract(
        Some("tool_protocol_error".to_string()),
        Some(true),
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Completion, Content, ToolSpec};
    use serde_json::json;

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "read".to_string(),
            description: "read a path".to_string(),
            parameters: json!({
                "$defs": {"path": {"type": "string", "minLength": 1}},
                "type": "object",
                "properties": {"path": {"$ref": "#/$defs/path"}},
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    fn completion(arguments: &str) -> Completion {
        Completion {
            content: vec![Content::ToolCall {
                id: "call_valid_1".to_string(),
                name: "read".to_string(),
                arguments: arguments.to_string(),
            }],
            ..Completion::default()
        }
    }

    #[test]
    fn validates_local_refs_and_rejects_schema_mismatch() {
        assert!(
            validate_client_tool_calls(
                &completion(r#"{"path":"README.md"}"#),
                &[tool()],
                ToolMode::Auto,
            )
            .is_ok()
        );
        assert!(
            validate_client_tool_calls(&completion(r#"{"path":7}"#), &[tool()], ToolMode::Auto,)
                .is_err()
        );
    }

    #[test]
    fn rejects_oversized_batches_before_execution() {
        let arguments = vec!["{}"; MAX_TOOL_CALLS + 1];
        assert!(validate_client_tool_batch_limits(arguments).is_err());
    }
}
