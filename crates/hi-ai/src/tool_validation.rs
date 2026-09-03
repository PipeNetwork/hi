#![allow(
    clippy::result_large_err,
    reason = "ProviderError intentionally preserves structured provider and API context by value"
)]

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, OnceLock},
};

use serde_json::Value;

use crate::{Completion, Content, ProviderError, ProviderErrorKind, ToolMode, ToolSpec};

const MAX_TOOL_ARGUMENT_BYTES: usize = 4 * 1024 * 1024;
/// Aggregate retained payload budget for one model-emitted tool batch. This is
/// a memory/resource guard, not a call-count ceiling: every call consumes a
/// conservative slot charge plus its encoded fields, so small valid batches
/// may contain well over the historical 128-call limit while empty-call floods
/// still cannot grow without bound.
pub(crate) const MAX_TOTAL_TOOL_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const TOOL_CALL_SLOT_OVERHEAD_BYTES: usize = 128;

/// Reserve retained memory against the aggregate per-response tool budget.
/// Checked arithmetic makes overflow a refusal rather than accidentally
/// turning saturation into permission.
pub(crate) fn try_reserve_tool_payload(total: &mut usize, additional: usize) -> bool {
    let Some(next) = total.checked_add(additional) else {
        return false;
    };
    if next > MAX_TOTAL_TOOL_PAYLOAD_BYTES {
        return false;
    }
    *total = next;
    true
}

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

/// Enforce an aggregate retained-memory limit before any call in a batch
/// executes. There is deliberately no independent call-count ceiling.
pub fn validate_client_tool_batch_limits<'a>(
    arguments: impl IntoIterator<Item = &'a str>,
) -> Result<(), ProviderError> {
    let mut total_bytes = 0usize;
    for argument in arguments {
        if !try_reserve_tool_payload(
            &mut total_bytes,
            TOOL_CALL_SLOT_OVERHEAD_BYTES.saturating_add(argument.len()),
        ) {
            return Err(tool_protocol_error(
                "model exceeded the total client tool payload size limit",
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
    let value = normalize_optional_nulls(&tool.parameters, &tool.parameters, &value);
    validate_schema(&tool.parameters, &value)
}

/// DeepSeek strict schemas encode formerly optional properties as nullable
/// required fields. Convert a model-emitted null back to omission before the
/// original client schema validates the call.
fn normalize_optional_nulls(root_schema: &Value, schema: &Value, value: &Value) -> Value {
    let schema = resolve_local_schema(root_schema, schema);
    match value {
        Value::Array(items) => {
            let Some(item_schema) = schema.get("items") else {
                return value.clone();
            };
            Value::Array(
                items
                    .iter()
                    .map(|item| normalize_optional_nulls(root_schema, item_schema, item))
                    .collect(),
            )
        }
        Value::Object(object) => {
            let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
                return value.clone();
            };
            let required: HashSet<&str> = schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let mut normalized = serde_json::Map::new();
            for (name, child) in object {
                if child.is_null() && !required.contains(name.as_str()) {
                    continue;
                }
                let child_schema = properties.get(name);
                normalized.insert(
                    name.clone(),
                    child_schema
                        .map(|child_schema| {
                            normalize_optional_nulls(root_schema, child_schema, child)
                        })
                        .unwrap_or_else(|| child.clone()),
                );
            }
            Value::Object(normalized)
        }
        _ => value.clone(),
    }
}

fn resolve_local_schema<'a>(root_schema: &'a Value, schema: &'a Value) -> &'a Value {
    let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
        return schema;
    };
    let Some(def_name) = reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))
    else {
        return schema;
    };
    root_schema
        .get("$defs")
        .or_else(|| root_schema.get("definitions"))
        .and_then(Value::as_object)
        .and_then(|defs| defs.get(def_name))
        .unwrap_or(schema)
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
    match validator.validate(value) {
        Ok(()) => Ok(()),
        Err(error) => {
            let mut detail = error.masked().to_string();
            let instance_path = error.instance_path().as_str();
            if !instance_path.is_empty() {
                detail.push_str(&format!(" at `{instance_path}`"));
            }
            if let Some(hint) = schema_requirement_hint(schema, value) {
                detail.push_str("; ");
                detail.push_str(&hint);
            }
            Err(tool_protocol_error(&format!(
                "invalid tool arguments: {detail}"
            )))
        }
    }
}

/// Add a stable, value-free hint for the common required/oneOf failures. The
/// validator's top-level message for an empty object against a schema such as
/// `oneOf: [{required: [path]}, {required: [paths]}]` only says "oneOf", which
/// is not enough for a model to repair its next call. Keep this derived from
/// schema keys rather than echoing argument values into the transcript.
fn schema_requirement_hint(schema: &Value, value: &Value) -> Option<String> {
    let schema = resolve_local_schema(schema, schema);
    let object = value.as_object()?;

    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        let missing = required
            .iter()
            .filter_map(Value::as_str)
            .filter(|name| !object.contains_key(*name))
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Some(format!(
                "missing required propert{}: {}",
                if missing.len() == 1 { "y" } else { "ies" },
                missing
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }

    for keyword in ["oneOf", "anyOf"] {
        let Some(alternatives) = schema.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        let missing_alternatives = alternatives
            .iter()
            .filter_map(|alternative| alternative.get("required"))
            .filter_map(Value::as_array)
            .map(|required| {
                required
                    .iter()
                    .filter_map(Value::as_str)
                    .filter(|name| !object.contains_key(*name))
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
            })
            .filter(|missing| !missing.is_empty())
            .map(|missing| missing.join(" and "))
            .collect::<Vec<_>>();
        if !missing_alternatives.is_empty() {
            return Some(format!(
                "provide one of: {}",
                missing_alternatives.join("; ")
            ));
        }
    }

    None
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
    fn validation_error_names_the_missing_required_property() {
        let error = validate_client_tool_calls(&completion("{}"), &[tool()], ToolMode::Auto)
            .expect_err("missing path must be rejected");
        assert!(error.to_string().contains("path"), "{error}");
    }

    #[test]
    fn accepts_batches_beyond_the_legacy_count_ceiling() {
        let arguments = vec!["{}"; 175];
        assert!(validate_client_tool_batch_limits(arguments).is_ok());

        let completion = Completion {
            content: (0..175)
                .map(|index| Content::ToolCall {
                    id: format!("call_{index}"),
                    name: "read".to_string(),
                    arguments: r#"{"path":"README.md"}"#.to_string(),
                })
                .collect(),
            ..Completion::default()
        };
        assert!(validate_client_tool_calls(&completion, &[tool()], ToolMode::Auto).is_ok());
    }

    #[test]
    fn rejects_aggregate_payload_overflow_before_execution() {
        let payload = "x".repeat(MAX_TOTAL_TOOL_PAYLOAD_BYTES / 2);
        assert!(
            validate_client_tool_batch_limits([payload.as_str(), payload.as_str(), "{}"]).is_err()
        );
    }

    #[test]
    fn optional_nulls_are_normalized_back_to_omitted_properties() {
        let optional = ToolSpec {
            name: "read".to_string(),
            description: "read".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"}
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        };
        let completion = Completion {
            content: vec![Content::ToolCall {
                id: "call_optional_null".to_string(),
                name: "read".to_string(),
                arguments: r#"{"path":"README.md","line":null}"#.to_string(),
            }],
            ..Completion::default()
        };
        assert!(validate_client_tool_calls(&completion, &[optional], ToolMode::Auto).is_ok());
    }
}
