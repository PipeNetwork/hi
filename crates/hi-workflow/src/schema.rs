//! Minimal JSON-Schema validation for workflow agent output.
//!
//! Hosts use this to check a structured reply against the `output_schema` an
//! `agent()` call declared, before the engine sees it — a mismatch earns one
//! corrective retry instead of silently degrading to "unusable output".
//! Deliberately a subset: `type`, `properties`, `required`, `items`,
//! `maxItems`, `enum`. Unknown keywords are ignored rather than rejected, so
//! scripts can carry richer schemas for the model without the host refusing
//! them.

use serde_json::Value;

/// Validate `value` against `schema`. `Err` carries a short, model-actionable
/// path + reason (e.g. `claims[2].confidence: expected one of ["high", …]`).
pub fn validate_output_schema(value: &Value, schema: &Value) -> Result<(), String> {
    validate_at(value, schema, "$")
}

fn validate_at(value: &Value, schema: &Value, path: &str) -> Result<(), String> {
    let Some(schema) = schema.as_object() else {
        return Ok(());
    };

    if let Some(expected) = schema.get("type").and_then(Value::as_str) {
        let matches = match expected {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "boolean" => value.is_boolean(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "null" => value.is_null(),
            _ => true,
        };
        if !matches {
            return Err(format!("{path}: expected {expected}, got {}", type_name(value)));
        }
    }

    if let Some(allowed) = schema.get("enum").and_then(Value::as_array)
        && !allowed.contains(value)
    {
        return Err(format!(
            "{path}: expected one of {}, got {value}",
            serde_json::to_string(allowed).unwrap_or_default()
        ));
    }

    if let Some(object) = value.as_object() {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(key) {
                    return Err(format!("{path}: missing required field \"{key}\""));
                }
            }
        }
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            for (key, subschema) in properties {
                if let Some(field) = object.get(key) {
                    validate_at(field, subschema, &format!("{path}.{key}"))?;
                }
            }
        }
    }

    if let Some(array) = value.as_array() {
        if let Some(max) = schema.get("maxItems").and_then(Value::as_u64)
            && array.len() as u64 > max
        {
            return Err(format!(
                "{path}: {} items exceeds maxItems {max}",
                array.len()
            ));
        }
        if let Some(items) = schema.get("items") {
            for (index, item) in array.iter().enumerate() {
                validate_at(item, items, &format!("{path}[{index}]"))?;
            }
        }
    }

    Ok(())
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn claims_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": { "type": "string" },
                            "confidence": { "type": "string", "enum": ["high", "low"] },
                        },
                        "required": ["claim", "confidence"],
                    },
                },
            },
            "required": ["claims"],
        })
    }

    #[test]
    fn conforming_value_passes() {
        let value = json!({"claims": [{"claim": "x", "confidence": "high"}]});
        assert_eq!(validate_output_schema(&value, &claims_schema()), Ok(()));
    }

    #[test]
    fn violations_name_the_path_and_reason() {
        let schema = claims_schema();
        let missing = json!({"other": 1});
        assert!(validate_output_schema(&missing, &schema)
            .unwrap_err()
            .contains("missing required field \"claims\""));
        let wrong_type = json!({"claims": "not an array"});
        assert!(validate_output_schema(&wrong_type, &schema)
            .unwrap_err()
            .contains("$.claims: expected array"));
        let bad_enum = json!({"claims": [{"claim": "x", "confidence": "medium"}]});
        let error = validate_output_schema(&bad_enum, &schema).unwrap_err();
        assert!(error.contains("$.claims[0].confidence"), "{error}");
        let too_many = json!({"claims": [
            {"claim": "a", "confidence": "high"},
            {"claim": "b", "confidence": "high"},
            {"claim": "c", "confidence": "high"},
        ]});
        assert!(validate_output_schema(&too_many, &schema)
            .unwrap_err()
            .contains("exceeds maxItems 2"));
    }

    #[test]
    fn unknown_keywords_and_non_object_schemas_are_ignored() {
        assert_eq!(
            validate_output_schema(&json!({"x": 1}), &json!({"weird": true})),
            Ok(())
        );
        assert_eq!(validate_output_schema(&json!(1), &json!("string-schema")), Ok(()));
    }
}
