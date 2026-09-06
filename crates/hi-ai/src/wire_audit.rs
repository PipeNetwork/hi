//! Provider-wire audit primitives shared by adapter-specific request builders.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The immutable local schema identity and the exact schema identity sent on
/// one provider attempt. Digests cover the ordered full tool-definition array.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireToolSchemaAudit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wire_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<String>,
}

pub(crate) fn canonical_value_digest(value: &Value) -> String {
    let canonical = canonicalize(value.clone());
    format!(
        "blake3:{}",
        blake3::hash(&serde_json::to_vec(&canonical).expect("canonical JSON serializes")).to_hex()
    )
}

/// Compare the provider-native definitions derived from the sealed request
/// with the exact ordered definitions placed on the wire. The adapter-specific
/// label describes a bounded rewrite; omission and injection retain common
/// labels so audits can be compared across providers.
pub(crate) fn tool_schema_audit(
    request_definitions: Option<Value>,
    wire_definitions: Option<&Value>,
    adapter_transform: &'static str,
) -> Option<WireToolSchemaAudit> {
    let request_digest = request_definitions.as_ref().map(canonical_value_digest);
    let wire_digest = wire_definitions
        .filter(|definitions| definitions.is_array())
        .map(canonical_value_digest);
    let transform = match (request_digest.as_deref(), wire_digest.as_deref()) {
        (None, None) => None,
        (Some(request), Some(wire)) if request == wire => None,
        (Some(_), Some(_)) => Some(adapter_transform.to_string()),
        (Some(_), None) => Some("tools_omitted_v1".to_string()),
        (None, Some(_)) => Some("provider_injected_tools_v1".to_string()),
    };

    (request_digest.is_some() || wire_digest.is_some()).then_some(WireToolSchemaAudit {
        request_digest,
        wire_digest,
        transform,
    })
}

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize).collect()),
        Value::Object(values) => {
            let mut fields = values.into_iter().collect::<Vec<_>>();
            fields.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            Value::Object(Map::from_iter(
                fields
                    .into_iter()
                    .map(|(key, value)| (key, canonicalize(value))),
            ))
        }
        scalar => scalar,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn canonical_digest_sorts_object_keys_but_preserves_tool_order() {
        let left = json!([{"name": "a", "schema": {"z": 1, "a": 2}}, {"name": "b"}]);
        let same = json!([{"schema": {"a": 2, "z": 1}, "name": "a"}, {"name": "b"}]);
        let reordered = json!([{"name": "b"}, {"name": "a", "schema": {"a": 2, "z": 1}}]);

        assert_eq!(canonical_value_digest(&left), canonical_value_digest(&same));
        assert_ne!(
            canonical_value_digest(&left),
            canonical_value_digest(&reordered)
        );
    }

    #[test]
    fn tool_schema_audit_distinguishes_rewrite_omission_and_injection() {
        let request = json!([{"name": "read", "schema": {"type": "object"}}]);
        let wire = json!([{"name": "read", "schema": {"type": "object"}, "strict": true}]);

        let rewritten = tool_schema_audit(Some(request.clone()), Some(&wire), "native_v1").unwrap();
        assert_ne!(rewritten.request_digest, rewritten.wire_digest);
        assert_eq!(rewritten.transform.as_deref(), Some("native_v1"));

        let omitted = tool_schema_audit(Some(request), None, "native_v1").unwrap();
        assert!(omitted.request_digest.is_some());
        assert_eq!(omitted.wire_digest, None);
        assert_eq!(omitted.transform.as_deref(), Some("tools_omitted_v1"));

        let injected = tool_schema_audit(None, Some(&wire), "native_v1").unwrap();
        assert_eq!(injected.request_digest, None);
        assert!(injected.wire_digest.is_some());
        assert_eq!(
            injected.transform.as_deref(),
            Some("provider_injected_tools_v1")
        );
    }
}
