//! Opaque transport form of the canonical harness tool envelope.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{EffectiveProviderCapabilities, MAX_TOOL_ARGUMENT_BYTES, ToolMode};

/// Capability and workspace evidence attached to one exact provider request.
/// Construction and execution stay in the workspace/tool layers; adapters
/// only transport and audit this value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestToolEnvelope {
    pub digest: String,
    pub payload: Value,
}

/// Build a canonical no-tool child envelope for provider-internal inference
/// (for example the private MoA reference call). Workspace identity is
/// inherited when present, but no executable tools or calls are admitted.
pub(crate) fn derived_chat_only(
    parent: Option<&RequestToolEnvelope>,
    provider: EffectiveProviderCapabilities,
    max_output_tokens: u32,
    scope: &str,
) -> RequestToolEnvelope {
    let workspace = parent
        .and_then(|envelope| envelope.payload.get("workspace").cloned())
        .unwrap_or_else(|| {
            json!({
                "authority": {"kind": "local"},
                "binding_id": format!("standalone-chat-only:{scope}"),
                "epoch": 0,
                "version": {"kind": "unknown"}
            })
        });
    let trust = parent
        .and_then(|envelope| envelope.payload.get("trust").cloned())
        .unwrap_or_else(|| json!("untrusted"));
    let argument_limit = provider
        .capabilities
        .request_limits
        .max_tool_argument_bytes
        .unwrap_or(MAX_TOOL_ARGUMENT_BYTES as u32)
        .min(MAX_TOOL_ARGUMENT_BYTES as u32);
    let capability_digest = provider.canonical_digest();
    let route = provider.target.route.clone();
    let requested_model = provider.target.model.clone();
    let actual_model_revision = provider.capabilities.actual_model_revision.clone();
    let payload = json!({
        "schema_version": 3,
        "tools": [],
        "provider": {
            "route": route,
            "requested_model": requested_model,
            "actual_model_revision": actual_model_revision,
            "capability_digest": capability_digest,
            "capability_record": provider,
        },
        "workspace": workspace,
        "trust": trust,
        "permissions": [format!("derived_request:{scope}"), "tools:disabled"],
        "limits": {
            "max_output_tokens": max_output_tokens,
            "max_parallel_calls": 1,
            "max_calls_per_round": 0,
            "max_inline_output_bytes": 50_000,
            "max_tool_argument_bytes": argument_limit,
        },
        "tool_mode": ToolMode::ChatOnly,
    });
    let canonical = canonicalize(payload);
    let digest = format!(
        "blake3:{}",
        blake3::hash(&serde_json::to_vec(&canonical).expect("request envelope serializes"))
            .to_hex()
    );
    RequestToolEnvelope {
        digest,
        payload: canonical,
    }
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
