//! Stable, side-effect-free protocol types for the optional decision engine.
//!
//! The native host owns every effect. A guest only consumes bounded input
//! events and returns validated actions. Keeping these types in a small crate
//! prevents the WASM boundary from depending on the large agent crate and
//! makes replay/differential testing possible without a live terminal.

use serde::{Deserialize, Serialize};

pub const ENGINE_API_MAJOR: u16 = 1;
pub const ENGINE_API_MINOR: u16 = 0;
pub const ENGINE_STATE_SCHEMA_VERSION: u32 = 1;
pub const MAX_ENGINE_ACTIONS: usize = 64;
pub const MAX_ENGINE_PAYLOAD_BYTES: usize = 256 * 1024;
pub const MAX_ENGINE_TOOL_BATCH: usize = 32;
pub const MAX_ENGINE_TOOLS: usize = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineMode {
    #[default]
    Native,
    Wasm,
}

impl EngineMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native" | "rust" | "off" => Some(Self::Native),
            "wasm" | "webassembly" | "component" => Some(Self::Wasm),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Wasm => "wasm",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineStateSnapshot {
    pub api_major: u16,
    pub api_minor: u16,
    pub state_schema_version: u32,
    pub turn_id: String,
    pub workspace_context_generation: u64,
    pub ledger_revision: u64,
    /// Opaque host-owned state. The guest may replace it only by returning a
    /// valid state update; it never relies on persistent linear memory.
    pub state: Vec<u8>,
}

impl EngineStateSnapshot {
    pub fn new(turn_id: impl Into<String>) -> Self {
        Self {
            api_major: ENGINE_API_MAJOR,
            api_minor: ENGINE_API_MINOR,
            state_schema_version: ENGINE_STATE_SCHEMA_VERSION,
            turn_id: turn_id.into(),
            workspace_context_generation: 0,
            ledger_revision: 0,
            state: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.api_major != ENGINE_API_MAJOR {
            return Err(ProtocolError::UnsupportedApiMajor(self.api_major));
        }
        if self.state_schema_version != ENGINE_STATE_SCHEMA_VERSION {
            return Err(ProtocolError::UnsupportedStateSchema(
                self.state_schema_version,
            ));
        }
        bounded_string(&self.turn_id, "turn_id")?;
        bounded_bytes(&self.state, "state")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineInput {
    TurnStarted {
        snapshot: EngineStateSnapshot,
        prompt: String,
        tools: Vec<ToolDescriptor>,
    },
    ProviderDelta {
        request_id: String,
        text: String,
        reasoning: String,
        tool_call_deltas: Vec<ToolCallDelta>,
        done: bool,
    },
    ToolResult {
        request_id: String,
        occurrence_id: String,
        name: String,
        status: String,
        output: String,
        workspace_context_generation: u64,
        ledger_revision: u64,
    },
    ApprovalResult {
        request_id: String,
        approved: bool,
        detail: String,
    },
    Cancelled {
        reason: String,
    },
    TimedOut,
    HostError {
        code: String,
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

impl ToolCallDelta {
    fn validate(&self) -> Result<(), ProtocolError> {
        if let Some(id) = &self.id {
            bounded_string(id, "tool call id")?;
        }
        if let Some(name) = &self.name {
            bounded_string(name, "tool call name")?;
        }
        bounded_bytes(self.arguments.as_bytes(), "tool call arguments")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub parameters_json: String,
}

impl ToolDescriptor {
    fn validate(&self) -> Result<(), ProtocolError> {
        bounded_string(&self.name, "tool descriptor name")?;
        bounded_string(&self.description, "tool descriptor description")?;
        bounded_string(&self.parameters_json, "tool descriptor parameters")?;
        serde_json::from_str::<serde_json::Value>(&self.parameters_json).map_err(|error| {
            ProtocolError::InvalidJson("tool descriptor parameters", error.to_string())
        })?;
        Ok(())
    }
}

impl EngineInput {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            Self::TurnStarted {
                snapshot,
                prompt,
                tools,
            } => {
                snapshot.validate()?;
                bounded_string(prompt, "turn prompt")?;
                if tools.len() > MAX_ENGINE_TOOLS {
                    return Err(ProtocolError::InvalidBatchSize(tools.len()));
                }
                for tool in tools {
                    tool.validate()?;
                }
            }
            Self::ProviderDelta {
                request_id,
                text,
                reasoning,
                tool_call_deltas,
                ..
            } => {
                bounded_string(request_id, "provider request_id")?;
                bounded_bytes(text.as_bytes(), "provider text")?;
                bounded_bytes(reasoning.as_bytes(), "provider reasoning")?;
                if tool_call_deltas.len() > MAX_ENGINE_TOOL_BATCH {
                    return Err(ProtocolError::InvalidBatchSize(tool_call_deltas.len()));
                }
                for delta in tool_call_deltas {
                    delta.validate()?;
                }
            }
            Self::ToolResult {
                request_id,
                occurrence_id,
                name,
                status,
                output,
                ..
            } => {
                bounded_string(request_id, "tool result request_id")?;
                bounded_string(occurrence_id, "tool result occurrence_id")?;
                bounded_string(name, "tool result name")?;
                bounded_string(status, "tool result status")?;
                bounded_bytes(output.as_bytes(), "tool result output")?;
            }
            Self::ApprovalResult {
                request_id, detail, ..
            } => {
                bounded_string(request_id, "approval request_id")?;
                bounded_bytes(detail.as_bytes(), "approval detail")?;
            }
            Self::Cancelled { reason } => bounded_string(reason, "cancellation reason")?,
            Self::TimedOut => {}
            Self::HostError { code, message } => {
                bounded_string(code, "host error code")?;
                bounded_string(message, "host error message")?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EngineAction {
    RequestModel {
        idempotency_key: String,
        request_id: String,
        messages_json: String,
    },
    ExecuteTool {
        request: ToolRequest,
    },
    ExecuteParallel {
        requests: Vec<ToolRequest>,
    },
    Present {
        idempotency_key: String,
        directive: PresentationDirective,
    },
    UpdateState {
        idempotency_key: String,
        state: Vec<u8>,
    },
    Wait {
        idempotency_key: String,
    },
    Complete {
        idempotency_key: String,
        result_json: String,
    },
    Fail {
        idempotency_key: String,
        code: String,
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolRequest {
    pub idempotency_key: String,
    pub request_id: String,
    pub occurrence_id: String,
    pub name: String,
    pub arguments_json: String,
}

impl ToolRequest {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        bounded_string(&self.idempotency_key, "tool idempotency_key")?;
        bounded_string(&self.request_id, "tool request_id")?;
        bounded_string(&self.occurrence_id, "tool occurrence_id")?;
        bounded_string(&self.name, "tool name")?;
        bounded_string(&self.arguments_json, "tool arguments")?;
        serde_json::from_str::<serde_json::Value>(&self.arguments_json)
            .map_err(|error| ProtocolError::InvalidJson("tool arguments", error.to_string()))?;
        Ok(())
    }

    /// Return the stable JSON representation used for action identity and
    /// replay matching. `serde_json::Map` is ordered in this crate, so object
    /// keys are normalized independent of the guest's whitespace/order.
    pub fn canonical_arguments(&self) -> Result<String, ProtocolError> {
        let value = serde_json::from_str::<serde_json::Value>(&self.arguments_json)
            .map_err(|error| ProtocolError::InvalidJson("tool arguments", error.to_string()))?;
        serde_json::to_string(&value)
            .map_err(|error| ProtocolError::InvalidJson("tool arguments", error.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PresentationDirective {
    Status { activity_id: String, text: String },
    Warning { activity_id: String, text: String },
    Activity { activity_id: String, text: String },
    ChangedFiles { files: Vec<String> },
    Completion { activity_id: String, text: String },
}

impl EngineAction {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            Self::RequestModel {
                idempotency_key,
                request_id,
                messages_json,
            } => {
                bounded_string(idempotency_key, "model idempotency_key")?;
                bounded_string(request_id, "model request_id")?;
                bounded_string(messages_json, "model messages")?;
            }
            Self::ExecuteTool { request } => request.validate()?,
            Self::ExecuteParallel { requests } => {
                if requests.is_empty() || requests.len() > MAX_ENGINE_TOOL_BATCH {
                    return Err(ProtocolError::InvalidBatchSize(requests.len()));
                }
                for request in requests {
                    request.validate()?;
                }
            }
            Self::Present {
                idempotency_key,
                directive,
            } => {
                bounded_string(idempotency_key, "presentation idempotency_key")?;
                directive.validate()?
            }
            Self::UpdateState {
                idempotency_key,
                state,
            } => {
                bounded_string(idempotency_key, "state idempotency_key")?;
                bounded_bytes(state, "state update")?
            }
            Self::Wait { idempotency_key } => {
                bounded_string(idempotency_key, "wait idempotency_key")?
            }
            Self::Complete {
                idempotency_key,
                result_json,
            } => {
                bounded_string(idempotency_key, "completion idempotency_key")?;
                bounded_string(result_json, "completion")?;
                serde_json::from_str::<serde_json::Value>(result_json)
                    .map_err(|error| ProtocolError::InvalidJson("completion", error.to_string()))?;
            }
            Self::Fail {
                idempotency_key,
                code,
                message,
            } => {
                bounded_string(idempotency_key, "failure idempotency_key")?;
                bounded_string(code, "failure code")?;
                bounded_string(message, "failure message")?;
            }
        }
        Ok(())
    }
}

impl PresentationDirective {
    fn validate(&self) -> Result<(), ProtocolError> {
        match self {
            Self::Status { activity_id, text }
            | Self::Warning { activity_id, text }
            | Self::Activity { activity_id, text }
            | Self::Completion { activity_id, text } => {
                bounded_string(activity_id, "activity_id")?;
                bounded_string(text, "presentation text")?;
            }
            Self::ChangedFiles { files } => {
                if files.len() > MAX_ENGINE_TOOL_BATCH {
                    return Err(ProtocolError::InvalidBatchSize(files.len()));
                }
                for file in files {
                    bounded_string(file, "changed file")?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineManifest {
    pub api_major: u16,
    pub api_minor: u16,
    pub guest_version: String,
    pub state_schema_version: u32,
    pub supported_features: Vec<String>,
    pub required_capabilities: Vec<String>,
    pub module_sha256: String,
    pub signature_hex: Option<String>,
    pub build_revision: Option<String>,
}

impl EngineManifest {
    pub fn unsigned(guest_version: impl Into<String>, module_sha256: impl Into<String>) -> Self {
        Self {
            api_major: ENGINE_API_MAJOR,
            api_minor: ENGINE_API_MINOR,
            guest_version: guest_version.into(),
            state_schema_version: ENGINE_STATE_SCHEMA_VERSION,
            supported_features: Vec::new(),
            required_capabilities: Vec::new(),
            module_sha256: module_sha256.into(),
            signature_hex: None,
            build_revision: None,
        }
    }

    /// The signature covers every manifest field except the signature itself.
    /// Struct field order is stable and the JSON representation is deliberately
    /// used as the small, inspectable release-envelope format.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut unsigned = self.clone();
        unsigned.signature_hex = None;
        serde_json::to_vec(&unsigned)
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        if self.api_major != ENGINE_API_MAJOR {
            return Err(ProtocolError::UnsupportedApiMajor(self.api_major));
        }
        if self.state_schema_version != ENGINE_STATE_SCHEMA_VERSION {
            return Err(ProtocolError::UnsupportedStateSchema(
                self.state_schema_version,
            ));
        }
        bounded_string(&self.guest_version, "guest version")?;
        if self.module_sha256.len() != 64
            || !self
                .module_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ProtocolError::InvalidHash);
        }
        for capability in &self.required_capabilities {
            bounded_string(capability, "required capability")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    #[error("unsupported engine API major version {0}; this binary supports {ENGINE_API_MAJOR}")]
    UnsupportedApiMajor(u16),
    #[error(
        "unsupported engine state schema version {0}; this binary supports {ENGINE_STATE_SCHEMA_VERSION}"
    )]
    UnsupportedStateSchema(u32),
    #[error("{0} exceeds the {MAX_ENGINE_PAYLOAD_BYTES}-byte engine payload limit")]
    PayloadTooLarge(&'static str),
    #[error("{0} is empty or too large")]
    InvalidString(&'static str),
    #[error("invalid JSON in {0}: {1}")]
    InvalidJson(&'static str, String),
    #[error("invalid engine action batch size {0}")]
    InvalidBatchSize(usize),
    #[error("invalid module SHA-256 hash")]
    InvalidHash,
}

fn bounded_string(value: &str, label: &'static str) -> Result<(), ProtocolError> {
    if value.is_empty() {
        return Err(ProtocolError::InvalidString(label));
    }
    if value.len() > MAX_ENGINE_PAYLOAD_BYTES {
        return Err(ProtocolError::PayloadTooLarge(label));
    }
    Ok(())
}

fn bounded_bytes(value: &[u8], label: &'static str) -> Result<(), ProtocolError> {
    if value.len() > MAX_ENGINE_PAYLOAD_BYTES {
        return Err(ProtocolError::PayloadTooLarge(label));
    }
    Ok(())
}

pub fn encode_actions(actions: &[EngineAction]) -> Result<String, ProtocolError> {
    if actions.len() > MAX_ENGINE_ACTIONS {
        return Err(ProtocolError::InvalidBatchSize(actions.len()));
    }
    for action in actions {
        action.validate()?;
    }
    let encoded = serde_json::to_string(actions)
        .map_err(|error| ProtocolError::InvalidJson("actions", error.to_string()))?;
    bounded_string(&encoded, "actions")?;
    Ok(encoded)
}

pub fn decode_actions(encoded: &str) -> Result<Vec<EngineAction>, ProtocolError> {
    bounded_string(encoded, "actions")?;
    let actions: Vec<EngineAction> = serde_json::from_str(encoded)
        .map_err(|error| ProtocolError::InvalidJson("actions", error.to_string()))?;
    if actions.len() > MAX_ENGINE_ACTIONS {
        return Err(ProtocolError::InvalidBatchSize(actions.len()));
    }
    for action in &actions {
        action.validate()?;
    }
    Ok(actions)
}

pub fn encode_input(input: &EngineInput) -> Result<String, ProtocolError> {
    input.validate()?;
    let encoded = serde_json::to_string(input)
        .map_err(|error| ProtocolError::InvalidJson("input", error.to_string()))?;
    bounded_string(&encoded, "input")?;
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ToolRequest {
        ToolRequest {
            idempotency_key: "turn-1:tool-0".into(),
            request_id: "req-1".into(),
            occurrence_id: "occ-1".into(),
            name: "read".into(),
            arguments_json: r#"{"path":"src/lib.rs"}"#.into(),
        }
    }

    #[test]
    fn protocol_round_trips_actions_and_rejects_invalid_json() {
        let encoded = encode_actions(&[EngineAction::ExecuteTool { request: request() }]).unwrap();
        let decoded = decode_actions(&encoded).unwrap();
        assert_eq!(decoded.len(), 1);
        assert!(decode_actions("[{\"type\":\"complete\",\"result_json\":\"bad\"}]").is_err());
    }

    #[test]
    fn manifest_signature_payload_omits_signature() {
        let mut manifest = EngineManifest::unsigned("0.1.0", "a".repeat(64));
        let first = manifest.signing_bytes().unwrap();
        manifest.signature_hex = Some("signature".into());
        assert_eq!(first, manifest.signing_bytes().unwrap());
    }

    #[test]
    fn input_and_snapshot_have_version_guards() {
        let mut snapshot = EngineStateSnapshot::new("turn-1");
        assert!(snapshot.validate().is_ok());
        snapshot.api_major = 99;
        assert!(matches!(
            snapshot.validate(),
            Err(ProtocolError::UnsupportedApiMajor(99))
        ));
    }
}
