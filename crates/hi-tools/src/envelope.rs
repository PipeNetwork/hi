//! Canonical, audit-safe description of the tools available to one model request.
//!
//! The envelope is built only after task-aware selection has produced the exact
//! ordered [`hi_ai::ToolSpec`] slice. Its digest is therefore suitable for
//! admission checks, trace comparison, and durable journal records. It is not
//! a permission grant by itself; execution still enforces the catalog and
//! workspace controller policies.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hi_ai::{EffectiveProviderCapabilities, ToolMode, ToolSpec};
use hi_workspace::{WorkspaceAuthority, WorkspaceBinding, WorkspaceVersion};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::catalog::{
    ArtifactPolicy, EffectScope, OutputPolicy, ReplayClass, ResourceAccess, tool_metadata,
};

pub const TOOL_ENVELOPE_SCHEMA_VERSION: u16 = 3;
pub const TOOL_ENVELOPE_MAX_INLINE_OUTPUT_BYTES: u64 = 50_000;
const DIGEST_PREFIX: &str = "blake3:";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceTrust {
    Trusted,
    Untrusted,
    OperatorOverride,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEnvelopeLimits {
    pub max_output_tokens: u32,
    pub max_parallel_calls: u16,
    pub max_calls_per_round: u16,
    pub max_inline_output_bytes: u64,
    pub max_tool_argument_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderEnvelope {
    pub route: String,
    pub requested_model: String,
    pub actual_model_revision: Option<String>,
    /// Canonical digest of the capability record used for this request.
    pub capability_digest: String,
    /// Full effective record, including fallback-member provenance. Keeping the
    /// record beside its digest makes request-shaping claims independently auditable.
    pub capability_record: EffectiveProviderCapabilities,
}

impl ProviderEnvelope {
    pub fn from_capability_record(record: EffectiveProviderCapabilities) -> Self {
        Self {
            route: record.target.route.clone(),
            requested_model: record.target.model.clone(),
            actual_model_revision: record.actual_model_revision().map(str::to_string),
            capability_digest: record.canonical_digest(),
            capability_record: record,
        }
    }

    pub fn capability_identity_is_valid(&self) -> bool {
        self.capability_record.schema_version == hi_ai::CAPABILITY_RECORD_SCHEMA_VERSION
            && self.capability_record.registry_version > 0
            && self.route == self.capability_record.target.route
            && self.requested_model == self.capability_record.target.model
            && self.actual_model_revision
                == self.capability_record.capabilities.actual_model_revision
            && self.capability_digest == self.capability_record.canonical_digest()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEnvelope {
    pub authority: WorkspaceAuthority,
    pub binding_id: String,
    pub epoch: u64,
    pub version: WorkspaceVersion,
}

impl From<&WorkspaceBinding> for WorkspaceEnvelope {
    fn from(binding: &WorkspaceBinding) -> Self {
        Self {
            authority: binding.authority.clone(),
            binding_id: binding.binding_id.as_str().to_owned(),
            epoch: binding.epoch,
            version: binding.version.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEnvelopeContext {
    pub provider: ProviderEnvelope,
    pub workspace: WorkspaceEnvelope,
    pub trust: WorkspaceTrust,
    /// Stable permission identifiers. A set deliberately makes ordering
    /// irrelevant and prevents caller-specific hash churn.
    pub permissions: BTreeSet<String>,
    pub limits: ToolEnvelopeLimits,
    /// Effective choice policy for this request. The tools may remain present
    /// during a ChatOnly wrap-up for cache stability, but the envelope must
    /// still record that none are executable.
    pub tool_mode: ToolMode,
    /// Runtime versions for dynamically discovered tools, keyed by advertised
    /// name. Built-ins default to this crate's version; unknown tools stay
    /// explicitly `unknown` unless the selector provides a version here.
    pub tool_versions: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvelopeTool {
    pub position: u32,
    pub name: String,
    pub implementation_version: String,
    pub schema_version: u16,
    /// Standalone provider token estimate for this tool's name,
    /// description, and input schema. Keeping it in the request envelope
    /// makes schema-cost decisions attributable per tool instead of exposing
    /// only an aggregate catalog total.
    pub schema_token_cost: u64,
    pub description: String,
    pub input_schema: Value,
    pub effect_scope: EffectScope,
    pub replay_class: ReplayClass,
    pub resource_access: ResourceAccess,
    pub output_policy: OutputPolicy,
    pub artifact_policy: ArtifactPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEnvelopePayload {
    pub schema_version: u16,
    /// Exact ordered provider-facing tool list.
    pub tools: Vec<EnvelopeTool>,
    /// Exact ordered host-tool list callable only from an admitted
    /// `run_program`. These tools are not provider-facing and therefore do
    /// not make a direct model tool call admissible.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub program_tools: Vec<EnvelopeTool>,
    pub provider: ProviderEnvelope,
    pub workspace: WorkspaceEnvelope,
    pub trust: WorkspaceTrust,
    pub permissions: BTreeSet<String>,
    pub limits: ToolEnvelopeLimits,
    pub tool_mode: ToolMode,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolEnvelope {
    pub digest: String,
    pub payload: ToolEnvelopePayload,
}

/// Sealed policy for a provider request that intentionally has no workspace
/// tools. Standalone evaluators and the higher-trust RSI executor still attach
/// the same canonical capability/workspace envelope as interactive turns;
/// `ChatOnly` ensures the synthetic local binding grants no executable tool.
#[derive(Clone, Debug)]
pub struct ChatOnlyRequestPolicy {
    pub max_output_tokens: u32,
    pub envelope: Arc<hi_ai::RequestToolEnvelope>,
}

/// Resolve the provider's conservative route capabilities and seal a no-tool
/// request at the last boundary before it is sent. `scope` must be a stable,
/// non-secret workflow identity (for example `rsi-stage` or `diff-lab`).
pub async fn seal_chat_only_request(
    provider: &dyn hi_ai::Provider,
    route: &str,
    model: &str,
    requested_max_output_tokens: u32,
    scope: &str,
) -> ChatOnlyRequestPolicy {
    let target = hi_ai::CapabilityRoute::new(route, model);
    let candidates = provider.capability_candidates(&target.route, &target.model);
    let effective = hi_ai::ProviderCapabilityRegistry::default()
        .resolve_candidates(target, &candidates)
        .await;
    let max_output_tokens = effective
        .capabilities
        .request_limits
        .max_output_tokens
        .map_or(requested_max_output_tokens, |limit| {
            requested_max_output_tokens.min(limit)
        });
    let client_argument_limit = u32::try_from(hi_ai::MAX_TOOL_ARGUMENT_BYTES).unwrap_or(u32::MAX);
    let max_tool_argument_bytes = effective
        .capabilities
        .request_limits
        .max_tool_argument_bytes
        .unwrap_or(client_argument_limit)
        .min(client_argument_limit);
    let envelope = ToolEnvelope::build(
        &[],
        ToolEnvelopeContext {
            provider: ProviderEnvelope::from_capability_record(effective),
            workspace: WorkspaceEnvelope {
                authority: WorkspaceAuthority::Local,
                binding_id: format!("standalone-chat-only:{scope}"),
                epoch: 0,
                version: WorkspaceVersion::Unknown,
            },
            trust: WorkspaceTrust::Untrusted,
            permissions: BTreeSet::from([
                "tools:disabled".to_string(),
                "workspace_access:none".to_string(),
                format!("request_scope:{scope}"),
            ]),
            limits: ToolEnvelopeLimits {
                max_output_tokens,
                max_parallel_calls: 1,
                max_calls_per_round: 0,
                max_inline_output_bytes: TOOL_ENVELOPE_MAX_INLINE_OUTPUT_BYTES,
                max_tool_argument_bytes,
            },
            tool_mode: ToolMode::ChatOnly,
            tool_versions: BTreeMap::new(),
        },
    );
    ChatOnlyRequestPolicy {
        max_output_tokens,
        envelope: Arc::new(hi_ai::RequestToolEnvelope {
            digest: envelope.digest,
            payload: serde_json::to_value(envelope.payload)
                .expect("tool envelope payload serializes"),
        }),
    }
}

impl ToolEnvelope {
    /// Build an envelope from the exact tool order advertised on a request.
    /// Unknown dynamic tools are deliberately treated as live, non-replayable
    /// external effects until their host supplies richer metadata.
    pub fn build(specs: &[ToolSpec], context: ToolEnvelopeContext) -> Self {
        Self::build_with_program_tools(specs, &[], context)
    }

    /// Build an envelope that also seals the non-provider-facing tools the
    /// admitted `run_program` host may execute.
    pub fn build_with_program_tools(
        specs: &[ToolSpec],
        program_specs: &[ToolSpec],
        context: ToolEnvelopeContext,
    ) -> Self {
        let tools = envelope_tools(specs, &context.tool_versions);
        let program_tools = envelope_tools(program_specs, &context.tool_versions);
        let payload = ToolEnvelopePayload {
            schema_version: TOOL_ENVELOPE_SCHEMA_VERSION,
            tools,
            program_tools,
            provider: context.provider,
            workspace: context.workspace,
            trust: context.trust,
            permissions: context.permissions,
            limits: context.limits,
            tool_mode: context.tool_mode,
        };
        let digest = digest_serializable(&payload);
        Self { digest, payload }
    }

    pub fn digest_is_valid(&self) -> bool {
        self.payload.provider.capability_identity_is_valid()
            && self.digest == digest_serializable(&self.payload)
    }

    pub fn admits(&self, tool_name: &str) -> bool {
        !matches!(self.payload.tool_mode, ToolMode::ChatOnly)
            && self.payload.tools.iter().any(|tool| tool.name == tool_name)
    }

    pub fn admits_program(&self, tool_name: &str) -> bool {
        !matches!(self.payload.tool_mode, ToolMode::ChatOnly)
            && self
                .payload
                .program_tools
                .iter()
                .any(|tool| tool.name == tool_name)
    }

    pub fn program_specs(&self) -> Vec<ToolSpec> {
        self.payload
            .program_tools
            .iter()
            .map(|tool| ToolSpec {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            })
            .collect()
    }

    /// Prove the executor received the same ordered provider-facing schemas
    /// sealed by this envelope, not a broader catalog reconstructed later.
    pub fn matches_specs(&self, specs: &[ToolSpec]) -> bool {
        matches_tools(&self.payload.tools, specs)
    }

    pub fn matches_program_specs(&self, specs: &[ToolSpec]) -> bool {
        matches_tools(&self.payload.program_tools, specs)
    }
}

fn matches_tools(tools: &[EnvelopeTool], specs: &[ToolSpec]) -> bool {
    tools.len() == specs.len()
        && tools.iter().zip(specs).all(|(tool, spec)| {
            tool.name == spec.name
                && tool.description == spec.description
                && tool.input_schema == canonicalize_json(spec.parameters.clone())
        })
}

fn envelope_tools(
    specs: &[ToolSpec],
    tool_versions: &BTreeMap<String, String>,
) -> Vec<EnvelopeTool> {
    specs
        .iter()
        .enumerate()
        .map(|(position, spec)| {
            let metadata = tool_metadata(&spec.name);
            let policy = metadata
                .map(EnvelopePolicy::from)
                .unwrap_or_else(EnvelopePolicy::conservative);
            EnvelopeTool {
                position: position as u32,
                name: spec.name.clone(),
                implementation_version: tool_versions.get(&spec.name).cloned().unwrap_or_else(
                    || {
                        if metadata.is_some() {
                            env!("CARGO_PKG_VERSION").to_owned()
                        } else {
                            "unknown".to_owned()
                        }
                    },
                ),
                schema_version: policy.schema_version,
                schema_token_cost: hi_ai::estimate_tool_schema_tokens(std::slice::from_ref(spec)),
                description: spec.description.clone(),
                input_schema: canonicalize_json(spec.parameters.clone()),
                effect_scope: policy.effect_scope,
                replay_class: policy.replay_class,
                resource_access: policy.resource_access,
                output_policy: policy.output_policy,
                artifact_policy: policy.artifact_policy,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct EnvelopePolicy {
    schema_version: u16,
    effect_scope: EffectScope,
    replay_class: ReplayClass,
    resource_access: ResourceAccess,
    output_policy: OutputPolicy,
    artifact_policy: ArtifactPolicy,
}

impl From<&crate::catalog::ToolMetadata> for EnvelopePolicy {
    fn from(metadata: &crate::catalog::ToolMetadata) -> Self {
        Self {
            schema_version: metadata.schema_version,
            effect_scope: metadata.policy.effect_scope,
            replay_class: metadata.policy.replay_class,
            resource_access: metadata.policy.resource_access,
            output_policy: metadata.policy.output,
            artifact_policy: metadata.policy.artifacts,
        }
    }
}

impl EnvelopePolicy {
    const fn conservative() -> Self {
        Self {
            schema_version: 0,
            ..Self::from_policy(crate::catalog::ToolPolicy::conservative())
        }
    }

    const fn from_policy(policy: crate::catalog::ToolPolicy) -> Self {
        Self {
            schema_version: TOOL_ENVELOPE_SCHEMA_VERSION,
            effect_scope: policy.effect_scope,
            replay_class: policy.replay_class,
            resource_access: policy.resource_access,
            output_policy: policy.output,
            artifact_policy: policy.artifacts,
        }
    }
}

/// Stable digest for just the provider-facing schemas. This is useful for
/// prompt-cache telemetry before the full workspace/provider context is known.
pub fn canonical_tool_schema_digest(specs: &[ToolSpec]) -> String {
    let value = Value::Array(
        specs
            .iter()
            .map(|spec| {
                Value::Object(Map::from_iter([
                    ("name".to_owned(), Value::String(spec.name.clone())),
                    (
                        "description".to_owned(),
                        Value::String(spec.description.clone()),
                    ),
                    (
                        "parameters".to_owned(),
                        canonicalize_json(spec.parameters.clone()),
                    ),
                ]))
            })
            .collect(),
    );
    digest_bytes(&serde_json::to_vec(&canonicalize_json(value)).expect("JSON Value serializes"))
}

pub fn canonical_value_digest(value: &Value) -> String {
    digest_bytes(
        &serde_json::to_vec(&canonicalize_json(value.clone())).expect("JSON Value serializes"),
    )
}

fn digest_serializable(value: &impl Serialize) -> String {
    let value = serde_json::to_value(value).expect("envelope fields serialize");
    let bytes = serde_json::to_vec(&canonicalize_json(value)).expect("JSON Value serializes");
    digest_bytes(&bytes)
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{DIGEST_PREFIX}{}", blake3::hash(bytes).to_hex())
}

fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        Value::Object(values) => {
            let mut keys = values.into_iter().collect::<Vec<_>>();
            keys.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            Value::Object(Map::from_iter(
                keys.into_iter()
                    .map(|(key, value)| (key, canonicalize_json(value))),
            ))
        }
        scalar => scalar,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use async_trait::async_trait;
    use serde_json::json;

    use super::*;

    fn spec(name: &str, schema: Value) -> ToolSpec {
        ToolSpec {
            name: name.to_owned(),
            description: format!("{name} description"),
            parameters: schema,
        }
    }

    fn context() -> ToolEnvelopeContext {
        let record = EffectiveProviderCapabilities::conservative(
            hi_ai::CapabilityRoute::new("provider/default", "model"),
            hi_ai::ProviderCapabilities {
                actual_model_revision: Some("model@2026-01-01".to_owned()),
                ..hi_ai::ProviderCapabilities::default()
            },
        );
        ToolEnvelopeContext {
            provider: ProviderEnvelope::from_capability_record(record),
            workspace: WorkspaceEnvelope {
                authority: WorkspaceAuthority::Local,
                binding_id: "binding".to_owned(),
                epoch: 0,
                version: WorkspaceVersion::Local {
                    generation: 7,
                    content_digest: Some("blake3:workspace".to_owned()),
                },
            },
            trust: WorkspaceTrust::Trusted,
            permissions: BTreeSet::from(["workspace.read".to_owned()]),
            limits: ToolEnvelopeLimits {
                max_output_tokens: 16_384,
                max_parallel_calls: 4,
                max_calls_per_round: 16,
                max_inline_output_bytes: 50_000,
                max_tool_argument_bytes: hi_ai::MAX_TOOL_ARGUMENT_BYTES as u32,
            },
            tool_mode: ToolMode::Auto,
            tool_versions: BTreeMap::new(),
        }
    }

    struct LimitedChatProvider;

    #[async_trait]
    impl hi_ai::Provider for LimitedChatProvider {
        async fn stream(
            &self,
            _request: hi_ai::ChatRequest,
            _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
        ) -> anyhow::Result<hi_ai::Completion> {
            unreachable!("envelope sealing never performs a model request")
        }

        fn capabilities(&self) -> hi_ai::ProviderCapabilities {
            hi_ai::ProviderCapabilities {
                request_limits: hi_ai::ProviderRequestLimits {
                    max_output_tokens: Some(64),
                    ..hi_ai::ProviderRequestLimits::default()
                },
                // A configured model identifier is not immutable revision
                // evidence, so the provider deliberately leaves this unset.
                actual_model_revision: None,
                ..hi_ai::ProviderCapabilities::default()
            }
        }
    }

    #[tokio::test]
    async fn standalone_chat_request_is_sealed_with_conservative_provider_limits() {
        let policy = seal_chat_only_request(
            &LimitedChatProvider,
            "test-route",
            "configured-model",
            1_024,
            "test-workflow",
        )
        .await;
        let envelope = ToolEnvelope {
            digest: policy.envelope.digest.clone(),
            payload: serde_json::from_value(policy.envelope.payload.clone())
                .expect("standalone payload is the canonical ToolEnvelope schema"),
        };

        assert_eq!(policy.max_output_tokens, 64);
        assert_eq!(policy.envelope.digest, envelope.digest);
        assert!(envelope.digest_is_valid());
        assert_eq!(envelope.payload.tool_mode, ToolMode::ChatOnly);
        assert!(envelope.payload.tools.is_empty());
        assert_eq!(
            envelope.payload.provider.requested_model,
            "configured-model"
        );
        assert_eq!(envelope.payload.provider.actual_model_revision, None);
    }

    #[test]
    fn object_key_order_does_not_change_digest() {
        let first = vec![spec(
            "read",
            json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )];
        let second = vec![spec(
            "read",
            json!({"properties": {"path": {"type": "string"}}, "type": "object"}),
        )];
        assert_eq!(
            ToolEnvelope::build(&first, context()).digest,
            ToolEnvelope::build(&second, context()).digest
        );
    }

    #[test]
    fn advertised_tool_order_is_digest_significant() {
        let first = vec![spec("read", json!({})), spec("bash", json!({}))];
        let second = vec![spec("bash", json!({})), spec("read", json!({}))];
        assert_ne!(
            ToolEnvelope::build(&first, context()).digest,
            ToolEnvelope::build(&second, context()).digest
        );
        assert!(ToolEnvelope::build(&first, context()).matches_specs(&first));
        assert!(!ToolEnvelope::build(&first, context()).matches_specs(&second));
    }

    #[test]
    fn chat_only_envelope_never_admits_catalog_or_program_tools() {
        let specs = vec![spec("read", json!({})), spec("run_program", json!({}))];
        let program_specs = vec![spec("grep", json!({}))];
        let mut chat_only = context();
        chat_only.tool_mode = ToolMode::ChatOnly;
        let envelope = ToolEnvelope::build_with_program_tools(&specs, &program_specs, chat_only);

        assert!(envelope.matches_specs(&specs));
        assert!(!envelope.admits("read"));
        assert!(!envelope.admits("run_program"));
        assert!(!envelope.admits_program("grep"));
    }

    #[test]
    fn envelope_records_attributable_schema_cost_per_tool() {
        let specs = vec![
            spec("read", json!({"type": "object"})),
            spec(
                "write",
                json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    }
                }),
            ),
        ];
        let envelope = ToolEnvelope::build(&specs, context());
        for (recorded, advertised) in envelope.payload.tools.iter().zip(&specs) {
            assert_eq!(
                recorded.schema_token_cost,
                hi_ai::estimate_tool_schema_tokens(std::slice::from_ref(advertised))
            );
            assert!(recorded.schema_token_cost > 0);
        }
    }

    #[test]
    fn program_host_tools_are_sealed_without_becoming_directly_advertised() {
        let advertised = [spec("run_program", json!({}))];
        let nested = [spec("read", json!({}))];
        let envelope = ToolEnvelope::build_with_program_tools(&advertised, &nested, context());
        assert!(envelope.admits("run_program"));
        assert!(!envelope.admits("read"));
        assert!(envelope.admits_program("read"));
        assert!(envelope.matches_program_specs(&envelope.program_specs()));
        assert!(envelope.digest_is_valid());
    }

    #[test]
    fn unknown_dynamic_tools_are_fail_closed() {
        let envelope = ToolEnvelope::build(&[spec("remote_dynamic", json!({}))], context());
        let tool = &envelope.payload.tools[0];
        assert_eq!(tool.schema_version, 0);
        assert_eq!(tool.implementation_version, "unknown");
        assert_eq!(tool.effect_scope, EffectScope::LiveWriter);
        assert_eq!(tool.replay_class, ReplayClass::NonReplayableExternal);
        assert!(tool.resource_access.network);
        assert!(envelope.digest_is_valid());
        assert!(envelope.admits("remote_dynamic"));
        assert!(!envelope.admits("read"));
    }

    #[test]
    fn workspace_epoch_and_permissions_are_digest_significant() {
        let specs = [spec("read", json!({}))];
        let first = ToolEnvelope::build(&specs, context());
        let mut changed = context();
        changed.workspace.epoch += 1;
        changed.permissions.insert("network".to_owned());
        let second = ToolEnvelope::build(&specs, changed);
        assert_ne!(first.digest, second.digest);
    }

    #[test]
    fn dynamic_version_is_recorded_and_digest_is_self_verifying() {
        let specs = [spec("remote_dynamic", json!({}))];
        let mut selected = context();
        selected
            .tool_versions
            .insert("remote_dynamic".to_owned(), "mcp-server@42".to_owned());
        let mut envelope = ToolEnvelope::build(&specs, selected);
        assert_eq!(
            envelope.payload.tools[0].implementation_version,
            "mcp-server@42"
        );
        assert!(envelope.digest_is_valid());
        envelope.payload.tools[0].input_schema = json!({"type": "string"});
        assert!(!envelope.digest_is_valid());
    }
}
