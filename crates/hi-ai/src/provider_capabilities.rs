//! Typed, canonical provider capability identity.

mod runtime;

use serde::{Deserialize, Serialize};

pub use runtime::{
    CAPABILITY_RECORD_SCHEMA_VERSION, CAPABILITY_REGISTRY_VERSION, CapabilityMemberRecord,
    CapabilityProbe, CapabilityProbeAuditRecord, CapabilityProbeDisposition,
    CapabilityProbeObservation, CapabilityRegistryConfig, CapabilityRoute,
    DEFAULT_CAPABILITY_CACHE_TTL, DEFAULT_CAPABILITY_PROBE_MEMBERS,
    DEFAULT_CAPABILITY_PROBE_TIMEOUT, EffectiveProviderCapabilities, MAX_CAPABILITY_PROBE_MEMBERS,
    MAX_CAPABILITY_PROBE_TIMEOUT, ProviderCapabilityCandidate, ProviderCapabilityRegistry,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolChoiceCapabilities {
    pub automatic: bool,
    pub required: bool,
    pub disabled: bool,
    pub specific_tool: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrictSchemaDialect {
    #[default]
    None,
    Draft7,
    Draft202012,
    OpenAiSubset,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderRequestLimits {
    pub max_input_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub max_tools: Option<u32>,
    pub max_tool_argument_bytes: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderModalities {
    pub text_input: bool,
    pub image_input: bool,
    pub audio_input: bool,
    pub text_output: bool,
    pub image_output: bool,
    pub audio_output: bool,
}

impl Default for ProviderModalities {
    fn default() -> Self {
        Self {
            text_input: true,
            image_input: false,
            audio_input: false,
            text_output: true,
            image_output: false,
            audio_output: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageReporting {
    #[default]
    None,
    Final,
    Streamed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningReplayCapabilities {
    pub plain_text: bool,
    pub signed_or_encrypted: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationSupport {
    #[default]
    None,
    Cooperative,
    TransportAbort,
}

/// Capabilities that affect request shaping, execution, and trace comparison.
/// Unknown providers get the conservative default: text-only, no optional
/// protocol promises, no asserted limits, and no model-revision claim.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub native_tool_calls: bool,
    pub tool_choice: ToolChoiceCapabilities,
    pub parallel_tool_calls: bool,
    pub strict_schema_dialect: StrictSchemaDialect,
    /// Whether the provider streams incremental tool name/argument fragments.
    pub streamed_tool_call_deltas: bool,
    pub request_limits: ProviderRequestLimits,
    pub structured_output: bool,
    pub modalities: ProviderModalities,
    pub usage_reporting: UsageReporting,
    pub reasoning_replay: ReasoningReplayCapabilities,
    pub cancellation: CancellationSupport,
    pub actual_model_revision: Option<String>,
}

impl ProviderCapabilities {
    /// Compatibility constructor for providers that support native tool calls
    /// but have not yet populated the richer negotiated fields.
    pub fn native_tools(streamed_arguments: bool) -> Self {
        Self {
            native_tool_calls: true,
            tool_choice: ToolChoiceCapabilities {
                automatic: true,
                required: true,
                disabled: true,
                specific_tool: false,
            },
            streamed_tool_call_deltas: streamed_arguments,
            ..Self::default()
        }
    }

    /// Conservative intersection for a route that may dispatch to any member.
    /// A feature survives only when every member supports it. Numeric limits
    /// survive only when every member reports one, using the smallest value.
    pub fn conservative_intersection(&self, other: &Self) -> Self {
        Self {
            native_tool_calls: self.native_tool_calls && other.native_tool_calls,
            tool_choice: ToolChoiceCapabilities {
                automatic: self.tool_choice.automatic && other.tool_choice.automatic,
                required: self.tool_choice.required && other.tool_choice.required,
                disabled: self.tool_choice.disabled && other.tool_choice.disabled,
                specific_tool: self.tool_choice.specific_tool && other.tool_choice.specific_tool,
            },
            parallel_tool_calls: self.parallel_tool_calls && other.parallel_tool_calls,
            strict_schema_dialect: if self.strict_schema_dialect == other.strict_schema_dialect {
                self.strict_schema_dialect
            } else {
                StrictSchemaDialect::None
            },
            streamed_tool_call_deltas: self.streamed_tool_call_deltas
                && other.streamed_tool_call_deltas,
            request_limits: self.request_limits.intersection(other.request_limits),
            structured_output: self.structured_output && other.structured_output,
            modalities: self.modalities.intersection(other.modalities),
            usage_reporting: self.usage_reporting.min(other.usage_reporting),
            reasoning_replay: ReasoningReplayCapabilities {
                plain_text: self.reasoning_replay.plain_text && other.reasoning_replay.plain_text,
                signed_or_encrypted: self.reasoning_replay.signed_or_encrypted
                    && other.reasoning_replay.signed_or_encrypted,
            },
            cancellation: self.cancellation.min(other.cancellation),
            actual_model_revision: (self.actual_model_revision == other.actual_model_revision)
                .then(|| self.actual_model_revision.clone())
                .flatten(),
        }
    }

    /// Canonical identifier attached to each audited tool envelope.
    pub fn canonical_digest(&self) -> String {
        let encoded = serde_json::to_vec(self).expect("provider capabilities serialize");
        format!("blake3:{}", blake3::hash(&encoded).to_hex())
    }
}

impl ProviderRequestLimits {
    fn intersection(self, other: Self) -> Self {
        Self {
            max_input_tokens: intersect_limit(self.max_input_tokens, other.max_input_tokens),
            max_output_tokens: intersect_limit(self.max_output_tokens, other.max_output_tokens),
            max_tools: intersect_limit(self.max_tools, other.max_tools),
            max_tool_argument_bytes: intersect_limit(
                self.max_tool_argument_bytes,
                other.max_tool_argument_bytes,
            ),
        }
    }
}

impl ProviderModalities {
    fn intersection(self, other: Self) -> Self {
        Self {
            text_input: self.text_input && other.text_input,
            image_input: self.image_input && other.image_input,
            audio_input: self.audio_input && other.audio_input,
            text_output: self.text_output && other.text_output,
            image_output: self.image_output && other.image_output,
            audio_output: self.audio_output && other.audio_output,
        }
    }
}

fn intersect_limit(left: Option<u32>, right: Option<u32>) -> Option<u32> {
    left.zip(right).map(|(left, right)| left.min(right))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capable(revision: &str) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calls: true,
            tool_choice: ToolChoiceCapabilities {
                automatic: true,
                required: true,
                disabled: true,
                specific_tool: false,
            },
            parallel_tool_calls: true,
            strict_schema_dialect: StrictSchemaDialect::Draft202012,
            streamed_tool_call_deltas: true,
            request_limits: ProviderRequestLimits {
                max_input_tokens: Some(128_000),
                max_output_tokens: Some(16_384),
                max_tools: Some(128),
                max_tool_argument_bytes: Some(1_000_000),
            },
            structured_output: true,
            modalities: ProviderModalities {
                image_input: true,
                ..ProviderModalities::default()
            },
            usage_reporting: UsageReporting::Streamed,
            reasoning_replay: ReasoningReplayCapabilities {
                plain_text: true,
                signed_or_encrypted: true,
            },
            cancellation: CancellationSupport::TransportAbort,
            actual_model_revision: Some(revision.to_owned()),
        }
    }

    #[test]
    fn capability_digest_is_stable_and_complete() {
        fn changed(
            capabilities: &ProviderCapabilities,
            mutate: impl FnOnce(&mut ProviderCapabilities),
        ) -> ProviderCapabilities {
            let mut changed = capabilities.clone();
            mutate(&mut changed);
            changed
        }
        let capabilities = capable("model@1");
        assert_eq!(
            capabilities.canonical_digest(),
            capabilities.canonical_digest()
        );
        let variants = [
            changed(&capabilities, |value| value.native_tool_calls = false),
            changed(&capabilities, |value| value.tool_choice.required = false),
            changed(&capabilities, |value| value.parallel_tool_calls = false),
            changed(&capabilities, |value| {
                value.strict_schema_dialect = StrictSchemaDialect::Draft7
            }),
            changed(&capabilities, |value| {
                value.streamed_tool_call_deltas = false
            }),
            changed(&capabilities, |value| {
                value.request_limits.max_output_tokens = Some(8_192)
            }),
            changed(&capabilities, |value| value.structured_output = false),
            changed(&capabilities, |value| value.modalities.image_input = false),
            changed(&capabilities, |value| {
                value.usage_reporting = UsageReporting::Final
            }),
            changed(&capabilities, |value| {
                value.reasoning_replay.signed_or_encrypted = false
            }),
            changed(&capabilities, |value| {
                value.cancellation = CancellationSupport::Cooperative
            }),
            changed(&capabilities, |value| {
                value.actual_model_revision = Some("model@2".to_owned())
            }),
        ];
        for (index, changed) in variants.iter().enumerate() {
            assert_ne!(
                capabilities.canonical_digest(),
                changed.canonical_digest(),
                "capability field variant {index} was omitted from the digest"
            );
        }
    }

    #[test]
    fn fallback_intersection_is_fail_closed_and_uses_smallest_known_limits() {
        let left = capable("model@1");
        let mut right = capable("model@2");
        right.tool_choice.required = false;
        right.request_limits.max_output_tokens = Some(8_192);
        right.request_limits.max_tools = None;
        right.modalities.image_input = false;
        right.cancellation = CancellationSupport::Cooperative;
        let merged = left.conservative_intersection(&right);
        assert!(!merged.tool_choice.required);
        assert_eq!(merged.request_limits.max_output_tokens, Some(8_192));
        assert_eq!(merged.request_limits.max_tools, None);
        assert!(!merged.modalities.image_input);
        assert_eq!(merged.cancellation, CancellationSupport::Cooperative);
        assert_eq!(merged.actual_model_revision, None);
    }
}

#[cfg(test)]
#[path = "provider_capabilities/runtime_tests.rs"]
mod runtime_tests;
