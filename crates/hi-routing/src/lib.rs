//! Capability-aware model and harness routing.
//!
//! This crate contains policy-neutral route contracts. Callers supply the
//! candidates allowed by their current scope and policy; the resolver only
//! selects candidates that satisfy the request and records why others were
//! rejected.

use std::collections::BTreeSet;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Streaming,
    StructuredTools,
    Reasoning,
    Vision,
    JsonSchema,
    ToolReplay,
    Network,
    WorkspaceRead,
    WorkspaceWrite,
    ProcessExecution,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    #[serde(default)]
    pub capabilities: BTreeSet<Capability>,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
}

impl CapabilitySet {
    pub fn with(mut self, capability: Capability) -> Self {
        self.capabilities.insert(capability);
        self
    }

    pub fn supports(&self, required: &CapabilitySet) -> bool {
        required
            .capabilities
            .iter()
            .all(|capability| self.capabilities.contains(capability))
            && required
                .context_window
                .is_none_or(|required| self.context_window.is_some_and(|actual| actual >= required))
            && required.max_output_tokens.is_none_or(|required| {
                self.max_output_tokens
                    .is_some_and(|actual| actual >= required)
            })
    }

    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        blake3::hash(&bytes).to_hex().to_string()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteRequirements {
    #[serde(default)]
    pub capabilities: CapabilitySet,
    pub require_available: bool,
    pub require_credentials: bool,
    pub scope_id: Option<String>,
    pub policy_digest: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCandidate {
    pub provider: String,
    pub model: String,
    pub capabilities: CapabilitySet,
    pub available: bool,
    pub credential_available: bool,
    pub health: String,
}

impl ModelCandidate {
    pub fn from_served_model(provider: impl Into<String>, model: &hi_ai::ServedModel) -> Self {
        let mut capabilities = CapabilitySet {
            capabilities: BTreeSet::new(),
            context_window: model.context_window,
            max_output_tokens: model.max_output_tokens,
        };
        for raw in &model.capabilities {
            match raw.trim().to_ascii_lowercase().as_str() {
                "stream" | "streaming" => capabilities.capabilities.insert(Capability::Streaming),
                "tools" | "tool_calls" | "tool-calls" => capabilities
                    .capabilities
                    .insert(Capability::StructuredTools),
                "reasoning" | "thinking" => capabilities.capabilities.insert(Capability::Reasoning),
                "vision" | "images" => capabilities.capabilities.insert(Capability::Vision),
                "json" | "json_schema" | "structured_output" => {
                    capabilities.capabilities.insert(Capability::JsonSchema)
                }
                "tool_replay" => capabilities.capabilities.insert(Capability::ToolReplay),
                _ => false,
            };
        }
        Self {
            provider: provider.into(),
            model: model.id.clone(),
            capabilities,
            available: model.available,
            credential_available: true,
            health: model.status.clone().unwrap_or_else(|| "unknown".into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessDescriptor {
    pub id: String,
    pub version: String,
    pub capabilities: CapabilitySet,
    pub isolation: String,
    pub network_allowed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteCandidate {
    pub harness: HarnessDescriptor,
    pub model: ModelCandidate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteRejection {
    Unavailable,
    CredentialsUnavailable,
    ModelCapabilities,
    HarnessCapabilities,
    PolicyScopeMissing,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedRoute {
    pub provider: String,
    pub model: String,
    pub harness: String,
    pub reason: RouteRejection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub selected: RouteCandidate,
    pub rejected: Vec<RejectedRoute>,
    pub requirements: RouteRequirements,
    pub capability_digest: String,
}

#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("no route satisfies the requested capabilities")]
    NoRoute { rejected: Vec<RejectedRoute> },
    #[error("harness execution failed: {0}")]
    Harness(String),
}

pub struct RouteResolver;

impl RouteResolver {
    pub fn resolve(
        requirements: RouteRequirements,
        candidates: impl IntoIterator<Item = RouteCandidate>,
    ) -> Result<RouteDecision, RoutingError> {
        let mut rejected = Vec::new();
        for candidate in candidates {
            let rejected_reason = if requirements.require_available
                && (!candidate.model.available
                    || candidate.model.health.eq_ignore_ascii_case("degraded"))
            {
                Some(RouteRejection::Unavailable)
            } else if requirements.require_credentials && !candidate.model.credential_available {
                Some(RouteRejection::CredentialsUnavailable)
            } else if !candidate
                .model
                .capabilities
                .supports(&requirements.capabilities)
            {
                Some(RouteRejection::ModelCapabilities)
            } else if !candidate
                .harness
                .capabilities
                .supports(&requirements.capabilities)
            {
                Some(RouteRejection::HarnessCapabilities)
            } else if requirements.scope_id.is_none()
                && candidate
                    .model
                    .capabilities
                    .capabilities
                    .contains(&Capability::WorkspaceWrite)
            {
                Some(RouteRejection::PolicyScopeMissing)
            } else {
                None
            };
            if let Some(reason) = rejected_reason {
                rejected.push(RejectedRoute {
                    provider: candidate.model.provider.clone(),
                    model: candidate.model.model.clone(),
                    harness: candidate.harness.id.clone(),
                    reason,
                });
                continue;
            }
            let capability_digest = route_digest(&candidate, &requirements);
            return Ok(RouteDecision {
                selected: candidate,
                rejected,
                requirements,
                capability_digest,
            });
        }
        Err(RoutingError::NoRoute { rejected })
    }
}

fn route_digest(candidate: &RouteCandidate, requirements: &RouteRequirements) -> String {
    let value = serde_json::json!({
        "harness": candidate.harness,
        "provider": candidate.model.provider,
        "model": candidate.model.model,
        "requirements": requirements,
    });
    blake3::hash(value.to_string().as_bytes())
        .to_hex()
        .to_string()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HarnessRequest {
    pub run_id: String,
    pub attempt_id: String,
    pub payload: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HarnessResponse {
    pub payload: serde_json::Value,
}

/// Future local/remote harness implementations use this seam. v1 only needs
/// the local hi agent adapter; the trait does not grant policy or scope access.
#[async_trait]
pub trait HarnessAdapter: Send + Sync {
    fn descriptor(&self) -> &HarnessDescriptor;
    async fn execute(&self, request: HarnessRequest) -> Result<HarnessResponse, RoutingError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(tools: bool, available: bool) -> RouteCandidate {
        let mut model_caps = CapabilitySet::default();
        if tools {
            model_caps = model_caps.with(Capability::StructuredTools);
        }
        RouteCandidate {
            harness: HarnessDescriptor {
                id: "hi".into(),
                version: "1".into(),
                capabilities: CapabilitySet::default().with(Capability::StructuredTools),
                isolation: "workspace".into(),
                network_allowed: false,
            },
            model: ModelCandidate {
                provider: "test".into(),
                model: "model".into(),
                capabilities: model_caps,
                available,
                credential_available: true,
                health: "available".into(),
            },
        }
    }

    #[test]
    fn rejects_missing_capabilities_and_selects_next_candidate() {
        let requirements = RouteRequirements {
            capabilities: CapabilitySet::default().with(Capability::StructuredTools),
            require_available: true,
            ..RouteRequirements::default()
        };
        let decision = RouteResolver::resolve(
            requirements,
            [candidate(false, true), candidate(true, true)],
        )
        .unwrap();
        assert_eq!(decision.selected.model.model, "model");
        assert_eq!(
            decision.rejected[0].reason,
            RouteRejection::ModelCapabilities
        );
    }

    #[test]
    fn write_capability_requires_a_scope() {
        let mut write_candidate = candidate(true, true);
        write_candidate
            .model
            .capabilities
            .capabilities
            .insert(Capability::WorkspaceWrite);
        write_candidate
            .harness
            .capabilities
            .capabilities
            .insert(Capability::WorkspaceWrite);
        let requirements = RouteRequirements {
            capabilities: CapabilitySet::default().with(Capability::WorkspaceWrite),
            ..RouteRequirements::default()
        };
        assert!(matches!(
            RouteResolver::resolve(requirements, [write_candidate]),
            Err(RoutingError::NoRoute { rejected }) if rejected[0].reason == RouteRejection::PolicyScopeMissing
        ));
    }
}
