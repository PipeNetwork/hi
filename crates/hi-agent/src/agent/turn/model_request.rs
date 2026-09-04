//! ChatRequest assembly helpers for the Model phase.
//!
//! Keeps tool-list filtering and schema accounting out of the main
//! `run_model_round` body.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use hi_ai::{ToolMode, ToolSpec};
use hi_tools::envelope::{
    ProviderEnvelope, ToolEnvelope, ToolEnvelopeContext, ToolEnvelopeLimits, WorkspaceEnvelope,
    WorkspaceTrust,
};

pub(super) mod provider_constraints;

#[derive(Clone)]
pub(crate) struct SealedRequestPolicy {
    pub tools: Arc<[ToolSpec]>,
    pub tool_mode: ToolMode,
    pub max_tokens: u32,
    pub envelope: Arc<hi_ai::RequestToolEnvelope>,
    pub execution_envelope: Arc<ToolEnvelope>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ToolEnvelopeRequestLimits {
    pub(super) max_output_tokens: u32,
    pub(super) executed_tool_calls: u32,
    pub(super) max_parallel_calls: usize,
    pub(super) max_tool_argument_bytes: u32,
}

/// Drop coordination/bookkeeping tools when the one-shot suppress flag is set,
/// unless that would leave the list empty under a required tool choice.
pub(super) fn apply_bookkeeping_suppress(
    tools: Arc<[ToolSpec]>,
    suppress: bool,
) -> Arc<[ToolSpec]> {
    if !suppress {
        return tools;
    }
    if tools
        .iter()
        .any(|tool| !hi_tools::is_coordination(&tool.name))
    {
        tools
            .iter()
            .filter(|tool| !hi_tools::is_coordination(&tool.name))
            .cloned()
            .collect::<Vec<_>>()
            .into()
    } else {
        // Cannot suppress — would empty the list.
        tools
    }
}

/// Track advertised names and peak schema tokens for telemetry.
pub(super) fn note_advertised_tools(
    tools: &[ToolSpec],
    advertised: &mut BTreeSet<String>,
    tool_schema_tokens: &mut u64,
) -> u64 {
    advertised.extend(tools.iter().map(|tool| tool.name.clone()));
    let tokens = hi_ai::estimate_tool_schema_tokens(tools);
    *tool_schema_tokens = (*tool_schema_tokens).max(tokens);
    tokens
}

/// Planning needs its state-transition tool even when task-aware selection
/// otherwise chooses a read-only inspection subset.
pub(super) fn ensure_plan_tool(tools: Arc<[ToolSpec]>, plan_mode: bool) -> Arc<[ToolSpec]> {
    if !plan_mode || tools.iter().any(|tool| tool.name == "update_plan") {
        return tools;
    }
    let Some(plan) = hi_tools::TOOL_SPECS
        .iter()
        .find(|tool| tool.name == "update_plan")
    else {
        return tools;
    };
    let mut selected = Vec::with_capacity(tools.len() + 1);
    selected.push(plan.clone());
    selected.extend(tools.iter().cloned());
    selected.into()
}

impl crate::Agent {
    pub(super) fn validate_and_audit_request_envelope(
        &mut self,
        request: &hi_ai::ChatRequest,
        execution: &ToolEnvelope,
    ) -> anyhow::Result<()> {
        let attached = request.tool_envelope.as_ref().ok_or_else(|| {
            anyhow::anyhow!("provider request is missing its sealed tool envelope")
        })?;
        let expected =
            serde_json::to_value(&execution.payload).expect("tool envelope payload serializes");
        anyhow::ensure!(
            attached.digest == execution.digest && attached.payload == expected,
            "provider request tool envelope does not match the execution envelope"
        );
        self.report
            .last_turn_telemetry
            .record_wire_audit(serde_json::json!({
                "provider": "request_admission",
                "route": self.config.routing.provider_route,
                "model": request.model,
                "request_attempt": request.retry_attempt.saturating_add(1),
                "tool_count": request.tools.len(),
                "tool_envelope_digest": attached.digest,
                "tool_envelope": attached.payload,
            }));
        Ok(())
    }

    pub(crate) async fn seal_chat_only_auxiliary_request(
        &mut self,
        model: &str,
        max_tokens: u32,
    ) -> SealedRequestPolicy {
        self.seal_auxiliary_request(model, Arc::new([]), ToolMode::ChatOnly, max_tokens, 1)
            .await
    }

    /// Seal an auxiliary request after applying the same conservative provider
    /// limits as the primary turn. Callers must use every returned field; the
    /// envelope describes this exact shaped request, not the pre-shape intent.
    pub(crate) async fn seal_auxiliary_request(
        &mut self,
        model: &str,
        tools: Arc<[ToolSpec]>,
        tool_mode: ToolMode,
        max_tokens: u32,
        max_parallel_calls: usize,
    ) -> SealedRequestPolicy {
        let provider_capabilities = self.effective_provider_capabilities_for_model(model).await;
        let shape = provider_constraints::constrain(
            tools,
            tool_mode,
            max_tokens,
            max_parallel_calls,
            &provider_capabilities.capabilities,
        );
        let tool_envelope = self.build_tool_envelope(
            &shape.tools,
            shape.tool_mode,
            shape.envelope_limits(shape.max_output_tokens, 0),
            provider_capabilities,
        );
        SealedRequestPolicy {
            tools: shape.tools,
            tool_mode: shape.tool_mode,
            max_tokens: shape.max_output_tokens,
            envelope: request_envelope(&tool_envelope),
            execution_envelope: tool_envelope,
        }
    }

    /// Refresh mutable host-owned resource aliases before the model can issue a
    /// `read` call. The reference is session-scoped and the body is bounded by
    /// the shared resource cache; an oversized transcript simply remains
    /// unavailable rather than bypassing read limits.
    pub(super) fn refresh_session_resource(&self) {
        let Ok(uri) = hi_workspace::ResourceUri::parse("session://current/transcript") else {
            return;
        };
        let Ok(body) = serde_json::to_string_pretty(self.messages.as_slice()) else {
            return;
        };
        if let Ok(mut cache) = self.runtime.read_cache().lock() {
            let _ = cache.register_resource(uri, body);
        }
    }

    /// Seal the exact post-selection tool list and the authority/policy facts
    /// that govern its execution. The returned value is shared unchanged with
    /// the model response's execution phase.
    pub(super) fn build_tool_envelope(
        &self,
        tools: &[ToolSpec],
        tool_mode: ToolMode,
        request_limits: ToolEnvelopeRequestLimits,
        provider_capabilities: hi_ai::EffectiveProviderCapabilities,
    ) -> Arc<ToolEnvelope> {
        let sandbox = self.runtime.process_runner().sandbox_policy();
        let mut permissions = BTreeSet::from([
            format!("permission_mode:{}", self.permission_mode.as_str()),
            format!("sandbox:{}", sandbox_label(sandbox)),
            format!(
                "network:{}",
                if sandbox.restricts_network() {
                    "restricted"
                } else {
                    "allowed"
                }
            ),
            format!("dry_run:{}", self.config.gates.dry_run),
        ]);
        if self.config.gates.confirm_edits {
            permissions.insert("writes:confirmation_required".to_string());
        }
        let remaining_calls = self
            .config
            .loop_limits
            .remaining_tool_calls(request_limits.executed_tool_calls)
            .min(u32::from(u16::MAX)) as u16;
        let context = ToolEnvelopeContext {
            provider: ProviderEnvelope::from_capability_record(provider_capabilities),
            workspace: WorkspaceEnvelope::from(&self.workspace_controller_binding()),
            trust: workspace_trust(self.runtime.root()),
            permissions,
            limits: ToolEnvelopeLimits {
                max_output_tokens: request_limits.max_output_tokens,
                max_parallel_calls: request_limits
                    .max_parallel_calls
                    .max(1)
                    .min(usize::from(u16::MAX)) as u16,
                max_calls_per_round: remaining_calls,
                max_inline_output_bytes: hi_tools::envelope::TOOL_ENVELOPE_MAX_INLINE_OUTPUT_BYTES,
                max_tool_argument_bytes: request_limits.max_tool_argument_bytes,
            },
            tool_mode,
            tool_versions: BTreeMap::new(),
        };
        let mut program_tools = Vec::new();
        if tools.iter().any(|tool| tool.name == "run_program") {
            program_tools.extend(hi_tools::TOOL_SPECS.iter().cloned());
            for tool in tools {
                if tool.name != "run_program"
                    && !program_tools.iter().any(|known| known.name == tool.name)
                {
                    program_tools.push(tool.clone());
                }
            }
        }
        Arc::new(ToolEnvelope::build_with_program_tools(
            tools,
            &program_tools,
            context,
        ))
    }
}

fn workspace_trust(root: &std::path::Path) -> WorkspaceTrust {
    if hi_tools::folder_trust::folder_trust_inert() {
        return WorkspaceTrust::OperatorOverride;
    }
    let key = hi_tools::folder_trust::workspace_key(root);
    let inputs = hi_tools::folder_trust::decide_inputs_with_interactive(root, &key, false);
    match hi_tools::folder_trust::decide(true, &inputs) {
        hi_tools::folder_trust::TrustOutcome::Trusted => WorkspaceTrust::Trusted,
        hi_tools::folder_trust::TrustOutcome::Untrusted
        | hi_tools::folder_trust::TrustOutcome::Prompt => WorkspaceTrust::Untrusted,
    }
}

fn sandbox_label(policy: hi_tools::sandbox::SandboxPolicy) -> &'static str {
    match policy {
        hi_tools::sandbox::SandboxPolicy::Off => "off",
        hi_tools::sandbox::SandboxPolicy::Workspace => "workspace",
        hi_tools::sandbox::SandboxPolicy::Strict => "strict",
        hi_tools::sandbox::SandboxPolicy::ReadOnly => "read_only",
    }
}

pub(super) fn attach_tool_envelope_audit(
    object: &mut serde_json::Map<String, serde_json::Value>,
    envelope: &ToolEnvelope,
) {
    object.insert(
        "tool_envelope_digest".to_string(),
        serde_json::Value::String(envelope.digest.clone()),
    );
    object.insert(
        "tool_envelope".to_string(),
        serde_json::to_value(&envelope.payload).expect("tool envelope payload serializes"),
    );
}

pub(super) fn request_envelope(envelope: &ToolEnvelope) -> Arc<hi_ai::RequestToolEnvelope> {
    Arc::new(hi_ai::RequestToolEnvelope {
        digest: envelope.digest.clone(),
        payload: serde_json::to_value(&envelope.payload).expect("tool envelope payload serializes"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppress_noop_when_flag_clear() {
        let tools: Arc<[ToolSpec]> = Arc::from([ToolSpec {
            name: "bash".into(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }]);
        let out = apply_bookkeeping_suppress(tools.clone(), false);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn plan_mode_restores_update_plan_to_a_narrow_read_catalog() {
        let tools: Arc<[ToolSpec]> = Arc::from([ToolSpec {
            name: "read".into(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }]);
        let selected = ensure_plan_tool(tools, true);
        assert_eq!(selected[0].name, "update_plan");
        assert_eq!(selected[1].name, "read");
    }
}
