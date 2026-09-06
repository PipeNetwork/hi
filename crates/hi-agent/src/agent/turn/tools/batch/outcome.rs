//! Tool-batch capability events and aggregate outcome state.

use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, RunEvent,
    SemanticActivity,
};
use hi_policy::{capability_is_read_only, capability_kind_for_tool};

use crate::Ui;
use crate::agent::turn::progress::ToolProgressLabel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::agent::turn) enum ToolProtocolFailureKind {
    /// The call named a tool that this exact request did not admit.
    UnavailableTool,
    /// The workspace changed after the provider request was sealed.
    StaleWorkspace,
    /// The admitted tool name was valid, but its call payload was not.
    InvalidArguments,
}

impl ToolProtocolFailureKind {
    pub(in crate::agent::turn) const fn code(self) -> &'static str {
        match self {
            Self::UnavailableTool => "unavailable_tool",
            Self::StaleWorkspace => "stale_workspace",
            Self::InvalidArguments => "invalid_arguments",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::agent::turn) struct ToolProtocolFailure {
    pub(in crate::agent::turn) tool: String,
    pub(in crate::agent::turn) message: String,
    pub(in crate::agent::turn) kind: ToolProtocolFailureKind,
}

impl ToolProtocolFailure {
    pub(super) fn stale_workspace(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            message: message.into(),
            kind: ToolProtocolFailureKind::StaleWorkspace,
        }
    }
}

pub(super) fn tool_protocol_failure_content(failure: &ToolProtocolFailure) -> String {
    serde_json::json!({
        "error": {
            "kind": "tool_protocol_error",
            "reason": failure.kind.code(),
            "message": failure.message,
        }
    })
    .to_string()
}

/// Validate the envelope itself before any policy branch, workspace admission,
/// or executor can observe its payload. Integrity failures are harness errors,
/// not model-correctable tool results, and therefore fail the turn directly.
pub(super) fn validate_sealed_tool_envelope(
    specs: &[hi_ai::ToolSpec],
    envelope: &hi_tools::envelope::ToolEnvelope,
) -> anyhow::Result<()> {
    if !envelope.digest_is_valid() {
        anyhow::bail!("sealed tool envelope integrity failure: digest does not match its payload");
    }
    if !envelope.matches_specs(specs) {
        anyhow::bail!(
            "sealed tool envelope integrity failure: execution schemas do not match the envelope"
        );
    }
    Ok(())
}

/// Compare every authority-bearing workspace field captured for the provider
/// request with the controller's current binding. A valid digest proves only
/// that the old payload was not tampered with; it does not make that payload
/// current after a rebind or a successful intervening settlement.
pub(super) fn sealed_workspace_staleness(
    envelope: &hi_tools::envelope::ToolEnvelope,
    current: &hi_workspace::WorkspaceBinding,
) -> Option<String> {
    crate::workspace_coordination::sealed_workspace_mismatch(&envelope.payload.workspace, current)
        .map(|error| error.to_string())
}

pub(super) fn validate_sealed_tool_call(
    batch_error: Option<&str>,
    id: &str,
    name: &str,
    arguments: &str,
    specs: &[hi_ai::ToolSpec],
    envelope: &hi_tools::envelope::ToolEnvelope,
) -> Result<(), ToolProtocolFailure> {
    let (message, kind) = if !envelope.admits(name) {
        let mode_detail = matches!(envelope.payload.tool_mode, hi_ai::ToolMode::ChatOnly)
            .then_some("; envelope mode is chat_only and admits no executable tools")
            .unwrap_or_default();
        let policy_detail = if matches!(envelope.payload.execution_mode, hi_ai::ToolMode::ReadOnly)
            && crate::steering::implementation_tool_call_mutates(name, arguments)
        {
            format!("Tool `{name}` blocked: this request is sealed for read-only execution; ")
        } else {
            String::new()
        };
        (
            format!(
                "{policy_detail}tool `{name}` is outside the model request's sealed envelope {}{mode_detail}",
                envelope.digest,
            ),
            ToolProtocolFailureKind::UnavailableTool,
        )
    } else if let Some(error) = batch_error {
        (error.to_string(), ToolProtocolFailureKind::InvalidArguments)
    } else {
        return hi_ai::validate_client_tool_call_with_limit(
            id,
            name,
            arguments,
            specs,
            envelope.payload.limits.max_tool_argument_bytes as usize,
        )
        .map_err(|error| ToolProtocolFailure {
            tool: name.to_string(),
            message: error.to_string(),
            kind: ToolProtocolFailureKind::InvalidArguments,
        });
    };
    Err(ToolProtocolFailure {
        tool: name.to_string(),
        message,
        kind,
    })
}

pub(super) fn emit_capability_request(ui: &mut dyn Ui, id: &str, tool: &str) {
    let capability = capability_kind_for_tool(tool);
    if capability_is_read_only(&capability) {
        return;
    }
    let capability_name = serde_json::to_string(&capability)
        .unwrap_or_else(|_| "unknown".to_string())
        .trim_matches('"')
        .to_string();
    ui.semantic_event(RunEvent::new(
        EventKind::CapabilityRequested,
        EventContext {
            correlation_id: Some(id.to_string()),
            ..EventContext::default()
        },
        SemanticActivity {
            verb: ActivityVerb::Request,
            object: ActivityObject::Capability,
            state: ActivityState::Waiting,
            group_key: format!("capability:{id}"),
            title: format!("{capability_name} capability requested"),
            detail: Some(format!("tool {tool}")),
            refs: Vec::new(),
            progress: None,
        },
    ));
}

pub(super) fn append_tool_images(
    output: &hi_tools::ToolOutcome,
    vision: &mut Vec<hi_tools::ToolImage>,
) {
    if !output.images.is_empty() {
        vision.extend(output.images.iter().cloned());
    }
}

/// Outcomes and counters produced by one Tools-phase batch.
pub(in crate::agent::turn) struct ToolBatchOutcome {
    pub(in crate::agent::turn) calls: Vec<(String, String, String)>,
    pub(in crate::agent::turn) read_only_intent: Option<crate::steering::ReviewIntent>,
    pub(in crate::agent::turn) hash_guard_applies: bool,
    pub(in crate::agent::turn) hashable_idempotent_results: usize,
    pub(in crate::agent::turn) repeated_idempotent_results: usize,
    /// Results that polled a still-running process, independent of output
    /// novelty (progress bars otherwise defeat waiting detection).
    pub(in crate::agent::turn) running_background_poll_results: usize,
    /// Running polls that delivered actionable failure diagnostics.
    pub(in crate::agent::turn) actionable_poll_results: usize,
    /// Calls compatible with a round that is only waiting on background work.
    pub(in crate::agent::turn) wait_flavored_results: usize,
    pub(in crate::agent::turn) tool_progress_labels: Vec<ToolProgressLabel>,
    pub(in crate::agent::turn) plan_changed_this_batch: bool,
    pub(in crate::agent::turn) interrupted_calls: usize,
    pub(in crate::agent::turn) interrupted_coordination_calls: usize,
    /// Calls rejected at the sealed client boundary. Keeping the reason typed
    /// prevents unavailable tools from being misdiagnosed as correctable
    /// JSON-schema errors. Envelope integrity faults fail before a batch starts.
    pub(in crate::agent::turn) protocol_validation_errors: Vec<ToolProtocolFailure>,
    /// Exact executable subset of the request catalog. `ChatOnly` therefore
    /// records an empty slice even when schemas remain attached for cache/audit.
    pub(in crate::agent::turn) admitted_tool_names: Vec<String>,
    /// Whether changing an Auto retry to Required preserves provider support.
    pub(in crate::agent::turn) required_tool_choice_supported: bool,
    /// Background handles named by the model this batch that the registry has
    /// never seen, most recent first.
    pub(in crate::agent::turn) unknown_background_handles: Vec<hi_tools::UnknownBackgroundHandle>,
    /// The one ordinary-tool recovery was already consumed and a second
    /// rejected program was received. The turn loop must use its typed error
    /// path instead of allowing an unbounded program/fallback cycle.
    pub(in crate::agent::turn) program_fallback_exhausted: bool,
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use hi_ai::{CapabilityRoute, EffectiveProviderCapabilities, ProviderCapabilities, ToolMode};
    use hi_tools::envelope::{
        ProviderEnvelope, ToolEnvelope, ToolEnvelopeContext, ToolEnvelopeLimits, WorkspaceEnvelope,
        WorkspaceTrust,
    };
    use hi_workspace::{WorkspaceAuthority, WorkspaceVersion};
    use serde_json::json;

    use super::validate_sealed_tool_envelope;

    fn envelope(specs: &[hi_ai::ToolSpec]) -> ToolEnvelope {
        let capabilities = EffectiveProviderCapabilities::conservative(
            CapabilityRoute::new("test", "model"),
            ProviderCapabilities::native_tools(true),
        );
        ToolEnvelope::build(
            specs,
            ToolEnvelopeContext {
                provider: ProviderEnvelope::from_capability_record(capabilities),
                workspace: WorkspaceEnvelope {
                    authority: WorkspaceAuthority::Local,
                    binding_id: "binding".into(),
                    epoch: 0,
                    version: WorkspaceVersion::Unknown,
                },
                trust: WorkspaceTrust::Trusted,
                permissions: BTreeSet::new(),
                limits: ToolEnvelopeLimits::default(),
                tool_mode: ToolMode::Auto,
                execution_mode: ToolMode::Auto,
                tool_versions: BTreeMap::new(),
            },
        )
    }

    fn spec(name: &str) -> hi_ai::ToolSpec {
        hi_ai::ToolSpec {
            name: name.into(),
            description: String::new(),
            parameters: json!({"type": "object"}),
        }
    }

    #[test]
    fn envelope_integrity_fails_before_call_level_recovery() {
        let specs = vec![spec("read")];
        let mut corrupt = envelope(&specs);
        corrupt.digest.push('0');
        assert!(
            validate_sealed_tool_envelope(&specs, &corrupt)
                .unwrap_err()
                .to_string()
                .contains("digest does not match")
        );

        let sealed = envelope(&specs);
        assert!(
            validate_sealed_tool_envelope(&[spec("write")], &sealed)
                .unwrap_err()
                .to_string()
                .contains("execution schemas do not match")
        );
    }
}
