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
    /// The sealed envelope or its execution schema no longer matches itself.
    EnvelopeIntegrity,
    /// The admitted tool name was valid, but its call payload was not.
    InvalidArguments,
}

impl ToolProtocolFailureKind {
    pub(in crate::agent::turn) const fn code(self) -> &'static str {
        match self {
            Self::UnavailableTool => "unavailable_tool",
            Self::EnvelopeIntegrity => "envelope_integrity",
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

pub(super) fn validate_sealed_tool_call(
    envelope_error: Option<&str>,
    batch_error: Option<&str>,
    id: &str,
    name: &str,
    arguments: &str,
    specs: &[hi_ai::ToolSpec],
    envelope: &hi_tools::envelope::ToolEnvelope,
) -> Result<(), ToolProtocolFailure> {
    let (message, kind) = if let Some(error) = envelope_error {
        (
            error.to_string(),
            ToolProtocolFailureKind::EnvelopeIntegrity,
        )
    } else if !envelope.admits(name) {
        let mode_detail = matches!(envelope.payload.tool_mode, hi_ai::ToolMode::ChatOnly)
            .then_some("; envelope mode is chat_only and admits no executable tools")
            .unwrap_or_default();
        (
            format!(
                "tool `{name}` is outside the model request's sealed envelope {}{mode_detail}",
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
    /// prevents unavailable tools and envelope faults from being misdiagnosed
    /// as correctable JSON-schema errors.
    pub(in crate::agent::turn) protocol_validation_errors: Vec<ToolProtocolFailure>,
    /// Exact executable subset of the request catalog. `ChatOnly` therefore
    /// records an empty slice even when schemas remain attached for cache/audit.
    pub(in crate::agent::turn) admitted_tool_names: Vec<String>,
    /// Background handles named by the model this batch that the registry has
    /// never seen, most recent first.
    pub(in crate::agent::turn) unknown_background_handles: Vec<hi_tools::UnknownBackgroundHandle>,
    /// The one ordinary-tool recovery was already consumed and a second
    /// rejected program was received. The turn loop must use its typed error
    /// path instead of allowing an unbounded program/fallback cycle.
    pub(in crate::agent::turn) program_fallback_exhausted: bool,
}
