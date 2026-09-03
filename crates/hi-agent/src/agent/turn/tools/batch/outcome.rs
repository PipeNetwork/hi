//! Tool-batch capability events and aggregate outcome state.

use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, RunEvent,
    SemanticActivity,
};
use hi_policy::{capability_is_read_only, capability_kind_for_tool};

use crate::Ui;
use crate::agent::turn::progress::ToolProgressLabel;

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
    /// Calls rejected by client-side JSON-schema validation, retained so the
    /// next model round can receive a concrete correction instead of a generic
    /// repeat nudge.
    pub(in crate::agent::turn) protocol_validation_errors: Vec<(String, String)>,
    /// Background handles named by the model this batch that the registry has
    /// never seen, most recent first.
    pub(in crate::agent::turn) unknown_background_handles: Vec<hi_tools::UnknownBackgroundHandle>,
    /// The one ordinary-tool recovery was already consumed and a second
    /// rejected program was received. The turn loop must use its typed error
    /// path instead of allowing an unbounded program/fallback cycle.
    pub(in crate::agent::turn) program_fallback_exhausted: bool,
}
