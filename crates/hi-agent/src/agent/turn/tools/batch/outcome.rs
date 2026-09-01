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
    pub(in crate::agent::turn) hash_guard_applies: bool,
    pub(in crate::agent::turn) hashable_idempotent_results: usize,
    pub(in crate::agent::turn) repeated_idempotent_results: usize,
    /// How many results were `bash_output` polls of a still-running process —
    /// with or without new output. A live progress bar produces fresh bytes on
    /// every poll, so waiting-detection keys on the process lifecycle instead
    /// of output novelty.
    pub(in crate::agent::turn) running_background_poll_results: usize,
    /// How many running-process polls delivered failure diagnostics (compiler
    /// errors, test failures, panics) in their fresh output. Those rounds are
    /// new work arriving, not waiting — they reset the wait-streak so the
    /// model can act on the evidence.
    pub(in crate::agent::turn) actionable_poll_results: usize,
    /// How many calls in this batch were wait-flavored: background polls, plan
    /// bookkeeping, or non-mutating non-validating shell probes (log tails,
    /// size checks). A batch of only these while a process runs is a turn
    /// waiting on external work, not making progress toward the plan.
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
    /// never seen, most recent first. Lets the turn loop tell a guessed id
    /// (never real) from a pruned one (a real process was forgotten at
    /// capacity) and steer the model accordingly.
    pub(in crate::agent::turn) unknown_background_handles: Vec<hi_tools::UnknownBackgroundHandle>,
}
