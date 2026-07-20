//! Agent-facing UI and transcript handling for bounded mutation discovery.

use crate::heuristics::plan_has_pending_steps;
use crate::steering::{
    DiscoveryRecovery, EvidenceTracker, IMPLEMENTATION_NO_CHANGES_NUDGE, ImplementationTracker,
    MutationRecovery,
};
use crate::transcript::NudgeKind;
use crate::{Agent, Ui};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MutationRecoveryControl {
    None,
    Continue,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn handle_mutation_recovery(
        &mut self,
        recovery: &mut MutationRecovery,
        expected_mutation: bool,
        tracker: &mut ImplementationTracker,
        evidence: &mut EvidenceTracker,
        plan_changed: bool,
        force_tools_next: &mut bool,
        ui: &mut dyn Ui,
    ) -> MutationRecoveryControl {
        if tracker.mutation_seen {
            return MutationRecoveryControl::None;
        }
        if !expected_mutation {
            return MutationRecoveryControl::None;
        }
        let has_pending_plan = plan_has_pending_steps(&self.goals.last_plan);
        if recovery.transition_after_plan(tracker, plan_changed, has_pending_plan) {
            *force_tools_next = true;
            ui.nudge(
                "implementation plan recorded after bounded discovery; starting its active step",
            );
            self.messages.push_nudge(
                NudgeKind::Continue,
                "The implementation plan is now concrete. Begin its active step immediately. Prefer an edit now; if one more read is genuinely necessary to make the edit safely, perform it and then edit without reopening broad discovery.",
            );
            return MutationRecoveryControl::Continue;
        }
        match recovery.after_discovery(tracker, has_pending_plan) {
            DiscoveryRecovery::None => MutationRecoveryControl::None,
            DiscoveryRecovery::ExistingPlan => {
                *force_tools_next = true;
                ui.nudge("active implementation plan already exists; resuming its active step");
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    "An implementation plan with an active or pending step already exists. Begin that step now. Prefer an edit; use further reads only when needed to make the edit safely, not to reopen broad discovery.",
                );
                MutationRecoveryControl::Continue
            }
            DiscoveryRecovery::PlanNudge => {
                *force_tools_next = true;
                ui.nudge("implementation plan got its final read round; requesting the edit now");
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    "The additional read completed successfully. Act on that evidence now: edit the workspace, or report a concrete blocker if the planned edit cannot be made. Do not reopen broad discovery.",
                );
                MutationRecoveryControl::Continue
            }
            DiscoveryRecovery::Nudge { attempt, maximum } => {
                evidence.quality_repair_nudges = evidence.quality_repair_nudges.saturating_add(1);
                *force_tools_next = true;
                ui.nudge(&format!(
                    "mutation request used {} model rounds ({} tools) without editing; requesting an implementation step ({attempt}/{maximum})",
                    tracker.pre_mutation_rounds, tracker.pre_mutation_tool_calls,
                ));
                self.messages.push_nudge(
                    NudgeKind::Continue,
                    format!(
                        "{IMPLEMENTATION_NO_CHANGES_NUDGE}\n\nYou have already used {} model tool rounds ({} individual tools) without editing. Stop open-ended inspection. Record a concrete active implementation plan or attempt the requested workspace change now. A narrowly targeted read remains available when it is genuinely necessary to make the edit safely; if the evidence is still insufficient, report the concrete blocker.",
                        tracker.pre_mutation_rounds, tracker.pre_mutation_tool_calls,
                    ),
                );
                MutationRecoveryControl::Continue
            }
        }
    }
}
