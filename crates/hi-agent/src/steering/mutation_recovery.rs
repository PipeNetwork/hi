//! Bounded discovery recovery for mutation turns.

use super::{
    ImplementationTracker, MAX_MUTATION_DISCOVERY_NUDGES, MUTATION_DISCOVERY_ROUND_CAP,
    MUTATION_DISCOVERY_ROUNDS_PER_NUDGE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiscoveryRecovery {
    None,
    ExistingPlan,
    PlanNudge,
    Nudge { attempt: u32, maximum: u32 },
}

#[derive(Debug, Default)]
pub(crate) struct MutationRecovery {
    phase_nudges: u32,
    plan_grace_used: bool,
    plan_nudge_sent: bool,
}

impl MutationRecovery {
    pub(crate) fn transition_after_plan(
        &mut self,
        tracker: &ImplementationTracker,
        plan_changed: bool,
        has_pending_plan: bool,
    ) -> bool {
        let discovery_started = tracker.discovery_nudges > 0
            || tracker.pre_mutation_rounds >= MUTATION_DISCOVERY_ROUND_CAP;
        if !plan_changed || !has_pending_plan || !discovery_started || self.plan_grace_used {
            return false;
        }
        self.start_plan_phase();
        true
    }

    pub(crate) fn after_discovery(
        &mut self,
        tracker: &mut ImplementationTracker,
        has_pending_plan: bool,
    ) -> DiscoveryRecovery {
        // A concrete/resumed plan gets one stronger advisory after its next
        // non-mutating round. Productive inspection remains non-terminal; the
        // common repeat/no-progress guards and model-call cap provide bounds.
        if self.plan_grace_used {
            if !self.plan_nudge_sent {
                self.plan_nudge_sent = true;
                return DiscoveryRecovery::PlanNudge;
            }
            return DiscoveryRecovery::None;
        }
        let limit = MUTATION_DISCOVERY_ROUND_CAP.saturating_add(
            self.phase_nudges
                .saturating_mul(MUTATION_DISCOVERY_ROUNDS_PER_NUDGE),
        );
        if tracker.pre_mutation_rounds < limit {
            return DiscoveryRecovery::None;
        }
        if has_pending_plan && !self.plan_grace_used {
            self.start_plan_phase();
            return DiscoveryRecovery::ExistingPlan;
        }
        if self.phase_nudges < MAX_MUTATION_DISCOVERY_NUDGES {
            self.phase_nudges += 1;
            tracker.discovery_nudges += 1;
            return DiscoveryRecovery::Nudge {
                attempt: self.phase_nudges,
                maximum: MAX_MUTATION_DISCOVERY_NUDGES,
            };
        }
        DiscoveryRecovery::None
    }

    fn start_plan_phase(&mut self) {
        self.plan_grace_used = true;
        self.plan_nudge_sent = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_budget_is_two_advisory_nudges_then_continues() {
        let mut recovery = MutationRecovery::default();
        let mut tracker = ImplementationTracker {
            pre_mutation_rounds: 10,
            pre_mutation_tool_calls: 10,
            ..Default::default()
        };
        assert_eq!(
            recovery.after_discovery(&mut tracker, false),
            DiscoveryRecovery::Nudge {
                attempt: 1,
                maximum: 2
            }
        );
        tracker.pre_mutation_rounds = 12;
        tracker.pre_mutation_tool_calls = 24;
        assert_eq!(
            recovery.after_discovery(&mut tracker, false),
            DiscoveryRecovery::Nudge {
                attempt: 2,
                maximum: 2
            }
        );
        tracker.pre_mutation_rounds = 14;
        tracker.pre_mutation_tool_calls = 40;
        assert_eq!(
            recovery.after_discovery(&mut tracker, false),
            DiscoveryRecovery::None
        );
        assert_eq!(tracker.discovery_nudges, 2);
    }

    #[test]
    fn plan_at_final_threshold_gets_advisory_without_terminal_stop() {
        let mut recovery = MutationRecovery::default();
        let mut tracker = ImplementationTracker {
            pre_mutation_rounds: 14,
            pre_mutation_tool_calls: 14,
            discovery_nudges: 2,
            ..Default::default()
        };
        assert!(recovery.transition_after_plan(&tracker, true, true));
        tracker.pre_mutation_rounds += 1;
        tracker.pre_mutation_tool_calls += 4;
        assert_eq!(
            recovery.after_discovery(&mut tracker, true),
            DiscoveryRecovery::PlanNudge
        );
        tracker.pre_mutation_rounds += 1;
        tracker.pre_mutation_tool_calls += 3;
        assert_eq!(
            recovery.after_discovery(&mut tracker, true),
            DiscoveryRecovery::None
        );
        assert_eq!(tracker.discovery_nudges, 2);
    }

    #[test]
    fn completed_plan_does_not_enter_plan_grace() {
        let mut recovery = MutationRecovery::default();
        let tracker = ImplementationTracker {
            pre_mutation_rounds: 14,
            discovery_nudges: 2,
            ..Default::default()
        };
        assert!(!recovery.transition_after_plan(&tracker, true, false));
    }
}
