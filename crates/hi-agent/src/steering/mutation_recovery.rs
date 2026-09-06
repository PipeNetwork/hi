//! Optional bounded discovery recovery for mutation turns.

use super::ImplementationTracker;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiscoveryRecovery {
    None,
    ExistingPlan,
    PlanNudge,
    Nudge {
        attempt: u32,
        maximum: u32,
    },
    /// Discovery nudges are spent and there is still no mutation. Require an
    /// edit on the next model round.
    ForceEdit,
    /// The force-edit round (or the post-plan extra read) still did not
    /// mutate. Stop the turn instead of inspecting indefinitely.
    Stop,
}

#[derive(Debug)]
pub(crate) struct MutationRecovery {
    /// This bounds only the discovery phase before the first mutation. The
    /// interactive turn itself remains unlimited once it is making progress.
    round_cap: Option<u32>,
    rounds_per_nudge: u32,
    max_nudges: u32,
    phase_nudges: u32,
    plan_grace_used: bool,
    plan_nudge_sent: bool,
    force_edit_sent: bool,
}

impl Default for MutationRecovery {
    fn default() -> Self {
        Self {
            round_cap: Some(10),
            rounds_per_nudge: 2,
            max_nudges: 2,
            phase_nudges: 0,
            plan_grace_used: false,
            plan_nudge_sent: false,
            force_edit_sent: false,
        }
    }
}

impl MutationRecovery {
    /// Whether the next model request must be sealed to mutation-capable tools.
    ///
    /// The recovery state, rather than the nudges/counters mirrored elsewhere,
    /// is authoritative here. In particular, a plan gets one final targeted
    /// read before `plan_nudge_sent` is set. After that read, continuing to
    /// advertise `read`/`grep` makes the model's next inspection legal even
    /// though [`Self::after_discovery`] will immediately stop the turn for it.
    pub(crate) fn requires_mutation_focus(&self) -> bool {
        self.round_cap.is_some()
            && (self.plan_nudge_sent
                || self.force_edit_sent
                || (self.max_nudges > 0 && self.phase_nudges >= self.max_nudges))
    }

    pub(crate) fn transition_after_plan(
        &mut self,
        tracker: &ImplementationTracker,
        plan_changed: bool,
        has_pending_plan: bool,
    ) -> bool {
        let Some(round_cap) = self.round_cap else {
            return false;
        };
        let discovery_started =
            tracker.discovery_nudges > 0 || tracker.pre_mutation_rounds >= round_cap;
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
        let Some(round_cap) = self.round_cap else {
            return DiscoveryRecovery::None;
        };
        // A concrete/resumed plan gets one stronger advisory after its next
        // non-mutating round, then the turn stops if it still has not edited.
        if self.plan_grace_used {
            if !self.plan_nudge_sent {
                self.plan_nudge_sent = true;
                return DiscoveryRecovery::PlanNudge;
            }
            return DiscoveryRecovery::Stop;
        }
        let limit =
            round_cap.saturating_add(self.phase_nudges.saturating_mul(self.rounds_per_nudge));
        if tracker.pre_mutation_rounds < limit {
            return DiscoveryRecovery::None;
        }
        if has_pending_plan && !self.plan_grace_used {
            self.start_plan_phase();
            return DiscoveryRecovery::ExistingPlan;
        }
        if self.phase_nudges < self.max_nudges {
            self.phase_nudges += 1;
            tracker.discovery_nudges += 1;
            return DiscoveryRecovery::Nudge {
                attempt: self.phase_nudges,
                maximum: self.max_nudges,
            };
        }
        if !self.force_edit_sent {
            self.force_edit_sent = true;
            return DiscoveryRecovery::ForceEdit;
        }
        DiscoveryRecovery::Stop
    }

    fn start_plan_phase(&mut self) {
        self.plan_grace_used = true;
        self.plan_nudge_sent = false;
    }

    #[cfg(test)]
    fn bounded_for_test() -> Self {
        Self {
            round_cap: Some(10),
            rounds_per_nudge: 2,
            max_nudges: 2,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_budget_is_two_advisory_nudges_then_force_edit_then_stop() {
        let mut recovery = MutationRecovery::bounded_for_test();
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
            DiscoveryRecovery::ForceEdit
        );
        tracker.pre_mutation_rounds = 15;
        assert_eq!(
            recovery.after_discovery(&mut tracker, false),
            DiscoveryRecovery::Stop
        );
        assert_eq!(tracker.discovery_nudges, 2);
    }

    #[test]
    fn plan_at_final_threshold_stops_after_the_advisory_read() {
        let mut recovery = MutationRecovery::bounded_for_test();
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
            DiscoveryRecovery::Stop
        );
        assert_eq!(tracker.discovery_nudges, 2);
    }

    #[test]
    fn completed_plan_does_not_enter_plan_grace() {
        let mut recovery = MutationRecovery::bounded_for_test();
        let tracker = ImplementationTracker {
            pre_mutation_rounds: 14,
            discovery_nudges: 2,
            ..Default::default()
        };
        assert!(!recovery.transition_after_plan(&tracker, true, false));
    }

    #[test]
    fn production_discovery_is_bounded_before_the_first_mutation() {
        let mut recovery = MutationRecovery::default();
        let mut tracker = ImplementationTracker {
            pre_mutation_rounds: 10,
            pre_mutation_tool_calls: 33,
            ..Default::default()
        };

        assert_eq!(
            recovery.after_discovery(&mut tracker, false),
            DiscoveryRecovery::Nudge {
                attempt: 1,
                maximum: 2
            }
        );
        assert_eq!(tracker.discovery_nudges, 1);
    }

    #[test]
    fn plan_grace_requires_mutation_focus_after_its_one_allowed_read() {
        let mut recovery = MutationRecovery::bounded_for_test();
        let mut tracker = ImplementationTracker {
            pre_mutation_rounds: 10,
            ..Default::default()
        };

        assert_eq!(
            recovery.after_discovery(&mut tracker, true),
            DiscoveryRecovery::ExistingPlan
        );
        assert!(!recovery.requires_mutation_focus());

        tracker.pre_mutation_rounds += 1;
        assert_eq!(
            recovery.after_discovery(&mut tracker, true),
            DiscoveryRecovery::PlanNudge
        );
        assert!(recovery.requires_mutation_focus());
    }
}
