//! Challenges for missing requested mutations and tests.
//! The configured verifier owns ordinary post-edit checks.
use crate::steering::{
    IMPLEMENTATION_NO_CHANGES_NUDGE, ImplementationIntent, ImplementationTracker,
    REQUESTED_VALIDATION_NUDGE,
};

pub(super) const IMPLEMENTATION_COMPLETENESS_CASCADE: &[ImplementationGate] = &[
    ImplementationGate::NoChanges,
    ImplementationGate::RequestedValidation,
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ImplementationGate {
    NoChanges,
    RequestedValidation,
}
#[derive(Debug)]
pub(super) enum ImplementationCascadeAction {
    Repair {
        gate: ImplementationGate,
        status: &'static str,
        nudge_body: String,
        force_tools: bool,
        text_tool_fallback: bool,
    },
    Exhausted {
        gate: ImplementationGate,
        status: &'static str,
    },
}
pub(super) fn select_implementation_completeness(
    implementation_intent: Option<ImplementationIntent>,
    expected_mutation: bool,
    requested_validation: bool,
    finished_text_answer: bool,
    tracker: &ImplementationTracker,
) -> Option<ImplementationCascadeAction> {
    if !finished_text_answer && implementation_intent.is_none() {
        return None;
    }
    for &gate in IMPLEMENTATION_COMPLETENESS_CASCADE {
        let (applies, used, status, body) = match gate {
            ImplementationGate::NoChanges => (
                (implementation_intent.is_some() || expected_mutation)
                    && !tracker.mutation_seen
                    && !tracker.dry_run_mutation_planned,
                tracker.no_change_nudges,
                "implementation request made no file changes; requesting an edit or explanation",
                IMPLEMENTATION_NO_CHANGES_NUDGE,
            ),
            ImplementationGate::RequestedValidation => (
                requested_validation && !tracker.tests_seen,
                tracker.requested_validation_nudges,
                "requested tests have not passed; requesting the test run",
                REQUESTED_VALIDATION_NUDGE,
            ),
        };
        if !applies {
            continue;
        }
        return Some(if used < 2 {
            // Valid narration is not a protocol error. Keep structured tools
            // available and allow an explanation when execution is unavailable.
            ImplementationCascadeAction::Repair {
                gate,
                status,
                nudge_body: body.into(),
                force_tools: false,
                text_tool_fallback: false,
            }
        } else {
            ImplementationCascadeAction::Exhausted { gate, status }
        });
    }
    None
}
pub(super) fn spend_implementation_gate(
    gate: ImplementationGate,
    tracker: &mut ImplementationTracker,
) {
    match gate {
        ImplementationGate::NoChanges => tracker.challenge_edit(true),
        ImplementationGate::RequestedValidation => tracker.requested_validation_nudges += 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_change_challenge_allows_an_answer_without_protocol_fallback() {
        for no_change_nudges in 0..2 {
            let tracker = ImplementationTracker {
                no_change_nudges,
                ..ImplementationTracker::default()
            };
            assert!(matches!(
                select_implementation_completeness(None, true, false, true, &tracker),
                Some(ImplementationCascadeAction::Repair {
                    gate: ImplementationGate::NoChanges,
                    force_tools: false,
                    text_tool_fallback: false,
                    ..
                })
            ));
        }
    }

    #[test]
    fn cascade_order_is_mutation_then_requested_tests() {
        assert_eq!(
            IMPLEMENTATION_COMPLETENESS_CASCADE,
            &[
                ImplementationGate::NoChanges,
                ImplementationGate::RequestedValidation
            ]
        );
    }

    #[test]
    fn no_mutation_selects_no_changes() {
        let tracker = ImplementationTracker::default();
        let action = select_implementation_completeness(
            Some(ImplementationIntent { tui: false }),
            false,
            false,
            true,
            &tracker,
        );
        assert!(matches!(
            action,
            Some(ImplementationCascadeAction::Repair {
                gate: ImplementationGate::NoChanges,
                ..
            })
        ));
    }

    #[test]
    fn explicit_mutation_without_implementation_intent_still_selects_no_changes() {
        let tracker = ImplementationTracker::default();
        let action = select_implementation_completeness(None, true, false, true, &tracker);
        assert!(matches!(
            action,
            Some(ImplementationCascadeAction::Repair {
                gate: ImplementationGate::NoChanges,
                ..
            })
        ));
    }

    #[test]
    fn structured_incomplete_turn_defers_to_plan_continue() {
        let tracker = ImplementationTracker::default();
        assert!(
            select_implementation_completeness(None, true, false, false, &tracker).is_none(),
            "an incomplete structured plan must not be hijacked into no-change repair"
        );
    }

    #[test]
    fn tool_using_expected_mutation_still_requires_no_change_gate() {
        let tracker = ImplementationTracker::default();
        assert!(matches!(
            select_implementation_completeness(None, true, false, true, &tracker),
            Some(ImplementationCascadeAction::Repair {
                gate: ImplementationGate::NoChanges,
                ..
            }),
        ));
    }

    #[test]
    fn explicit_mutation_after_edit_skips_scaffold_and_validation_gates() {
        let tracker = ImplementationTracker {
            mutation_seen: true,
            validation_after_last_mutation: false,
            ..Default::default()
        };
        assert!(
            select_implementation_completeness(None, true, false, true, &tracker).is_none(),
            "ordinary fix turns must not demand post-edit validation repair"
        );
    }

    #[test]
    fn plain_non_mutation_turn_skips_cascade() {
        let tracker = ImplementationTracker::default();
        assert!(select_implementation_completeness(None, false, false, true, &tracker).is_none());
    }

    #[test]
    fn dry_run_mutation_plan_satisfies_no_change_without_claiming_an_edit() {
        let tracker = ImplementationTracker {
            dry_run_mutation_planned: true,
            ..Default::default()
        };

        assert!(
            select_implementation_completeness(
                Some(ImplementationIntent { tui: false }),
                true,
                false,
                true,
                &tracker,
            )
            .is_none()
        );
        assert!(!tracker.mutation_seen);
    }

    #[test]
    fn requested_validation_requires_observed_success_without_mutation() {
        let tracker = ImplementationTracker::default();
        assert!(matches!(
            select_implementation_completeness(None, false, true, true, &tracker),
            Some(ImplementationCascadeAction::Repair {
                gate: ImplementationGate::RequestedValidation,
                ..
            })
        ));

        let tracker = ImplementationTracker {
            tests_seen: true,
            ..Default::default()
        };
        assert!(select_implementation_completeness(None, false, true, true, &tracker).is_none());
    }
}
