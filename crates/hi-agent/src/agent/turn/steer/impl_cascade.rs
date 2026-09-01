//! Table-driven implementation completeness gates (text-only Steer).
//!
//! Separate from [`super::cascade`] (review quality): different counters,
//! budgets (hardcoded ×2), and force-flag semantics.

use crate::steering::{
    IMPLEMENTATION_NO_CHANGES_NUDGE, IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE, ImplementationIntent,
    ImplementationTracker, REQUESTED_VALIDATION_NUDGE, implementation_missing_validation_nudge,
    implementation_text_tool_nudge,
};

/// Ordered implementation completeness steps after unfinished/plan gates.
pub(super) const IMPLEMENTATION_COMPLETENESS_CASCADE: &[ImplementationGate] = &[
    ImplementationGate::NoChanges,
    ImplementationGate::ScaffoldOnly,
    ImplementationGate::MissingValidation,
];

const EXPECTED_MUTATION_WITH_VALIDATION: &[ImplementationGate] = &[
    ImplementationGate::NoChanges,
    ImplementationGate::RequestedValidation,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ImplementationGate {
    NoChanges,
    ScaffoldOnly,
    MissingValidation,
    RequestedValidation,
}

impl ImplementationGate {
    pub(super) fn budget(self) -> u32 {
        2
    }

    fn counter(self, tracker: &ImplementationTracker) -> u32 {
        match self {
            Self::NoChanges => tracker.no_change_nudges,
            Self::ScaffoldOnly => tracker.scaffold_only_nudges,
            Self::MissingValidation => tracker.missing_validation_nudges,
            Self::RequestedValidation => tracker.requested_validation_nudges,
        }
    }

    fn bump(self, tracker: &mut ImplementationTracker) {
        match self {
            Self::NoChanges => tracker.no_change_nudges += 1,
            Self::ScaffoldOnly => tracker.scaffold_only_nudges += 1,
            Self::MissingValidation => tracker.missing_validation_nudges += 1,
            Self::RequestedValidation => tracker.requested_validation_nudges += 1,
        }
    }
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
        status: &'static str,
    },
}

/// First matching implementation completeness gate, or `None` if satisfied.
///
/// Structured implementation tasks (`/build`, keep-building, …) run the full
/// cascade (no-change → scaffold-only → missing-validation).
///
/// Ordinary explicit mutation turns (`expected_mutation`, e.g. "fix the parser
/// bug") get the no-change gate whenever the model presents a finished answer,
/// including after it used read/fetch/wait tools. This is important because a
/// model can inspect evidence and then claim completion without editing; that
/// must receive an edit-or-explain challenge before the turn can settle.
/// Incomplete structured plans keep their bounded plan-continue path. Scaffold and implicit post-mutation
/// validation remain implementation-only; an explicit request to run tests or
/// checks gets its own validation gate without inventing a mutation obligation.
pub(super) fn select_implementation_completeness(
    implementation_intent: Option<ImplementationIntent>,
    expected_mutation: bool,
    requested_validation: bool,
    finished_text_answer: bool,
    tracker: &ImplementationTracker,
) -> Option<ImplementationCascadeAction> {
    let gates: &[ImplementationGate] = if implementation_intent.is_some() {
        IMPLEMENTATION_COMPLETENESS_CASCADE
    } else if expected_mutation && requested_validation && finished_text_answer {
        EXPECTED_MUTATION_WITH_VALIDATION
    } else if expected_mutation && finished_text_answer {
        &[ImplementationGate::NoChanges]
    } else if requested_validation && finished_text_answer {
        &[ImplementationGate::RequestedValidation]
    } else {
        return None;
    };
    for &gate in gates {
        if let Some(action) = evaluate_gate(gate, tracker) {
            return Some(action);
        }
    }
    None
}

fn evaluate_gate(
    gate: ImplementationGate,
    tracker: &ImplementationTracker,
) -> Option<ImplementationCascadeAction> {
    let applies = match gate {
        ImplementationGate::NoChanges => {
            !tracker.mutation_seen && !tracker.dry_run_mutation_planned
        }
        ImplementationGate::ScaffoldOnly => tracker.mutation_seen && !tracker.substantive_edit_seen,
        ImplementationGate::MissingValidation => {
            tracker.mutation_seen && !tracker.validation_after_last_mutation
        }
        ImplementationGate::RequestedValidation => !tracker.validation_seen,
    };
    if !applies {
        return None;
    }
    let used = gate.counter(tracker);
    if used < gate.budget() {
        let next = used + 1;
        let use_text_fallback = next >= gate.budget();
        let (status, body) = match gate {
            ImplementationGate::NoChanges => (
                "implementation answer had no file changes; nudging the model to edit or scaffold",
                IMPLEMENTATION_NO_CHANGES_NUDGE.to_string(),
            ),
            ImplementationGate::ScaffoldOnly => (
                "implementation only scaffolded setup files; nudging the model to edit source files",
                IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE.to_string(),
            ),
            ImplementationGate::MissingValidation => (
                "implementation changed files without validation; nudging the model to run tests or build",
                implementation_missing_validation_nudge(tracker),
            ),
            ImplementationGate::RequestedValidation => (
                "requested validation did not run; nudging the model to execute it",
                REQUESTED_VALIDATION_NUDGE.to_string(),
            ),
        };
        let nudge_body = if use_text_fallback {
            implementation_text_tool_nudge(&body)
        } else {
            body
        };
        Some(ImplementationCascadeAction::Repair {
            gate,
            status,
            nudge_body,
            force_tools: !use_text_fallback,
            text_tool_fallback: use_text_fallback,
        })
    } else {
        let status = match gate {
            ImplementationGate::NoChanges => {
                "implementation still had no file changes after repair"
            }
            ImplementationGate::ScaffoldOnly => {
                "implementation still only had scaffold/setup changes after repair"
            }
            ImplementationGate::MissingValidation => {
                "implementation still lacked validation after repair"
            }
            ImplementationGate::RequestedValidation => {
                "requested validation still did not run after repair"
            }
        };
        Some(ImplementationCascadeAction::Exhausted { status })
    }
}

/// Apply counter bump when spending a repair action.
pub(super) fn spend_implementation_gate(
    gate: ImplementationGate,
    tracker: &mut ImplementationTracker,
) {
    gate.bump(tracker);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_order_is_no_change_scaffold_validation() {
        assert_eq!(
            IMPLEMENTATION_COMPLETENESS_CASCADE,
            &[
                ImplementationGate::NoChanges,
                ImplementationGate::ScaffoldOnly,
                ImplementationGate::MissingValidation,
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
            substantive_edit_seen: true,
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
            validation_seen: true,
            ..Default::default()
        };
        assert!(select_implementation_completeness(None, false, true, true, &tracker).is_none());
    }
}
