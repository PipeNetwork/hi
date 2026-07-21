//! Table-driven implementation completeness gates (text-only Steer).
//!
//! Separate from [`super::cascade`] (review quality): different counters,
//! budgets (hardcoded ×2), and force-flag semantics.

use crate::steering::{
    IMPLEMENTATION_NO_CHANGES_NUDGE, IMPLEMENTATION_SCAFFOLD_ONLY_NUDGE, ImplementationIntent,
    ImplementationTracker, implementation_missing_validation_nudge, implementation_text_tool_nudge,
};

/// Ordered implementation completeness steps after unfinished/plan gates.
pub(super) const IMPLEMENTATION_COMPLETENESS_CASCADE: &[ImplementationGate] = &[
    ImplementationGate::NoChanges,
    ImplementationGate::ScaffoldOnly,
    ImplementationGate::MissingValidation,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ImplementationGate {
    NoChanges,
    ScaffoldOnly,
    MissingValidation,
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
        }
    }

    fn bump(self, tracker: &mut ImplementationTracker) {
        match self {
            Self::NoChanges => tracker.no_change_nudges += 1,
            Self::ScaffoldOnly => tracker.scaffold_only_nudges += 1,
            Self::MissingValidation => tracker.missing_validation_nudges += 1,
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
/// bug") only get the no-change gate when the text answer looks finished *and*
/// the turn never used tools (`text_only_turn`). That is the live failure:
/// pure diagnosis text → "verification skipped — no files changed" →
/// `incomplete · stalled` with no edit attempt. Unfinished narration, incomplete
/// plans, and turns that already used tools (plan/read/wait) keep their existing
/// silent-continue / plan-continue paths. Scaffold/validation remain
/// implementation-only so a normal fix that lands an edit is not forced into a
/// validation loop.
pub(super) fn select_implementation_completeness(
    implementation_intent: Option<ImplementationIntent>,
    expected_mutation: bool,
    finished_text_answer: bool,
    text_only_turn: bool,
    tracker: &ImplementationTracker,
) -> Option<ImplementationCascadeAction> {
    let gates: &[ImplementationGate] = if implementation_intent.is_some() {
        IMPLEMENTATION_COMPLETENESS_CASCADE
    } else if expected_mutation && finished_text_answer && text_only_turn {
        &[ImplementationGate::NoChanges]
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
        ImplementationGate::NoChanges => !tracker.mutation_seen,
        ImplementationGate::ScaffoldOnly => {
            tracker.mutation_seen && !tracker.substantive_edit_seen
        }
        ImplementationGate::MissingValidation => {
            tracker.mutation_seen && !tracker.validation_after_last_mutation
        }
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
            true,
            false,
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
        let action = select_implementation_completeness(None, true, true, true, &tracker);
        assert!(matches!(
            action,
            Some(ImplementationCascadeAction::Repair {
                gate: ImplementationGate::NoChanges,
                ..
            })
        ));
    }

    #[test]
    fn unfinished_expected_mutation_defers_to_silent_continue() {
        let tracker = ImplementationTracker::default();
        assert!(
            select_implementation_completeness(None, true, false, true, &tracker).is_none(),
            "plan/unfinished narration must not be hijacked into no-change repair"
        );
    }

    #[test]
    fn tool_using_expected_mutation_skips_text_only_no_change_gate() {
        let tracker = ImplementationTracker::default();
        assert!(
            select_implementation_completeness(None, true, true, false, &tracker).is_none(),
            "plan/read/wait turns already used tools; do not force edit via text cascade"
        );
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
            select_implementation_completeness(None, true, true, true, &tracker).is_none(),
            "ordinary fix turns must not demand post-edit validation repair"
        );
    }

    #[test]
    fn plain_non_mutation_turn_skips_cascade() {
        let tracker = ImplementationTracker::default();
        assert!(select_implementation_completeness(None, false, true, true, &tracker).is_none());
    }
}
