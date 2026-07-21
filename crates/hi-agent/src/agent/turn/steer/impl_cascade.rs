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
pub(super) fn select_implementation_completeness(
    implementation_intent: Option<ImplementationIntent>,
    tracker: &ImplementationTracker,
) -> Option<ImplementationCascadeAction> {
    if implementation_intent.is_none() {
        return None;
    }
    for &gate in IMPLEMENTATION_COMPLETENESS_CASCADE {
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
}
