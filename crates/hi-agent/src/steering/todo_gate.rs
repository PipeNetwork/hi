//! Grok-build turn-end TodoGate: leftover pending / unbacked in-progress
//! checklist items continue the turn with one reminder, up to a per-prompt cap.

/// Why the gate fired. Single variant today, matching grok-build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TodoGateReason {
    InFlight,
}

/// Outcome of [`evaluate_todo_gate`]. The caller owns the per-prompt cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TodoGateDecision {
    Continue,
    Nudge {
        reminder: String,
        reason: TodoGateReason,
    },
}

/// Borrowed checklist snapshot. Pending and unbacked in-progress fire the gate;
/// backed in-progress items (live background work) do not.
#[derive(Clone, Debug, Default)]
pub(crate) struct TodoGateInput<'a> {
    pub pending: Vec<&'a str>,
    pub in_progress_unbacked: Vec<&'a str>,
    pub in_progress_backed: Vec<&'a str>,
    #[allow(dead_code)]
    pub backing_task_count: usize,
}

impl TodoGateReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::InFlight => "in_flight",
        }
    }
}

/// Pure decision. Does not consult the fire cap.
pub(crate) fn evaluate_todo_gate(input: &TodoGateInput<'_>) -> TodoGateDecision {
    if input.pending.is_empty() && input.in_progress_unbacked.is_empty() {
        return TodoGateDecision::Continue;
    }
    TodoGateDecision::Nudge {
        reminder: build_todo_gate_reminder(&input.pending, &input.in_progress_unbacked),
        reason: TodoGateReason::InFlight,
    }
}

pub(crate) fn build_todo_gate_reminder(pending: &[&str], unbacked_in_progress: &[&str]) -> String {
    use std::fmt::Write as _;
    let mut buf =
        String::from("You have outstanding todos but ended your turn without a tool call.\n\n");
    if !unbacked_in_progress.is_empty() {
        buf.push_str("In-progress (no backing background task):\n");
        for title in unbacked_in_progress {
            let _ = writeln!(buf, "- {title}");
        }
        buf.push('\n');
    }
    if !pending.is_empty() {
        buf.push_str("Pending:\n");
        for title in pending {
            let _ = writeln!(buf, "- {title}");
        }
        buf.push('\n');
    }
    buf.push_str(
        "Advance the next pending todo with a tool call NOW. If you have a genuine \
         external blocker, state it explicitly AND mark the affected todos done or \
         drop them via `update_plan` with a reason in the same turn.",
    );
    buf
}

/// Build the gate input from hi's checklist. Active steps count as unbacked
/// unless a live background process is the only remaining work.
pub(crate) fn todo_gate_input_from_plan<'a>(
    steps: &'a [hi_tools::PlanStep],
    awaiting_background: bool,
) -> TodoGateInput<'a> {
    use hi_tools::PlanStatus;
    let mut pending = Vec::new();
    let mut in_progress = Vec::new();
    for step in steps {
        match step.status {
            PlanStatus::Pending => pending.push(step.title.as_str()),
            PlanStatus::Active => in_progress.push(step.title.as_str()),
            PlanStatus::Done => {}
        }
    }
    let backing = if awaiting_background {
        in_progress.len()
    } else {
        0
    };
    let in_progress_backed = in_progress[..backing].to_vec();
    let in_progress_unbacked = in_progress[backing..].to_vec();
    TodoGateInput {
        pending,
        in_progress_unbacked,
        in_progress_backed,
        backing_task_count: backing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_tools::{PlanStatus, PlanStep};

    #[test]
    fn evaluate_todo_gate_fires_on_pending_and_unbacked() {
        let pending = evaluate_todo_gate(&TodoGateInput {
            pending: vec!["Fix reply drop"],
            ..TodoGateInput::default()
        });
        assert!(matches!(
            pending,
            TodoGateDecision::Nudge {
                reason: TodoGateReason::InFlight,
                ..
            }
        ));
        let unbacked = evaluate_todo_gate(&TodoGateInput {
            in_progress_unbacked: vec!["Add TLS"],
            ..TodoGateInput::default()
        });
        assert!(matches!(unbacked, TodoGateDecision::Nudge { .. }));
        let empty = evaluate_todo_gate(&TodoGateInput::default());
        assert_eq!(empty, TodoGateDecision::Continue);
        let only_backed = evaluate_todo_gate(&TodoGateInput {
            in_progress_backed: vec!["wait for download"],
            backing_task_count: 1,
            ..TodoGateInput::default()
        });
        assert_eq!(only_backed, TodoGateDecision::Continue);
        assert_eq!(TodoGateReason::InFlight.as_str(), "in_flight");
    }

    #[test]
    fn reminder_lists_pending_and_unbacked() {
        let text = build_todo_gate_reminder(&["next"], &["active"]);
        assert!(text.contains("- next"));
        assert!(text.contains("- active"));
        assert!(text.contains("update_plan"));
    }

    #[test]
    fn plan_steps_split_backed_when_waiting() {
        let steps = vec![
            PlanStep {
                title: "pending work".into(),
                status: PlanStatus::Pending,
            },
            PlanStep {
                title: "active wait".into(),
                status: PlanStatus::Active,
            },
        ];
        let idle = todo_gate_input_from_plan(&steps, false);
        assert_eq!(idle.pending, ["pending work"]);
        assert_eq!(idle.in_progress_unbacked, ["active wait"]);
        assert!(idle.in_progress_backed.is_empty());
        let waiting = todo_gate_input_from_plan(&steps, true);
        assert_eq!(waiting.in_progress_backed, ["active wait"]);
        assert!(waiting.in_progress_unbacked.is_empty());
        assert_eq!(
            evaluate_todo_gate(&waiting),
            evaluate_todo_gate(&TodoGateInput {
                pending: vec!["pending work"],
                ..TodoGateInput::default()
            })
        );
    }
}
