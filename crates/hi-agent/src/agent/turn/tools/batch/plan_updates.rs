use hi_tools::PlanStatus;

/// Constrain a model-authored checklist while the session is in plan mode.
///
/// A previously completed step may stay completed when the user re-enters
/// planning to revise the remaining work. Every other step is executable work
/// and therefore remains pending until a non-plan turn supplies real evidence.
pub(super) fn normalize_plan_mode_update(
    current: &[hi_tools::PlanStep],
    proposed: &mut [hi_tools::PlanStep],
) -> usize {
    let mut corrected = 0;
    for (index, step) in proposed.iter_mut().enumerate() {
        let status = if current.get(index).is_some_and(|existing| {
            existing.title == step.title && existing.status == PlanStatus::Done
        }) {
            PlanStatus::Done
        } else {
            PlanStatus::Pending
        };
        if step.status != status {
            step.status = status;
            corrected += 1;
        }
    }
    corrected
}

/// Whether a checklist title describes work that normally needs concrete
/// workspace or validation evidence before it can truthfully become `Done`.
/// Read-only milestones may still be completed from inspection evidence.
pub(super) fn plan_step_requires_execution_evidence(title: &str) -> bool {
    crate::agent::plan_goal::plan_step_requires_execution_evidence(title)
}

/// Extract a model-supplied reason that this particular implementation step
/// did not require a workspace change. A generic top-level recap cannot
/// self-certify an entire checklist.
fn plan_step_has_no_change_justification(arguments: &str, index: usize) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments) else {
        return false;
    };
    value
        .get("steps")
        .and_then(serde_json::Value::as_array)
        .and_then(|steps| steps.get(index))
        .and_then(|step| step.get("completion_evidence"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .is_some_and(concrete_no_change_justification)
}

fn concrete_no_change_justification(reason: &str) -> bool {
    let normalized = reason
        .to_ascii_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '/' | '_' | '-') {
                character
            } else {
                ' '
            }
        })
        .collect::<String>();
    let words = normalized.split_whitespace().collect::<Vec<_>>();
    if reason.chars().count() < 12 || words.len() < 3 {
        return false;
    }
    !matches!(
        words.join(" ").as_str(),
        "done"
            | "already done"
            | "already complete"
            | "already completed"
            | "no change needed"
            | "no changes needed"
            | "no change required"
            | "no changes required"
            | "not needed"
            | "not required"
    )
}

/// Reject unsupported completion claims on implementation-shaped checklist
/// steps. Successful mutation, successful validation, or an explicit per-step
/// no-change justification is required.
pub(super) fn normalize_unsupported_plan_completion(
    current: &[hi_tools::PlanStep],
    proposed: &mut [hi_tools::PlanStep],
    arguments: &str,
    execution_evidence: bool,
) -> Vec<usize> {
    // Turn-global mutation/validation proves at most the step that was active
    // when the work began. Letting one unrelated write or an old passing test
    // authorize every `done` entry allowed a weak model to clear an entire
    // durable checklist at once. Additional implementation steps need their
    // own concrete `completion_evidence`.
    let evidenced_index = execution_evidence
        .then(|| {
            current
                .iter()
                .position(|step| step.status == PlanStatus::Active)
                .or_else(|| {
                    current
                        .iter()
                        .position(|step| step.status == PlanStatus::Pending)
                })
                .or_else(|| {
                    proposed.iter().position(|step| {
                        step.status == PlanStatus::Done
                            && plan_step_requires_execution_evidence(&step.title)
                    })
                })
        })
        .flatten();

    let mut corrected = Vec::new();
    for (index, step) in proposed.iter_mut().enumerate() {
        if step.status != PlanStatus::Done || !plan_step_requires_execution_evidence(&step.title) {
            continue;
        }
        let already_done = current.get(index).is_some_and(|existing| {
            existing.title == step.title && existing.status == PlanStatus::Done
        });
        if already_done
            || evidenced_index == Some(index)
            || plan_step_has_no_change_justification(arguments, index)
        {
            continue;
        }

        step.status = current
            .get(index)
            .filter(|existing| existing.title == step.title)
            .map(|existing| existing.status)
            .filter(|status| *status != PlanStatus::Done)
            .unwrap_or(PlanStatus::Pending);
        corrected.push(index);
    }
    corrected
}
