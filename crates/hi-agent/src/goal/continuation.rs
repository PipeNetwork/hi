//! Grok-build-style goal-drive continuation.
//!
//! The frontend still enqueues [`crate::GOAL_CONTINUE_PROMPT`]. This module
//! expands it into a concrete next-step directive so auto-drive turns do not
//! collapse into "keep going" narration.

use hi_tools::{PlanStatus, PlanStep};

use super::{Goal, GoalStatus};
use crate::GOAL_CONTINUE_NUDGE;
use crate::GoalKind;
use crate::steering::{BAIL_CONTINUE_NUDGE, matched_bail_out};

const DRIVE_CADENCE: &str = "Tool-call first, narration second. Do not ask permission to continue \
in-flight work. Do not stop with easy remaining work.";

pub(crate) fn render_drive_continuation(
    goal: &Goal,
    plan: &[PlanStep],
    last_assistant: Option<&str>,
    plan_file_next: Option<String>,
) -> String {
    let active = goal.active_sub_goal();
    let active_desc = active
        .map(|s| s.description.as_str())
        .unwrap_or("the next unfinished step");
    let notes = active
        .map(|s| s.notes.as_slice())
        .filter(|n| !n.is_empty())
        .map(|notes| format!("\nPrior failed attempts:\n- {}", notes.join("\n- ")))
        .unwrap_or_default();
    let upcoming = upcoming_sub_goal(goal)
        .map(|desc| format!("\nUpcoming: {desc}"))
        .unwrap_or_default();
    let plan_next = next_plan_item(plan)
        .filter(|title| Some(title.as_str()) != Some(active_desc))
        .map(|title| format!("\nPlan next: {title}"))
        .unwrap_or_default();
    let bail = last_assistant
        .and_then(matched_bail_out)
        .map(|_| format!("{BAIL_CONTINUE_NUDGE}\n\n"))
        .unwrap_or_default();
    let file_next = plan_file_next
        .filter(|title| Some(title.as_str()) != Some(active_desc))
        .map(|title| format!("\nPlan next: {title}"))
        .unwrap_or(plan_next);
    let verify_now = verification_block(goal);
    let gaps = if goal.last_gaps.is_empty() {
        String::new()
    } else {
        format!("\nVerifier gaps:\n{}", goal.last_gaps)
    };
    let strategy = if goal.strategy_note.is_empty() {
        String::new()
    } else {
        format!("\nStrategist:\n{}", goal.strategy_note)
    };
    format!(
        "{bail}Continue the long-horizon goal: complete the active sub-goal now.\n\
         Objective: {}\n\
         {}Active sub-goal: {active_desc}{notes}{upcoming}{file_next}{verify_now}{gaps}{strategy}\n\
         {}\n\
         {DRIVE_CADENCE}",
        goal.objective,
        goal.contract_prompt_lines(),
        super::kind_lens::drive_discipline(goal.kind),
    )
}

fn verification_block(goal: &Goal) -> String {
    if !goal.kind.requires_workspace_evidence() || goal.verification.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nRun the plan's verification now:");
    for item in goal.verification.iter().take(3) {
        out.push_str("\n- ");
        out.push_str(item);
    }
    out
}

fn upcoming_sub_goal(goal: &Goal) -> Option<&str> {
    let i = goal.active_index()? + 1;
    goal.sub_goals[i..]
        .iter()
        .find(|s| matches!(s.status, GoalStatus::Pending | GoalStatus::Active))
        .map(|s| s.description.as_str())
}

/// Mid-turn leftover-goal nudge. Code-change keeps [`GOAL_CONTINUE_NUDGE`] as a
/// prefix so existing steer tests still match; analysis/research name the
/// write-up contract instead of "implementation steps".
pub(crate) fn mid_turn_continue_nudge(goal: &Goal) -> String {
    let active = goal
        .active_sub_goal()
        .map(|step| step.description.as_str())
        .unwrap_or("the active sub-goal");
    match goal.kind {
        GoalKind::CodeChange => {
            format!("{GOAL_CONTINUE_NUDGE}\nActive sub-goal: {active}.")
        }
        GoalKind::Analysis => format!(
            "The long-horizon goal still has remaining sub-goals. \
             Continue `{active}` now — the deliverable is a cited write-up (the workspace diff may be empty). \
             Then call `update_plan` with the full goal checklist in its existing order, updating statuses. \
             If this sub-goal is genuinely complete, mark it done in `update_plan` and start the next one."
        ),
        GoalKind::Research => format!(
            "The long-horizon goal still has remaining sub-goals. \
             Continue `{active}` now — the deliverable is a source-backed summary (the workspace diff may be empty). \
             Then call `update_plan` with the full goal checklist in its existing order, updating statuses. \
             If this sub-goal is genuinely complete, mark it done in `update_plan` and start the next one."
        ),
    }
}

fn next_plan_item(plan: &[PlanStep]) -> Option<String> {
    plan.iter()
        .find(|step| step.status != PlanStatus::Done)
        .map(|step| step.title.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GoalKind;

    #[test]
    fn continuation_names_kind_next_step_and_discipline() {
        let goal = Goal::new(
            "port the parser to Rust",
            vec!["write the lexer".into(), "write the parser".into()],
        );
        assert_eq!(goal.kind, GoalKind::CodeChange);
        let plan = vec![PlanStep {
            title: "add lexer.rs".into(),
            status: PlanStatus::Pending,
        }];
        let text = render_drive_continuation(&goal, &plan, None, None);
        assert!(
            text.contains("Objective: port the parser to Rust"),
            "{text}"
        );
        assert!(text.contains("Kind: code-change"), "{text}");
        assert!(text.contains("Active sub-goal: write the lexer"), "{text}");
        assert!(text.contains("Upcoming: write the parser"), "{text}");
        assert!(text.contains("Plan next: add lexer.rs"), "{text}");
        assert!(text.contains("Goal NOT complete"), "{text}");
        assert!(text.contains("update_plan"), "{text}");
        assert!(text.contains("Tool-call first"), "{text}");
        assert!(!text.contains("Acceptance:"), "{text}");
        assert!(!text.contains("Verify:"), "{text}");
        assert!(!text.contains("You appear to be stopping"), "{text}");
    }

    #[test]
    fn continuation_injects_first_acceptance_and_verify() {
        let mut goal = Goal::new("port the parser to Rust", vec!["write the lexer".into()]);
        goal.acceptance = vec![
            "the lexer emits tokens for representative input".into(),
            "the parser builds an AST".into(),
        ];
        goal.verification = vec!["exercise the shipped lexer on a sample file".into()];
        let text = render_drive_continuation(&goal, &[], None, None);
        assert!(
            text.contains("Acceptance: the lexer emits tokens for representative input"),
            "{text}"
        );
        assert!(
            text.contains("Verify: exercise the shipped lexer on a sample file"),
            "{text}"
        );
        assert!(
            !text.contains("the parser builds an AST"),
            "only the first criterion is injected: {text}"
        );
    }

    #[test]
    fn continuation_analysis_asks_for_a_cited_write_up() {
        let goal = Goal::new(
            "explain how the auth middleware works",
            vec!["name the request path".into()],
        );
        assert_eq!(goal.kind, GoalKind::Analysis);
        let text = render_drive_continuation(&goal, &[], None, None);
        assert!(text.contains("cited, evidence-grounded write-up"), "{text}");
        assert!(
            !text.contains("Drive the shipped code"),
            "analysis must not inherit the code-change lens: {text}"
        );
        assert!(text.contains("update_plan"), "{text}");
    }

    #[test]
    fn continuation_omits_duplicate_plan_title() {
        let goal = Goal::new("ship it", vec!["write the lexer".into()]);
        let plan = vec![PlanStep {
            title: "write the lexer".into(),
            status: PlanStatus::Active,
        }];
        let text = render_drive_continuation(&goal, &plan, None, None);
        assert!(!text.contains("Plan next:"), "{text}");
    }

    #[test]
    fn continuation_prepends_bail_preface_after_a_surrender() {
        let goal = Goal::new("port the parser to Rust", vec!["write the lexer".into()]);
        let text = render_drive_continuation(
            &goal,
            &[],
            Some("I can't proceed with this without more context."),
            None,
        );
        assert!(
            text.starts_with("You appear to be stopping or handing off"),
            "{text}"
        );
        assert!(
            text.contains("Objective: port the parser to Rust"),
            "{text}"
        );
    }

    #[test]
    fn continuation_asks_code_change_to_run_stored_verification() {
        let mut goal = Goal::new("port the parser to Rust", vec!["write the lexer".into()]);
        goal.verification = vec!["exercise the shipped lexer on a sample file".into()];
        let text = render_drive_continuation(&goal, &[], None, None);
        assert!(text.contains("Run the plan's verification now:"), "{text}");
        assert!(
            text.contains("exercise the shipped lexer on a sample file"),
            "{text}"
        );
    }

    #[test]
    fn continuation_prefers_plan_file_next_step() {
        let goal = Goal::new(
            "port the parser to Rust",
            vec!["write the lexer".into(), "write the parser".into()],
        );
        let text = render_drive_continuation(&goal, &[], None, Some("add lexer.rs".into()));
        assert!(text.contains("Plan next: add lexer.rs"), "{text}");
    }

    #[test]
    fn mid_turn_nudge_names_the_active_step_and_kind() {
        let code = Goal::new(
            "port the parser to Rust",
            vec!["write the lexer".into(), "write the parser".into()],
        );
        let text = mid_turn_continue_nudge(&code);
        assert!(text.contains(GOAL_CONTINUE_NUDGE), "{text}");
        assert!(text.contains("Active sub-goal: write the lexer"), "{text}");

        let analysis = Goal::new(
            "explain how the auth middleware works",
            vec!["name the request path".into()],
        );
        let text = mid_turn_continue_nudge(&analysis);
        assert!(text.contains("`name the request path`"), "{text}");
        assert!(text.contains("cited write-up"), "{text}");
        assert!(
            !text.contains("implementation steps"),
            "analysis must not inherit the code-change continue: {text}"
        );
    }
}
