//! Unit tests for planner contracts, gap stall, and drive-budget helpers.

use super::*;

fn goal() -> Goal {
    Goal::new(
        "refactor the parser",
        vec![
            "write tests".into(),
            "rewrite parser".into(),
            "update callers".into(),
        ],
    )
}

#[test]
fn a_goal_saved_before_these_fields_existed_still_loads() {
    // Sessions on disk predate every counter added here. If any of them
    // failed to default, resuming an existing long-horizon goal would error
    // instead of picking up where it left off — losing exactly the progress
    // record these changes exist to protect.
    let legacy = r#"{
        "objective": "review plan.md and fully build this",
        "sub_goals": [
            {"description": "step one", "status": "Done", "attempts": 0, "notes": []},
            {"description": "step two", "status": "Active", "attempts": 2, "notes": ["a note"]}
        ],
        "status": "Active",
        "paused": true,
        "team": true
    }"#;
    let goal: Goal = serde_json::from_str(legacy).expect("legacy goal must deserialize");

    assert_eq!(goal.sub_goals.len(), 2);
    assert_eq!(goal.completed_count(), 1);
    assert!(goal.is_paused());
    // Every field added across these changes defaults sanely.
    assert_eq!(goal.consecutive_skips, 0);
    assert_eq!(goal.turn_budget, None);
    assert_eq!(goal.turns_spent, 0);
    assert_eq!(goal.sub_goals[1].unjudged_turns, 0);
    assert_eq!(goal.sub_goals[1].productive_turns, 0);
    assert_eq!(goal.sub_goals[1].cap_continuations, 0);
    // And the pre-existing state survives untouched.
    assert_eq!(goal.sub_goals[1].attempts, 2);
    assert!(!goal.is_thrashing(), "a legacy goal must not read as stuck");
    assert!(goal.acceptance.is_empty());
    assert!(goal.verification.is_empty());
    assert_eq!(goal.kind, crate::GoalKind::CodeChange);
}

#[test]
fn from_goal_plan_freezes_kind_and_caps_criteria() {
    let plan = GoalPlan {
        kind: Some(crate::GoalKind::Analysis),
        milestones: vec!["name the request path".into()],
        acceptance: vec!["the auth middleware's request path is named".into()],
        verification: vec!["the write-up cites the responsible functions".into()],
    };
    let g = Goal::from_goal_plan("explain how the auth middleware works", plan);
    assert_eq!(g.kind, crate::GoalKind::Analysis);
    assert_eq!(g.sub_goals[0].description, "name the request path");
    assert_eq!(
        g.acceptance,
        vec!["the auth middleware's request path is named"]
    );
    let section = g.prompt_section().expect("renders");
    assert!(
        section.contains("the deliverable is a cited write-up"),
        "{section}"
    );
    assert!(section.contains("Kind: analysis"), "{section}");
    assert!(
        section.contains("Acceptance: the auth middleware's request path is named"),
        "{section}"
    );
    assert!(
        section.contains("Verify: the write-up cites the responsible functions"),
        "{section}"
    );
    let md = g.to_markdown();
    assert!(md.contains("**Kind:** analysis"), "{md}");
    assert!(md.contains("## Acceptance criteria"), "{md}");
    let report = g.status_report();
    assert!(report.contains("kind: analysis"), "{report}");
    assert!(
        report.contains("acceptance: the auth middleware's request path is named"),
        "{report}"
    );
}

#[test]
fn from_goal_plan_caps_acceptance_items() {
    let plan = GoalPlan {
        kind: None,
        milestones: vec!["write the lexer".into()],
        acceptance: (0..12).map(|i| format!("criterion {i}")).collect(),
        verification: vec![" ".into(), "exercise the shipped lexer".into()],
    };
    let g = Goal::from_goal_plan("port the parser to Rust", plan);
    assert_eq!(g.kind, crate::GoalKind::CodeChange);
    assert_eq!(g.acceptance.len(), MAX_PLAN_ITEMS);
    assert_eq!(g.verification, vec!["exercise the shipped lexer"]);
}

#[test]
fn identical_panel_gaps_stall_on_the_second_repeat() {
    let mut g = Goal::new("ship it", vec!["one".into()]);
    g.last_gaps = "missing CSRF path".into();
    assert!(!g.record_repeated_gaps());
    g.last_gaps = "missing CSRF path".into();
    assert!(g.record_repeated_gaps());
    g.last_gaps = "a different gap".into();
    assert!(!g.record_repeated_gaps());
}

#[test]
fn empty_gaps_do_not_count_as_a_repeat() {
    let mut g = Goal::new("ship it", vec!["one".into()]);
    assert!(!g.record_repeated_gaps());
    assert!(!g.record_repeated_gaps());
}

#[test]
fn fresh_goals_have_no_default_turn_budget() {
    let g = goal();
    assert_eq!(g.turn_budget, None);
    assert!(!g.budget_auto);
    assert!(!g.budget_exhausted());
}

#[test]
fn plan_growth_does_not_install_a_default_turn_budget() {
    let mut g = Goal::new("ship it", vec!["one".into()]);
    let grown: Vec<String> = (0..80).map(|i| format!("step {i}")).collect();
    g.append_missing(&grown);
    assert_eq!(g.turn_budget, None);
    assert!(!g.budget_auto);
}

#[test]
fn a_legacy_automatic_budget_is_removed_and_reopened() {
    let mut g = Goal::new("ship it", vec!["one".into()]);
    g.turn_budget = Some(25);
    g.budget_auto = true;
    g.pause(GoalPauseReason::Budget);

    assert!(g.clear_legacy_automatic_budget());
    assert_eq!(g.turn_budget, None);
    assert!(!g.budget_auto);
    assert!(!g.is_paused());
    assert_eq!(g.pause_reason, GoalPauseReason::None);
    assert!(!g.clear_legacy_automatic_budget());
}

#[test]
fn an_explicit_budget_stops_the_rescaling() {
    let mut g = Goal::new("ship it", vec!["one".into()]);
    g.turn_budget = Some(7);
    g.budget_auto = false; // as `/goal budget 7` does
    let grown: Vec<String> = (0..40).map(|i| format!("step {i}")).collect();
    g.append_missing(&grown);
    assert_eq!(
        g.turn_budget,
        Some(7),
        "a number the user chose must not move under them"
    );
}

#[test]
fn compatibility_auto_budget_helper_is_unlimited() {
    assert_eq!(auto_budget_for(0), u32::MAX);
    assert_eq!(auto_budget_for(1), u32::MAX);
    assert_eq!(auto_budget_for(100_000), u32::MAX);
}

#[test]
fn a_turn_budget_bounds_an_open_ended_objective() {
    // "fully build this" against a multi-phase plan has no reachable end
    // state, so without a ceiling it simply runs until someone notices.
    let mut g = goal();
    // `/goal budget off` — the explicit opt-out.
    g.turn_budget = None;
    g.budget_auto = false;
    assert!(!g.budget_exhausted(), "no budget set = runs until done");
    assert_eq!(g.turns_remaining(), None);
    assert!(!g.spend_turn(), "spending against no budget never exhausts");

    g.turn_budget = Some(2);
    g.turns_spent = 0;
    assert!(!g.spend_turn(), "one of two");
    assert_eq!(g.turns_remaining(), Some(1));
    assert!(g.spend_turn(), "the second turn exhausts it");
    assert!(g.budget_exhausted());
    assert_eq!(g.turns_remaining(), Some(0));
}

#[test]
fn the_progress_report_accounts_for_every_step() {
    // A goal that stops without saying what it finished, what it couldn't
    // reach, and what's left is no more useful than one that ran forever.
    let mut g = Goal::new(
        "ship it",
        vec!["one".into(), "two".into(), "three".into(), "four".into()],
    );
    g.advance(); // one: done
    g.block_active("a running PostgreSQL"); // two: blocked
    g.record_failure("verification failed", 0); // three: failed
    g.turns_spent = 7;

    let report = g.progress_report();
    assert!(report.contains("1 done"), "{report}");
    assert!(report.contains("1 failed"), "{report}");
    assert!(report.contains("1 blocked"), "{report}");
    assert!(report.contains("across 7 turn(s)"), "{report}");
    assert!(
        report.contains("a running PostgreSQL"),
        "the actionable prerequisite must appear: {report}"
    );
    assert!(
        report.contains("Next up: 4."),
        "the user needs to know where it would resume: {report}"
    );
}

#[test]
fn blocking_a_step_costs_no_retry_budget_and_is_not_a_failure() {
    // A missing prerequisite is not a rejected attempt. Marking it `Failed`
    // tells the user their work was judged and found wanting, and hides the
    // one thing they can act on.
    let mut g = goal();
    assert!(g.block_active("a running PostgreSQL reachable via DATABASE_URL"));

    assert_eq!(g.sub_goals[0].status, GoalStatus::Blocked);
    assert_ne!(g.sub_goals[0].status, GoalStatus::Failed);
    assert_eq!(g.sub_goals[0].attempts, 0, "no retry budget spent");
    assert_eq!(g.active_index(), Some(1), "the drive moves on");
    assert_eq!(g.status, GoalStatus::Active);

    let blocked = g.blocked_steps();
    assert_eq!(blocked.len(), 1);
    assert!(
        blocked[0].1.notes.iter().any(|n| n.contains("PostgreSQL")),
        "the prerequisite must be recorded verbatim: {:?}",
        blocked[0].1.notes
    );
}

#[test]
fn a_wholly_blocked_plan_reports_blocked_not_failed() {
    let mut g = goal();
    g.block_active("no database");
    g.block_active("no database");
    assert!(!g.block_active("no database"), "nothing left to drive");
    assert_eq!(
        g.status,
        GoalStatus::Blocked,
        "the goal is waiting on prerequisites, not broken"
    );

    // A genuine failure alongside blocks dominates — claiming merely
    // "blocked" would overstate how recoverable the run is.
    let mut mixed = goal();
    mixed.block_active("no database");
    mixed.record_failure("verification failed", 0);
    mixed.block_active("no tofu");
    assert_eq!(mixed.status, GoalStatus::Failed);
}
