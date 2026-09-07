use super::*;

#[test]
fn goal_drive_stall_skips_stuck_step_and_keeps_driving() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["first".into(), "second".into(), "third".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    let mut last = crate::GoalDriveProgress::Unchanged;
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        last = agent.note_goal_drive_progress(false);
    }
    assert!(
        matches!(
            last,
            crate::GoalDriveProgress::Skipped { ref failed, .. } if failed == "first"
        ),
        "{last:?}"
    );
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal)
    );
    assert_eq!(agent.goal_drive_stall(), 0);
    let goal = agent.structured_goal().expect("goal");
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Failed);
    assert_eq!(goal.sub_goals[1].status, crate::GoalStatus::Active);
    assert_eq!(goal.pause_reason, crate::GoalPauseReason::None);
}

#[test]
fn goal_drive_two_skips_without_completion_parks() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["first".into(), "second".into(), "third".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        agent.note_goal_drive_progress(false);
    }
    let mut last = crate::GoalDriveProgress::Unchanged;
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        last = agent.note_goal_drive_progress(false);
    }
    assert_eq!(last, crate::GoalDriveProgress::Parked);
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::GoalParked
        }
    );
    let goal = agent.structured_goal().expect("goal");
    assert!(goal.is_thrashing());
    assert_eq!(goal.pause_reason, crate::GoalPauseReason::None);
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Failed);
    assert_eq!(goal.sub_goals[1].status, crate::GoalStatus::Failed);
}

#[test]
fn stall_skip_then_completion_requeues_failed_step() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["first".into(), "second".into(), "third".into()],
            )))
            .unwrap()
    );
    let mut last = crate::GoalDriveProgress::Unchanged;
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        last = agent.note_goal_drive_progress(false);
    }
    assert!(
        matches!(
            last,
            crate::GoalDriveProgress::Skipped { ref failed, .. } if failed == "first"
        ),
        "{last:?}"
    );
    agent
        .update_structured_goal(|goal| {
            goal.sub_goals[1].status = crate::GoalStatus::Done;
            goal.sub_goals[2].status = crate::GoalStatus::Done;
            goal.rederive_status();
        })
        .unwrap();
    let progress = agent.note_goal_drive_progress(true);
    assert_eq!(progress, crate::GoalDriveProgress::Requeued { count: 1 });
    let goal = agent.structured_goal().expect("goal");
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Active);
    assert!(goal.sub_goals[0].stall_skipped);
    assert!(goal.sub_goals[0].requeued);
    assert_eq!(goal.pause_reason, crate::GoalPauseReason::None);
    let outcome = completed_outcome(agent.leftover_work());
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal)
    );
}

#[test]
fn requeued_step_that_stalls_again_stays_failed_and_parks() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["first".into(), "second".into(), "third".into()],
            )))
            .unwrap()
    );
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        agent.note_goal_drive_progress(false);
    }
    agent
        .update_structured_goal(|goal| {
            goal.sub_goals[1].status = crate::GoalStatus::Done;
            goal.sub_goals[2].status = crate::GoalStatus::Done;
            goal.rederive_status();
        })
        .unwrap();
    assert_eq!(
        agent.note_goal_drive_progress(true),
        crate::GoalDriveProgress::Requeued { count: 1 }
    );
    let mut last = crate::GoalDriveProgress::Unchanged;
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        last = agent.note_goal_drive_progress(false);
    }
    assert_eq!(last, crate::GoalDriveProgress::Parked);
    let goal = agent.structured_goal().expect("goal");
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Failed);
    assert!(goal.sub_goals[0].requeued);
    assert_eq!(goal.pause_reason, crate::GoalPauseReason::None);
    assert!(!goal.has_drive_work());
    let outcome = completed_outcome(agent.leftover_work());
    assert!(
        !matches!(
            agent.drive_decision(Some(&outcome)),
            crate::DriveAction::Enqueue(_)
        ),
        "failed requeued step must not keep driving"
    );
}
