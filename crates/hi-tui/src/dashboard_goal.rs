#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RowGoal {
    pub(crate) done: usize,
    pub(crate) total: usize,
    pub(crate) active: bool,
    pub(crate) paused: bool,
    /// `running|paused|parked|off` from report `goal.drive`. Absent on old reports.
    pub(crate) drive: Option<String>,
    /// Phase-level trail from the report's `goal.phases` array: `(title,
    /// state)` where state is `"done"`, `"active"`, or `"pending"`. Empty when
    /// the child didn't emit phases (older binaries or non-goal rows).
    pub(crate) phases: Vec<(String, String)>,
}

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RowPlan {
    pub(crate) done: usize,
    pub(crate) total: usize,
    pub(crate) next: Option<String>,
    pub(crate) pending: bool,
    pub(crate) drive: String,
}

/// The fields the dashboard consumes from a child turn's schema-v2 report.
pub(crate) struct TurnReport {
    pub(crate) total_tokens: u64,
    pub(crate) goal: Option<RowGoal>,
    pub(crate) goal_raw: Option<String>,
    pub(crate) leftover: Option<String>,
    pub(crate) plan: Option<RowPlan>,
    pub(crate) changed_files: Vec<String>,
    pub(crate) progress_events: Vec<hi_agent::ProgressEvent>,
    /// Agent-owned exact cross-turn stall counters. Absent on older child reports.
    pub(crate) plan_drive_stall: Option<u32>,
    pub(crate) goal_drive_stall: Option<u32>,
}

pub(crate) fn parse_report(text: &str) -> Option<TurnReport> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let goal_value = value.get("goal").filter(|goal| !goal.is_null());
    let goal = goal_value.map(|goal| RowGoal {
        done: goal
            .get("done")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        total: goal
            .get("total")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        active: goal.get("status").and_then(|value| value.as_str()) == Some("Active"),
        paused: goal
            .get("paused")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        drive: goal
            .get("drive")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        phases: goal
            .get("phases")
            .and_then(|value| value.as_array())
            .map(|array| {
                array
                    .iter()
                    .filter_map(|entry| {
                        let title = entry.get("title")?.as_str()?.to_string();
                        let state = entry
                            .get("state")
                            .and_then(|value| value.as_str())
                            .unwrap_or("pending")
                            .to_string();
                        Some((title, state))
                    })
                    .collect()
            })
            .unwrap_or_default(),
    });
    let plan_value = value.get("plan").filter(|plan| !plan.is_null());
    let plan = plan_value.map(|plan| RowPlan {
        done: plan
            .get("done")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        total: plan
            .get("total")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        next: plan
            .get("next")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        pending: plan
            .get("pending")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
        drive: plan
            .get("drive")
            .and_then(|value| value.as_str())
            .unwrap_or("off")
            .to_string(),
    });
    let leftover = value
        .pointer("/outcome/plan_leftover")
        .and_then(|leftover| leftover.as_str())
        .map(str::to_string)
        .or_else(|| {
            value
                .pointer("/outcome/leftover")
                .and_then(|leftover| leftover.as_str())
                .map(str::to_string)
        })
        .or_else(|| {
            plan.as_ref().and_then(|plan| {
                plan.pending
                    .then(|| {
                        plan.next
                            .as_deref()
                            .map(|next| format!("remaining — {next}"))
                    })
                    .flatten()
            })
        });
    Some(TurnReport {
        total_tokens: value
            .pointer("/usage/session/total_tokens")
            .or_else(|| value.get("total_tokens"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
        goal_raw: goal_value.map(|goal| goal.to_string()),
        goal,
        leftover,
        plan,
        changed_files: value
            .pointer("/outcome/changed_files")
            .and_then(|files| files.as_array())
            .map(|files| {
                files
                    .iter()
                    .filter_map(|file| file.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        progress_events: value
            .pointer("/telemetry/progress_events")
            .and_then(|events| serde_json::from_value(events.clone()).ok())
            .unwrap_or_default(),
        plan_drive_stall: value
            .pointer("/telemetry/plan_drive_stall")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok()),
        goal_drive_stall: value
            .pointer("/telemetry/goal_drive_stall")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok()),
    })
}

pub(crate) fn next_drive_stall(
    was_driving: bool,
    previous_goal: &Option<String>,
    new_goal: &Option<String>,
    productive_evidence: bool,
    current: u32,
) -> u32 {
    if was_driving && same_drive_state(previous_goal, new_goal) && !productive_evidence {
        current.saturating_add(1)
    } else {
        0
    }
}

/// Compare the parts of a reported goal that represent work. `turns_spent` is
/// accounting, not progress: it changes on every synthetic drive turn and
/// must not keep an unchanged fleet row alive indefinitely. Keep the comparison
/// tolerant of older/malformed reports by falling back to the raw JSON.
fn same_drive_state(previous: &Option<String>, new: &Option<String>) -> bool {
    if previous == new {
        return true;
    }
    let (Some(previous), Some(new)) = (previous, new) else {
        return false;
    };
    let Ok(mut previous) = serde_json::from_str::<serde_json::Value>(previous) else {
        return false;
    };
    let Ok(mut new) = serde_json::from_str::<serde_json::Value>(new) else {
        return false;
    };
    for goal in [&mut previous, &mut new] {
        if let Some(object) = goal.as_object_mut() {
            object.remove("turns_spent");
        }
    }
    previous == new
}

pub(crate) fn leftover_step_title(leftover: Option<&str>) -> Option<&str> {
    leftover.and_then(|line| line.split_once(" — ").map(|(_, title)| title))
}

pub(crate) fn drive_action(
    goal: Option<&RowGoal>,
    plan: Option<&RowPlan>,
    leftover: Option<&str>,
    plan_drive_stall: u32,
) -> hi_agent::DriveAction {
    if let Some(goal) = goal
        && let Some(drive) = goal.drive.as_deref()
    {
        return match drive {
            "running" => hi_agent::DriveAction::Enqueue(hi_agent::DriveKind::Goal),
            "paused" => hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalPaused,
            },
            "parked" => hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalParked,
            },
            _ => hi_agent::DriveAction::from_plan(plan_drive_action(
                Some(goal),
                plan,
                leftover,
                plan_drive_stall,
            )),
        };
    }
    if goal.is_some_and(|goal| goal.active && !goal.paused) {
        return hi_agent::DriveAction::Enqueue(hi_agent::DriveKind::Goal);
    }
    hi_agent::DriveAction::from_plan(plan_drive_action(goal, plan, leftover, plan_drive_stall))
}

pub(crate) fn plan_drive_action(
    goal: Option<&RowGoal>,
    plan: Option<&RowPlan>,
    leftover: Option<&str>,
    plan_drive_stall: u32,
) -> hi_agent::PlanDriveAction {
    let goal_driving = goal.is_some_and(|goal| goal.active && !goal.paused);
    if let Some(plan) = plan {
        let paused = plan.drive == "paused";
        let stall = if plan.drive == "parked" {
            hi_agent::PLAN_DRIVE_STALL_LIMIT
        } else {
            0
        };
        hi_agent::PlanDriveAction::decide(plan.pending, false, paused, stall, goal_driving, None)
    } else {
        hi_agent::PlanDriveAction::decide(
            leftover.is_some(),
            false,
            false,
            plan_drive_stall,
            goal_driving,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_reads_tokens_and_goal() {
        let json = r#"{"schema_version":2,"usage":{"session":{"total_tokens":12345}},
            "goal":{"done":2,"total":7,"status":"Active","paused":false}}"#;
        let report = parse_report(json).unwrap();
        assert_eq!(report.total_tokens, 12345);
        let goal = report.goal.unwrap();
        assert_eq!((goal.done, goal.total), (2, 7));
        assert!(goal.active && !goal.paused);
        assert!(
            parse_report(r#"{"schema_version":2,"goal":null}"#)
                .unwrap()
                .goal
                .is_none()
        );
        assert!(parse_report("not json").is_none());
    }

    #[test]
    fn report_reads_phases_trail() {
        let json = r#"{"schema_version":2,"goal":{"done":1,"total":3,"status":"Active","paused":false,
            "phases":[
                {"title":"Scan","state":"done"},
                {"title":"Analyze","state":"active"},
                {"title":"Synthesize","state":"pending"}
            ]}}"#;
        let report = parse_report(json).unwrap();
        let goal = report.goal.unwrap();
        assert_eq!(goal.phases.len(), 3);
        assert_eq!(goal.phases[0], ("Scan".into(), "done".into()));
        assert_eq!(goal.phases[1], ("Analyze".into(), "active".into()));
        assert_eq!(goal.phases[2], ("Synthesize".into(), "pending".into()));
    }

    #[test]
    fn report_without_phases_yields_empty_trail() {
        let json =
            r#"{"schema_version":2,"goal":{"done":0,"total":2,"status":"Active","paused":false}}"#;
        let report = parse_report(json).unwrap();
        let goal = report.goal.unwrap();
        assert!(goal.phases.is_empty());
    }

    #[test]
    fn stall_counts_only_unchanged_drive_turns() {
        let first = Some(r#"{"done":1,"total":3}"#.to_string());
        let next = Some(r#"{"done":2,"total":3}"#.to_string());
        assert_eq!(next_drive_stall(false, &first, &first, false, 5), 0);
        assert_eq!(next_drive_stall(true, &first, &next, false, 1), 0);
        assert_eq!(next_drive_stall(true, &first, &first, false, 0), 1);
        assert_eq!(next_drive_stall(true, &first, &first, true, 3), 0);
    }

    #[test]
    fn stall_ignores_drive_turn_accounting() {
        let before =
            Some(r#"{"done":1,"total":3,"turns_spent":4,"events":[{"kind":"set"}]}"#.to_string());
        let after =
            Some(r#"{"done":1,"total":3,"turns_spent":5,"events":[{"kind":"set"}]}"#.to_string());
        assert_eq!(next_drive_stall(true, &before, &after, false, 2), 3);
    }

    #[test]
    fn stall_counter_saturates() {
        let goal = Some(r#"{"done":1,"total":3}"#.to_string());
        assert_eq!(
            next_drive_stall(true, &goal, &goal, false, u32::MAX),
            u32::MAX
        );
    }

    #[test]
    fn retry_notes_and_cursor_changes_reset_stall() {
        let raw = |attempts: u8, active: usize| {
            parse_report(&format!(
                r#"{{"schema_version":2,"goal":{{"done":0,"total":2,"status":"Active","paused":false,"active_index":{active},"sub_goals":[{{"attempts":{attempts}}}]}}}}"#
            ))
            .unwrap()
            .goal_raw
        };
        let before = raw(0, 0);
        let retried = raw(1, 0);
        let advanced = raw(1, 1);
        assert_eq!(next_drive_stall(true, &before, &retried, false, 1), 0);
        assert_eq!(next_drive_stall(true, &retried, &advanced, false, 1), 0);
    }

    #[test]
    fn leftover_pending_plan_enqueues_drive_when_goal_is_not_driving() {
        let json = r#"{"schema_version":2,"outcome":{"status":"completed","leftover":"1/2 remaining — wire the scheduler","changed_files":[]}}"#;
        let report = parse_report(json).unwrap();
        assert_eq!(
            report.leftover.as_deref(),
            Some("1/2 remaining — wire the scheduler")
        );
        assert_eq!(
            leftover_step_title(report.leftover.as_deref()),
            Some("wire the scheduler")
        );
        assert!(
            plan_drive_action(None, report.plan.as_ref(), report.leftover.as_deref(), 0)
                .should_enqueue()
        );
        let active = RowGoal {
            done: 0,
            total: 2,
            active: true,
            paused: false,
            drive: None,
            phases: Vec::new(),
        };
        assert!(
            !plan_drive_action(
                Some(&active),
                report.plan.as_ref(),
                report.leftover.as_deref(),
                0
            )
            .should_enqueue()
        );
        assert!(
            !plan_drive_action(
                None,
                report.plan.as_ref(),
                report.leftover.as_deref(),
                hi_agent::PLAN_DRIVE_STALL_LIMIT
            )
            .should_enqueue()
        );
    }

    #[test]
    fn report_plan_pending_enqueues_and_paused_parked_do_not() {
        let json = r#"{"schema_version":2,"plan":{"done":1,"total":2,"next":"wire the scheduler","pending":true,"drive":"running"}}"#;
        let report = parse_report(json).unwrap();
        assert!(report.plan.as_ref().is_some_and(|plan| plan.pending));
        assert!(
            plan_drive_action(None, report.plan.as_ref(), report.leftover.as_deref(), 0)
                .should_enqueue()
        );

        let paused = r#"{"schema_version":2,"plan":{"done":1,"total":2,"next":"wire the scheduler","pending":true,"drive":"paused"}}"#;
        let paused = parse_report(paused).unwrap();
        assert!(
            !plan_drive_action(None, paused.plan.as_ref(), paused.leftover.as_deref(), 0)
                .should_enqueue()
        );

        let parked = r#"{"schema_version":2,"plan":{"done":1,"total":2,"next":"wire the scheduler","pending":true,"drive":"parked"}}"#;
        let parked = parse_report(parked).unwrap();
        assert_eq!(
            plan_drive_action(None, parked.plan.as_ref(), parked.leftover.as_deref(), 0),
            hi_agent::PlanDriveAction::Idle {
                reason: hi_agent::PlanDriveIdleReason::Parked
            }
        );
    }

    #[test]
    fn report_goal_drive_enqueues_and_paused_parked_do_not() {
        let running = r#"{"schema_version":2,"goal":{"done":0,"total":2,"status":"Active","paused":false,"drive":"running"}}"#;
        let running = parse_report(running).unwrap();
        assert_eq!(
            drive_action(running.goal.as_ref(), None, None, 0),
            hi_agent::DriveAction::Enqueue(hi_agent::DriveKind::Goal)
        );

        let paused = r#"{"schema_version":2,"goal":{"done":0,"total":2,"status":"Active","paused":true,"drive":"paused"}}"#;
        let paused = parse_report(paused).unwrap();
        assert_eq!(
            drive_action(paused.goal.as_ref(), None, None, 0),
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalPaused
            }
        );

        let parked = r#"{"schema_version":2,"goal":{"done":0,"total":2,"status":"Active","paused":false,"drive":"parked"}}"#;
        let parked = parse_report(parked).unwrap();
        assert_eq!(
            drive_action(parked.goal.as_ref(), None, None, 0),
            hi_agent::DriveAction::Idle {
                reason: hi_agent::DriveIdleReason::GoalParked
            }
        );

        let plan = r#"{"schema_version":2,"plan":{"done":1,"total":2,"next":"wire the scheduler","pending":true,"drive":"running"}}"#;
        let plan = parse_report(plan).unwrap();
        assert_eq!(
            drive_action(None, plan.plan.as_ref(), plan.leftover.as_deref(), 0),
            hi_agent::DriveAction::Enqueue(hi_agent::DriveKind::Plan)
        );
    }
}
