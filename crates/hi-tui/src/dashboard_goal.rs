#[derive(Clone, PartialEq, Eq)]
pub(crate) struct RowGoal {
    pub(crate) done: usize,
    pub(crate) total: usize,
    pub(crate) active: bool,
    pub(crate) paused: bool,
}

/// The fields the dashboard consumes from a child turn's schema-v2 report.
pub(crate) struct TurnReport {
    pub(crate) total_tokens: u64,
    pub(crate) goal: Option<RowGoal>,
    pub(crate) goal_raw: Option<String>,
    pub(crate) outcome_status: Option<String>,
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
    });
    Some(TurnReport {
        total_tokens: value
            .pointer("/usage/session/total_tokens")
            .or_else(|| value.get("total_tokens"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
        goal_raw: goal_value.map(|goal| goal.to_string()),
        goal,
        outcome_status: value
            .pointer("/outcome/status")
            .and_then(|status| status.as_str())
            .map(str::to_string),
    })
}

pub(crate) fn next_drive_stall(
    was_driving: bool,
    previous_goal: &Option<String>,
    new_goal: &Option<String>,
    current: u32,
) -> u32 {
    if was_driving && new_goal == previous_goal {
        current + 1
    } else {
        0
    }
}

pub(crate) fn should_retry_goal_turn(
    was_driving: bool,
    outcome_status: Option<&str>,
    goal: Option<&RowGoal>,
) -> bool {
    was_driving
        && outcome_status == Some("incomplete")
        && goal.is_some_and(|goal| goal.active && !goal.paused)
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
    fn stall_counts_only_unchanged_drive_turns() {
        let first = Some(r#"{"done":1,"total":3}"#.to_string());
        let next = Some(r#"{"done":2,"total":3}"#.to_string());
        assert_eq!(next_drive_stall(false, &first, &first, 5), 0);
        assert_eq!(next_drive_stall(true, &first, &next, 1), 0);
        assert_eq!(next_drive_stall(true, &first, &first, 0), 1);
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
        assert_eq!(next_drive_stall(true, &before, &retried, 1), 0);
        assert_eq!(next_drive_stall(true, &retried, &advanced, 1), 0);
    }

    #[test]
    fn incomplete_drive_retries_but_infrastructure_failure_does_not() {
        let active = RowGoal {
            done: 0,
            total: 2,
            active: true,
            paused: false,
        };
        assert!(should_retry_goal_turn(
            true,
            Some("incomplete"),
            Some(&active)
        ));
        assert!(!should_retry_goal_turn(true, Some("failed"), Some(&active)));
        assert!(!should_retry_goal_turn(
            false,
            Some("incomplete"),
            Some(&active)
        ));
        let paused = RowGoal {
            paused: true,
            ..active
        };
        assert!(!should_retry_goal_turn(
            true,
            Some("incomplete"),
            Some(&paused)
        ));
    }
}
