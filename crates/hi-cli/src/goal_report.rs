pub(crate) fn report_goal(goal: Option<&hi_agent::Goal>) -> Option<serde_json::Value> {
    goal.map(|goal| {
        let phases: Vec<serde_json::Value> = goal
            .sub_goals
            .iter()
            .map(|sg| {
                serde_json::json!({
                    "title": sg.description,
                    "state": goal_status_phase_state(sg.status),
                })
            })
            .collect();
        serde_json::json!({
            "objective": goal.objective,
            "done": goal.sub_goals.iter().filter(|step| step.status == hi_agent::GoalStatus::Done).count(),
            "total": goal.sub_goals.len(),
            "status": format!("{:?}", goal.status),
            "paused": goal.paused,
            "active_index": goal.active_index(),
            "sub_goals": goal.sub_goals,
            "phases": phases,
            "skeptic_objections": goal.skeptic_objections,
            "skeptic_unavailable": goal.skeptic_unavailable,
            "last_skeptic_status": goal.last_skeptic_status,
        })
    })
}

/// Map a [`hi_agent::GoalStatus`] to the phase-state string the dashboard's
/// phase trail renders: `"done"`, `"active"`, or `"pending"`. Failed and
/// Blocked sub-goals render as pending (not done) — the trail shows progress,
/// not failure mode; the `◎done/total` counter and attention badge carry that.
fn goal_status_phase_state(status: hi_agent::GoalStatus) -> &'static str {
    match status {
        hi_agent::GoalStatus::Done => "done",
        hi_agent::GoalStatus::Active => "active",
        hi_agent::GoalStatus::Pending
        | hi_agent::GoalStatus::Failed
        | hi_agent::GoalStatus::Blocked => "pending",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_progress_is_visible_beyond_aggregate_counts() {
        let mut goal = hi_agent::Goal::new(
            "ship the parser",
            vec!["implement parser".into(), "update callers".into()],
        );
        let before = report_goal(Some(&goal)).unwrap();
        assert!(goal.record_failure("first approach broke escaped input", 2));
        let after = report_goal(Some(&goal)).unwrap();

        assert_eq!(before["done"], after["done"]);
        assert_eq!(before["total"], after["total"]);
        assert_ne!(before, after);
        assert_eq!(after["active_index"], 0);
        assert_eq!(after["sub_goals"][0]["attempts"], 1);
        assert_eq!(after["sub_goals"][0]["notes"][0],
            "first approach broke escaped input"
        );
    }

    #[test]
    fn phases_array_mirrors_sub_goal_states() {
        let mut goal = hi_agent::Goal::new(
            "ship it",
            vec!["step one".into(), "step two".into(), "step three".into()],
        );
        // Mark step one done, step two is active (the default for index 0
        // after construction is Active; advance to make it realistic).
        goal.sub_goals[0].status = hi_agent::GoalStatus::Done;
        goal.sub_goals[1].status = hi_agent::GoalStatus::Active;
        goal.sub_goals[2].status = hi_agent::GoalStatus::Pending;

        let report = report_goal(Some(&goal)).unwrap();
        let phases = report["phases"].as_array().unwrap();
        assert_eq!(phases.len(), 3);
        assert_eq!(phases[0]["title"], "step one");
        assert_eq!(phases[0]["state"], "done");
        assert_eq!(phases[1]["title"], "step two");
        assert_eq!(phases[1]["state"], "active");
        assert_eq!(phases[2]["title"], "step three");
        assert_eq!(phases[2]["state"], "pending");
    }

    #[test]
    fn failed_and_blocked_render_as_pending_in_phase_trail() {
        let mut goal = hi_agent::Goal::new("x", vec!["a".into(), "b".into()]);
        goal.sub_goals[0].status = hi_agent::GoalStatus::Failed;
        goal.sub_goals[1].status = hi_agent::GoalStatus::Blocked;
        let report = report_goal(Some(&goal)).unwrap();
        let phases = report["phases"].as_array().unwrap();
        assert_eq!(phases[0]["state"], "pending");
        assert_eq!(phases[1]["state"], "pending");
    }
}
