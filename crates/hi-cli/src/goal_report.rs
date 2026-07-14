pub(crate) fn report_goal(goal: Option<&hi_agent::Goal>) -> Option<serde_json::Value> {
    goal.map(|goal| {
        serde_json::json!({
            "objective": goal.objective,
            "done": goal.sub_goals.iter().filter(|step| step.status == hi_agent::GoalStatus::Done).count(),
            "total": goal.sub_goals.len(),
            "status": format!("{:?}", goal.status),
            "paused": goal.paused,
            "active_index": goal.active_index(),
            "sub_goals": goal.sub_goals,
            "skeptic_objections": goal.skeptic_objections,
            "skeptic_unavailable": goal.skeptic_unavailable,
            "last_skeptic_status": goal.last_skeptic_status,
        })
    })
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
        assert_eq!(
            after["sub_goals"][0]["notes"][0],
            "first approach broke escaped input"
        );
    }
}
