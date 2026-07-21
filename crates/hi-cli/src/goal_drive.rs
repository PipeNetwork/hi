/// Seed the REPL's synthetic input queue when a durable active goal is loaded.
pub(crate) fn initial_goal_drive(goal: Option<&hi_agent::Goal>) -> Option<String> {
    goal.is_some_and(hi_agent::Goal::should_auto_drive)
        .then(|| hi_agent::GOAL_CONTINUE_PROMPT.to_string())
}

pub(crate) fn long_horizon_enabled(is_subagent: bool) -> bool {
    !is_subagent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resumed_active_goal_is_queued_before_repl_input() {
        let active = hi_agent::Goal::new("build the plan", vec!["review plan.md".into()]);
        assert_eq!(
            initial_goal_drive(Some(&active)).as_deref(),
            Some(hi_agent::GOAL_CONTINUE_PROMPT)
        );

        let mut paused = active.clone();
        paused.pause(hi_agent::GoalPauseReason::User);
        assert_eq!(initial_goal_drive(Some(&paused)), None);

        let mut done = active;
        done.advance();
        assert_eq!(initial_goal_drive(Some(&done)), None);
        assert_eq!(initial_goal_drive(None), None);
    }

    #[test]
    fn structured_goals_are_enabled_for_every_top_level_provider() {
        assert!(long_horizon_enabled(false));
        assert!(!long_horizon_enabled(true));
    }
}
