/// After a turn, enqueue the Agent-owned leftover-work drive when it says so.
pub(crate) fn pending_drive_prompt(
    agent: &hi_agent::Agent,
    outcome: Option<&hi_agent::TurnOutcome>,
) -> Option<String> {
    agent.drive_decision(outcome).prompt().map(str::to_string)
}

pub(crate) fn long_horizon_enabled(is_subagent: bool) -> bool {
    !is_subagent
}

/// One-shot leftover after the drive loop: unfinished plan or goal work,
/// including parked leftover. Paused leftover also counts.
pub(crate) fn one_shot_leftover_remains(agent: &hi_agent::Agent) -> bool {
    agent.plan_leftover_work().is_some()
        || agent
            .structured_goal()
            .is_some_and(hi_agent::Goal::has_drive_work)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resumed_active_goal_is_queued_before_repl_input() {
        let initial_goal_drive = |goal: Option<&hi_agent::Goal>| {
            goal.is_some_and(hi_agent::Goal::should_auto_drive)
                .then(|| hi_agent::GOAL_CONTINUE_PROMPT.to_string())
        };
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

    fn test_agent() -> (std::path::PathBuf, hi_agent::Agent) {
        let root = std::env::temp_dir().join(format!(
            "hi-cli-plan-drive-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let config = hi_agent::AgentConfig {
            paths: hi_agent::AgentPaths {
                workspace_root: root.clone(),
                state_root: root.join(".hi-state"),
            },
            routing: hi_agent::AgentRouting {
                model: "test-model".into(),
                ..hi_agent::AgentRouting::default()
            },
            subagents: hi_agent::AgentSubagents {
                long_horizon: true,
                ..hi_agent::AgentSubagents::default()
            },
            ..hi_agent::AgentConfig::default()
        };
        let agent = hi_agent::Agent::new(
            std::sync::Arc::new(hi_ai::OpenAiProvider::new(
                "http://127.0.0.1:1/v1".into(),
                "test".into(),
            )),
            config,
        )
        .unwrap();
        (root, agent)
    }

    fn completed_outcome(agent: &hi_agent::Agent) -> hi_agent::TurnOutcome {
        hi_agent::TurnOutcome {
            status: hi_agent::TurnStatus::Completed,
            verification: hi_agent::VerificationStatus::Unverified,
            review: hi_agent::ReviewStatus::NotRequired,
            stop_reason: hi_agent::TurnStopReason::Completed,
            changed_files: Vec::new(),
            verified_workspace_revision: None,
            effective_route: hi_agent::EffectiveModelRoute {
                provider: Some("test".into()),
                model: "m".into(),
            },
            review_same_model: false,
            leftover: agent.leftover_work(),
            plan_leftover: agent.plan_leftover_work(),
        }
    }

    #[test]
    fn completed_leftover_enqueues_plan_drive() {
        let (root, mut agent) = test_agent();
        agent.restore_plan(vec![hi_agent::PlanStep {
            title: "wire the scheduler".into(),
            status: hi_agent::PlanStatus::Pending,
        }]);
        let outcome = completed_outcome(&agent);
        assert_eq!(
            pending_drive_prompt(&agent, Some(&outcome)).as_deref(),
            Some(hi_agent::PLAN_DRIVE_PROMPT)
        );
        agent.set_plan_drive_paused(true);
        assert_eq!(pending_drive_prompt(&agent, Some(&outcome)), None);
        drop(agent);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn goal_leftover_wins_over_plan_and_pause_stops_enqueue() {
        let (root, mut agent) = test_agent();
        agent.restore_plan(vec![hi_agent::PlanStep {
            title: "wire the scheduler".into(),
            status: hi_agent::PlanStatus::Pending,
        }]);
        assert!(
            agent
                .set_structured_goal(Some(hi_agent::Goal::new(
                    "ship it",
                    vec!["implement it".into()],
                )))
                .unwrap()
        );
        let outcome = completed_outcome(&agent);
        assert_eq!(
            pending_drive_prompt(&agent, Some(&outcome)).as_deref(),
            Some(hi_agent::GOAL_CONTINUE_PROMPT)
        );
        assert!(
            agent
                .try_set_goal_pause_reason(hi_agent::GoalPauseReason::User)
                .unwrap()
        );
        assert_eq!(pending_drive_prompt(&agent, Some(&outcome)), None);
        drop(agent);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn structured_goals_are_enabled_for_every_top_level_provider() {
        assert!(long_horizon_enabled(false));
        assert!(!long_horizon_enabled(true));
    }
}
