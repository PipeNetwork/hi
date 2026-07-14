use super::common::*;
use super::*;
use std::sync::Arc;

struct GoalRecordingSession {
    goals: Arc<Mutex<Vec<Goal>>>,
}

impl SessionSink for GoalRecordingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_goal(&mut self, goal: &Goal) -> anyhow::Result<()> {
        self.goals.lock().unwrap().push(goal.clone());
        Ok(())
    }
}

fn update_goal_plan_completion(steps: &[(&str, &str)]) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: "goal-plan".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": steps
                    .iter()
                    .map(|(title, status)| serde_json::json!({
                        "title": title,
                        "status": status,
                    }))
                    .collect::<Vec<_>>()
            })
            .to_string(),
        }],
        1,
        1,
    )
}

#[tokio::test]
async fn failed_verification_cannot_commit_partial_update_plan_progress() {
    let workspace = IsolatedWorkspace::new("goal-provisional-partial");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.review = ReviewPolicy::Off;
    cfg.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.max_verify_repairs = 0;
    let responses = vec![
        write_completion(&changed.to_string_lossy()),
        update_goal_plan_completion(&[("step one", "done"), ("step two", "active")]),
        completion(vec![Content::Text("Finished the first step.".into())], 1, 1),
        completion(
            vec![Content::Text("The verifier is still failing.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    agent
        .set_structured_goal(Some(Goal::new(
            "ship the parser",
            vec!["step one".into(), "step two".into()],
        )))
        .unwrap();

    let outcome = agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    assert_eq!(outcome.verification, VerificationStatus::Failed);
    let goal = agent.structured_goal().unwrap();
    assert_eq!(goal.active_index(), Some(0));
    assert_eq!(goal.sub_goals[0].status, GoalStatus::Active);
    assert_eq!(goal.sub_goals[1].status, GoalStatus::Pending);
    assert_eq!(goal.sub_goals[0].attempts, 1);
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|note| note.contains("verification failed"))
    );
}

#[tokio::test]
async fn failed_verification_cannot_mark_entire_goal_done() {
    let workspace = IsolatedWorkspace::new("goal-provisional-done-failure");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.review = ReviewPolicy::Off;
    cfg.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.max_verify_repairs = 0;
    let responses = vec![
        write_completion(&changed.to_string_lossy()),
        update_goal_plan_completion(&[("only step", "done")]),
        completion(vec![Content::Text("Everything is done.".into())], 1, 1),
        completion(
            vec![Content::Text("The verifier is still failing.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    agent
        .set_structured_goal(Some(Goal::new("ship the parser", vec!["only step".into()])))
        .unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(goal.status, GoalStatus::Active);
    assert_eq!(goal.active_index(), Some(0));
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("long-horizon goal complete")),
        "completion must not be announced before a passing verifier: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn provider_failure_after_update_plan_cannot_leak_goal_progress() {
    let workspace = IsolatedWorkspace::new("goal-provisional-provider-error");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.review = ReviewPolicy::Off;
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(update_goal_plan_completion(&[("only step", "done")])),
            ProviderStep::Error(ProviderErrorKind::Auth),
        ],
        cfg,
    );
    agent
        .set_structured_goal(Some(Goal::new("ship it", vec!["only step".into()])))
        .unwrap();

    assert!(agent.run_turn("go", &mut RecUi::default()).await.is_err());

    let goal = agent.structured_goal().unwrap();
    assert_eq!(goal.status, GoalStatus::Active);
    assert_eq!(goal.active_index(), Some(0));
}

#[tokio::test]
async fn verified_update_plan_completion_is_persisted_as_done() {
    let workspace = IsolatedWorkspace::new("goal-provisional-done-pass");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.review = ReviewPolicy::Off;
    cfg.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let responses = vec![
        write_completion(&changed.to_string_lossy()),
        update_goal_plan_completion(&[("only step", "done")]),
        completion(vec![Content::Text("Everything is done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let goals = Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(GoalRecordingSession {
        goals: goals.clone(),
    }));
    agent
        .set_structured_goal(Some(Goal::new("ship the parser", vec!["only step".into()])))
        .unwrap();

    let outcome = agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(agent.structured_goal().unwrap().status, GoalStatus::Done);
    assert_eq!(
        goals.lock().unwrap().last().unwrap().status,
        GoalStatus::Done
    );
}

#[tokio::test]
async fn exact_plan_goal_continuation_uses_real_context_and_implementation_guards() {
    let workspace = IsolatedWorkspace::new("goal-plan-context");
    std::fs::write(
        workspace.path("plan.md"),
        "Build the parser and preserve all regression tests.",
    )
    .unwrap();
    let changed = workspace.path("built.rs");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.review = ReviewPolicy::Off;
    cfg.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&changed.to_string_lossy())),
            ProviderStep::Completion(bash_completion("cargo test --help")),
            ProviderStep::Completion(completion(
                vec![Content::Text("Implemented and validated.".into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text("All implementation work is complete.".into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(vec![Content::Text("Done.".into())], 1, 1)),
        ],
        cfg,
    );
    agent
        .set_structured_goal(Some(Goal::new(
            "review the plan.md document and fully build this",
            vec!["Implement every requirement in plan.md".into()],
        )))
        .unwrap();

    let mut ui = RecUi::default();
    let outcome = agent.run_turn(GOAL_CONTINUE_PROMPT, &mut ui).await.unwrap();

    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "outcome={outcome:?}, statuses={:?}, telemetry={:?}",
        ui.statuses,
        agent.last_turn_telemetry()
    );
    assert_eq!(agent.last_turn_telemetry().effective_max_steps, 120);
    assert_eq!(
        agent.last_task_contract.as_ref().unwrap().referenced_paths,
        vec!["plan.md"]
    );
    let first_request = &requests.lock().unwrap()[0];
    let request_text = first_request
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(request_text.contains("Objective: review the plan.md document"));
    assert!(request_text.contains("Implementation guard:"));
    assert!(request_text.contains("# Task context index"));
    assert!(request_text.contains("plan.md"));
}
