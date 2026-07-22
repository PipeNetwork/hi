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
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.gates.max_verify_repairs = 0;
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
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.gates.max_verify_repairs = 0;
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
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
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
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let responses = vec![
        write_completion(&changed.to_string_lossy()),
        update_goal_plan_completion(&[("only step", "done")]),
        completion(vec![Content::Text("Everything is done.".into())], 1, 1),
        // The completion auditor approves the finish.
        completion(vec![Content::Text("COMPLETE".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let goals = Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(GoalRecordingSession {
        goals: goals.clone(),
    }));
    let mut done_goal = Goal::new("ship the parser", vec!["only step".into()]);
    done_goal.team = false; // isolate the commit gate from the default-on skeptic
    agent.set_structured_goal(Some(done_goal)).unwrap();

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
async fn verified_bulk_done_update_plan_advances_only_one_step() {
    // The small-quant production failure: right after scaffolding stubs, the
    // model's update_plan marked (nearly) every milestone done and the whole
    // goal completed after two drive turns. Bounded application must cap a
    // verified turn at one advance and keep the drive alive.
    let workspace = IsolatedWorkspace::new("goal-bulk-done-bounded");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let responses = vec![
        write_completion(&changed.to_string_lossy()),
        update_goal_plan_completion(&[
            ("scaffold", "done"),
            ("implement parser", "done"),
            ("implement runtime", "done"),
            ("implement kernels", "done"),
        ]),
        completion(vec![Content::Text("All milestones complete.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let goals = Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(GoalRecordingSession {
        goals: goals.clone(),
    }));
    let mut bulk_goal = Goal::new(
        "fully implement the plan",
        vec![
            "scaffold".into(),
            "implement parser".into(),
            "implement runtime".into(),
            "implement kernels".into(),
        ],
    );
    bulk_goal.team = false; // isolate the anchor rule from the default-on skeptic
    agent.set_structured_goal(Some(bulk_goal)).unwrap();

    let outcome = agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    assert_eq!(outcome.verification, VerificationStatus::Passed);
    let goal = agent.structured_goal().unwrap();
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Done,
        "anchor advanced"
    );
    assert_eq!(goal.active_index(), Some(1), "exactly one step per turn");
    for sub_goal in &goal.sub_goals[2..] {
        assert_eq!(sub_goal.status, GoalStatus::Pending, "no teleport to Done");
        assert_eq!(sub_goal.attempts, 0);
        assert!(
            sub_goal.notes.iter().any(|note| note == CLAIM_NOTE),
            "bulk done-claim downgraded to a note: {:?}",
            sub_goal.notes
        );
    }
    assert_eq!(goal.status, GoalStatus::Active, "goal is NOT done");
    assert!(goal.should_auto_drive(), "auto-drive must keep going");
    let persisted = goals.lock().unwrap();
    let last = persisted.last().unwrap();
    assert_eq!(last.status, GoalStatus::Active);
    assert_eq!(last.active_index(), Some(1));
}

#[tokio::test]
async fn multiple_update_plan_calls_in_one_turn_advance_one_step() {
    let workspace = IsolatedWorkspace::new("goal-multi-plan-one-advance");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let responses = vec![
        write_completion(&changed.to_string_lossy()),
        update_goal_plan_completion(&[
            ("step one", "done"),
            ("step two", "active"),
            ("step three", "pending"),
        ]),
        update_goal_plan_completion(&[
            ("step one", "done"),
            ("step two", "done"),
            ("step three", "active"),
        ]),
        completion(vec![Content::Text("Two steps in one turn!".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut multi_plan_goal = Goal::new(
        "ship the parser",
        vec!["step one".into(), "step two".into(), "step three".into()],
    );
    multi_plan_goal.team = false; // isolate the anchor rule from the default-on skeptic
    agent.set_structured_goal(Some(multi_plan_goal)).unwrap();

    let outcome = agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    assert_eq!(outcome.verification, VerificationStatus::Passed);
    let goal = agent.structured_goal().unwrap();
    assert_eq!(
        goal.active_index(),
        Some(1),
        "the second update_plan in the same turn cannot compound the advance"
    );
    assert_eq!(goal.sub_goals[2].status, GoalStatus::Pending);
    assert!(
        goal.sub_goals[1]
            .notes
            .iter()
            .any(|note| note == CLAIM_NOTE),
        "the compounding attempt is recorded as a claim note"
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
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&changed.to_string_lossy())),
            ProviderStep::Completion(bash_completion("true # validate")),
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
        agent
            .task
            .last_task_contract
            .as_ref()
            .unwrap()
            .referenced_paths,
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

#[tokio::test]
async fn appended_validation_milestone_is_rejected() {
    // The qtest failure vector: the executor's update_plan appended a "Final
    // workspace validation" step, which is structurally unwinnable and later
    // killed the goal. Meta appends must be dropped; real appends kept.
    let workspace = IsolatedWorkspace::new("goal-meta-append");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let responses = vec![
        write_completion(&changed.to_string_lossy()),
        update_goal_plan_completion(&[
            ("step one", "done"),
            ("step two", "active"),
            ("Implement the discovered exporter module", "pending"),
            ("Final workspace validation", "pending"),
        ]),
        completion(vec![Content::Text("advancing".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new("ship it", vec!["step one".into(), "step two".into()]);
    goal.team = false;
    agent.set_structured_goal(Some(goal)).unwrap();

    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(
        goal.sub_goals.len(),
        3,
        "real append kept, validation-only append dropped: {:?}",
        goal.sub_goals
            .iter()
            .map(|s| &s.description)
            .collect::<Vec<_>>()
    );
    assert!(goal.sub_goals[2].description.contains("exporter module"));
}
