use super::common::*;
use super::*;

struct PlanRecordingSession {
    plans: std::sync::Arc<Mutex<Vec<Vec<PlanStep>>>>,
    clears: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

type PlanDrivePolicyRecords = std::sync::Arc<Mutex<Vec<(bool, u32, bool, bool, Vec<String>)>>>;

struct PlanDrivePolicyRecordingSession {
    records: PlanDrivePolicyRecords,
}

impl SessionSink for PlanDrivePolicyRecordingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> Result<()> {
        Ok(())
    }

    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        stall: u32,
        resume_on_user_input: bool,
        reset_evidence: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.records.lock().unwrap().push((
            paused,
            stall,
            resume_on_user_input,
            reset_evidence,
            evidence_add.to_vec(),
        ));
        Ok(())
    }
}

impl SessionSink for PlanRecordingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> Result<()> {
        Ok(())
    }

    fn record_plan(&mut self, plan: &[PlanStep]) -> Result<()> {
        self.plans.lock().unwrap().push(plan.to_vec());
        Ok(())
    }

    fn clear_plan(&mut self) -> Result<()> {
        self.clears
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

struct FailingPlanSession;

struct FailingPlanDriveSession;

impl SessionSink for FailingPlanSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> Result<()> {
        Ok(())
    }

    fn record_plan(&mut self, _plan: &[PlanStep]) -> Result<()> {
        Err(anyhow::anyhow!("injected plan persistence failure"))
    }
}

impl SessionSink for FailingPlanDriveSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> Result<()> {
        Ok(())
    }

    fn record_plan_drive_state_with_policy(
        &mut self,
        _paused: bool,
        _stall: u32,
        _resume_on_user_input: bool,
        _reset_evidence: bool,
        _evidence_add: &[String],
    ) -> Result<()> {
        Err(anyhow::anyhow!("injected plan-drive persistence failure"))
    }
}

#[derive(Clone)]
struct ModeTransitionRequest {
    messages: Vec<Message>,
    tool_names: Vec<String>,
}

struct ModeTransitionProvider {
    responses: Mutex<Vec<Completion>>,
    requests: std::sync::Arc<Mutex<Vec<ModeTransitionRequest>>>,
}

#[async_trait::async_trait]
impl hi_ai::Provider for ModeTransitionProvider {
    async fn stream(
        &self,
        request: hi_ai::ChatRequest,
        _sink: &mut (dyn FnMut(hi_ai::StreamEvent) + Send),
    ) -> Result<Completion> {
        self.requests.lock().unwrap().push(ModeTransitionRequest {
            messages: request.messages.to_vec(),
            tool_names: request.tools.iter().map(|tool| tool.name.clone()).collect(),
        });
        pop_canned_completion(&self.responses, "ModeTransitionProvider")
    }
}

#[tokio::test]
async fn distinct_discovery_plan_transitions_to_verified_mutation_without_a_count_cap() {
    let workspace = IsolatedWorkspace::new("mixed-review-build");
    let mut responses = Vec::new();
    // Reproduce the failed live turn: thirteen distinct reads, then a concrete
    // plan. Every read must remain available without count-based steering.
    for index in 0..13 {
        let relative = format!("src/context-{index}.rs");
        let path = workspace.path(&relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!("pub const VALUE_{index}: usize = {index};\n"),
        )
        .unwrap();
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("read-{index}"),
                name: "read".into(),
                arguments: serde_json::json!({"path": relative}).to_string(),
            }],
            1,
            1,
        ));
    }
    responses.push(completion(
        vec![Content::ToolCall {
            id: "plan-active".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Implement the selected component", "status": "active"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    let post_plan_read = workspace.path("src/post-plan-context.rs");
    std::fs::write(&post_plan_read, "final context before the edit\n").unwrap();
    responses.push(completion(
        vec![Content::ToolCall {
            id: "post-plan-read".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "src/post-plan-context.rs"}).to_string(),
        }],
        1,
        1,
    ));
    let changed = workspace.path("src/implemented.rs");
    responses.push(write_completion(&changed.to_string_lossy()));
    responses.push(completion(
        vec![Content::ToolCall {
            id: "plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Implement the selected component", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    responses.push(bash_completion("true # validate"));
    responses.push(completion(vec![Content::Text("implemented".into())], 1, 1));

    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(responses),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn("review plan.md and lets keep building this", &mut ui)
        .await
        .unwrap();

    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "outcome={outcome:?}; statuses={:?}",
        ui.statuses
    );
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.stop_reason, TurnStopReason::Completed);
    assert!(
        outcome
            .changed_files
            .iter()
            .any(|path| path == "src/implemented.rs")
    );
    assert!(outcome.verified_workspace_revision.is_some());
    assert!(changed.exists());
    assert!(
        ui.statuses.iter().all(
            |status| !status.contains("requesting an implementation step")
                && !status.contains("bounded discovery")
                && !status.contains("discovery budget")
        ),
        "distinct discovery must not hit a hidden count limit: {:?}",
        ui.statuses
    );
    let read_results = ui
        .tool_results
        .iter()
        .filter(|(name, _)| name == "read")
        .collect::<Vec<_>>();
    assert_eq!(read_results.len(), 14, "all scripted reads must execute");
    assert!(
        read_results
            .iter()
            .all(|(_, result)| !result.to_ascii_lowercase().contains("denied")),
        "discovery steering must never deny read: {read_results:?}"
    );
    assert!(
        agent
            .last_turn_telemetry()
            .tool_timeline
            .iter()
            .filter(|entry| entry.tool == "read")
            .all(|entry| entry.status == hi_tools::ToolStatus::Succeeded),
        "every read must have typed Succeeded status"
    );
    let guided_tools = tool_names.lock().unwrap()[10]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(guided_tools.contains("read"));
    assert!(guided_tools.contains("update_plan"));
    assert!(guided_tools.contains("write"));
    assert_ne!(modes.lock().unwrap()[10], ToolMode::ChatOnly);
    assert_ne!(modes.lock().unwrap()[12], ToolMode::ChatOnly);
    let post_plan_tools = tool_names.lock().unwrap()[14]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(post_plan_tools.contains("read"));
    assert!(post_plan_tools.contains("write"));
    assert_ne!(modes.lock().unwrap()[14], ToolMode::ChatOnly);
    assert_ne!(modes.lock().unwrap()[15], ToolMode::ChatOnly);
}

#[tokio::test]
async fn resumed_active_plan_allows_distinct_discovery_until_mutation() {
    let workspace = IsolatedWorkspace::new("resumed-plan-build");
    let mut responses = Vec::new();
    for index in 0..10 {
        let relative = format!("src/resumed-context-{index}.rs");
        std::fs::create_dir_all(workspace.path("src")).unwrap();
        std::fs::write(workspace.path(&relative), format!("context {index}\n")).unwrap();
        responses.push(completion(
            vec![Content::ToolCall {
                id: format!("resumed-read-{index}"),
                name: "read".into(),
                arguments: serde_json::json!({"path": relative}).to_string(),
            }],
            1,
            1,
        ));
    }
    let changed = workspace.path("src/resumed-implemented.rs");
    responses.push(write_completion(&changed.to_string_lossy()));
    responses.push(completion(
        vec![Content::ToolCall {
            id: "resumed-plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "Resume implementation", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    ));
    responses.push(bash_completion("true # validate"));
    responses.push(completion(vec![Content::Text("implemented".into())], 1, 1));

    let tool_names = std::sync::Arc::new(Mutex::new(Vec::new()));
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordRequests {
        responses: Mutex::new(responses),
        tool_names: tool_names.clone(),
        modes: modes.clone(),
    };
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    agent.goals.last_plan = vec![PlanStep {
        title: "Resume implementation".into(),
        status: PlanStatus::Active,
    }];
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("continue building this", &mut ui)
        .await
        .unwrap();

    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "outcome={outcome:?}; statuses={:?}",
        ui.statuses
    );
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert!(changed.exists());
    assert!(ui.statuses.iter().all(|status| {
        !status.contains("active implementation plan already exists")
            && !status.contains("discovery budget")
    }));
    let recovery_tools = tool_names.lock().unwrap()[10]
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert!(recovery_tools.contains("read"));
    assert!(recovery_tools.contains("write"));
    assert!(
        agent
            .last_turn_telemetry()
            .tool_timeline
            .iter()
            .filter(|entry| entry.tool == "read")
            .all(|entry| entry.status == hi_tools::ToolStatus::Succeeded)
    );
    assert_ne!(modes.lock().unwrap()[10], ToolMode::ChatOnly);
}

#[tokio::test]
async fn plan_with_pending_steps_continues_past_recap() {
    // The model posts a plan (2/3 done), does one step, then stops with a
    // finished-looking recap. Without plan-awareness, the text heuristic
    // sees a finished recap and ends the turn — leaving the plan at 2/3.
    // With plan-awareness, the agent detects pending steps and nudges the
    // model to continue until the plan is complete.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 5;
    // Helper: an update_plan call with given step statuses.
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"Review step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: model posts the initial plan (0/3 done) and starts step 1.
        plan_call("p1", &["active", "pending", "pending"]),
        // R2: model does a read for step 1.
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R3: model updates plan (1/3 done, step 2 active) and does a read.
        plan_call("p2", &["done", "active", "pending"]),
        // R4: model stops with a finished-looking recap — but plan is 1/3!
        // The plan-aware continue should nudge it to keep going.
        completion(
            vec![Content::Text(
                "I've completed step 1. The implementation looks good.".into(),
            )],
            1,
            1,
        ),
        // R5 (nudged): model does step 2.
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // R6: model updates plan (2/3 done, step 3 active).
        plan_call("p3", &["done", "done", "active"]),
        // R7: model stops with recap again — plan is 2/3, nudge again.
        completion(
            vec![Content::Text("Step 2 is done. Moving on.".into())],
            1,
            1,
        ),
        // R8 (nudged): model does step 3.
        completion(
            vec![Content::ToolCall {
                id: "r3".into(),
                name: "read".into(),
                arguments: r#"{"path":"z"}"#.into(),
            }],
            1,
            1,
        ),
        // R9: model updates plan (3/3 done) — all complete.
        plan_call("p4", &["done", "done", "done"]),
        // R10: model gives final recap — plan is complete, turn ends.
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("work through the plan", &mut ui)
        .await
        .unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // The turn should have run all the way to the final recap (R10),
    // not stopped at R4 or R7 when the model gave a partial recap.
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("All steps complete"),
        "turn ran to the final recap with plan complete: {:?}",
        agent.messages().last().unwrap().text()
    );
}

#[tokio::test]
async fn new_task_keeps_incomplete_plan_and_folds_replace_nudge() {
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("new task done".into())],
            1,
            1,
        )],
        config(),
    );
    agent.goals.last_plan = vec![PlanStep {
        title: "old unfinished step".into(),
        status: PlanStatus::Pending,
    }];
    let mut ui = RecUi::default();

    agent
        .run_turn("do a different task", &mut ui)
        .await
        .unwrap();

    assert_eq!(agent.goals.last_plan.len(), 1);
    assert_eq!(agent.goals.last_plan[0].title, "old unfinished step");
    assert!(
        ui.plans.is_empty(),
        "incomplete plans must stay pinned: {:?}",
        ui.plans
    );
    let prompt = agent
        .messages()
        .iter()
        .find(|m| m.role == hi_ai::Role::User)
        .map(|m| m.text())
        .unwrap_or_default();
    assert!(
        prompt.contains("If this message is a new task"),
        "replace-plan nudge should fold into the new-task prompt: {prompt}"
    );
}

#[tokio::test]
async fn continue_does_not_preserve_a_completed_plan_box() {
    let mut agent = agent(
        vec![completion(vec![Content::Text("done".into())], 1, 1)],
        config(),
    );
    agent.goals.last_plan = vec![PlanStep {
        title: "old completed step".into(),
        status: PlanStatus::Done,
    }];
    let mut ui = RecUi::default();

    agent.run_turn("continue", &mut ui).await.unwrap();

    assert!(agent.goals.last_plan.is_empty());
    assert_eq!(ui.plans, vec![Vec::<PlanStep>::new()]);
}

#[tokio::test]
async fn complete_plan_ends_turn_without_spurious_continue() {
    // When the plan is fully done (all steps "done"), the model's recap
    // should end the turn cleanly — no plan-driven continue nudge.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 5;
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"Review step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // Model posts plan (all done) and gives final recap.
        plan_call("p1", &["done", "done"]),
        completion(vec![Content::Text("All done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // No spurious continue — the turn ended after exactly 2 responses.
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no incomplete warning when plan is done: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn plan_mode_cannot_mark_unexecuted_checklist_done() {
    // Regression for a live plan-mode failure: after drafting a pending plan,
    // unfinished-plan steering told a read-only planner to "do the work". The
    // model escaped the contradiction with one all-done update_plan call, and
    // the UI showed N/N despite zero edits or checks.
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "draft-all-done".into(),
                name: "update_plan".into(),
                arguments: r#"{"steps":[{"title":"Add follow graph","status":"done"},{"title":"Add thread UI","status":"done"}]}"#.into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "The implementation plan is ready for approval: first add the follow graph, then expose threads in the UI. No implementation was performed in plan mode."
                    .into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, config());
    let persisted_plans = std::sync::Arc::new(Mutex::new(Vec::<Vec<PlanStep>>::new()));
    let plan_clears = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    agent.set_session(Box::new(PlanRecordingSession {
        plans: persisted_plans.clone(),
        clears: plan_clears.clone(),
    }));
    agent.set_plan_mode(true);
    let mut ui = RecUi::default();

    let outcome = agent.run_turn("build all of that", &mut ui).await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(outcome.changed_files.is_empty());
    assert_eq!(agent.current_plan().len(), 2);
    assert!(
        agent
            .current_plan()
            .iter()
            .all(|step| step.status == PlanStatus::Pending),
        "planning cannot self-certify execution: {:?}",
        agent.current_plan()
    );
    assert!(
        ui.plans
            .last()
            .is_some_and(|plan| plan.iter().all(|step| step.status == PlanStatus::Pending)),
        "the UI must receive the constrained pending plan: {:?}",
        ui.plans
    );
    assert!(
        persisted_plans
            .lock()
            .unwrap()
            .last()
            .is_some_and(|plan| plan.iter().all(|step| step.status == PlanStatus::Pending)),
        "session persistence must retain the plan as unfinished"
    );
    assert_eq!(
        plan_clears.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "an unexecuted plan must not be persisted as cleared"
    );
    assert!(
        !ui.statuses.iter().any(|status| {
            status.contains("incomplete steps")
                || status.contains("no successful file changes")
                || status.contains("read-only review answer")
        }),
        "plan mode must not enter execution/review repair: {:?}",
        ui.statuses
    );

    agent.set_plan_mode(false);
    assert!(
        agent.drive_decision(Some(&outcome)).should_enqueue(),
        "leaving plan mode must expose the real pending implementation work"
    );
}

#[tokio::test]
async fn failed_plan_persistence_does_not_publish_live_only_state() {
    let mut agent = agent(
        vec![completion(
            vec![Content::ToolCall {
                id: "persist-fails".into(),
                name: "update_plan".into(),
                arguments: r#"{"steps":[{"title":"Build parser","status":"active"}]}"#.into(),
            }],
            1,
            1,
        )],
        config(),
    );
    agent.set_session(Box::new(FailingPlanSession));

    let error = agent
        .run_turn("build the parser", &mut RecUi::default())
        .await
        .expect_err("injected durable write must fail the tool batch");

    assert!(
        error
            .to_string()
            .contains("injected plan persistence failure")
    );
    assert!(
        agent.current_plan().is_empty(),
        "a plan that never became durable leaked into live state: {:?}",
        agent.current_plan()
    );
}

#[tokio::test]
async fn synthetic_plan_drive_cannot_complete_an_implementation_step_before_it_edits() {
    let workspace = IsolatedWorkspace::new("plan-drive-done-needs-evidence");
    let changed = workspace.path("vote_transaction.rs");
    let title = "Build VoteInstruction::Vote transaction from tower decision";
    let update = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [{"title": title, "status": "done"}]
                })
                .to_string(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        update("unsupported-done"),
        write_completion(&changed.to_string_lossy()),
        update("supported-done"),
        completion(
            vec![Content::Text(
                "Implemented the vote transaction and retained the completed plan step.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    let mut agent = agent(responses, cfg);
    agent.restore_plan(vec![PlanStep {
        title: title.into(),
        status: PlanStatus::Active,
    }]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .expect("plan drive should recover from the unsupported claim and edit");

    assert!(changed.is_file(), "the corrective mutation did not run");
    assert!(
        !agent.plan_incomplete(),
        "evidenced completion was not retained"
    );
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        ui.statuses.iter().any(|status| {
            status.contains("kept 1 unsupported implementation completion claim")
        }),
        "the zero-evidence completion was not rejected: {:?}",
        ui.statuses
    );
    assert!(
        ui.plans.iter().any(|plan| {
            plan.first()
                .is_some_and(|step| step.status == PlanStatus::Active)
        }),
        "the active step disappeared before the write: {:?}",
        ui.plans
    );
    assert!(
        ui.plans.last().is_some_and(|plan| {
            plan.first()
                .is_some_and(|step| step.status == PlanStatus::Done)
        }),
        "the post-mutation completion was not accepted: {:?}",
        ui.plans
    );
}

#[tokio::test]
async fn stalled_plan_drive_starts_a_fresh_strategy_epoch_without_renewing_evidence() {
    let workspace = IsolatedWorkspace::new("plan-drive-fresh-strategy");
    let changed = workspace.path("recovered.rs");
    let title = "Implement vote transaction signing";
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = ModeTransitionProvider {
        responses: Mutex::new(vec![
            write_completion(&changed.to_string_lossy()),
            completion(
                vec![Content::ToolCall {
                    id: "done-after-recovery".into(),
                    name: "update_plan".into(),
                    arguments: serde_json::json!({
                        "steps": [{"title": title, "status": "done"}]
                    })
                    .to_string(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text("Recovered and implemented.".into())],
                1,
                1,
            ),
        ]),
        requests: requests.clone(),
    };
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let evidence_hash = "b".repeat(64);
    agent.restore_plan(vec![PlanStep {
        title: title.into(),
        status: PlanStatus::Active,
    }]);
    agent.restore_plan_drive_with_policy(false, false, 1, vec![evidence_hash.clone()]);
    agent
        .messages_mut()
        .push(Message::user("old synthetic drive"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "Now I have everything I need, but I will read it again.".into(),
        )]));
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .expect("fresh strategy epoch should reach the corrective write");

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(changed.is_file());
    assert_eq!(
        agent.plan_drive_stall(),
        1,
        "turn-local recovery reset the stall"
    );
    assert_eq!(agent.plan_drive_evidence.snapshot(), vec![evidence_hash]);
    let first = requests.lock().unwrap().first().cloned().unwrap();
    let prompt = first
        .messages
        .iter()
        .filter(|message| message.role == Role::User)
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(prompt.contains("[hi:plan-recovery]"), "prompt={prompt}");
    assert!(
        prompt.contains("make the smallest safe mutation"),
        "prompt={prompt}"
    );
    assert!(
        first
            .messages
            .iter()
            .all(|message| !message.text().contains("old synthetic drive")),
        "poisoned prior drive leaked into the fresh request: {:?}",
        first.messages
    );
}

#[tokio::test]
async fn plan_off_removes_stale_controls_and_executes_build_all_follow_up() {
    // Regression for a live session where `/plan off` restored all mutation
    // tools, but the provider history still contained the old imperative
    // "You are in PLAN MODE" user wrapper. The model kept refusing to edit.
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "draft".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [{"title": "Build the feature", "status": "pending"}]
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "I'm in plan mode. The implementation plan is ready.".into(),
            )],
            1,
            1,
        ),
        write_completion("implemented.txt"),
        bash_completion("true # validate"),
        completion(
            vec![Content::ToolCall {
                id: "implemented".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [{"title": "Build the feature", "status": "done"}]
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "Implemented implemented.txt and validated the requested change.".into(),
            )],
            1,
            1,
        ),
    ];
    let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = ModeTransitionProvider {
        responses: Mutex::new(responses),
        requests: requests.clone(),
    };
    let mut cfg = config();
    cfg.gates.allow_unverified = true;
    let workspace = cfg.paths.workspace_root.clone();
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    agent.set_plan_mode(true);
    agent
        .run_turn("create a robust plan", &mut ui)
        .await
        .unwrap();
    assert!(agent.plan_incomplete());

    agent.set_plan_mode(false);
    agent.set_permission_mode(PermissionMode::Always);
    let result = agent.run_turn("build all of that", &mut ui).await;
    assert!(
        result.is_ok(),
        "execution failed: {result:?}; requests={}; statuses={:?}",
        requests.lock().unwrap().len(),
        ui.statuses
    );
    let outcome = result.unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(workspace.join("implemented.txt").is_file());
    let requests = requests.lock().unwrap();
    let execution_request = requests
        .get(2)
        .expect("the first request after leaving plan mode");
    assert!(
        execution_request
            .tool_names
            .iter()
            .any(|name| name == "write"),
        "mutation tools must be restored: {:?}",
        execution_request.tool_names
    );
    assert!(execution_request.messages.iter().any(|message| {
        message.role == Role::Assistant && message.text().contains("I'm in plan mode")
    }));
    assert!(execution_request.messages.iter().all(|message| {
        message.role != Role::User
            || (!message.text().contains("Plan mode is ON for this turn")
                && !message.text().contains("Read-only review guard:"))
    }));
    let current_user = execution_request
        .messages
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .expect("current user message")
        .text();
    assert!(current_user.contains("Plan mode is OFF for this turn"));
    assert!(current_user.contains("build all of that"));
}

#[test]
fn resumed_legacy_plan_session_is_clean_before_first_turn() {
    let legacy = format!(
        "{}\nold task index\n{}\n\n\
You are in PLAN MODE. Do not modify files or run mutating commands.\n\
Produce a clear plan and wait.\n\nUser request:\ncreate the profiles page\n\n\
Read-only review guard: use only the currently advertised read-only inspection tools; never invent tool names. Do not write.",
        crate::transcript::CONTEXT_BLOCK_START,
        crate::transcript::CONTEXT_BLOCK_END,
    );
    let agent = resumed_agent(
        vec![
            Message::system("system"),
            Message::user(legacy),
            Message::assistant(vec![Content::Text("I'm in plan mode; plan ready.".into())]),
        ],
        Usage::default(),
        None,
        config(),
    );

    let user = agent
        .messages()
        .iter()
        .find(|message| message.role == Role::User)
        .expect("legacy user request")
        .text();
    assert_eq!(user, "create the profiles page");
    assert!(!agent.plan_mode(), "resumed sessions start in normal mode");
}

#[tokio::test]
async fn long_plan_10_steps_runs_to_completion() {
    // A 10-step plan where the model does one step per round, then stops
    // with a recap. The plan-aware continue should nudge it to keep going
    // until all 10 steps are done. The silent_continues counter resets on
    // each tool call, so this should work regardless of plan length.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3; // the default
    // The dynamic catalog omits coordination tools for ordinary read-only
    // requests; this fixture specifically exercises update_plan mechanics.
    cfg.memory.tool_set = ToolSet::Full;
    let n_steps = 10;
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"Review step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let read_call = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        )
    };
    let recap = |text: &str| completion(vec![Content::Text(text.into())], 1, 1);

    let mut responses = Vec::new();
    for step in 0..n_steps {
        // Build statuses: steps before `step` are done, step `step` is active,
        // steps after are pending.
        let statuses: Vec<&str> = (0..n_steps)
            .map(|i| {
                if i < step {
                    "done"
                } else if i == step {
                    "active"
                } else {
                    "pending"
                }
            })
            .collect();
        // Model posts plan + does a read for this step.
        responses.push(plan_call(&format!("p{step}"), &statuses));
        responses.push(read_call(&format!("r{step}")));
        // Model stops with a recap (unless it's the last step).
        if step < n_steps - 1 {
            responses.push(recap(&format!(
                "Step {} is done. The implementation looks good.",
                step + 1
            )));
        }
    }
    // Final: all steps done + final recap.
    let all_done: Vec<&str> = (0..n_steps).map(|_| "done").collect();
    responses.push(plan_call("pfinal", &all_done));
    responses.push(recap("All 10 steps complete. Done."));

    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        // This fixture exercises plan continuation with inspection-only tool
        // calls. Keep the request explicitly read-only so the mutation
        // contract does not correctly stop it after bounded discovery.
        .run_turn("review the feature plan", &mut ui)
        .await
        .unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // The turn should have run all the way to the final recap.
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All 10 steps complete"),
        "turn ran to the final recap, got: {last_text}; statuses: {:?}",
        ui.statuses,
    );
    // Should NOT have ended with an incomplete warning.
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no incomplete warning on a completed 10-step plan: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn long_plan_survives_text_only_response_to_nudge() {
    // A plan where the model sometimes responds to the continue-nudge with
    // text-only (no tool call) before eventually doing the work. This is
    // the real-world pattern that causes stalls: the model writes a recap,
    // gets nudged, writes another recap instead of acting, gets nudged
    // again, and eventually does the work. The silent_continues budget
    // must be high enough to survive a few text-only responses.
    //
    // With max_silent_continues=3, the model can text-only 3 times in a
    // row before the turn ends. On the 4th text-only, the budget is
    // exhausted. This test has 3 text-only responses (within budget)
    // before the model finally acts.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3;
    let plan_call = |id: &str, s1: &str, s2: &str, s3: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"Review a","status":"{s1}"}},{{"title":"Review b","status":"{s2}"}},{{"title":"Review c","status":"{s3}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: plan + read for step 1.
        plan_call("p1", "active", "pending", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R2: recap, no tools → nudge (silent_continues=1, force_tools).
        completion(vec![Content::Text("Step 1 done. Looks good.".into())], 1, 1),
        // R3: text-only again (ignores force) → nudge (silent_continues=2).
        completion(
            vec![Content::Text(
                "The implementation is clean. No issues found.".into(),
            )],
            1,
            1,
        ),
        // R4: text-only again (ignores force) → nudge (silent_continues=3).
        completion(
            vec![Content::Text("Everything looks correct so far.".into())],
            1,
            1,
        ),
        // R5: finally does a tool call → silent_continues resets to 0.
        plan_call("p2", "done", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // R6: recap → nudge (silent_continues=1).
        completion(vec![Content::Text("Step 2 done.".into())], 1, 1),
        // R7: does step 3.
        plan_call("p3", "done", "done", "active"),
        completion(
            vec![Content::ToolCall {
                id: "r3".into(),
                name: "read".into(),
                arguments: r#"{"path":"z"}"#.into(),
            }],
            1,
            1,
        ),
        // R8: all done + final recap.
        plan_call("p4", "done", "done", "done"),
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All steps complete"),
        "turn ran to completion despite text-only responses to nudges, got: {last_text}"
    );
}

#[tokio::test]
async fn plan_settles_after_max_consecutive_text_only_responses() {
    // When the model responds to the continue-nudge with text-only (no tool
    // call) more than max_silent_continues times in a row, the turn ends
    // with the pending plan left durable for a later drive. This is the safety
    // valve for a model that keeps narrating without acting. It fires after
    // exactly max_silent_continues+1 text-only responses (the original recap +
    // max_silent_continues nudged retries) without manufacturing an error.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3;
    let plan_call = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: r#"{"steps":[{"title":"a","status":"active"},{"title":"b","status":"pending"}]}"#.into(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: plan + read for step 1.
        plan_call("p1"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R2: recap → nudge (1/3).
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        // R3: text-only → nudge (2/3).
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // R4: text-only → nudge (3/3).
        completion(vec![Content::Text("Correct.".into())], 1, 1),
        // R5: text-only → budget exhausted, turn settles.
        completion(vec![Content::Text("Fine.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn ended");
    assert!(
        !ui.statuses.iter().any(|status| {
            status.contains("incomplete") || status.to_ascii_lowercase().contains("stalled")
        }),
        "bounded plan settlement must not manufacture a legacy error: {:?}",
        ui.statuses
    );
    assert!(plan_has_pending_steps(&agent.goals.last_plan));
}

#[tokio::test]
async fn keep_working_finishes_plan_after_silent_continue_budget() {
    // After the silent-continue budget is spent, production keeps working
    // in-turn instead of asking the user to retry. The model then acts and
    // finishes the plan.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 1;
    cfg.loop_limits.max_keep_working = 2;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"Review a","status":"{s1}"}},{{"title":"Review b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        plan_call("p1", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // Silent-continue budget spent; keep-working must fire here.
        completion(vec![Content::Text("Still recapping.".into())], 1, 1),
        plan_call("p2", "done", "done"),
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(
        ui.statuses.iter().any(|s| s.contains("still working")),
        "keep-working recovery should fire after silent-continue budget: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("/retry")),
        "must not ask the user to retry: {:?}",
        ui.statuses
    );
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All steps complete"),
        "turn should finish the plan, got: {last_text}"
    );
}

#[tokio::test]
async fn plan_persists_across_turns_for_continue() {
    // When a turn ends with an incomplete plan and the user types
    // "continue", the plan state should persist so the plan-aware continue
    // logic can fire. Without persistence, last_plan is cleared at the
    // start of the new turn and the agent can't detect the incomplete plan.
    let mut cfg = config();
    cfg.loop_limits.max_silent_continues = 3;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };

    // Turn 1: model posts plan (step 1 active), does step 1, then stops
    // with a recap. The plan-continue nudges, but the model text-only's
    // past the budget, so the turn ends with an incomplete plan (1/2).
    let turn1_responses = vec![
        plan_call("p1", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // Recap → nudge (1/3).
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        // Text-only → nudge (2/3).
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // Text-only → nudge (3/3).
        completion(vec![Content::Text("Correct.".into())], 1, 1),
        // Text-only → budget exhausted, turn ends.
        completion(vec![Content::Text("Fine.".into())], 1, 1),
    ];
    let mut agent = agent(turn1_responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    // Turn 1 settled normally while the plan remained 1/2.
    assert!(
        !ui.statuses.iter().any(|status| {
            status.contains("incomplete") || status.to_ascii_lowercase().contains("stalled")
        }),
        "turn 1 must not manufacture a legacy error: {:?}",
        ui.statuses
    );

    // Verify the plan state persisted after turn 1 — it should still have
    // pending steps so the plan-aware continue can fire on "continue".
    let plan_after_turn1 = &agent.goals.last_plan;
    assert!(
        plan_has_pending_steps(plan_after_turn1),
        "plan should persist with pending steps after turn 1: {:?}",
        plan_after_turn1
    );

    // Turn 2: user types "fix a different bug" (NOT "continue"). The plan
    // stays pinned; a replace-plan nudge tells the model to swap the
    // checklist if this is a new task.
    assert!(
        !looks_like_continue("fix a different bug"),
        "a new task should not look like continue"
    );
    assert!(
        looks_like_continue("continue"),
        "'continue' should look like continue"
    );
    assert!(
        plan_has_pending_steps(&agent.goals.last_plan),
        "incomplete plan must survive a new-task follow-up: {:?}",
        agent.goals.last_plan
    );
}

fn pending_step(title: &str) -> PlanStep {
    PlanStep {
        title: title.into(),
        status: PlanStatus::Pending,
    }
}

fn completed_outcome(leftover: Option<String>) -> TurnOutcome {
    TurnOutcome {
        status: TurnStatus::Completed,
        verification: VerificationStatus::Unverified,
        review: ReviewStatus::NotRequired,
        stop_reason: TurnStopReason::Completed,
        changed_files: Vec::new(),
        verified_workspace_revision: None,
        effective_route: crate::EffectiveModelRoute {
            provider: Some("test".into()),
            model: "m".into(),
        },
        review_same_model: false,
        leftover: leftover.clone(),
        plan_leftover: leftover,
    }
}

#[tokio::test]
async fn plan_drive_prompt_finishes_next_pending_step() {
    let workspace = IsolatedWorkspace::new("plan-drive-finish");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        plan_call("p1", "active", "pending"),
        write_completion(&workspace.path("src/a.rs").to_string_lossy()),
        plan_call("p1-done", "done", "pending"),
        completion(vec![Content::Text("Step a is done.".into())], 1, 1),
        write_completion(&workspace.path("src/b.rs").to_string_lossy()),
        plan_call("p2", "done", "done"),
        completion(vec![Content::Text("All steps complete.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let first = agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(
        agent.plan_incomplete(),
        "first turn should leave step b pending: {:?}",
        agent.current_plan()
    );
    assert!(
        agent.drive_decision(Some(&first)).should_enqueue(),
        "leftover plan should auto-drive after the first turn: {first:?}"
    );

    let second = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .unwrap();
    assert!(
        !agent.plan_incomplete(),
        "plan-drive should finish the remaining step: {:?}",
        agent.current_plan()
    );
    assert!(!agent.drive_decision(Some(&second)).should_enqueue());
}

#[tokio::test]
async fn repeated_generic_completion_fails_one_plan_drive_without_parking_loop() {
    let workspace = IsolatedWorkspace::new("plan-drive-generic-completion");
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let generic = || {
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        )
    };
    let mut agent = agent(vec![generic(), generic()], cfg);
    agent.restore_plan(vec![pending_step("wire timeline replies and likes")]);
    let mut ui = RecUi::default();

    let error = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .unwrap_err();

    assert!(
        error.to_string().contains("no usable final answer"),
        "unexpected bounded provider error: {error:#}"
    );
    assert!(
        !ui.assistant.contains("Completed the requested action"),
        "a rejected completion claim leaked into the UI: {}",
        ui.assistant
    );
    assert!(
        agent.plan_incomplete(),
        "the unfinished plan must remain durable"
    );
    assert_eq!(
        agent.plan_drive_stall(),
        0,
        "one failed provider attempt must not manufacture four completed drive turns"
    );
    let cleanup = agent
        .cleanup_turn(crate::TurnCleanupKind::Fail)
        .await
        .unwrap();
    assert_eq!(
        cleanup.outcome.stop_reason,
        crate::TurnStopReason::InfrastructureFailure
    );
    assert!(
        !agent
            .drive_decision(Some(&cleanup.outcome))
            .should_enqueue(),
        "the frontend must not auto-queue another identical plan drive after provider exhaustion"
    );
}

#[tokio::test]
async fn provider_exhaustion_after_mutation_is_infrastructure_and_does_not_auto_drive() {
    let workspace = IsolatedWorkspace::new("plan-drive-provider-exhausted-after-write");
    let changed = workspace.path("main.rs");
    let mut cfg = workspace.config();
    cfg.loop_limits.max_empty_retries = 0;
    cfg.loop_limits.max_keep_working = 0;
    let (mut agent, _) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&changed.to_string_lossy())),
            ProviderStep::Error(hi_ai::ProviderErrorKind::EmptyCompletion),
        ],
        cfg,
    );
    agent.restore_plan(vec![
        pending_step("persist follow graph"),
        pending_step("run the final verification"),
    ]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .expect("a landed tool result must settle even if the provider then exhausts");

    assert!(changed.exists(), "the landed mutation must be retained");
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::InfrastructureFailure,
        "provider exhaustion must not masquerade as a verification failure"
    );
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Infrastructure,
        },
        "an unavailable provider must not be restarted autonomously"
    );
    assert!(
        agent.plan_incomplete(),
        "pending work must remain resumable"
    );
}

#[tokio::test]
async fn completed_plan_settles_without_an_optional_provider_recap() {
    let workspace = IsolatedWorkspace::new("plan-drive-empty-recap-after-complete");
    let changed = workspace.path("completed.txt");
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    let done = completion(
        vec![Content::ToolCall {
            id: "plan-done".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [{"title": "create the requested file", "status": "done"}]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&changed.to_string_lossy())),
            ProviderStep::Completion(done),
            ProviderStep::ErrorMessage(
                hi_ai::ProviderErrorKind::ModelUnavailable,
                "requested model is not currently serviceable".into(),
            ),
        ],
        cfg,
    );
    agent.restore_plan(vec![pending_step("create the requested file")]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .expect("a completed plan must settle without an optional recap request");

    assert!(changed.exists(), "the successful write was not retained");
    assert!(!agent.plan_incomplete(), "the completed plan was reopened");
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(
        ui.assistant
            .contains("The plan is complete and the successful tool results were retained"),
        "the truthful deterministic closeout was missing: {}",
        ui.assistant
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "the completed plan made an unnecessary provider recap request"
    );
    assert!(
        !agent.drive_decision(Some(&outcome)).should_enqueue(),
        "a completed plan must not start another synthetic drive"
    );
}

#[tokio::test]
async fn empty_stream_error_after_completed_plan_preserves_productive_outcome() {
    let workspace = IsolatedWorkspace::new("plan-drive-empty-stream-after-complete");
    let changed = workspace.path("completed.txt");
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    cfg.loop_limits.max_empty_retries = 0;
    cfg.loop_limits.max_keep_working = 0;
    // Put the mutation and final bookkeeping in one model batch so the
    // bookkeeping-only early-settlement path does not apply. This directly
    // exercises the defensive provider-error fallback for a completed plan.
    let work_and_done = completion(
        vec![
            Content::ToolCall {
                id: "write".into(),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": changed.to_string_lossy(),
                    "content": "x"
                })
                .to_string(),
            },
            Content::ToolCall {
                id: "plan-done".into(),
                name: "update_plan".into(),
                arguments: serde_json::json!({
                    "steps": [{"title": "create the requested file", "status": "done"}]
                })
                .to_string(),
            },
        ],
        1,
        1,
    );
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(work_and_done),
            ProviderStep::Error(hi_ai::ProviderErrorKind::EmptyCompletion),
        ],
        cfg,
    );
    agent.restore_plan(vec![pending_step("create the requested file")]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .expect("an empty stream must not erase completed plan evidence");

    assert!(changed.exists(), "the successful write was not retained");
    assert!(!agent.plan_incomplete(), "the completed plan was reopened");
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(
        ui.assistant.contains("The plan is complete")
            && ui.assistant.contains("did not return a final recap"),
        "the truthful deterministic closeout was missing: {}",
        ui.assistant
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "the defensive fallback made an additional provider request"
    );
    assert!(
        !agent.drive_decision(Some(&outcome)).should_enqueue(),
        "a completed plan must not start another synthetic drive"
    );
}

#[tokio::test]
async fn generic_completion_after_mutation_keeps_progress_and_plan_drive_alive() {
    let workspace = IsolatedWorkspace::new("plan-drive-generic-after-progress");
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let changed = workspace.path("main.rs");
    let generic = || {
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        )
    };
    let mut agent = agent(
        vec![
            write_completion(&changed.to_string_lossy()),
            generic(),
            generic(),
        ],
        cfg,
    );
    agent.restore_plan(vec![
        pending_step("persist follow graph"),
        pending_step("run the final verification"),
    ]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .expect("a bad summary must not fail a productive plan turn");

    assert!(
        outcome
            .changed_files
            .iter()
            .any(|path| path.ends_with("main.rs")),
        "the landed mutation was lost: {outcome:?}"
    );
    assert!(
        ui.assistant
            .contains("Made concrete progress on the current step"),
        "the truthful deterministic closeout was missing: {}",
        ui.assistant
    );
    assert!(
        !ui.assistant.contains("Completed the requested action"),
        "the rejected canned summary leaked into the UI: {}",
        ui.assistant
    );
    assert!(
        agent.plan_incomplete(),
        "the remaining plan must stay durable"
    );
    assert!(
        agent.drive_decision(Some(&outcome)).should_enqueue(),
        "productive partial work should continue with the next plan drive: {outcome:?}"
    );
}

#[tokio::test]
async fn generic_completion_after_validation_keeps_final_plan_step_alive() {
    let workspace = IsolatedWorkspace::new("plan-drive-generic-after-validation");
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let generic = || {
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        )
    };
    let mut agent = agent(
        vec![bash_completion("true # validate"), generic(), generic()],
        cfg,
    );
    agent.restore_plan(vec![pending_step("run the final verification")]);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(crate::PLAN_DRIVE_PROMPT, &mut ui)
        .await
        .expect("a bad summary must not fail a successful validation turn");

    assert!(
        ui.assistant
            .contains("Made concrete progress on the current step"),
        "the truthful deterministic closeout was missing: {}",
        ui.assistant
    );
    assert!(
        agent.plan_incomplete(),
        "the final plan step must remain durable"
    );
    assert!(
        agent.drive_decision(Some(&outcome)).should_enqueue(),
        "successful validation should leave the final plan step resumable: {outcome:?}"
    );
}

#[test]
fn completed_turn_with_pending_plan_still_auto_drives() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    let leftover = agent.leftover_work();
    assert_eq!(
        leftover.as_deref(),
        Some("1/1 remaining — wire the scheduler")
    );
    let outcome = completed_outcome(leftover.clone());
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Enqueue
    );
    agent.set_plan_mode(true);
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::PlanMode
        }
    );
    agent.set_plan_mode(false);
    agent.set_plan_drive_paused(true);
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::Paused
        }
    );
}

#[test]
fn terminal_turn_with_pending_work_never_auto_drives_without_explicit_resume() {
    let mut agent = goal_agent();
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    let mut outcome = completed_outcome(agent.leftover_work());
    outcome.status = TurnStatus::Failed;
    outcome.stop_reason = TurnStopReason::InfrastructureFailure;

    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Infrastructure,
        },
        "an infrastructure failure must not start another goal or plan drive"
    );
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::Infrastructure,
        },
        "the plan-only gate must agree with the unified drive gate"
    );

    agent.report.last_turn_outcome = Some(outcome.clone());
    assert_eq!(
        agent.drive_decision(None),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Infrastructure,
        },
        "automatic drive must remain latched by the remembered infrastructure failure"
    );
    assert_eq!(
        agent.explicit_goal_drive_decision(),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal),
        "an explicit goal resume/replacement must bypass only the stale terminal outcome"
    );

    let mut cancelled = outcome;
    cancelled.status = TurnStatus::Cancelled;
    cancelled.stop_reason = TurnStopReason::Cancelled;
    agent.report.last_turn_outcome = Some(cancelled);
    assert_eq!(
        agent.drive_decision(None),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Cancelled,
        }
    );
    assert_eq!(
        agent.explicit_goal_drive_decision(),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal),
        "the user's explicit resume is also authoritative after cancellation"
    );

    let mut no_progress = completed_outcome(agent.leftover_work());
    no_progress.status = TurnStatus::Failed;
    no_progress.stop_reason = TurnStopReason::NoProgress;
    agent.report.last_turn_outcome = Some(no_progress.clone());
    assert_eq!(
        agent.drive_decision(None),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::NoProgress,
        },
        "a spent no-progress circuit must not auto-loop"
    );
    assert_eq!(
        agent.plan_drive_decision(Some(&no_progress)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::NoProgress,
        }
    );
    assert_eq!(
        agent.explicit_goal_drive_decision(),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal),
        "the user may explicitly retry after changing the approach"
    );

    let mut blocked = completed_outcome(agent.leftover_work());
    blocked.status = TurnStatus::Blocked;
    blocked.stop_reason = TurnStopReason::ToolModeDenied;
    agent.report.last_turn_outcome = Some(blocked.clone());
    assert_eq!(
        agent.drive_decision(None),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Blocked,
        },
        "a blocked turn needs user or configuration input and must not auto-loop"
    );
    assert_eq!(
        agent.plan_drive_decision(Some(&blocked)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::Blocked,
        },
        "the plan-only gate must also stop immediately after a block"
    );
    assert_eq!(
        agent.explicit_goal_drive_decision(),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal),
        "an explicit resume may retry after the user changes the blocking condition"
    );

    let mut failed = completed_outcome(agent.leftover_work());
    failed.status = TurnStatus::Failed;
    failed.verification = VerificationStatus::Failed;
    failed.stop_reason = TurnStopReason::VerificationFailed;
    assert_eq!(
        agent.drive_decision(Some(&failed)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Blocked,
        },
        "a typed failure must not start another autonomous goal or plan turn"
    );
    assert_eq!(
        agent.plan_drive_decision(Some(&failed)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::Blocked,
        },
        "the plan-only gate must fail closed for typed failures too"
    );
}

#[test]
fn parked_plan_approval_blocks_drive_without_becoming_plan_pause() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    agent.set_plan_approval_parked(true);

    assert!(!agent.plan_drive_paused());
    assert!(agent.plan_approval_parked());
    assert_eq!(
        agent.plan_drive_decision(None),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::ApprovalParked,
        }
    );
    assert_eq!(
        agent.drive_decision(None),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::PlanApprovalParked,
        }
    );

    agent.set_plan_approval_parked(false);
    assert_eq!(
        agent.plan_drive_decision(None),
        crate::PlanDriveAction::Enqueue
    );
}

#[test]
fn plan_resume_does_not_bypass_a_parked_approval() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    agent.set_plan_drive_paused(true);
    agent.set_plan_approval_parked(true);

    let effect =
        crate::handle_session_command(&mut agent, &crate::Command::Plan("resume".into()), &[])
            .expect("plan command");

    assert!(!agent.plan_drive_paused(), "resume consumes only the pause");
    assert!(agent.plan_approval_parked());
    assert!(effect.follow_up_prompt.is_none());
    assert!(effect.message.contains("/view-plan"));
}

#[test]
fn four_no_progress_plan_drives_park_and_mutation_resets() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    let outcome = completed_outcome(agent.leftover_work());
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Enqueue
    );
    for _ in 0..4 {
        agent.note_plan_drive_progress(false);
    }
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Idle {
            reason: crate::PlanDriveIdleReason::Parked
        }
    );
    assert_eq!(agent.plan_drive_status(), "parked");
    agent.note_plan_drive_progress(true);
    assert_eq!(
        agent.plan_drive_decision(Some(&outcome)),
        crate::PlanDriveAction::Enqueue
    );
}

fn set_signed_drive_evidence(agent: &mut Agent, signature: &str) {
    agent.report.last_turn_telemetry.progress_events = vec![crate::ProgressEvent {
        kind: "meaningful".into(),
        reason: "new file evidence".into(),
        signature: Some(signature.into()),
    }];
    agent.workspace.last_changed_files.clear();
}

#[test]
fn repeated_cross_turn_evidence_parks_but_distinct_read_only_evidence_does_not() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("review the existing behavior")]);

    // Five distinct evidence turns remain productive; this is the shape used by
    // the curated PTY regression and must not acquire a hidden count ceiling.
    for index in 0..5 {
        set_signed_drive_evidence(&mut agent, &format!("read:evidence-{index}:1:default"));
        let made_progress =
            agent.plan_drive_turn_made_progress(Some("review the existing behavior"));
        assert!(made_progress, "distinct evidence {index} was rejected");
        agent.note_plan_drive_progress(made_progress);
        assert_eq!(agent.plan_drive_stall(), 0);
    }

    // Cycling through that finite set cannot reset the stall forever. Every
    // signature has already been credited in this structural scope.
    for index in 0..crate::PLAN_DRIVE_STALL_LIMIT {
        set_signed_drive_evidence(
            &mut agent,
            &format!("read:evidence-{}:1:default", index % 5),
        );
        let made_progress =
            agent.plan_drive_turn_made_progress(Some("review the existing behavior"));
        assert!(!made_progress, "repeated evidence {index} looked novel");
        agent.note_plan_drive_progress(made_progress);
    }
    assert_eq!(agent.plan_drive_status(), "parked");
}

#[test]
fn implementation_step_only_credits_one_read_only_orientation_turn() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step(
        "Build VoteInstruction transaction from tower decision",
    )]);

    set_signed_drive_evidence(&mut agent, "read:src/network.rs:1432:45");
    let oriented = agent.plan_drive_turn_made_progress(Some(
        "Build VoteInstruction transaction from tower decision",
    ));
    assert!(oriented, "the initial orientation turn should count");
    agent.note_plan_drive_progress(oriented);
    assert_eq!(agent.plan_drive_stall(), 0);

    // New paths and offsets no longer erase the stall once this implementation
    // step has been oriented. This is the exact shape of the 1,367-call live
    // regression: every turn found something novel, but no turn edited code.
    for attempt in 1..=crate::PLAN_DRIVE_STALL_LIMIT {
        set_signed_drive_evidence(
            &mut agent,
            &format!("read:src/runtime-{attempt}.rs:{attempt}:45"),
        );
        let made_progress = agent.plan_drive_turn_made_progress(Some(
            "Build VoteInstruction transaction from tower decision",
        ));
        assert!(
            !made_progress,
            "novel inspection {attempt} incorrectly reset implementation progress"
        );
        agent.note_plan_drive_progress(made_progress);
        assert_eq!(agent.plan_drive_stall(), attempt);
    }

    assert_eq!(agent.plan_drive_status(), "parked");

    // The circuit is semantic rather than a total-turn ceiling: once real
    // implementation lands (for example after an explicit resume), it clears
    // the stall and the next work may continue normally.
    agent.workspace.last_changed_files = vec!["src/vote.rs".into()];
    let made_progress = agent.plan_drive_turn_made_progress(Some(
        "Build VoteInstruction transaction from tower decision",
    ));
    assert!(made_progress);
    agent.note_plan_drive_progress(made_progress);
    assert_eq!(agent.plan_drive_stall(), 0);
    assert_eq!(agent.plan_drive_status(), "running");
    assert!(agent.plan_drive_evidence.is_empty());
}

#[test]
fn restored_implementation_orientation_cannot_be_renewed_with_novel_reads() {
    let step = "Wire vote transaction signing and submission";
    let mut first = agent(Vec::new(), config());
    first.restore_plan(vec![pending_step(step)]);
    set_signed_drive_evidence(&mut first, "read:src/network.rs:1400:45");
    assert!(first.plan_drive_turn_made_progress(Some(step)));

    let mut resumed = agent(Vec::new(), config());
    resumed.restore_plan(vec![pending_step(step)]);
    resumed.restore_plan_drive_with_policy(false, false, 0, first.plan_drive_evidence.snapshot());
    set_signed_drive_evidence(&mut resumed, "read:src/validator.rs:900:45");

    assert!(
        !resumed.plan_drive_turn_made_progress(Some(step)),
        "restart plus a novel read must not grant another orientation turn"
    );
}

#[test]
fn restored_drive_evidence_remains_non_novel_until_scope_changes() {
    let mut first = agent(Vec::new(), config());
    first.restore_plan(vec![pending_step("investigate")]);
    set_signed_drive_evidence(&mut first, "read:src/lib.rs:1:default");
    assert!(first.plan_drive_turn_made_progress(Some("investigate")));
    let persisted = first.plan_drive_evidence.snapshot();
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].len(), 64);
    assert!(!persisted[0].contains("src/lib.rs"));

    let mut resumed = agent(Vec::new(), config());
    resumed.restore_plan(vec![pending_step("investigate")]);
    resumed.restore_plan_drive_with_policy(false, false, 0, persisted);
    set_signed_drive_evidence(&mut resumed, "read:src/lib.rs:1:default");
    assert!(
        !resumed.plan_drive_turn_made_progress(Some("investigate")),
        "restart must not make already-credited evidence novel"
    );

    // A real step transition starts a new scope, so the same file can be
    // legitimately inspected again for the next piece of work.
    assert!(resumed.plan_drive_turn_made_progress(Some("previous step")));
    set_signed_drive_evidence(&mut resumed, "read:src/lib.rs:1:default");
    assert!(resumed.plan_drive_turn_made_progress(Some("investigate")));
}

#[test]
fn goal_drive_uses_the_same_cross_turn_novelty_ledger() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new("ship", vec!["investigate".into()],)))
            .unwrap()
    );
    let before = agent.structured_goal().cloned();
    set_signed_drive_evidence(&mut agent, "grep:needle::src:0");
    assert!(agent.goal_drive_turn_made_progress(before.as_ref()));
    set_signed_drive_evidence(&mut agent, "grep:needle::src:0");
    assert!(!agent.goal_drive_turn_made_progress(before.as_ref()));

    agent.begin_drive_turn(crate::DriveKind::User).unwrap();
    set_signed_drive_evidence(&mut agent, "grep:needle::src:0");
    assert!(
        agent.goal_drive_turn_made_progress(before.as_ref()),
        "genuine user input must start a fresh evidence scope"
    );
}

#[tokio::test]
async fn completed_turn_stamps_leftover_when_plan_pending() {
    let workspace = IsolatedWorkspace::new("completed-leftover");
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "p1".into(),
                    name: "update_plan".into(),
                    arguments: r#"{"steps":[{"title":"a","status":"done"},{"title":"b","status":"pending"}]}"#.into(),
                }],
                1,
                1,
            ),
            write_completion(&workspace.path("src/a.rs").to_string_lossy()),
            completion(vec![Content::Text("Step a is done.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("do it", &mut ui).await.unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(
        outcome
            .leftover
            .as_deref()
            .is_some_and(|line| line.contains("remaining")),
        "Completed leftover should name remaining work: {outcome:?}"
    );
    assert!(agent.drive_decision(Some(&outcome)).should_enqueue());
}

#[tokio::test]
async fn new_task_without_update_plan_keeps_stale_pin() {
    let workspace = IsolatedWorkspace::new("stale-pin");
    let mut cfg = workspace.config();
    cfg.loop_limits.max_silent_continues = 0;
    cfg.loop_limits.max_keep_working = 0;
    let plan_call = completion(
        vec![Content::ToolCall {
            id: "p1".into(),
            name: "update_plan".into(),
            arguments: r#"{"steps":[{"title":"old work","status":"pending"}]}"#.into(),
        }],
        1,
        1,
    );
    let mut agent = agent(
        vec![
            plan_call,
            completion(vec![Content::Text("planned.".into())], 1, 1),
            completion(
                vec![Content::Text("starting the new auth work.".into())],
                1,
                1,
            ),
            completion(
                vec![Content::Text("still on the old checklist.".into())],
                1,
                1,
            ),
            completion(vec![Content::Text("steering noted.".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(agent.plan_incomplete());

    agent
        .run_turn(
            "implement a new auth system from scratch that replaces the old one",
            &mut ui,
        )
        .await
        .unwrap();
    assert!(
        agent.plan_incomplete(),
        "new-task without update_plan should keep the pin: {:?}",
        agent.current_plan()
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.text().contains("/plan replace")),
        "new-task fold should name /plan replace"
    );

    agent.restore_plan(vec![pending_step("old work")]);
    agent.run_turn("continue", &mut ui).await.unwrap();
    assert!(
        agent.plan_incomplete(),
        "continue should keep the pin: {:?}",
        agent.current_plan()
    );

    agent.run_turn("use a BTreeMap", &mut ui).await.unwrap();
    assert!(
        agent.plan_incomplete(),
        "short steering should keep the pin: {:?}",
        agent.current_plan()
    );
}

#[tokio::test]
async fn ask_user_fails_closed_when_not_offered() {
    let mut cfg = config();
    cfg.memory.offer_ask_user = false;
    let mut agent = agent(Vec::new(), cfg);
    let mut ui = RecUi::default();
    let out = agent
        .handle_ask_user(
            r#"{"question":"REST or gRPC for the public API?"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(out.status, hi_tools::ToolStatus::Failed);
    assert!(
        out.content.contains("unavailable"),
        "eval/report must fail-close ask_user: {}",
        out.content
    );
    assert!(
        ui.ask_user_questions.is_empty(),
        "no overlay in eval: {:?}",
        ui.ask_user_questions
    );
}

#[tokio::test]
async fn ask_user_rejects_continue_and_caps_at_one() {
    let mut cfg = config();
    cfg.memory.offer_ask_user = true;
    let mut agent = agent(Vec::new(), cfg);
    let mut ui = RecUi::default();
    let confirm = agent
        .handle_ask_user(r#"{"question":"should I continue?"}"#, &mut ui)
        .await;
    assert_eq!(confirm.status, hi_tools::ToolStatus::Failed);
    assert!(confirm.content.contains("keep working"));
    assert!(
        ui.ask_user_questions.is_empty(),
        "continue-shaped questions must not open the overlay: {:?}",
        ui.ask_user_questions
    );

    let first = agent
        .handle_ask_user(
            r#"{"question":"REST or gRPC for the public API?"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(first.status, hi_tools::ToolStatus::Succeeded);
    assert_eq!(ui.ask_user_questions.len(), 1);

    let second = agent
        .handle_ask_user(r#"{"question":"And which error type?"}"#, &mut ui)
        .await;
    assert_eq!(second.status, hi_tools::ToolStatus::Failed);
    assert!(second.content.contains("already asked this turn"));
    assert_eq!(ui.ask_user_questions.len(), 1);
}

#[tokio::test]
async fn ask_user_drive_streak_and_what_next_fail_closed() {
    let mut cfg = config();
    cfg.memory.offer_ask_user = true;
    let mut agent = agent(Vec::new(), cfg);
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    agent.begin_drive_turn(crate::DriveKind::Plan).unwrap();
    let mut ui = RecUi::default();

    let next = agent
        .handle_ask_user(r#"{"question":"what should I do next?"}"#, &mut ui)
        .await;
    assert_eq!(next.status, hi_tools::ToolStatus::Failed);
    assert!(next.content.contains("keep working"));
    assert!(ui.ask_user_questions.is_empty());

    let first = agent
        .handle_ask_user(
            r#"{"question":"REST or gRPC for the public API?"}"#,
            &mut ui,
        )
        .await;
    assert_eq!(first.status, hi_tools::ToolStatus::Succeeded);
    assert_eq!(ui.ask_user_questions.len(), 1);

    agent.ask_user_calls = 0;
    let second = agent
        .handle_ask_user(r#"{"question":"And which error type?"}"#, &mut ui)
        .await;
    assert_eq!(second.status, hi_tools::ToolStatus::Failed);
    assert!(second.content.contains("already asked this drive"));
    assert_eq!(ui.ask_user_questions.len(), 1);
}

#[tokio::test]
async fn ask_user_timeout_resumes_with_best_option_instruction() {
    let mut cfg = config();
    cfg.memory.offer_ask_user = true;
    let mut agent = agent(Vec::new(), cfg);
    agent.side_call_timeout = Some(std::time::Duration::from_millis(10));
    let mut ui = RecUi {
        pending_ask_user: true,
        ..RecUi::default()
    };

    let outcome = agent
        .handle_ask_user(
            r#"{"question":"REST or gRPC for the public API?","options":["REST","gRPC"]}"#,
            &mut ui,
        )
        .await;

    assert_eq!(outcome.status, hi_tools::ToolStatus::Succeeded);
    assert!(outcome.content.contains("timed out"));
    assert!(
        outcome
            .content
            .contains("pick the best option and continue")
    );
    assert_eq!(ui.ask_user_questions.len(), 1);
}

#[test]
fn plan_clear_drops_the_pin() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("old work")]);
    let clear = handle_session_command(&mut agent, &Command::Plan("clear".into()), &[])
        .expect("clear effect");
    assert!(agent.current_plan().is_empty());
    assert!(clear.follow_up_prompt.is_none());
    assert!(clear.message.contains("cleared"));

    agent.restore_plan(vec![pending_step("old work")]);
    let replace = handle_session_command(&mut agent, &Command::Plan("replace".into()), &[])
        .expect("replace effect");
    assert!(agent.current_plan().is_empty());
    assert!(replace.message.contains("update_plan"));
}

#[test]
fn plan_pause_and_resume_are_reserved() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    let pause = handle_session_command(&mut agent, &Command::Plan("pause".into()), &[])
        .expect("pause effect");
    assert!(agent.plan_drive_paused());
    assert!(pause.follow_up_prompt.is_none());
    assert!(pause.message.contains("paused"));

    let outcome = completed_outcome(agent.leftover_work());
    assert!(!agent.drive_decision(Some(&outcome)).should_enqueue());

    let status = handle_session_command(&mut agent, &Command::Plan("status".into()), &[])
        .expect("status effect");
    assert!(status.message.contains("plan drive: paused"));

    let resume = handle_session_command(&mut agent, &Command::Plan("resume".into()), &[])
        .expect("resume effect");
    assert!(!agent.plan_drive_paused());
    assert_eq!(
        resume.follow_up_prompt.as_deref(),
        Some(crate::PLAN_DRIVE_PROMPT)
    );
}

#[test]
fn real_user_turn_consumes_interruption_pause_but_not_manual_pause() {
    let mut interrupted = agent(Vec::new(), config());
    interrupted.restore_plan(vec![pending_step("wire the scheduler")]);
    interrupted.restore_plan_drive_with_policy(true, true, 0, Vec::new());
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    interrupted.set_session(Box::new(PlanDrivePolicyRecordingSession {
        records: records.clone(),
    }));

    interrupted
        .begin_drive_turn(crate::DriveKind::User)
        .unwrap();

    assert!(!interrupted.plan_drive_paused());
    assert!(
        records.lock().unwrap().is_empty(),
        "the durable interruption latch must remain until user work succeeds"
    );
    interrupted.settle_plan_interruption_resume(true).unwrap();
    assert_eq!(
        records.lock().unwrap().as_slice(),
        &[(false, 0, false, false, Vec::new())],
        "resuming a zero-stall interruption must still persist paused=false"
    );

    let mut manual = agent(Vec::new(), config());
    manual.restore_plan(vec![pending_step("keep this deliberately paused")]);
    // The legacy public restore API retains manual-pause semantics.
    manual.restore_plan_drive(true, 0, Vec::new());
    manual.begin_drive_turn(crate::DriveKind::User).unwrap();
    assert!(
        manual.plan_drive_paused(),
        "ordinary conversation must not override explicit /plan pause intent"
    );

    let mut cancelled = agent(Vec::new(), config());
    cancelled.restore_plan(vec![pending_step("retry interrupted steering")]);
    cancelled.restore_plan_drive_with_policy(true, true, 0, Vec::new());
    assert!(
        cancelled
            .prepare_plan_drive_for_turn(crate::DriveKind::User)
            .unwrap()
    );
    cancelled.begin_drive_turn(crate::DriveKind::User).unwrap();
    assert!(!cancelled.plan_drive_paused());
    cancelled.settle_plan_interruption_resume(false).unwrap();
    assert!(
        cancelled.plan_drive_paused(),
        "unsuccessful user steering must reveal the durable interruption again"
    );
}

#[tokio::test]
async fn failed_user_turn_keeps_the_interruption_pause_and_blocks_auto_drive() {
    let workspace = IsolatedWorkspace::new("failed-user-interruption-resume");
    let changed = workspace.path("unverified.rs");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![crate::VerifyStage::new(
        "intentional failure",
        "false",
    )]);
    cfg.gates.max_verify_repairs = 0;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut subject = agent(
        vec![
            write_completion(&changed.to_string_lossy()),
            completion(vec![Content::Text("verification failed".into())], 1, 1),
            completion(vec![Content::Text("still failing".into())], 1, 1),
        ],
        cfg,
    );
    subject.restore_plan(vec![pending_step("finish the verified implementation")]);
    subject.restore_plan_drive_with_policy(true, true, 0, Vec::new());
    assert!(
        subject
            .prepare_plan_drive_for_turn(crate::DriveKind::User)
            .unwrap()
    );

    let outcome = subject
        .run_turn("resume by changing the implementation", &mut NullUi)
        .await
        .expect("the failed verification should settle as a typed failure");

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationFailed);
    assert!(
        changed.exists(),
        "productive work remains available to the user"
    );
    assert!(
        subject.plan_drive_paused(),
        "a failed steering turn must retain the interruption latch"
    );
    assert!(
        subject.plan_drive_resumes_on_user_input(),
        "the retained latch must still be resumable by the next real user turn"
    );
    assert_eq!(
        subject.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Blocked,
        }
    );
}

#[test]
fn synthetic_plan_turn_records_zero_stall_resume() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    agent.restore_plan_drive_with_policy(true, false, 0, Vec::new());
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    agent.set_session(Box::new(PlanDrivePolicyRecordingSession {
        records: records.clone(),
    }));

    agent.begin_drive_turn(crate::DriveKind::Plan).unwrap();

    assert!(!agent.plan_drive_paused());
    assert_eq!(
        records.lock().unwrap().as_slice(),
        &[(false, 0, false, false, Vec::new())],
        "empty-Enter/synthetic resume must not resurrect pause after restart"
    );
}

#[test]
fn failed_plan_drive_persistence_reverts_pause_and_resume_transitions() {
    let mut pause = agent(Vec::new(), config());
    pause.restore_plan(vec![pending_step("remain running")]);
    pause.set_session(Box::new(FailingPlanDriveSession));
    assert!(pause.try_set_plan_drive_paused(true).is_err());
    assert!(!pause.plan_drive_paused());

    let mut resume = agent(Vec::new(), config());
    let evidence_hash = "d".repeat(64);
    resume.restore_plan(vec![pending_step("remain paused")]);
    resume.restore_plan_drive(
        true,
        crate::PLAN_DRIVE_STALL_LIMIT,
        vec![evidence_hash.clone()],
    );
    resume.set_session(Box::new(FailingPlanDriveSession));
    assert!(resume.resume_plan_drive().is_err());
    assert!(resume.plan_drive_paused());
    assert_eq!(resume.plan_drive_stall(), crate::PLAN_DRIVE_STALL_LIMIT);
    assert_eq!(resume.plan_drive_evidence.snapshot(), vec![evidence_hash]);

    let mut interrupted = agent(Vec::new(), config());
    interrupted.restore_plan(vec![pending_step("retry steering")]);
    interrupted.restore_plan_drive_with_policy(true, true, 0, Vec::new());
    interrupted
        .begin_drive_turn(crate::DriveKind::User)
        .unwrap();
    interrupted.set_session(Box::new(FailingPlanDriveSession));
    assert!(interrupted.settle_plan_interruption_resume(true).is_err());
    assert!(
        interrupted.plan_drive_paused(),
        "failed unpause append must expose the retained interruption"
    );
}

#[test]
fn ordinary_synthetic_plan_turn_retains_cross_turn_stall_and_evidence() {
    let mut agent = agent(Vec::new(), config());
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    let evidence_hash = "c".repeat(64);
    agent.restore_plan_drive_with_policy(
        false,
        false,
        crate::PLAN_DRIVE_STALL_LIMIT - 1,
        vec![evidence_hash.clone()],
    );

    agent.begin_drive_turn(crate::DriveKind::Plan).unwrap();

    assert_eq!(agent.plan_drive_stall(), crate::PLAN_DRIVE_STALL_LIMIT - 1);
    assert_eq!(agent.plan_drive_evidence.snapshot(), vec![evidence_hash]);
    agent.note_plan_drive_progress(false);
    assert_eq!(agent.plan_drive_stall(), crate::PLAN_DRIVE_STALL_LIMIT);
    assert!(matches!(
        agent.drive_decision(None),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::PlanParked
        }
    ));
}

fn goal_agent() -> Agent {
    let mut cfg = config();
    cfg.subagents.long_horizon = true;
    agent(Vec::new(), cfg)
}

#[test]
fn drive_decision_goal_wins_over_plan_and_plan_mode_idles() {
    let mut agent = goal_agent();
    agent.restore_plan(vec![pending_step("wire the scheduler")]);
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal)
    );
    agent.set_plan_mode(true);
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::PlanMode
        }
    );
}

#[test]
fn drive_decision_enqueues_goal_after_step_limit() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    let mut outcome = completed_outcome(agent.leftover_work());
    outcome.stop_reason = TurnStopReason::StepLimit;
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Enqueue(crate::DriveKind::Goal),
        "a per-turn step cap must not park leftover goal work"
    );
}

#[test]
fn drive_decision_stops_goal_after_session_turn_limit() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    let mut outcome = completed_outcome(agent.leftover_work());
    outcome.stop_reason = TurnStopReason::TurnLimit;
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::Cancelled
        },
        "a session turn cap must not requeue unfinished goal work"
    );
}

#[test]
fn drive_decision_goal_paused_and_parked() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    assert!(
        agent
            .try_set_goal_pause_reason(crate::GoalPauseReason::User)
            .unwrap()
    );
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::GoalPaused
        }
    );
    assert!(
        agent
            .try_set_goal_pause_reason(crate::GoalPauseReason::None)
            .unwrap()
    );
    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        agent.note_goal_drive_progress(false);
    }
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::GoalParked
        }
    );
    let goal = agent.structured_goal().expect("goal");
    assert!(!goal.is_paused());
    assert_ne!(goal.pause_reason, crate::GoalPauseReason::Stall);
}

#[test]
fn begin_drive_turn_resumes_paused_and_parked_goal() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    assert!(
        agent
            .try_set_goal_pause_reason(crate::GoalPauseReason::User)
            .unwrap()
    );
    agent.begin_drive_turn(crate::DriveKind::Goal).unwrap();
    assert!(!agent.structured_goal().unwrap().is_paused());
    assert_eq!(agent.goal_drive_stall(), 0);

    for _ in 0..crate::GOAL_DRIVE_STALL_LIMIT {
        agent.note_goal_drive_progress(false);
    }
    assert_eq!(agent.goal_drive_status(), "parked");
    agent.begin_drive_turn(crate::DriveKind::Goal).unwrap();
    assert_eq!(agent.goal_drive_stall(), 0);
    assert_eq!(agent.goal_drive_status(), "running");
}

#[test]
fn interactive_drive_turn_demotes_always_and_restores() {
    let mut agent = goal_agent();
    agent.set_interactive_session(true);
    agent.set_permission_mode(crate::PermissionMode::Always);
    agent.begin_drive_turn(crate::DriveKind::Plan).unwrap();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Auto);
    agent.finish_drive_turn();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Always);

    agent.set_interactive_session(false);
    agent.set_permission_mode(crate::PermissionMode::Always);
    agent.begin_drive_turn(crate::DriveKind::Goal).unwrap();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Always);
}

#[test]
fn interactive_goal_drive_demotes_always_when_not_unattended() {
    let mut agent = goal_agent();
    agent.set_interactive_session(true);
    agent.set_permission_mode(crate::PermissionMode::Always);
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    agent.begin_drive_turn(crate::DriveKind::Goal).unwrap();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Auto);
    agent.finish_drive_turn();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Always);
}

#[test]
fn interactive_unattended_goal_drive_keeps_auto() {
    let mut agent = goal_agent();
    agent.set_interactive_session(true);
    agent.set_permission_mode(crate::PermissionMode::Auto);
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    assert!(agent.try_set_goal_unattended(true).unwrap());
    agent.begin_drive_turn(crate::DriveKind::Goal).unwrap();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Auto);
    agent.finish_drive_turn();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Auto);
}

#[test]
fn interactive_unattended_goal_drive_demotes_always_and_restores() {
    let mut agent = goal_agent();
    agent.set_interactive_session(true);
    agent.set_permission_mode(crate::PermissionMode::Always);
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    assert!(agent.try_set_goal_unattended(true).unwrap());
    agent.begin_drive_turn(crate::DriveKind::Goal).unwrap();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Auto);
    agent.finish_drive_turn();
    assert_eq!(agent.permission_mode(), crate::PermissionMode::Always);
}

#[test]
fn unattended_goal_esc_still_user_pauses() {
    let mut agent = goal_agent();
    assert!(
        agent
            .set_structured_goal(Some(crate::Goal::new(
                "ship it",
                vec!["implement it".into()],
            )))
            .unwrap()
    );
    assert!(agent.try_set_goal_unattended(true).unwrap());
    assert!(
        agent
            .try_set_goal_pause_reason(crate::GoalPauseReason::User)
            .unwrap()
    );
    let outcome = completed_outcome(agent.leftover_work());
    assert_eq!(
        agent.drive_decision(Some(&outcome)),
        crate::DriveAction::Idle {
            reason: crate::DriveIdleReason::GoalPaused
        }
    );
    let goal = agent.structured_goal().expect("goal");
    assert!(goal.unattended);
    assert_eq!(goal.pause_reason, crate::GoalPauseReason::User);
}

#[test]
fn ingest_checkbox_plan_becomes_sub_goals_without_planner() {
    let workspace = IsolatedWorkspace::new("ingest-checkbox");
    std::fs::write(
        workspace.path("plan.md"),
        "- [x] already shipped\n- [ ] wire the CLI\n- [ ] pass tests\n",
    )
    .unwrap();
    let mut cfg = workspace.config();
    cfg.paths.workspace_root = std::fs::canonicalize(&cfg.paths.workspace_root).unwrap();
    cfg.subagents.long_horizon = true;
    let agent = agent(Vec::new(), cfg);
    let goal = agent
        .try_ingest_goal("implement plan.md")
        .expect("checklist ingest");
    assert_eq!(goal.sub_goals.len(), 3);
    assert_eq!(goal.sub_goals[0].status, crate::GoalStatus::Done);
    assert_eq!(goal.sub_goals[1].status, crate::GoalStatus::Active);
    assert_eq!(goal.sub_goals[1].description, "wire the CLI");
}

#[test]
fn ingest_prose_plan_falls_back_to_planner() {
    let workspace = IsolatedWorkspace::new("ingest-prose");
    std::fs::write(
        workspace.path("plan.md"),
        "# Design\n\nThis document describes the architecture in prose.\n",
    )
    .unwrap();
    let mut cfg = workspace.config();
    cfg.paths.workspace_root = std::fs::canonicalize(&cfg.paths.workspace_root).unwrap();
    cfg.subagents.long_horizon = true;
    let agent = agent(Vec::new(), cfg);
    assert!(agent.try_ingest_goal("implement plan.md").is_none());
}

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
