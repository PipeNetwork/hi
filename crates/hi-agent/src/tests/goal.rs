use super::common::*;
use super::*;
use std::sync::Arc;

type CompactionRecords = Arc<Mutex<Vec<Vec<Message>>>>;

struct CompactionRecordingSession {
    records: CompactionRecords,
}

impl SessionSink for CompactionRecordingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, messages: &[Message]) -> anyhow::Result<()> {
        self.records.lock().unwrap().push(messages.to_vec());
        Ok(())
    }
}

struct FailingCompactionSession;

impl SessionSink for FailingCompactionSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("disk full"))
    }
}

struct FailingGoalSession;

impl SessionSink for FailingGoalSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_goal(&mut self, _goal: &Goal) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("disk full"))
    }

    fn clear_goal(&mut self) -> anyhow::Result<()> {
        Err(anyhow::anyhow!("disk full"))
    }
}

struct GoalClearingSession {
    clears: Arc<Mutex<usize>>,
}

impl SessionSink for GoalClearingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn clear_goal(&mut self) -> anyhow::Result<()> {
        *self.clears.lock().unwrap() += 1;
        Ok(())
    }
}

#[test]
fn goal_updates_system_prompt_and_clear_history_keeps_it() {
    let mut agent = agent(vec![], config());
    agent.set_goal(Some("ship a stable TUI".into()));

    assert_eq!(agent.goal(), Some("ship a stable TUI"));
    assert!(
        agent.messages()[0]
            .text()
            .contains("[Current session goal]"),
        "goal marker included"
    );
    assert!(
        agent.messages()[0].text().contains("ship a stable TUI"),
        "goal text included"
    );

    agent.messages_mut().push(Message::user("noise"));
    agent.clear_history().unwrap();
    assert_eq!(agent.messages().len(), 1);
    assert!(
        agent.messages()[0].text().contains("ship a stable TUI"),
        "goal survives clear-history"
    );

    agent.set_goal(None);
    assert_eq!(agent.goal(), None);
    assert!(
        !agent.messages()[0]
            .text()
            .contains("[Current session goal]"),
        "goal marker removed"
    );
}

#[test]
fn clear_history_records_durable_compaction_boundary() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("old context"));
    agent.set_session(Box::new(CompactionRecordingSession {
        records: records.clone(),
    }));

    agent.clear_history().unwrap();

    assert_eq!(agent.messages().len(), 1);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].len(), 1);
    assert_eq!(records[0][0].role, Role::System);
    assert_eq!(records[0][0].text(), agent.messages()[0].text());
}

#[test]
fn clear_history_keeps_visible_history_when_persistence_fails() {
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("old context"));
    agent.set_session(Box::new(FailingCompactionSession));

    let err = agent.clear_history().unwrap_err();

    assert!(err.to_string().contains("disk full"));
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(agent.messages()[1].text(), "old context");
}

#[test]
fn structured_goal_set_keeps_visible_state_when_persistence_fails() {
    let mut cfg = config();
    cfg.long_horizon = true;
    let mut agent = agent(vec![], cfg);
    agent.set_session(Box::new(FailingGoalSession));

    let err = agent
        .set_structured_goal(Some(Goal::new("ship it", vec!["ship it".into()])))
        .unwrap_err();

    assert!(err.to_string().contains("disk full"));
    assert!(agent.structured_goal().is_none());
    assert!(
        !agent.messages()[0].text().contains("Long-horizon goal"),
        "system prompt should not show an unpersisted goal"
    );
}

#[test]
fn structured_goal_clear_keeps_visible_state_when_persistence_fails() {
    let mut cfg = config();
    cfg.long_horizon = true;
    let mut agent = agent(vec![], cfg);
    agent
        .set_structured_goal(Some(Goal::new("ship it", vec!["ship it".into()])))
        .unwrap();
    agent.set_session(Box::new(FailingGoalSession));

    let err = agent.set_structured_goal(None).unwrap_err();

    assert!(err.to_string().contains("disk full"));
    assert!(agent.structured_goal().is_some());
    assert!(
        agent.messages()[0].text().contains("ship it"),
        "system prompt should keep the still-active goal"
    );
}

#[test]
fn structured_goal_clear_records_marker_even_when_long_horizon_is_off() {
    let persisted_goal = Goal::new("old durable goal", vec!["old durable goal".into()]);
    let history = vec![Message::system("old prompt")];
    let cfg = config();
    let mut agent = resumed_agent(history, Usage::default(), Some(persisted_goal), cfg);
    assert!(
        agent.structured_goal().is_none(),
        "long-horizon-off resume intentionally hides structured goal"
    );
    let clears = Arc::new(Mutex::new(0));
    agent.set_session(Box::new(GoalClearingSession {
        clears: clears.clone(),
    }));

    agent.set_structured_goal(None).unwrap();

    assert_eq!(
        *clears.lock().unwrap(),
        1,
        "clear should write a goal_cleared marker even when no goal is visible"
    );
}

#[test]
fn transient_goal_set_clears_hidden_persisted_structured_goal() {
    let persisted_goal = Goal::new("old durable goal", vec!["old durable goal".into()]);
    let history = vec![Message::system("old prompt")];
    let cfg = config();
    let mut agent = resumed_agent(history, Usage::default(), Some(persisted_goal), cfg);
    let clears = Arc::new(Mutex::new(0));
    agent.set_session(Box::new(GoalClearingSession {
        clears: clears.clone(),
    }));

    agent
        .set_transient_goal(Some("new transient goal".into()))
        .unwrap();

    assert_eq!(
        *clears.lock().unwrap(),
        1,
        "setting a plain goal should tombstone any hidden durable structured goal"
    );
    assert_eq!(agent.goal(), Some("new transient goal"));
    assert!(agent.structured_goal().is_none());
    assert!(agent.messages()[0].text().contains("new transient goal"));
    assert!(!agent.messages()[0].text().contains("old durable goal"));
}

#[test]
fn transient_goal_set_keeps_visible_state_when_hidden_goal_clear_fails() {
    let persisted_goal = Goal::new("old durable goal", vec!["old durable goal".into()]);
    let history = vec![Message::system("old prompt")];
    let cfg = config();
    let mut agent = resumed_agent(history, Usage::default(), Some(persisted_goal), cfg);
    agent.set_session(Box::new(FailingGoalSession));

    let err = agent
        .set_transient_goal(Some("new transient goal".into()))
        .unwrap_err();

    assert!(err.to_string().contains("disk full"));
    assert!(agent.goal().is_none());
    assert!(!agent.messages()[0].text().contains("new transient goal"));
}

#[tokio::test]
async fn structured_goal_state_injected_into_system_prompt_when_long_horizon_on() {
    // With long_horizon on, a structured goal's state (objective + sub-goal
    // checklist + retry notes) is injected into the system prompt so the
    // agent resumes the active sub-goal coherently each turn.
    let mut cfg = config();
    cfg.long_horizon = true;
    let mut agent = agent(
        vec![completion(vec![Content::Text("ok".into())], 1, 1)],
        cfg,
    );
    let mut goal = Goal::new(
        "refactor the parser",
        vec!["write tests".into(), "rewrite parser".into()],
    );
    // Record a failed attempt so the prompt surfaces "don't repeat" notes.
    goal.record_failure("approach A didn't compile", DEFAULT_SUBGOAL_RETRIES);
    assert!(
        agent.set_structured_goal(Some(goal)).unwrap(),
        "accepted when long_horizon on"
    );

    let sys = agent.messages()[0].text();
    assert!(sys.contains("Long-horizon goal"), "header: {sys}");
    assert!(sys.contains("refactor the parser"), "objective: {sys}");
    assert!(sys.contains("write tests"), "sub-goal: {sys}");
    assert!(
        sys.contains("don't repeat these"),
        "retry notes surfaced: {sys}"
    );

    // Clearing the goal removes the section.
    agent.set_structured_goal(None).unwrap();
    let sys_after = agent.messages()[0].text();
    assert!(
        !sys_after.contains("Long-horizon goal"),
        "goal section cleared: {sys_after}"
    );
}

#[test]
fn resume_restores_structured_goal_and_rebuilds_system_prompt() {
    let mut cfg = config();
    cfg.long_horizon = true;
    let mut goal = Goal::new(
        "ship resumed parser",
        vec!["write tests".into(), "merge parser".into()],
    );
    goal.advance();
    let history = vec![
        Message::system("old prompt\n\n[Long-horizon goal]\nstale objective\nstale step"),
        Message::user("previous request"),
    ];

    let agent = resumed_agent(history, Usage::default(), Some(goal), cfg);

    let sys = agent.messages()[0].text();
    assert!(
        agent.structured_goal().is_some(),
        "structured goal restored"
    );
    assert!(
        sys.contains("Long-horizon goal"),
        "goal section restored: {sys}"
    );
    assert!(
        sys.contains("ship resumed parser"),
        "objective restored: {sys}"
    );
    assert!(
        sys.contains("merge parser"),
        "active sub-goal restored: {sys}"
    );
    assert!(
        !sys.contains("stale objective") && !sys.contains("stale step"),
        "resume should rebuild the system prompt from loaded metadata, not keep stale saved goal text: {sys}"
    );
}

#[tokio::test]
async fn structured_goal_rejected_when_long_horizon_off() {
    // Default config has long_horizon off — setting a structured goal is
    // rejected (the single-turn loop is unchanged), so the system prompt
    // gains no goal section.
    let mut agent = agent(
        vec![completion(vec![Content::Text("ok".into())], 1, 1)],
        config(),
    );
    let goal = Goal::new("do a thing", vec!["step one".into()]);
    assert!(
        !agent.set_structured_goal(Some(goal)).unwrap(),
        "rejected when off"
    );
    assert!(agent.structured_goal().is_none());
    let sys = agent.messages()[0].text();
    assert!(
        !sys.contains("Long-horizon goal"),
        "no goal section when off: {sys}"
    );
}

#[tokio::test]
async fn long_horizon_driver_advances_on_clean_turn() {
    // With long_horizon on and a structured goal set, a turn that verifies
    // clean advances the active
    // sub-goal, and the system prompt reflects the new active sub-goal.
    let workspace = IsolatedWorkspace::new("goal-clean");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    // One turn: model writes a file (tool), then a clean text finish.
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    agent
        .set_structured_goal(Some(Goal::new(
            "refactor",
            vec!["step one".into(), "step two".into()],
        )))
        .unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Done,
        "advanced past step 1"
    );
    assert_eq!(goal.active_index(), Some(1), "step 2 now active");
    // The system prompt reflects the new active sub-goal.
    assert!(
        agent.messages()[0].text().contains("step two"),
        "system prompt shows new active sub-goal"
    );
}

#[tokio::test]
async fn skeptic_gate_objection_blocks_advance_and_records_note() {
    // With `/goal team` on and a skeptic model configured, a turn that would
    // otherwise advance is reviewed first; an OBJECT sends the sub-goal back to
    // retry (objections become notes) instead of advancing.
    let workspace = IsolatedWorkspace::new("goal-skeptic-object");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.skeptic_model = Some("skeptic".into());
    cfg.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("done".into())], 1, 1),
        // The skeptic call, fired at goal-turn end, objects.
        completion(
            vec![Content::Text(
                "OBJECT\n- the empty-input edge case isn't handled".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.active_index(),
        Some(0),
        "objection blocked the advance"
    );
    assert_eq!(goal.sub_goals[0].status, GoalStatus::Active);
    assert_eq!(goal.skeptic_objections, 1, "objection counted");
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|n| n.contains("empty-input edge case")),
        "objection recorded as a retry note: {:?}",
        goal.sub_goals[0].notes
    );
    // The note surfaces in the system prompt so the next turn addresses it.
    assert!(
        agent.messages()[0].text().contains("empty-input edge case"),
        "objection in the system prompt"
    );
}

#[tokio::test]
async fn skeptic_gate_approval_advances_and_actually_calls_the_skeptic() {
    let workspace = IsolatedWorkspace::new("goal-skeptic-approve");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.skeptic_model = Some("skeptic".into());
    cfg.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let steps = vec![
        ProviderStep::Completion(write_completion(&p)),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Done,
        "approved → advanced"
    );
    assert_eq!(goal.active_index(), Some(1));
    assert_eq!(goal.skeptic_objections, 0);
    // Exactly one extra call beyond the turn, and it was the skeptic (its system
    // prompt is distinctive) — proving *which* call reviewed, not positional trust.
    let reqs = requests.lock().unwrap();
    assert_eq!(reqs.len(), 3, "turn (2 calls) + skeptic (1)");
    assert!(
        reqs.last()
            .unwrap()
            .iter()
            .any(|m| m.text().contains("code reviewer acting as a merge gate")),
        "the extra call carried the skeptic review prompt"
    );
}

#[tokio::test]
async fn skeptic_gate_off_makes_no_extra_call() {
    // A skeptic model is configured, but `/goal team` is off (default): the gate
    // must not fire — no extra provider call, and advancing is byte-identical to
    // single-agent driving.
    let workspace = IsolatedWorkspace::new("goal-skeptic-off");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.skeptic_model = Some("skeptic".into());
    cfg.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    // Only the turn's two calls are scripted; a spurious skeptic call would pop an
    // absent step and panic — so this fails loudly on a regression too.
    let steps = vec![
        ProviderStep::Completion(write_completion(&p)),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    // team defaults to false.
    agent
        .set_structured_goal(Some(Goal::new(
            "refactor",
            vec!["step one".into(), "step two".into()],
        )))
        .unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(
        agent.structured_goal().unwrap().active_index(),
        Some(1),
        "advanced normally with the gate off"
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "no extra skeptic call when team is off"
    );
}

#[tokio::test]
async fn skeptic_gate_fails_open_on_provider_error() {
    // A skeptic that errors must not wedge the goal — the gate is fail-open, so the
    // sub-goal advances as if there were no gate.
    let workspace = IsolatedWorkspace::new("goal-skeptic-error");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.skeptic_model = Some("skeptic".into());
    cfg.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let steps = vec![
        ProviderStep::Completion(write_completion(&p)),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Error(ProviderErrorKind::Outage),
    ];
    let (mut agent, _requests) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Done,
        "fail-open advanced despite the skeptic error"
    );
    assert_eq!(goal.skeptic_objections, 0);
    assert_eq!(goal.skeptic_unavailable, 1);
    assert_eq!(goal.last_skeptic_status, Some(SkepticStatus::Unavailable));
    assert_eq!(agent.last_turn_telemetry().skeptic_unavailable_count, 1);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("skeptic unavailable"))
    );
}

#[tokio::test]
async fn skeptic_gate_reviews_update_plan_completion_and_reverts_on_objection() {
    // The common case a live run exposed: the model marks a sub-goal done via
    // update_plan (not the heuristic advance). The skeptic must STILL review it,
    // and on an objection revert the update_plan advance (re-open the sub-goal) —
    // otherwise the gate is bypassed exactly when a capable model claims "done".
    let workspace = IsolatedWorkspace::new("goal-skeptic-plan");
    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.skeptic_model = Some("skeptic".into());
    cfg.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let update_plan = completion(
        vec![Content::ToolCall {
            id: "up".into(),
            name: "update_plan".into(),
            arguments: serde_json::json!({
                "steps": [
                    {"title": "step one", "status": "done"},
                    {"title": "step two", "status": "active"},
                ]
            })
            .to_string(),
        }],
        1,
        1,
    );
    let steps = vec![
        ProviderStep::Completion(write_completion(&p)),
        ProviderStep::Completion(update_plan),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(completion(
            vec![Content::Text(
                "OBJECT\n- step one wasn't actually finished".into(),
            )],
            1,
            1,
        )),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    // The update_plan advance was REVERTED — step one active again, not step two.
    assert_eq!(
        goal.active_index(),
        Some(0),
        "objection reverted the update_plan advance"
    );
    assert_eq!(goal.sub_goals[0].status, GoalStatus::Active);
    assert_eq!(goal.sub_goals[1].status, GoalStatus::Pending);
    assert_eq!(goal.skeptic_objections, 1);
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|n| n.contains("wasn't actually finished")),
        "objection recorded as a note: {:?}",
        goal.sub_goals[0].notes
    );
    // The skeptic really ran (4th call) and reviewed the pre-turn active sub-goal.
    let reqs = requests.lock().unwrap();
    assert_eq!(reqs.len(), 4, "write + update_plan + finish + skeptic");
    assert!(
        reqs.last()
            .unwrap()
            .iter()
            .any(|m| m.text().contains("step one")),
        "skeptic reviewed the sub-goal active at turn start"
    );
}

#[tokio::test]
async fn long_horizon_driver_records_failure_on_stall() {
    // A turn that stalls (repeat guard exhausts) records a sub-goal attempt
    // so the next turn sees the prior note (and doesn't repeat the approach).
    let mut cfg = config();
    cfg.long_horizon = true;
    cfg.max_repeat_nudges = 1;
    // Model re-issues the same tool call → repeat guard stalls the turn
    // after exhausting the (1) nudge budget. Three identical writes: the
    // second triggers a nudge, the third exhausts the budget and breaks
    // stalled.
    let responses = vec![
        write_completion("lhstall"),
        write_completion("lhstall"),
        write_completion("lhstall"),
    ];
    let mut agent = agent(responses, cfg);
    agent
        .set_structured_goal(Some(Goal::new(
            "refactor",
            vec!["step one".into(), "step two".into()],
        )))
        .unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    let _ = std::fs::remove_file("lhstall");
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(goal.active_index(), Some(0), "didn't advance (stalled)");
    assert!(
        goal.sub_goals[0].attempts > 0,
        "recorded a failure attempt: {:?}",
        goal.sub_goals[0]
    );
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|n| n.contains("stalled")),
        "stall reason recorded as a note: {:?}",
        goal.sub_goals[0].notes
    );
    // The system prompt surfaces the "don't repeat" notes on the active
    // sub-goal, so the next turn doesn't repeat the failed approach.
    assert!(
        agent.messages()[0].text().contains("don't repeat these"),
        "retry notes in system prompt"
    );
}

#[tokio::test]
async fn long_horizon_driver_records_failure_on_unfinished_turn() {
    // A turn can be unfinished without being an exact repeat stall, for example
    // when an implementation task only scaffolds setup and never edits source.
    // That should count as a failed attempt on the active sub-goal, not advance
    // as a clean changed-files turn.
    let workspace = IsolatedWorkspace::new("goal-unfinished-scaffold");
    let dir = workspace.path("scaffold");
    let dir_string = dir.to_string_lossy().to_string();

    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    let responses = vec![
        bash_completion(&format!("mkdir -p {dir_string}")),
        completion(vec![Content::Text("Implemented it.".into())], 1, 1),
        completion(vec![Content::Text("Done.".into())], 1, 1),
        completion(vec![Content::Text("Final recap.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    agent
        .set_structured_goal(Some(Goal::new(
            "build estimator",
            vec!["implement estimator".into(), "validate estimator".into()],
        )))
        .unwrap();
    let mut ui = RecUi::default();

    agent
        .run_turn("/build a small CLI project tracker", &mut ui)
        .await
        .unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.active_index(),
        Some(0),
        "unfinished turn did not advance"
    );
    assert!(
        goal.sub_goals[0].attempts > 0,
        "unfinished turn should record an attempt: {:?}",
        goal.sub_goals[0]
    );
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|note| note.contains("without completing")),
        "unfinished reason recorded: {:?}",
        goal.sub_goals[0].notes
    );
    assert!(
        agent.messages()[0].text().contains("don't repeat these"),
        "retry notes in system prompt"
    );
}

#[tokio::test]
async fn long_horizon_driver_records_verify_failure_reason_after_exhaustion() {
    let workspace = IsolatedWorkspace::new("goal-verify-failure");
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();

    let mut cfg = workspace.config();
    cfg.long_horizon = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.max_verify_repairs = 0;
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("attempt 1".into())], 1, 1),
        completion(vec![Content::Text("attempt 2".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    agent
        .set_structured_goal(Some(Goal::new(
            "ship parser",
            vec!["make parser pass tests".into(), "cleanup".into()],
        )))
        .unwrap();

    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(agent.last_verify(), Some(false));
    assert!(agent.last_turn_telemetry().stalled_unfinished);
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.active_index(),
        Some(0),
        "verify failure did not advance"
    );
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|note| note.contains("verification failed")),
        "verify failure reason recorded: {:?}",
        goal.sub_goals[0].notes
    );
}
