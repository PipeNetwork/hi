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
    cfg.subagents.long_horizon = true;
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
    cfg.subagents.long_horizon = true;
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
    cfg.subagents.long_horizon = true;
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
    cfg.subagents.long_horizon = true;
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
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    // One turn: model writes a file (tool), then a clean text finish.
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut driver_goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    driver_goal.team = false; // isolate the driver from the (default-on) skeptic
    agent.set_structured_goal(Some(driver_goal)).unwrap();
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
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
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
async fn skeptic_gate_works_unconfigured_by_reviewing_with_the_session_model() {
    // `/goal team on` must work with zero configuration: no `skeptic_model`
    // set, the gate reviews with the session model instead of reporting
    // "no skeptic model configured".
    let workspace = IsolatedWorkspace::new("goal-skeptic-default");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = None;
    cfg.gates.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let steps = vec![
        ProviderStep::Completion(write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        )),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    assert_eq!(
        agent.effective_skeptic_model(),
        "m",
        "session model reviews"
    );
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
    assert_eq!(goal.skeptic_unavailable, 0, "the gate actually reviewed");
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
async fn skeptic_gate_approval_advances_and_actually_calls_the_skeptic() {
    let workspace = IsolatedWorkspace::new("goal-skeptic-approve");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let steps = vec![
        ProviderStep::Completion(write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        )),
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
    // A skeptic model is configured, but `/goal team` is off (set explicitly —
    // new goals default to team on): the gate must not fire — no extra provider
    // call, and advancing is byte-identical to single-agent driving. This is
    // the `/goal team off` contract.
    let workspace = IsolatedWorkspace::new("goal-skeptic-off");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    // Only the turn's two calls are scripted; a spurious skeptic call would pop an
    // absent step and panic — so this fails loudly on a regression too.
    let steps = vec![
        ProviderStep::Completion(write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        )),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut team_off_goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    team_off_goal.team = false; // the explicit `/goal team off` state
    agent.set_structured_goal(Some(team_off_goal)).unwrap();
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
async fn skeptic_gate_skips_review_on_trivial_diff() {
    // `/goal team` on, verify passing, but the turn's net change is tiny: the
    // gate skips the second-model round-trip and advances directly — the
    // defect classes the skeptic catches can't hide in a one-byte write.
    let workspace = IsolatedWorkspace::new("goal-skeptic-trivial");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    // Only the two drive calls are scripted; a spurious skeptic call would pop
    // an absent step and panic.
    let steps = vec![
        ProviderStep::Completion(write_completion(&p)), // 1-byte write = trivial
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    goal.team = true; // gate explicitly on
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(
        agent.structured_goal().unwrap().active_index(),
        Some(1),
        "trivial verified change advances without review"
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "no skeptic call for a trivial diff"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("trivial diff")),
        "the skip is visible: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn skeptic_gate_fails_open_on_provider_error() {
    // A skeptic that errors must not wedge the goal — the gate is fail-open, so the
    // sub-goal advances as if there were no gate.
    let workspace = IsolatedWorkspace::new("goal-skeptic-error");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let steps = vec![
        ProviderStep::Completion(write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        )),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        // The reviewer retries a transient error once before giving up, so a
        // persistent outage takes both scripted attempts.
        ProviderStep::Error(ProviderErrorKind::Outage),
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
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
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
        ProviderStep::Completion(write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        )),
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
    cfg.subagents.long_horizon = true;
    cfg.loop_limits.max_repeat_nudges = 1;
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
    cfg.subagents.long_horizon = true;
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
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.gates.max_verify_repairs = 0;
    let responses = vec![
        write_content_completion(
            &p,
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
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

fn quant_plan_doc() -> &'static str {
    "Quantization-aware training for the GLM transformer: binary and ternary \
     fake-quantization with group-128 scales, teacher distillation losses, CUDA \
     GEMV decode kernels, artifact packing manifests, expert coverage tracking, \
     progressive quantization schedules, inference runtime backends. Quantization \
     kernels, distillation, quantization schedules, teacher logits, expert routing, \
     GEMV kernels, artifact manifests, runtime backends, transformer layers."
}

const GENERIC_WEB_DECOMPOSITION: &str = "Implement all missing frontend UI components and pages\n\
Set up authentication and API endpoints\n\
Add client-side state management\n";

const GROUNDED_DECOMPOSITION: &str = "Implement binary fake-quantization with group-128 scales\n\
Implement CUDA GEMV decode kernels\n\
Add teacher distillation losses\n";

#[tokio::test]
async fn planner_retry_on_mismatched_decomposition_then_success() {
    let workspace = IsolatedWorkspace::new("planner-grounding-retry");
    std::fs::write(workspace.path("plan.md"), quant_plan_doc()).unwrap();
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.subagents.planner_model = Some("planner".into());
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::Text(GENERIC_WEB_DECOMPOSITION.into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text(GROUNDED_DECOMPOSITION.into())],
                1,
                1,
            )),
        ],
        cfg,
    );

    let steps = agent
        .decompose_goal("review plan.md and fully build this")
        .await
        .expect("retry recovers a grounded decomposition");

    assert_eq!(steps.len(), 3);
    assert!(steps[0].contains("fake-quantization"), "steps: {steps:?}");
    let recorded = requests.lock().unwrap();
    assert_eq!(recorded.len(), 2, "initial call plus one retry");
    let retry_text = recorded[1]
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        retry_text.contains("did not correspond"),
        "retry names the mismatch: {retry_text}"
    );
    assert!(
        retry_text.contains("frontend UI components"),
        "retry cites an unmatched milestone"
    );
}

#[tokio::test]
async fn planner_mismatch_after_retry_returns_err() {
    let workspace = IsolatedWorkspace::new("planner-grounding-err");
    std::fs::write(workspace.path("plan.md"), quant_plan_doc()).unwrap();
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.subagents.planner_model = Some("planner".into());
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::Text(GENERIC_WEB_DECOMPOSITION.into())],
                1,
                1,
            )),
            ProviderStep::Completion(completion(
                vec![Content::Text(GENERIC_WEB_DECOMPOSITION.into())],
                1,
                1,
            )),
        ],
        cfg,
    );

    let err = agent
        .decompose_goal("review plan.md and fully build this")
        .await
        .expect_err("two ungrounded decompositions must not drive a goal");

    assert!(
        err.to_string()
            .contains("did not match the referenced documents"),
        "error explains the fallback: {err}"
    );
    assert_eq!(requests.lock().unwrap().len(), 2, "exactly one retry");
}

#[tokio::test]
async fn skeptic_context_includes_stub_findings() {
    let workspace = IsolatedWorkspace::new("goal-skeptic-stubs");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let stub_write = completion(
        vec![Content::ToolCall {
            id: "w".into(),
            name: "write".into(),
            arguments: serde_json::json!({
                "path": workspace.path("stubbed.rs").to_string_lossy(),
                // Big enough to clear the skeptic gate's trivial-diff exemption —
                // this test exists to check the stub findings reach the review.
                "content": "pub fn quantize() { todo!(\"later\") }\n// padding to make this write non-trivial in size\n",
            })
            .to_string(),
        }],
        1,
        1,
    );
    let steps = vec![
        ProviderStep::Completion(stub_write),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("implement it", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();

    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let recorded = requests.lock().unwrap();
    let skeptic_request = recorded
        .last()
        .unwrap()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        skeptic_request.contains("Stub markers present in files changed this turn:"),
        "skeptic sees the stub section: {skeptic_request}"
    );
    assert!(
        skeptic_request.contains("stubbed.rs") && skeptic_request.contains("todo!("),
        "the finding names the file and marker: {skeptic_request}"
    );
}

fn audit_cfg(workspace: &IsolatedWorkspace) -> AgentConfig {
let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg
}

fn single_step_goal() -> Goal {
    // team=false isolates the audit call from the (default-on) skeptic call.
    let mut goal = Goal::new(
        "review plan.md and fully build this",
        vec!["implement everything in plan.md".into()],
    );
    goal.team = false;
    goal
}

#[tokio::test]
async fn completion_audit_appends_missing_work_and_goal_stays_active() {
    let workspace = IsolatedWorkspace::new("goal-audit-appends");
    std::fs::write(workspace.path("plan.md"), quant_plan_doc()).unwrap();
    let changed = workspace.path("changed.rs");
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                &changed.to_string_lossy(),
                "a substantial implementation body, comfortably past the trivial-diff exemption",
            )),
            ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
            // The auditor flags two missing deliverables.
            ProviderStep::Completion(completion(
                vec![Content::Text(
                    "Implement the inference runtime backends\nImplement Metal decode kernels"
                        .into(),
                )],
                1,
                1,
            )),
        ],
        audit_cfg(&workspace),
    );
    agent.set_structured_goal(Some(single_step_goal())).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(goal.status, GoalStatus::Active, "audit reopened the goal");
    assert_eq!(goal.sub_goals.len(), 3, "two missing milestones appended");
    assert_eq!(
        goal.active_index(),
        Some(1),
        "first appended step is active"
    );
    assert_eq!(goal.audit_rounds, 1);
    assert!(goal.should_auto_drive(), "the drive continues");
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("completion audit found 2 missing")),
        "statuses: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("long-horizon goal complete")),
        "no completion announcement while work remains: {:?}",
        ui.statuses
    );
    // The auditor saw the requirements doc and the sized repository listing.
    let recorded = requests.lock().unwrap();
    let audit_request = recorded
        .last()
        .unwrap()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(audit_request.contains("completion auditor"));
    assert!(
        audit_request.contains("group-128 scales"),
        "plan.md inlined"
    );
    assert!(
        audit_request.contains("Repository files (path, bytes):"),
        "listing present"
    );
    assert!(
        audit_request.contains("plan.md "),
        "listing has sized entries: {audit_request}"
    );
}

#[tokio::test]
async fn completion_audit_complete_finishes_goal() {
    let workspace = IsolatedWorkspace::new("goal-audit-complete");
    let changed = workspace.path("changed.rs");
    let responses = vec![
        write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("COMPLETE".into())], 1, 1),
    ];
    let mut agent = agent(responses, audit_cfg(&workspace));
    agent.set_structured_goal(Some(single_step_goal())).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(agent.structured_goal().unwrap().status, GoalStatus::Done);
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("completion audit passed")),
        "statuses: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("long-horizon goal complete")),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn completion_audit_cap_reached_finishes_without_another_call() {
    let workspace = IsolatedWorkspace::new("goal-audit-cap");
    let changed = workspace.path("changed.rs");
    // No auditor completion scripted: at the cap no audit call may fire (the
    // Canned provider would panic if one did).
    let responses = vec![
        write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, audit_cfg(&workspace));
    let mut goal = single_step_goal();
    goal.audit_rounds = crate::agent::audit_goal::MAX_AUDIT_ROUNDS;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(agent.structured_goal().unwrap().status, GoalStatus::Done);
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("long-horizon goal complete")),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn completion_audit_unavailable_fails_open() {
    let workspace = IsolatedWorkspace::new("goal-audit-failopen");
    let changed = workspace.path("changed.rs");
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                &changed.to_string_lossy(),
                "a substantial implementation body, comfortably past the trivial-diff exemption",
            )),
            ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
            ProviderStep::Error(ProviderErrorKind::Auth),
        ],
        audit_cfg(&workspace),
    );
    agent.set_structured_goal(Some(single_step_goal())).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(
        agent.structured_goal().unwrap().status,
        GoalStatus::Done,
        "auditor failure must not wedge the goal"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("auditor unavailable")),
        "honest about the missing audit: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn completion_audit_step_limit_saturated_finishes_with_warning() {
    let workspace = IsolatedWorkspace::new("goal-audit-steplimit");
    let changed = workspace.path("changed.rs");
    let responses = vec![
        write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text("Implement the missing deliverable".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, audit_cfg(&workspace));
    let mut goal = single_step_goal();
    goal.step_limit = Some(1); // saturated — the audit cannot grow the plan
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(goal.status, GoalStatus::Done, "cannot grow → finish");
    assert_eq!(goal.sub_goals.len(), 1);
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("step limit") && s.contains("missing deliverable")),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn exhausted_sub_goal_skips_to_next_step_instead_of_failing_goal() {
    // The qtest failure: the last sub-goal thrashed to budget exhaustion and
    // record_failure marked the WHOLE goal Failed — killing the drive with
    // 20/21 milestones done. With pending work remaining, the driver must
    // skip past the dead step and keep driving.
    let mut cfg = config();
    cfg.subagents.long_horizon = true;
    cfg.loop_limits.max_repeat_nudges = 1;
    let responses = vec![
        write_completion("lhskip"),
        write_completion("lhskip"),
        write_completion("lhskip"),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new("refactor", vec!["stuck step".into(), "next step".into()]);
    goal.team = false;
    goal.sub_goals[0].attempts = 2; // one more failure exhausts the budget
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();
    let _ = std::fs::remove_file("lhskip");

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Failed,
        "the dead step stays visible as Failed"
    );
    assert_eq!(goal.active_index(), Some(1), "skipped to the next step");
    assert_eq!(goal.status, GoalStatus::Active, "goal survives");
    assert!(goal.should_auto_drive(), "the drive keeps its momentum");
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("skipping to the next step")),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn step_capped_turn_with_progress_is_a_continuation_not_a_failure() {
    // The qtest2 failure: a milestone big enough to hit the per-turn step cap
    // three turns running burned its whole retry budget and was marked Failed
    // — while its crate sat complete and compiling on disk. A capped turn
    // that changed files must continue the milestone, not count against it.
    let workspace = IsolatedWorkspace::new("goal-cap-continuation");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.loop_limits.max_steps = 1; // the write below consumes the whole turn budget
    cfg.loop_limits.max_steps_explicit = true;
    let responses = vec![
        write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
        completion(vec![Content::Text("ran out of turn budget".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new("build it", vec!["big milestone".into(), "next".into()]);
    goal.team = false;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(goal.active_index(), Some(0), "same milestone stays active");
    assert_eq!(goal.sub_goals[0].attempts, 0, "no retry budget burned");
    assert_eq!(
        goal.sub_goals[0].cap_continuations, 1,
        "continuation counted"
    );
    assert!(goal.sub_goals[0].notes.is_empty(), "no failure note");
    assert_eq!(goal.status, GoalStatus::Active);
    assert!(goal.should_auto_drive(), "drive keeps going");
    assert!(
        ui.statuses.iter().any(|s| s.contains("continuing (1/")),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn step_capped_turn_past_continuation_budget_records_failure() {
    let workspace = IsolatedWorkspace::new("goal-cap-exhausted");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.loop_limits.max_steps = 1;
    cfg.loop_limits.max_steps_explicit = true;
    let responses = vec![
        write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
        completion(vec![Content::Text("ran out of turn budget".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new(
        "build it",
        vec!["thrashing milestone".into(), "next".into()],
    );
    goal.team = false;
    goal.sub_goals[0].cap_continuations = MAX_CAP_CONTINUATIONS;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].attempts, 1,
        "past the continuation budget, capped turns burn retries again"
    );
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|n| n.contains("step cap")),
        "notes: {:?}",
        goal.sub_goals[0].notes
    );
}

#[tokio::test]
async fn step_capped_barren_turn_continues_under_the_barren_limit() {
    // A capped turn that lands NO edits is "barren" — a big milestone may still
    // need an exploration turn, so a single barren cap continues (doesn't burn a
    // retry) but is counted toward the barren limit.
    let workspace = IsolatedWorkspace::new("goal-barren-continue");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.loop_limits.max_steps = 1; // the non-editing bash call below consumes the whole budget
    cfg.loop_limits.max_steps_explicit = true;
    let responses = vec![
        bash_completion("echo exploring"),
        completion(vec![Content::Text("ran out of turn budget".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new("build it", vec!["big milestone".into(), "next".into()]);
    goal.team = false;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(goal.active_index(), Some(0), "milestone stays active");
    assert_eq!(
        goal.sub_goals[0].attempts, 0,
        "a barren cap under the limit burns no retry"
    );
    assert_eq!(
        goal.sub_goals[0].cap_continuations, 1,
        "continuation counted"
    );
    assert_eq!(goal.sub_goals[0].barren_caps, 1, "barren cap counted");
    assert!(goal.should_auto_drive(), "drive keeps going");
}

#[tokio::test]
async fn step_capped_barren_run_fails_at_the_limit() {
    // A run of MAX_BARREN_CAPS capped turns without any edits means the model
    // can't make progress — it fails rather than continuing forever.
    let workspace = IsolatedWorkspace::new("goal-barren-fail");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.loop_limits.max_steps = 1;
    cfg.loop_limits.max_steps_explicit = true;
    let responses = vec![
        bash_completion("echo exploring"),
        completion(vec![Content::Text("ran out of turn budget".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new("build it", vec!["thrashing".into(), "next".into()]);
    goal.team = false;
    // Already at the edge — one more barren cap tips it over the limit.
    goal.sub_goals[0].barren_caps = crate::goal::MAX_BARREN_CAPS - 1;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.sub_goals[0].attempts, 1,
        "a run of barren caps burns a retry"
    );
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|n| n.contains("step cap")),
        "notes: {:?}",
        goal.sub_goals[0].notes
    );
}

#[tokio::test]
async fn oversized_milestone_decomposes_into_substeps() {
    // A milestone that keeps hitting the step cap while making progress is too
    // big for one turn: instead of grinding it out, it's split into turn-sized
    // sub-steps via a planner call.
    let workspace = IsolatedWorkspace::new("goal-decompose");
    let changed = workspace.path("changed.rs");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.subagents.planner_model = Some("planner".into());
    cfg.loop_limits.max_steps = 1; // the write consumes the whole turn budget (one model call)
    cfg.loop_limits.max_steps_explicit = true;
    let substeps = "Scaffold the crate with core types\n\
Implement the encoder with tests\n\
Implement the decoder with tests\n";
    let (mut agent, _requests) = scripted_agent(
        vec![
            // Drive turn: one write (progress) that hits the step cap.
            ProviderStep::Completion(write_content_completion(
                &changed.to_string_lossy(),
                "a substantial implementation body, comfortably past the trivial-diff exemption",
            )),
            // Then the milestone-split planner call returns the sub-steps.
            ProviderStep::Completion(completion(vec![Content::Text(substeps.into())], 1, 1)),
        ],
        cfg,
    );
    let mut goal = Goal::new(
        "build it",
        vec!["huge crate milestone".into(), "next".into()],
    );
    goal.team = false;
    // On the brink of the decomposition trigger: one more productive cap splits it.
    goal.sub_goals[0].cap_continuations = crate::goal::DECOMPOSE_AFTER_CONTINUATIONS - 1;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().expect("goal still set");
    let descs: Vec<&str> = goal
        .sub_goals
        .iter()
        .map(|s| s.description.as_str())
        .collect();
    assert_eq!(
        goal.sub_goals.len(),
        4,
        "the oversized milestone became 3 sub-steps (+ the untouched next): {descs:?}"
    );
    assert_eq!(
        goal.sub_goals[0].description,
        "Scaffold the crate with core types"
    );
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Active,
        "first sub-step active"
    );
    assert_eq!(goal.sub_goals[0].split_depth, 1, "sub-steps carry depth 1");
    assert_eq!(
        goal.sub_goals[3].description, "next",
        "rest of the plan preserved"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("split into")),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn skeptic_endpoint_routes_the_review_to_the_side_provider() {
    // With a skeptic endpoint configured, the per-turn review runs against the
    // side provider (e.g. a local model), not the session provider. The main
    // provider scripts ONLY the drive turn; the skeptic's OBJECT comes from the
    // side provider — if the review wrongly used the main provider it would run
    // out of script and panic.
    let workspace = IsolatedWorkspace::new("goal-skeptic-endpoint");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("local-skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    // Main provider: the drive turn only.
    let mut agent = agent(
        vec![
            write_content_completion(
                &p,
                "a substantial implementation body, comfortably past the trivial-diff exemption",
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    // Side provider: the skeptic's OBJECT verdict.
    let side_requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let side: std::sync::Arc<dyn hi_ai::Provider> = std::sync::Arc::new(ScriptedProvider {
        steps: std::sync::Mutex::new(vec![ProviderStep::Completion(completion(
            vec![Content::Text("OBJECT\n- side-provider objection".into())],
            1,
            1,
        ))]),
        requests: side_requests.clone(),
        max_tokens: None,
    });
    agent.skeptic_provider = Some(side);
    let mut goal = Goal::new("refactor", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    // The skeptic review was routed to the side provider...
    assert_eq!(
        side_requests.lock().unwrap().len(),
        1,
        "skeptic review routed to the side provider"
    );
    // ...and its objection took effect (blocked the advance, recorded).
    let goal = agent.structured_goal().expect("goal still set");
    assert_eq!(
        goal.skeptic_objections, 1,
        "side-provider objection counted"
    );
    assert_eq!(
        goal.active_index(),
        Some(0),
        "objection blocked the advance"
    );
}

#[tokio::test]
async fn skeptic_escalate_skips_step_and_keeps_driving() {
    // ESCALATE means retrying can't fix it (contradiction / needs the user):
    // the driver skips the step with a visible Failed scar instead of burning
    // the retry budget, and the drive moves to the next step.
    let workspace = IsolatedWorkspace::new("goal-skeptic-escalate");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let changed = workspace.path("changed.rs");
    let responses = vec![
        write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        ),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text(
                "ESCALATE\n- the sub-goal contradicts the objective's schema choice".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut goal = Goal::new(
        "ship it",
        vec!["conflicted step".into(), "next step".into()],
    );
    goal.team = true;
    agent.set_structured_goal(Some(goal)).unwrap();
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    let goal = agent.structured_goal().unwrap();
    assert_eq!(
        goal.sub_goals[0].status,
        GoalStatus::Failed,
        "escalated step is a visible scar"
    );
    assert_eq!(goal.sub_goals[0].attempts, 0, "no retry budget burned");
    assert!(
        goal.sub_goals[0]
            .notes
            .iter()
            .any(|n| n.contains("reviewer escalated") && n.contains("schema choice")),
        "notes: {:?}",
        goal.sub_goals[0].notes
    );
    assert_eq!(goal.active_index(), Some(1), "drive moved to the next step");
    assert_eq!(goal.status, GoalStatus::Active);
    assert!(goal.should_auto_drive());
    assert_eq!(goal.skeptic_escalations, 1);
    assert_eq!(goal.last_skeptic_status, Some(SkepticStatus::Escalated));
    assert!(
        ui.statuses.iter().any(|s| s.contains("skeptic escalated")),
        "statuses: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn skeptic_context_includes_prior_notes_for_anti_ratchet() {
    // On a re-review the skeptic must see the step's prior notes so it can
    // confirm they're addressed rather than raising fresh objections.
    let workspace = IsolatedWorkspace::new("goal-skeptic-prior-notes");
    let mut cfg = workspace.config();
    cfg.subagents.long_horizon = true;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.subagents.skeptic_model = Some("skeptic".into());
    cfg.gates.review = ReviewPolicy::Off;
    let changed = workspace.path("changed.rs");
    let steps = vec![
        ProviderStep::Completion(write_content_completion(
            &changed.to_string_lossy(),
            "a substantial implementation body, comfortably past the trivial-diff exemption",
        )),
        ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ProviderStep::Completion(completion(vec![Content::Text("APPROVE".into())], 1, 1)),
    ];
    let (mut agent, requests) = scripted_agent(steps, cfg);
    let mut goal = Goal::new("ship it", vec!["step one".into(), "step two".into()]);
    goal.team = true;
    goal.sub_goals[0]
        .notes
        .push("reviewer objected — address then continue:\nthe empty-input case crashes".into());
    agent.set_structured_goal(Some(goal)).unwrap();

    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let recorded = requests.lock().unwrap();
    let skeptic_request = recorded
        .last()
        .unwrap()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        skeptic_request.contains("Prior review notes on this step"),
        "anti-ratchet section present: {skeptic_request}"
    );
    assert!(
        skeptic_request.contains("empty-input case crashes"),
        "the prior note itself is shown"
    );
    assert!(
        skeptic_request.contains("the bar does not rise"),
        "anti-ratchet contract stated in context"
    );
}

#[tokio::test]
async fn completion_audit_input_names_the_round() {
    let workspace = IsolatedWorkspace::new("goal-audit-round-line");
    std::fs::write(workspace.path("plan.md"), quant_plan_doc()).unwrap();
    let changed = workspace.path("changed.rs");
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                &changed.to_string_lossy(),
                "a substantial implementation body, comfortably past the trivial-diff exemption",
            )),
            ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
            ProviderStep::Completion(completion(vec![Content::Text("COMPLETE".into())], 1, 1)),
        ],
        audit_cfg(&workspace),
    );
    let mut goal = single_step_goal();
    goal.audit_rounds = 2;
    agent.set_structured_goal(Some(goal)).unwrap();

    agent.run_turn("go", &mut RecUi::default()).await.unwrap();

    let recorded = requests.lock().unwrap();
    let audit_request = recorded
        .last()
        .unwrap()
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        audit_request.contains("Audit round: 2"),
        "round number anchors the anti-ratchet rule: {audit_request}"
    );
    assert!(
        audit_request.contains("the bar does NOT rise between rounds"),
        "auditor prompt carries the anti-ratchet contract"
    );
}
