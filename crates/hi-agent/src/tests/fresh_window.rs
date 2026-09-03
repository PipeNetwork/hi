use super::common::*;
use super::*;

fn has_new_context(agent: &Agent) -> bool {
    agent.tools.iter().any(|tool| tool.name == "new_context")
}

#[test]
fn new_context_is_inject_only() {
    assert!(
        !hi_tools::TOOL_SPECS
            .iter()
            .any(|tool| tool.name == "new_context")
    );
    assert!(
        !hi_tools::MINIMAL_TOOL_SPECS
            .iter()
            .any(|tool| tool.name == "new_context")
    );
    assert_eq!(hi_tools::new_context_tool_spec().name, "new_context");
}

#[test]
fn new_context_advertises_at_half_occupancy_and_stays_sticky() {
    let mut cfg = config();
    cfg.routing.context_window = Some(100_000);
    let mut agent = agent(vec![], cfg);
    agent.report.context_used = 10_000;
    agent.refresh_tools_for_task("fix the parser", TaskIntent::Mutation);
    assert!(
        !has_new_context(&agent),
        "must not advertise below 50% occupancy"
    );

    agent.report.context_used = 50_000;
    agent.refresh_tools_for_task("fix the parser", TaskIntent::Mutation);
    assert!(has_new_context(&agent), "advertise at ≥50% occupancy");

    agent.report.context_used = 10_000;
    agent.refresh_tools_for_task("fix the parser", TaskIntent::Mutation);
    assert!(
        has_new_context(&agent),
        "catalog stays sticky after the first advertisement"
    );
}

#[test]
fn new_context_is_not_advertised_for_subagents_or_minimal() {
    let mut sub_cfg = config();
    sub_cfg.routing.context_window = Some(100_000);
    sub_cfg.subagents.is_subagent = true;
    let mut subagent = agent(vec![], sub_cfg);
    subagent.report.context_used = 90_000;
    subagent.refresh_tools_for_task("investigate", TaskIntent::ReadOnly);
    assert!(
        !has_new_context(&subagent),
        "subagents never get new_context"
    );

    let mut min_cfg = config();
    min_cfg.routing.context_window = Some(100_000);
    min_cfg.memory.tool_set = ToolSet::Minimal;
    let mut minimal = agent(vec![], min_cfg);
    minimal.report.context_used = 90_000;
    minimal.refresh_tools_for_task("fix the parser", TaskIntent::Mutation);
    assert!(
        !has_new_context(&minimal),
        "Minimal tool set does not inject new_context"
    );
}

#[test]
fn new_context_rejects_low_occupancy_and_a_second_call() {
    let mut cfg = config();
    cfg.routing.context_window = Some(100_000);
    let mut agent = agent(vec![], cfg);
    agent.report.context_used = 20_000;

    let denied = agent.handle_new_context();
    assert_eq!(denied.status, hi_tools::ToolStatus::Failed);
    assert!(
        denied.content.contains("50%"),
        "low occupancy error: {}",
        denied.content
    );

    agent.report.context_used = 50_000;
    let first = agent.handle_new_context();
    assert_eq!(first.status, hi_tools::ToolStatus::Succeeded);
    let second = agent.handle_new_context();
    assert_eq!(second.status, hi_tools::ToolStatus::Failed);
    assert!(
        second.content.contains("already used"),
        "once per turn: {}",
        second.content
    );
}

#[test]
fn fresh_window_keeps_goal_and_current_task_without_a_summary() {
    let mut agent = agent(vec![], config());
    agent.set_goal(Some("keep this goal".into()));
    agent.task.set_task(Some("fix the parser".into()), None);
    agent.messages_mut().push(Message::user("old question"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    agent.messages_mut().push(Message::user("later aside"));

    let mut ui = RecordingUi::default();
    agent.compact_fresh_window(&mut ui).unwrap();

    assert_eq!(agent.messages().len(), 2, "{:?}", agent.messages());
    assert_eq!(agent.messages()[0].role, Role::System);
    let kept = agent.messages()[1].text();
    assert!(kept.contains("fix the parser"), "current task kept: {kept}");
    assert!(!kept.contains("old question"), "old user turn dropped");
    assert!(!kept.contains("later aside"), "other user turns dropped");
    assert!(
        !kept.contains("[CONTEXT COMPACTION"),
        "no-summary reset must not insert a recap: {kept}"
    );
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(block.contains("keep this goal"), "goal survives: {block}");
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("fresh context window")),
        "status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn compact_window_kind_uses_the_no_summary_path() {
    let mut agent = agent(vec![], config());
    agent.task.set_task(Some("ship the exporter".into()), None);
    agent.messages_mut().push(Message::user("q1"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("a1".into())]));

    agent
        .compact_with(CompactionKind::FreshWindow, &mut NullUi)
        .await
        .unwrap();

    assert_eq!(agent.messages().len(), 2);
    assert!(agent.messages()[1].text().contains("ship the exporter"));
    assert!(
        !agent.messages()[1].text().contains("q1"),
        "the prior conversation must not survive in the retained task message"
    );
}

#[test]
fn applying_new_context_drops_conversation_and_keeps_the_task() {
    let mut cfg = config();
    cfg.routing.context_window = Some(100_000);
    let mut agent = agent(vec![], cfg);
    agent.task.set_task(Some("fix the parser".into()), None);
    agent.report.context_used = 60_000;
    agent.messages_mut().push(Message::user("old question"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));

    let result = agent.handle_new_context();
    assert_eq!(result.status, hi_tools::ToolStatus::Succeeded);
    assert!(agent.token_budget.take_pending_fresh_window());

    let mut ui = RecordingUi::default();
    let task = agent.task.last_task_prompt.clone();
    agent.apply_fresh_window(&mut ui, task.as_deref()).unwrap();

    assert_eq!(agent.messages().len(), 2);
    assert!(agent.messages()[1].text().contains("fix the parser"));
    assert!(
        !agent
            .messages()
            .iter()
            .any(|m| m.text().contains("old question"))
    );
    assert_eq!(agent.report.context_used, 0);
    assert_eq!(agent.token_budget.window_id, 1);
}

#[tokio::test]
async fn new_context_during_turn_reanchors_turn_boundaries() {
    let mut cfg = config();
    cfg.routing.context_window = Some(100_000);
    cfg.memory.finalize = true;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "fresh".into(),
                    name: "new_context".into(),
                    arguments: "{}".into(),
                }],
                60_000,
                1,
            ),
            completion(
                vec![Content::Text("The current state is healthy.".into())],
                10,
                8,
            ),
        ],
        cfg,
    );
    agent.report.context_used = 60_000;
    for index in 0..3 {
        agent
            .messages_mut()
            .push(Message::user(format!("old question {index}")));
        agent
            .messages_mut()
            .push(Message::assistant(vec![Content::Text(format!(
                "old answer {index}"
            ))]));
    }

    let outcome = agent
        .run_turn("explain the current state", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(agent.token_budget.window_id, 1);
    assert!(
        agent
            .messages()
            .iter()
            .all(|message| !message.text().contains("old question")),
        "discarded history must stay discarded: {:?}",
        agent.messages()
    );
    assert!(
        agent
            .messages()
            .iter()
            .any(|message| message.text().contains("The current state is healthy.")),
        "the post-reset answer must remain in the active turn"
    );
}

#[test]
fn fresh_window_resets_drive_stall_so_the_goal_can_keep_running() {
    let mut cfg = config();
    cfg.subagents.long_horizon = true;
    let mut agent = agent(vec![], cfg);
    assert!(
        agent
            .set_structured_goal(Some(Goal::new("ship it", vec!["step one".into()])))
            .unwrap()
    );
    agent.restore_goal_drive(3, Vec::new());
    assert_eq!(agent.goal_drive_stall(), 3);

    agent.compact_fresh_window(&mut NullUi).unwrap();

    assert_eq!(agent.goal_drive_stall(), 0);
    assert!(
        matches!(
            agent.drive_decision(None),
            DriveAction::Enqueue(DriveKind::Goal)
        ),
        "unparked after a new window epoch"
    );
}

#[test]
fn plan_recovery_window_drops_poisoned_history_but_keeps_stall_evidence() {
    let mut agent = agent(vec![], config());
    let evidence_hash = "a".repeat(64);
    agent.restore_plan(vec![hi_tools::PlanStep {
        title: "Build vote transaction".into(),
        status: hi_tools::PlanStatus::Active,
    }]);
    agent.restore_plan_drive_with_policy(false, false, 2, vec![evidence_hash.clone()]);
    agent.messages_mut().push(Message::user("old drive prompt"));
    agent.messages_mut().push(Message::assistant(vec![
        Content::Text("Now I have everything I need.".into()),
        Content::ToolCall {
            id: "read-again".into(),
            name: "read".into(),
            arguments: r#"{"path":"src/network.rs"}"#.into(),
        },
    ]));
    agent
        .messages_mut()
        .push(Message::tool_result("read-again", "same old output"));

    let mut ui = RecordingUi::default();
    agent.apply_plan_recovery_window(&mut ui).unwrap();

    assert_eq!(agent.messages().len(), 1, "old loop history was retained");
    assert_eq!(agent.messages()[0].role, Role::System);
    assert_eq!(agent.plan_drive_stall(), 2, "recovery renewed the stall");
    assert_eq!(
        agent.plan_drive_evidence.snapshot(),
        vec![evidence_hash],
        "recovery made prior inspection novel again"
    );
    assert!(agent.plan_incomplete());
    assert!(ui.statuses.iter().any(|status| {
        status.contains("fresh plan-recovery context")
            && status.contains("no-progress evidence kept")
    }));
}

#[tokio::test]
async fn automatic_drive_window_does_not_renew_plan_orientation() {
    let mut cfg = config();
    cfg.memory.auto_compact = true;
    cfg.routing.context_window = Some(100_000);
    let mut agent = agent(vec![], cfg);
    let evidence_hash = "c".repeat(64);
    agent.restore_plan(vec![hi_tools::PlanStep {
        title: "Implement the parser".into(),
        status: hi_tools::PlanStatus::Active,
    }]);
    agent.restore_plan_drive_with_policy(false, false, 0, vec![evidence_hash.clone()]);
    agent.report.context_used = 90_000;
    agent.messages_mut().push(Message::user("old drive"));

    assert!(
        agent
            .maybe_reclaim_context(&mut NullUi, true)
            .await
            .unwrap()
    );
    assert_eq!(agent.plan_drive_stall(), 0);
    assert_eq!(agent.plan_drive_evidence.snapshot(), vec![evidence_hash]);
    assert_eq!(agent.messages().len(), 1, "old context was not dropped");
}

#[tokio::test]
async fn goal_drive_auto_fresh_window_instead_of_summarize() {
    let mut cfg = config();
    cfg.memory.auto_compact = true;
    cfg.routing.context_window = Some(200_000);
    cfg.subagents.long_horizon = true;
    let mut agent = agent(
        vec![
            completion(vec![Content::Text("kept going".into())], 50, 8),
            completion(vec![Content::Text("still going".into())], 20, 4),
            completion(vec![Content::Text("done".into())], 10, 2),
        ],
        cfg,
    );
    assert!(
        agent
            .set_structured_goal(Some(Goal::new(
                "ship the exporter",
                vec!["step one".into(), "step two".into()],
            )))
            .unwrap()
    );
    agent.restore_goal_drive(3, Vec::new());
    agent.report.context_used = 180_000;
    agent.messages_mut().push(Message::user("old question"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));

    let mut ui = RecordingUi::default();
    agent.run_turn(GOAL_CONTINUE_PROMPT, &mut ui).await.unwrap();

    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("fresh window so the goal can keep running")),
        "status: {:?}",
        ui.statuses
    );
    assert!(
        !agent
            .messages()
            .iter()
            .any(|m| m.text().contains("old question") || m.text().contains("[CONTEXT COMPACTION")),
        "drive occupancy must not summarize: {:?}",
        agent
            .messages()
            .iter()
            .map(|m| m.text())
            .collect::<Vec<_>>()
    );
    let prompt = agent
        .messages()
        .iter()
        .find(|m| m.role == Role::User)
        .map(|m| m.text())
        .unwrap_or_default();
    assert!(
        prompt.contains("ship the exporter") && prompt.contains("step one"),
        "goal continuation kept: {prompt}"
    );
    assert!(
        prompt.contains("new context window"),
        "re-orient note: {prompt}"
    );
    assert_eq!(
        agent.goal_drive_stall(),
        3,
        "automatic transport rollover must not renew semantic progress"
    );
    assert_eq!(agent.token_budget.window_id, 1);
}
