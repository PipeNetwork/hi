use super::common::*;
use super::*;
use std::sync::Arc;

type CompactionRecords = Arc<Mutex<Vec<Vec<Message>>>>;
type StateReplacementRecords =
    Arc<Mutex<Vec<(Vec<Message>, Option<Goal>, Vec<Decision>, Vec<PlanStep>)>>>;
type PlanDriveStateRecords = Arc<Mutex<Vec<(bool, u32, bool, bool, Vec<String>)>>>;

struct CompactionRecordingSession {
    records: CompactionRecords,
}

struct StateReplacementRecordingSession {
    records: StateReplacementRecords,
    plan_drive_records: Option<PlanDriveStateRecords>,
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

impl SessionSink for StateReplacementRecordingSession {
    fn record(&mut self, _messages: &[Message], _usage: Usage) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_compaction(&mut self, _messages: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }

    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&Goal>,
        decisions: &DecisionLog,
        plan: &[crate::PlanStep],
    ) -> anyhow::Result<()> {
        self.records.lock().unwrap().push((
            messages.to_vec(),
            goal.cloned(),
            decisions.entries().to_vec(),
            plan.to_vec(),
        ));
        Ok(())
    }

    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        stall: u32,
        resume_on_user_input: bool,
        reset_evidence: bool,
        evidence_add: &[String],
    ) -> anyhow::Result<()> {
        if let Some(records) = &self.plan_drive_records {
            records.lock().unwrap().push((
                paused,
                stall,
                resume_on_user_input,
                reset_evidence,
                evidence_add.to_vec(),
            ));
        }
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

#[test]
fn durable_truncate_records_compaction_boundary() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("old attempt"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    agent.set_session(Box::new(CompactionRecordingSession {
        records: records.clone(),
    }));

    agent.truncate_messages_durable(1).unwrap();

    assert_eq!(agent.messages().len(), 1);
    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].len(), 1);
    assert_eq!(records[0][0].role, Role::System);
}

#[test]
fn durable_truncate_keeps_live_history_when_persistence_fails() {
    let mut agent = agent(vec![], config());
    agent.messages_mut().push(Message::user("old attempt"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    agent.set_session(Box::new(FailingCompactionSession));

    let err = agent.truncate_messages_durable(1).unwrap_err();

    assert!(err.to_string().contains("disk full"));
    assert_eq!(agent.messages().len(), 3);
    assert_eq!(agent.messages()[1].text(), "old attempt");
}

#[test]
fn retry_rewind_restores_state_snapshot_and_rebuilt_system_prompt() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let mut agent = agent(vec![], config());
    agent.set_goal(Some("keep this goal".into()));
    agent.decisions.record(Decision {
        summary: "kept decision".into(),
        rationale: "pre-turn state".into(),
        files: vec!["src/lib.rs".into()],
    });
    agent.refresh_system_message();

    let start = agent.messages().len();
    let snapshot = agent.state_snapshot();

    agent.messages_mut().push(Message::user("old attempt"));
    agent.decisions.record(Decision {
        summary: "discarded decision".into(),
        rationale: "recorded during abandoned attempt".into(),
        files: vec!["src/bad.rs".into()],
    });
    agent.set_goal(Some("discarded goal".into()));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    agent.set_session(Box::new(StateReplacementRecordingSession {
        records: records.clone(),
        plan_drive_records: None,
    }));

    agent.rewind_to_snapshot_durable(start, &snapshot).unwrap();

    assert_eq!(agent.messages().len(), 1);
    assert_eq!(agent.goal(), Some("keep this goal"));
    assert_eq!(agent.decisions().entries().len(), 1);
    assert_eq!(agent.decisions().entries()[0].summary, "kept decision");
    let block = agent.volatile_context_block().unwrap_or_default();
    assert!(block.contains("keep this goal"), "context block: {block}");
    assert!(block.contains("kept decision"), "context block: {block}");
    assert!(
        !block.contains("discarded decision") && !block.contains("discarded goal"),
        "discarded state leaked into the context block: {block}"
    );

    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].0.len(), 1);
    assert!(
        !records[0].0[0].text().contains("kept decision")
            && !records[0].0[0].text().contains("discarded decision"),
        "the recorded system message is stable and carries no decision state"
    );
    assert!(records[0].1.is_none());
    assert_eq!(records[0].2.len(), 1);
    assert_eq!(records[0].2[0].summary, "kept decision");
}

#[test]
fn interrupt_rewind_keeps_plan_progress_instead_of_rolling_back() {
    use hi_tools::{PlanStatus, PlanStep};

    let mut agent = agent(vec![], config());
    // Pre-turn: plan still incomplete.
    agent.goals.last_plan = vec![
        PlanStep {
            title: "step 1".into(),
            status: PlanStatus::Done,
        },
        PlanStep {
            title: "step 2".into(),
            status: PlanStatus::Active,
        },
    ];
    let start = agent.messages().len();
    let snapshot = agent.state_snapshot();

    // Abandoned turn finished the plan (user hit Esc right after update_plan).
    agent.goals.last_plan = vec![
        PlanStep {
            title: "step 1".into(),
            status: PlanStatus::Done,
        },
        PlanStep {
            title: "step 2".into(),
            status: PlanStatus::Done,
        },
    ];
    agent.messages_mut().push(Message::user("work on the plan"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "finished both steps".into(),
        )]));

    agent.rewind_to_snapshot_durable(start, &snapshot).unwrap();

    assert_eq!(agent.messages().len(), start);
    let plan = agent.current_plan();
    assert_eq!(plan.len(), 2, "finished plan stays visible: {plan:?}");
    assert!(
        plan.iter().all(|s| s.status == PlanStatus::Done),
        "interrupt must not roll a completed checklist back: {plan:?}"
    );
}

#[test]
fn cancelled_workspace_rollback_reopens_implementation_completion_durably() {
    use hi_tools::{PlanStatus, PlanStep};

    let records = Arc::new(Mutex::new(Vec::new()));
    let mut subject = agent(vec![], config());
    subject.goals.last_plan = vec![PlanStep {
        title: "Implement the parser fix".into(),
        status: PlanStatus::Active,
    }];
    let start = subject.messages().len();
    let snapshot = subject.state_snapshot();

    // The abandoned turn claimed completion from an edit which cancellation
    // subsequently removed by restoring the pre-turn workspace checkpoint.
    subject.goals.last_plan[0].status = PlanStatus::Done;
    subject
        .messages_mut()
        .push(Message::user("continue the plan"));
    subject.set_session(Box::new(StateReplacementRecordingSession {
        records: records.clone(),
        plan_drive_records: None,
    }));

    subject
        .rewind_to_snapshot_durable_with_workspace_rollback(start, &snapshot, true)
        .unwrap();

    assert_eq!(subject.current_plan()[0].status, PlanStatus::Active);
    let persisted = records.lock().unwrap();
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].3[0].status, PlanStatus::Active);

    // The state handed to a restarted Agent must remain unfinished too.
    let mut resumed = agent(vec![], config());
    resumed.restore_plan(persisted[0].3.clone());
    assert_eq!(resumed.current_plan()[0].status, PlanStatus::Active);
}

#[test]
fn cancelled_rewind_retains_meta_progress_and_resets_drive_scope_durably() {
    use hi_tools::{PlanStatus, PlanStep};

    let state_records = Arc::new(Mutex::new(Vec::new()));
    let drive_records = Arc::new(Mutex::new(Vec::new()));
    let evidence_hash = "a".repeat(64);
    let mut subject = agent(vec![], config());
    subject.goals.last_plan = vec![
        PlanStep {
            title: "Inspect the existing parser".into(),
            status: PlanStatus::Active,
        },
        PlanStep {
            title: "Implement the parser fix".into(),
            status: PlanStatus::Pending,
        },
    ];
    subject.restore_plan_drive_with_policy(false, false, 3, vec![evidence_hash.clone()]);
    let start = subject.messages().len();
    let snapshot = subject.state_snapshot();

    // Read-only bookkeeping survives the workspace rewind, so the next step
    // changes. Stall/evidence from the inspection scope must not poison it.
    subject.goals.last_plan[0].status = PlanStatus::Done;
    subject.goals.last_plan[1].status = PlanStatus::Active;
    subject.set_session(Box::new(StateReplacementRecordingSession {
        records: state_records.clone(),
        plan_drive_records: Some(drive_records.clone()),
    }));

    subject
        .rewind_to_snapshot_durable_with_workspace_rollback(start, &snapshot, true)
        .unwrap();

    assert_eq!(subject.current_plan()[0].status, PlanStatus::Done);
    assert_eq!(subject.current_plan()[1].status, PlanStatus::Active);
    assert_eq!(subject.plan_drive_stall(), 0);
    assert!(subject.plan_drive_evidence.is_empty());

    let persisted_plan = state_records.lock().unwrap()[0].3.clone();
    let drive_record = drive_records.lock().unwrap()[0].clone();
    assert_eq!(drive_record, (false, 0, false, true, Vec::new()));

    // Replay the prior evidence plus the emitted reset exactly as session
    // loading does: the newly-active step starts with a clean drive scope.
    let mut resumed_evidence = vec![evidence_hash];
    if drive_record.3 {
        resumed_evidence.clear();
    }
    resumed_evidence.extend(drive_record.4);
    let mut resumed = agent(vec![], config());
    resumed.restore_plan(persisted_plan);
    resumed.restore_plan_drive_with_policy(
        drive_record.0,
        drive_record.2,
        drive_record.1,
        resumed_evidence,
    );
    assert_eq!(
        resumed.next_plan_step_title(),
        Some("Implement the parser fix")
    );
    assert_eq!(resumed.plan_drive_stall(), 0);
    assert!(resumed.plan_drive_evidence.is_empty());
}

#[test]
fn transactional_user_resume_rewind_serializes_the_durable_interruption_latch() {
    use hi_tools::{PlanStatus, PlanStep};

    let state_records = Arc::new(Mutex::new(Vec::new()));
    let drive_records = Arc::new(Mutex::new(Vec::new()));
    let mut subject = agent(vec![], config());
    subject.restore_plan(vec![
        PlanStep {
            title: "Implement the parser fix".into(),
            status: PlanStatus::Active,
        },
        PlanStep {
            title: "Validate the parser fix".into(),
            status: PlanStatus::Pending,
        },
    ]);
    subject.restore_plan_drive_with_policy(true, true, 0, Vec::new());
    let start = subject.messages().len();
    let snapshot = subject.state_snapshot();

    assert!(
        subject
            .prepare_plan_drive_for_turn(crate::DriveKind::User)
            .unwrap()
    );
    subject.begin_drive_turn(crate::DriveKind::User).unwrap();
    assert!(
        !subject.plan_drive_paused(),
        "the active user turn should render as resumed"
    );
    subject.goals.last_plan[0].status = PlanStatus::Done;
    subject.goals.last_plan[1].status = PlanStatus::Active;
    subject.set_session(Box::new(StateReplacementRecordingSession {
        records: state_records,
        plan_drive_records: Some(drive_records.clone()),
    }));

    subject
        .rewind_to_snapshot_durable(start, &snapshot)
        .unwrap();

    assert_eq!(
        drive_records.lock().unwrap().as_slice(),
        &[(true, 0, true, true, Vec::new())],
        "a rewind during transactional resume must not persist the hidden UI state as running"
    );
}

#[tokio::test]
async fn request_too_large_compacts_prior_context_and_retries_latest_prompt() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
        ],
        config(),
    );
    let huge_old_output = "old tool output ".repeat(20_000);
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "read-1".into(),
            name: "read".into(),
            arguments: r#"{"path":"LICENSE"}"#.into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("read-1", huge_old_output.clone()));

    let mut ui = RecordingUi::default();
    agent
        .run_turn("what is the current bug status?", &mut ui)
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    let contains = |messages: &[Message], needle: &str| {
        messages.iter().flat_map(|m| &m.content).any(|c| match c {
            Content::Text(t) => t.contains(needle),
            Content::Thinking { text, .. } => text.contains(needle),
            Content::ToolCall {
                name, arguments, ..
            } => name.contains(needle) || arguments.contains(needle),
            Content::ToolResult { output, .. } => output.contains(needle),
            _ => false,
        })
    };
    assert_eq!(requests.len(), 2);
    assert!(
        contains(&requests[0], &huge_old_output),
        "first request includes existing context"
    );
    assert!(
        !contains(&requests[1], &huge_old_output),
        "retry omits oversized prior context"
    );
    assert!(
        requests[1]
            .iter()
            .any(|m| m.text().contains("what is the current bug status?")),
        "latest user request is preserved"
    );
    assert!(
        requests[1]
            .iter()
            .any(|m| m.text().contains("[CONTEXT COMPACTION")
                || m.text().contains("previous task")),
        "retry contains a summary of prior turns: {:?}",
        requests[1].iter().map(|m| m.text()).collect::<Vec<_>>()
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("summarized prior turns")),
        "user sees compact recovery status: {:?}",
        ui.statuses
    );
    assert_eq!(agent.messages().last().unwrap().text(), "ok");
}

#[tokio::test]
async fn request_too_large_keeps_last_assistant_recap() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
        ],
        config(),
    );
    let huge_old_output = "old tool output ".repeat(20_000);
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "Gap #1: fold the standalone stream_area into the Run row.".into(),
        )]));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::ToolCall {
            id: "read-1".into(),
            name: "read".into(),
            arguments: r#"{"path":"LICENSE"}"#.into(),
        }]));
    agent
        .messages_mut()
        .push(Message::tool_result("read-1", huge_old_output.clone()));

    agent
        .run_turn(
            "what is the current bug status?",
            &mut RecordingUi::default(),
        )
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let retry = requests[1]
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        retry.contains("Gap #1: fold the standalone stream_area into the Run row."),
        "retry must keep the last assistant recap: {retry}"
    );
    assert!(
        retry.contains("[CONTEXT COMPACTION") || retry.contains("Assistant:"),
        "recap must ride in the compact summary: {retry}"
    );
    assert!(
        !retry.contains(&huge_old_output),
        "retry must still omit oversized tool output"
    );
}

#[tokio::test]
async fn request_too_large_context_drop_records_durable_boundary() {
    let records = Arc::new(Mutex::new(Vec::new()));
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
        ],
        config(),
    );
    agent.messages_mut().push(Message::user("previous task"));
    let huge_recap = "old answer with huge context".repeat(1000);
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(huge_recap.clone())]));
    agent.set_session(Box::new(CompactionRecordingSession {
        records: records.clone(),
    }));

    // See request_too_large_drops_prior_context_and_retries_latest_prompt:
    // avoid expected_mutation so the post-recovery text answer is final.
    agent
        .run_turn(
            "what is the current bug status?",
            &mut RecordingUi::default(),
        )
        .await
        .unwrap();

    let records = records.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert!(
        records[0]
            .iter()
            .any(|message| message.role == Role::System),
        "compact boundary should persist the rebuilt system prompt"
    );
    assert!(
        records[0].iter().any(|message| {
            message.role == Role::User && message.text().contains("[CONTEXT COMPACTION")
        }),
        "compact boundary should persist the folded latest prompt"
    );
    assert!(
        !records[0]
            .iter()
            .any(|message| message.text().contains(&huge_recap)),
        "verbatim discarded context must not survive the durable boundary"
    );
}

#[tokio::test]
async fn request_too_large_keeps_live_context_when_boundary_persistence_fails() {
    let (mut agent, requests) = scripted_agent(vec![ProviderStep::RequestTooLarge], config());
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));
    let start_len = agent.messages().len();
    agent.set_session(Box::new(FailingCompactionSession));
    let mut ui = RecordingUi::default();

    let err = agent.run_turn("fix it", &mut ui).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "failed durable boundary should abort recovery instead of retrying from divergent state"
    );
    assert_eq!(agent.messages().len(), start_len);
    assert_eq!(agent.messages()[1].text(), "previous task");
    assert!(
        ui.statuses.iter().any(
            |s| s.contains("couldn't persist compacted-context retry state")
                || s.contains("couldn't persist dropped-context retry state")
        ),
        "user sees persistence failure: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn request_too_large_latest_prompt_is_removed_after_failed_retry() {
    let (mut agent, _requests) = scripted_agent(vec![ProviderStep::RequestTooLarge], config());
    let start_len = agent.messages().len();
    let mut ui = RecordingUi::default();

    let err = agent
        .run_turn(&"single huge prompt ".repeat(20_000), &mut ui)
        .await
        .unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert_eq!(
        agent.messages().len(),
        start_len,
        "failed oversized prompt is not left in live history"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
        "user gets actionable status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn request_too_large_failed_retry_after_dropping_context_removes_latest_prompt() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::RequestTooLarge,
            ProviderStep::RequestTooLarge,
        ],
        config(),
    );
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "old answer with lots of prior context".into(),
        )]));
    let start_len = agent.messages().len();
    let huge_prompt = "still too large ".repeat(20_000);
    let mut ui = RecordingUi::default();

    let err = agent.run_turn(&huge_prompt, &mut ui).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        3,
        "first overflow compact-retries, second drops prior context, third fails"
    );
    assert_eq!(
        agent.messages().len(),
        1,
        "failed retry should remove the rewritten latest prompt instead of leaving it in history"
    );
    assert!(
        agent
            .messages()
            .iter()
            .all(|message| !message.text().contains(&huge_prompt[..64])),
        "oversized latest prompt should not remain in live history"
    );
    assert!(
        start_len > agent.messages().len(),
        "test must exercise the context-dropping retry path"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
        "user gets actionable status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn request_too_large_second_overflow_drops_after_compact() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::RequestTooLarge,
            ProviderStep::RequestTooLarge,
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 12, 3)),
        ],
        config(),
    );
    let huge_old_output = "old tool output ".repeat(20_000);
    agent.messages_mut().push(Message::user("previous task"));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text(
            "old recap before overflow".into(),
        )]));
    agent
        .messages_mut()
        .push(Message::tool_result("read-1", huge_old_output.clone()));

    let mut ui = RecordingUi::default();
    agent
        .run_turn("what is the current bug status?", &mut ui)
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[1]
            .iter()
            .any(|m| m.text().contains("[CONTEXT COMPACTION")),
        "second request is the compact retry"
    );
    let drop_retry = requests[2]
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        drop_retry.contains("Earlier conversation context was omitted"),
        "third request is the drop-context retry: {drop_retry}"
    );
    assert!(
        drop_retry.contains("what is the current bug status?"),
        "drop retry keeps the latest prompt"
    );
    assert!(
        !drop_retry.contains(&huge_old_output),
        "drop retry omits oversized tool output"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("dropped prior conversation context")),
        "user sees drop recovery status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn context_preflight_rejects_hopeless_oversized_prompt_without_provider_call() {
    let mut cfg = config();
    cfg.routing.context_window = Some(1);
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    let (mut agent, requests) = scripted_agent(vec![], cfg);
    let start_len = agent.messages().len();
    let mut ui = RecordingUi::default();

    let err = agent.run_turn("x", &mut ui).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RequestTooLarge)
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "locally impossible requests should not be sent to the provider"
    );
    assert_eq!(
        agent.messages().len(),
        start_len,
        "failed oversized prompt is not left in live history"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("shorten the prompt")),
        "user gets actionable status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn context_preflight_reduces_output_budget_to_available_headroom() {
    let mut cfg = config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.routing.max_tokens = 8192;
    cfg.routing.requested_max_tokens = 8192;
    // Suppress the ask_user line for non-subagents. Other volatile context
    // (such as the token-budget notice) is intentionally included and is
    // accounted for from the request captured below.
    cfg.subagents.is_subagent = true;
    let (mut agent, requests, max_tokens) = scripted_agent_recording_max_tokens(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            10,
            2,
        ))],
        cfg,
    );
    let prompt = "hello";
    let prompt_estimate =
        hi_ai::estimate_messages_tokens(agent.messages()) + hi_ai::estimate_text_tokens(prompt);
    let expected_headroom = 2048;
    let model_window = prompt_estimate + expected_headroom;
    agent.set_model("m".into(), Some(model_window.try_into().unwrap()), None);

    agent.run_turn(prompt, &mut NullUi).await.unwrap();

    let requests = requests.lock().unwrap();
    let request_prompt_tokens = hi_ai::estimate_messages_tokens(&requests[0]);
    let available = model_window.saturating_sub(request_prompt_tokens) as u32;
    assert_eq!(*max_tokens.lock().unwrap(), vec![available]);
    assert!(available > 0 && available < 8192);
}

#[tokio::test]
async fn context_preflight_drops_prior_context_before_first_provider_call() {
    let mut cfg = config();
    cfg.routing.tool_mode = ToolMode::ChatOnly;
    cfg.routing.max_tokens = 512;
    cfg.routing.requested_max_tokens = 512;
    let (mut agent, requests) = scripted_agent(
        vec![ProviderStep::Completion(completion(
            vec![Content::Text("ok".into())],
            10,
            2,
        ))],
        cfg,
    );
    let huge_old = "old context ".repeat(20_000);
    agent.messages_mut().push(Message::user(huge_old.clone()));
    agent
        .messages_mut()
        .push(Message::assistant(vec![Content::Text("old answer".into())]));

    let prompt = "answer the current question";
    let system_estimate = hi_ai::estimate_messages_tokens(&agent.messages()[..1]);
    let latest_estimate = hi_ai::estimate_text_tokens(prompt);
    let window = system_estimate + latest_estimate + 512 + 128;
    assert!(
        hi_ai::estimate_messages_tokens(agent.messages()) + latest_estimate + 512 > window,
        "test must start over the window before dropping old context"
    );
    agent.set_model("m".into(), Some(window.try_into().unwrap()), None);
    let mut ui = RecordingUi::default();

    agent.run_turn(prompt, &mut ui).await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(
        requests.len(),
        1,
        "prior context should be dropped before the first provider call"
    );
    let sent = requests[0]
        .iter()
        .map(Message::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        sent.contains("Earlier conversation context was omitted"),
        "request includes context omission marker: {sent}"
    );
    assert!(
        sent.contains(prompt),
        "latest user request is preserved: {sent}"
    );
    assert!(
        !sent.contains(&huge_old[..64]),
        "oversized prior context should not be sent"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("dropped prior conversation context")),
        "user sees recovery status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn malformed_stream_retries_and_recovers() {
    // A garbled stream on the first call is silently re-run (with recovery
    // sampling) rather than failing the turn — then it recovers.
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::MalformedStream),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    let mut ui = RecordingUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "retried once after the garble"
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("retrying")),
        "shows a retry, got: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn malformed_stream_retry_is_internal_not_user_visible_status() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::MalformedStream),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    let mut ui = SplitUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();

    assert!(
        ui.statuses.iter().all(|s| !s.contains("retrying")),
        "malformed-stream recovery must not be user-visible status: {:?}",
        ui.statuses
    );
    assert!(
        ui.nudges.iter().any(|s| s.contains("retrying")),
        "internal retry telemetry should remain available to tests: {:?}",
        ui.nudges
    );
}

#[tokio::test]
async fn retry_counts_usage_from_failed_attempt() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::ErrorWithUsage(
                ProviderErrorKind::MalformedStream,
                Usage {
                    input_tokens: 7,
                    output_tokens: 100,
                    ..Default::default()
                },
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(agent.totals().input_tokens, 12);
    assert_eq!(agent.totals().output_tokens, 103);
}

#[tokio::test]
async fn empty_completion_error_is_resampled_too() {
    // The same path catches a provider's empty-completion *error*, not just a
    // content-less Ok response.
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    agent.run_turn("go", &mut NullUi).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn keep_working_recovers_after_empty_retry_budget() {
    let mut cfg = config();
    cfg.loop_limits.max_empty_retries = 1;
    cfg.loop_limits.max_keep_working = 2;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Completion(bash_completion("true")),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert!(
        ui.statuses.iter().any(|s| s.contains("still working")),
        "keep-working should fire after the empty-retry budget: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("/retry")),
        "must not ask the user to retry: {:?}",
        ui.statuses
    );
    assert_eq!(requests.lock().unwrap().len(), 4);
}

#[tokio::test]
async fn unlimited_default_still_bounds_persistent_empty_recovery_at_six_requests() {
    let mut cfg = config();
    // The common test fixture disables keep-working. Restore the production
    // default so this covers the complete composed recovery budget while the
    // ordinary model-round limit is unlimited.
    cfg.loop_limits.max_keep_working = crate::MAX_KEEP_WORKING;
    assert_eq!(cfg.loop_limits.max_steps, u32::MAX);
    assert_eq!(cfg.loop_limits.max_empty_retries, crate::MAX_EMPTY_RETRIES);

    let steps = (0..6)
        .map(|_| ProviderStep::Error(ProviderErrorKind::EmptyCompletion))
        .collect();
    let (mut agent, requests) = scripted_agent(steps, cfg);

    let err = agent
        .run_turn("answer despite a persistently empty provider", &mut NullUi)
        .await
        .unwrap_err();

    assert!(
        err.to_string().contains("scripted provider error"),
        "the terminal empty-completion error should surface: {err:#}"
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        6,
        "initial plus two empty retries on each side of one keep-working recovery"
    );
    assert_eq!(agent.last_turn_telemetry().effective_max_steps, u32::MAX);
    assert!(!agent.last_turn_telemetry().hit_step_cap);
}

#[tokio::test]
async fn unlimited_default_interleaved_503_does_not_reopen_empty_recovery_budget() {
    let mut cfg = config();
    cfg.loop_limits.max_keep_working = crate::MAX_KEEP_WORKING;
    assert_eq!(cfg.loop_limits.max_steps, u32::MAX);

    let empty = || ProviderStep::Error(ProviderErrorKind::EmptyCompletion);
    let temporary_503 = ProviderStep::ErrorMessage(
        ProviderErrorKind::ModelUnavailable,
        r#"{"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#
            .into(),
    );
    let (mut agent, requests) = scripted_agent(
        vec![
            empty(),
            empty(),
            empty(),
            temporary_503,
            empty(),
            empty(),
            empty(),
        ],
        cfg,
    );

    let err = agent
        .run_turn("answer through empty responses and one 503", &mut NullUi)
        .await
        .unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::EmptyCompletion)
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        7,
        "the one 503 replay composes with, but does not reset, the six-request empty budget"
    );
    assert_eq!(agent.last_turn_telemetry().effective_max_steps, u32::MAX);
    assert!(!agent.last_turn_telemetry().hit_step_cap);
}

#[tokio::test]
async fn keep_working_refuses_a_repeated_identical_read() {
    let workspace = IsolatedWorkspace::new("keep-working-same-sig");
    std::fs::write(workspace.path("notes.txt"), "hello\n").unwrap();
    let mut cfg = workspace.config();
    cfg.loop_limits.max_empty_retries = 1;
    cfg.loop_limits.max_keep_working = 2;
    let read = Content::ToolCall {
        id: "r".into(),
        name: "read".into(),
        arguments: serde_json::json!({ "path": "notes.txt", "limit": 20 }).to_string(),
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(vec![read.clone()], 5, 1)),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Completion(completion(vec![read], 5, 1)),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("inspect notes", &mut ui).await.unwrap();
    let still_working = ui
        .statuses
        .iter()
        .filter(|s| s.contains("still working"))
        .count();
    assert_eq!(
        still_working, 1,
        "same-signature read after keep-working is not another recovery: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("different class of action") || s.contains("still working")),
        "first recovery should fire: {:?}",
        ui.statuses
    );
    assert_ne!(
        outcome.status,
        TurnStatus::Completed,
        "repeating the stalled read must not look like progress"
    );
    assert!(
        requests.lock().unwrap().len() >= 4,
        "read, empty stall, keep-working, repeated read"
    );
}

#[tokio::test]
async fn empty_completion_after_tool_results_gets_continuation_nudge() {
    let read_cargo = Content::ToolCall {
        id: "r".into(),
        name: "read".into(),
        arguments: serde_json::json!({ "path": "Cargo.toml", "limit": 20 }).to_string(),
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(vec![read_cargo], 5, 1)),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 8, 2)),
        ],
        config(),
    );

    agent.run_turn("say hi", &mut NullUi).await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    let retry_request = &requests[3];
    assert!(
        retry_request
            .last()
            .is_some_and(|message| message.role == Role::User
                && message
                    .text()
                    .contains("previous model response after the tool results was empty")),
        "retry should include a post-tool empty-response nudge: {retry_request:#?}"
    );
    assert!(
        retry_request
            .windows(2)
            .all(|pair| !(pair[0].role == Role::User && pair[1].role == Role::User)),
        "nudge must not create consecutive user messages: {retry_request:#?}"
    );
}

#[tokio::test]
async fn contentless_completion_after_tool_results_gets_continuation_nudge() {
    let read_cargo = Content::ToolCall {
        id: "r".into(),
        name: "read".into(),
        arguments: serde_json::json!({ "path": "Cargo.toml", "limit": 20 }).to_string(),
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(vec![read_cargo], 5, 1)),
            ProviderStep::Completion(completion(vec![], 8, 0)),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 8, 2)),
        ],
        config(),
    );

    agent.run_turn("say hi", &mut NullUi).await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let retry_request = &requests[2];
    assert!(
        retry_request
            .last()
            .is_some_and(|message| message.role == Role::User
                && message
                    .text()
                    .contains("previous model response after the tool results was empty")),
        "retry should include a post-tool empty-response nudge: {retry_request:#?}"
    );
}

#[tokio::test]
async fn output_cap_error_retries_once_with_advertised_budget() {
    let mut cfg = config();
    cfg.routing.max_tokens = 8192;
    cfg.routing.requested_max_tokens = 8192;
    let (mut agent, requests, max_tokens) = scripted_agent_recording_max_tokens(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::RequestTooLarge,
                "API error 400 Bad Request: max_tokens must be less than or equal to 4096".into(),
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        cfg,
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(*max_tokens.lock().unwrap(), vec![8192, 4096]);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn output_cap_error_without_limit_halves_budget_not_2048() {
    let mut cfg = config();
    cfg.routing.max_tokens = 8192;
    cfg.routing.requested_max_tokens = 8192;
    let (mut agent, _requests, max_tokens) = scripted_agent_recording_max_tokens(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::UnsupportedRequestShape,
                "max_tokens is greater than the provider output limit".into(),
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        cfg,
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(*max_tokens.lock().unwrap(), vec![8192, 4096]);
}

#[tokio::test]
async fn retryable_route_rejection_retries_and_recovers() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                r#"API error 503 Service Unavailable: {"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn temporary_provider_overload_retries_on_the_capacity_budget() {
    let overload = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"glm-5.2 is temporarily overloaded","code":1305},"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            overload(),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn ordinary_rate_limit_gets_one_bounded_retry() {
    let limited = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"quota exceeded","code":"rate_limit"},"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            limited(),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 2);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn ordinary_rate_limit_exhausts_backoff_budget() {
    let limited = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::RateLimit,
            r#"API error 429 Too Many Requests: {"error":{"message":"too many requests","code":"rate_limit"},"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(vec![limited(), limited()], config());

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::RateLimit)
    );
    // Initial attempt plus one client-owned replay. The API already exhausted
    // its compatible provider ladder inside each request.
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn capacity_unavailable_retries_until_the_backend_recovers() {
    let capacity = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::CapacityUnavailable,
            r#"API error 429 Too Many Requests: {"error":"capacity temporarily unavailable","code":"capacity_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            capacity(),
            capacity(),
            capacity(),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 4);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("capacity limited") && status.contains("1/6")),
        "should retry capacity before failing: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("/retry")),
        "must not ask the user to retry while capacity retries remain: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn capacity_unavailable_exhausts_then_surfaces_error() {
    let capacity = || {
        ProviderStep::ErrorMessage(
            ProviderErrorKind::CapacityUnavailable,
            r#"API error 429 Too Many Requests: {"error":"capacity temporarily unavailable","code":"capacity_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
        )
    };
    let (mut agent, requests) = scripted_agent(
        vec![
            capacity(),
            capacity(),
            capacity(),
            capacity(),
            capacity(),
            capacity(),
            capacity(),
        ],
        config(),
    );

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::CapacityUnavailable)
    );
    // Initial attempt plus MAX_CAPACITY_RETRIES (6) client-owned replays.
    assert_eq!(requests.lock().unwrap().len(), 7);
}

#[tokio::test]
async fn capacity_retries_do_not_consume_route_retry_budget() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::CapacityUnavailable,
                r#"API error 429 Too Many Requests: {"error":"capacity temporarily unavailable","code":"capacity_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
            ProviderStep::ErrorMessage(
                ProviderErrorKind::Outage,
                r#"{"error":"external model service unavailable","code":"service_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    assert_eq!(requests.lock().unwrap().len(), 3);
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
}

#[tokio::test]
async fn retryable_route_rejection_exhausts_then_surfaces_error() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                r#"{"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                r#"{"error":"model temporarily unavailable","code":"model_unavailable","retryable":true,"retry_after_seconds":0}"#.into(),
            ),
        ],
        config(),
    );

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::ModelUnavailable)
    );
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn tool_protocol_error_is_resampled_too() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    agent.run_turn("go", &mut NullUi).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert_eq!(requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn invalid_tool_arguments_get_schema_specific_repair_and_recover() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(completion(
                vec![Content::ToolCall {
                    id: "bad-read".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                }],
                5,
                3,
            )),
            ProviderStep::Completion(completion(
                vec![Content::ToolCall {
                    id: "good-read".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"Cargo.toml"}"#.into(),
                }],
                5,
                3,
            )),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );

    agent
        .run_turn("read Cargo.toml and summarize", &mut NullUi)
        .await
        .unwrap();

    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let repair = requests[1]
        .iter()
        .map(Message::text)
        .find(|text| text.contains("schema validation"))
        .expect("retry request should contain schema-specific guidance");
    assert!(repair.contains("`read`"), "{repair}");
    assert!(repair.contains("path"), "{repair}");
}

#[tokio::test]
async fn repeated_invalid_mutation_arguments_fall_back_to_text_and_survive_narration() {
    let workspace = IsolatedWorkspace::new("validation-text-fallback");
    let source = workspace.path("src/main.rs");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    let source_text = source.to_string_lossy();
    let xmlish_write = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{source_text}</arg_value><arg_key>content</arg_key><arg_value>fn main() {{}}\n</arg_value></tool_call>"
    );
    let invalid_write = |id: &str, arguments: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "write".into(),
                arguments: arguments.into(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        invalid_write("bad-1", "{}"),
        invalid_write(
            "bad-2",
            &serde_json::json!({"path": source_text}).to_string(),
        ),
        completion(
            vec![Content::Text("Let me construct the edit carefully.".into())],
            1,
            1,
        ),
        completion(vec![Content::Text(xmlish_write)], 1, 1),
        completion(vec![Content::Text("Implemented the app.".into())], 1, 1),
        bash_completion("true # validate"),
        completion(
            vec![Content::Text(
                "Implemented src/main.rs and validated it successfully.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut cfg = workspace.config();
    cfg.loop_limits.max_repeat_nudges = 1;
    cfg.gates.allow_unverified = true;
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("Build a small command-line app.", &mut ui)
        .await
        .unwrap();

    // This fixture intentionally disables deterministic verification; the
    // fallback behavior under test is the tool-channel recovery itself.
    // Permit the resulting unverified workspace to settle as completed.
    // Production defaults still require a verification seal.
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(std::fs::read_to_string(source).unwrap(), "fn main() {}\n");
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("structured tool arguments kept failing validation")),
        "missing validation fallback status: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses.iter().any(|status| status
            .contains("plain-text tool fallback returned narration instead of a call")),
        "missing fallback-miss recovery status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn read_only_tool_protocol_retry_does_not_recommend_bash() {
    let mut read_only_config = config();
    read_only_config.routing.tool_mode = ToolMode::ReadOnly;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        read_only_config,
    );

    agent.run_turn("go", &mut NullUi).await.unwrap();

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let retry_guidance = requests[1]
        .iter()
        .map(Message::text)
        .find(|text| text.contains("only available tool names"))
        .expect("retry request should contain tool-aware protocol guidance");
    assert!(retry_guidance.contains("`read`"), "{retry_guidance}");
    assert!(!retry_guidance.contains("`bash`"), "{retry_guidance}");
    assert!(!retry_guidance.contains("`write`"), "{retry_guidance}");
}

#[tokio::test]
async fn tool_protocol_after_tool_progress_gets_guidance_nudge() {
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(bash_completion("true")),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        config(),
    );
    agent.run_turn("go", &mut NullUi).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[2]
            .iter()
            .any(|message| message.text().contains("only available tool names")),
        "expected protocol guidance in retry request: {:?}",
        requests[2]
    );
}

#[tokio::test]
async fn alternating_invalid_tool_turns_hit_the_cumulative_circuit_breaker() {
    // A model that alternates a valid tool call with an invalid tool turn keeps
    // resetting the *consecutive* protocol counter (MAX_TOOL_PROTOCOL_RETRIES), so
    // without the cumulative cap the nudge-and-retry loop runs forever (the qtest4
    // wedge). The cumulative circuit-breaker must end the turn instead. Distinct
    // valid calls each round keep the repeat-tool-call guard from firing first, so
    // this isolates the protocol cap; far more pairs than the cap are scripted, so
    // a non-terminating loop would exhaust the script and panic in the provider.
    let mut steps = Vec::new();
    for i in 0..16 {
        steps.push(ProviderStep::Completion(bash_completion(&format!(
            "echo {i}"
        ))));
        steps.push(ProviderStep::Error(ProviderErrorKind::ToolProtocol));
    }
    let (mut agent, _requests) = scripted_agent(steps, config());
    let mut ui = RecUi::default();

    agent.run_turn("go", &mut ui).await.unwrap();

    assert!(
        ui.statuses.iter().any(|s| s.contains("invalid tool turns")),
        "the circuit-breaker should end the turn once cumulative invalid turns are spent: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn keep_working_recovers_after_invalid_tool_turn_budget() {
    let mut cfg = config();
    cfg.loop_limits.max_keep_working = 2;
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(bash_completion("true")),
            ProviderStep::Completion(completion(vec![Content::Text("recovered".into())], 5, 3)),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    assert_eq!(agent.messages().last().unwrap().text(), "recovered");
    assert!(
        ui.statuses.iter().any(|s| s.contains("still working")),
        "keep-working should fire after the invalid-tool-turn budget: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("/retry")),
        "must not ask the user to retry: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn implementation_tool_protocol_exhaustion_falls_back_to_text_tool_calls() {
    let path = temp_file("protocol-text-fallback");
    let path_string = path.to_string_lossy().to_string();
    let xmlish_write = format!(
        "<tool_call>write<arg_key>path</arg_key><arg_value>{path_string}</arg_value><arg_key>content</arg_key><arg_value>ok\n</arg_value></tool_call>"
    );
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Completion(completion(vec![Content::Text(xmlish_write)], 5, 3)),
            ProviderStep::Completion(bash_completion("true # validate")),
            ProviderStep::Completion(completion(
                vec![Content::Text(format!(
                    "Changed {path_string} and validated with true # validate."
                ))],
                5,
                3,
            )),
        ],
        config(),
    );
    let mut ui = RecordingUi::default();
    agent
        .run_turn("/build a small CLI project tracker", &mut ui)
        .await
        .unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "ok\n");
    let _ = std::fs::remove_file(&path);

    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("plain-text tool-call parsing")),
        "expected text-tool fallback status: {:?}",
        ui.statuses
    );
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("validated with true # validate")
    );
    assert!(requests.lock().unwrap().len() >= 7);
}

#[tokio::test]
async fn terminal_error_aborts_without_retry() {
    // A non-resamplable error (auth) fails the turn immediately — no retry.
    let (mut agent, requests) =
        scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], config());
    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();
    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "a terminal error is not retried"
    );
}

#[tokio::test]
async fn terminal_error_resets_stale_turn_telemetry() {
    let (mut agent, _requests) =
        scripted_agent(vec![ProviderStep::Error(ProviderErrorKind::Auth)], config());
    agent.report.last_turn_telemetry = TurnTelemetry {
        repeat_nudges: 99,
        no_progress_streak: 99,
        tool_calls: 42,
        ..TurnTelemetry::default()
    };

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.repeat_nudges, 0);
    assert_eq!(telemetry.tool_calls, 0);
    assert_eq!(telemetry.no_progress_streak, 0);
}

#[tokio::test]
async fn terminal_error_after_recovery_retry_reports_retry_count() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::MalformedStream),
            ProviderStep::Error(ProviderErrorKind::Auth),
        ],
        config(),
    );

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    assert_eq!(
        agent.last_turn_telemetry().recovery_retries,
        1,
        "retry telemetry must survive a terminal error after recovery sampling"
    );
}

#[tokio::test]
async fn terminal_error_after_tool_progress_reports_changed_files_and_tools() {
    let workspace = IsolatedWorkspace::new("retry-terminal-error-progress");
    let path = workspace.path("changed.rs");
    let path_string = path.to_string_lossy().to_string();
    let file_name = path.file_name().unwrap().to_string_lossy().to_string();
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&path_string)),
            ProviderStep::Error(ProviderErrorKind::Auth),
        ],
        workspace.config(),
    );

    let err = agent
        .run_turn("write the file then continue", &mut NullUi)
        .await
        .unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );
    assert!(
        agent
            .last_changed_files()
            .iter()
            .any(|changed| changed == &file_name),
        "changed file should be retained after terminal error: {:?}",
        agent.last_changed_files()
    );
    let telemetry = agent.last_turn_telemetry();
    assert_eq!(telemetry.tool_calls, 1);
    assert!(
        telemetry
            .tool_timeline
            .iter()
            .any(|entry| entry.tool == "write" && entry.path == path_string),
        "write tool telemetry should be retained after terminal error: {:?}",
        telemetry.tool_timeline
    );
}

#[tokio::test]
async fn pipe_serviceability_blip_after_tool_progress_retries_and_settles() {
    let workspace = IsolatedWorkspace::new("retry-pipe-serviceability-after-tool");
    let path = workspace.path("live-write-proof.txt");
    let path_string = path.to_string_lossy().to_string();
    let mut cfg = workspace.config();
    cfg.gates.allow_unverified = true;
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_completion(&path_string)),
            ProviderStep::Error(ProviderErrorKind::EmptyCompletion),
            ProviderStep::ErrorMessage(
                ProviderErrorKind::ModelUnavailable,
                "requested model is not currently serviceable".into(),
            ),
            ProviderStep::Completion(completion(
                vec![Content::Text("Created the requested file.".into())],
                5,
                3,
            )),
        ],
        cfg,
    );
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("write the requested file, then finish", &mut ui)
        .await
        .expect("a transient Pipe serviceability blip should recover");

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert!(path.exists(), "the completed tool mutation was lost");
    assert_eq!(requests.lock().unwrap().len(), 4);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("provider overloaded") && status.contains("1/6")),
        "the serviceability error did not use the capacity retry lane: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn terminal_error_drops_failed_prompt_before_next_turn() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::Auth),
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 1, 1)),
        ],
        config(),
    );

    let err = agent.run_turn("first task", &mut NullUi).await.unwrap_err();
    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );

    agent.run_turn("second task", &mut NullUi).await.unwrap();
    let last_user = agent
        .messages()
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .expect("user prompt recorded");
    let text = last_user.text();
    assert!(
        !text.contains("first task"),
        "failed prompt should not fold into the next turn: {text}"
    );
    assert!(
        text.contains("second task"),
        "next prompt should be cleanly recorded: {text}"
    );
}

#[tokio::test]
async fn protocol_retry_terminal_error_drops_retry_guidance_before_next_turn() {
    let (mut agent, _requests) = scripted_agent(
        vec![
            ProviderStep::Error(ProviderErrorKind::ToolProtocol),
            ProviderStep::Error(ProviderErrorKind::Auth),
            ProviderStep::Completion(completion(vec![Content::Text("ok".into())], 1, 1)),
        ],
        config(),
    );

    let err = agent.run_turn("first task", &mut NullUi).await.unwrap_err();
    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Auth)
    );

    agent.run_turn("second task", &mut NullUi).await.unwrap();
    let last_user = agent
        .messages()
        .iter()
        .rev()
        .find(|message| message.role == Role::User)
        .expect("user prompt recorded");
    let text = last_user.text();
    assert!(
        !text.contains("valid tool calls") && !text.contains("[hi:nudge:"),
        "retry guidance should not leak into the next turn: {text}"
    );
    assert!(
        !text.contains("first task") && text.contains("second task"),
        "next prompt should be cleanly recorded: {text}"
    );
}

#[tokio::test]
async fn terminal_error_persists_usage_before_returning() {
    let records = std::sync::Arc::new(Mutex::new(Vec::new()));
    let (mut agent, _requests) = scripted_agent(
        vec![ProviderStep::ErrorWithUsage(
            ProviderErrorKind::Outage,
            Usage {
                input_tokens: 11,
                output_tokens: 100,
                ..Default::default()
            },
        )],
        config(),
    );
    agent.set_session(Box::new(RecordingSession {
        records: records.clone(),
    }));

    let err = agent.run_turn("go", &mut NullUi).await.unwrap_err();

    assert_eq!(
        hi_ai::provider_error_kind(&err),
        Some(ProviderErrorKind::Outage)
    );
    assert_eq!(
        *records.lock().unwrap(),
        vec![Usage {
            input_tokens: 11,
            output_tokens: 100,
            ..Default::default()
        }]
    );
}
