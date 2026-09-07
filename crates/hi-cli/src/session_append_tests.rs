use super::*;

#[test]
fn terminal_credit_cannot_appear_in_a_crash_prefix_without_its_receipt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut session = JsonlSession::new(path.clone());
    let mut recovery = hi_agent::TaskRecoveryState::new("finish".into(), 3);
    recovery.remaining = 1;
    session.record_task_recovery(&recovery).unwrap();
    let conservative = hi_agent::Goal::new("finish", vec!["implement".into()]);
    session.record_goal(&conservative).unwrap();
    let prefix = fs::read_to_string(&path).unwrap();
    let mut credited = recovery.clone();
    credited.remaining = 3;
    let mut settled_goal = conservative.clone();
    settled_goal.advance();
    let mut outcome = hi_agent::TurnOutcome::infrastructure_failure("test", None, vec![]);
    outcome.status = hi_agent::TurnStatus::Completed;
    outcome.stop_reason = hi_agent::TurnStopReason::Completed;
    outcome.verification = hi_agent::VerificationStatus::Passed;
    session
        .record_turn_settlement(&outcome, None, Some(&credited), Some(&settled_goal))
        .unwrap();
    let complete = fs::read_to_string(&path).unwrap();
    assert_eq!(complete.lines().count(), prefix.lines().count() + 1);
    let receipt: serde_json::Value =
        serde_json::from_str(complete.lines().last().unwrap()).unwrap();
    assert_eq!(receipt["type"], "turn_outcome");
    assert_eq!(receipt["task_recovery"]["remaining"], 3);
    for (bytes, expected, expected_goal) in [
        (&prefix, &recovery, &conservative),
        (&complete, &credited, &settled_goal),
    ] {
        let crashed = dir.path().join("crash.jsonl");
        fs::write(&crashed, bytes).unwrap();
        let records = bytes
            .lines()
            .map(|payload| RemoteRecord {
                record_type: serde_json::from_str::<serde_json::Value>(payload).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .into(),
                payload_json: payload.into(),
            })
            .collect::<Vec<_>>();
        for loaded in [
            load_history(&crashed).unwrap(),
            load_history_from_records(&records).unwrap(),
        ] {
            assert_eq!(&loaded.task_recovery, expected);
            assert_eq!(loaded.goal.as_ref(), Some(expected_goal));
        }
        let summary = session_goal_summary(&crashed).unwrap();
        assert_eq!(summary.done, expected_goal.completed_count());
    }
    for invalid in [
        serde_json::json!({"schema_version":99}),
        serde_json::json!({}),
    ] {
        let mut broken = receipt.clone();
        broken["task_recovery"] = invalid;
        let payload = broken.to_string();
        fs::write(&path, format!("{prefix}{payload}\n")).unwrap();
        assert!(load_history(&path).is_err());
        assert_eq!(session_goal_summary(&path).unwrap().done, 0);
        assert!(
            load_history_from_records(&[RemoteRecord {
                record_type: "turn_outcome".into(),
                payload_json: payload,
            }])
            .is_err()
        );
    }
}

#[test]
fn task_recovery_survives_local_remote_cache_compaction_and_rewind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut session = JsonlSession::new(path.clone());
    let mut recovery = hi_agent::TaskRecoveryState::new("finish the same task".into(), 3);
    recovery.remaining = 0;
    recovery.interventions = 3;
    recovery.exhausted = true;
    recovery.last_reason = Some("no objective improvement".into());
    session.record_task_recovery(&recovery).unwrap();
    session
        .record(
            &[Message::user(hi_agent::PLAN_DRIVE_PROMPT)],
            Usage::default(),
        )
        .unwrap();
    session
        .record_compaction(&[Message::user("summary")])
        .unwrap();
    session
        .record_state_replacement(&[], None, &hi_agent::DecisionLog::default(), &[])
        .unwrap();
    let records = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            RemoteRecord {
                record_type: value
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("message")
                    .into(),
                payload_json: line.into(),
            }
        })
        .collect::<Vec<_>>();
    for restored in [
        load_history(&path).unwrap(),
        load_history_from_records(&records).unwrap(),
    ] {
        assert_eq!(restored.task_recovery, recovery);
        assert!(restored.messages.is_empty());
        let cache = dir.path().join("cached.jsonl");
        cache_loaded_session(&cache, &restored).unwrap();
        assert_eq!(load_history(&cache).unwrap().task_recovery, recovery);
    }
}

#[test]
fn recognized_future_or_malformed_control_state_fails_both_restoration_paths() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut future = hi_agent::TaskRecoveryState::default();
    future.schema_version += 1;
    for payload in [
        serde_json::json!({"type":"task_recovery","state":future}).to_string(),
        r#"{"type":"task_recovery","state":{}}"#.into(),
        r#"{"schema_version":99,"kind":{"type":"opaque_boundary"}}"#.into(),
    ] {
        fs::write(&path, format!("{payload}\n")).unwrap();
        assert!(load_history(&path).is_err());
        assert!(
            load_history_from_records(&[RemoteRecord {
                record_type: "task_recovery".into(),
                payload_json: payload
            }])
            .is_err()
        );
    }
}

#[test]
fn versioned_snapshot_records_restore_legacy_state_and_reject_future_versions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut snapshot = hi_agent::SessionReducer::new().snapshot();
    snapshot.reducer_version = 2;
    snapshot.state.reducer_version = 2;
    snapshot
        .state
        .messages
        .push(Message::user("older snapshot"));
    let payload = serde_json::json!({"type":"reducer_snapshot", "snapshot":snapshot}).to_string();
    fs::write(&path, format!("{payload}\n")).unwrap();
    assert_eq!(
        load_history(&path).unwrap().messages[0].text(),
        "older snapshot"
    );
    let remote = [RemoteRecord {
        record_type: "reducer_snapshot".into(),
        payload_json: payload,
    }];
    assert_eq!(
        load_history_from_records(&remote).unwrap().messages[0].text(),
        "older snapshot"
    );
    snapshot.reducer_version = u32::MAX;
    let payload = serde_json::json!({"type":"reducer_snapshot", "snapshot":snapshot}).to_string();
    fs::write(&path, format!("{payload}\n")).unwrap();
    assert!(load_history(&path).is_err());
    assert!(
        load_history_from_records(&[RemoteRecord {
            record_type: "reducer_snapshot".into(),
            payload_json: payload
        }])
        .is_err()
    );
}

#[test]
fn staged_workspace_execution_replays_once_across_every_crash_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let assistant_content = vec![hi_ai::Content::ToolCall {
        id: "call-1".into(),
        name: "write".into(),
        arguments: r#"{"path":"note.txt"}"#.into(),
    }];
    let execution = hi_agent::WorkspaceTranscriptExecution {
        schema_version: hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new("operation-1"),
        assistant_content: assistant_content.clone(),
        calls: vec![hi_agent::WorkspaceTranscriptCall {
            call_id: "call-1".into(),
            name: "write".into(),
            result: "wrote note.txt".into(),
        }],
        execution: hi_workspace::ExecutionReport::succeeded(None),
    };
    let mut session = JsonlSession::new(path.clone());

    assert!(
        session.stage_workspace_execution(&execution).is_err(),
        "local JSONL alone must not satisfy a PipeFS causal stage"
    );
    session
        .stage_local_workspace_execution(&execution, true)
        .unwrap();

    let staged = fs::read_to_string(&path).unwrap();
    assert!(staged.contains("workspace_execution_staged"));
    assert!(staged.contains("wrote note.txt"));
    let pending = load_history(&path).unwrap();
    assert!(
        pending
            .messages
            .last()
            .unwrap()
            .text()
            .contains("workspace recovery pending"),
        "a pre-settlement crash must remain visibly recovery-pending"
    );
    assert_eq!(
        pending.messages.len(),
        1,
        "an unacknowledged success must not be reconstructed as settled"
    );

    let portable_cache = dir.path().join("portable.jsonl");
    cache_loaded_session(&portable_cache, &pending).unwrap();
    let cached = load_history(&portable_cache).unwrap();
    assert_eq!(cached.messages.len(), 1);
    assert!(cached.pending_execution_snapshot.is_some());
    let mut adopted = JsonlSession::new(portable_cache.clone());
    adopted
        .settle_local_workspace_execution(&execution.operation_id)
        .unwrap();
    let recovered_cache = load_history(&portable_cache).unwrap();
    assert_eq!(recovered_cache.messages.len(), 2);
    assert_eq!(
        serde_json::to_value(&recovered_cache.messages[1]).unwrap(),
        serde_json::to_value(Message::tool_result("call-1", "wrote note.txt")).unwrap()
    );

    session
        .settle_local_workspace_execution(&execution.operation_id)
        .unwrap();
    let settled = load_history(&path).unwrap();
    assert_eq!(settled.messages.len(), 2);
    assert_eq!(
        serde_json::to_value(&settled.messages[1]).unwrap(),
        serde_json::to_value(Message::tool_result("call-1", "wrote note.txt")).unwrap()
    );

    let visible = [
        Message::assistant(assistant_content),
        Message::tool_result("call-1", "wrote note.txt"),
    ];
    session.record(&visible, Usage::default()).unwrap();
    let loaded = load_history(&path).unwrap();
    assert_eq!(
        serde_json::to_value(loaded.messages).unwrap(),
        serde_json::to_value(visible).unwrap(),
        "normal post-settlement persistence remains the sole visible copy"
    );

    let mut audit = execution;
    audit.operation_id = hi_workspace::OperationId::new("audit-only-operation");
    session
        .stage_local_workspace_execution(&audit, false)
        .unwrap();
    session
        .settle_local_workspace_execution(&audit.operation_id)
        .unwrap();
    assert_eq!(
        load_history(&path).unwrap().messages.len(),
        2,
        "settled audit-only operations never leak synthetic tool calls on resume"
    );
}

#[test]
fn settled_stage_is_not_deduplicated_against_an_identical_earlier_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let visible = [
        Message::assistant(vec![hi_ai::Content::ToolCall {
            id: "same-call".into(),
            name: "write".into(),
            arguments: "{}".into(),
        }]),
        Message::tool_result("same-call", "same result"),
    ];
    let mut session = JsonlSession::new(path.clone());
    session.record(&visible, Usage::default()).unwrap();
    let execution = hi_agent::WorkspaceTranscriptExecution {
        schema_version: hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new("repeated-operation"),
        assistant_content: visible[0].content.clone(),
        calls: vec![hi_agent::WorkspaceTranscriptCall {
            call_id: "same-call".into(),
            name: "write".into(),
            result: "same result".into(),
        }],
        execution: hi_workspace::ExecutionReport::succeeded(None),
    };
    session
        .stage_local_workspace_execution(&execution, true)
        .unwrap();
    session
        .settle_local_workspace_execution(&execution.operation_id)
        .unwrap();

    let loaded = load_history(&path).unwrap();
    assert_eq!(loaded.messages.len(), 4);
    assert_eq!(
        serde_json::to_value(&loaded.messages[..2]).unwrap(),
        serde_json::to_value(&loaded.messages[2..]).unwrap(),
        "the stage boundary distinguishes a repeated operation from prior history"
    );
}

#[test]
fn recovered_stage_keeps_its_original_order_before_later_messages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut session = JsonlSession::new(path.clone());
    session
        .record(&[Message::user("before")], Usage::default())
        .unwrap();
    let execution = hi_agent::WorkspaceTranscriptExecution {
        schema_version: hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new("ordered-operation"),
        assistant_content: vec![hi_ai::Content::ToolCall {
            id: "ordered-call".into(),
            name: "write".into(),
            arguments: "{}".into(),
        }],
        calls: vec![hi_agent::WorkspaceTranscriptCall {
            call_id: "ordered-call".into(),
            name: "write".into(),
            result: "ordered result".into(),
        }],
        execution: hi_workspace::ExecutionReport::succeeded(None),
    };
    session
        .stage_local_workspace_execution(&execution, true)
        .unwrap();
    session
        .settle_local_workspace_execution(&execution.operation_id)
        .unwrap();
    session
        .record(&[Message::user("later")], Usage::default())
        .unwrap();

    let loaded = load_history(&path).unwrap();
    assert_eq!(loaded.messages.len(), 4);
    assert_eq!(loaded.messages[0].text(), "before");
    assert_eq!(loaded.messages[1].role, Role::Assistant);
    assert_eq!(loaded.messages[3].text(), "later");
    assert_eq!(
        serde_json::to_value(&loaded.messages[2]).unwrap(),
        serde_json::to_value(Message::tool_result("ordered-call", "ordered result")).unwrap()
    );
}

#[test]
fn append_after_interrupted_tail_preserves_next_message_and_approval() {
    for tail in [
        b"{\"role\":\"assistant\"".as_slice(),
        b"{\"text\":\"\xf0\x9f",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let original = Message::user("original task");
        let mut bytes = serde_json::to_vec(&original).unwrap();
        bytes.push(b'\n');
        bytes.extend_from_slice(tail);
        fs::write(&path, &bytes).unwrap();

        let mut session = JsonlSession::new(path.clone());
        session.record_plan_approval_parked(true).unwrap();
        let next = Message::user("continue after restart");
        session
            .record(std::slice::from_ref(&next), Usage::default())
            .unwrap();

        let loaded = load_history(&path).unwrap();
        assert!(
            loaded.plan_approval_parked,
            "the first new record must survive a broken tail"
        );
        assert_eq!(
            serde_json::to_value(loaded.messages).unwrap(),
            serde_json::to_value([original, next]).unwrap()
        );
        assert!(
            fs::read(&path).unwrap().starts_with(&bytes),
            "recovery must preserve the original bytes"
        );
    }
}

#[test]
fn append_preserves_complete_record_without_final_newline() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let original = Message::user("original task");
    fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    let mut session = JsonlSession::new(path.clone());
    let next = Message::user("next task");
    session
        .record(std::slice::from_ref(&next), Usage::default())
        .unwrap();
    assert_eq!(
        serde_json::to_value(load_history(&path).unwrap().messages).unwrap(),
        serde_json::to_value([original, next]).unwrap()
    );
}

#[test]
fn concurrent_appenders_preserve_large_records_and_settings() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    fs::write(&path, b"{\"unfinished\":").unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(5));
    std::thread::scope(|scope| {
        for writer in 0..4 {
            let path = &path;
            let barrier = barrier.clone();
            scope.spawn(move || {
                let mut session = JsonlSession::new(path.clone());
                barrier.wait();
                for record in 0..8 {
                    session
                        .record(
                            &[Message::user(format!(
                                "{writer}:{record} {}",
                                "x".repeat(32 * 1024)
                            ))],
                            Usage::default(),
                        )
                        .unwrap();
                }
            });
        }
        let path = &path;
        scope.spawn(move || {
            barrier.wait();
            for _ in 0..8 {
                crate::session_harness::append(path, &crate::session_harness::empty_layer())
                    .unwrap();
            }
        });
    });
    assert_eq!(load_history(&path).unwrap().messages.len(), 32);
    let records = fs::read_to_string(&path).unwrap();
    assert_eq!(
        records
            .lines()
            .skip(1)
            .filter(|line| serde_json::from_str::<serde_json::Value>(line).is_ok())
            .count(),
        72
    );
}

#[tokio::test(flavor = "current_thread")]
async fn jsonl_lock_wait_keeps_runtime_responsive_and_retains_accepted_recovery_write() {
    struct SignallingJsonl {
        inner: JsonlSession,
        started: Option<tokio::sync::oneshot::Sender<()>>,
        committed: Option<tokio::sync::oneshot::Sender<()>>,
    }
    impl SessionSink for SignallingJsonl {
        fn record(&mut self, messages: &[Message], usage: Usage) -> Result<()> {
            self.inner.record(messages, usage)
        }
        fn record_compaction(&mut self, messages: &[Message]) -> Result<()> {
            self.inner.record_compaction(messages)
        }
        fn record_task_recovery(&mut self, state: &hi_agent::TaskRecoveryState) -> Result<()> {
            if let Some(sender) = self.started.take() {
                let _ = sender.send(());
            }
            self.inner.record_task_recovery(state)?;
            if let Some(sender) = self.committed.take() {
                let _ = sender.send(());
            }
            Ok(())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let locked = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)
        .unwrap();
    locked.lock().unwrap();
    let provider = std::sync::Arc::new(hi_ai::OpenAiProvider::new(
        "http://127.0.0.1:1/v1".into(),
        "unused".into(),
    ));
    let mut config = hi_agent::AgentConfig::default();
    config.paths.workspace_root = dir.path().to_owned();
    config.paths.state_root = dir.path().join(".state");
    let mut agent = hi_agent::Agent::new(provider, config).unwrap();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (finished, committed) = tokio::sync::oneshot::channel();
    agent.set_session(Box::new(SignallingJsonl {
        inner: JsonlSession::new(path.clone()),
        started: Some(entered),
        committed: Some(finished),
    }));
    let mut ui = hi_agent::ui::NullUi;
    let mut turn = Box::pin(agent.run_turn("explain the project", &mut ui));
    tokio::select! {
        result = &mut turn => panic!("turn ended before blocked write: {result:?}"),
        started = tokio::time::timeout(std::time::Duration::from_secs(2), started) => { started.unwrap().unwrap(); }
    }
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(20), &mut turn)
            .await
            .is_err(),
        "the locked write cannot acknowledge commit"
    );
    drop(turn);
    locked.unlock().unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), committed)
        .await
        .unwrap()
        .unwrap();
    agent.session_barrier().await.unwrap();
    assert!(
        !load_history(&path)
            .unwrap()
            .task_recovery
            .objective
            .is_empty()
    );
}
