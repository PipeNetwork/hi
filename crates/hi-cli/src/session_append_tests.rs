use super::*;

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
