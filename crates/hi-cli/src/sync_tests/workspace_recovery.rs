//! Local workspace transcript handoff into remote session sync.

use super::*;

fn unreachable_config() -> SyncConfig {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    SyncConfig {
        base_url: format!("http://{addr}"),
        api_key: "test-key".into(),
        machine_id: None,
        cwd_digest: None,
    }
}

#[test]
fn local_sync_stage_stays_hidden_and_is_not_mirrored_twice() {
    let session_id = "local-workspace-stage";
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), session_id.into());
    let store = sink.store.clone();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut sync = SyncSession::new(crate::session::JsonlSession::new(path.clone()), sink).unwrap();
    let record = hi_agent::WorkspaceTranscriptExecution {
        schema_version: hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new("local-operation"),
        assistant_content: vec![hi_ai::Content::Text("edited".into())],
        calls: Vec::new(),
        execution: hi_workspace::ExecutionReport::succeeded(Some("digest".into())),
    };

    sync.stage_local_workspace_execution(&record, true).unwrap();
    sync.settle_local_workspace_execution(&record.operation_id)
        .unwrap();
    sync.record(&[], Usage::default()).unwrap();

    let transcript = std::fs::read_to_string(path).unwrap();
    assert!(transcript.contains("workspace_execution_staged"));
    assert!(transcript.contains("workspace_execution_settled"));
    assert!(
        store.ready_records(session_id, 10).unwrap().is_empty(),
        "the hidden local outbox entry must not become a second remote record"
    );
}

#[test]
fn enabling_sync_materializes_a_crash_recovered_tool_batch_before_mirroring() {
    let session_id = "recovered-local-workspace-stage";
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let execution = hi_agent::WorkspaceTranscriptExecution {
        schema_version: hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new("settled-before-crash"),
        assistant_content: vec![hi_ai::Content::ToolCall {
            id: "recovered-call".into(),
            name: "write".into(),
            arguments: "{}".into(),
        }],
        calls: vec![hi_agent::WorkspaceTranscriptCall {
            call_id: "recovered-call".into(),
            name: "write".into(),
            result: "recovered result".into(),
        }],
        execution: hi_workspace::ExecutionReport::succeeded(None),
    };
    let mut local = crate::session::JsonlSession::new(path.clone());
    local
        .stage_local_workspace_execution(&execution, true)
        .unwrap();
    local
        .settle_local_workspace_execution(&execution.operation_id)
        .unwrap();
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), session_id.into());
    let store = sink.store.clone();

    let sync = SyncSession::new(local, sink).unwrap();
    sync.remote_handle().reconcile_jsonl(&path).unwrap();

    assert!(
        std::fs::read_to_string(&path)
            .unwrap()
            .contains("state_replacement"),
        "sync activation must durably tombstone the recovered projection"
    );
    assert_eq!(
        crate::session::load_history(&path).unwrap().messages.len(),
        2
    );
    let records = store.ready_records(session_id, 10).unwrap();
    let replacements = records
        .iter()
        .filter(|record| record.record_type == "state_replacement")
        .collect::<Vec<_>>();
    assert_eq!(replacements.len(), 1);
    assert!(replacements[0].payload_json.contains("recovered result"));
    assert!(records.iter().all(|record| !matches!(
        record.record_type.as_str(),
        "workspace_execution_staged" | "workspace_execution_settled"
    )));
}

#[test]
fn pipefs_pinned_message_write_fails_closed_when_its_local_prefix_cannot_stage() {
    let session_id = "pipefs-prefix-stage-failure";
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), session_id.into());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.jsonl");
    let mut sync = SyncSession::new(crate::session::JsonlSession::new(path.clone()), sink).unwrap();
    sync.remote_handle().set_pipefs_sync_required(true);

    std::fs::write(&path, b"{not-valid-json}\n").unwrap();
    let error = sync
        .record(
            &[hi_ai::Message::user("must precede mutation")],
            Usage::default(),
        )
        .unwrap_err();

    assert!(
        format!("{error:#}").contains("invalid JSONL record"),
        "{error:#}"
    );
    assert!(
        format!("{error:#}").contains("staging the PipeFS transcript prefix"),
        "{error:#}"
    );
}
