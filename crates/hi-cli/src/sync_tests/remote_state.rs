//! Session-state, flushing, persistence, and UI-event tests.

use super::*;

#[tokio::test]
async fn remote_session_sink_flushes_records() {
    let server = MockServer::start().await;
    let config = SyncConfig {
        base_url: server.base_url.clone(),
        api_key: "test-key".to_string(),
        machine_id: Some("test-machine".to_string()),
        cwd_digest: Some("0123456789abcdef".to_string()),
    };
    let sink = RemoteSessionSink::new_for_test(config, "test-session-1".to_string());

    // Push a message record via SyncSession (which delegates to the remote sink).
    let local = crate::session::JsonlSession::new(
        std::env::temp_dir().join(format!("hi-sync-test-{}.jsonl", std::process::id())),
    );
    let mut sync = SyncSession::new(local, sink);
    let messages = vec![Message::user("hello world")];
    sync.record(&messages, Usage::default()).unwrap();

    // Flush — should send a POST to the server.
    sync.remote_handle().flush().await.unwrap();

    // The server should have received at least one POST (registration + records).
    assert!(
        server.post_count() >= 1,
        "expected at least 1 POST, got {}",
        server.post_count()
    );

    // Clean up.
    let _ = std::fs::remove_file(
        std::env::temp_dir().join(format!("hi-sync-test-{}.jsonl", std::process::id())),
    );
}

/// A locked or broken outbox store must never fail the turn: the local
/// JSONL is the source of truth, and the outbox mirror is offset-tracked
/// and idempotent, so a failed reconcile defers to the next record. The
/// old behavior surfaced "database is locked" as a failed turn.
#[test]
fn turn_record_survives_broken_outbox_store() {
    let dir = std::env::temp_dir().join(format!(
        "hi-sync-broken-store-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let store_path = dir.join("outbox.sqlite3");
    let store = Arc::new(crate::sync_store::SyncStore::open_at(store_path.clone()).unwrap());
    let sink = RemoteSessionSink::with_store(
        unreachable_config(),
        "broken-store".to_string(),
        None,
        remote_session_http_client(),
        store,
    );
    // Break the store out from under the sink: every reconcile now errors.
    rusqlite::Connection::open(&store_path)
        .unwrap()
        .execute_batch("DROP TABLE session_sync; DROP TABLE record_outbox;")
        .unwrap();

    let jsonl_path = dir.join("session.jsonl");
    let mut sync = SyncSession::new(crate::session::JsonlSession::new(jsonl_path.clone()), sink);
    sync.record(&[Message::user("hello")], Usage::default())
        .expect("turn recording must not fail on outbox errors");
    assert!(
        std::fs::metadata(&jsonl_path).unwrap().len() > 0,
        "local JSONL still records the turn"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn workspace_execution_stages_locally_without_a_live_remote_lease() {
    let sink = RemoteSessionSink::new_for_test(
        unreachable_config(),
        "workspace-stage-after-lease-loss".to_string(),
    );
    sink.set_pipefs_sync_required(true);
    let record = hi_agent::WorkspaceTranscriptExecution {
        schema_version: hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new("operation-1"),
        assistant_content: vec![hi_ai::Content::Text("running edit".into())],
        calls: vec![hi_agent::WorkspaceTranscriptCall {
            call_id: "call-1".into(),
            name: "edit".into(),
            result: "done".into(),
        }],
        execution: hi_workspace::ExecutionReport::succeeded(Some("digest-1".into())),
    };

    sink.stage_workspace_execution(&record)
        .expect("local recovery evidence must not depend on a remote lease");
    let queued = sink
        .store
        .ready_records("workspace-stage-after-lease-loss", 10)
        .unwrap();
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].record_type, RECORD_TYPE_USAGE);
    let compatibility_payload =
        serde_json::from_str::<serde_json::Value>(&queued[0].payload_json).unwrap();
    assert_eq!(compatibility_payload["type"], "workspace_execution");
    assert_eq!(compatibility_payload["operation_id"], "operation-1");
    let replayed = crate::session::load_history_from_records(&[crate::session::RemoteRecord {
        record_type: queued[0].record_type.clone(),
        payload_json: queued[0].payload_json.clone(),
    }])
    .unwrap();
    assert!(
        replayed.messages.is_empty(),
        "carrier is not a visible message"
    );
    assert_eq!(replayed.usage, Usage::default());

    let store = sink.store.clone();
    drop(sink);
    let restarted = RemoteSessionSink::with_store(
        unreachable_config(),
        "workspace-stage-after-lease-loss".to_string(),
        None,
        remote_session_http_client(),
        store,
    );
    restarted.set_pipefs_sync_required(true);
    let causal = restarted.causal_pipefs_transcript_batch().unwrap();
    assert_eq!(causal.records.len(), 1);
    assert_eq!(causal.records[0].record_type, "workspace_execution");
    assert_eq!(
        causal.records[0].payload,
        serde_json::to_value(&record).unwrap()
    );
    let operation = hi_pipefs::CausalOperationReceipt {
        operation_id: "operation-1".into(),
        idempotency_key: "operation-key".into(),
        binding_id: "binding-1".into(),
        binding_epoch: 1,
        replay_class: hi_workspace::ReplayClass::PureWorkspace,
        execution: record.execution.clone(),
    };
    restarted
        .ensure_compatibility_workspace_execution(&operation)
        .expect("the exact queued execution is compatibility recovery proof");
    let mut wrong_batch = causal.clone();
    wrong_batch.records[0].payload["execution"]["detail"] =
        serde_json::Value::String("different result".into());
    let mismatch = restarted
        .acknowledge_causal_pipefs_transcript(&wrong_batch, 1)
        .unwrap_err();
    assert!(mismatch.to_string().contains("does not exactly match"));
    let stale = restarted
        .acknowledge_causal_pipefs_transcript(&causal, 0)
        .unwrap_err();
    assert!(stale.to_string().contains("outbox retained"));
    assert_eq!(
        restarted
            .store
            .ready_records("workspace-stage-after-lease-loss", 10)
            .unwrap()
            .len(),
        1
    );
    restarted
        .acknowledge_causal_pipefs_transcript(&causal, 1)
        .unwrap();
    assert!(
        restarted
            .store
            .ready_records("workspace-stage-after-lease-loss", 10)
            .unwrap()
            .is_empty()
    );
    restarted
        .ensure_compatibility_workspace_execution(&operation)
        .expect("acknowledgement and outbox deletion must be one durable proof transition");
    assert_eq!(
        restarted
            .compatibility_workspace_execution_cursor(&operation)
            .unwrap(),
        1
    );
    let mut mismatched = operation;
    mismatched.execution.detail = Some("different result".into());
    assert!(
        restarted
            .ensure_compatibility_workspace_execution(&mismatched)
            .is_err(),
        "a cursor for another execution must not release recovery evidence"
    );
}

/// The `--session-file` collision bug: session ids derive from the file
/// stem, so a second session at a same-named path inherits the first
/// session's byte offset into a different file. Reconcile must reset the
/// stale offset and proceed — the old behavior failed with "invalid JSONL
/// record at byte N" on every reconcile forever, poisoning whole turns as
/// infrastructure errors.
#[test]
fn reconcile_jsonl_recovers_from_stale_offset_of_a_previous_session_file() {
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), "stale-offset".to_string());
    let dir = std::env::temp_dir().join(format!(
        "hi-sync-stale-offset-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("session.jsonl");

    // Session A: tracking begins before the file exists (as in the real
    // flow), then content appends and reconcile commits the EOF offset.
    sink.reconcile_jsonl(&path).unwrap();
    std::fs::write(
        &path,
        "{\"type\":\"usage\",\"input_tokens\":1,\"output_tokens\":1,\"padding\":\"a fairly long first-session record\"}\n",
    )
    .unwrap();
    sink.reconcile_jsonl(&path).unwrap();
    assert!(
        !sink
            .store
            .ready_records("stale-offset", 32)
            .unwrap()
            .is_empty(),
        "session A records enqueued"
    );

    // Session B replaces the file with a shorter transcript: the stored
    // offset now points past the new EOF.
    std::fs::write(
        &path,
        "{\"type\":\"usage\",\"input_tokens\":2,\"output_tokens\":2}\n",
    )
    .unwrap();
    sink.reconcile_jsonl(&path)
        .expect("past-EOF offset must reset, not fail");

    // A stale offset can also land mid-record (byte before it is not a
    // newline). That, too, must reset instead of failing.
    sink.store.set_jsonl_offset("stale-offset", 5).unwrap();
    sink.reconcile_jsonl(&path)
        .expect("mid-record offset must reset, not fail");

    // After recovery the committed offset is the current file's EOF.
    let len = std::fs::metadata(&path).unwrap().len();
    assert_eq!(sink.store.track_jsonl("stale-offset", &path).unwrap(), len);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn reconcile_jsonl_streams_a_large_backlog_and_leaves_partial_tail_uncommitted() {
    use std::io::{BufWriter, Write};

    const RECORDS: usize = 512;
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), "large-backlog".to_string());
    let dir = std::env::temp_dir().join(format!(
        "hi-sync-large-backlog-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("session.jsonl");
    let mut writer = BufWriter::new(std::fs::File::create(&path).unwrap());
    let padding = "x".repeat(8 * 1024);
    for index in 0..RECORDS {
        writeln!(
            writer,
            "{}",
            serde_json::json!({
                "type": "usage",
                "input_tokens": index,
                "output_tokens": 1,
                "padding": padding,
            })
        )
        .unwrap();
    }
    writer.flush().unwrap();
    drop(writer);
    let complete_len = std::fs::metadata(&path).unwrap().len();
    assert!(complete_len > 4 * 1024 * 1024);
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"{\"type\":\"usage\",\"input_tokens\":999")
        .unwrap();

    sink.reconcile_jsonl(&path).unwrap();
    assert_eq!(
        sink.store.track_jsonl("large-backlog", &path).unwrap(),
        complete_len,
        "an unterminated tail must remain pending"
    );
    assert_eq!(
        sink.store
            .ready_records("large-backlog", RECORDS + 1)
            .unwrap()
            .len(),
        RECORDS
    );

    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b",\"output_tokens\":1}\n")
        .unwrap();
    sink.reconcile_jsonl(&path).unwrap();
    assert_eq!(
        sink.store.track_jsonl("large-backlog", &path).unwrap(),
        std::fs::metadata(&path).unwrap().len()
    );
    assert_eq!(
        sink.store
            .ready_records("large-backlog", RECORDS + 1)
            .unwrap()
            .len(),
        RECORDS + 1
    );

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn session_snapshot_backfills_state_and_title() {
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), "snapshot".to_string());
    let loaded = crate::session::LoadedSession {
        messages: vec![Message::user("first portal prompt")],
        usage: Usage {
            input_tokens: 10,
            output_tokens: 2,
            ..Usage::default()
        },
        checkpoint_refs: vec!["checkpoint-1".into()],
        harness_settings: hi_workspace::SettingLayer {
            source: hi_workspace::SettingSource::Session,
            values: std::collections::BTreeMap::from([(
                hi_workspace::JOB_MAX_ACTIVE.to_string(),
                hi_workspace::SettingValue::Integer(7),
            )]),
        },
        remote_session_id: None,
        pipefs_enabled: Some(true),
        name: Some("Named portal session".into()),
        goal: None,
        decisions: hi_agent::DecisionLog::default(),
        plan: vec![hi_agent::PlanStep {
            title: "finish synced work".into(),
            status: hi_agent::PlanStatus::Pending,
        }],
        plan_drive_paused: true,
        plan_drive_resume_on_user_input: true,
        plan_approval_parked: true,
        plan_drive_stall: 2,
        goal_drive_stall: 3,
        plan_drive_evidence: vec!["a".repeat(64)],
        goal_drive_evidence: vec!["b".repeat(64)],
    };

    sink.seed_snapshot(&loaded).unwrap();

    assert_eq!(
        sink.title.lock().unwrap().as_deref(),
        Some("Named portal session")
    );
    let record_types = sink
        .store
        .ready_records("snapshot", 32)
        .unwrap()
        .iter()
        .map(|record| record.record_type.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        record_types,
        vec![
            RECORD_TYPE_STATE_REPLACEMENT.to_string(),
            RECORD_TYPE_USAGE.to_string(),
            RECORD_TYPE_CHECKPOINTS.to_string(),
            crate::session_harness::RECORD_TYPE.to_string(),
            RECORD_TYPE_PLAN_DRIVE.to_string(),
            RECORD_TYPE_PLAN_APPROVAL.to_string(),
            RECORD_TYPE_GOAL_DRIVE.to_string(),
        ]
    );

    let remote_records = sink
        .store
        .ready_records("snapshot", 32)
        .unwrap()
        .into_iter()
        .map(|record| crate::session::RemoteRecord {
            record_type: record.record_type,
            payload_json: record.payload_json,
        })
        .collect::<Vec<_>>();
    let restored = crate::session::load_history_from_records(&remote_records).unwrap();
    assert!(restored.plan_drive_paused);
    assert!(restored.plan_drive_resume_on_user_input);
    assert!(restored.plan_approval_parked);
    assert_eq!(restored.plan_drive_stall, 2);
    assert_eq!(restored.goal_drive_stall, 3);
    assert_eq!(restored.plan_drive_evidence, vec!["a".repeat(64)]);
    assert_eq!(restored.goal_drive_evidence, vec!["b".repeat(64)]);
    assert_eq!(restored.harness_settings, loaded.harness_settings);
}

#[test]
fn session_snapshot_emits_default_drive_state_to_clear_remote_stale_values() {
    let sink =
        RemoteSessionSink::new_for_test(unreachable_config(), "snapshot-default".to_string());
    let loaded = crate::session::LoadedSession {
        messages: Vec::new(),
        usage: Usage::default(),
        checkpoint_refs: Vec::new(),
        harness_settings: crate::session_harness::empty_layer(),
        remote_session_id: None,
        pipefs_enabled: None,
        name: None,
        goal: None,
        decisions: hi_agent::DecisionLog::default(),
        plan: Vec::new(),
        plan_drive_paused: false,
        plan_drive_resume_on_user_input: false,
        plan_approval_parked: false,
        plan_drive_stall: 0,
        goal_drive_stall: 0,
        plan_drive_evidence: Vec::new(),
        goal_drive_evidence: Vec::new(),
    };

    sink.seed_snapshot(&loaded).unwrap();

    let record_types = sink
        .store
        .ready_records("snapshot-default", 32)
        .unwrap()
        .into_iter()
        .map(|record| record.record_type)
        .collect::<Vec<_>>();
    assert_eq!(
        record_types,
        vec![
            RECORD_TYPE_STATE_REPLACEMENT.to_string(),
            crate::session_harness::RECORD_TYPE.to_string(),
            RECORD_TYPE_PLAN_DRIVE.to_string(),
            RECORD_TYPE_PLAN_APPROVAL.to_string(),
            RECORD_TYPE_GOAL_DRIVE.to_string(),
        ]
    );
}

/// The heartbeat advertises what the host knows: model and context spend.
/// Empty model clears, zero occupancy is ignored (a turn that reported no
/// usage must not erase the last real number), and the window converts to
/// tokens.
#[test]
fn heartbeat_telemetry_tracks_model_and_context() {
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), "telemetry".to_string());

    assert_eq!(lock_recover(&sink.telemetry).model, None);

    sink.set_model_context(" pipe/deepseek-v4-flash-0731 ", Some(1_000_000));
    sink.observe_context_used(81_000);
    {
        let telemetry = lock_recover(&sink.telemetry);
        assert_eq!(
            telemetry.model.as_deref(),
            Some("pipe/deepseek-v4-flash-0731")
        );
        assert_eq!(telemetry.context_max_tokens, Some(1_000_000));
        assert_eq!(telemetry.context_used_tokens, Some(81_000));
    }

    // Zero occupancy: keep the last real number.
    sink.observe_context_used(0);
    assert_eq!(
        lock_recover(&sink.telemetry).context_used_tokens,
        Some(81_000)
    );

    // A model switch replaces both model and window.
    sink.set_model_context("pipe/kimi-3", Some(262_144));
    {
        let telemetry = lock_recover(&sink.telemetry);
        assert_eq!(telemetry.model.as_deref(), Some("pipe/kimi-3"));
        assert_eq!(telemetry.context_max_tokens, Some(262_144));
    }

    // Blank model means "unknown", not an empty string on the wire.
    sink.set_model_context("  ", None);
    let telemetry = lock_recover(&sink.telemetry);
    assert_eq!(telemetry.model, None);
    assert_eq!(telemetry.context_max_tokens, None);
}

#[test]
fn oversized_record_cannot_jam_later_sync() {
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), "oversized".to_string());
    let huge = serde_json::to_string(&Message::user("x".repeat(MAX_RECORD_WIRE_BYTES))).unwrap();
    sink.push(RECORD_TYPE_MESSAGE, &huge);
    sink.push(RECORD_TYPE_STATE_REPLACEMENT, &huge);
    sink.push(
        RECORD_TYPE_MESSAGE,
        &serde_json::to_string(&Message::user("next turn")).unwrap(),
    );

    let pending = sink.store.ready_records("oversized", 64).unwrap();
    assert!(
        pending
            .iter()
            .any(|record| record.record_type == "chunk_part")
    );
    assert_eq!(
        pending
            .iter()
            .filter(|record| record.record_type == "chunk_commit")
            .count(),
        2,
        "both oversized logical records must be committed"
    );
    assert!(
        pending
            .iter()
            .any(|record| record.payload_json.contains("next turn"))
    );
}

#[tokio::test]
async fn title_discovered_after_registration_is_synced() {
    let server = MockServer::start().await;
    let sink = RemoteSessionSink::new_for_test(
        SyncConfig {
            base_url: server.base_url.clone(),
            api_key: "test-key".into(),
            machine_id: None,
            cwd_digest: None,
        },
        "title-sync".into(),
    );
    sink.ensure_registered_now().await.unwrap();
    assert_eq!(
        server.post_count(),
        2,
        "registration plus lease capability probe"
    );

    sink.update_title("Portal work");
    sink.flush().await.unwrap();

    assert_eq!(
        server.post_count(),
        3,
        "registration, lease, and title update"
    );
    assert_eq!(
        sink.registered_title.lock().unwrap().as_deref(),
        Some("Portal work")
    );
}

#[tokio::test]
async fn replacement_session_waits_for_background_handoff() {
    let server = MockServer::start().await;
    let (handoff_tx, handoff_rx) = tokio::sync::oneshot::channel();
    let sink = std::sync::Arc::new(RemoteSessionSink::new_after_drain(
        SyncConfig {
            base_url: server.base_url.clone(),
            api_key: "test-key".into(),
            machine_id: None,
            cwd_digest: None,
        },
        "replacement".into(),
        handoff_rx,
    ));
    sink.push(
        RECORD_TYPE_MESSAGE,
        &serde_json::to_string(&Message::user("after switch")).unwrap(),
    );
    let flushing = {
        let sink = sink.clone();
        tokio::spawn(async move { sink.flush().await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(
        server.post_count(),
        0,
        "registered before predecessor drained"
    );

    handoff_tx.send(()).unwrap();
    flushing.await.unwrap().unwrap();
    assert_eq!(
        server.post_count(),
        3,
        "registration, lease, and record append"
    );
}

#[tokio::test]
async fn remote_ui_flushes_events() {
    let server = MockServer::start().await;
    let config = SyncConfig {
        base_url: server.base_url.clone(),
        api_key: "test-key".to_string(),
        machine_id: None,
        cwd_digest: None,
    };
    let rui = RemoteUi::new_for_test(config, "test-session-2".to_string());

    // Push some events via the MultiplexUi (which calls push_event).
    let mut multi = MultiplexUi {
        primary: Box::new(crate::ui::QuietUi),
        remote: std::sync::Arc::new(rui),
    };
    use hi_agent::Ui;
    multi.assistant_text("hello");
    multi.assistant_end();
    multi.turn_end("[10 in · 5 out]");

    // Flush — should send a POST to the events endpoint.
    // We need to get the RemoteUi back to flush. Since it's behind Arc in
    // the MultiplexUi, we can access it via the remote field.
    multi.remote.flush().await.unwrap();

    assert!(
        server.post_count() >= 1,
        "expected at least 1 POST, got {}",
        server.post_count()
    );
}

#[tokio::test]
async fn uievent_roundtrips_through_sync() {
    // Verify that a UiEvent can be serialized, sent as event_json, and
    // deserialized back — the core of the live streaming protocol.
    let original = hi_tui::event::UiEvent::Text {
        text: "hello from the agent".to_string(),
    };
    let json = serde_json::to_string(&original).unwrap();
    // Simulate what the server receives: an event_json string inside a POST body.
    let body = serde_json::json!({
        "events": [{"event_json": json}]
    });
    let body_str = serde_json::to_string(&body).unwrap();
    // Parse it back.
    let parsed: serde_json::Value = serde_json::from_str(&body_str).unwrap();
    let event_json = parsed["events"][0]["event_json"].as_str().unwrap();
    let decoded: hi_tui::event::UiEvent = serde_json::from_str(event_json).unwrap();
    match decoded {
        hi_tui::event::UiEvent::Text { text } => {
            assert_eq!(text, "hello from the agent");
        }
        _ => panic!("expected Text event"),
    }
}

fn unreachable_config() -> SyncConfig {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    SyncConfig {
        base_url: format!("http://{addr}"),
        api_key: "test-key".to_string(),
        machine_id: None,
        cwd_digest: None,
    }
}

#[tokio::test]
async fn failed_record_flush_keeps_records_and_retries_registration() {
    let sink = RemoteSessionSink::new_for_test(unreachable_config(), "safe-id".to_string());
    sink.push(RECORD_TYPE_MESSAGE, r#"{"role":"user","content":[]}"#);

    assert!(sink.flush().await.is_err());
    assert_eq!(sink.store.ready_records("safe-id", 10).unwrap().len(), 1);
    assert!(!*sink.registered.lock().unwrap());
}

#[tokio::test]
async fn failed_event_flush_keeps_events() {
    let ui = RemoteUi::new_for_test(unreachable_config(), "safe-id".to_string());
    ui.push_event(hi_tui::event::UiEvent::Text {
        text: "keep me".to_string(),
    });

    assert!(ui.flush().await.is_err());
    assert_eq!(ui.store.ready_events("safe-id", 10).unwrap().len(), 1);
}

#[tokio::test]
async fn flush_chunks_batches_to_server_contract_limits() {
    let server = MockServer::start().await;
    let config = SyncConfig {
        base_url: server.base_url.clone(),
        api_key: "test-key".to_string(),
        machine_id: None,
        cwd_digest: None,
    };
    let records = RemoteSessionSink::new_for_test(config.clone(), "record-chunks".to_string());
    for _ in 0..513 {
        records.push(RECORD_TYPE_MESSAGE, r#"{"role":"user","content":[]}"#);
    }
    records.flush().await.unwrap();
    // Registration, lease capability probe, plus two record batches.
    assert_eq!(server.post_count(), 4);

    let events = RemoteUi::new_for_test(config, "event-chunks".to_string());
    for _ in 0..257 {
        events.push_event(hi_tui::event::UiEvent::AssistantEnd);
    }
    events.flush().await.unwrap();
    // Two more event batches (256 + 1).
    assert_eq!(server.post_count(), 6);
}

#[test]
fn session_ids_are_safe_single_path_segments() {
    for valid in ["session-123", "abc_DEF", "2026.07.12"] {
        validate_session_id(valid).unwrap();
    }
    for invalid in ["", "../escape", "with/slash", "contains space", "é"] {
        assert!(
            validate_session_id(invalid).is_err(),
            "accepted {invalid:?}"
        );
    }
}

#[test]
fn unicode_tool_output_clipping_stays_on_char_boundaries() {
    let input = "🦀".repeat(201);
    let clipped = clip_chars(&input, 200);
    assert_eq!(clipped.chars().count(), 201);
    assert!(clipped.ends_with('…'));
}
