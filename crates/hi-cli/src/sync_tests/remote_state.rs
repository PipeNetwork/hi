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
        name: Some("Named portal session".into()),
        goal: None,
        decisions: hi_agent::DecisionLog::default(),
        plan: Vec::new(),
        plan_drive_paused: false,
        plan_drive_stall: 0,
        goal_drive_stall: 0,
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
            RECORD_TYPE_CHECKPOINTS.to_string()
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
