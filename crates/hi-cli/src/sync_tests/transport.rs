//! Remote transport, retry, lease, and admission tests.

use super::*;

#[test]
fn stale_or_unauthorized_remote_input_polls_are_terminal() {
    for status in [
        reqwest::StatusCode::UNAUTHORIZED,
        reqwest::StatusCode::FORBIDDEN,
        reqwest::StatusCode::NOT_FOUND,
        reqwest::StatusCode::CONFLICT,
        reqwest::StatusCode::GONE,
    ] {
        assert!(remote_input_poll_status_is_terminal(status), "{status}");
    }
    assert!(!remote_input_poll_status_is_terminal(
        reqwest::StatusCode::TOO_MANY_REQUESTS
    ));
    assert!(!remote_input_poll_status_is_terminal(
        reqwest::StatusCode::BAD_GATEWAY
    ));
}

/// A mock that 400s any records POST whose body mentions a poison
/// record type, and 200s everything else (registration included).
async fn start_poison_rejecting_server(
    reject_poison: Arc<std::sync::atomic::AtomicBool>,
) -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
    let (base_url, bodies, _posts) = start_recording_server(reject_poison).await;
    (base_url, bodies)
}

/// The same mock, also handing back the request line of every POST so a
/// test can assert lifecycle calls (`/lease`, `/end`) and not just bodies.
async fn start_recording_server(
    reject_poison: Arc<std::sync::atomic::AtomicBool>,
) -> (
    String,
    Arc<std::sync::Mutex<Vec<String>>>,
    Arc<std::sync::Mutex<Vec<String>>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let accepted_bodies = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let posts = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let bodies = accepted_bodies.clone();
    let post_lines = posts.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(s) => s,
                Err(_) => break,
            };
            let bodies = bodies.clone();
            let post_lines = post_lines.clone();
            let reject_poison = reject_poison.clone();
            tokio::spawn(async move {
                let Ok(request) = read_mock_http_request(&mut sock).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&request).to_string();
                if request.starts_with("POST") {
                    let line = request.lines().next().unwrap_or_default().to_string();
                    post_lines.lock().unwrap().push(line);
                }
                let is_records = request.starts_with("POST") && request.contains("/records");
                let response = if is_records
                    && reject_poison.load(Ordering::SeqCst)
                    && request.contains("\"record_type\":\"poison\"")
                {
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 37\r\nConnection: close\r\n\r\n{\"error\":\"unsupported record_type\"}\n"
                        .to_string()
                } else {
                    if is_records {
                        bodies.lock().unwrap().push(request.clone());
                        let cursor = post_lines.lock().unwrap().len().saturating_mul(1_000);
                        let body = serde_json::json!({ "record_count": cursor }).to_string();
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                            body.len(),
                            body
                        )
                    } else {
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}".to_string()
                    }
                };
                let _ = sock.write_all(response.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}"), accepted_bodies, posts)
}

/// Models a protocol-1 records endpoint: legacy record types are accepted,
/// but the newer native workspace-execution type is rejected.
async fn start_protocol_one_server() -> (String, Arc<std::sync::Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let accepted_bodies = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let bodies = accepted_bodies.clone();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(connection) => connection,
                Err(_) => break,
            };
            let bodies = bodies.clone();
            tokio::spawn(async move {
                let Ok(request) = read_mock_http_request(&mut socket).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&request).to_string();
                let is_records = request.starts_with("POST") && request.contains("/records");
                let rejects_native_type =
                    is_records && request.contains(r#""record_type":"workspace_execution""#);
                let response = if rejects_native_type {
                    "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 37\r\nConnection: close\r\n\r\n{\"error\":\"unsupported record_type\"}\n"
                        .to_string()
                } else {
                    if is_records {
                        bodies.lock().unwrap().push(request);
                    }
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 18\r\nConnection: close\r\n\r\n{\"record_count\":1}"
                        .to_string()
                };
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    (format!("http://{addr}"), accepted_bodies)
}

#[tokio::test]
async fn protocol_one_flushes_workspace_execution_as_invisible_legacy_record() {
    let (base_url, accepted) = start_protocol_one_server().await;
    let session_id = format!("protocol-one-workspace-{}", std::process::id());
    let store = unique_test_sync_store();
    let config = SyncConfig {
        base_url,
        api_key: "test-key".to_string(),
        machine_id: None,
        cwd_digest: None,
    };
    let sink = RemoteSessionSink::with_store(
        config.clone(),
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    sink.set_pipefs_sync_required(true);
    let record = hi_agent::WorkspaceTranscriptExecution {
        schema_version: hi_agent::WorkspaceTranscriptExecution::SCHEMA_VERSION,
        operation_id: hi_workspace::OperationId::new("legacy-operation"),
        assistant_content: vec![hi_ai::Content::Text("edited".into())],
        calls: Vec::new(),
        execution: hi_workspace::ExecutionReport::succeeded(Some("digest".into())),
    };

    sink.stage_workspace_execution(&record).unwrap();
    drop(sink);
    let restarted = RemoteSessionSink::with_store(
        config,
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    restarted.set_pipefs_sync_required(true);
    restarted.stage_workspace_execution(&record).unwrap();
    restarted
        .flush_required()
        .await
        .expect("a protocol-1 server accepts the compatibility carrier");

    let uploaded = accepted.lock().unwrap().join("\n");
    assert!(uploaded.contains(r#""record_type":"usage""#));
    assert!(!uploaded.contains(r#""record_type":"workspace_execution""#));
    assert!(!uploaded.contains(r#""record_type":"message""#));
    assert!(uploaded.contains(r#"\"type\":\"workspace_execution\""#));
    assert_eq!(accepted.lock().unwrap().len(), 1, "retry is idempotent");
    assert!(store.ready_records(&session_id, 10).unwrap().is_empty());
}

#[tokio::test]
async fn malformed_success_ack_keeps_transcript_outbox_evidence() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let _request = read_mock_http_request(&mut socket).await.unwrap();
                let body = "{}";
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            });
        }
    });
    let store = unique_test_sync_store();
    let session_id = format!("malformed-ack-{}", std::process::id());
    let sink = RemoteSessionSink::with_store(
        SyncConfig {
            base_url: format!("http://{addr}"),
            api_key: "test-key".into(),
            machine_id: None,
            cwd_digest: None,
        },
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    sink.set_pipefs_sync_required(true);
    sink.push("message", r#"{"role":"assistant","text":"durable"}"#);

    let error = sink.flush_required().await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("typed transcript acknowledgement")
    );
    assert_eq!(store.ready_records(&session_id, 10).unwrap().len(), 1);
    assert_eq!(store.status(Some(&session_id)).unwrap().server_cursor, 0);
}

#[tokio::test]
async fn causal_admission_preflushes_a_backlog_larger_than_one_commit_batch() {
    let (base_url, _accepted) =
        start_poison_rejecting_server(Arc::new(AtomicBool::new(false))).await;
    let store = unique_test_sync_store();
    let session_id = format!("causal-preflight-{}", std::process::id());
    let sink = RemoteSessionSink::with_store(
        SyncConfig {
            base_url,
            api_key: "test-key".into(),
            machine_id: None,
            cwd_digest: None,
        },
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    sink.set_pipefs_sync_required(true);
    for index in 0..513 {
        sink.push("usage", &serde_json::json!({"index": index}).to_string());
    }
    assert_eq!(store.status(Some(&session_id)).unwrap().queue_rows, 513);

    sink.prepare_causal_pipefs_mutation().await.unwrap();
    assert_eq!(store.status(Some(&session_id)).unwrap().queue_rows, 0);
    assert!(
        sink.causal_pipefs_transcript_batch()
            .unwrap()
            .records
            .is_empty()
    );
}

#[tokio::test]
async fn permanent_batch_rejection_quarantines_only_the_poison_record() {
    let reject = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let (base_url, accepted) = start_poison_rejecting_server(reject.clone()).await;
    let config = SyncConfig {
        base_url,
        api_key: "test-key".to_string(),
        machine_id: None,
        cwd_digest: None,
    };
    let store = unique_test_sync_store();
    let session_id = format!("poison-isolation-{}", std::process::id());
    let sink = RemoteSessionSink::with_store(
        config.clone(),
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );

    sink.push("message", r#"{"role":"user","n":1}"#);
    sink.push("poison", r#"{"kind":"poison"}"#);
    sink.push("message", r#"{"role":"assistant","n":2}"#);
    sink.flush()
        .await
        .expect("flush completes despite the poison record");

    // Both good records made it around the poison one…
    // Payloads travel JSON-escaped inside the append body.
    let uploaded = accepted.lock().unwrap().join("\n");
    assert!(uploaded.contains(r#"\"n\":1"#), "first message uploaded");
    assert!(uploaded.contains(r#"\"n\":2"#), "second message uploaded");
    // …and exactly the offender is quarantined, not the whole batch.
    let status = store.status(Some(&session_id)).unwrap();
    assert_eq!(status.quarantined_records, 1);
    assert!(store.ready_records(&session_id, 10).unwrap().is_empty());

    // A later process retries the quarantined record once; with the
    // server fixed (accepting the type), it drains clean.
    reject.store(false, Ordering::SeqCst);
    let second_process = RemoteSessionSink::with_store(
        config,
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    second_process.flush().await.expect("requeue flush");
    let status = store.status(Some(&session_id)).unwrap();
    assert_eq!(status.quarantined_records, 0);
    let uploaded = accepted.lock().unwrap().join("\n");
    assert!(
        uploaded.contains("poison"),
        "healed server received the record"
    );
}

#[tokio::test]
async fn unreachable_endpoint_trips_breaker_and_later_flushes_skip_silently() {
    // Port 9 (discard) refuses connections instantly: the first flush
    // pays one fast connect failure and trips the shared breaker; every
    // later flush must return Ok without touching the network, leaving
    // the records queued. This is what keeps a dead portal from stacking
    // multi-second timeouts onto startup, turn ends, and exits.
    let sink = RemoteSessionSink::new_for_test(
        SyncConfig {
            base_url: "http://127.0.0.1:9".to_string(),
            api_key: "test-key".to_string(),
            machine_id: None,
            cwd_digest: None,
        },
        format!("breaker-sink-{}", std::process::id()),
    );
    sink.push("message", r#"{"role":"user"}"#);

    let first = sink.flush().await;
    assert!(first.is_err(), "the first attempt surfaces one error");

    let started = std::time::Instant::now();
    let second = sink.flush().await;
    assert!(
        second.is_ok(),
        "breaker-open flushes skip silently: {second:?}"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_millis(500),
        "no network wait while cooling down: {:?}",
        started.elapsed()
    );
    assert!(
        sink.store
            .breaker_open_until()
            .unwrap()
            .is_some_and(|until| until > unix_now()),
        "connect failure tripped the breaker"
    );
}

#[tokio::test]
async fn pipefs_sync_pin_keeps_a_live_sink_flushing_after_global_sync_off() {
    let reject = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (base_url, accepted) = start_poison_rejecting_server(reject).await;
    let store = unique_test_sync_store();
    store
        .set_mode(crate::sync_store::SyncMode::Off)
        .expect("turning global sync off");
    let session_id = format!("pipefs-sync-pin-{}", std::process::id());
    let sink = RemoteSessionSink::with_store(
        SyncConfig {
            base_url,
            api_key: "test-key".to_string(),
            machine_id: None,
            cwd_digest: None,
        },
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );

    // Without the pin, an ordinary sink obeys the persisted mode.
    sink.push("message", r#"{"role":"user","n":0}"#);
    assert!(store.ready_records(&session_id, 10).unwrap().is_empty());

    sink.set_pipefs_sync_required(true);
    sink.push("message", r#"{"role":"user","n":1}"#);
    sink.flush()
        .await
        .expect("PipeFS pin must keep the transcript transport alive");
    assert!(
        accepted.lock().unwrap().join("\n").contains(r#"\"n\":1"#),
        "the pinned record reached the server"
    );
    assert_eq!(
        store.effective_mode().unwrap(),
        crate::sync_store::SyncMode::Off
    );

    // Removing the pin restores normal per-process behavior without changing
    // the persisted preference.
    sink.set_pipefs_sync_required(false);
    sink.push("message", r#"{"role":"user","n":2}"#);
    assert!(store.ready_records(&session_id, 10).unwrap().is_empty());
}

#[tokio::test]
async fn lease_acquisition_retries_a_timeout_with_the_same_token() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let observed_tokens = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let attempts = Arc::new(AtomicUsize::new(0));
    let server_tokens = observed_tokens.clone();
    let server_attempts = attempts.clone();
    let server = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let tokens = server_tokens.clone();
            let attempts = server_attempts.clone();
            tokio::spawn(async move {
                let request = read_mock_http_request(&mut socket).await.unwrap();
                let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                let body_offset = request
                    .windows(4)
                    .position(|window| window == b"\r\n\r\n")
                    .unwrap()
                    + 4;
                let body: serde_json::Value =
                    serde_json::from_slice(&request[body_offset..]).unwrap();
                let token = body["lease_token"].as_str().unwrap().to_string();
                tokens.lock().unwrap().push(token.clone());
                if attempt == 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
                }
                let response_body = serde_json::json!({
                    "lease_token": token,
                    "generation": 1,
                    "expires_at_unix": 4_000_000_000_u64,
                })
                .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });

    let session_id = format!("lease-retry-{}", std::process::id());
    let sink = RemoteSessionSink::new_for_test(
        SyncConfig {
            base_url: format!("http://{addr}"),
            api_key: "test-key".to_string(),
            machine_id: Some("test-machine".to_string()),
            cwd_digest: None,
        },
        session_id.clone(),
    );
    sink.acquire_lease_with_policy(
        true,
        std::time::Duration::from_millis(30),
        std::time::Duration::from_millis(5),
        2,
    )
    .await
    .unwrap();

    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    let tokens = observed_tokens.lock().unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0], tokens[1]);
    assert_eq!(sink.lease_token().as_deref(), Some(tokens[1].as_str()));
    server.abort();
}

#[tokio::test]
async fn synchronous_lease_confirmation_fails_closed_and_marks_takeover() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = read_mock_http_request(&mut socket).await.unwrap();
        assert!(
            String::from_utf8_lossy(&request).contains("x-hi-lease-token: test-token"),
            "confirmation must authenticate with the writer lease"
        );
        let body = r#"{"error":"lease_lost: replaced"}"#;
        let response = format!(
            "HTTP/1.1 409 Conflict\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    let store = unique_test_sync_store();
    let session_id = format!("lease-confirm-{}", std::process::id());
    store
        .store_lease(&session_id, "test-token", 3, "test-owner", 4_000_000_000)
        .unwrap();
    let sink = RemoteSessionSink::with_store(
        SyncConfig {
            base_url: format!("http://{addr}"),
            api_key: "test-key".to_string(),
            machine_id: Some("test-machine".to_string()),
            cwd_digest: None,
        },
        session_id,
        None,
        remote_session_http_client(),
        store,
    );

    let mut lease_status = sink.subscribe_writer_lease_status();
    let error = sink.confirm_writer_lease().await.unwrap_err();
    assert!(error.to_string().contains("409"));
    assert!(sink.writer_lease_is_lost());
    lease_status.changed().await.unwrap();
    assert_eq!(*lease_status.borrow(), hi_pipefs::PipeFsLeaseStatus::Lost);
    server.await.unwrap();
}

#[tokio::test]
async fn failed_lease_confirmation_publishes_uncertainty() {
    let store = unique_test_sync_store();
    let session_id = format!("lease-uncertain-{}", std::process::id());
    store
        .store_lease(&session_id, "test-token", 3, "test-owner", 4_000_000_000)
        .unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let sink = RemoteSessionSink::with_store(
        SyncConfig {
            base_url: format!("http://{addr}"),
            api_key: "test-key".into(),
            machine_id: None,
            cwd_digest: None,
        },
        session_id,
        None,
        remote_session_http_client(),
        store,
    );
    let mut lease_status = sink.subscribe_writer_lease_status();

    sink.confirm_writer_lease().await.unwrap_err();
    lease_status.changed().await.unwrap();
    assert_eq!(
        *lease_status.borrow(),
        hi_pipefs::PipeFsLeaseStatus::Uncertain
    );
    assert!(!sink.writer_lease_is_lost());
}

#[tokio::test]
async fn synchronous_lease_confirmation_renews_local_freshness_evidence() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let _ = read_mock_http_request(&mut socket).await.unwrap();
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}",
            )
            .await
            .unwrap();
    });

    let store = unique_test_sync_store();
    let session_id = format!("lease-renew-{}", std::process::id());
    store
        .store_lease(&session_id, "test-token", 3, "test-owner", 1)
        .unwrap();
    let sink = RemoteSessionSink::with_store(
        SyncConfig {
            base_url: format!("http://{addr}"),
            api_key: "test-key".to_string(),
            machine_id: Some("test-machine".to_string()),
            cwd_digest: None,
        },
        session_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );

    sink.confirm_writer_lease().await.unwrap();
    assert!(store.status(Some(&session_id)).unwrap().lease_expiry_unix > unix_now().max(0) as u64);
    server.await.unwrap();
}

#[test]
fn attach_stream_deduplicates_replayed_cursor_events() {
    let mut cursor = 41;
    assert_eq!(
        accept_streamed_event(
            cursor,
            StreamedEvent {
                event_json: "duplicate".into(),
                event_seq: 41
            }
        ),
        None
    );
    let (next, json) = accept_streamed_event(
        cursor,
        StreamedEvent {
            event_json: "next".into(),
            event_seq: 42,
        },
    )
    .unwrap();
    assert_eq!(json, "next");
    cursor = next;
    assert_eq!(cursor, 42);
}

#[test]
fn attach_retry_backoff_is_bounded() {
    assert_eq!(attach_retry_delay(0), std::time::Duration::from_millis(500));
    assert_eq!(attach_retry_delay(3), std::time::Duration::from_secs(4));
    assert_eq!(attach_retry_delay(20), std::time::Duration::from_secs(8));
}

#[test]
fn classify_session_join_prefers_host_alive() {
    let hosted = serde_json::json!({
        "host_alive": true,
        "accepts_input": true,
        "status": "active",
        "lease_expires_at_unix": 1
    });
    assert_eq!(classify_session_join(&hosted), SessionJoinMode::SteerHost);

    let portable = serde_json::json!({
        "host_alive": false,
        "accepts_input": false,
        "status": "ended",
        "lease_expires_at_unix": 0
    });
    assert_eq!(
        classify_session_join(&portable),
        SessionJoinMode::ContinueHere
    );
}

#[tokio::test]
async fn stranded_sessions_from_earlier_runs_are_drained_and_ended() {
    let reject = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (base_url, accepted, posts) = start_recording_server(reject).await;
    let config = SyncConfig {
        base_url,
        api_key: "test-key".to_string(),
        machine_id: None,
        cwd_digest: None,
    };
    let store = unique_test_sync_store();
    let pid = std::process::id();

    // An earlier process queued records for its session and died before its
    // flush (breaker open, interrupted): nothing of its own will run again.
    let stranded_id = format!("stranded-{pid}");
    let earlier = RemoteSessionSink::with_store(
        config.clone(),
        stranded_id.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    earlier.push("message", r#"{"role":"user","n":"stranded"}"#);
    drop(earlier);

    // The next run, for a different session, sweeps it up…
    let current = RemoteSessionSink::with_store(
        config,
        format!("current-{pid}"),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    assert_eq!(current.drain_stranded_sessions().await, 1);

    // …uploads its records, and leaves it ended rather than "active" under a
    // lease the drain just took.
    let uploaded = accepted.lock().unwrap().join("\n");
    assert!(
        uploaded.contains(r#"\"n\":\"stranded\""#),
        "stranded record uploaded"
    );
    assert!(store.ready_records(&stranded_id, 10).unwrap().is_empty());
    let post_lines = posts.lock().unwrap().clone();
    assert!(
        post_lines
            .iter()
            .any(|line| line.contains(&format!("/hi/sessions/{stranded_id}/end"))),
        "stranded session ended: {post_lines:?}"
    );
    // The current session's own records are untouched by the sweep — its
    // own flush owns them.
    assert!(
        !post_lines
            .iter()
            .any(|line| line.contains(&format!("current-{pid}/records"))),
        "{post_lines:?}"
    );
    // A second sweep finds nothing.
    assert_eq!(current.drain_stranded_sessions().await, 0);

    // With the breaker open, flush skips the network and reports Ok — the
    // sweep must not mistake that for delivery: no /end, nothing counted,
    // records still queued for a later run.
    let stranded_again = format!("stranded-breaker-{pid}");
    let later = RemoteSessionSink::with_store(
        SyncConfig {
            base_url: "http://127.0.0.1:9".to_string(),
            api_key: "test-key".to_string(),
            machine_id: None,
            cwd_digest: None,
        },
        stranded_again.clone(),
        None,
        remote_session_http_client(),
        store.clone(),
    );
    later.push("message", r#"{"role":"user","n":"queued"}"#);
    drop(later);
    store.trip_breaker(unix_now()).unwrap();
    assert_eq!(current.drain_stranded_sessions().await, 0);
    assert_eq!(store.ready_records(&stranded_again, 10).unwrap().len(), 1);
    let post_lines = posts.lock().unwrap().clone();
    assert!(
        !post_lines
            .iter()
            .any(|line| line.contains(&format!("{stranded_again}/end"))),
        "{post_lines:?}"
    );
    store.reset_breaker().unwrap();
}
