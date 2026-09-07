use super::*;
use crate::openai::OpenAiProvider;
use crate::provider::Provider;
use crate::token::StaticToken;
use crate::types::{ChatRequest, Content, Message, RequestProfile};

fn request(model: &str) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        request_id: None,
        retry_attempt: 0,
        execution: Default::default(),
        user_turn: false,
        canonical_objective: None,
        messages: vec![Message::user("hello")].into(),
        tools: Vec::new().into(),
        tool_envelope: None,
        max_tokens: 32,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: RequestProfile::default(),
    }
}

fn raw_chat_request(body: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn wait_for_request_tickets(
    server: &ScriptedOpenAiServer,
    count: usize,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if server.state.next_sequence.load(Ordering::Acquire) >= count {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    server.state.next_sequence.load(Ordering::Acquire) >= count
}

fn wait_for_order_waiter(
    server: &ScriptedOpenAiServer,
    sequence: usize,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    let mut order = server.state.request_order.values.lock().unwrap();
    loop {
        if order.waiting.contains(&sequence) {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        let (next, result) = server
            .state
            .request_order
            .changed
            .wait_timeout(order, deadline.saturating_duration_since(now))
            .unwrap();
        order = next;
        if result.timed_out() {
            return order.waiting.contains(&sequence);
        }
    }
}

#[tokio::test]
async fn model_discovery_is_repeatable_and_does_not_consume_chat_steps() {
    let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::expecting(
        RequestMatcher::any().json_eq("/model", json!("test-model")),
        ScriptedResponse::text("hello back"),
    )]) else {
        return;
    };
    let provider = OpenAiProvider::with_token_source(
        server.v1_url(),
        Arc::new(StaticToken("test".to_string())),
    );

    let first = provider.list_models().await.unwrap();
    let second = provider.list_models().await.unwrap();
    assert_eq!(first[0].id, "test-model");
    assert_eq!(second[0].id, "test-model");

    let mut sink = |_| {};
    let completion = provider
        .stream(request("test-model"), &mut sink)
        .await
        .unwrap();
    assert!(
        matches!(completion.content.first(), Some(Content::Text(text)) if text == "hello back")
    );
    assert_eq!(server.requests().len(), 3);
    server.assert_clean().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn idle_speculative_connection_does_not_block_real_requests() {
    let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(ScriptedResponse::text(
        "real request completed",
    ))]) else {
        return;
    };
    let idle = TcpStream::connect(server.url().trim_start_matches("http://")).unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();
    let response = tokio::time::timeout(
        Duration::from_secs(1),
        client
            .post(server.chat_url())
            .json(&json!({"model": "test-model", "messages": []}))
            .send(),
    )
    .await
    .expect("real request was head-of-line blocked")
    .unwrap();
    assert!(response.status().is_success());

    drop(idle);
    server.assert_clean().unwrap();
}

#[test]
fn concurrent_connections_consume_chat_steps_in_request_arrival_order() {
    let Some(server) = ScriptedOpenAiServer::new(vec![
        ChatStep::expecting(
            RequestMatcher::any().json_eq("/model", json!("first")),
            ScriptedResponse::text("first response"),
        ),
        ChatStep::expecting(
            RequestMatcher::any().json_eq("/model", json!("second")),
            ScriptedResponse::text("second response"),
        ),
    ]) else {
        return;
    };
    let address = server.url().trim_start_matches("http://");

    let first_body = r#"{"model":"first","messages":[]}"#;
    let first_request = raw_chat_request(first_body);
    let first_body_start = first_request.len() - first_body.len();
    let mut first = TcpStream::connect(address).unwrap();
    first
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    first
        .write_all(&first_request.as_bytes()[..first_body_start])
        .unwrap();
    assert!(wait_for_request_tickets(&server, 1, Duration::from_secs(1)));

    let second_body = r#"{"model":"second","messages":[]}"#;
    let mut second = TcpStream::connect(address).unwrap();
    second
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    second
        .write_all(raw_chat_request(second_body).as_bytes())
        .unwrap();
    second.shutdown(Shutdown::Write).unwrap();
    assert!(wait_for_request_tickets(&server, 2, Duration::from_secs(1)));

    // The complete second request must wait for the earlier request body,
    // rather than consuming the first scripted step in its worker thread.
    assert!(wait_for_order_waiter(&server, 1, Duration::from_secs(1)));
    assert!(server.requests().is_empty());

    first
        .write_all(&first_request.as_bytes()[first_body_start..])
        .unwrap();
    first.shutdown(Shutdown::Write).unwrap();
    let mut first_response = String::new();
    first.read_to_string(&mut first_response).unwrap();
    let mut second_response = String::new();
    second.read_to_string(&mut second_response).unwrap();

    assert!(
        first_response.contains("first response"),
        "{first_response}"
    );
    assert!(
        second_response.contains("second response"),
        "{second_response}"
    );
    let requests = server.requests();
    assert_eq!(
        requests
            .iter()
            .map(|request| request.sequence)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(requests[0].json.as_ref().unwrap()["model"], "first");
    assert_eq!(requests[1].json.as_ref().unwrap()["model"], "second");
    server.assert_clean().unwrap();
}

#[test]
fn drip_fed_request_hits_absolute_read_deadline() {
    let Some(server) = ScriptedOpenAiServer::new(Vec::new()) else {
        return;
    };
    let mut stream = TcpStream::connect(server.url().trim_start_matches("http://")).unwrap();
    stream
        .set_write_timeout(Some(Duration::from_millis(250)))
        .unwrap();
    let keep_writing = Arc::new(AtomicBool::new(true));
    let writer_flag = Arc::clone(&keep_writing);
    let writer = std::thread::spawn(move || {
        while writer_flag.load(Ordering::Acquire) {
            if stream.write_all(b"x").is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    });

    let started = Instant::now();
    let observation_deadline = started + REQUEST_READ_DEADLINE + Duration::from_secs(1);
    let failure = loop {
        if let Some(failure) = server.failures().into_iter().next() {
            break Some(failure);
        }
        if Instant::now() >= observation_deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    keep_writing.store(false, Ordering::Release);
    writer.join().unwrap();

    let failure = failure.expect("drip-fed request outlived its absolute read deadline");
    assert_eq!(failure.kind, ScriptedFailureKind::RequestRead);
    assert_eq!(failure.request_sequence, Some(0));
    assert!(
        started.elapsed() < REQUEST_READ_DEADLINE + Duration::from_secs(1),
        "request failure exceeded its absolute deadline: {:?}",
        started.elapsed()
    );
}

#[test]
fn shutdown_interrupts_partial_request_promptly() {
    let Some(server) = ScriptedOpenAiServer::new(Vec::new()) else {
        return;
    };
    let mut stream = TcpStream::connect(server.url().trim_start_matches("http://")).unwrap();
    stream.write_all(b"POST /v1/chat").unwrap();
    assert!(wait_for_request_tickets(&server, 1, Duration::from_secs(1)));

    let started = Instant::now();
    server.shutdown();
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "shutdown waited too long for a partial request: {:?}",
        started.elapsed()
    );
    assert!(server.failures().is_empty());
}

#[test]
fn request_matcher_reports_all_differences() {
    let request = RecordedRequest {
        sequence: 7,
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        headers: BTreeMap::from([("authorization".to_string(), "Bearer real".to_string())]),
        body: r#"{"model":"other","messages":[]}"#.to_string(),
        json: Some(json!({"model": "other", "messages": []})),
    };
    let mismatches = RequestMatcher::any()
        .method("GET")
        .header("authorization", "Bearer expected")
        .body_contains("needle")
        .body_excludes("messages")
        .json_eq("/model", json!("expected"))
        .json_present("/stream")
        .json_absent("/messages")
        .mismatches(&request);
    assert_eq!(mismatches.len(), 7, "{mismatches:#?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn named_gate_holds_response_without_blocking_observation() {
    let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(
        ScriptedResponse::text("released").wait_for_gate("model"),
    )]) else {
        return;
    };
    let url = server.chat_url();
    let request = tokio::spawn(async move {
        reqwest::Client::new()
            .post(url)
            .json(&json!({"model": "test-model", "messages": []}))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    });
    assert!(server.wait_for_gate_waiter("model", Duration::from_secs(2)));
    assert!(!request.is_finished());
    assert!(server.release_gate("model"));
    assert!(request.await.unwrap().contains("released"));
    server.assert_clean().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hold_open_gate_waits_after_headers_and_body() {
    let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(
        ScriptedResponse::raw_sse("").hold_open_until("close"),
    )]) else {
        return;
    };
    let response = reqwest::Client::new()
        .post(server.chat_url())
        .json(&json!({"model": "test-model", "messages": []}))
        .send()
        .await
        .unwrap();
    assert!(server.wait_for_gate_waiter("close", Duration::from_secs(2)));
    let body = tokio::spawn(async move { response.text().await.unwrap() });
    tokio::task::yield_now().await;
    assert!(!body.is_finished());
    assert!(server.release_gate("close"));
    assert_eq!(body.await.unwrap(), "");
    server.assert_clean().unwrap();
}

#[tokio::test]
async fn orderly_eof_and_tcp_reset_are_distinct_scripted_failures() {
    let Some(server) = ScriptedOpenAiServer::new(vec![
        ChatStep::new(ScriptedResponse::eof()),
        ChatStep::new(ScriptedResponse::reset()),
    ]) else {
        return;
    };
    let client = reqwest::Client::new();
    let eof = client
        .post(server.chat_url())
        .json(&json!({"model": "test-model"}))
        .send()
        .await;
    assert!(eof.is_err(), "clean EOF before HTTP headers must fail");
    let reset = client
        .post(server.chat_url())
        .json(&json!({"model": "test-model"}))
        .send()
        .await;
    assert!(reset.is_err(), "TCP reset before HTTP headers must fail");
    server.assert_clean().unwrap();
}

#[tokio::test]
async fn mismatch_and_unexpected_requests_are_durable_failures() {
    let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::expecting(
        RequestMatcher::any().json_eq("/model", json!("expected")),
        ScriptedResponse::text("bounded"),
    )]) else {
        return;
    };
    let client = reqwest::Client::new();
    let first = client
        .post(server.chat_url())
        .json(&json!({"model": "wrong"}))
        .send()
        .await
        .unwrap();
    assert!(first.status().is_success());
    let second = client
        .post(server.chat_url())
        .json(&json!({"model": "extra"}))
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 500);
    assert_eq!(
        server
            .failures()
            .iter()
            .map(|failure| &failure.kind)
            .collect::<Vec<_>>(),
        vec![
            &ScriptedFailureKind::MismatchedChatRequest,
            &ScriptedFailureKind::UnexpectedChatRequest,
        ]
    );
    assert!(server.assert_clean().is_err());
}

#[test]
fn unconsumed_required_steps_fail_but_optional_steps_do_not() {
    let Some(server) = ScriptedOpenAiServer::new(vec![
        ChatStep::new(ScriptedResponse::text("optional")).optional(),
        ChatStep::new(ScriptedResponse::text("required")).named("must-run"),
    ]) else {
        return;
    };
    let error = server.assert_clean().unwrap_err();
    assert_eq!(error.unconsumed_required_steps.len(), 1);
    assert_eq!(
        error.unconsumed_required_steps[0].label.as_deref(),
        Some("must-run")
    );
}

#[test]
fn response_builders_cover_text_tools_fragmentation_and_transport_faults() {
    let text = ScriptedResponse::text("hello").fragmented(3, Duration::from_millis(1));
    assert!(text.chunks.len() > 1);
    assert_eq!(text.body_len(), super::super::sse_text("hello").len());

    let tool = ScriptedResponse::tool_call(ScriptedToolCall::new(
        "call-1",
        "bash",
        json!({"command": "true"}),
    ));
    let body = String::from_utf8(
        tool.chunks
            .iter()
            .flat_map(|chunk| chunk.bytes.iter().copied())
            .collect(),
    )
    .unwrap();
    assert!(body.contains("tool_calls"));
    assert!(body.contains("call-1"));
    assert!(body.contains(r#"{\"command\":\"true\"}"#));

    assert!(!ScriptedResponse::eof().send_head);
    assert_eq!(ScriptedResponse::reset().terminal, ResponseTerminal::Reset);
    assert_eq!(
        ScriptedResponse::raw_sse("partial")
            .finish_with_eof()
            .terminal,
        ResponseTerminal::Eof
    );
}
