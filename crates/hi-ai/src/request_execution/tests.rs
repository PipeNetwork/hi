use super::*;
use crate::test_support::{ChatStep, ScriptedOpenAiServer, ScriptedResponse};
use crate::{Backend, ChatRequest, FallbackProvider, OpenAiProvider, Provider};
use serde_json::json;

fn attempt_event(physical_attempt: u32, state: ProviderAttemptState) -> StreamEvent {
    StreamEvent::ProviderAttempt(Box::new(ProviderAttemptEvent {
        operation_id: "operation".into(),
        request_id: "request".into(),
        physical_attempt,
        provider: "test".into(),
        route: "route".into(),
        model: "model".into(),
        state,
    }))
}

fn started(physical_attempt: u32) -> StreamEvent {
    attempt_event(
        physical_attempt,
        ProviderAttemptState::Started { replay_attempt: 0 },
    )
}

fn request(execution: Arc<RequestExecution>) -> ChatRequest {
    ChatRequest {
        model: "selected-model".into(),
        request_id: Some("test_operation".into()),
        retry_attempt: 0,
        execution,
        user_turn: true,
        canonical_objective: None,
        messages: vec![].into(),
        tools: vec![].into(),
        tool_envelope: None,
        max_tokens: 32,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: Default::default(),
    }
}

#[tokio::test]
async fn physical_sends_share_allowance_and_payload_identity() {
    let Some(server) = ScriptedOpenAiServer::new(
        (0..4)
            .map(|_| ChatStep::new(ScriptedResponse::text("ok")))
            .collect(),
    ) else {
        return;
    };
    let execution = RequestExecution::default();
    let client = crate::agent_http_client();
    let mut events = Vec::new();
    let mut sink = |event| {
        if let StreamEvent::ProviderAttempt(event) = event {
            events.push(*event);
        }
    };
    for model in ["first", "first", "second", "first"] {
        let response = execution
            .dispatch(
                client
                    .post(format!("{}/chat/completions", server.url()))
                    .json(&json!({"model":model})),
                "openai",
                model,
                &mut sink,
            )
            .await
            .unwrap();
        let _ = response.bytes().await.unwrap();
    }
    let error = execution
        .dispatch(
            client
                .post(format!("{}/chat/completions", server.url()))
                .json(&json!({"model":"first"})),
            "openai",
            "first",
            &mut sink,
        )
        .await
        .unwrap_err();
    assert_eq!(error.code.as_deref(), Some("request_attempts_exhausted"));
    assert_eq!(error.retryable, Some(false));
    let evidence = error.request_failure.as_ref().unwrap();
    assert_eq!(evidence.reason, RequestFailureReason::AttemptsExhausted);
    assert_eq!((evidence.attempts, evidence.attempt_limit), (4, 4));
    assert_eq!(evidence.recent_attempts.len(), 4);
    assert!(
        evidence
            .recent_attempts
            .iter()
            .all(|attempt| attempt.http_status == Some(200))
    );
    assert_eq!(server.inspection().requests.len(), 4);
    let started: Vec<_> = events
        .iter()
        .filter(|event| matches!(event.state, ProviderAttemptState::Started { .. }))
        .collect();
    assert_eq!(
        started
            .iter()
            .map(|e| e.physical_attempt)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    assert_eq!(started[0].request_id, started[1].request_id);
    assert_eq!(started[0].request_id, started[3].request_id);
    assert_ne!(started[0].request_id, started[2].request_id);
    assert_eq!(
        started[3].state,
        ProviderAttemptState::Started { replay_attempt: 2 }
    );
    server.assert_clean().unwrap();
}

#[tokio::test]
async fn repeated_empty_fallback_chains_stop_at_four_physical_sends() {
    let mut servers = Vec::new();
    for _ in 0..3 {
        let Some(server) = ScriptedOpenAiServer::new(
            (0..2)
                .map(|_| ChatStep::new(ScriptedResponse::text("")).optional())
                .collect(),
        ) else {
            return;
        };
        servers.push(server);
    }
    let chain = FallbackProvider::new(
        servers
            .iter()
            .enumerate()
            .map(|(index, server)| Backend {
                provider: Box::new(OpenAiProvider::new(server.url().into(), "test".into())),
                model: format!("configured-{index}"),
                label: format!("openai/configured-{index}"),
                capability_route: format!("route-{index}"),
            })
            .collect(),
    )
    .unwrap();
    let execution = Arc::new(RequestExecution::default());
    assert!(
        chain
            .stream(request(execution.clone()), &mut |_| {})
            .await
            .is_err()
    );
    let error = chain
        .stream(request(execution.clone()), &mut |_| {})
        .await
        .unwrap_err();
    assert_eq!(crate::provider_error_retryable(&error), Some(false));
    let evidence = crate::provider_error_details(&error)
        .unwrap()
        .request_failure
        .as_ref()
        .unwrap();
    assert_eq!(evidence.recent_attempts.len(), 4);
    assert!(evidence.recent_attempts.iter().all(|attempt| {
        attempt.failure_kind.as_deref() == Some("empty_completion")
            && attempt.http_status == Some(200)
    }));
    assert!(
        error
            .to_string()
            .contains("4/4 sends; last attempt: empty_completion")
    );
    assert_eq!(execution.attempts(), 4);
    assert_eq!(
        servers
            .iter()
            .map(|server| server.inspection().requests.len())
            .sum::<usize>(),
        4
    );
    assert!(
        servers[0].inspection().requests[0]
            .body
            .contains("selected-model")
    );
    for server in servers {
        server.assert_clean().unwrap();
    }
}

#[test]
fn request_failure_evidence_is_bounded_private_and_attributed_to_its_own_attempt() {
    let execution = RequestExecution::default();
    {
        let mut state = execution.state.lock().unwrap();
        // Composite fanout can exceed the ordinary limit; diagnostic retention
        // remains fixed even when a configured operation has many children.
        for physical_attempt in 1..=12 {
            let StreamEvent::ProviderAttempt(event) = started(physical_attempt) else {
                unreachable!()
            };
            diagnostics::record_dispatch(&mut state.recent_attempts, &event);
        }
        state.attempts = 12;
    }
    let provider_error: anyhow::Error = ProviderError::new(
        ProviderErrorKind::MalformedStream,
        "secret provider message containing submitted source and token",
    )
    .with_api_contract(Some("secret_provider_code".into()), None, None)
    .into();
    execution.record_provider_failure(Some(9), &provider_error);
    execution.record_attempt_result(12, Some(503), None);
    let error = execution.failure(RequestFailureReason::DeadlineExceeded);
    let evidence = error.request_failure.as_ref().unwrap();
    assert_eq!(evidence.recent_attempts.len(), 8);
    assert_eq!(evidence.recent_attempts[0].physical_attempt, 5);
    assert_eq!(evidence.recent_attempts[4].physical_attempt, 9);
    assert_eq!(
        evidence.recent_attempts[4].failure_kind.as_deref(),
        Some("malformed_stream")
    );
    assert_eq!(evidence.recent_attempts[7].http_status, Some(503));
    assert!(evidence.recent_attempts[7].failure_kind.is_none());
    let serialized = serde_json::to_string(evidence).unwrap();
    assert!(!serialized.contains("secret"));
    assert!(!error.to_string().contains("secret"));
    assert_eq!(error.retryable, Some(false));
    assert_eq!(error.code.as_deref(), Some("request_deadline"));
}

#[tokio::test]
async fn explicit_nonretryable_shape_error_does_not_enter_compatibility_ladder() {
    let Some(server) =
        ScriptedOpenAiServer::new(vec![ChatStep::new(ScriptedResponse::http_error(
            400,
            r#"{"error":{"message":"unsupported parameter stream_options","retryable":false}}"#,
        ))])
    else {
        return;
    };
    let provider = OpenAiProvider::new(server.url().into(), "test".into());
    let execution = Arc::new(RequestExecution::default());
    let error = provider
        .stream(request(execution.clone()), &mut |_| {})
        .await
        .unwrap_err();
    assert_eq!(crate::provider_error_retryable(&error), Some(false));
    assert_eq!(execution.attempts(), 1);
    server.assert_clean().unwrap();
}

#[tokio::test]
async fn reservation_protects_aggregator_without_replenishing_total() {
    let execution = Arc::new(RequestExecution::new(RequestExecutionPolicy {
        max_attempts: 1,
        ..Default::default()
    }));
    let reservation = execution.reserve(1);
    assert!(execution.ensure_available().is_err());
    drop(reservation);
    assert!(execution.ensure_available().is_ok());
    assert_eq!(execution.attempts(), 0);
}

#[tokio::test]
async fn fanout_initial_dispatches_share_three_recoveries_and_reserved_aggregation() {
    for children in [1_u32, 3] {
        let planned = children + 1;
        let total = planned + 3;
        let Some(server) = ScriptedOpenAiServer::new(
            (0..total)
                .map(|_| ChatStep::new(ScriptedResponse::text("ok")))
                .collect(),
        ) else {
            return;
        };
        let execution = Arc::new(RequestExecution::default());
        execution.plan_fanout(planned).unwrap();
        let parent = execution.clone();
        assert_eq!(parent.attempt_limit(), total);
        let reservation = execution.reserve(1);
        let client = crate::inference_http_client_for_socket(None);
        let send = || {
            client
                .post(format!("{}/chat/completions", server.url()))
                .json(&json!({"model":"child"}))
        };
        for _ in 0..children + 3 {
            let response = execution
                .dispatch(send(), "openai", "child", &mut |_| {})
                .await
                .unwrap();
            response.bytes().await.unwrap();
        }
        assert!(
            execution
                .dispatch(send(), "openai", "child", &mut |_| {})
                .await
                .is_err()
        );
        assert_eq!(parent.attempts(), total - 1);
        execution.plan_fanout(planned).unwrap();
        assert_eq!(execution.remaining_attempts(), 0);
        assert!(execution.plan_fanout(planned + 1).is_err());
        drop(reservation);
        let response = execution
            .dispatch(send(), "openai", "aggregator", &mut |_| {})
            .await
            .unwrap();
        response.bytes().await.unwrap();
        execution.plan_fanout(planned).unwrap();
        assert!(execution.ensure_available().is_err());
        assert_eq!(parent.attempts(), total);
        assert_eq!(server.inspection().requests.len(), total as usize);
        assert_eq!(execution.fresh_operation().attempt_limit(), 4);
        server.assert_clean().unwrap();
    }
}

#[tokio::test(start_paused = true)]
async fn declaring_fanout_does_not_reset_prior_sends_or_backoff() {
    let execution = RequestExecution::default();
    execution.state.lock().unwrap().attempts = 2;
    execution.backoff(Duration::from_secs(60)).await.unwrap();
    execution.plan_fanout(2).unwrap();
    assert_eq!(execution.attempts(), 2);
    assert_eq!(execution.remaining_attempts(), 3);
    assert!(execution.backoff(Duration::from_millis(1)).await.is_err());
}

#[tokio::test(start_paused = true)]
async fn backoff_allowance_is_shared_and_never_refilled() {
    let execution = RequestExecution::default();
    execution.backoff(Duration::from_secs(40)).await.unwrap();
    assert!(execution.backoff(Duration::from_secs(21)).await.is_err());
    execution.backoff(Duration::from_secs(20)).await.unwrap();
    assert!(execution.backoff(Duration::from_millis(1)).await.is_err());
    assert_eq!(execution.backoff_spent(), Duration::from_secs(60));
}

#[tokio::test(start_paused = true)]
async fn silent_headers_heartbeats_and_name_only_tool_deltas_expire() {
    for heartbeat in [
        None,
        Some(StreamEvent::Status("heartbeat".into())),
        Some(StreamEvent::ToolCallDelta {
            index: 0,
            id_delta: None,
            name_delta: Some("write".into()),
            arguments_delta: String::new(),
        }),
    ] {
        let execution = RequestExecution::default();
        let progress = execution.progress();
        progress.observe(&started(1));
        let start = tokio::time::Instant::now();
        let error = progress
            .watch(async {
                loop {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    if let Some(event) = &heartbeat {
                        progress.observe(event);
                    }
                }
                #[allow(unreachable_code)]
                Ok::<(), anyhow::Error>(())
            })
            .await
            .unwrap_err();
        assert_eq!(start.elapsed(), Duration::from_secs(300));
        assert_eq!(
            error
                .downcast_ref::<ProviderError>()
                .unwrap()
                .code
                .as_deref(),
            Some("request_stalled")
        );
    }
}

#[tokio::test(start_paused = true)]
async fn decoded_reasoning_keeps_a_productive_long_stream_alive() {
    let execution = RequestExecution::default();
    let progress = execution.progress();
    progress.observe(&started(1));
    progress
        .watch(async {
            for _ in 0..8 {
                tokio::time::sleep(Duration::from_secs(100)).await;
                progress.observe(&StreamEvent::Reasoning("new reasoning".into()));
            }
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test(start_paused = true)]
async fn approval_pause_preserves_prior_silence_and_repeated_statuses_cannot_extend_it() {
    let execution = RequestExecution::default();
    let progress = execution.progress();
    progress.observe(&started(1));
    let attempt = |state| attempt_event(1, state);
    let start = tokio::time::Instant::now();
    let err = progress
        .watch(async {
            tokio::time::sleep(Duration::from_secs(250)).await;
            progress.observe(&attempt(ProviderAttemptState::WaitingForApproval));
            tokio::time::sleep(Duration::from_secs(300)).await;
            progress.observe(&attempt(ProviderAttemptState::WaitingForApproval));
            tokio::time::sleep(Duration::from_secs(300)).await;
            progress.observe(&attempt(ProviderAttemptState::ApprovalCompleted));
            tokio::time::sleep(Duration::from_secs(25)).await;
            progress.observe(&attempt(ProviderAttemptState::ApprovalCompleted));
            progress.observe(&attempt(ProviderAttemptState::Started {
                replay_attempt: 1,
            }));
            std::future::pending::<anyhow::Result<()>>().await
        })
        .await
        .unwrap_err();
    assert_eq!(start.elapsed(), Duration::from_secs(900));
    assert_eq!(
        crate::provider_error_details(&err).unwrap().code.as_deref(),
        Some("request_stalled")
    );
}

#[tokio::test(start_paused = true)]
async fn silence_starts_at_physical_dispatch_and_only_new_attempts_restart_it() {
    let execution = RequestExecution::default();
    let progress = execution.progress();
    let start = tokio::time::Instant::now();
    let error = progress
        .watch(async {
            // Pre-dispatch metadata/credential work is outside the model clock.
            tokio::time::sleep(Duration::from_secs(1000)).await;
            progress.observe(&started(1));
            tokio::time::sleep(Duration::from_secs(250)).await;
            progress.observe(&started(2));
            tokio::time::sleep(Duration::from_secs(250)).await;
            progress.observe(&started(2));
            std::future::pending::<anyhow::Result<()>>().await
        })
        .await
        .unwrap_err();
    assert_eq!(start.elapsed(), Duration::from_secs(1550));
    assert_eq!(
        crate::provider_error_details(&error)
            .unwrap()
            .code
            .as_deref(),
        Some("request_stalled")
    );
}

#[tokio::test(start_paused = true)]
async fn disabled_watchdog_and_fresh_operations_keep_resolved_policy() {
    let execution = RequestExecution::new(RequestExecutionPolicy {
        max_attempts: 1,
        max_backoff: Duration::from_secs(2),
        no_progress_timeout: Duration::ZERO,
    });
    execution.backoff(Duration::from_secs(2)).await.unwrap();
    let fresh = execution.fresh_operation();
    assert_eq!(fresh.backoff_spent(), Duration::ZERO);
    assert_eq!(fresh.policy.max_attempts, 1);
    assert_ne!(fresh.operation_id, execution.operation_id);
    fresh
        .progress()
        .watch(async {
            tokio::time::sleep(Duration::from_secs(86_400)).await;
            Ok(())
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn concrete_openai_header_wait_and_raw_heartbeats_are_bounded() {
    use crate::test_support::ResponseChunk;
    for response in [
        ScriptedResponse::text("late").delayed(Duration::from_secs(1)),
        ScriptedResponse::text("").with_chunks(
            (0..100).map(|_| ResponseChunk::after(Duration::from_millis(10), b": heartbeat\n\n")),
        ),
    ] {
        let Some(server) = ScriptedOpenAiServer::new(vec![ChatStep::new(response)]) else {
            return;
        };
        let execution = Arc::new(RequestExecution::new(RequestExecutionPolicy {
            no_progress_timeout: Duration::from_millis(100),
            ..Default::default()
        }));
        let provider = OpenAiProvider::new(server.url().into(), "test".into());
        let start = tokio::time::Instant::now();
        let error = provider
            .stream(request(execution.clone()), &mut |_| {})
            .await
            .unwrap_err();
        let error = error.downcast_ref::<ProviderError>().unwrap();
        assert_eq!(error.code.as_deref(), Some("request_stalled"));
        assert_eq!(execution.attempts(), 1);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(error.usage.estimated);
    }
}

#[tokio::test]
async fn redirects_are_charged_and_one_shot_dispatch_never_replays_them() {
    let Some(server) = ScriptedOpenAiServer::new(vec![
        ChatStep::new(
            ScriptedResponse::http_error(307, "redirect")
                .with_header("location", "/v2/chat/completions"),
        ),
        ChatStep::new(ScriptedResponse::text("ok")),
        ChatStep::new(
            ScriptedResponse::http_error(307, "redirect")
                .with_header("location", "/v2/chat/completions"),
        ),
    ]) else {
        return;
    };
    let execution = RequestExecution::default();
    let client = crate::inference_http_client_for_socket(None);
    let builder = || {
        client
            .post(format!("{}/chat/completions", server.url()))
            .json(&json!({"model":"test"}))
    };
    let response = execution
        .dispatch(builder(), "test", "test", &mut |_| {})
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let _ = response.bytes().await.unwrap();
    assert_eq!(execution.attempts(), 2);
    let response = execution
        .dispatch_once(builder(), "test", "test", &mut |_| {})
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 307);
    assert_eq!(execution.attempts(), 3);
    server.assert_clean().unwrap();
}
