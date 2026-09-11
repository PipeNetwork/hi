use std::sync::Arc;
use std::time::Duration;

use hi_ai::test_support::{ChatStep, ScriptedOpenAiServer, ScriptedResponse, ScriptedToolCall};
use hi_ai::{OpenAiProvider, ProviderError, ProviderErrorKind};

use super::TurnRetryState;
use crate::tests::common::{NullUi, config};

fn rejected_generation() -> ProviderError {
    ProviderError::new(ProviderErrorKind::ToolProtocol, "generation rejected")
        .with_api_contract(Some("tool_protocol_error".into()), Some(true), None)
        .with_http_status(Some(503))
}

#[test]
fn only_explicit_terminal_rejections_rotate_generation_identity() {
    let mut state = TurnRetryState::default();
    let original = state.request_id();
    state.record_recovery_attempt();
    state.start_protocol_recovery(&rejected_generation().into());
    let retry = state.request_id();
    assert_ne!(retry, original);
    assert_eq!(state.request_attempt(), 1);
    state.record_recovery_attempt();
    assert_eq!(state.request_id(), retry);
    assert_eq!(state.request_attempt(), 2);

    let mut untyped = rejected_generation();
    untyped.code = None;
    let mut not_retryable = rejected_generation();
    not_retryable.retryable = Some(false);
    let mut inferred_retryable = rejected_generation();
    inferred_retryable.retryable = None;
    let mut partial_stream = rejected_generation();
    partial_stream.http_status = None;
    let mut different_status = rejected_generation();
    different_status.http_status = Some(200);
    let mut outage = rejected_generation();
    outage.kind = ProviderErrorKind::Outage;
    let mut unknown_code = rejected_generation();
    unknown_code.code = Some("upstream_error".into());
    for error in [
        untyped,
        not_retryable,
        inferred_retryable,
        partial_stream,
        different_status,
        outage,
        unknown_code,
    ] {
        state.start_protocol_recovery(&error.into());
        assert_eq!(state.request_id(), retry);
    }
    state.start_protocol_recovery(&anyhow::anyhow!(
        "transport failed after tool_protocol_error appeared in diagnostic text"
    ));
    assert_eq!(state.request_id(), retry);
}

async fn forced_final_requests(
    error_body: &str,
    resample: bool,
) -> Vec<hi_ai::test_support::RecordedRequest> {
    // Four harmless no-op tool results reach the real forced-final branch.
    // Its retry must retain the same tool-free prompt, so the wire identity
    // cannot depend on guidance mutating the request body.
    let mut steps = ["stop", "quit", "exit", "done"]
        .into_iter()
        .enumerate()
        .map(|(index, word)| {
            ChatStep::new(ScriptedResponse::tool_call(ScriptedToolCall::new(
                format!("noop-{index}"),
                "bash",
                serde_json::json!({"command": format!("echo {word}")}),
            )))
        })
        .collect::<Vec<_>>();
    steps.extend([
        ChatStep::new(ScriptedResponse::json(503, error_body)),
        // The server receives the replacement generation, then its response
        // is lost. The transport retry must replay that replacement identity.
        ChatStep::new(ScriptedResponse::reset()),
    ]);
    if resample {
        steps.push(ChatStep::new(ScriptedResponse::text(
            "Stopped after the available no-op output.",
        )));
    }
    let server = ScriptedOpenAiServer::builder()
        .chat_steps(steps)
        .start()
        .expect("bind local mock server");
    let provider = OpenAiProvider::new(server.v1_url(), "synthetic-test-key".into());
    let mut cfg = config();
    cfg.loop_limits.max_repeat_nudges = 2;
    let mut agent = crate::Agent::new(Arc::new(provider), cfg).unwrap();
    tokio::time::timeout(
        Duration::from_secs(30),
        agent.run_turn("stop when complete", &mut NullUi),
    )
    .await
    .expect("mocked turn must remain bounded")
    .unwrap();
    server.assert_clean().unwrap();
    if resample {
        assert_eq!(
            agent.messages().last().unwrap().text(),
            "Stopped after the available no-op output."
        );
    } else {
        assert!(
            agent
                .last_turn_telemetry()
                .wire_audit
                .iter()
                .any(|event| { event["request_failure"]["reason"] == "attempts_exhausted" })
        );
    }
    let requests = server.requests();
    assert_eq!(requests.len(), if resample { 7 } else { 6 });
    for request in &requests[4..] {
        assert!(request.body.contains("Stop using tools now"));
        assert_eq!(request.json.as_ref().unwrap()["tool_choice"], "none");
        assert!(
            request
                .header("x-request-id")
                .is_some_and(|id| !id.is_empty())
        );
        assert!(
            request
                .header("idempotency-key")
                .is_some_and(|key| !key.is_empty())
        );
    }
    requests
}

#[tokio::test]
async fn rejected_generation_gets_fresh_key_then_lost_response_reuses_it() {
    let requests = forced_final_requests(
        r#"{"error":{"message":"generation rejected","code":"tool_protocol_error","retryable":true}}"#,
        true,
    )
    .await;
    let rejected = &requests[4];
    let replacement = &requests[5];
    let transport_retry = &requests[6];
    assert_eq!(rejected.body, replacement.body);
    assert_ne!(
        rejected.header("x-request-id"),
        replacement.header("x-request-id")
    );
    assert_ne!(
        rejected.header("idempotency-key"),
        replacement.header("idempotency-key")
    );
    assert_eq!(replacement.header("x-request-attempt"), Some("0"));
    assert_eq!(transport_retry.header("x-request-attempt"), Some("1"));
    assert_eq!(replacement.body, transport_retry.body);
    assert_eq!(
        replacement.header("x-request-id"),
        transport_retry.header("x-request-id")
    );
    assert_eq!(
        replacement.header("idempotency-key"),
        transport_retry.header("idempotency-key")
    );
}

#[tokio::test]
async fn unknown_503_and_lost_response_preserve_possibly_paid_generation_key() {
    let requests = forced_final_requests(
        r#"{"error":{"message":"upstream unavailable","code":"upstream_error","retryable":true,"retry_after_seconds":0}}"#,
        false,
    )
    .await;
    for request in &requests[5..] {
        assert_eq!(requests[4].body, request.body);
        assert_eq!(
            requests[4].header("x-request-id"),
            request.header("x-request-id")
        );
        assert_eq!(
            requests[4].header("idempotency-key"),
            request.header("idempotency-key")
        );
    }
}
