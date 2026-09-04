use super::*;
use crate::test_support::{FakeOpenAiServer, Response, sse_text};
use crate::{Message, RequestProfile, ToolSpec};

fn request(tools: Vec<ToolSpec>, profile: RequestProfile) -> ChatRequest {
    ChatRequest {
        model: "m".into(),
        request_id: None,
        retry_attempt: 0,
        user_turn: false,
        canonical_objective: None,
        messages: vec![Message::user("hi")].into(),
        tools: tools.into(),
        tool_envelope: None,
        max_tokens: 16,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile,
    }
}

fn measured_gateway_tool_response(input: u64, output: u64, cached: u64) -> String {
    let chunk = json!({
        "choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{
            "name":"read","arguments":"{\"path\":\"README.md\"}"
        }}]},"finish_reason":"tool_calls"}],
        "usage":{"prompt_tokens":input,"completion_tokens":output,
            "prompt_tokens_details":{"cached_tokens":cached}}
    });
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

#[tokio::test]
async fn stripped_reasoning_retry_accounts_for_both_completed_responses() {
    let Some(server) = FakeOpenAiServer::new(vec![
        Response::sse(measured_gateway_tool_response(100, 10, 40)),
        Response::sse(measured_gateway_tool_response(200, 20, 80)),
    ]) else {
        return;
    };
    let provider = OpenAiProvider::new(server.url().into(), "unused".into())
        .with_capability_base_url("https://gateway.example/v1");
    let mut req = request(vec![], RequestProfile::default());
    req.model = "deepseek-v4-flash".into();
    let completion = provider.stream(req, &mut |_| {}).await.unwrap();
    assert_eq!(server.bodies().len(), 2);
    assert_eq!(completion.usage.input_tokens, 300);
    assert_eq!(completion.usage.output_tokens, 30);
    assert_eq!(completion.usage.cache_read_tokens, 120);
    assert_eq!(completion.usage.context_occupancy, 200);
    assert!(!completion.usage.estimated);
    assert_eq!(completion.tool_calls().len(), 1);
}

#[tokio::test]
async fn stripped_reasoning_retry_failure_keeps_already_consumed_usage() {
    for failure in [
        Response::json(400, r#"{"error":"unrecoverable bad shape"}"#),
        Response::sse("data: {\"error\":{\"message\":\"provider stream failed\"}}\n\n"),
    ] {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::sse(measured_gateway_tool_response(100, 10, 40)),
            failure,
        ]) else {
            return;
        };
        let provider = OpenAiProvider::new(server.url().into(), "unused".into())
            .with_capability_base_url("https://gateway.example/v1");
        let mut req = request(vec![], RequestProfile::default());
        req.model = "deepseek-v4-flash".into();
        let error = provider.stream(req, &mut |_| {}).await.unwrap_err();
        let usage = crate::provider_error_usage(&error);
        assert_eq!(server.bodies().len(), 2);
        assert!(usage.input_tokens >= 100);
        assert_eq!(usage.output_tokens, 10);
        assert_eq!(usage.cache_read_tokens, 40);
    }
}

#[tokio::test]
async fn fake_server_rejects_stream_options_then_succeeds() {
    let Some(server) = FakeOpenAiServer::new(vec![
        Response::json(400, r#"{"error":"stream_options unsupported"}"#),
        Response::sse(sse_text("ok")),
    ]) else {
        return;
    };
    let provider = OpenAiProvider::new(server.url().to_string(), "test".into());
    let request = request(vec![], Default::default());
    let mut statuses = Vec::new();
    let mut sink = |event| {
        if let StreamEvent::Status(status) = event {
            statuses.push(status);
        }
    };
    let completion = provider.stream(request, &mut sink).await.unwrap();
    assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "ok"));
    assert!(
        completion.usage.input_tokens > 0,
        "fallback request gets estimated input usage: {:?}",
        completion.usage
    );
    assert!(
        completion.usage.output_tokens > 0,
        "fallback request gets estimated output usage: {:?}",
        completion.usage
    );
    assert!(
        statuses.is_empty(),
        "provider wire-shape retries must not appear as user status: {statuses:?}"
    );
    let bodies = server.bodies();
    assert!(bodies[0].contains("stream_options"));
    assert!(!bodies[1].contains("stream_options"));
    let request_ids = server.request_ids();
    let idempotency_keys = server.idempotency_keys();
    assert_ne!(request_ids[0], request_ids[1]);
    assert_ne!(idempotency_keys[0], idempotency_keys[1]);
}

#[tokio::test]
async fn optional_field_retries_are_monotonic_and_cached_without_losing_tools() {
    let Some(server) = FakeOpenAiServer::new(vec![
        Response::json(400, r#"{"error":"stream_options unsupported"}"#),
        Response::json(400, r#"{"error":"frequency_penalty unsupported"}"#),
        Response::sse(sse_text("first")),
        Response::sse(sse_text("second")),
    ]) else {
        return;
    };
    let provider = OpenAiProvider::new(server.url().into(), "unused".into());
    let mut req = request(
        vec![ToolSpec {
            name: "read".into(),
            description: "Read workspace source files before editing them.".repeat(20),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        }],
        RequestProfile::default(),
    );
    req.frequency_penalty = Some(0.3);
    for expected in ["first", "second"] {
        let completion = provider.stream(req.clone(), &mut |_| {}).await.unwrap();
        assert!(
            matches!(completion.content.first(), Some(Content::Text(text)) if text == expected)
        );
    }
    let raw = server.bodies();
    let bodies: Vec<Value> = raw
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect();
    assert_eq!(
        bodies.len(),
        4,
        "three discovery requests, then one cached request"
    );
    assert!(bodies[0].get("stream_options").is_some());
    assert!(bodies[0].get("frequency_penalty").is_some());
    assert!(bodies[1].get("stream_options").is_none());
    assert!(bodies[1].get("frequency_penalty").is_some());
    for body in &bodies[2..] {
        assert!(body.get("stream_options").is_none());
        assert!(body.get("frequency_penalty").is_none());
    }
    for body in &bodies {
        assert_eq!(body["tools"], bodies[0]["tools"]);
        assert_eq!(body["messages"], bodies[0]["messages"]);
    }

    // Before the fix each turn sent (usage+penalty), (penalty), (usage), ().
    // Compare serialized bytes with the same complete schemas and messages.
    let mut readded_usage = bodies[0].clone();
    readded_usage
        .as_object_mut()
        .unwrap()
        .remove("frequency_penalty");
    let before_bytes = 2
        * (raw[0].len()
            + raw[1].len()
            + serde_json::to_vec(&readded_usage).unwrap().len()
            + raw[2].len());
    let after_bytes: usize = raw.iter().map(String::len).sum();
    assert!(after_bytes * 100 < before_bytes * 55);
    eprintln!(
        "compatibility fixture: 8 -> 4 HTTP requests; {before_bytes} -> {after_bytes} serialized request bytes"
    );
}

#[test]
fn learned_optional_fields_respect_model_strict_and_explicit_usage_scope() {
    let cache = compatibility::CompatibilityCache::default();
    let mut req = request(Vec::new(), RequestProfile::default());
    req.frequency_penalty = Some(0.3);
    let mut accepted = request::request_attempts(&req)[0];
    accepted.include_usage = false;
    accepted.include_frequency_penalty = false;
    cache.remember(&req, accepted);

    let mut other_model = req.clone();
    other_model.model = "other-model".into();
    cache.apply(&mut other_model);
    assert!(other_model.profile.stream_usage.is_none());
    assert_eq!(other_model.frequency_penalty, Some(0.3));

    let mut strict = req.clone();
    strict.profile.compat = CompatMode::Strict;
    cache.apply(&mut strict);
    assert!(strict.profile.stream_usage.is_none());
    assert_eq!(strict.frequency_penalty, Some(0.3));

    req.profile.stream_usage = Some(true);
    cache.apply(&mut req);
    assert_eq!(req.profile.stream_usage, Some(true));
    assert_eq!(req.frequency_penalty, None);
}
