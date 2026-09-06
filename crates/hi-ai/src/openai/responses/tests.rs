use super::{OpenAiProvider, request::build_body, supports_model};
use crate::test_support::{ChatStep, RequestMatcher, ScriptedOpenAiServer, ScriptedResponse};
use crate::{
    ChatRequest, Content, Message, Provider, ReasoningEffort, RequestProfile, StreamEvent,
    ToolMode, ToolSpec,
};
use serde_json::{Value, json};

fn request() -> ChatRequest {
    ChatRequest {
        model: "gpt-6-astra".into(),
        request_id: Some("astra_test".into()),
        retry_attempt: 0,
        user_turn: true,
        canonical_objective: None,
        messages: vec![
            Message::system("Help with code."),
            Message::user("Read the file."),
        ]
        .into(),
        tools: vec![ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
            parameters: json!({"type":"object", "properties": {
                "path":{"type":"string"}, "limit":{"type":"integer"}
            }, "required":["path"]}),
        }]
        .into(),
        tool_envelope: None,
        max_tokens: 256,
        temperature: Some(0.7),
        top_p: Some(0.95),
        frequency_penalty: Some(0.4),
        thinking_budget: None,
        reasoning_effort: Some(ReasoningEffort::Minimal),
        profile: RequestProfile::default(),
    }
}

fn completed(output: Value) -> String {
    format!(
        "data: {}\n\n",
        json!({
            "type":"response.completed", "response": {
                "id":"resp_test", "status":"completed", "output":output,
                "usage":{"input_tokens":100,"output_tokens":20,
                    "input_tokens_details":{"cached_tokens":40,"cache_write_tokens":30}}
            }
        })
    )
}

#[test]
fn astra_request_uses_supported_parameters_and_keeps_optional_tool_fields() {
    let mut req = request();
    let body = build_body(&req);
    assert_eq!(body["reasoning"]["effort"], "low");
    assert_eq!(body["max_output_tokens"], 256);
    assert_eq!(body["store"], false);
    assert_eq!(body["prompt_cache_options"]["ttl"], "30m");
    assert_eq!(body["input"][0]["role"], "developer");
    for field in [
        "temperature",
        "top_p",
        "frequency_penalty",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_effort",
        "messages",
        "prompt_cache_retention",
    ] {
        assert!(body.get(field).is_none(), "unexpected field: {field}");
    }
    assert_eq!(body["tools"][0]["strict"], false);
    assert_eq!(body["tools"][0]["parameters"], req.tools[0].parameters);
    req.reasoning_effort = Some(ReasoningEffort::High);
    assert_eq!(build_body(&req)["reasoning"]["effort"], "high");
    req.reasoning_effort = None;
    assert!(build_body(&req).get("reasoning").is_none());
    req.profile.tool_mode = ToolMode::ChatOnly;
    let disabled = build_body(&req);
    assert_eq!(disabled["tool_choice"], "none");
    assert_eq!(disabled["tools"], body["tools"]);
    req.profile.tool_mode = ToolMode::Required;
    assert_eq!(build_body(&req)["tool_choice"], "required");
    assert!(supports_model("gpt-6-astra"));
    assert!(supports_model("openai/gpt-6-astra"));
    assert!(!supports_model("gpt-5.6-sol"));
    assert!(!supports_model("not-gpt-6-astra"));
}

#[test]
fn astra_capabilities_allow_parallel_tools_without_changing_other_models() {
    let provider = OpenAiProvider::new("https://api.openai.com/v1".into(), "test".into());
    let astra = provider.capability_candidates("openai", "gpt-6-astra");
    let legacy = provider.capability_candidates("openai", "legacy");
    assert!(astra[0].declared.parallel_tool_calls);
    assert!(astra[0].declared.reasoning_replay.signed_or_encrypted);
    assert_eq!(legacy[0].declared, provider.capabilities());
}

#[test]
fn astra_replays_order_phase_and_reasoning_and_ignores_stale_metadata() {
    let items = vec![
        json!({"id":"rs_1", "type":"reasoning", "summary":[], "encrypted_content":"opaque"}),
        json!({"id":"msg_1", "type":"message", "role":"assistant", "phase":"commentary",
            "status":"completed", "content":[{"type":"output_text", "text":"Looking.", "annotations":[]}]}),
        json!({"id":"fc_1", "type":"function_call", "call_id":"call_1", "name":"read",
            "arguments":"{\"path\":\"a.rs\"}", "status":"completed"}),
    ];
    let assistant = Message::assistant(vec![
        Content::Text("Looking.".into()),
        Content::ToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: "{\"path\":\"a.rs\"}".into(),
        },
    ])
    .with_provider_replay("openai-responses", items.clone());
    let mut req = request();
    req.messages = vec![
        assistant.clone(),
        Message::tool_result("call_1", "contents"),
    ]
    .into();
    let body = build_body(&req);
    let input = body["input"].as_array().unwrap();
    assert_eq!(&input[..3], items);
    assert_eq!(
        input[3],
        json!({"type":"function_call_output","call_id":"call_1","output":"contents"})
    );
    let mut edited = assistant;
    edited.content[0] = Content::Text("Edited.".into());
    req.messages = vec![edited].into();
    let body = build_body(&req);
    assert_eq!(body["input"][0]["content"], "Edited.");
    assert!(!body.to_string().contains("opaque"));
}

#[tokio::test]
async fn astra_transport_executes_two_responses_with_call_ids_and_replay() {
    let output = json!([
        {"id":"rs_1","type":"reasoning","summary":[],"encrypted_content":"cipher"},
        {"id":"msg_1","type":"message","role":"assistant","status":"completed","phase":"commentary",
            "content":[{"type":"output_text","text":"Reading.","annotations":[]}]},
        {"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1",
            "name":"read","arguments":"{\"path\":\"a.rs\"}"}
    ]);
    let final_output = json!([{"id":"msg_2","type":"message","role":"assistant","status":"completed",
        "phase":"final_answer","content":[{"type":"output_text","text":"Done.","annotations":[]}]}]);
    let matcher = || {
        RequestMatcher::any()
            .method("POST")
            .path("/v1/responses")
            .body_contains("max_output_tokens")
            .body_excludes("temperature")
            .body_excludes("top_p")
    };
    let server = ScriptedOpenAiServer::new(vec![
        ChatStep::expecting(
            matcher(),
            ScriptedResponse::raw_sse(completed(output.clone())),
        ),
        ChatStep::expecting(
            matcher()
                .body_contains("function_call_output")
                .body_contains("cipher"),
            ScriptedResponse::raw_sse(completed(final_output)),
        ),
    ])
    .expect("Responses transport regression requires loopback socket access");
    let provider = OpenAiProvider::new(server.v1_url(), "test".into());
    let mut req = request();
    let mut events = Vec::new();
    let first = provider
        .stream(req.clone(), &mut |event| events.push(event))
        .await
        .unwrap();
    assert_eq!(first.tool_calls()[0].id, "call_1");
    assert_eq!(first.usage.cache_read_tokens, 40);
    assert_eq!(first.usage.cache_creation_tokens, 30);
    let mut messages = req.messages.to_vec();
    messages.push(Message::assistant(first.content));
    messages.push(Message::tool_result("call_1", "file contents"));
    req.messages = messages.into();
    let second = provider.stream(req, &mut |_| {}).await.unwrap();
    assert_eq!(Message::assistant(second.content).text(), "Done.");
    server.assert_clean().unwrap();
    let requests = server.requests();
    let body: Value = serde_json::from_str(&requests[1].body).unwrap();
    assert_eq!(
        &body["input"].as_array().unwrap()[2..5],
        output.as_array().unwrap()
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, StreamEvent::WireAudit(audit)
        if audit.output_token_parameter == "max_output_tokens" && audit.temperature.is_none()
        && audit.accepted && audit.provider == "openai"))
    );
}

#[tokio::test]
async fn astra_rejection_does_not_retry_chat_completions_or_drop_tools() {
    let server = ScriptedOpenAiServer::new(vec![ChatStep::expecting(
        RequestMatcher::any().path("/v1/responses"),
        ScriptedResponse::json(
            400,
            json!({"error":{"message":"unsupported tool schema"}}).to_string(),
        ),
    )])
    .expect("Responses transport regression requires loopback socket access");
    let provider = OpenAiProvider::new(server.v1_url(), "test".into());
    assert!(provider.stream(request(), &mut |_| {}).await.is_err());
    assert_eq!(server.requests().len(), 1);
    server.assert_clean().unwrap();
}
