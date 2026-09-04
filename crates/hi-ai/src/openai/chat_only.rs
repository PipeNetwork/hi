//! ChatOnly wrap-up keeps the tool catalog on the wire when the endpoint
//! supports `tool_choice: none`, so the tool-prefix cache still hits.

use super::super::deepseek::ProviderCapabilities;
use super::{build_body, build_body_with_capabilities, request_attempts, request_attempts_for};
use crate::types::{DeepSeekCompat, Message, RequestProfile, ToolMode, ToolSpec};

fn chat_only_request(model: &str, deepseek_compat: DeepSeekCompat) -> crate::types::ChatRequest {
    crate::types::ChatRequest {
        model: model.into(),
        request_id: None,
        retry_attempt: 0,
        user_turn: false,
        canonical_objective: None,
        messages: vec![Message::user("hi")].into(),
        tools: vec![ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
            parameters: serde_json::json!({"type": "object"}),
        }]
        .into(),
        tool_envelope: None,
        max_tokens: 16,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: RequestProfile {
            tool_mode: ToolMode::ChatOnly,
            deepseek_compat,
            ..Default::default()
        },
    }
}

#[test]
fn chat_only_keeps_tools_and_sets_tool_choice_none() {
    let req = chat_only_request("m", DeepSeekCompat::Off);
    let attempts = request_attempts(&req);
    assert!(attempts[0].include_tools);
    let body = build_body(&req, attempts[0], None);
    assert!(body.get("tools").is_some());
    assert_eq!(body["tool_choice"], "none");
}

#[test]
fn chat_only_omits_tools_without_tool_choice_support() {
    let req = chat_only_request("deepseek-chat", DeepSeekCompat::On);
    let caps = ProviderCapabilities::detect(
        "https://api.deepseek.com",
        &req.model,
        req.profile.deepseek_compat,
    );
    assert!(!caps.supports_tool_choice);
    let attempts = request_attempts_for(&req, &caps);
    assert!(!attempts[0].include_tools);
    let body = build_body_with_capabilities(&req, attempts[0], None, &caps);
    assert!(body.get("tools").is_none());
    assert!(body.get("tool_choice").is_none());
}
