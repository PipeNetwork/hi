use std::sync::Arc;

use hi_ai::{ChatRequest, Message, OpenAiProvider, RequestProfile, ToolMode};

pub(super) async fn build(provider: &OpenAiProvider, model: &str, prompt: &str) -> ChatRequest {
    let policy = hi_tools::envelope::seal_chat_only_request(
        provider,
        "team-bench-local",
        model,
        3_500,
        "team-bench",
    )
    .await;
    ChatRequest {
        model: model.to_string(),
        request_id: None,
        retry_attempt: 0,
        user_turn: false,
        canonical_objective: None,
        messages: Arc::new(vec![Message::user(prompt)]),
        tools: Arc::new([]),
        tool_envelope: Some(policy.envelope),
        max_tokens: policy.max_output_tokens,
        temperature: Some(0.0),
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: RequestProfile {
            tool_mode: ToolMode::ChatOnly,
            ..RequestProfile::default()
        },
    }
}
