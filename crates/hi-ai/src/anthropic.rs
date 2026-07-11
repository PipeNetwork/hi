//! Native Anthropic Messages API adapter.
//!
//! Unlike the OpenAI shape, Anthropic uses a top-level `system` string,
//! content-block messages, tool results carried inside `user` messages, and
//! an event-typed SSE stream. Extended thinking is surfaced as `thinking`
//! blocks whose `signature` must be echoed back on the next turn.

use anyhow::{Context, Result};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::provider::{Provider, ProviderError, ProviderErrorKind};
use crate::types::{
    ChatRequest, Completion, Content, Message, Role, StreamEvent,
    estimate_completion_output_tokens, estimate_messages_tokens,
};

const API_VERSION: &str = "2023-06-01";

/// Upper bound on content blocks in one response — the per-event `index`
/// arrives straight from the provider's SSE JSON.
const MAX_CONTENT_BLOCKS: usize = 512;

pub struct AnthropicProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            http: crate::http::agent_http_client(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = build_body(&request);

        let resp = crate::http::send_with_retry(
            self.http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .json(&body),
        )
        .await
        .context("request to Anthropic endpoint failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::new(
                classify_http_error(status),
                format!("API error {status}: {text}"),
            )
            .into());
        }

        // `debug_tap` optionally echoes the raw wire bytes when HI_DEBUG_STREAM
        // is set.
        let mut stream = crate::http::debug_tap(resp.bytes_stream()).eventsource();
        let mut blocks: Vec<Option<BlockBuilder>> = Vec::new();
        let mut completion = Completion::default();
        let mut stream_complete = false;
        let mut progressed = false;

        loop {
            let Some(event) = stream.next().await else {
                break;
            };
            let event = match event {
                Ok(event) => event,
                // Mirror the OpenAI path: an unclean mid-stream close AFTER the
                // answer finished or after content has already streamed must not
                // discard a (near-)complete response and force a full re-bill —
                // return what we have (the input tokens from `message_start` are
                // already in `completion.usage`; output is estimated below). With
                // no progress yet it's a genuine failure: propagate.
                Err(err) => {
                    if stream_complete || progressed {
                        break;
                    }
                    return Err(err).context("error reading stream");
                }
            };
            let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };

            match event.event.as_str() {
                "message_start" => {
                    if let Some(tokens) = data["message"]["usage"]["input_tokens"].as_u64() {
                        completion.usage.input_tokens = tokens;
                    }
                    if let Some(tokens) =
                        data["message"]["usage"]["cache_read_input_tokens"].as_u64()
                    {
                        completion.usage.cache_read_tokens = tokens;
                    }
                    if let Some(tokens) =
                        data["message"]["usage"]["cache_creation_input_tokens"].as_u64()
                    {
                        completion.usage.cache_creation_tokens = tokens;
                    }
                    // Anthropic reports cache tokens separately from
                    // `input_tokens`, so the full context window occupancy is
                    // the sum of all three. Saturating: the counts come straight
                    // off the wire, so a corrupt frame can't overflow-panic here.
                    completion.usage.context_occupancy = completion
                        .usage
                        .input_tokens
                        .saturating_add(completion.usage.cache_read_tokens)
                        .saturating_add(completion.usage.cache_creation_tokens);
                }
                "content_block_start" => {
                    let index = data["index"].as_u64().unwrap_or(0) as usize;
                    // The index comes straight off the wire — bound it so a
                    // corrupt frame can't force a huge `resize_with` allocation.
                    if index >= MAX_CONTENT_BLOCKS {
                        continue;
                    }
                    if blocks.len() <= index {
                        blocks.resize_with(index + 1, || None);
                    }
                    blocks[index] = Some(BlockBuilder::start(&data["content_block"]));
                }
                "content_block_delta" => {
                    let index = data["index"].as_u64().unwrap_or(0) as usize;
                    if let Some(Some(builder)) = blocks.get_mut(index) {
                        builder.apply_delta(&data["delta"], sink);
                        progressed = true;
                    }
                }
                "message_delta" => {
                    if let Some(reason) = data["delta"]["stop_reason"].as_str() {
                        completion.stop_reason = Some(reason.to_string());
                        stream_complete = true;
                    }
                    if let Some(tokens) = data["usage"]["output_tokens"].as_u64() {
                        completion.usage.output_tokens = tokens;
                    }
                }
                "error" => {
                    let message = data["error"]["message"].as_str().unwrap_or("unknown error");
                    let error_type = data["error"]["type"].as_str().unwrap_or("");
                    let kind = match error_type {
                        "overloaded_error" | "rate_limit_error" => ProviderErrorKind::RateLimit,
                        "authentication_error" => ProviderErrorKind::Auth,
                        "invalid_request_error" => ProviderErrorKind::UnsupportedRequestShape,
                        _ => ProviderErrorKind::Other,
                    };
                    return Err(ProviderError::new(
                        kind,
                        format!("Anthropic stream error: {message}"),
                    )
                    .with_usage(completion.usage)
                    .into());
                }
                _ => {}
            }
            if stream_complete {
                break;
            }
        }

        completion.content = blocks
            .into_iter()
            .flatten()
            .filter_map(BlockBuilder::finish)
            .collect();
        if completion.usage.input_tokens == 0 {
            completion.usage.input_tokens = estimate_messages_tokens(&request.messages);
            completion.usage.estimated = true;
        }
        if completion.usage.output_tokens == 0 {
            completion.usage.output_tokens = estimate_completion_output_tokens(&completion.content);
            completion.usage.estimated = true;
        }
        // Keep the occupancy gauge alive on the estimate path too (matches the
        // OpenAI path's backfill): a proxy that omits `message_start` usage
        // would otherwise leave it at 0 all session.
        if completion.usage.context_occupancy == 0 {
            completion.usage.context_occupancy = completion.usage.input_tokens;
        }
        Ok(completion)
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ServedModel>> {
        let url = format!("{}/v1/models", self.base_url);
        crate::http::fetch_models(
            self.http
                .get(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION),
        )
        .await
    }
}

fn classify_http_error(status: reqwest::StatusCode) -> ProviderErrorKind {
    match status {
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
            ProviderErrorKind::Auth
        }
        reqwest::StatusCode::TOO_MANY_REQUESTS => ProviderErrorKind::RateLimit,
        s if s.is_server_error() => ProviderErrorKind::Outage,
        reqwest::StatusCode::BAD_REQUEST | reqwest::StatusCode::UNPROCESSABLE_ENTITY => {
            ProviderErrorKind::UnsupportedRequestShape
        }
        _ => ProviderErrorKind::Other,
    }
}

fn build_body(request: &ChatRequest) -> Value {
    let (system, messages) = to_anthropic_messages(&request.messages);
    let mut body = json!({
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
        "stream": true,
    });
    if !system.is_empty() {
        // Use the array form with cache_control so the system prompt is cached
        // on the provider side. After the first request in a session, this ~500-
        // token block is served from cache at ~10% of normal input cost.
        body["system"] = json!([
            {
                "type": "text",
                "text": system,
                "cache_control": { "type": "ephemeral" },
            }
        ]);
    }
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut tool = json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                });
                // Cache the tool definitions (they never change within a
                // session). cache_control goes on the last tool.
                if i == request.tools.len() - 1 {
                    tool["cache_control"] = json!({ "type": "ephemeral" });
                }
                tool
            })
            .collect();
        body["tools"] = json!(tools);
    }
    if let Some(budget) = request.thinking_budget {
        body["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
        // Extended thinking requires default sampling, so set neither temperature
        // nor top_p. (Anthropic has no frequency_penalty; it's ignored either way.)
    } else {
        if let Some(temperature) = request.temperature {
            body["temperature"] = json!(temperature);
        }
        if let Some(top_p) = request.top_p {
            body["top_p"] = json!(top_p);
        }
    }
    body
}

/// Build Anthropic's `(system, messages)` pair. System messages are hoisted to
/// the top-level `system` string; consecutive tool-result messages are merged
/// into a single `user` message, as the API requires.
fn to_anthropic_messages(messages: &[Message]) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut out: Vec<Value> = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let message = &messages[i];
        match message.role {
            Role::System => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&message.text());
                i += 1;
            }
            Role::User => {
                let content = anthropic_user_content(message);
                out.push(json!({ "role": "user", "content": content }));
                i += 1;
            }
            Role::Assistant => {
                let mut content = Vec::new();
                for block in &message.content {
                    match block {
                        // Anthropic rejects thinking blocks without a signature.
                        Content::Thinking {
                            text,
                            signature: Some(signature),
                        } => {
                            content.push(json!({
                                "type": "thinking",
                                "thinking": text,
                                "signature": signature,
                            }));
                        }
                        Content::Text(text) if !text.is_empty() => {
                            content.push(json!({ "type": "text", "text": text }));
                        }
                        Content::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            let input: Value =
                                serde_json::from_str(arguments).unwrap_or_else(|_| json!({}));
                            content.push(json!({
                                "type": "tool_use",
                                "id": id,
                                "name": name,
                                "input": input,
                            }));
                        }
                        _ => {}
                    }
                }
                out.push(json!({ "role": "assistant", "content": content }));
                i += 1;
            }
            Role::Tool => {
                let mut content = Vec::new();
                while i < messages.len() && messages[i].role == Role::Tool {
                    for block in &messages[i].content {
                        if let Content::ToolResult { call_id, output } = block {
                            content.push(json!({
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": output,
                            }));
                        }
                    }
                    i += 1;
                }
                while i < messages.len() && messages[i].role == Role::User {
                    content.extend(anthropic_user_content(&messages[i]));
                    i += 1;
                }
                out.push(json!({ "role": "user", "content": content }));
            }
        }
    }

    (system, out)
}

fn anthropic_user_content(message: &Message) -> Vec<Value> {
    let mut content = Vec::new();
    for block in &message.content {
        match block {
            Content::Image { data, media_type } => content.push(json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            })),
            Content::Text(text) if !text.is_empty() => {
                content.push(json!({ "type": "text", "text": text }));
            }
            _ => {}
        }
    }
    if content.is_empty() {
        content.push(json!({ "type": "text", "text": message.text() }));
    }
    content
}

/// Accumulates one streamed content block (text, thinking, or tool_use).
enum BlockBuilder {
    Text(String),
    Thinking {
        text: String,
        signature: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
}

impl BlockBuilder {
    fn start(content_block: &Value) -> Self {
        match content_block["type"].as_str() {
            Some("tool_use") => BlockBuilder::ToolUse {
                id: content_block["id"].as_str().unwrap_or_default().to_string(),
                name: content_block["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                input: String::new(),
            },
            Some("thinking") => BlockBuilder::Thinking {
                text: content_block["thinking"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                signature: String::new(),
            },
            _ => BlockBuilder::Text(
                content_block["text"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            ),
        }
    }

    fn apply_delta(&mut self, delta: &Value, sink: &mut (dyn FnMut(StreamEvent) + Send)) {
        match (self, delta["type"].as_str()) {
            (BlockBuilder::Text(text), Some("text_delta")) => {
                if let Some(chunk) = delta["text"].as_str() {
                    text.push_str(chunk);
                    sink(StreamEvent::Text(chunk.to_string()));
                }
            }
            (BlockBuilder::Thinking { text, .. }, Some("thinking_delta")) => {
                if let Some(chunk) = delta["thinking"].as_str() {
                    text.push_str(chunk);
                    sink(StreamEvent::Reasoning(chunk.to_string()));
                }
            }
            (BlockBuilder::Thinking { signature, .. }, Some("signature_delta")) => {
                if let Some(chunk) = delta["signature"].as_str() {
                    signature.push_str(chunk);
                }
            }
            (BlockBuilder::ToolUse { input, .. }, Some("input_json_delta")) => {
                if let Some(chunk) = delta["partial_json"].as_str() {
                    input.push_str(chunk);
                }
            }
            _ => {}
        }
    }

    fn finish(self) -> Option<Content> {
        match self {
            BlockBuilder::Text(text) if !text.is_empty() => Some(Content::Text(text)),
            BlockBuilder::Text(_) => None,
            BlockBuilder::Thinking { text, signature } => Some(Content::Thinking {
                text,
                signature: (!signature.is_empty()).then_some(signature),
            }),
            BlockBuilder::ToolUse { id, name, input } => Some(Content::ToolCall {
                id,
                name,
                arguments: if input.is_empty() { "{}".into() } else { input },
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::to_anthropic_messages;
    use crate::types::{Content, Message};

    #[test]
    fn system_is_hoisted_to_top_level() {
        let (system, msgs) = to_anthropic_messages(&[Message::system("S"), Message::user("U")]);
        assert_eq!(system, "S");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
    }

    #[test]
    fn consecutive_tool_results_coalesce_into_one_user_message() {
        let (_s, out) = to_anthropic_messages(&[
            Message::tool_result("a", "ra"),
            Message::tool_result("b", "rb"),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "a");
        assert_eq!(content[1]["tool_use_id"], "b");
    }

    #[test]
    fn tool_results_and_following_user_prompt_coalesce_into_one_user_message() {
        let (_s, out) = to_anthropic_messages(&[
            Message::assistant(vec![Content::ToolCall {
                id: "a".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]),
            Message::tool_result("a", "ra"),
            Message::user("next prompt"),
        ]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[1]["role"], "user");
        let content = out[1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "a");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "next prompt");
    }

    #[test]
    fn thinking_with_signature_is_sent_back() {
        let (_s, out) = to_anthropic_messages(&[Message::assistant(vec![
            Content::Thinking {
                text: "t".into(),
                signature: Some("sig".into()),
            },
            Content::Text("hi".into()),
        ])]);
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["signature"], "sig");
    }

    #[test]
    fn thinking_without_signature_is_dropped() {
        let (_s, out) = to_anthropic_messages(&[Message::assistant(vec![
            Content::Thinking {
                text: "t".into(),
                signature: None,
            },
            Content::Text("hi".into()),
        ])]);
        let content = out[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
    }
}
