//! Native Anthropic Messages API adapter.
//!
//! Unlike the OpenAI shape, Anthropic uses a top-level `system` string,
//! content-block messages, tool results carried inside `user` messages, and
//! an event-typed SSE stream. Extended thinking is surfaced as `thinking`
//! blocks whose `signature` must be echoed back on the next turn.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{Value, json};

use crate::provider::Provider;
use crate::types::{ChatRequest, Completion, Content, Message, Role, StreamEvent};

const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
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
            bail!("API error {status}: {text}");
        }

        // `debug_tap` optionally echoes the raw wire bytes when HI_DEBUG_STREAM
        // is set. We give up only if no real output arrives for STREAM_IDLE_TIMEOUT:
        // `last_progress` is reset on every event except `ping` (the heartbeat),
        // so a streaming model never trips it but a heartbeat-only stall does.
        let mut stream = crate::http::debug_tap(resp.bytes_stream()).eventsource();
        let mut blocks: Vec<Option<BlockBuilder>> = Vec::new();
        let mut completion = Completion::default();
        let idle = crate::http::stream_idle_timeout();
        let mut last_progress = Instant::now();

        loop {
            let budget = idle.saturating_sub(last_progress.elapsed());
            let event = match tokio::time::timeout(budget, stream.next()).await {
                Ok(Some(event)) => event.context("error reading stream")?,
                Ok(None) => break,
                Err(_) => bail!(
                    "model produced no output for {}s — the provider streamed only \
                     keep-alive heartbeats. It may be overloaded or the model unavailable; \
                     try again, or switch with /model.",
                    idle.as_secs()
                ),
            };
            let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };
            if event.event != "ping" {
                last_progress = Instant::now();
            }

            match event.event.as_str() {
                "message_start" => {
                    if let Some(tokens) = data["message"]["usage"]["input_tokens"].as_u64() {
                        completion.usage.input_tokens = tokens;
                    }
                }
                "content_block_start" => {
                    let index = data["index"].as_u64().unwrap_or(0) as usize;
                    if blocks.len() <= index {
                        blocks.resize_with(index + 1, || None);
                    }
                    blocks[index] = Some(BlockBuilder::start(&data["content_block"]));
                }
                "content_block_delta" => {
                    let index = data["index"].as_u64().unwrap_or(0) as usize;
                    if let Some(Some(builder)) = blocks.get_mut(index) {
                        builder.apply_delta(&data["delta"], sink);
                    }
                }
                "message_delta" => {
                    if let Some(reason) = data["delta"]["stop_reason"].as_str() {
                        completion.stop_reason = Some(reason.to_string());
                    }
                    if let Some(tokens) = data["usage"]["output_tokens"].as_u64() {
                        completion.usage.output_tokens = tokens;
                    }
                }
                "error" => {
                    let message = data["error"]["message"].as_str().unwrap_or("unknown error");
                    bail!("Anthropic stream error: {message}");
                }
                _ => {}
            }
        }

        completion.content = blocks
            .into_iter()
            .flatten()
            .filter_map(BlockBuilder::finish)
            .collect();
        Ok(completion)
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/v1/models", self.base_url);
        crate::http::fetch_model_ids(
            self.http
                .get(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION),
        )
        .await
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
        body["system"] = json!(system);
    }
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect();
        body["tools"] = json!(tools);
    }
    if let Some(budget) = request.thinking_budget {
        body["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
        // Extended thinking requires the default temperature, so don't set it.
    } else if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
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
                out.push(json!({
                    "role": "user",
                    "content": [{ "type": "text", "text": message.text() }],
                }));
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
                out.push(json!({ "role": "user", "content": content }));
            }
        }
    }

    (system, out)
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
