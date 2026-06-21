//! OpenAI Chat Completions adapter.
//!
//! Covers OpenRouter, terminaili.com, and local servers (Ollama, llama.cpp,
//! LM Studio, vLLM) — they differ only by base URL and API key.

use std::time::Instant;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::provider::Provider;
use crate::types::{ChatRequest, Completion, Content, Message, Role, StreamEvent, Usage};

pub struct OpenAiProvider {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiProvider {
    pub fn new(base_url: String, api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn stream(
        &self,
        request: ChatRequest,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> Result<Completion> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = build_body(&request);

        let resp = crate::http::send_with_retry(
            self.http.post(&url).bearer_auth(&self.api_key).json(&body),
        )
        .await
        .context("request to model endpoint failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("API error {status}: {text}");
        }

        // `debug_tap` optionally echoes the raw wire bytes when HI_DEBUG_STREAM
        // is set. We give up only if no real output arrives for STREAM_IDLE_TIMEOUT
        // — `last_progress` is reset on each token, so keep-alive heartbeats (which
        // produce no token) eventually trip it, while a streaming model never does.
        let mut stream = crate::http::debug_tap(resp.bytes_stream()).eventsource();
        let mut text = String::new();
        let mut tool_calls: Vec<ToolCallBuilder> = Vec::new();
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
            if event.data == "[DONE]" {
                break;
            }
            let Ok(chunk) = serde_json::from_str::<ChatChunk>(&event.data) else {
                continue;
            };

            if let Some(usage) = chunk.usage {
                completion.usage = Usage {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                };
            }
            for choice in chunk.choices {
                let delta = choice.delta;
                if let Some(reasoning) = delta.reasoning.or(delta.reasoning_content)
                    && !reasoning.is_empty()
                {
                    sink(StreamEvent::Reasoning(reasoning));
                    last_progress = Instant::now();
                }
                if let Some(content) = delta.content
                    && !content.is_empty()
                {
                    text.push_str(&content);
                    sink(StreamEvent::Text(content));
                    last_progress = Instant::now();
                }
                if let Some(deltas) = delta.tool_calls {
                    last_progress = Instant::now();
                    for tcd in deltas {
                        if tool_calls.len() <= tcd.index {
                            tool_calls.resize_with(tcd.index + 1, ToolCallBuilder::default);
                        }
                        let builder = &mut tool_calls[tcd.index];
                        if let Some(id) = tcd.id
                            && !id.is_empty()
                        {
                            builder.id = id;
                        }
                        if let Some(func) = tcd.function {
                            if let Some(name) = func.name {
                                builder.name.push_str(&name);
                            }
                            if let Some(args) = func.arguments {
                                builder.arguments.push_str(&args);
                            }
                        }
                    }
                }
                if let Some(finish_reason) = choice.finish_reason {
                    completion.stop_reason = Some(finish_reason);
                }
            }
        }

        if !text.is_empty() {
            completion.content.push(Content::Text(text));
        }
        for builder in tool_calls {
            if !builder.name.is_empty() {
                completion.content.push(builder.finish());
            }
        }
        Ok(completion)
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/models", self.base_url);
        crate::http::fetch_model_ids(self.http.get(&url).bearer_auth(&self.api_key)).await
    }
}

fn build_body(request: &ChatRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "messages": to_openai_messages(&request.messages),
        "stream": true,
        "stream_options": { "include_usage": true },
        "max_tokens": request.max_tokens,
    });
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = json!(tools);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    body
}

/// Flatten neutral messages into OpenAI's wire shape. Thinking blocks are
/// dropped (the Chat Completions API has no place to put them).
fn to_openai_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message.role {
            Role::System => out.push(json!({ "role": "system", "content": message.text() })),
            Role::User => out.push(json!({ "role": "user", "content": message.text() })),
            Role::Assistant => {
                let mut thinking = String::new();
                let mut text = String::new();
                let mut tool_calls = Vec::new();
                for block in &message.content {
                    match block {
                        Content::Text(t) => text.push_str(t),
                        // Cross-provider handoff: the Chat Completions API has no
                        // reasoning field, so preserve Anthropic-style thinking as
                        // inline tags rather than dropping it.
                        Content::Thinking { text: t, .. } => thinking.push_str(t),
                        Content::ToolCall {
                            id,
                            name,
                            arguments,
                        } => tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": { "name": name, "arguments": arguments },
                        })),
                        Content::ToolResult { .. } => {}
                    }
                }
                let mut content = String::new();
                if !thinking.is_empty() {
                    content.push_str(&format!("<thinking>\n{thinking}\n</thinking>\n"));
                }
                content.push_str(&text);

                let mut msg = json!({ "role": "assistant" });
                if tool_calls.is_empty() {
                    msg["content"] = json!(content);
                } else {
                    msg["tool_calls"] = json!(tool_calls);
                    // Omit content when empty; OpenAI allows it and some servers
                    // (e.g. Ollama) reject an explicit null.
                    if !content.is_empty() {
                        msg["content"] = json!(content);
                    }
                }
                out.push(msg);
            }
            Role::Tool => {
                for block in &message.content {
                    if let Content::ToolResult { call_id, output } = block {
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": output,
                        }));
                    }
                }
            }
        }
    }
    out
}

#[derive(Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallBuilder {
    fn finish(self) -> Content {
        Content::ToolCall {
            id: self.id,
            name: self.name,
            arguments: if self.arguments.is_empty() {
                "{}".into()
            } else {
                self.arguments
            },
        }
    }
}

// --- Streaming response shapes -------------------------------------------

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::to_openai_messages;
    use crate::types::{Content, Message};

    #[test]
    fn system_and_user_become_text_messages() {
        let out = to_openai_messages(&[Message::system("sys"), Message::user("hi")]);
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[0]["content"], "sys");
        assert_eq!(out[1]["role"], "user");
        assert_eq!(out[1]["content"], "hi");
    }

    #[test]
    fn assistant_tool_call_omits_content_rather_than_null() {
        let out = to_openai_messages(&[Message::assistant(vec![Content::ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: "{}".into(),
        }])]);
        // Ollama rejects null content; we omit the key entirely.
        assert!(out[0].get("content").is_none());
        assert!(out[0]["tool_calls"].is_array());
        assert_eq!(out[0]["tool_calls"][0]["function"]["name"], "bash");
    }

    #[test]
    fn thinking_is_preserved_as_inline_tag() {
        let out = to_openai_messages(&[Message::assistant(vec![
            Content::Thinking {
                text: "my reasoning".into(),
                signature: None,
            },
            Content::Text("the answer".into()),
        ])]);
        let content = out[0]["content"].as_str().unwrap();
        assert!(content.contains("<thinking>"));
        assert!(content.contains("my reasoning"));
        assert!(content.contains("the answer"));
    }

    #[test]
    fn tool_result_maps_to_tool_role() {
        let out = to_openai_messages(&[Message::tool_result("call_1", "the output")]);
        assert_eq!(out[0]["role"], "tool");
        assert_eq!(out[0]["tool_call_id"], "call_1");
        assert_eq!(out[0]["content"], "the output");
    }
}
