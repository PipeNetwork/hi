//! Pipe Network chat-completions client.
//!
//! OpenAI-compatible wire format against `https://api.pipenetwork.ai/v1`.
//! No compatibility probing, strict schemas, or multi-provider fallbacks.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::TurnCancellation;
use anyhow::{Context, Result, anyhow, bail};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use hi_ai::{Content, Message, ReasoningEffort, Role, ServedModel, ToolSpec, Usage};
use reqwest::StatusCode;
use serde_json::{Value, json};

pub const DEFAULT_BASE_URL: &str = "https://api.pipenetwork.ai/v1";
pub const DEFAULT_MODEL: &str = "pipe/deepseek-v4-flash-0731";
pub const DEFAULT_MAX_TOKENS: u32 = 8192;

#[derive(Debug)]
pub struct PipeError {
    pub status: Option<u16>,
    pub message: String,
    pub retryable: bool,
    pub retry_after: Option<u64>,
}

impl PipeError {
    fn cancelled() -> Self {
        Self {
            status: None,
            message: "cancelled".into(),
            retryable: false,
            retry_after: None,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.message == "cancelled" && self.status.is_none()
    }

    pub fn is_auth(&self) -> bool {
        matches!(self.status, Some(401 | 403))
    }
}

impl std::fmt::Display for PipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(status) => write!(f, "pipe http {status}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for PipeError {}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default)]
pub struct PipeCompletion {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub finish_reason: Option<String>,
}

impl PipeCompletion {
    /// Grok's turn loop treats a clean stop with no payload as a finished
    /// attempt. A 200 stream that carried nothing is retried like 429.
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
            && self.reasoning.trim().is_empty()
            && self.tool_calls.is_empty()
    }
}

pub struct PipeClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl PipeClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: hi_ai::inference_http_client_for_socket(None),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn set_api_key(&mut self, api_key: impl Into<String>) {
        self.api_key = api_key.into();
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    pub fn has_api_key(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    pub async fn list_models(&self) -> Result<Vec<ServedModel>> {
        let url = format!("{}/models", self.base_url);
        let response = hi_ai::agent_http_client_quick()
            .get(&url)
            .bearer_auth(&self.api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .context("listing Pipe models")?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!(PipeError {
                status: Some(status.as_u16()),
                message: body,
                retryable: status.is_server_error(),
                retry_after: None,
            });
        }
        parse_models(&body)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolSpec],
        max_tokens: u32,
        reasoning_effort: Option<ReasoningEffort>,
        on_event: &mut dyn FnMut(StreamDelta),
        cancel: &TurnCancellation,
    ) -> Result<PipeCompletion> {
        let body = build_body(model, messages, tools, max_tokens, reasoning_effort);
        let url = format!("{}/chat/completions", self.base_url);
        let mut attempts = 0u32;
        loop {
            if cancel.is_cancelled() {
                return Err(PipeError::cancelled().into());
            }
            let request = self
                .http
                .post(&url)
                .bearer_auth(&self.api_key)
                .header("Accept", "text/event-stream")
                .json(&body);
            let response = match request.send().await {
                Ok(response) => response,
                Err(err) if attempts < 1 && (err.is_connect() || err.is_timeout()) => {
                    attempts += 1;
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                Err(err) => return Err(err).context("pipe chat request"),
            };
            let status = response.status();
            if status == StatusCode::TOO_MANY_REQUESTS && attempts < 3 {
                let retry_after = retry_after_seconds(response.headers());
                attempts += 1;
                tokio::time::sleep(Duration::from_secs(retry_after.unwrap_or(1).min(8))).await;
                continue;
            }
            if status.is_server_error() && attempts < 2 {
                attempts += 1;
                tokio::time::sleep(Duration::from_millis(250 * 2u64.pow(attempts))).await;
                continue;
            }
            if !status.is_success() {
                let retry_after = retry_after_seconds(response.headers());
                let message = response.text().await.unwrap_or_default();
                return Err(PipeError {
                    status: Some(status.as_u16()),
                    message,
                    retryable: status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS,
                    retry_after,
                }
                .into());
            }
            let completion = read_sse(response, on_event, cancel).await?;
            if completion.is_empty() && attempts < 3 {
                attempts += 1;
                tokio::time::sleep(Duration::from_millis(250)).await;
                continue;
            }
            return Ok(completion);
        }
    }
}

#[derive(Debug)]
pub enum StreamDelta {
    Text(String),
    Reasoning(String),
}

fn retry_after_seconds(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
}

fn build_body(
    model: &str,
    messages: &[Message],
    tools: &[ToolSpec],
    max_tokens: u32,
    reasoning_effort: Option<ReasoningEffort>,
) -> Value {
    let uses_tools = !tools.is_empty();
    let mut body = json!({
        "model": model,
        "messages": to_openai_messages(messages),
        "stream": true,
        "stream_options": { "include_usage": true },
        "max_tokens": max_tokens,
        "metadata": {
            "endpoint_name": "pipenetworkai",
            "request_type": if uses_tools { "agent_tool_invocation" } else { "code_generation" },
            "selected_agent_model": model,
            "max_output_tokens": max_tokens,
        }
    });
    if uses_tools {
        body["metadata"]["agent_turn_kind"] = json!("root_agent_turn");
        body["tools"] = json!(
            tools
                .iter()
                .map(|spec| json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    }
                }))
                .collect::<Vec<_>>()
        );
        body["tool_choice"] = json!("auto");
    }
    if is_deepseek_model(model) {
        // DeepSeek-V4 thinking is off unless the request enables it. Grok shows
        // thinking blocks by default; match that so Ctrl+E has a body to expand.
        body["thinking"] = json!({ "type": "enabled" });
    }
    if let Some(effort) = reasoning_effort {
        body["reasoning_effort"] = json!(reasoning_wire_value(model, effort));
    }
    body
}

fn is_deepseek_model(model: &str) -> bool {
    model.to_ascii_lowercase().contains("deepseek")
}

/// Pipe's DeepSeek routes accept the gateway `low`/`high` surface; other
/// models get the OpenAI-style level names grok-build uses.
fn reasoning_wire_value(model: &str, effort: ReasoningEffort) -> &'static str {
    if model.to_ascii_lowercase().contains("deepseek") {
        match effort {
            ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
            ReasoningEffort::Medium | ReasoningEffort::High | ReasoningEffort::Xhigh => "high",
        }
    } else {
        effort.as_str()
    }
}

fn to_openai_messages(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    for message in messages {
        match message.role {
            Role::System => out.push(json!({ "role": "system", "content": message.text() })),
            Role::User => {
                let has_image = message
                    .content
                    .iter()
                    .any(|block| matches!(block, Content::Image { .. }));
                if has_image {
                    let mut parts = Vec::new();
                    for block in &message.content {
                        match block {
                            Content::Image { data, media_type } => parts.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{media_type};base64,{data}"),
                                },
                            })),
                            Content::Text(text) if !text.is_empty() => {
                                parts.push(json!({ "type": "text", "text": text }));
                            }
                            _ => {}
                        }
                    }
                    if parts.is_empty() {
                        parts.push(json!({ "type": "text", "text": message.text() }));
                    }
                    out.push(json!({ "role": "user", "content": parts }));
                } else {
                    out.push(json!({ "role": "user", "content": message.text() }));
                }
            }
            Role::Assistant => {
                let mut thinking = String::new();
                let mut text = String::new();
                let mut tool_calls = Vec::new();
                for block in &message.content {
                    match block {
                        Content::Text(t) => text.push_str(t),
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
                        _ => {}
                    }
                }
                let mut msg = if text.is_empty() && !tool_calls.is_empty() {
                    json!({ "role": "assistant", "content": null })
                } else {
                    json!({ "role": "assistant", "content": text })
                };
                if !thinking.is_empty() || !tool_calls.is_empty() {
                    msg["reasoning_content"] = json!(if thinking.is_empty() {
                        "Inspected the requested files."
                    } else {
                        thinking.as_str()
                    });
                }
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = json!(tool_calls);
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

fn parse_models(body: &str) -> Result<Vec<ServedModel>> {
    let value: Value = serde_json::from_str(body).context("parsing /models")?;
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| value.as_array())
        .context("Pipe /models missing data")?;
    Ok(data
        .iter()
        .filter_map(|item| {
            let id = item.get("id").and_then(Value::as_str)?.to_string();
            Some(ServedModel {
                id,
                context_window: item
                    .get("context_window")
                    .or_else(|| item.get("context_length"))
                    .and_then(Value::as_u64)
                    .map(|n| n as u32),
                max_output_tokens: item
                    .get("max_output_tokens")
                    .and_then(Value::as_u64)
                    .map(|n| n as u32),
                price: None,
                provider_label: item
                    .get("owned_by")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                status: None,
                available: true,
                availability_reason: None,
                capabilities: Vec::new(),
            })
        })
        .collect())
}

#[derive(Default)]
struct PartialTool {
    id: String,
    name: String,
    arguments: String,
}

fn sse_idle_timeout() -> Duration {
    std::env::var("HI_PIPE_SSE_IDLE_SECS")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .filter(|seconds: &u64| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(90))
}

async fn read_sse(
    response: reqwest::Response,
    on_event: &mut dyn FnMut(StreamDelta),
    cancel: &TurnCancellation,
) -> Result<PipeCompletion> {
    let mut stream = response.bytes_stream().eventsource();
    let mut completion = PipeCompletion::default();
    let mut tools: BTreeMap<usize, PartialTool> = BTreeMap::new();
    let cancelled = cancel.cancelled();
    tokio::pin!(cancelled);
    // Grok does not wait forever on a silent 200 stream. After cargo/test
    // timeouts the next completion can stall with no SSE bytes.
    let idle = sse_idle_timeout();
    loop {
        let event = tokio::select! {
            _ = &mut cancelled => return Err(PipeError::cancelled().into()),
            event = tokio::time::timeout(idle, stream.next()) => match event {
                Ok(Some(Ok(event))) => event,
                Ok(Some(Err(err))) => return Err(anyhow!("pipe sse: {err}")),
                Ok(None) => break,
                Err(_) => break,
            }
        };
        if event.data.trim() == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };
        if let Some(usage) = chunk.get("usage") {
            completion.usage = parse_usage(usage);
        }
        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        else {
            continue;
        };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            completion.finish_reason = Some(reason.to_string());
        }
        let Some(delta) = choice.get("delta").or_else(|| choice.get("message")) else {
            continue;
        };
        apply_text_delta(&mut completion, on_event, delta.get("content"));
        apply_reasoning_delta(
            &mut completion,
            on_event,
            delta
                .get("reasoning_content")
                .or_else(|| delta.get("reasoning")),
        );
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let entry = tools.entry(index).or_default();
                if let Some(id) = call.get("id").and_then(Value::as_str) {
                    entry.id.push_str(id);
                }
                let function = call.get("function");
                if let Some(name) = function.and_then(|f| f.get("name")).and_then(Value::as_str) {
                    entry.name.push_str(name);
                }
                if let Some(arguments) = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                {
                    entry.arguments.push_str(arguments);
                }
            }
        }
    }
    completion.tool_calls = tools
        .into_values()
        .map(|partial| ToolCall {
            id: if partial.id.is_empty() {
                format!("call_{}", uuid::Uuid::new_v4().simple())
            } else {
                partial.id
            },
            name: partial.name,
            arguments: partial.arguments,
        })
        .collect();
    Ok(completion)
}

fn apply_text_delta(
    completion: &mut PipeCompletion,
    on_event: &mut dyn FnMut(StreamDelta),
    content: Option<&Value>,
) {
    let Some(content) = content else {
        return;
    };
    match content {
        Value::String(text) if !text.is_empty() => {
            completion.text.push_str(text);
            on_event(StreamDelta::Text(text.clone()));
        }
        Value::Array(parts) => {
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("text");
                let Some(text) = part_text(part) else {
                    continue;
                };
                if matches!(kind, "thinking" | "reasoning") {
                    completion.reasoning.push_str(&text);
                    on_event(StreamDelta::Reasoning(text));
                } else {
                    completion.text.push_str(&text);
                    on_event(StreamDelta::Text(text));
                }
            }
        }
        _ => {}
    }
}

fn apply_reasoning_delta(
    completion: &mut PipeCompletion,
    on_event: &mut dyn FnMut(StreamDelta),
    reasoning: Option<&Value>,
) {
    let Some(text) = reasoning.and_then(value_text) else {
        return;
    };
    completion.reasoning.push_str(&text);
    on_event(StreamDelta::Reasoning(text));
}

fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Object(_) => part_text(value),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part_text(part) {
                    out.push_str(&text);
                }
            }
            if out.is_empty() { None } else { Some(out) }
        }
        _ => None,
    }
}

fn part_text(part: &Value) -> Option<String> {
    part.get("text")
        .or_else(|| part.get("thinking"))
        .or_else(|| part.get("reasoning"))
        .or_else(|| part.get("content"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
        .or_else(|| {
            part.as_str()
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        })
}

fn parse_usage(value: &Value) -> Usage {
    let input = value
        .get("prompt_tokens")
        .or_else(|| value.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = value
        .get("completion_tokens")
        .or_else(|| value.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = value
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input_tokens: input,
        output_tokens: output,
        cache_read_tokens: cache_read,
        input_includes_cache: true,
        context_occupancy: input,
        estimated: input == 0 && output == 0,
        ..Usage::default()
    }
}

pub fn default_base_url() -> String {
    match std::env::var("PIPENETWORK_API_BASE") {
        Ok(value) => {
            let value = value.trim().trim_end_matches('/').to_string();
            if value.is_empty() {
                DEFAULT_BASE_URL.to_string()
            } else if value.ends_with("/v1") {
                value
            } else {
                format!("{value}/v1")
            }
        }
        Err(_) => DEFAULT_BASE_URL.to_string(),
    }
}

#[cfg(test)]
pub mod test_support {
    use std::collections::VecDeque;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[derive(Clone)]
    pub enum Scripted {
        Sse(Vec<String>),
        Http {
            status: u16,
            body: String,
            retry_after: Option<u64>,
        },
    }

    pub struct MockPipe {
        pub url: String,
        pub bodies: Arc<Mutex<Vec<String>>>,
    }

    impl MockPipe {
        pub fn new(scripts: Vec<Scripted>) -> Option<Self> {
            let listener = match TcpListener::bind("127.0.0.1:0") {
                Ok(listener) => listener,
                Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return None,
                Err(err) => panic!("bind mock pipe: {err}"),
            };
            let url = format!("http://{}", listener.local_addr().unwrap());
            let bodies = Arc::new(Mutex::new(Vec::new()));
            let scripts = Arc::new(Mutex::new(VecDeque::from(scripts)));
            let thread_bodies = bodies.clone();
            let thread_scripts = scripts.clone();
            thread::spawn(move || {
                while let Ok((mut stream, _)) = listener.accept() {
                    let mut buf = vec![0u8; 64 * 1024];
                    let n = stream.read(&mut buf).unwrap_or(0);
                    let request = String::from_utf8_lossy(&buf[..n]);
                    if let Some(idx) = request.find("\r\n\r\n") {
                        thread_bodies
                            .lock()
                            .unwrap()
                            .push(request[idx + 4..].to_string());
                    }
                    let script =
                        thread_scripts
                            .lock()
                            .unwrap()
                            .pop_front()
                            .unwrap_or(Scripted::Http {
                                status: 500,
                                body: "no script".into(),
                                retry_after: None,
                            });
                    let _ = stream.write_all(&response_bytes(&script));
                }
            });
            Some(Self { url, bodies })
        }
    }

    fn response_bytes(script: &Scripted) -> Vec<u8> {
        match script {
            Scripted::Sse(chunks) => {
                let mut body = String::new();
                for chunk in chunks {
                    body.push_str("data: ");
                    body.push_str(chunk);
                    body.push_str("\n\n");
                }
                body.push_str("data: [DONE]\n\n");
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .into_bytes()
            }
            Scripted::Http {
                status,
                body,
                retry_after,
            } => {
                let reason = if *status == 429 {
                    "Too Many Requests"
                } else if *status == 401 {
                    "Unauthorized"
                } else {
                    "Error"
                };
                let retry = retry_after
                    .map(|secs| format!("Retry-After: {secs}\r\n"))
                    .unwrap_or_default();
                format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n{retry}Connection: close\r\n\r\n{body}",
                    body.len()
                )
                .into_bytes()
            }
        }
    }

    pub fn text_chunk(text: &str) -> String {
        serde_json::json!({
            "choices": [{ "index": 0, "delta": { "content": text } }]
        })
        .to_string()
    }

    pub fn tool_chunk(index: usize, id: &str, name: &str, arguments: &str) -> String {
        serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": index,
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": arguments }
                    }]
                }
            }]
        })
        .to_string()
    }

    pub fn usage_chunk(prompt: u64, completion: u64) -> String {
        serde_json::json!({
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }],
            "usage": { "prompt_tokens": prompt, "completion_tokens": completion }
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{MockPipe, Scripted, text_chunk, tool_chunk, usage_chunk};
    use super::*;
    use hi_ai::{Content, Message};

    #[tokio::test]
    async fn streams_text_and_usage() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("hello "),
            text_chunk("world"),
            usage_chunk(3, 2),
        ])]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let mut events = Vec::new();
        let completion = client
            .stream(
                DEFAULT_MODEL,
                &[Message::user("hi")],
                &[],
                128,
                None,
                &mut |delta| events.push(delta),
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        assert_eq!(completion.text, "hello world");
        assert_eq!(completion.usage.output_tokens, 2);
        assert!(matches!(events.first(), Some(StreamDelta::Text(_))));
        let body = server.bodies.lock().unwrap()[0].clone();
        assert!(body.contains("endpoint_name"));
        assert!(body.contains("pipenetworkai"));
        assert!(!body.contains("reasoning_effort"));
        assert!(
            body.contains("\"thinking\":{\"type\":\"enabled\"}")
                || body.contains("\"type\":\"enabled\""),
            "deepseek default enables thinking so the TUI has a thought block: {body}"
        );
    }

    #[tokio::test]
    async fn streams_reasoning_content() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            serde_json::json!({
                "choices": [{ "delta": { "reasoning_content": "inspect " } }]
            })
            .to_string(),
            serde_json::json!({
                "choices": [{ "delta": { "reasoning": { "text": "the file" } } }]
            })
            .to_string(),
            text_chunk("done"),
            usage_chunk(3, 2),
        ])]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let mut events = Vec::new();
        let completion = client
            .stream(
                DEFAULT_MODEL,
                &[Message::user("hi")],
                &[],
                128,
                None,
                &mut |delta| events.push(delta),
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        assert_eq!(completion.reasoning, "inspect the file");
        assert!(
            events
                .iter()
                .any(|delta| matches!(delta, StreamDelta::Reasoning(text) if text == "inspect ")),
            "{events:?}"
        );
    }

    #[tokio::test]
    async fn non_deepseek_omits_thinking_object() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("ok"),
            usage_chunk(1, 1),
        ])]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        client
            .stream(
                "pipe/kimi-k2.5",
                &[Message::user("hi")],
                &[],
                128,
                None,
                &mut |_| {},
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        let body = server.bodies.lock().unwrap()[0].clone();
        assert!(
            !body.contains("\"thinking\""),
            "non-deepseek models should not send a thinking envelope: {body}"
        );
    }

    #[tokio::test]
    async fn request_sends_reasoning_effort() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("ok"),
            usage_chunk(1, 1),
        ])]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        client
            .stream(
                "pipe/kimi-k2.5",
                &[Message::user("hi")],
                &[],
                128,
                Some(ReasoningEffort::Xhigh),
                &mut |_| {},
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        let body = server.bodies.lock().unwrap()[0].clone();
        assert!(body.contains("\"reasoning_effort\":\"xhigh\""), "{body}");
    }

    #[tokio::test]
    async fn deepseek_on_pipe_maps_effort_to_gateway_low_high() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("ok"),
            usage_chunk(1, 1),
        ])]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        client
            .stream(
                DEFAULT_MODEL,
                &[Message::user("hi")],
                &[],
                128,
                Some(ReasoningEffort::Xhigh),
                &mut |_| {},
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        let body = server.bodies.lock().unwrap()[0].clone();
        assert!(body.contains("\"reasoning_effort\":\"high\""), "{body}");
    }

    #[tokio::test]
    async fn reassembles_split_tool_calls() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            tool_chunk(0, "call_1", "read", ""),
            tool_chunk(0, "", "", "{\"path\":\"a.rs\"}"),
            usage_chunk(1, 1),
        ])]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let completion = client
            .stream(
                DEFAULT_MODEL,
                &[Message::user("read it")],
                &[],
                128,
                None,
                &mut |_| {},
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        assert_eq!(completion.tool_calls.len(), 1);
        assert_eq!(completion.tool_calls[0].name, "read");
        assert_eq!(completion.tool_calls[0].arguments, "{\"path\":\"a.rs\"}");
    }

    #[tokio::test]
    async fn empty_stream_retries_the_same_request() {
        let Some(server) = MockPipe::new(vec![
            Scripted::Sse(vec![usage_chunk(2, 0)]),
            Scripted::Sse(vec![text_chunk("ok"), usage_chunk(2, 1)]),
        ]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let completion = client
            .stream(
                DEFAULT_MODEL,
                &[Message::user("hi")],
                &[],
                128,
                None,
                &mut |_| {},
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        assert_eq!(completion.text, "ok");
        assert_eq!(server.bodies.lock().unwrap().len(), 2);
    }

    #[test]
    fn user_image_uses_multipart_content() {
        let messages = vec![Message::user_with_image("see this", "abc", "image/png")];
        let encoded = super::to_openai_messages(&messages);
        assert_eq!(encoded[0]["content"][0]["type"], "image_url");
        assert!(
            encoded[0]["content"][0]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn tool_call_assistant_uses_null_content() {
        let messages = vec![Message::assistant(vec![Content::ToolCall {
            id: "call_1".into(),
            name: "read".into(),
            arguments: "{\"path\":\"a.rs\"}".into(),
        }])];
        let encoded = super::to_openai_messages(&messages);
        assert_eq!(encoded[0]["content"], serde_json::Value::Null);
        assert!(encoded[0]["tool_calls"].is_array());
    }

    #[tokio::test]
    async fn retries_429_then_succeeds() {
        let Some(server) = MockPipe::new(vec![
            Scripted::Http {
                status: 429,
                body: "slow down".into(),
                retry_after: Some(0),
            },
            Scripted::Sse(vec![text_chunk("ok"), usage_chunk(1, 1)]),
        ]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let completion = client
            .stream(
                DEFAULT_MODEL,
                &[Message::user("hi")],
                &[],
                128,
                None,
                &mut |_| {},
                &TurnCancellation::new(),
            )
            .await
            .unwrap();
        assert_eq!(completion.text, "ok");
    }

    #[tokio::test]
    async fn auth_error_does_not_succeed() {
        let Some(server) = MockPipe::new(vec![Scripted::Http {
            status: 401,
            body: "bad key".into(),
            retry_after: None,
        }]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let err = client
            .stream(
                DEFAULT_MODEL,
                &[Message::user("hi")],
                &[],
                128,
                None,
                &mut |_| {},
                &TurnCancellation::new(),
            )
            .await
            .unwrap_err();
        let pipe = err.downcast_ref::<PipeError>().unwrap();
        assert!(pipe.is_auth());
    }
}
