//! Pipe Network chat-completions client.
//!
//! OpenAI-compatible wire format against `https://api.pipenetwork.ai/v1`.
//! No compatibility probing, strict schemas, or multi-provider fallbacks.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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
/// How many times a TLS handshake or dead keep-alive may fail before the
/// turn surfaces the error. A second prompt used to succeed because it
/// opened a fresh connection; the first request did not retry that class.
const MAX_TRANSPORT_RETRIES: u32 = 3;

/// Cap body/header `retry_after_seconds` so a bad payload cannot stall a turn.
const MAX_RETRY_AFTER_SECS: u64 = 30;

#[derive(Debug)]
pub struct PipeError {
    pub status: Option<u16>,
    pub message: String,
    pub retryable: bool,
    pub retry_after: Option<u64>,
}

impl PipeError {
    pub(crate) fn cancelled() -> Self {
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
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

    pub fn is_truncated(&self) -> bool {
        matches!(
            self.finish_reason.as_deref(),
            Some("length" | "max_tokens" | "max_output_tokens")
        )
    }
}

pub struct PipeClient {
    pub(crate) managed: crate::managed::ManagedState,
    pub(crate) http: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) sse_timing: SseTiming,
}

/// How long a model stream may go without bytes or without output.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SseTiming {
    /// No bytes at all, not even keep-alive comments: the connection is dead.
    pub idle: Duration,
    /// How often the reader wakes to check the timers.
    pub tick: Duration,
    /// Tell the user why the turn is waiting once output has been absent
    /// this long, and again at the same interval.
    pub queue_notice: Duration,
    /// Give up on a request the server keeps open without output.
    pub queue_cap: Duration,
}

impl SseTiming {
    fn from_env() -> Self {
        let idle = sse_idle_timeout();
        Self {
            idle,
            tick: Duration::from_secs(1).min(idle),
            queue_notice: Duration::from_secs(15),
            queue_cap: env_secs("HI_PIPE_QUEUE_WAIT_SECS").unwrap_or(Duration::from_secs(600)),
        }
    }
}

/// The server held a request open with keep-alives and never started a reply.
#[derive(Debug)]
pub(crate) struct PipeQueueTimeout {
    pub waited: Duration,
}

impl std::fmt::Display for PipeQueueTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pipe kept the request queued for {}s without starting a reply (HI_PIPE_QUEUE_WAIT_SECS)",
            self.waited.as_secs()
        )
    }
}

impl std::error::Error for PipeQueueTimeout {}

impl PipeClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            managed: Default::default(),
            http: hi_ai::inference_http_client_for_socket(None),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            sse_timing: SseTiming::from_env(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_sse_timing(mut self, timing: SseTiming) -> Self {
        self.sse_timing = timing;
        self
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
        if model == "pipe/auto" {
            return self.stream_managed(body, tools, on_event, cancel).await;
        }
        let url = format!("{}/chat/completions", self.base_url);
        let idle = self.sse_timing.idle;
        let mut attempts = 0u32;
        let mut transport_attempts = 0u32;
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
            // The inference client has no total or read timeout so long
            // streams survive, which leaves the wait for response headers
            // unbounded. Bound it with the same idle budget as the SSE body
            // and keep Esc working while it waits.
            let sent = tokio::select! {
                _ = cancel.cancelled() => return Err(PipeError::cancelled().into()),
                sent = tokio::time::timeout(idle, request.send()) => sent,
            };
            let response = match sent {
                Ok(Ok(response)) => response,
                Ok(Err(err))
                    if transport_attempts < MAX_TRANSPORT_RETRIES
                        && pipe_transport_retryable(&err) =>
                {
                    transport_attempts += 1;
                    on_event(StreamDelta::Status(format!(
                        "pipe request failed ({}); retrying",
                        brief_transport_error(&err.to_string())
                    )));
                    sleep_unless_cancelled(backoff_unit() * transport_attempts, cancel).await?;
                    continue;
                }
                Ok(Err(err)) => return Err(err).context("pipe chat request"),
                Err(_elapsed) if attempts < 1 => {
                    attempts += 1;
                    on_event(StreamDelta::Status(format!(
                        "pipe sent no response headers for {}s; retrying",
                        idle.as_secs()
                    )));
                    continue;
                }
                Err(_elapsed) => {
                    return Err(PipeError {
                        status: None,
                        message: format!(
                            "pipe sent no response headers within {}s (HI_PIPE_SSE_IDLE_SECS)",
                            idle.as_secs()
                        ),
                        retryable: true,
                        retry_after: None,
                    }
                    .into());
                }
            };
            let status = response.status();
            if status == StatusCode::TOO_MANY_REQUESTS && attempts < 3 {
                let header = retry_after_seconds(response.headers());
                let body = response.text().await.unwrap_or_default();
                let wait = header
                    .or_else(|| retry_after_from_json_body(&body))
                    .unwrap_or(1)
                    .min(MAX_RETRY_AFTER_SECS);
                attempts += 1;
                on_event(StreamDelta::Status(format!(
                    "pipe rate limited; retrying in {wait}s"
                )));
                sleep_unless_cancelled(Duration::from_secs(wait), cancel).await?;
                continue;
            }
            if status.is_server_error() && attempts < 2 {
                attempts += 1;
                sleep_unless_cancelled(Duration::from_millis(250 * 2u64.pow(attempts)), cancel)
                    .await?;
                continue;
            }
            if !status.is_success() {
                let header = retry_after_seconds(response.headers());
                let message = response.text().await.unwrap_or_default();
                let retry_after = header.or_else(|| retry_after_from_json_body(&message));
                return Err(PipeError {
                    status: Some(status.as_u16()),
                    message,
                    retryable: status.is_server_error() || status == StatusCode::TOO_MANY_REQUESTS,
                    retry_after,
                }
                .into());
            }
            let completion = match read_sse(response, on_event, cancel, self.sse_timing).await {
                Ok(completion) => completion,
                Err(err)
                    if transport_attempts < MAX_TRANSPORT_RETRIES
                        && stream_error_retryable(&err) =>
                {
                    transport_attempts += 1;
                    on_event(StreamDelta::Status(format!(
                        "pipe stream failed ({}); retrying",
                        brief_transport_error(&err.to_string())
                    )));
                    sleep_unless_cancelled(backoff_unit() * transport_attempts, cancel).await?;
                    continue;
                }
                Err(err) => return Err(err),
            };
            if completion.is_empty() && attempts < 3 {
                attempts += 1;
                on_event(StreamDelta::Status(
                    "pipe stream carried nothing; retrying".into(),
                ));
                sleep_unless_cancelled(Duration::from_millis(250), cancel).await?;
                continue;
            }
            return Ok(completion);
        }
    }
}

/// Retry backoff that still honors Esc. A plain `sleep` here made a 30s
/// `Retry-After` uncancellable.
async fn sleep_unless_cancelled(wait: Duration, cancel: &TurnCancellation) -> Result<()> {
    tokio::select! {
        _ = cancel.cancelled() => Err(PipeError::cancelled().into()),
        _ = tokio::time::sleep(wait) => Ok(()),
    }
}

#[derive(Debug)]
pub enum StreamDelta {
    Status(String),
    Text(String),
    Reasoning(String),
}

/// Connect, timeout, and mid-send failures. Reqwest reports a TLS handshake
/// on a reused connection as `is_request`, not `is_connect`, so a connect-only
/// check lets `tls handshake eof` fail the turn.
/// What the turn loop should do with a failed model stream before giving up.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum StreamRetry {
    /// Sleep, then send the same request again.
    Again { wait: Duration, status: String },
    /// The prompt does not fit. Stub old tool output, then send again.
    Shrink,
    /// Pipe rejected the model's tool call. Send again with this hint so the
    /// model produces a different, smaller call.
    Repair { hint: &'static str, status: String },
}

/// Sent with the retry after Pipe rejects the model's tool call.
pub(crate) const TOOL_PROTOCOL_REPAIR_HINT: &str = "\
Your previous reply was rejected by the model gateway because its tool call \
was not valid: usually a call so long it was cut off, or arguments that did \
not match the tool schema. Make one small, well-formed tool call now. Split \
large changes into several `edit` calls with short `old_string`/`new_string` \
values, and do not paste whole files into a single call.";

/// Pipe's rejection of a reply whose tool call it could not parse or
/// validate. Resending the same request usually produces the same call.
pub(crate) fn tool_protocol_rejection(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "tool protocol",
        "tool-enabled chat output must be valid json",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

/// Consecutive failed model requests the turn absorbs before it surfaces the
/// error. The count resets after every successful round.
pub(crate) const MAX_STREAM_RECOVERIES: u32 = 5;

/// Retry pacing. Scaled down under test so the retry paths run in
/// milliseconds instead of seconds.
pub(crate) fn backoff_unit() -> Duration {
    if cfg!(test) {
        Duration::from_millis(5)
    } else {
        Duration::from_millis(250)
    }
}

/// Decide whether a failed Pipe stream can be recovered inside the turn.
///
/// Everything is retried except a cancel, bad credentials, and a 4xx that a
/// resend cannot fix. Matching on error text missed each new wording (a TLS
/// handshake, a body decode error, an in-stream error frame), and every miss
/// ended a turn that a second prompt then finished. A context-length
/// rejection shrinks history once instead.
pub(crate) fn recover_pipe_stream_error(
    err: &anyhow::Error,
    retries: u32,
    shrunk: bool,
) -> Option<StreamRetry> {
    let pipe = err.downcast_ref::<PipeError>();
    if pipe.is_some_and(|pipe| pipe.is_cancelled() || pipe.is_auth()) {
        return None;
    }
    // One fresh request after a long queue wait; a second would only queue
    // the turn for another full cap.
    if err.downcast_ref::<PipeQueueTimeout>().is_some() && retries >= 1 {
        return None;
    }
    let text = format!("{err:#}");
    if context_overflow(&text) || pipe.is_some_and(|pipe| pipe.status == Some(413)) {
        return (!shrunk).then_some(StreamRetry::Shrink);
    }
    if retries >= MAX_STREAM_RECOVERIES {
        return None;
    }
    // Checked before the permanent-4xx rule: Pipe sends this as a 400 too,
    // and a different request is exactly what fixes it.
    if tool_protocol_rejection(&text) {
        return Some(StreamRetry::Repair {
            hint: TOOL_PROTOCOL_REPAIR_HINT,
            status: format!(
                "Pipe rejected the model's tool call ({}); asking for a smaller call, retry {}/{MAX_STREAM_RECOVERIES}",
                brief_transport_error(&text),
                retries + 1
            ),
        });
    }
    if pipe.is_some_and(permanent_client_error) {
        return None;
    }
    let backoff = backoff_unit() * 4 * 2u32.pow(retries.min(4));
    let wait = pipe
        .and_then(|pipe| pipe.retry_after)
        .map(|after| Duration::from_secs(after.min(MAX_RETRY_AFTER_SECS)).max(backoff))
        .unwrap_or(backoff);
    Some(StreamRetry::Again {
        wait,
        status: format!(
            "model request failed ({}); retry {}/{MAX_STREAM_RECOVERIES}",
            brief_transport_error(&text),
            retries + 1
        ),
    })
}

/// A 4xx the same request will hit again: bad shape, unknown model, payment.
fn permanent_client_error(pipe: &PipeError) -> bool {
    !pipe.retryable && matches!(pipe.status, Some(400 | 402 | 404 | 405 | 406 | 410 | 422))
}

fn context_overflow(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "context_length_exceeded",
        "context length",
        "maximum context length",
        "maximum prompt length",
        "prompt length",
        "too many tokens",
        "request too large",
        "exceeds the context",
        "exceed context",
        "input exceeds",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn sse_error_message(chunk: &Value) -> Option<String> {
    let err = chunk.get("error")?;
    if let Some(message) = err.as_str() {
        let message = message.trim();
        if message.is_empty() {
            return None;
        }
        return Some(message.to_string());
    }
    err.get("message")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .map(str::to_string)
        .or_else(|| {
            let rendered = err.to_string();
            let rendered = rendered.trim();
            if rendered.is_empty() || rendered == "null" {
                None
            } else {
                Some(rendered.to_string())
            }
        })
}

fn stream_error_retryable(err: &anyhow::Error) -> bool {
    if looks_like_transport_loss(&err.to_string()) {
        return true;
    }
    err.downcast_ref::<PipeError>()
        .is_some_and(|pipe| pipe.retryable && !pipe.is_cancelled() && !pipe.is_auth())
}

fn pipe_transport_retryable(err: &reqwest::Error) -> bool {
    err.is_timeout()
        || err.is_connect()
        || err.is_request()
        || looks_like_transport_loss(&err.to_string())
}

fn looks_like_transport_loss(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "tls",
        "handshake",
        "connection reset",
        "broken pipe",
        "incomplete message",
        "connection closed",
        "error trying to connect",
        "error sending request",
        "unexpected eof",
        "unexpected end of file",
        "stream closed",
        "connection aborted",
        "reset by peer",
        "overloaded",
        "temporarily unavailable",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn brief_transport_error(text: &str) -> String {
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::new();
    for ch in one_line.chars() {
        if out.chars().count() >= 160 {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn retry_after_seconds(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse().ok())
}

/// Pipe capacity 429s put `retry_after_seconds` in JSON and often omit Retry-After.
fn retry_after_from_json_body(body: &str) -> Option<u64> {
    let trimmed = body.trim();
    let json = if trimmed.starts_with('{') {
        trimmed
    } else {
        trimmed.find('{').map(|idx| &trimmed[idx..])?
    };
    let value: Value = serde_json::from_str(json).ok()?;
    fn find(v: &Value) -> Option<u64> {
        match v {
            Value::Object(map) => {
                if let Some(n) = map.get("retry_after_seconds").and_then(json_u64) {
                    return Some(n);
                }
                map.values().find_map(find)
            }
            Value::Array(items) => items.iter().find_map(find),
            _ => None,
        }
    }
    find(&value)
}

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|n| u64::try_from(n).ok()))
}

pub(crate) fn build_body(
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
pub(crate) fn reasoning_wire_value(model: &str, effort: ReasoningEffort) -> &'static str {
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
                    .pointer("/managed_coding/limits/context_tokens")
                    .or_else(|| item.get("context_window"))
                    .or_else(|| item.get("context_length"))
                    .and_then(Value::as_u64)
                    .map(|n| n as u32),
                max_output_tokens: item
                    .pointer("/managed_coding/limits/output_tokens")
                    .or_else(|| item.get("max_output_tokens"))
                    .and_then(Value::as_u64)
                    .map(|n| n as u32),
                price: None,
                provider_label: item
                    .get("owned_by")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                status: None,
                available: item
                    .get("managed_coding")
                    .is_none_or(|c| c["available"] == true),
                availability_reason: item
                    .pointer("/managed_coding/reason")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                capabilities: item
                    .pointer("/managed_coding/capabilities")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default(),
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
    env_secs("HI_PIPE_SSE_IDLE_SECS").unwrap_or(Duration::from_secs(90))
}

fn env_secs(name: &str) -> Option<Duration> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .filter(|seconds: &u64| *seconds > 0)
        .map(Duration::from_secs)
}

/// Grows whenever a stream event adds output. Keep-alives and empty deltas
/// leave it unchanged.
fn output_marker(completion: &PipeCompletion, tools: &BTreeMap<usize, PartialTool>) -> usize {
    completion.text.len()
        + completion.reasoning.len()
        + usize::from(completion.finish_reason.is_some())
        + tools
            .values()
            .map(|tool| tool.id.len() + tool.name.len() + tool.arguments.len())
            .sum::<usize>()
}

async fn read_sse(
    response: reqwest::Response,
    on_event: &mut dyn FnMut(StreamDelta),
    cancel: &TurnCancellation,
    timing: SseTiming,
) -> Result<PipeCompletion> {
    // The SSE parser drops `: keep-alive` comments without yielding an
    // event. DeepSeek sends them while a request waits for a slot, so timing
    // events alone called a queued request dead after 90s, resent it, and
    // put it back at the end of the queue. Liveness is measured on raw body
    // bytes; output progress is measured separately.
    let opened = Instant::now();
    let last_bytes_ms = Arc::new(AtomicU64::new(0));
    let bytes_seen = Arc::clone(&last_bytes_ms);
    let mut stream = response
        .bytes_stream()
        .inspect(move |chunk| {
            if chunk.is_ok() {
                bytes_seen.store(opened.elapsed().as_millis() as u64, Ordering::Relaxed);
            }
        })
        .eventsource();
    let mut completion = PipeCompletion::default();
    let mut tools: BTreeMap<usize, PartialTool> = BTreeMap::new();
    let cancelled = cancel.cancelled();
    tokio::pin!(cancelled);
    let idle = timing.idle;
    let mut last_output = Instant::now();
    let mut last_marker = 0usize;
    let mut last_notice = Instant::now();
    loop {
        let marker = output_marker(&completion, &tools);
        if marker != last_marker {
            last_marker = marker;
            last_output = Instant::now();
        }
        let event = tokio::select! {
            _ = &mut cancelled => return Err(PipeError::cancelled().into()),
            event = tokio::time::timeout(timing.tick, stream.next()) => match event {
                Ok(Some(Ok(event))) => event,
                Ok(Some(Err(err))) => {
                    // The round is not committed until the stream ends, so a
                    // drop after partial reasoning or text is still safe to
                    // resend. Treating it as fatal ended drme turns that a
                    // second prompt then finished.
                    let message = format!("pipe stream dropped: {err}");
                    let overflow = context_overflow(&message);
                    return Err(PipeError {
                        status: None,
                        message,
                        retryable: !overflow,
                        retry_after: None,
                    }
                    .into());
                }
                Ok(None) => break,
                Err(_tick) => {
                    let quiet = opened
                        .elapsed()
                        .saturating_sub(Duration::from_millis(last_bytes_ms.load(Ordering::Relaxed)));
                    if quiet >= idle {
                        on_event(StreamDelta::Status(format!(
                            "pipe stream went silent for {}s; using what arrived",
                            idle.as_secs()
                        )));
                        break;
                    }
                    let waiting = last_output.elapsed();
                    if waiting >= timing.queue_cap {
                        return Err(PipeQueueTimeout { waited: waiting }.into());
                    }
                    if waiting >= timing.queue_notice && last_notice.elapsed() >= timing.queue_notice {
                        last_notice = Instant::now();
                        on_event(StreamDelta::Status(if last_marker == 0 {
                            format!(
                                "pipe is queuing the request (no output for {}s); keeping its place",
                                waiting.as_secs()
                            )
                        } else {
                            format!(
                                "pipe reply paused (no output for {}s); connection is alive, waiting",
                                waiting.as_secs()
                            )
                        }));
                    }
                    continue;
                }
            }
        };
        if event.data.trim() == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };
        if let Some(message) = sse_error_message(&chunk) {
            // An error frame with no payload yet is the provider asking us to
            // try again (or to shrink, when the prompt does not fit). Leaving
            // it on the floor made the turn die; the next user prompt worked.
            if completion.is_empty() && tools.is_empty() {
                // A rejected tool call is the model's output, not the
                // transport: resending the same request unchanged repeats it.
                // The turn retries with a repair hint instead.
                let retryable = !context_overflow(&message) && !tool_protocol_rejection(&message);
                return Err(PipeError {
                    status: None,
                    message: message.clone(),
                    retryable,
                    retry_after: retry_after_from_json_body(&message),
                }
                .into());
            }
            return Err(anyhow!("pipe sse: {message}"));
        }
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

pub(crate) fn parse_usage(value: &Value) -> Usage {
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
        /// Accept the request and hold the socket open without ever writing
        /// a byte: a server that connected but never sends headers.
        Silent,
        /// Send these SSE frames, then close the socket before the declared
        /// Content-Length: a connection that drops mid-reply.
        Truncated(Vec<String>),
        /// Send headers, then each raw body piece after its delay in
        /// milliseconds, then close. Models keep-alive comments and pauses.
        Paced(Vec<(u64, String)>),
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
                // Silent connections stay open here so the client sees a
                // live socket with nothing on it, not a reset.
                let mut held = Vec::new();
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
                    if matches!(script, Scripted::Silent) {
                        held.push(stream);
                        continue;
                    }
                    if let Scripted::Paced(parts) = script {
                        thread::spawn(move || {
                            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";
                            if stream.write_all(head.as_bytes()).is_err() {
                                return;
                            }
                            for (delay_ms, raw) in parts {
                                thread::sleep(std::time::Duration::from_millis(delay_ms));
                                if stream.write_all(raw.as_bytes()).is_err()
                                    || stream.flush().is_err()
                                {
                                    return;
                                }
                            }
                        });
                        continue;
                    }
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
            Scripted::Silent | Scripted::Paced(_) => Vec::new(),
            Scripted::Truncated(chunks) => {
                let mut body = String::new();
                for chunk in chunks {
                    body.push_str("data: ");
                    body.push_str(chunk);
                    body.push_str("\n\n");
                }
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len() + 4096
                )
                .into_bytes()
            }
        }
    }

    pub fn reasoning_chunk(text: &str) -> String {
        serde_json::json!({
            "choices": [{ "index": 0, "delta": { "reasoning_content": text } }]
        })
        .to_string()
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
        usage_chunk_with_reason(prompt, completion, "stop")
    }

    pub fn usage_chunk_with_reason(prompt: u64, completion: u64, finish_reason: &str) -> String {
        serde_json::json!({
            "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason }],
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

    #[test]
    fn tls_handshake_text_is_a_transport_loss() {
        assert!(looks_like_transport_loss(
            "error sending request for url (https://api.pipenetwork.ai/v1/chat/completions): error trying to connect: tls handshake eof"
        ));
        assert!(looks_like_transport_loss(
            "received fatal alert: handshake_failure"
        ));
        assert!(!looks_like_transport_loss(
            "pipe returned 400: unknown model"
        ));
        assert!(context_overflow(
            "pipe http 400: This model's maximum context length is 131072 tokens"
        ));
        assert!(!context_overflow("pipe http 400: unknown model"));
        let brief = brief_transport_error("tls   handshake\n eof");
        assert_eq!(brief, "tls handshake eof");
    }

    #[test]
    fn context_overflow_shrinks_once_and_transient_errors_retry() {
        let overflow = anyhow!(PipeError {
            status: Some(400),
            message: "maximum context length is 131072 tokens".into(),
            retryable: false,
            retry_after: None,
        });
        assert_eq!(
            recover_pipe_stream_error(&overflow, 0, false),
            Some(StreamRetry::Shrink)
        );
        assert_eq!(recover_pipe_stream_error(&overflow, 0, true), None);

        let gateway = anyhow!(PipeError {
            status: Some(503),
            message: "bad gateway".into(),
            retryable: true,
            retry_after: Some(2),
        });
        match recover_pipe_stream_error(&gateway, 0, false) {
            Some(StreamRetry::Again { wait, status }) => {
                assert_eq!(wait, Duration::from_secs(2));
                assert!(status.contains("bad gateway"), "{status}");
            }
            other => panic!("expected a retry, got {other:?}"),
        }
        assert_eq!(
            recover_pipe_stream_error(&gateway, MAX_STREAM_RECOVERIES, false),
            None
        );

        let unknown_model = anyhow!(PipeError {
            status: Some(400),
            message: "unknown model pipe/nope".into(),
            retryable: false,
            retry_after: None,
        });
        assert_eq!(recover_pipe_stream_error(&unknown_model, 0, false), None);

        let rejected_call = anyhow!(PipeError {
            status: Some(400),
            message: r#"{"error":"model output did not satisfy the tool protocol"}"#.into(),
            retryable: false,
            retry_after: None,
        });
        assert!(matches!(
            recover_pipe_stream_error(&rejected_call, 0, false),
            Some(StreamRetry::Repair {
                hint: TOOL_PROTOCOL_REPAIR_HINT,
                ..
            })
        ));
        assert_eq!(
            recover_pipe_stream_error(&rejected_call, MAX_STREAM_RECOVERIES, false),
            None
        );

        let body_decode = anyhow!(PipeError {
            status: None,
            message: "pipe stream dropped: Transport error: error decoding response body".into(),
            retryable: true,
            retry_after: None,
        });
        assert!(matches!(
            recover_pipe_stream_error(&body_decode, 0, false),
            Some(StreamRetry::Again { .. })
        ));

        let auth = anyhow!(PipeError {
            status: Some(401),
            message: "unauthorized".into(),
            retryable: false,
            retry_after: None,
        });
        assert_eq!(recover_pipe_stream_error(&auth, 0, false), None);

        let tls = anyhow!("pipe chat request: error trying to connect: tls handshake eof");
        assert!(matches!(
            recover_pipe_stream_error(&tls, 0, false),
            Some(StreamRetry::Again { .. })
        ));
    }

    #[test]
    fn sse_error_frame_is_read_as_a_message() {
        let chunk = json!({"error":{"message":"The service is busy, please try again later"}});
        assert_eq!(
            sse_error_message(&chunk).as_deref(),
            Some("The service is busy, please try again later")
        );
        assert!(sse_error_message(&json!({"choices":[]})).is_none());
    }

    fn fast_timing() -> SseTiming {
        SseTiming {
            idle: Duration::from_millis(300),
            tick: Duration::from_millis(20),
            queue_notice: Duration::from_millis(100),
            queue_cap: Duration::from_secs(10),
        }
    }

    async fn stream_with(client: &PipeClient) -> (anyhow::Result<PipeCompletion>, Vec<String>) {
        let mut statuses = Vec::new();
        let result = client
            .stream(
                "pipe/test",
                &[Message::user("continue")],
                &[],
                64,
                None,
                &mut |delta| {
                    if let StreamDelta::Status(text) = delta {
                        statuses.push(text);
                    }
                },
                &TurnCancellation::new(),
            )
            .await;
        (result, statuses)
    }

    /// Live ~/drme: one model round took 12 minutes. DeepSeek sends
    /// `: keep-alive` comments while a request waits for a slot; hi called
    /// that silence, resent after 90s, and went back to the end of the queue.
    #[tokio::test]
    async fn keep_alive_comments_hold_a_queued_request() {
        let mut parts: Vec<(u64, String)> = (0..12)
            .map(|_| (60, ": keep-alive\n\n".to_string()))
            .collect();
        parts.push((
            0,
            format!("data: {}\n\n", text_chunk("served after the queue")),
        ));
        parts.push((0, "data: [DONE]\n\n".to_string()));
        let Some(server) = MockPipe::new(vec![
            Scripted::Paced(parts),
            Scripted::Sse(vec![text_chunk("second request")]),
        ]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test").with_sse_timing(fast_timing());
        let (result, statuses) = stream_with(&client).await;
        let completion = result.expect("a queued request completes");
        assert!(
            completion.text.contains("served after the queue"),
            "{completion:?} {statuses:?}"
        );
        assert_eq!(
            server.bodies.lock().unwrap().len(),
            1,
            "a queued request must keep its place, not be resent: {statuses:?}"
        );
        assert!(
            statuses.iter().any(|status| status.contains("queuing")),
            "the wait must be visible: {statuses:?}"
        );
    }

    #[tokio::test]
    async fn silence_without_keep_alives_is_still_retried() {
        let Some(server) = MockPipe::new(vec![
            Scripted::Paced(vec![(
                1_500,
                format!("data: {}\n\n", text_chunk("too late")),
            )]),
            Scripted::Sse(vec![text_chunk("second request")]),
        ]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test").with_sse_timing(fast_timing());
        let (result, statuses) = stream_with(&client).await;
        let completion = result.expect("the retry succeeds");
        assert!(
            completion.text.contains("second request"),
            "{completion:?} {statuses:?}"
        );
        assert_eq!(server.bodies.lock().unwrap().len(), 2, "{statuses:?}");
        assert!(
            statuses.iter().any(|status| status.contains("went silent")),
            "{statuses:?}"
        );
    }

    #[tokio::test]
    async fn queue_cap_ends_a_request_that_never_starts() {
        let parts: Vec<(u64, String)> = (0..40)
            .map(|_| (50, ": keep-alive\n\n".to_string()))
            .collect();
        let Some(server) = MockPipe::new(vec![Scripted::Paced(parts)]) else {
            return;
        };
        let timing = SseTiming {
            queue_cap: Duration::from_millis(400),
            ..fast_timing()
        };
        let client = PipeClient::new(server.url.clone(), "pk_test").with_sse_timing(timing);
        let (result, statuses) = stream_with(&client).await;
        let err = result.expect_err("a request that never starts must end");
        assert!(
            err.downcast_ref::<PipeQueueTimeout>().is_some(),
            "{err:#} {statuses:?}"
        );
        assert_eq!(
            server.bodies.lock().unwrap().len(),
            1,
            "the client must not requeue by itself"
        );
        assert!(matches!(
            recover_pipe_stream_error(&err, 0, false),
            Some(StreamRetry::Again { .. })
        ));
        assert_eq!(recover_pipe_stream_error(&err, 1, false), None);
    }

    #[tokio::test]
    async fn empty_sse_error_is_retried_until_the_stream_succeeds() {
        let Some(server) = MockPipe::new(vec![
            Scripted::Sse(vec![
                r#"{"error":{"message":"unexpected EOF while reading stream"}}"#.into(),
            ]),
            Scripted::Sse(vec![text_chunk("recovered")]),
        ]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let mut statuses = Vec::new();
        let completion = client
            .stream(
                "pipe/test",
                &[Message::user("continue")],
                &[],
                64,
                None,
                &mut |delta| {
                    if let StreamDelta::Status(text) = delta {
                        statuses.push(text);
                    }
                },
                &TurnCancellation::new(),
            )
            .await
            .expect("retry must recover an empty stream error");
        assert!(
            completion.text.contains("recovered"),
            "completion: {completion:?}"
        );
        assert!(
            statuses.iter().any(|status| status.contains("retrying")),
            "expected a retry status, got {statuses:?}"
        );
        assert_eq!(
            server.bodies.lock().unwrap().len(),
            2,
            "the failed frame must be followed by one retry"
        );
    }

    /// A server that connects and then sends nothing used to park the turn
    /// in `send().await` with Esc ignored. Cancel must end that wait.
    #[tokio::test]
    async fn esc_ends_wait_for_response_headers() {
        let Some(server) = MockPipe::new(vec![Scripted::Silent, Scripted::Silent]) else {
            return;
        };
        let client = PipeClient::new(server.url.clone(), "pk_test");
        let cancel = TurnCancellation::new();
        let trigger = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            trigger.cancel();
        });
        let started = std::time::Instant::now();
        let result = client
            .stream(
                "pipe/test",
                &[Message::user("hi")],
                &[],
                64,
                None,
                &mut |_| {},
                &cancel,
            )
            .await;
        let err = result.expect_err("a silent server must not yield a completion");
        assert!(
            err.downcast_ref::<PipeError>()
                .is_some_and(PipeError::is_cancelled),
            "{err:#}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "cancel took {:?}",
            started.elapsed()
        );
    }

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
    fn parse_models_reads_advertised_output_cap() {
        let models = super::parse_models(
            r#"{"data":[{"id":"pipe/deepseek-v4-flash-0731","context_window":212992,"max_output_tokens":16384}]}"#,
        )
        .unwrap();
        assert_eq!(models[0].id, "pipe/deepseek-v4-flash-0731");
        assert_eq!(models[0].context_window, Some(212_992));
        assert_eq!(models[0].max_output_tokens, Some(16_384));
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

    #[test]
    fn retry_after_from_json_body_reads_pipe_capacity_payload() {
        let body = r#"{"code":"capacity_unavailable","retry_after_seconds":5,"retryable":true}"#;
        assert_eq!(super::retry_after_from_json_body(body), Some(5));
        let wrapped = r#"pipe http 429: {"error":{"retry_after_seconds":7}}"#;
        assert_eq!(super::retry_after_from_json_body(wrapped), Some(7));
        assert_eq!(super::retry_after_from_json_body("slow down"), None);
    }

    #[tokio::test]
    async fn retries_429_using_json_retry_after_without_header() {
        let Some(server) = MockPipe::new(vec![
            Scripted::Http {
                status: 429,
                body: r#"{"code":"capacity_unavailable","retry_after_seconds":0,"retryable":true}"#
                    .into(),
                retry_after: None,
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
        assert_eq!(server.bodies.lock().unwrap().len(), 2);
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
