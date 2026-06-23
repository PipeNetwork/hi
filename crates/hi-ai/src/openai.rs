//! OpenAI Chat Completions adapter.
//!
//! Covers OpenRouter, terminaili.com, and local servers (Ollama, llama.cpp,
//! LM Studio, vLLM) — they differ only by base URL and API key.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures_util::{Stream, StreamExt};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::provider::{Provider, ProviderError, ProviderErrorKind};
use crate::types::{
    ChatRequest, CompatMode, Completion, Content, Message, Role, StreamEvent, ToolMode, Usage,
};

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
        let attempts = request_attempts(&request);
        let mut last_error: Option<ProviderError> = None;
        let mut resp = None;
        let mut idx = 0;
        while idx < attempts.len() {
            let attempt = attempts[idx];
            let body = build_body(&request, attempt);
            let response = crate::http::send_with_retry(
                self.http.post(&url).bearer_auth(&self.api_key).json(&body),
            )
            .await
            .context("request to model endpoint failed")?;

            if response.status().is_success() {
                if let Some(status) = attempt.status {
                    sink(StreamEvent::Status(status.into()));
                }
                resp = Some(response);
                break;
            }

            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let kind = classify_http_error(status, &text);
            last_error = Some(ProviderError::new(
                kind,
                format!("API error {status}: {text}"),
            ));
            if request.profile.compat == CompatMode::Strict {
                break;
            }
            // Degrade toward the attempt that actually addresses this error — a
            // tool rejection jumps straight to chat-only rather than first dropping
            // usage streaming. `None` means nothing more will help: surface it.
            match next_degraded_attempt(&attempts, idx, kind, &text) {
                Some(next) => idx = next,
                None => break,
            }
        }
        let Some(resp) = resp else {
            return Err(last_error
                .unwrap_or_else(|| {
                    ProviderError::new(ProviderErrorKind::Other, "request failed before streaming")
                })
                .into());
        };

        // `debug_tap` optionally echoes the raw wire bytes when HI_DEBUG_STREAM
        // is set. Reduce the stream to its SSE `data` strings so the collection
        // loop is provider-agnostic and unit-testable.
        let stream = crate::http::debug_tap(resp.bytes_stream())
            .eventsource()
            .map(|res| res.map(|event| event.data).context("error reading stream"));
        let completion = collect_completion(
            Box::pin(stream),
            crate::http::stream_idle_timeout(),
            crate::http::stream_stall_timeout(),
            sink,
        )
        .await
        .map_err(classify_stream_error)?;
        if completion.content.is_empty() {
            return Err(ProviderError::new(
                ProviderErrorKind::EmptyCompletion,
                "model returned an empty completion",
            )
            .into());
        }
        Ok(completion)
    }

    async fn list_models(&self) -> Result<Vec<crate::provider::ServedModel>> {
        let url = format!("{}/models", self.base_url);
        crate::http::fetch_models(self.http.get(&url).bearer_auth(&self.api_key)).await
    }
}

fn classify_stream_error(err: anyhow::Error) -> anyhow::Error {
    let text = err.to_string();
    let kind = if text.contains("no output") {
        ProviderErrorKind::StreamTimeout
    } else if text.contains("error reading stream") || text.contains("malformed SSE JSON chunk") {
        ProviderErrorKind::MalformedStream
    } else {
        ProviderErrorKind::Other
    };
    ProviderError::new(kind, text).into()
}

/// Once the model reports a `finish_reason`, it has stopped generating; we wait
/// only this long for the trailing usage chunk / `[DONE]` before ending the
/// turn. Without it, a provider that emits `finish_reason` but never sends
/// `[DONE]` (nor closes the socket) would wedge the turn until the much longer
/// idle timeout expires — a completed answer left spinning for ~2 minutes.
const FINISH_GRACE: Duration = Duration::from_secs(3);

/// Drain an OpenAI SSE stream (already reduced to `data` strings) into a
/// [`Completion`], forwarding text/reasoning/tool tokens to `sink`.
///
/// Three deadlines bound the wait, all measured from the last real token
/// (content/reasoning/tool — keep-alive heartbeats carry none, so they don't
/// reset the clock):
/// - **cold start** (no output yet): wait up to `idle` — a request can be
///   legitimately queued before the first token.
/// - **stall** (output flowed, then stopped without a finish signal): end after
///   the shorter `stall`, returning what we have. A multi-second gap *between*
///   tokens means the stream has effectively ended; this stops a provider that
///   streams a full answer then holds the socket open from hanging the turn.
/// - **finish** (`finish_reason` seen): a short [`FINISH_GRACE`] catches any
///   trailing usage chunk, then stop even if `[DONE]` never comes.
async fn collect_completion<S>(
    mut stream: S,
    idle: Duration,
    stall: Duration,
    sink: &mut (dyn FnMut(StreamEvent) + Send),
) -> Result<Completion>
where
    S: Stream<Item = Result<String>> + Unpin,
{
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCallBuilder> = Vec::new();
    let mut completion = Completion::default();
    let mut last_progress = Instant::now();
    let mut finished: Option<Instant> = None;
    let mut progressed = false;

    loop {
        let budget = match finished {
            Some(at) => FINISH_GRACE.saturating_sub(at.elapsed()),
            None if progressed => stall.saturating_sub(last_progress.elapsed()),
            None => idle.saturating_sub(last_progress.elapsed()),
        };
        let data = match tokio::time::timeout(budget, stream.next()).await {
            Ok(Some(data)) => data?,
            Ok(None) => break,
            // Past finish_reason the answer is complete; don't let a provider that
            // omits `[DONE]` (or never closes the socket) hang a finished turn.
            Err(_) if finished.is_some() => break,
            // Output flowed then stalled with no finish signal: treat what we have
            // as the response rather than waiting out the full cold-start timeout.
            Err(_) if progressed => break,
            Err(_) => bail!(
                "model produced no output for {}s — the provider streamed only \
                 keep-alive heartbeats. It may be overloaded or the model unavailable; \
                 try again, or switch with /model.",
                idle.as_secs()
            ),
        };
        if data == "[DONE]" {
            break;
        }
        let chunk = serde_json::from_str::<ChatChunk>(&data).with_context(|| {
            format!(
                "malformed SSE JSON chunk: {}",
                data.chars().take(160).collect::<String>()
            )
        })?;

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
                progressed = true;
            }
            if let Some(content) = delta.content
                && !content.is_empty()
            {
                text.push_str(&content);
                sink(StreamEvent::Text(content));
                last_progress = Instant::now();
                progressed = true;
            }
            if let Some(deltas) = delta.tool_calls {
                last_progress = Instant::now();
                progressed = true;
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
                finished.get_or_insert_with(Instant::now);
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

#[derive(Clone, Copy)]
struct RequestAttempt {
    include_usage: bool,
    include_tools: bool,
    status: Option<&'static str>,
}

/// Given the attempt that just failed (at `current`) and its error, the index of
/// the next attempt to try — the one whose degradation actually addresses this
/// error — or `None` to stop and surface the error. A tool rejection jumps
/// straight to the chat-only attempt instead of first dropping usage streaming;
/// in Required tool mode there is no chat-only attempt, so the error surfaces.
fn next_degraded_attempt(
    attempts: &[RequestAttempt],
    current: usize,
    kind: ProviderErrorKind,
    text: &str,
) -> Option<usize> {
    let cur = attempts[current];
    let after = current + 1;
    // Usage streaming rejected → retry without it (keeping tools).
    if cur.include_usage && mentions(text, &["stream_options", "include_usage"]) {
        return attempts[after..]
            .iter()
            .position(|a| !a.include_usage)
            .map(|i| after + i);
    }
    // Tool schema rejected → drop to chat-only.
    if cur.include_tools
        && matches!(
            kind,
            ProviderErrorKind::UnsupportedTools | ProviderErrorKind::UnsupportedRequestShape
        )
    {
        return attempts[after..]
            .iter()
            .position(|a| !a.include_tools)
            .map(|i| after + i);
    }
    // A persistent outage that already survived transport retries: try the next
    // (degraded) shape in case the 5xx was really a request-shape problem.
    if matches!(kind, ProviderErrorKind::Outage) && after < attempts.len() {
        return Some(after);
    }
    None
}

fn request_attempts(request: &ChatRequest) -> Vec<RequestAttempt> {
    let include_usage = request.profile.stream_usage.unwrap_or(true);
    let include_tools =
        !request.tools.is_empty() && request.profile.tool_mode != ToolMode::ChatOnly;
    let mut attempts = vec![RequestAttempt {
        include_usage,
        include_tools,
        status: None,
    }];
    if request.profile.compat == CompatMode::Strict {
        return attempts;
    }
    if include_usage {
        attempts.push(RequestAttempt {
            include_usage: false,
            include_tools,
            status: Some(
                "compat: provider rejected stream_options; retried without usage streaming",
            ),
        });
    }
    if include_tools && request.profile.tool_mode != ToolMode::Required {
        attempts.push(RequestAttempt {
            include_usage: false,
            include_tools: false,
            status: Some(
                "compat: provider rejected tool calling; degraded to chat-only for this request",
            ),
        });
    }
    attempts
}

fn classify_http_error(status: StatusCode, text: &str) -> ProviderErrorKind {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => ProviderErrorKind::Auth,
        StatusCode::TOO_MANY_REQUESTS => ProviderErrorKind::RateLimit,
        s if s.is_server_error() => ProviderErrorKind::Outage,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
            if mentions(
                text,
                &[
                    "maximum allowed size",
                    "input exceeds",
                    "context length",
                    "context_length_exceeded",
                    "too many tokens",
                    "request too large",
                ],
            ) {
                ProviderErrorKind::RequestTooLarge
            } else if mentions(text, &["tool", "function_call", "function"]) {
                ProviderErrorKind::UnsupportedTools
            } else {
                ProviderErrorKind::UnsupportedRequestShape
            }
        }
        _ => ProviderErrorKind::Other,
    }
}

fn mentions(text: &str, needles: &[&str]) -> bool {
    let lower = text.to_ascii_lowercase();
    needles.iter().any(|needle| lower.contains(needle))
}

fn build_body(request: &ChatRequest, attempt: RequestAttempt) -> Value {
    let mut messages = to_openai_messages(&request.messages);
    if !attempt.include_tools && !request.tools.is_empty() {
        messages.push(json!({
            "role": "user",
            "content": "Tool calling is unavailable for this request because the provider rejected the tool schema. If the user asked for file edits, shell commands, or other workspace changes, say that this cannot be completed with the current provider/tool mode instead of claiming changes were made.",
        }));
    }
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
        "max_tokens": request.max_tokens,
    });
    if attempt.include_usage {
        body["stream_options"] = json!({ "include_usage": true });
    }
    if attempt.include_tools {
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
        if request.profile.tool_mode == ToolMode::Required {
            body["tool_choice"] = json!("required");
        }
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }
    if let Some(frequency_penalty) = request.frequency_penalty {
        body["frequency_penalty"] = json!(frequency_penalty);
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
    use std::time::Duration;

    use futures_util::{StreamExt, stream};

    use super::{Result, build_body, collect_completion, request_attempts, to_openai_messages};
    use crate::provider::{Provider, ProviderErrorKind, provider_error_kind};
    use crate::test_support::{FakeOpenAiServer, Response, sse_text};
    use crate::types::{
        ChatRequest, Content, Message, RequestProfile, StreamEvent, ToolMode, ToolSpec,
    };

    /// A stream of SSE `data` strings that never ends (no `[DONE]`, socket stays
    /// open) — `pending()` models a provider that just stops talking.
    fn never_ending(events: Vec<&str>) -> impl futures_util::Stream<Item = Result<String>> + Unpin {
        let items: Vec<Result<String>> = events.into_iter().map(|s| Ok(s.to_string())).collect();
        stream::iter(items).chain(stream::pending())
    }

    const STALL: Duration = Duration::from_secs(15);
    const IDLE: Duration = Duration::from_secs(120);

    #[tokio::test(start_paused = true)]
    async fn stops_after_finish_reason_without_done() {
        // The bug: terminaili sends `finish_reason` then neither `[DONE]` nor a
        // socket close, so a finished answer used to spin until the 120s idle
        // timeout. Now the short finish-grace ends the turn promptly.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"the answer"},"finish_reason":null}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert_eq!(completion.stop_reason.as_deref(), Some("stop"));
        assert!(
            matches!(completion.content.first(), Some(Content::Text(t)) if t == "the answer"),
            "text is still collected: {:?}",
            completion.content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn trailing_usage_chunk_after_finish_is_captured() {
        // Providers send the usage chunk right after `finish_reason`; the grace
        // window must be long enough to catch it.
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert_eq!(completion.usage.input_tokens, 10);
        assert_eq!(completion.usage.output_tokens, 2);
    }

    #[tokio::test(start_paused = true)]
    async fn returns_partial_when_stream_stalls_after_content() {
        // The reported bug: the model streams a full answer, then the provider
        // sends no finish_reason / [DONE] and holds the socket open. Once output
        // has flowed, the short stall window ends the turn with what we have,
        // instead of spinning out the full cold-start idle timeout.
        let stream = never_ending(vec![r#"{"choices":[{"delta":{"content":"the answer"}}]}"#]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        assert!(
            matches!(completion.content.first(), Some(Content::Text(t)) if t == "the answer"),
            "the streamed answer is returned: {:?}",
            completion.content
        );
    }

    #[tokio::test(start_paused = true)]
    async fn bails_when_no_output_at_all() {
        // Nothing ever streamed (cold-start): the long idle timeout trips and we
        // surface the "no output" error rather than returning an empty success.
        let stream = never_ending(vec![]);
        let mut sink = |_: StreamEvent| {};
        let err = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no output"), "got: {err}");
    }

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

    #[test]
    fn request_body_can_omit_stream_options() {
        let req = crate::types::ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: Default::default(),
        };

        let normal = build_body(&req, request_attempts(&req)[0]);
        assert_eq!(normal["stream_options"]["include_usage"], true);

        let fallback = build_body(&req, request_attempts(&req)[1]);
        assert!(fallback.get("stream_options").is_none());
        assert_eq!(fallback["stream"], true);
    }

    #[test]
    fn request_body_carries_recovery_sampling() {
        // top_p/frequency_penalty (set by recovery sampling on a retry) reach the
        // wire; absent fields stay absent so the provider default applies.
        let mut req = crate::types::ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile: Default::default(),
        };
        let plain = build_body(&req, request_attempts(&req)[0]);
        assert!(plain.get("top_p").is_none(), "omitted when unset");
        assert!(plain.get("frequency_penalty").is_none());

        req.temperature = Some(0.9);
        req.top_p = Some(0.95);
        req.frequency_penalty = Some(0.4);
        let hot = build_body(&req, request_attempts(&req)[0]);
        // f32 → JSON f64 isn't exact (0.9f32 ≈ 0.89999996), so compare with tolerance.
        let near = |v: &serde_json::Value, want: f64| (v.as_f64().unwrap() - want).abs() < 1e-6;
        assert!(
            near(&hot["temperature"], 0.9),
            "temperature: {}",
            hot["temperature"]
        );
        assert!(near(&hot["top_p"], 0.95), "top_p: {}", hot["top_p"]);
        assert!(
            near(&hot["frequency_penalty"], 0.4),
            "frequency_penalty: {}",
            hot["frequency_penalty"]
        );
    }

    #[tokio::test]
    async fn fake_server_rejects_stream_options_then_succeeds() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(400, r#"{"error":"stream_options unsupported"}"#),
            Response::sse(sse_text("ok")),
        ]) else {
            return;
        };
        let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
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
            statuses.iter().any(|s| s.contains("stream_options")),
            "{statuses:?}"
        );
        let bodies = server.bodies();
        assert!(bodies[0].contains("stream_options"));
        assert!(!bodies[1].contains("stream_options"));
    }

    #[tokio::test]
    async fn fake_server_rejects_tools_then_degrades_to_chat_only() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(400, r#"{"error":"tools unsupported"}"#),
            Response::sse(sse_text("cannot edit without tools")),
        ]) else {
            return;
        };
        let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
        let mut statuses = Vec::new();
        let mut sink = |event| {
            if let StreamEvent::Status(status) = event {
                statuses.push(status);
            }
        };
        let completion = provider
            .stream(request(vec![tool()], Default::default()), &mut sink)
            .await
            .unwrap();
        assert!(
            matches!(completion.content.first(), Some(Content::Text(t)) if t.contains("without tools"))
        );
        assert!(
            statuses.iter().any(|s| s.contains("tool calling")),
            "{statuses:?}"
        );
        let bodies = server.bodies();
        assert!(bodies[0].contains("\"tools\""));
        assert!(!bodies[1].contains("\"tools\""));
        assert!(bodies[1].contains("Tool calling is unavailable"));
    }

    #[tokio::test]
    async fn required_tool_mode_does_not_degrade() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"tools unsupported"}"#,
        )]) else {
            return;
        };
        let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
        let profile = RequestProfile {
            tool_mode: ToolMode::Required,
            ..Default::default()
        };
        let err = provider
            .stream(request(vec![tool()], profile), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::UnsupportedTools)
        );
        assert_eq!(server.bodies().len(), 1);
    }

    #[tokio::test]
    async fn auth_rate_limit_and_malformed_stream_are_classified() {
        for (status, kind) in [
            (401, ProviderErrorKind::Auth),
            (403, ProviderErrorKind::Auth),
            (429, ProviderErrorKind::RateLimit),
        ] {
            let Some(server) =
                FakeOpenAiServer::new(vec![Response::json(status, r#"{"error":"nope"}"#)])
            else {
                return;
            };
            let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
            let err = provider
                .stream(request(vec![], Default::default()), &mut |_| {})
                .await
                .unwrap_err();
            assert_eq!(provider_error_kind(&err), Some(kind), "status {status}");
        }

        let Some(server) = FakeOpenAiServer::new(vec![Response::sse("data: {not-json}\n\n")])
        else {
            return;
        };
        let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::MalformedStream)
        );
    }

    #[tokio::test]
    async fn request_too_large_400_is_classified() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::json(
            400,
            r#"{"error":"chat input exceeds the maximum allowed size of 131072 bytes","error_type":"invalid_request_error"}"#,
        )]) else {
            return;
        };
        let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
        let err = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(
            provider_error_kind(&err),
            Some(ProviderErrorKind::RequestTooLarge)
        );
    }

    #[tokio::test]
    async fn server_error_retries_then_succeeds() {
        let Some(server) = FakeOpenAiServer::new(vec![
            Response::json(500, r#"{"error":"temporary outage"}"#),
            Response::sse(sse_text("recovered")),
        ]) else {
            return;
        };
        let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "recovered"));
        assert_eq!(server.bodies().len(), 2);
    }

    #[tokio::test]
    async fn fake_server_stream_can_finish_without_done() {
        let Some(server) = FakeOpenAiServer::new(vec![Response::sse(
            "data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\n",
        )]) else {
            return;
        };
        let provider = super::OpenAiProvider::new(server.url().to_string(), "test".into());
        let completion = provider
            .stream(request(vec![], Default::default()), &mut |_| {})
            .await
            .unwrap();
        assert!(matches!(completion.content.first(), Some(Content::Text(t)) if t == "done"));
    }

    #[tokio::test(start_paused = true)]
    async fn fragmented_tool_calls_are_reassembled() {
        let stream = never_ending(vec![
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"ba","arguments":"{\"com"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"sh","arguments":"mand\":\"echo hi\"}"}}]},"finish_reason":"tool_calls"}]}"#,
        ]);
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream, IDLE, STALL, &mut sink)
            .await
            .unwrap();
        let calls = completion.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].arguments, r#"{"command":"echo hi"}"#);
    }

    fn request(tools: Vec<ToolSpec>, profile: RequestProfile) -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("hi")],
            tools,
            max_tokens: 16,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            thinking_budget: None,
            profile,
        }
    }

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run shell command".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        }
    }
}
