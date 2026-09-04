//! Drain an xAI Responses SSE stream into a [`Completion`].

use anyhow::{Context, Result};
use futures_util::{Stream, StreamExt};
use serde_json::Value;

use crate::provider::{ProviderError, ProviderErrorKind};
use crate::types::{
    Completion, Content, StreamEvent, ToolCallChannel, Usage, estimate_completion_output_tokens,
};

const MAX_STREAM_TOOL_NAME_BYTES: usize = 256;
const MAX_STREAM_TOOL_ARGUMENT_BYTES: usize = 4 * 1024 * 1024;
/// Allow gateway-specific trailing usage without waiting indefinitely after a
/// terminal response when the server leaves its SSE connection open.
const POST_FINISH_USAGE_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

struct StreamAcc {
    text: String,
    reasoning: String,
    encrypted: Option<String>,
    refusal: String,
    calls: Vec<FunctionCallAcc>,
    tool_payload_bytes: usize,
    native_tool_calls: bool,
}

struct FunctionCallAcc {
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    metadata_emitted: bool,
}

impl FunctionCallAcc {
    fn finish(self, index: usize) -> Content {
        let id = if self.call_id.is_empty() {
            format!("call_hi_{}_{}", uuid::Uuid::new_v4().simple(), index)
        } else {
            self.call_id
        };
        Content::ToolCall {
            id,
            name: self.name,
            arguments: if self.arguments.is_empty() {
                "{}".into()
            } else {
                self.arguments
            },
        }
    }
}

/// Collect a Responses SSE stream. Event type is taken from the SSE `event`
/// field when present, otherwise from the JSON `type` field — xAI and the
/// OpenAI SDK both appear in the wild.
pub(crate) async fn collect_completion(
    mut stream: impl Stream<Item = Result<eventsource_stream::Event, anyhow::Error>> + Unpin,
    sink: &mut (dyn FnMut(StreamEvent) + Send),
) -> Result<Completion> {
    let mut acc = StreamAcc {
        text: String::new(),
        reasoning: String::new(),
        encrypted: None,
        refusal: String::new(),
        calls: Vec::new(),
        tool_payload_bytes: 0,
        native_tool_calls: false,
    };
    let mut completion = Completion::default();
    let mut progressed = false;
    let mut stream_complete = false;
    let mut usage_seen = false;
    let mut post_finish_deadline = None;

    loop {
        let next = if stream_complete {
            let deadline = *post_finish_deadline
                .get_or_insert_with(|| tokio::time::Instant::now() + POST_FINISH_USAGE_GRACE);
            match tokio::time::timeout_at(deadline, stream.next()).await {
                Ok(next) => next,
                Err(_) => break,
            }
        } else {
            stream.next().await
        };
        let Some(event) = next else { break };
        let event = match event {
            Ok(event) => event,
            Err(err) => {
                if stream_complete || progressed {
                    break;
                }
                return Err(err).context("error reading stream");
            }
        };
        if event.data.trim() == "[DONE]" {
            break;
        }
        let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
            continue;
        };
        if let Some(error) = data.get("error") {
            let message = error
                .as_str()
                .or_else(|| error.get("message").and_then(Value::as_str))
                .unwrap_or("unknown error");
            return Err(ProviderError::new(
                super::request::classify_http_error(
                    reqwest::StatusCode::BAD_REQUEST,
                    &error.to_string(),
                ),
                format!("xAI stream error: {message}"),
            )
            .into());
        }
        let event_type = event_type(&event.event, &data);
        match event_type {
            "response.output_text.delta" => {
                if let Some(delta) = delta_text(&data) {
                    acc.text.push_str(delta);
                    sink(StreamEvent::Text(delta.to_string()));
                    progressed = true;
                }
            }
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                if let Some(delta) = delta_text(&data) {
                    acc.reasoning.push_str(delta);
                    sink(StreamEvent::Reasoning(delta.to_string()));
                    progressed = true;
                }
            }
            "response.refusal.delta" => {
                if let Some(delta) = delta_text(&data) {
                    acc.refusal.push_str(delta);
                    progressed = true;
                }
            }
            "response.output_item.added" | "response.output_item.done" => {
                if let Some(item) = data.get("item") {
                    ingest_item(item, &mut acc)?;
                    if acc.native_tool_calls {
                        progressed = true;
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                acc.native_tool_calls = true;
                progressed = true;
                let delta = delta_text(&data).unwrap_or("");
                let index = call_index(&acc.calls, &data);
                let (id_delta, name_delta) = {
                    let call = locate_call_mut(&mut acc.calls, &data, &mut acc.tool_payload_bytes)?;
                    append_bounded(
                        &mut call.arguments,
                        delta,
                        MAX_STREAM_TOOL_ARGUMENT_BYTES,
                        &mut acc.tool_payload_bytes,
                    )?;
                    (
                        (!call.metadata_emitted)
                            .then(|| call.call_id.clone())
                            .filter(|value| !value.is_empty()),
                        (!call.metadata_emitted)
                            .then(|| call.name.clone())
                            .filter(|value| !value.is_empty()),
                    )
                };
                if let Ok(call) =
                    locate_call_mut(&mut acc.calls, &data, &mut acc.tool_payload_bytes)
                {
                    call.metadata_emitted = true;
                }
                sink(StreamEvent::ToolCallDelta {
                    index,
                    id_delta,
                    name_delta,
                    arguments_delta: delta.to_string(),
                });
            }
            "response.function_call_arguments.done" => {
                acc.native_tool_calls = true;
                progressed = true;
                if let Some(arguments) = data.get("arguments").and_then(Value::as_str) {
                    let call = locate_call_mut(&mut acc.calls, &data, &mut acc.tool_payload_bytes)?;
                    if call.arguments.is_empty() {
                        append_bounded(
                            &mut call.arguments,
                            arguments,
                            MAX_STREAM_TOOL_ARGUMENT_BYTES,
                            &mut acc.tool_payload_bytes,
                        )?;
                    }
                }
            }
            "response.completed" | "response.incomplete" => {
                stream_complete = true;
                if let Some(response) = data.get("response") {
                    ingest_completed(response, &mut acc, &mut completion, &mut usage_seen)?;
                    if event_type == "response.incomplete"
                        && let Some(reason) = response
                            .pointer("/incomplete_details/reason")
                            .and_then(Value::as_str)
                    {
                        // The agent's existing truncation recovery recognizes
                        // max_tokens, so preserve the provider's terminal cause
                        // instead of treating partial output as a normal answer.
                        completion.stop_reason = Some(
                            match reason {
                                "max_output_tokens" => "max_tokens",
                                reason => reason,
                            }
                            .to_string(),
                        );
                    }
                } else if let Some(usage) = data.get("usage") {
                    apply_usage(&mut completion, usage);
                    usage_seen = true;
                }
            }
            "response.failed" => {
                let message = data
                    .pointer("/response/error/message")
                    .and_then(Value::as_str)
                    .or_else(|| data.get("error").and_then(Value::as_str))
                    .unwrap_or("response failed");
                return Err(ProviderError::new(
                    ProviderErrorKind::Other,
                    format!("xAI response failed: {message}"),
                )
                .into());
            }
            _ => {
                if let Some(usage) = data.get("usage") {
                    apply_usage(&mut completion, usage);
                    usage_seen = true;
                }
            }
        }
        if stream_complete && usage_seen {
            break;
        }
    }

    if !acc.reasoning.is_empty() || acc.encrypted.is_some() {
        completion.content.push(Content::Thinking {
            text: acc.reasoning,
            signature: acc.encrypted,
        });
    }
    if !acc.text.is_empty() {
        completion.content.push(Content::Text(acc.text));
    }
    for (index, call) in acc.calls.into_iter().enumerate() {
        if !call.name.is_empty() || !call.call_id.is_empty() {
            completion.content.push(call.finish(index));
        }
    }
    completion.tool_call_channel = if acc.native_tool_calls {
        ToolCallChannel::Native
    } else {
        ToolCallChannel::None
    };
    if !acc.refusal.is_empty() {
        completion.refusal = Some(acc.refusal);
    }
    if completion.usage.output_tokens == 0 {
        completion.usage.output_tokens = estimate_completion_output_tokens(&completion.content);
        completion.usage.estimated = true;
    }
    Ok(completion)
}

fn call_index(calls: &[FunctionCallAcc], data: &Value) -> usize {
    let item_id = data.get("item_id").and_then(Value::as_str).unwrap_or("");
    if !item_id.is_empty()
        && let Some(index) = calls.iter().position(|call| call.item_id == item_id)
    {
        return index;
    }
    calls.len().saturating_sub(1)
}

fn event_type<'a>(sse_event: &'a str, data: &'a Value) -> &'a str {
    if !sse_event.is_empty() && sse_event != "message" {
        sse_event
    } else {
        data.get("type").and_then(Value::as_str).unwrap_or("")
    }
}

fn delta_text(data: &Value) -> Option<&str> {
    data.get("delta")
        .and_then(Value::as_str)
        .or_else(|| data.pointer("/delta/text").and_then(Value::as_str))
}

fn ingest_item(item: &Value, acc: &mut StreamAcc) -> Result<()> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => {
            acc.native_tool_calls = true;
            upsert_function_call(&mut acc.calls, item, &mut acc.tool_payload_bytes)?;
        }
        Some("reasoning") => {
            if let Some(enc) = item.get("encrypted_content").and_then(Value::as_str)
                && !enc.is_empty()
            {
                acc.encrypted = Some(enc.to_string());
            }
            if acc.reasoning.is_empty()
                && let Some(summary) = item.get("summary").and_then(Value::as_array)
            {
                for part in summary {
                    if let Some(chunk) = part.get("text").and_then(Value::as_str) {
                        acc.reasoning.push_str(chunk);
                    }
                }
            }
        }
        Some("message") if acc.text.is_empty() => {
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for part in content {
                    if part.get("type").and_then(Value::as_str) == Some("output_text")
                        && let Some(chunk) = part.get("text").and_then(Value::as_str)
                    {
                        acc.text.push_str(chunk);
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn ingest_completed(
    response: &Value,
    acc: &mut StreamAcc,
    completion: &mut Completion,
    usage_seen: &mut bool,
) -> Result<()> {
    if let Some(status) = response.get("status").and_then(Value::as_str) {
        completion.stop_reason = Some(status.to_string());
    }
    if let Some(usage) = response.get("usage") {
        apply_usage(completion, usage);
        *usage_seen = true;
    }
    if let Some(output) = response.get("output").and_then(Value::as_array) {
        for item in output {
            ingest_item(item, acc)?;
        }
    }
    Ok(())
}

fn upsert_function_call(
    calls: &mut Vec<FunctionCallAcc>,
    item: &Value,
    tool_payload_bytes: &mut usize,
) -> Result<(), anyhow::Error> {
    let item_id = item
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if name.len() > MAX_STREAM_TOOL_NAME_BYTES {
        return Err(tool_protocol("model exceeded the streamed tool-name size limit").into());
    }
    if let Some(existing) = calls.iter_mut().find(|call| {
        (!item_id.is_empty() && call.item_id == item_id)
            || (!call_id.is_empty() && call.call_id == call_id)
    }) {
        if existing.call_id.is_empty() {
            reserve_tool_payload(tool_payload_bytes, call_id.len())?;
            existing.call_id = call_id;
        }
        if existing.name.is_empty() {
            reserve_tool_payload(tool_payload_bytes, name.len())?;
            existing.name = name;
        }
        if existing.arguments.is_empty() && !arguments.is_empty() {
            append_bounded(
                &mut existing.arguments,
                &arguments,
                MAX_STREAM_TOOL_ARGUMENT_BYTES,
                tool_payload_bytes,
            )?;
        }
        return Ok(());
    }
    reserve_tool_payload(
        tool_payload_bytes,
        crate::tool_validation::TOOL_CALL_SLOT_OVERHEAD_BYTES
            .saturating_add(item_id.len())
            .saturating_add(call_id.len())
            .saturating_add(name.len()),
    )?;
    let mut acc = FunctionCallAcc {
        item_id,
        call_id,
        name,
        arguments: String::new(),
        metadata_emitted: false,
    };
    if !arguments.is_empty() {
        append_bounded(
            &mut acc.arguments,
            &arguments,
            MAX_STREAM_TOOL_ARGUMENT_BYTES,
            tool_payload_bytes,
        )?;
    }
    calls.push(acc);
    Ok(())
}

fn locate_call_mut<'a>(
    calls: &'a mut Vec<FunctionCallAcc>,
    data: &Value,
    tool_payload_bytes: &mut usize,
) -> Result<&'a mut FunctionCallAcc, anyhow::Error> {
    let item_id = data.get("item_id").and_then(Value::as_str).unwrap_or("");
    if !item_id.is_empty() {
        if let Some(index) = calls.iter().position(|call| call.item_id == item_id) {
            return Ok(&mut calls[index]);
        }
        push_empty_call(calls, item_id, tool_payload_bytes)?;
        return Ok(calls.last_mut().expect("just inserted"));
    }
    if calls.is_empty() {
        push_empty_call(calls, "", tool_payload_bytes)?;
    }
    Ok(calls.last_mut().expect("call slot exists"))
}

fn push_empty_call(
    calls: &mut Vec<FunctionCallAcc>,
    item_id: &str,
    tool_payload_bytes: &mut usize,
) -> Result<(), anyhow::Error> {
    reserve_tool_payload(
        tool_payload_bytes,
        crate::tool_validation::TOOL_CALL_SLOT_OVERHEAD_BYTES.saturating_add(item_id.len()),
    )?;
    calls.push(FunctionCallAcc {
        item_id: item_id.to_string(),
        call_id: String::new(),
        name: String::new(),
        arguments: String::new(),
        metadata_emitted: false,
    });
    Ok(())
}

fn append_bounded(
    current: &mut String,
    fragment: &str,
    per_call: usize,
    total: &mut usize,
) -> Result<(), anyhow::Error> {
    if fragment.is_empty() {
        return Ok(());
    }
    if current.len().saturating_add(fragment.len()) > per_call {
        return Err(tool_protocol("model exceeded the streamed tool-argument size limit").into());
    }
    reserve_tool_payload(total, fragment.len())?;
    current.push_str(fragment);
    Ok(())
}

fn reserve_tool_payload(total: &mut usize, additional: usize) -> Result<(), anyhow::Error> {
    if crate::tool_validation::try_reserve_tool_payload(total, additional) {
        Ok(())
    } else {
        Err(tool_protocol("model exceeded the streamed tool payload size limit").into())
    }
}

fn apply_usage(completion: &mut Completion, usage: &Value) {
    let input = usage
        .get("input_tokens")
        .or_else(|| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("output_tokens")
        .or_else(|| usage.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cached = usage
        .pointer("/input_tokens_details/cached_tokens")
        .or_else(|| usage.pointer("/prompt_tokens_details/cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    completion.usage = Usage {
        input_tokens: input,
        output_tokens: output,
        cache_read_tokens: cached,
        cache_creation_tokens: 0,
        input_includes_cache: true,
        context_occupancy: input,
        rate_limits: None,
        estimated: false,
    };
}

fn tool_protocol(message: &str) -> ProviderError {
    ProviderError::new(ProviderErrorKind::ToolProtocol, message).with_api_contract(
        Some("tool_protocol_error".to_string()),
        Some(true),
        None,
    )
}

pub(crate) fn classify_stream_error(err: anyhow::Error) -> ProviderError {
    if let Some(provider) = err.downcast_ref::<ProviderError>() {
        return provider.clone();
    }
    let text = err.to_string();
    let kind = if text.contains("went silent") {
        ProviderErrorKind::Outage
    } else if text.contains("error reading stream") {
        ProviderErrorKind::MalformedStream
    } else {
        ProviderErrorKind::Other
    };
    ProviderError::new(kind, text).with_api_contract(None, Some(true), None)
}

pub(crate) fn backfill_missing_usage(
    completion: &mut Completion,
    request: &crate::types::ChatRequest,
) {
    if completion.usage.input_tokens == 0 {
        completion.usage.input_tokens =
            crate::types::estimate_request_input_tokens(&request.messages, &request.tools);
        completion.usage.estimated = true;
    }
    if completion.usage.context_occupancy == 0 {
        completion.usage.context_occupancy = completion.usage.input_tokens;
    }
    if completion.usage.output_tokens == 0 {
        completion.usage.output_tokens = estimate_completion_output_tokens(&completion.content);
        completion.usage.estimated = true;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::Result;
    use futures_util::{StreamExt, stream};

    use super::collect_completion;
    use crate::types::{Content, StreamEvent};

    fn event(data: serde_json::Value) -> Result<eventsource_stream::Event> {
        Ok(eventsource_stream::Event {
            data: data.to_string(),
            ..Default::default()
        })
    }

    #[tokio::test(start_paused = true)]
    async fn completed_response_without_usage_does_not_wait_for_socket_close() {
        let events = stream::iter(vec![
            event(serde_json::json!({"type": "response.output_text.delta", "delta": "done"})),
            event(serde_json::json!({
                "type": "response.completed",
                "response": {"status": "completed"}
            })),
        ])
        .chain(stream::pending());
        let started = tokio::time::Instant::now();
        let completion = tokio::time::timeout(
            Duration::from_secs(3),
            collect_completion(events, &mut |_| {}),
        )
        .await
        .expect("terminal response must not wait indefinitely for missing usage")
        .unwrap();
        assert_eq!(started.elapsed(), super::POST_FINISH_USAGE_GRACE);
        assert_eq!(completion.stop_reason.as_deref(), Some("completed"));
        assert!(matches!(&completion.content[0], Content::Text(text) if text == "done"));
    }

    #[tokio::test(start_paused = true)]
    async fn terminal_usage_grace_is_not_extended_by_heartbeats() {
        let terminal = stream::iter(vec![event(serde_json::json!({
            "type": "response.completed", "response": {"status": "completed"}
        }))]);
        let heartbeats = stream::unfold((), |_| async {
            tokio::time::sleep(Duration::from_millis(500)).await;
            Some((event(serde_json::json!({"type": "ping"})), ()))
        });
        let events = Box::pin(terminal.chain(heartbeats));
        let started = tokio::time::Instant::now();
        tokio::time::timeout(
            Duration::from_secs(3),
            collect_completion(events, &mut |_| {}),
        )
        .await
        .expect("heartbeats must not keep a terminal response alive")
        .unwrap();
        assert_eq!(started.elapsed(), super::POST_FINISH_USAGE_GRACE);
    }

    #[tokio::test(start_paused = true)]
    async fn trailing_usage_after_terminal_response_is_preserved() {
        let events = stream::iter(vec![
            event(serde_json::json!({
                "type": "response.completed", "response": {"status": "completed"}
            })),
            event(serde_json::json!({"usage": {"input_tokens": 12, "output_tokens": 8}})),
        ])
        .chain(stream::pending());
        let completion = tokio::time::timeout(
            Duration::from_secs(3),
            collect_completion(events, &mut |_| {}),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(completion.usage.input_tokens, 12);
        assert_eq!(completion.usage.output_tokens, 8);
    }

    #[tokio::test(start_paused = true)]
    async fn incomplete_response_preserves_output_usage_and_token_limit_reason() {
        let events = stream::iter(vec![event(serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 20, "output_tokens": 10},
                "output": [{"type": "message", "content": [{
                    "type": "output_text", "text": "partial answer"
                }]}]
            }
        }))])
        .chain(stream::pending());
        let completion = tokio::time::timeout(
            Duration::from_secs(3),
            collect_completion(events, &mut |_| {}),
        )
        .await
        .expect("incomplete is a terminal response too")
        .unwrap();
        assert_eq!(completion.stop_reason.as_deref(), Some("max_tokens"));
        assert_eq!(completion.usage.input_tokens, 20);
        assert_eq!(completion.usage.output_tokens, 10);
        assert!(matches!(
            &completion.content[0], Content::Text(text) if text == "partial answer"
        ));
    }

    #[tokio::test]
    async fn responses_stream_accepts_more_than_128_parallel_tool_calls() {
        let events = (0..175)
            .map(|index| {
                Ok(eventsource_stream::Event {
                    event: "response.output_item.done".into(),
                    data: serde_json::json!({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "id": format!("item_{index}"),
                            "call_id": format!("call_{index}"),
                            "name": "read",
                            "arguments": "{}"
                        }
                    })
                    .to_string(),
                    ..Default::default()
                })
            })
            .collect::<Vec<Result<eventsource_stream::Event>>>();
        let mut sink = |_: StreamEvent| {};
        let completion = collect_completion(stream::iter(events), &mut sink)
            .await
            .unwrap();
        assert_eq!(completion.tool_calls().len(), 175);
    }
}
