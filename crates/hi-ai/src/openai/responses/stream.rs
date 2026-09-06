//! Collect OpenAI Responses events without exposing unfinished tool calls.

use std::collections::HashSet;

use anyhow::Result;
use futures_util::{Stream, StreamExt};
use serde_json::Value;

use crate::provider::{ProviderError, ProviderErrorKind};
use crate::types::{Completion, Content, Message, StreamEvent, ToolCallChannel, Usage};

const MAX_TOOL_NAME_BYTES: usize = 256;

#[derive(Default)]
struct StreamAcc {
    calls: Vec<FunctionCall>,
    tool_payload_bytes: usize,
    refusal: String,
}

#[derive(Default)]
struct FunctionCall {
    output_index: Option<u64>,
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    arguments_done: bool,
    id_emitted: bool,
    name_emitted: bool,
}

/// Only a Responses terminal event commits a completion. In particular, SSE
/// `[DONE]`, EOF, and transport errors cannot authorize execution of a partial
/// tool batch, even if the visible text or arguments looked complete.
pub(super) async fn collect_completion(
    mut stream: impl Stream<Item = Result<eventsource_stream::Event>> + Unpin,
    sink: &mut (dyn FnMut(StreamEvent) + Send),
) -> Result<Completion> {
    let mut acc = StreamAcc::default();
    while let Some(event) = stream.next().await {
        let event = event
            .map_err(|err| malformed(format!("error reading OpenAI Responses stream: {err}")))?;
        if event.data.trim() == "[DONE]" {
            return Err(malformed("OpenAI Responses stream ended before a terminal event").into());
        }
        if event.data.trim().is_empty() {
            continue;
        }
        let data: Value = serde_json::from_str(&event.data)
            .map_err(|_| malformed("malformed SSE JSON chunk in OpenAI Responses stream"))?;
        let event_type = if event.event.is_empty() || event.event == "message" {
            string(&data, "type").unwrap_or("")
        } else {
            &event.event
        };
        match event_type {
            "response.output_text.delta" => {
                sink(StreamEvent::Text(required_string(&data, "delta")?.into()));
            }
            "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
                sink(StreamEvent::Reasoning(
                    required_string(&data, "delta")?.into(),
                ));
            }
            "response.refusal.delta" => {
                acc.refusal.push_str(required_string(&data, "delta")?);
            }
            "response.refusal.done" => {
                acc.refusal = required_string(&data, "refusal")?.into();
            }
            "response.output_item.added" | "response.output_item.done" => {
                let item = data.get("item").filter(|v| v.is_object()).ok_or_else(|| {
                    malformed("OpenAI Responses output event is missing its item")
                })?;
                if string(item, "type") == Some("function_call") {
                    let index = acc.locate_call(string(item, "id"), data.get("output_index"))?;
                    let delta = acc.update_call(index, item)?;
                    if event_type == "response.output_item.done" {
                        acc.calls[index].arguments_done = true;
                    }
                    acc.emit_call(index, &delta, sink);
                }
            }
            "response.function_call_arguments.delta" => {
                let index = acc.locate_call(string(&data, "item_id"), data.get("output_index"))?;
                let delta = required_string(&data, "delta")?;
                let call = &mut acc.calls[index];
                if call.arguments_done {
                    return Err(tool_protocol(
                        "OpenAI tool arguments continued after finalization",
                    )
                    .into());
                }
                append_arguments(&mut call.arguments, delta, &mut acc.tool_payload_bytes)?;
                acc.emit_call(index, delta, sink);
            }
            "response.function_call_arguments.done" => {
                let index = acc.locate_call(string(&data, "item_id"), data.get("output_index"))?;
                let arguments = required_string(&data, "arguments")?;
                let delta = acc.complete_arguments(index, arguments)?;
                acc.calls[index].arguments_done = true;
                acc.emit_call(index, &delta, sink);
            }
            "response.completed" | "response.incomplete" => {
                let response = data
                    .get("response")
                    .filter(|v| v.is_object())
                    .ok_or_else(|| malformed("OpenAI Responses terminal event has no response"))?;
                if let Some(error) = response.get("error").filter(|v| !v.is_null()) {
                    return Err(api_error(error).into());
                }
                return finish(response, event_type == "response.incomplete", &acc);
            }
            "response.failed" => {
                let error = data
                    .pointer("/response/error")
                    .or_else(|| data.get("error"))
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({"message": "OpenAI response failed"}));
                return Err(api_error(&error).into());
            }
            "error" => {
                return Err(api_error(data.get("error").unwrap_or(&data)).into());
            }
            // These snapshot events repeat data delivered in deltas and in the
            // terminal response. The terminal output is the authoritative,
            // ordered representation, including unknown future item fields.
            "response.created"
            | "response.in_progress"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_text.done"
            | "response.reasoning_text.done"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done" => {}
            _ => {
                if let Some(error) = data.get("error").filter(|v| !v.is_null()) {
                    return Err(api_error(error).into());
                }
            }
        }
    }
    Err(malformed("OpenAI Responses stream ended before a terminal event").into())
}

impl StreamAcc {
    fn locate_call(
        &mut self,
        item_id: Option<&str>,
        output_index: Option<&Value>,
    ) -> Result<usize> {
        let item_id = item_id.filter(|id| !id.is_empty()).unwrap_or("");
        let output_index = output_index.and_then(Value::as_u64);
        if item_id.is_empty() && output_index.is_none() {
            return Err(tool_protocol("OpenAI tool event has no item identity").into());
        }
        if let Some(index) = self.calls.iter().position(|call| {
            (!item_id.is_empty() && call.item_id == item_id)
                || (output_index.is_some() && call.output_index == output_index)
        }) {
            let call = &mut self.calls[index];
            if (!item_id.is_empty() && !call.item_id.is_empty() && call.item_id != item_id)
                || (output_index.is_some()
                    && call.output_index.is_some()
                    && call.output_index != output_index)
            {
                return Err(tool_protocol("OpenAI tool event changed its item identity").into());
            }
            if call.item_id.is_empty() {
                reserve(&mut self.tool_payload_bytes, item_id.len())?;
                call.item_id = item_id.into();
            }
            if call.output_index.is_none() {
                call.output_index = output_index;
            }
            return Ok(index);
        }
        reserve(
            &mut self.tool_payload_bytes,
            crate::tool_validation::TOOL_CALL_SLOT_OVERHEAD_BYTES.saturating_add(item_id.len()),
        )?;
        self.calls.push(FunctionCall {
            item_id: item_id.into(),
            output_index,
            ..FunctionCall::default()
        });
        Ok(self.calls.len() - 1)
    }

    fn update_call(&mut self, index: usize, item: &Value) -> Result<String> {
        let call = &mut self.calls[index];
        for (field, current) in [("call_id", &mut call.call_id), ("name", &mut call.name)] {
            if let Some(value) = string(item, field).filter(|s| !s.is_empty()) {
                if field == "name" && value.len() > MAX_TOOL_NAME_BYTES {
                    return Err(
                        tool_protocol("model exceeded the streamed tool-name size limit").into(),
                    );
                }
                if current.is_empty() {
                    reserve(&mut self.tool_payload_bytes, value.len())?;
                    *current = value.into();
                } else if current != value {
                    return Err(
                        tool_protocol("OpenAI tool event changed its call id or name").into(),
                    );
                }
            }
        }
        if let Some(arguments) = string(item, "arguments").filter(|s| !s.is_empty()) {
            return self.complete_arguments(index, arguments);
        }
        Ok(String::new())
    }

    fn complete_arguments(&mut self, index: usize, arguments: &str) -> Result<String> {
        let call = &mut self.calls[index];
        let suffix = arguments.strip_prefix(&call.arguments).ok_or_else(|| {
            tool_protocol("OpenAI tool arguments disagree with streamed fragments")
        })?;
        let suffix = suffix.to_string();
        append_arguments(&mut call.arguments, &suffix, &mut self.tool_payload_bytes)?;
        Ok(suffix)
    }

    fn emit_call(
        &mut self,
        index: usize,
        arguments_delta: &str,
        sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) {
        let call = &mut self.calls[index];
        let id_delta = (!call.id_emitted && !call.call_id.is_empty()).then(|| call.call_id.clone());
        let name_delta = (!call.name_emitted && !call.name.is_empty()).then(|| call.name.clone());
        call.id_emitted |= id_delta.is_some();
        call.name_emitted |= name_delta.is_some();
        if id_delta.is_some() || name_delta.is_some() || !arguments_delta.is_empty() {
            sink(StreamEvent::ToolCallDelta {
                index,
                id_delta,
                name_delta,
                arguments_delta: arguments_delta.into(),
            });
        }
    }
}

fn finish(response: &Value, incomplete: bool, acc: &StreamAcc) -> Result<Completion> {
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("OpenAI Responses terminal response has no output array"))?;
    let expected_status = if incomplete {
        "incomplete"
    } else {
        "completed"
    };
    if string(response, "status").is_some_and(|status| status != expected_status) {
        return Err(
            malformed("OpenAI Responses terminal event disagrees with response status").into(),
        );
    }
    if !incomplete {
        validate_final_calls(output, &acc.calls)?;
    }
    let mut content = Vec::new();
    let mut replay = Vec::with_capacity(output.len());
    let mut refusal = String::new();
    let mut tool_payload_bytes = 0;
    let mut call_ids = HashSet::new();
    let mut tool_call_channel = ToolCallChannel::None;
    for item in output {
        match string(item, "type") {
            Some("message") => {
                if let Some(parts) = item.get("content").and_then(Value::as_array) {
                    for part in parts {
                        match string(part, "type") {
                            Some("output_text") => {
                                if let Some(text) = string(part, "text").filter(|s| !s.is_empty()) {
                                    content.push(Content::Text(text.into()));
                                }
                            }
                            Some("refusal") => {
                                if let Some(text) = string(part, "refusal") {
                                    refusal.push_str(text);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some("reasoning") => {
                let text = item
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|part| string(part, "text"))
                    .collect::<String>();
                if !text.is_empty() {
                    // OpenAI encrypted content is retained only in its scoped
                    // replay item, never in another provider's signature field.
                    content.push(Content::Thinking {
                        text,
                        signature: None,
                    });
                }
            }
            Some("function_call") => {
                validate_tool_size(item, &mut tool_payload_bytes)?;
                // A truncated generation is recovered as a whole by the agent.
                // Do not execute or replay any part of its pending tool batch.
                if incomplete {
                    continue;
                }
                let id = string(item, "call_id")
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| tool_protocol("OpenAI tool call is missing call_id"))?;
                let name = string(item, "name")
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| tool_protocol("OpenAI tool call is missing its name"))?;
                let arguments = string(item, "arguments")
                    .ok_or_else(|| tool_protocol("OpenAI tool call is missing its arguments"))?;
                if matches!(string(item, "status"), Some(status) if status != "completed")
                    || !serde_json::from_str::<Value>(arguments)
                        .is_ok_and(|value| value.is_object())
                {
                    return Err(tool_protocol(
                        "OpenAI returned unfinished or invalid tool arguments",
                    )
                    .into());
                }
                if !call_ids.insert(id) {
                    return Err(tool_protocol("OpenAI returned duplicate tool call ids").into());
                }
                content.push(Content::ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments: arguments.into(),
                });
                tool_call_channel = ToolCallChannel::Native;
            }
            _ => {}
        }
        replay.push(item.clone());
    }
    if refusal.is_empty() {
        refusal = acc.refusal.clone();
    }
    let stop_reason = if incomplete {
        match response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => "max_tokens",
            Some(reason) => reason,
            None => "incomplete",
        }
    } else {
        "completed"
    };
    Ok(Completion {
        content: Message::assistant(content)
            .with_provider_replay("openai-responses", replay)
            .content,
        stop_reason: Some(stop_reason.into()),
        refusal: (!refusal.is_empty()).then_some(refusal),
        usage: usage(response.get("usage").unwrap_or(&Value::Null)),
        tool_call_channel,
    })
}

fn validate_final_calls(output: &[Value], calls: &[FunctionCall]) -> Result<()> {
    for call in calls {
        let item = output
            .iter()
            .enumerate()
            .find_map(|(index, item)| {
                (string(item, "type") == Some("function_call")
                    && ((!call.item_id.is_empty()
                        && string(item, "id") == Some(call.item_id.as_str()))
                        || (!call.call_id.is_empty()
                            && string(item, "call_id") == Some(call.call_id.as_str()))
                        || call.output_index == Some(index as u64)))
                .then_some(item)
            })
            .ok_or_else(|| tool_protocol("OpenAI terminal output omitted a streamed tool call"))?;
        for (field, previous) in [
            ("id", &call.item_id),
            ("call_id", &call.call_id),
            ("name", &call.name),
        ] {
            if !previous.is_empty() && string(item, field) != Some(previous.as_str()) {
                return Err(tool_protocol(
                    "OpenAI terminal output changed a streamed tool identity",
                )
                .into());
            }
        }
        let arguments = string(item, "arguments").unwrap_or("");
        if !arguments.starts_with(&call.arguments)
            || (call.arguments_done && arguments != call.arguments)
        {
            return Err(
                tool_protocol("OpenAI terminal output changed streamed tool arguments").into(),
            );
        }
    }
    Ok(())
}

fn validate_tool_size(item: &Value, total: &mut usize) -> Result<()> {
    let name = string(item, "name").unwrap_or("");
    let arguments = string(item, "arguments").unwrap_or("");
    if name.len() > MAX_TOOL_NAME_BYTES
        || arguments.len() > crate::tool_validation::MAX_TOOL_ARGUMENT_BYTES
    {
        return Err(
            tool_protocol("model exceeded the tool-name or tool-argument size limit").into(),
        );
    }
    reserve(
        total,
        crate::tool_validation::TOOL_CALL_SLOT_OVERHEAD_BYTES
            .saturating_add(string(item, "id").unwrap_or("").len())
            .saturating_add(string(item, "call_id").unwrap_or("").len())
            .saturating_add(name.len())
            .saturating_add(arguments.len()),
    )
}

fn append_arguments(current: &mut String, delta: &str, total: &mut usize) -> Result<()> {
    if current.len().saturating_add(delta.len()) > crate::tool_validation::MAX_TOOL_ARGUMENT_BYTES {
        return Err(tool_protocol("model exceeded the streamed tool-argument size limit").into());
    }
    reserve(total, delta.len())?;
    current.push_str(delta);
    Ok(())
}

fn reserve(total: &mut usize, additional: usize) -> Result<()> {
    if crate::tool_validation::try_reserve_tool_payload(total, additional) {
        Ok(())
    } else {
        Err(tool_protocol("model exceeded the streamed tool payload size limit").into())
    }
}

fn usage(value: &Value) -> Usage {
    let input_tokens = value
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Usage {
        input_tokens,
        output_tokens: value
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cache_read_tokens: value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cache_creation_tokens: value
            .pointer("/input_tokens_details/cache_write_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        input_includes_cache: true,
        context_occupancy: input_tokens,
        rate_limits: None,
        estimated: value.get("input_tokens").and_then(Value::as_u64).is_none()
            || value.get("output_tokens").and_then(Value::as_u64).is_none(),
    }
}

fn string<'a>(data: &'a Value, field: &str) -> Option<&'a str> {
    data.get(field).and_then(Value::as_str)
}

fn required_string<'a>(data: &'a Value, field: &str) -> Result<&'a str> {
    string(data, field)
        .ok_or_else(|| malformed(format!("OpenAI Responses event is missing {field}")).into())
}

fn malformed(message: impl Into<String>) -> ProviderError {
    ProviderError::new(ProviderErrorKind::MalformedStream, message).with_api_contract(
        None,
        Some(true),
        None,
    )
}

fn tool_protocol(message: impl Into<String>) -> ProviderError {
    ProviderError::new(ProviderErrorKind::ToolProtocol, message).with_api_contract(
        Some("tool_protocol_error".into()),
        Some(true),
        None,
    )
}

fn api_error(error: &Value) -> ProviderError {
    let body = serde_json::json!({"error": error}).to_string();
    crate::openai::request::parse_api_error(None, &body).into_provider_error(None)
}

#[cfg(test)]
mod tests;
