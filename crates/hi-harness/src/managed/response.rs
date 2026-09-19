use super::*;

pub(crate) fn reported_outcome(outcome: &hi_tools::ToolOutcome) -> String {
    json!({"evidence_source":"client_reported","status":outcome.status,"process":outcome.process,"truncation":outcome.truncation,"effects":outcome.effects,"content":outcome.content}).to_string()
}
/// Strict buffered decoder. No text/tool payload is delivered until [DONE], a
/// finish reason and accepted terminal settlement metadata all agree.
pub(crate) fn parse_response(bytes: &[u8]) -> Result<Value> {
    let text = std::str::from_utf8(bytes)?;
    if text.trim_start().starts_with('{') {
        return serde_json::from_str(text).context("invalid managed JSON response");
    }
    let mut result = json!({"choices":[{"message":{"role":"assistant","content":""}}]});
    let mut calls: BTreeMap<usize, Value> = BTreeMap::new();
    let mut done = false;
    let mut terminal = false;
    for line in text.lines().filter_map(|l| l.strip_prefix("data:")) {
        let data = line.trim();
        ensure!(!done, "managed SSE has data after [DONE]");
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let chunk: Value = serde_json::from_str(data).context("invalid managed SSE JSON")?;
        ensure!(
            chunk.get("error").is_none(),
            "managed SSE error: {}",
            chunk["error"]
        );
        if let Some(id) = chunk.get("id") {
            if let Some(old) = result.get("id") {
                ensure!(old == id, "managed response ID changed");
            }
            result["id"] = id.clone();
        }
        if let Some(meta) = chunk.get("pipe") {
            ensure!(
                result.get("pipe").is_none(),
                "duplicate managed terminal metadata"
            );
            result["pipe"] = meta.clone();
        }
        if let Some(usage) = chunk.get("usage") {
            result["usage"] = usage.clone();
        }
        let choices = chunk["choices"]
            .as_array()
            .context("managed SSE choices missing")?;
        ensure!(
            choices.len() == 1 && choices[0]["index"] == 0,
            "invalid managed SSE choices"
        );
        let choice = &choices[0];
        if let Some(reason) = choice.get("finish_reason").filter(|v| !v.is_null()) {
            ensure!(!terminal, "duplicate managed finish");
            terminal = true;
            result["choices"][0]["finish_reason"] = reason.clone();
        }
        if let Some(delta) = choice.get("delta") {
            if let Some(content) = delta.get("content").filter(|v| !v.is_null()) {
                ensure!(!terminal, "text after managed finish");
                let t = content.as_str().context("managed text must be a string")?;
                let out = result["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap()
                    .to_owned()
                    + t;
                result["choices"][0]["message"]["content"] = json!(out);
            }
            if let Some(items) = delta.get("tool_calls") {
                ensure!(!terminal, "tool arguments after managed finish");
                for call in items
                    .as_array()
                    .context("managed tool_calls array required")?
                {
                    let index = call["index"]
                        .as_u64()
                        .context("managed tool index required")?
                        as usize;
                    ensure!(index < 8192, "managed tool index too large");
                    let entry = calls.entry(index).or_insert_with(
                        || json!({"id":"","type":"function","function":{"name":"","arguments":""}}),
                    );
                    if let Some(kind) = call.get("type") {
                        ensure!(kind == "function", "unsupported tool type");
                    }
                    for (pointer, part) in [
                        ("/id", call.get("id")),
                        ("/function/name", call.pointer("/function/name")),
                        ("/function/arguments", call.pointer("/function/arguments")),
                    ] {
                        if let Some(part) = part {
                            let field = entry.pointer_mut(pointer).unwrap();
                            *field = json!(
                                field.as_str().unwrap().to_owned()
                                    + part.as_str().context("invalid tool delta")?
                            );
                        }
                    }
                }
            }
        }
    }
    ensure!(
        done && terminal,
        "managed SSE interrupted: [DONE] and terminal finish required"
    );
    if !calls.is_empty() {
        ensure!(
            calls.keys().copied().eq(0..calls.len()),
            "managed tool indices are not contiguous"
        );
        result["choices"][0]["message"]["tool_calls"] =
            json!(calls.into_values().collect::<Vec<_>>());
    }
    Ok(result)
}
pub(super) fn accepted(value: &Value, tools: &[ToolSpec]) -> Result<PipeCompletion> {
    let pipe = &value["pipe"];
    ensure!(
        value.get("error").is_none()
            && pipe["status"] == "completed"
            && pipe["verification"]["status"] == "passed"
            && pipe["verification"]["scope"] == "client_reported_execution_evidence"
            && pipe["result_not_retained"] != true,
        "managed response was not accepted: {}",
        pipe
    );
    ensure!(
        value["id"].as_str().is_some_and(|s| !s.is_empty())
            && value["choices"].as_array().is_some_and(|v| v.len() == 1),
        "managed accepted response is incomplete"
    );
    let message = &value["choices"][0]["message"];
    ensure!(
        message["role"] == "assistant",
        "invalid managed message role"
    );
    ensure!(
        message.get("tool_calls").is_none_or(Value::is_array),
        "managed tool_calls must be an array"
    );
    ensure!(
        message["content"].is_null() || message["content"].is_string(),
        "invalid managed content"
    );
    let mut completion = PipeCompletion {
        text: message["content"].as_str().unwrap_or("").into(),
        finish_reason: value["choices"][0]["finish_reason"]
            .as_str()
            .map(str::to_owned),
        usage: crate::pipe::parse_usage(&value["usage"]),
        ..Default::default()
    };
    let mut neutral = hi_ai::Completion::default();
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        ensure!(call["type"] == "function", "invalid managed tool type");
        let call = ToolCall {
            id: call["id"]
                .as_str()
                .context("managed call ID missing")?
                .into(),
            name: call["function"]["name"]
                .as_str()
                .context("managed tool name missing")?
                .into(),
            arguments: call["function"]["arguments"]
                .as_str()
                .context("managed tool arguments missing")?
                .into(),
        };
        neutral.content.push(Content::ToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        });
        completion.tool_calls.push(call);
    }
    hi_ai::validate_client_tool_calls(&neutral, tools, hi_ai::ToolMode::Auto)?;
    ensure!(
        completion.finish_reason.as_deref()
            == Some(if completion.tool_calls.is_empty() {
                "stop"
            } else {
                "tool_calls"
            })
            && !completion.is_empty(),
        "managed response was empty or truncated"
    );
    Ok(completion)
}
