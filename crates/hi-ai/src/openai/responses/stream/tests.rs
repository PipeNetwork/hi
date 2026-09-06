use anyhow::Result;
use eventsource_stream::Eventsource;
use futures_util::{StreamExt, stream};
use serde_json::{Value, json};

use super::collect_completion;
use crate::{Content, Message, ProviderError, ProviderErrorKind, StreamEvent, ToolCallChannel};

fn event(data: Value) -> Result<eventsource_stream::Event> {
    Ok(eventsource_stream::Event {
        data: data.to_string(),
        ..Default::default()
    })
}

fn completed(output: Vec<Value>) -> Result<eventsource_stream::Event> {
    event(
        json!({"type": "response.completed", "response": {"status": "completed", "output": output}}),
    )
}

fn call(id: &str, call_id: &str, name: &str, arguments: &str) -> Value {
    json!({"type": "function_call", "id": id, "call_id": call_id, "name": name, "arguments": arguments, "status": "completed"})
}

fn message(id: &str, phase: &str, text: &str) -> Value {
    json!({"type": "message", "id": id, "role": "assistant", "phase": phase, "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn error_kind(error: &anyhow::Error) -> ProviderErrorKind {
    error
        .downcast_ref::<ProviderError>()
        .expect("typed provider error")
        .kind
}

#[tokio::test]
async fn interleaved_calls_keep_call_ids_and_literal_argument_fragments() {
    let first = call("fc_1", "call_1", "read", r#"{"path":"α.txt"}"#);
    let second = call("fc_2", "call_2", "list", r#"{"path":"src"}"#);
    let events = vec![
        event(
            json!({"type": "response.output_item.added", "output_index": 1, "item": {
            "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read", "arguments": ""}}),
        ),
        event(
            json!({"type": "response.output_item.added", "output_index": 2, "item": {
            "type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "list", "arguments": ""}}),
        ),
        event(
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 1, "delta": "{\"path\":"}),
        ),
        event(
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_2", "output_index": 2, "delta": "{\"path\":\"src\"}"}),
        ),
        event(
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 1, "delta": "\"α.txt\"}"}),
        ),
        event(
            json!({"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 1, "arguments": first["arguments"]}),
        ),
        event(json!({"type": "response.output_item.done", "output_index": 1, "item": first})),
        completed(vec![first.clone(), second.clone()]),
    ];
    let mut deltas = Vec::new();
    let result = collect_completion(stream::iter(events), &mut |event| deltas.push(event))
        .await
        .unwrap();
    let calls = result.tool_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_1");
    assert_eq!(calls[1].id, "call_2");
    assert_eq!(calls[0].arguments, first["arguments"].as_str().unwrap());
    assert_eq!(calls[1].arguments, second["arguments"].as_str().unwrap());
    assert_eq!(result.tool_call_channel, ToolCallChannel::Native);
    let mut arguments = [String::new(), String::new()];
    let mut ids = Vec::new();
    for delta in deltas {
        if let StreamEvent::ToolCallDelta {
            index,
            id_delta,
            arguments_delta,
            ..
        } = delta
        {
            arguments[index].push_str(&arguments_delta);
            if let Some(id) = id_delta {
                ids.push(id);
            }
        }
    }
    assert_eq!(ids, ["call_1", "call_2"]);
    assert_eq!(arguments[0], first["arguments"]);
    assert_eq!(arguments[1], second["arguments"]);
}

#[tokio::test]
async fn fragmented_sse_and_json_type_fallback_preserve_utf8() {
    let output = vec![message("msg_1", "final_answer", "héllo")];
    let wire = format!(
        "event: response.output_text.delta\r\ndata: {{\"delta\":\"héllo\"}}\r\n\r\ndata: {}\n\n",
        json!({"type": "response.completed", "response": {"status": "completed", "output": output}})
    );
    // Single-byte chunks split both SSE delimiters and the multi-byte é.
    let bytes = wire
        .into_bytes()
        .into_iter()
        .map(|byte| Ok::<_, std::io::Error>(vec![byte]));
    let events = stream::iter(bytes)
        .eventsource()
        .map(|event| event.map_err(Into::into));
    let mut streamed = String::new();
    let result = collect_completion(events, &mut |event| {
        if let StreamEvent::Text(text) = event {
            streamed.push_str(&text);
        }
    })
    .await
    .unwrap();
    assert_eq!(streamed, "héllo");
    assert_eq!(Message::assistant(result.content).text(), "héllo");
}

#[tokio::test]
async fn final_output_preserves_reasoning_phase_order_and_cache_usage_without_duplicate_text() {
    let output = vec![
        json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque",
            "summary": [{"type": "summary_text", "text": "Checking details"}]}),
        message("msg_1", "commentary", "I will check."),
        call("fc_1", "call_1", "read", "{}"),
        message("msg_2", "final_answer", "Done."),
    ];
    let mut events = Vec::new();
    for text in ["I will ", "check.", "Done."] {
        events.push(event(
            json!({"type": "response.output_text.delta", "delta": text}),
        ));
    }
    events.push(event(
        json!({"type": "response.reasoning_summary_text.delta", "delta": "Checking details"}),
    ));
    events.push(event(json!({"type": "response.completed", "response": {
        "status": "completed", "output": output, "usage": {"input_tokens": 120, "output_tokens": 25,
        "input_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 15}}}})));
    let mut reasoning = String::new();
    let result = collect_completion(stream::iter(events), &mut |event| {
        if let StreamEvent::Reasoning(text) = event {
            reasoning.push_str(&text);
        }
    })
    .await
    .unwrap();
    assert_eq!(reasoning, "Checking details");
    assert_eq!(result.usage.input_tokens, 120);
    assert_eq!(result.usage.context_occupancy, 120);
    assert_eq!(result.usage.cache_read_tokens, 80);
    assert_eq!(result.usage.cache_creation_tokens, 15);
    assert_eq!(result.usage.output_tokens, 25);
    assert!(result.usage.input_includes_cache);
    assert!(!result.usage.estimated);
    assert!(
        matches!(&result.content[0], Content::Thinking { text, signature: None } if text == "Checking details")
    );
    let message = Message::assistant(result.content);
    assert_eq!(message.text(), "I will check.Done.");
    assert_eq!(
        message.provider_replay("openai-responses"),
        Some(output.as_slice())
    );
    assert!(message.provider_replay("anthropic").is_none());
}

#[tokio::test]
async fn refusal_and_empty_output_are_valid_completions() {
    let result = collect_completion(
        stream::iter(vec![
            event(json!({"type": "response.refusal.delta", "delta": "Cannot "})),
            event(json!({"type": "response.refusal.delta", "delta": "help."})),
            event(json!({"type": "response.refusal.done", "refusal": "Cannot help."})),
            completed(vec![]),
        ]),
        &mut |_| {},
    )
    .await
    .unwrap();
    assert_eq!(result.refusal.as_deref(), Some("Cannot help."));
    assert!(result.tool_calls().is_empty());
    assert!(Message::assistant(result.content).text().is_empty());

    let result = collect_completion(stream::iter(vec![completed(vec![json!({
        "type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "refusal", "refusal": "No."}]
    })])]), &mut |_| {}).await.unwrap();
    assert_eq!(result.refusal.as_deref(), Some("No."));
}

#[tokio::test]
async fn incomplete_generation_preserves_recovery_reason_but_never_executes_or_replays_tools() {
    let result = collect_completion(stream::iter(vec![event(json!({"type": "response.incomplete", "response": {
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 40, "output_tokens": 20},
        "output": [message("msg_1", "commentary", "Working"),
            call("fc_1", "call_1", "read", "{}"),
            {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "write", "arguments": "{\"path\":", "status": "incomplete"}]
    }}))]), &mut |_| {}).await.unwrap();
    assert_eq!(result.stop_reason.as_deref(), Some("max_tokens"));
    assert_eq!(result.usage.output_tokens, 20);
    assert!(result.tool_calls().is_empty());
    assert_eq!(result.tool_call_channel, ToolCallChannel::None);
    let message = Message::assistant(result.content);
    assert_eq!(message.text(), "Working");
    assert_eq!(
        message.provider_replay("openai-responses").unwrap().len(),
        1
    );
}

#[tokio::test]
async fn eof_done_and_read_errors_before_terminal_never_commit_partial_tools() {
    for tail in [
        None,
        Some(Ok(eventsource_stream::Event {
            data: "[DONE]".into(),
            ..Default::default()
        })),
        Some(Err(anyhow::anyhow!("connection reset"))),
    ] {
        let mut events = vec![
            event(json!({"type": "response.output_text.delta", "delta": "Useful text"})),
            event(
                json!({"type": "response.output_item.done", "output_index": 0, "item": call("fc_1", "call_1", "write", "{}")}),
            ),
        ];
        events.extend(tail);
        let error = collect_completion(stream::iter(events), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(error_kind(&error), ProviderErrorKind::MalformedStream);
    }
}

#[tokio::test]
async fn completed_response_rejects_incomplete_invalid_or_duplicate_tools() {
    for output in [
        vec![call("fc_1", "call_1", "write", "{\"path\":")],
        vec![call("fc_1", "call_1", "write", "[]")],
        vec![call("fc_1", "call_1", "write", "")],
        vec![
            call("fc_1", "call_1", "write", "{}"),
            call("fc_2", "call_1", "read", "{}"),
        ],
        vec![
            json!({"type": "function_call", "call_id": "call_1", "name": "write", "arguments": "{}", "status": "incomplete"}),
        ],
    ] {
        let error = collect_completion(stream::iter(vec![completed(output)]), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(error_kind(&error), ProviderErrorKind::ToolProtocol);
    }
}

#[tokio::test]
async fn tool_argument_growth_is_bounded_before_terminal_event() {
    let events = vec![
        event(
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0,
            "delta": "a".repeat(crate::tool_validation::MAX_TOOL_ARGUMENT_BYTES)}),
        ),
        event(
            json!({"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": "b"}),
        ),
    ];
    let error = collect_completion(stream::iter(events), &mut |_| {})
        .await
        .unwrap_err();
    assert_eq!(error_kind(&error), ProviderErrorKind::ToolProtocol);
}

#[tokio::test]
async fn failed_and_error_events_keep_structured_api_error_classification() {
    for payload in [
        json!({"type": "response.failed", "response": {"error": {"code": "context_length_exceeded", "message": "Context length exceeded"}}}),
        json!({"type": "error", "code": "context_length_exceeded", "message": "Context length exceeded"}),
    ] {
        let error = collect_completion(stream::iter(vec![event(payload)]), &mut |_| {})
            .await
            .unwrap_err();
        assert_eq!(error_kind(&error), ProviderErrorKind::RequestTooLarge);
    }
}

#[tokio::test(start_paused = true)]
async fn terminal_event_finishes_without_waiting_for_socket_close_or_usage() {
    let events = stream::iter(vec![completed(vec![])]).chain(stream::pending());
    tokio::time::timeout(
        std::time::Duration::from_millis(1),
        collect_completion(events, &mut |_| {}),
    )
    .await
    .expect("terminal output is already authoritative")
    .unwrap();
}

#[tokio::test]
async fn malformed_json_is_not_silently_ignored_after_text() {
    let error = collect_completion(
        stream::iter(vec![
            event(json!({"type": "response.output_text.delta", "delta": "Partial"})),
            Ok(eventsource_stream::Event {
                data: "{broken".into(),
                ..Default::default()
            }),
            completed(vec![message("msg_1", "final_answer", "Partial")]),
        ]),
        &mut |_| {},
    )
    .await
    .unwrap_err();
    assert_eq!(error_kind(&error), ProviderErrorKind::MalformedStream);
}

#[tokio::test]
async fn missing_usage_is_estimated_but_provider_zero_counts_are_authoritative() {
    let missing = collect_completion(stream::iter(vec![completed(vec![])]), &mut |_| {})
        .await
        .unwrap();
    assert!(missing.usage.estimated);
    let zero = collect_completion(
        stream::iter(vec![event(json!({
            "type": "response.completed", "response": {"status": "completed", "output": [],
            "usage": {"input_tokens": 0, "output_tokens": 0}}
        }))]),
        &mut |_| {},
    )
    .await
    .unwrap();
    assert!(!zero.usage.estimated);
    assert_eq!(zero.usage.input_tokens, 0);
    assert_eq!(zero.usage.output_tokens, 0);
}

#[tokio::test]
async fn terminal_cannot_change_or_drop_a_finalized_streamed_tool() {
    let original = call("fc_1", "call_1", "read", r#"{"path":"safe"}"#);
    for output in [
        vec![],
        vec![call("fc_1", "call_1", "read", r#"{"path":"different"}"#)],
        vec![call("fc_1", "call_1", "write", r#"{"path":"safe"}"#)],
        vec![call("fc_1", "call_2", "read", r#"{"path":"safe"}"#)],
    ] {
        let error = collect_completion(stream::iter(vec![
            event(json!({"type": "response.output_item.done", "output_index": 0, "item": original})),
            completed(output),
        ]), &mut |_| {}).await.unwrap_err();
        assert_eq!(error_kind(&error), ProviderErrorKind::ToolProtocol);
    }
}
