use std::io;

use eventsource_stream::Event;
use futures_util::stream;
use serde_json::{Value, json};

use super::stream::collect_completion;
use crate::{ChatRequest, Content, Message, ToolCallChannel};

fn request() -> ChatRequest {
    ChatRequest {
        model: "claude-test".into(),
        request_id: None,
        retry_attempt: 0,
        user_turn: false,
        canonical_objective: None,
        messages: vec![Message::user("Update the source")].into(),
        tools: vec![].into(),
        tool_envelope: None,
        max_tokens: 128,
        temperature: None,
        top_p: None,
        frequency_penalty: None,
        thinking_budget: None,
        reasoning_effort: None,
        profile: Default::default(),
    }
}

fn event(kind: &str, data: Value) -> Result<Event, io::Error> {
    Ok(Event {
        event: kind.into(),
        data: data.to_string(),
        ..Default::default()
    })
}

fn tool_events() -> Vec<Result<Event, io::Error>> {
    vec![
        event(
            "message_start",
            json!({"message":{"usage":{
                "input_tokens":10,"cache_read_input_tokens":20
            }}}),
        ),
        event(
            "content_block_start",
            json!({"index":0,"content_block":{
                "type":"tool_use","id":"call-1","name":"write","input":{}
            }}),
        ),
        event(
            "content_block_delta",
            json!({"index":0,"delta":{
                "type":"input_json_delta", "partial_json":"{\"path\":\"main.rs\",\"content\":\"updated\"}"
            }}),
        ),
        event("content_block_stop", json!({"index":0})),
    ]
}

#[tokio::test]
async fn valid_tool_json_without_message_completion_is_truncated() {
    for transport_error in [false, true] {
        let mut events = tool_events();
        if transport_error {
            events.push(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "connection closed",
            )));
        }
        let completion = collect_completion(stream::iter(events), &request(), &mut |_| {})
            .await
            .unwrap();
        assert_eq!(completion.stop_reason.as_deref(), Some("length"));
        assert_eq!(completion.tool_call_channel, ToolCallChannel::Native);
        assert!(
            matches!(completion.content.first(), Some(Content::ToolCall { name, arguments, .. })
            if name == "write" && serde_json::from_str::<Value>(arguments).unwrap()["path"] == "main.rs")
        );
        assert_eq!(completion.usage.input_tokens, 10);
        assert_eq!(completion.usage.cache_read_tokens, 20);
        assert_eq!(completion.usage.context_occupancy, 30);
        assert!(completion.usage.output_tokens > 0);
        assert!(completion.usage.estimated);
    }
}

#[tokio::test]
async fn interrupted_text_preserves_partial_answer_for_recovery() {
    let events = vec![
        event(
            "content_block_start",
            json!({"index":0,"content_block":{"type":"text","text":""}}),
        ),
        event(
            "content_block_delta",
            json!({"index":0,"delta":{"type":"text_delta","text":"Work in progress"}}),
        ),
    ];
    let completion = collect_completion(stream::iter(events), &request(), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(completion.stop_reason.as_deref(), Some("length"));
    assert!(
        matches!(completion.content.first(), Some(Content::Text(text)) if text == "Work in progress")
    );
}

#[tokio::test]
async fn completed_message_keeps_authoritative_stop_and_usage() {
    for reason in ["tool_use", "end_turn", "max_tokens"] {
        let mut events = tool_events();
        events.push(event(
            "message_delta",
            json!({
                "delta":{"stop_reason":reason},"usage":{"output_tokens":3}
            }),
        ));
        let completion = collect_completion(stream::iter(events), &request(), &mut |_| {})
            .await
            .unwrap();
        assert_eq!(completion.stop_reason.as_deref(), Some(reason));
        assert_eq!(completion.usage.input_tokens, 10);
        assert_eq!(completion.usage.output_tokens, 3);
        assert!(!completion.usage.estimated);
    }
}

#[tokio::test]
async fn completed_stream_preserves_authoritative_zero_usage() {
    let events = vec![
        event(
            "message_start",
            json!({"message":{"usage":{
                "input_tokens":0,"cache_read_input_tokens":100
            }}}),
        ),
        event(
            "message_delta",
            json!({
                "delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}
            }),
        ),
    ];
    let completion = collect_completion(stream::iter(events), &request(), &mut |_| {})
        .await
        .unwrap();
    assert_eq!(completion.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(completion.usage.input_tokens, 0);
    assert_eq!(completion.usage.output_tokens, 0);
    assert_eq!(completion.usage.cache_read_tokens, 100);
    assert_eq!(completion.usage.context_occupancy, 100);
    assert!(!completion.usage.estimated);
}

#[tokio::test]
async fn transport_failure_before_content_remains_an_error() {
    let events = vec![Err(io::Error::new(io::ErrorKind::ConnectionReset, "reset"))];
    let error = collect_completion(stream::iter(events), &request(), &mut |_| {})
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("reset"));
}
