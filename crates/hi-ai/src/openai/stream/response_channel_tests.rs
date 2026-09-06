use anyhow::Result;
use futures_util::stream;

use super::collect_completion_with_protocol;
use crate::{
    Content, ProviderErrorKind, StreamEvent, ToolCallChannel, ToolSpec, provider_error_kind,
};

async fn collect_text(text: &str, text_tool_fallback: bool) -> (String, crate::Completion) {
    let (streamed, completion) = collect_text_result(text, text_tool_fallback).await;
    (streamed, completion.unwrap())
}

async fn collect_text_result(
    text: &str,
    text_tool_fallback: bool,
) -> (String, anyhow::Result<crate::Completion>) {
    let events: Vec<Result<String>> = vec![
        Ok(serde_json::json!({"choices": [{"delta": {"content": text}}]}).to_string()),
        Ok(serde_json::json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}).to_string()),
        Ok("[DONE]".to_string()),
    ];
    let mut streamed = String::new();
    let mut sink = |event: StreamEvent| {
        if let StreamEvent::Text(text) = event {
            streamed.push_str(&text);
        }
    };
    let tools = [ToolSpec {
        name: "read".to_string(),
        description: String::new(),
        parameters: serde_json::json!({"type": "object"}),
    }];
    let completion = collect_completion_with_protocol(
        stream::iter(events),
        &mut sink,
        super::super::deepseek::ToolProtocol::Auto,
        text_tool_fallback.then_some(tools.as_slice()),
    )
    .await;
    (streamed, completion)
}

#[tokio::test]
async fn ordinary_tool_shaped_json_and_xml_remain_visible_text() {
    let text = "Examples:\n{\"name\":\"write\",\"arguments\":{}}\n<tool_call>read</tool_call>";
    let (streamed, completion) = collect_text(text, false).await;
    assert_eq!(streamed, text);
    assert!(matches!(completion.content.as_slice(), [Content::Text(value)] if value == text));
}

#[tokio::test]
async fn sealed_text_tool_fallback_suppresses_protocol_display() {
    let text = "I will inspect it.\n<tool_call>read<arg_key>path</arg_key><arg_value>README.md</arg_value></tool_call>";
    let (streamed, completion) = collect_text(text, true).await;
    assert_eq!(streamed, "I will inspect it.\n");
    assert!(matches!(
        completion.content.as_slice(),
        [Content::Text(value), Content::ToolCall { name, arguments, .. }]
            if value == "I will inspect it."
                && name == "read"
                && arguments == r#"{"path":"README.md"}"#
    ));
    assert_eq!(completion.tool_call_channel, ToolCallChannel::TextFallback);
}

#[tokio::test]
async fn sealed_text_tool_fallback_does_not_promote_unadmitted_tool() {
    let text =
        "Before.\n<tool_call>bash<arg_key>command</arg_key><arg_value>true</arg_value></tool_call>";
    let (streamed, completion) = collect_text_result(text, true).await;
    assert_eq!(streamed, "Before.\n");
    let error = completion.unwrap_err();
    assert_eq!(
        provider_error_kind(&error),
        Some(ProviderErrorKind::ToolProtocol)
    );
    assert!(error.to_string().contains("outside the sealed request"));
}

#[tokio::test]
async fn sealed_text_tool_fallback_rejects_multiple_admitted_calls() {
    let text = concat!(
        "Before.\n",
        "<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value></tool_call>",
        "<tool_call>read<arg_key>path</arg_key><arg_value>b</arg_value></tool_call>"
    );
    let (streamed, completion) = collect_text_result(text, true).await;
    assert_eq!(streamed, "Before.\n");
    let error = completion.unwrap_err();
    assert_eq!(
        provider_error_kind(&error),
        Some(ProviderErrorKind::ToolProtocol)
    );
    assert!(error.to_string().contains("more than one tool call"));
}
