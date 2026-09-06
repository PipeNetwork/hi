//! Promotion policy for the one sealed plain-text tool recovery response.

use hi_ai::{Completion, Content, ToolCallChannel};

use crate::heuristics::{parse_text_tool_calls, textcall_id_offset};
use crate::transcript::Transcript;

pub(super) type ToolCallTuple = (String, String, String);

/// Promote exactly one textual call. Multiple calls are rejected as a whole,
/// regardless of the ordinary parallel native-call limit.
pub(super) fn promote(
    completion: &mut Completion,
    transcript: &Transcript,
    enabled: bool,
    calls: Vec<ToolCallTuple>,
) -> (Vec<ToolCallTuple>, bool) {
    if !enabled {
        return (calls, false);
    }

    let mut rejected_multiple =
        completion.tool_call_channel == ToolCallChannel::TextFallback && calls.len() > 1;
    let calls = if calls.is_empty() {
        let full_text = completion
            .content
            .iter()
            .filter_map(|content| match content {
                Content::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let parsed = parse_text_tool_calls(&full_text, textcall_id_offset(transcript));
        let count = parsed
            .iter()
            .filter(|content| matches!(content, Content::ToolCall { .. }))
            .count();
        if count == 1 {
            let mut content = Vec::new();
            let mut parsed = parsed.into_iter();
            for existing in &completion.content {
                match existing {
                    Content::Text(_) => content.extend(parsed.by_ref()),
                    Content::Thinking { .. } => content.push(existing.clone()),
                    _ => {}
                }
            }
            content.extend(parsed);
            completion.content = content;
            completion.tool_call_channel = ToolCallChannel::TextFallback;
            completion
                .tool_calls()
                .into_iter()
                .map(|call| {
                    (
                        call.id.to_string(),
                        call.name.to_string(),
                        call.arguments.to_string(),
                    )
                })
                .collect()
        } else {
            rejected_multiple = count > 1;
            Vec::new()
        }
    } else {
        calls
    };

    if rejected_multiple {
        completion
            .content
            .retain(|content| matches!(content, Content::Thinking { .. }));
        completion.content.push(Content::Text(
            "[plain-text tool retry: expected exactly one executable call]".into(),
        ));
        (Vec::new(), true)
    } else {
        (calls, false)
    }
}
