//! Per-request census for the harness judge. Counts what a provider send
//! actually carries — not the on-disk session — so elision and caps are
//! visible without a live model.

use hi_ai::{Content, Message, Role};

/// One provider send: token estimate and the largest remaining payloads.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RequestCensus {
    pub input_tokens_est: u64,
    pub max_tool_result_chars: u64,
    pub max_image_chars: u64,
    pub max_thinking_chars: u64,
    pub elided_images: u32,
    pub elided_thinking: u32,
    pub elided_tool_results: u32,
    /// Recorded only. User text is never a budget fail.
    pub user_message_chars: u64,
    pub user_messages: u32,
    pub assistant_messages: u32,
    pub tool_messages: u32,
}

/// Elide/compact event recorded on the turn tape.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompactionEvent {
    pub freed_chars: u64,
    pub keep_recent: usize,
}

pub fn tool_kind(name: &str) -> &'static str {
    match name {
        "read" => "read",
        "write" | "edit" | "multi_edit" | "apply_patch" | "delete" | "move" => "mutate",
        "bash" | "process" => "shell",
        "grep" | "glob" | "list" | "repo_map" | "web_search" | "web_fetch" | "find_symbol" => {
            "search"
        }
        _ => "other",
    }
}

/// Census of the messages that would be sent on the wire.
pub fn census_messages(messages: &[Message]) -> RequestCensus {
    let mut out = RequestCensus {
        input_tokens_est: hi_ai::estimate_messages_tokens(messages),
        ..RequestCensus::default()
    };
    for message in messages {
        match message.role {
            Role::User => {
                out.user_messages = out.user_messages.saturating_add(1);
                for block in &message.content {
                    if let Content::Text(text) = block {
                        out.user_message_chars = out
                            .user_message_chars
                            .saturating_add(text.chars().count() as u64);
                    }
                    census_block(&mut out, block);
                }
            }
            Role::Assistant => {
                out.assistant_messages = out.assistant_messages.saturating_add(1);
                for block in &message.content {
                    census_block(&mut out, block);
                }
            }
            Role::Tool => {
                out.tool_messages = out.tool_messages.saturating_add(1);
                for block in &message.content {
                    census_block(&mut out, block);
                }
            }
            Role::System => {
                for block in &message.content {
                    census_block(&mut out, block);
                }
            }
        }
    }
    out
}

fn census_block(out: &mut RequestCensus, block: &Content) {
    match block {
        Content::ToolResult { output, .. } => {
            let n = output.chars().count() as u64;
            out.max_tool_result_chars = out.max_tool_result_chars.max(n);
            if output.starts_with("[elided") {
                out.elided_tool_results = out.elided_tool_results.saturating_add(1);
            }
        }
        Content::Image { data, .. } => {
            let n = data.chars().count() as u64;
            out.max_image_chars = out.max_image_chars.max(n);
        }
        Content::Text(text) if text.starts_with("[elided image") => {
            let n = text.chars().count() as u64;
            out.max_image_chars = out.max_image_chars.max(n);
            out.elided_images = out.elided_images.saturating_add(1);
        }
        Content::Thinking { text, .. } => {
            let n = text.chars().count() as u64;
            out.max_thinking_chars = out.max_thinking_chars.max(n);
            if text.starts_with("[elided thinking") {
                out.elided_thinking = out.elided_thinking.saturating_add(1);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::{Content, Message};

    #[test]
    fn census_counts_elided_images_and_large_results() {
        let messages = vec![
            Message::system("sys"),
            Message {
                role: Role::User,
                content: vec![
                    Content::Text("see this".into()),
                    Content::Text("[elided image — was 2000000 chars]".into()),
                ],
            },
            Message::tool_result("r1", "x".repeat(20_000)),
        ];
        let census = census_messages(&messages);
        assert!(census.elided_images >= 1, "{census:?}");
        assert!(
            census.max_image_chars < 2_000,
            "stub, not the original screenshot: {}",
            census.max_image_chars
        );
        assert_eq!(census.max_tool_result_chars, 20_000);
        assert_eq!(census.user_messages, 1);
    }
}
