//! Paint a loaded transcript into the TUI before the idle loop.

use ratatui::style::Style;
use ratatui::text::Line;

use crate::App;
use crate::render::dim;

pub(super) fn hydrate_transcript(app: &mut App, messages: &[hi_ai::Message]) {
    for message in messages {
        match message.role {
            hi_ai::Role::User => {
                let text = message.text();
                if !text.trim().is_empty() {
                    app.push_user_prompt(Line::styled(
                        format!("❯ {text}"),
                        Style::default().fg(crate::theme::theme().accent_user),
                    ));
                }
            }
            hi_ai::Role::Assistant => {
                let text = message.text();
                if !text.trim().is_empty() {
                    app.transcript
                        .push(crate::TranscriptEntry::AssistantMessage { text });
                }
                for block in &message.content {
                    if let hi_ai::Content::ToolCall {
                        name, arguments, ..
                    } = block
                    {
                        let preview: String = arguments.chars().take(80).collect();
                        app.push(Line::styled(format!("→ {name} {preview}"), dim()));
                    }
                }
            }
            hi_ai::Role::Tool => {
                for block in &message.content {
                    if let hi_ai::Content::ToolResult { output, .. } = block {
                        let preview: String = output.chars().take(120).collect();
                        app.push(Line::styled(format!("← {preview}"), dim()));
                    }
                }
            }
            hi_ai::Role::System => {}
        }
    }
    app.bump_transcript();
    app.follow();
}

/// Index `/retry` truncates from. The in-flight user line is already in
/// `messages`, unlike a normal turn which records the index before pushing.
pub(super) fn resume_last_turn_start(messages: &[hi_ai::Message]) -> usize {
    match messages.last() {
        Some(message) if message.role == hi_ai::Role::User => messages.len().saturating_sub(1),
        _ => messages.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::Message;

    #[test]
    fn resume_last_turn_start_points_at_inflight_user_line() {
        assert_eq!(
            resume_last_turn_start(&[Message::user("fix the parser")]),
            0
        );
        assert_eq!(
            resume_last_turn_start(&[Message::user("old"), Message::user("fix the parser")]),
            1
        );
        let empty: [Message; 0] = [];
        assert_eq!(resume_last_turn_start(&empty), 0);
        assert_eq!(
            resume_last_turn_start(&[
                Message::user("fix the parser"),
                Message::assistant(vec![hi_ai::Content::Text("partial".into())]),
            ]),
            2
        );
    }
}
