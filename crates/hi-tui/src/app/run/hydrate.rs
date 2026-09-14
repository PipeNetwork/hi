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
