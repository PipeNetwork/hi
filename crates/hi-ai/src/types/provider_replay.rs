//! Opaque provider output preservation for stateless conversation replay.

use super::{Content, Message};
use serde_json::Value;

impl Message {
    /// Preserve a provider's complete output items for stateless continuation.
    /// Replaces older replay metadata and binds these items to the current role
    /// and content, so later edits or pruning cannot resurrect stale output.
    pub fn with_provider_replay(mut self, provider: &str, items: Vec<Value>) -> Self {
        self.content
            .retain(|block| !matches!(block, Content::ProviderReplay { .. }));
        let content_digest = self.provider_replay_digest();
        self.content.push(Content::ProviderReplay {
            provider: provider.to_owned(),
            content_digest,
            items,
        });
        self
    }

    /// Return opaque output items only for their originating provider and only
    /// while the message still matches the content that produced those items.
    pub fn provider_replay(&self, provider: &str) -> Option<&[Value]> {
        self.content.iter().find_map(|block| match block {
            Content::ProviderReplay {
                provider: origin,
                content_digest,
                items,
            } if origin == provider && content_digest == &self.provider_replay_digest() => {
                Some(items.as_slice())
            }
            _ => None,
        })
    }

    fn provider_replay_digest(&self) -> String {
        let content: Vec<_> = self
            .content
            .iter()
            .filter(|block| !matches!(block, Content::ProviderReplay { .. }))
            .collect();
        let encoded = serde_json::to_vec(&(self.role, content))
            .expect("provider-neutral message content serializes to JSON");
        blake3::hash(&encoded).to_hex().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Role, estimate_completion_output_tokens, estimate_messages_tokens};

    #[test]
    fn provider_replay_survives_serialization_without_visible_or_estimated_output() {
        let items = vec![
            serde_json::json!({"type": "reasoning", "encrypted_content": "opaque"}),
            serde_json::json!({"type": "message", "phase": "final_answer", "content": [{"type": "output_text", "text": "Done"}]}),
        ];
        let plain = Message::assistant(vec![Content::Text("Done".into())]);
        let message = plain
            .clone()
            .with_provider_replay("openai-responses", items.clone());
        let decoded: Message =
            serde_json::from_str(&serde_json::to_string(&message).unwrap()).unwrap();
        assert_eq!(
            decoded.provider_replay("openai-responses"),
            Some(items.as_slice())
        );
        assert_eq!(decoded.text(), "Done");
        assert_eq!(
            estimate_messages_tokens(std::slice::from_ref(&decoded)),
            estimate_messages_tokens(std::slice::from_ref(&plain))
        );
        assert_eq!(
            estimate_completion_output_tokens(&decoded.content),
            estimate_completion_output_tokens(&plain.content)
        );
    }

    #[test]
    fn provider_replay_is_invalidated_by_content_edits_and_pruning() {
        let message = Message::assistant(vec![
            Content::Text("Inspecting".into()),
            Content::ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                arguments: r#"{"path":"before"}"#.into(),
            },
        ])
        .with_provider_replay(
            "openai-responses",
            vec![serde_json::json!({"id": "output"})],
        );
        let mut text_changed = message.clone();
        text_changed.content[0] = Content::Text("Edited".into());
        assert!(text_changed.provider_replay("openai-responses").is_none());
        let mut arguments_changed = message.clone();
        if let Content::ToolCall { arguments, .. } = &mut arguments_changed.content[1] {
            *arguments = r#"{"path":"after"}"#.into();
        }
        assert!(
            arguments_changed
                .provider_replay("openai-responses")
                .is_none()
        );
        let mut pruned = message.clone();
        pruned.content.remove(1);
        assert!(pruned.provider_replay("openai-responses").is_none());
        let mut role_changed = message;
        role_changed.role = Role::User;
        assert!(role_changed.provider_replay("openai-responses").is_none());
    }

    #[test]
    fn provider_replay_is_provider_scoped_and_replaces_previous_metadata() {
        let message = Message::assistant(vec![Content::Text("Done".into())])
            .with_provider_replay("first", vec![serde_json::json!({"id": "old"})]);
        assert!(message.provider_replay("second").is_none());
        let items = vec![serde_json::json!({"id": "new"})];
        let message = message.with_provider_replay("second", items.clone());
        assert!(message.provider_replay("first").is_none());
        assert_eq!(message.provider_replay("second"), Some(items.as_slice()));
        assert_eq!(message.content.len(), 2);
    }
}
