//! Prompt-cache health: compares each model request's messages and tool catalog
//! against the previous request. A provider prompt cache (Anthropic breakpoints
//! or OpenAI-style implicit prefix caching) can only reuse the unchanged prefix,
//! so an append-only request with a frozen catalog is a cache hit. Message
//! rewrites and tool-schema churn are breaks — measured here so `hi metrics`
//! can see prefix-cache health instead of guessing from bills.

#[derive(Clone, Debug, Default)]
pub(crate) struct PrefixStability {
    /// Per-message content hashes of the most recent request sent.
    prev_hashes: Vec<u64>,
    /// Fingerprint of the tool schemas sent with that request.
    prev_tools_digest: String,
    /// This turn's append-only request count (messages and tools unchanged).
    pub(crate) stable_rounds: u32,
    /// This turn's prefix-breaking request count (messages or tools).
    pub(crate) break_rounds: u32,
    /// This turn's breaks caused by a different tool catalog.
    pub(crate) tool_break_rounds: u32,
    /// Smallest message-index divergence observed this turn (0 = system).
    pub(crate) earliest_break: Option<u32>,
}

impl PrefixStability {
    /// Reset the per-turn counters; the previous request's hashes persist so
    /// the first request of a new turn is measured against the last request
    /// of the previous one (turn boundaries are where the context block
    /// strip legitimately breaks the prefix once).
    pub(crate) fn begin_turn(&mut self) {
        self.stable_rounds = 0;
        self.break_rounds = 0;
        self.tool_break_rounds = 0;
        self.earliest_break = None;
    }

    /// Record the request about to be sent. Counters are read into turn
    /// telemetry at turn end. `tool_mode` is not part of the fingerprint:
    /// Anthropic/OpenAI `tool_choice` sits outside cached content blocks.
    pub(crate) fn record_request(
        &mut self,
        messages: &[hi_ai::Message],
        tools: &[hi_ai::ToolSpec],
    ) {
        let hashes = message_hashes(messages);
        let tools_digest = hi_tools::envelope::canonical_tool_schema_digest(tools);
        if !self.prev_hashes.is_empty() {
            let shared = self.prev_hashes.len().min(hashes.len());
            let divergence = (0..shared)
                .find(|&i| self.prev_hashes[i] != hashes[i])
                .or_else(|| (hashes.len() < self.prev_hashes.len()).then_some(hashes.len()));
            let tools_changed = self.prev_tools_digest != tools_digest;
            if divergence.is_none() && !tools_changed {
                self.stable_rounds = self.stable_rounds.saturating_add(1);
            } else {
                self.break_rounds = self.break_rounds.saturating_add(1);
                if let Some(index) = divergence {
                    let index = index as u32;
                    self.earliest_break = Some(match self.earliest_break {
                        Some(previous) => previous.min(index),
                        None => index,
                    });
                }
                if tools_changed {
                    self.tool_break_rounds = self.tool_break_rounds.saturating_add(1);
                }
            }
        }
        self.prev_hashes = hashes;
        self.prev_tools_digest = tools_digest;
    }
}

fn message_hashes(messages: &[hi_ai::Message]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    messages
        .iter()
        .map(|message| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::mem::discriminant(&message.role).hash(&mut hasher);
            for block in &message.content {
                match block {
                    hi_ai::Content::Text(text) => {
                        0u8.hash(&mut hasher);
                        text.hash(&mut hasher);
                    }
                    hi_ai::Content::Thinking { text, signature } => {
                        1u8.hash(&mut hasher);
                        text.hash(&mut hasher);
                        signature.hash(&mut hasher);
                    }
                    hi_ai::Content::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        2u8.hash(&mut hasher);
                        id.hash(&mut hasher);
                        name.hash(&mut hasher);
                        arguments.hash(&mut hasher);
                    }
                    hi_ai::Content::ToolResult { call_id, output } => {
                        3u8.hash(&mut hasher);
                        call_id.hash(&mut hasher);
                        output.hash(&mut hasher);
                    }
                    hi_ai::Content::Image { data, media_type } => {
                        4u8.hash(&mut hasher);
                        data.hash(&mut hasher);
                        media_type.hash(&mut hasher);
                    }
                }
            }
            hasher.finish()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::PrefixStability;
    use hi_ai::{Content, Message, ToolSpec};

    fn read_tool() -> ToolSpec {
        ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    fn bash_tool() -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a command".into(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn counts_append_only_requests_and_prefix_breaks() {
        let mut tracker = PrefixStability::default();
        let first = vec![Message::system("sys"), Message::user("question")];
        tracker.record_request(&first, &[]);
        assert_eq!(
            tracker.stable_rounds + tracker.break_rounds,
            0,
            "the first request has no predecessor to compare against"
        );

        let mut appended = first.clone();
        appended.push(Message::assistant(vec![Content::Text("answer".into())]));
        tracker.record_request(&appended, &[]);
        assert_eq!(tracker.stable_rounds, 1, "pure append is cache-friendly");
        assert_eq!(tracker.break_rounds, 0);

        let mut rewritten = appended.clone();
        rewritten[0] = Message::system("sys CHANGED");
        tracker.record_request(&rewritten, &[]);
        assert_eq!(tracker.break_rounds, 1);
        assert_eq!(
            tracker.earliest_break,
            Some(0),
            "a system rewrite breaks the prefix at message 0"
        );

        let truncated = rewritten[..2].to_vec();
        tracker.record_request(&truncated, &[]);
        assert_eq!(tracker.break_rounds, 2);
        assert_eq!(tracker.earliest_break, Some(0), "keeps the minimum");

        tracker.begin_turn();
        assert_eq!(tracker.stable_rounds, 0);
        assert_eq!(tracker.tool_break_rounds, 0);
        assert_eq!(tracker.earliest_break, None);
        let mut next_turn = truncated.clone();
        next_turn.push(Message::user("next"));
        tracker.record_request(&next_turn, &[]);
        assert_eq!(
            tracker.stable_rounds, 1,
            "previous hashes survive the per-turn counter reset"
        );
    }

    #[test]
    fn tool_catalog_change_is_a_prefix_break() {
        let mut tracker = PrefixStability::default();
        let messages = vec![Message::system("sys"), Message::user("question")];
        tracker.record_request(&messages, &[read_tool()]);

        let mut appended = messages.clone();
        appended.push(Message::assistant(vec![Content::Text("ok".into())]));
        tracker.record_request(&appended, &[read_tool()]);
        assert_eq!(tracker.stable_rounds, 1);
        assert_eq!(tracker.break_rounds, 0);
        assert_eq!(tracker.tool_break_rounds, 0);

        tracker.record_request(&appended, &[read_tool(), bash_tool()]);
        assert_eq!(tracker.stable_rounds, 1);
        assert_eq!(tracker.break_rounds, 1);
        assert_eq!(tracker.tool_break_rounds, 1);
        assert_eq!(
            tracker.earliest_break, None,
            "a tool-only break has no message index"
        );

        tracker.record_request(&appended, &[read_tool(), bash_tool()]);
        assert_eq!(tracker.stable_rounds, 2, "frozen catalog is append-stable");
    }
}
