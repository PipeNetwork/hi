//! Context compaction strategies.
//!
//! When a session's history grows toward the model's context window, the agent
//! reclaims room. The cheap, deterministic strategy ([`elide_tool_outputs`])
//! shrinks the bulky tool output that dominates a coding session; the richer
//! ones (summarize / hybrid) make a model call. The agent wires these into a
//! two-tier auto policy (elide first, summarize only if still heavy) — see
//! `Agent::compact_with`.

use std::collections::HashMap;

use hi_ai::{Content, Message, Role};

/// User turns kept verbatim by `Hybrid`/`ElideToolOutput` by default.
pub const DEFAULT_KEEP_RECENT: usize = 3;

/// Rough characters-per-token ratio for the local size estimate.
const CHARS_PER_TOKEN: usize = 4;
/// Tool outputs shorter than this aren't worth eliding.
const ELIDE_MIN_CHARS: usize = 200;
/// Marker an elided output starts with, so elision is idempotent.
const ELIDED_MARK: &str = "[elided";

/// How a turn's history is compacted when the context fills up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompactionKind {
    /// Summarize the whole conversation into one brief and reset to it.
    Summarize,
    /// Keep the last `keep_recent` user turns verbatim; summarize everything
    /// older into a brief folded into the first kept turn.
    Hybrid { keep_recent: usize },
    /// Deterministic, no model call: replace the output of tool results older
    /// than `keep_recent` turns with a short stub.
    ElideToolOutput { keep_recent: usize },
}

impl CompactionKind {
    /// Map a `/compact <arg>` argument to a kind. Empty or unrecognized input
    /// returns `None`, so the caller can fall back to the configured default.
    pub fn from_arg(arg: &str) -> Option<Self> {
        match arg.trim().to_lowercase().as_str() {
            "full" | "summarize" | "summary" => Some(Self::Summarize),
            "hybrid" => Some(Self::Hybrid {
                keep_recent: DEFAULT_KEEP_RECENT,
            }),
            "elide" | "tools" | "tool" => Some(Self::ElideToolOutput {
                keep_recent: DEFAULT_KEEP_RECENT,
            }),
            _ => None,
        }
    }
}

/// Indices where each user turn starts (skips index 0, the system message).
pub(crate) fn user_turn_starts(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, m)| m.role == Role::User)
        .map(|(i, _)| i)
        .collect()
}

/// The split index: start of the `keep_recent`-th user turn from the end, so
/// `[1..split)` is "old" and `[split..)` is "recent". Returns `None` when there
/// aren't more than `keep_recent` user turns (too small to be worth compacting).
/// Splitting only ever falls on a user-turn boundary, so a tool call and its
/// result are never separated.
pub(crate) fn recent_split(messages: &[Message], keep_recent: usize) -> Option<usize> {
    if keep_recent == 0 {
        return (messages.len() > 1).then_some(messages.len());
    }
    let starts = user_turn_starts(messages);
    (starts.len() > keep_recent).then(|| starts[starts.len() - keep_recent])
}

/// A rough token estimate (~4 chars/token) across all message content — used to
/// decide whether deterministic elision freed enough to skip a summary call.
pub(crate) fn estimate_tokens(messages: &[Message]) -> u64 {
    let chars: usize = messages
        .iter()
        .flat_map(|m| &m.content)
        .map(|c| match c {
            Content::Text(t) => t.len(),
            Content::Thinking { text, .. } => text.len(),
            Content::ToolCall {
                arguments, name, ..
            } => arguments.len() + name.len(),
            Content::ToolResult { output, .. } => output.len(),
        })
        .sum();
    (chars / CHARS_PER_TOKEN) as u64
}

/// `call_id` → tool name, from the assistant's ToolCall blocks, so an elision
/// stub can name the tool it replaced.
fn tool_names(messages: &[Message]) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for message in messages {
        for block in &message.content {
            if let Content::ToolCall { id, name, .. } = block {
                names.insert(id.clone(), name.clone());
            }
        }
    }
    names
}

/// Replace tool-result outputs in `messages[1..up_to]` longer than
/// [`ELIDE_MIN_CHARS`] with a short stub, keeping the call/result skeleton (and
/// `call_id`) intact so tool pairing stays valid. Idempotent — already-elided
/// outputs are skipped. Returns the number of characters reclaimed.
pub(crate) fn elide_tool_outputs(messages: &mut [Message], up_to: usize) -> usize {
    let names = tool_names(messages);
    let mut freed = 0;
    let up_to = up_to.min(messages.len());
    for message in &mut messages[1..up_to] {
        for block in &mut message.content {
            if let Content::ToolResult { call_id, output } = block
                && output.len() > ELIDE_MIN_CHARS
                && !output.starts_with(ELIDED_MARK)
            {
                let lines = output.lines().count();
                let name = names.get(call_id).map_or("tool", String::as_str);
                freed += output.len();
                *output = format!("{ELIDED_MARK} {name} output — was {lines} lines]");
            }
        }
    }
    freed
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_ai::{Content, Message};

    fn convo() -> Vec<Message> {
        vec![
            Message::system("sys"),
            Message::user("turn one"),
            Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]),
            Message::tool_result("c1", "x".repeat(500)),
            Message::user("turn two"),
            Message::assistant(vec![Content::Text("answer".into())]),
        ]
    }

    #[test]
    fn turn_starts_and_split() {
        let m = convo();
        assert_eq!(user_turn_starts(&m), vec![1, 4]);
        // Two user turns: keeping 1 splits at the second (index 4).
        assert_eq!(recent_split(&m, 1), Some(4));
        // Keeping ≥ all turns → nothing old to compact.
        assert_eq!(recent_split(&m, 2), None);
        assert_eq!(recent_split(&m, 5), None);
        // Keeping zero turns means everything after the system message is old.
        assert_eq!(recent_split(&m, 0), Some(m.len()));
    }

    #[test]
    fn from_arg_maps_known_kinds() {
        assert_eq!(
            CompactionKind::from_arg("full"),
            Some(CompactionKind::Summarize)
        );
        assert_eq!(
            CompactionKind::from_arg("Hybrid"),
            Some(CompactionKind::Hybrid {
                keep_recent: DEFAULT_KEEP_RECENT
            })
        );
        assert!(matches!(
            CompactionKind::from_arg("elide"),
            Some(CompactionKind::ElideToolOutput { .. })
        ));
        assert_eq!(CompactionKind::from_arg(""), None);
        assert_eq!(CompactionKind::from_arg("bogus"), None);
    }

    #[test]
    fn elide_shrinks_old_outputs_only_and_is_idempotent() {
        let mut m = convo();
        // keep_recent = 1 → "turn two" is recent; c1's output (in turn one) is old.
        let split = recent_split(&m, 1).unwrap();
        let freed = elide_tool_outputs(&mut m, split);
        assert!(freed >= 500, "reclaimed the big output: {freed}");

        let outputs: Vec<String> = m
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert!(
            outputs[0].starts_with(ELIDED_MARK),
            "old elided: {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("read"),
            "names the tool: {}",
            outputs[0]
        );

        // Running again frees nothing (idempotent).
        assert_eq!(elide_tool_outputs(&mut m, split), 0);
    }

    #[test]
    fn elide_keeps_small_and_recent_outputs() {
        let mut m = vec![
            Message::system("sys"),
            Message::user("q"),
            Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "bash".into(),
                arguments: "{}".into(),
            }]),
            Message::tool_result("c1", "tiny"), // below threshold
        ];
        // No recent split (one turn) → caller passes len; small output untouched.
        let len = m.len();
        assert_eq!(elide_tool_outputs(&mut m, len), 0);
    }

    #[test]
    fn estimate_counts_outputs_and_args() {
        let m = vec![
            Message::user("a".repeat(40)),             // 10 tokens
            Message::tool_result("c", "b".repeat(40)), // 10 tokens
        ];
        assert_eq!(estimate_tokens(&m), 20);
    }
}
