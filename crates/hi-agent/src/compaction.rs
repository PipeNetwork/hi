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

/// Tool outputs shorter than this aren't worth eliding.
const ELIDE_MIN_CHARS: usize = 200;
/// Marker an elided output starts with, so elision is idempotent.
const ELIDED_MARK: &str = "[elided";
/// Payload fields on executed `write`/`edit`/`apply_patch`/`bash` calls. Below
/// this, a short command or identifier stays; above it, the bytes are already
/// on disk (or in the tool result) and must not be resent every round.
const ELIDE_ARG_MIN_CHARS: usize = 400;

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
    /// Elide-first, summarize-only-the-conversational-tail. Keep the last
    /// `keep_recent` user turns verbatim (with their tool results elided, not
    /// summarized, so the call/result skeleton stays). For turns older than the
    /// recent window: elide the ones that carry tool results (their *shape* —
    /// which tool, which file — stays; only bulky output is stubbed), and
    /// summarize only the tool-free Q&A turns into a brief folded into the
    /// first kept turn. This is the right default for tool-heavy coding
    /// sessions, where the recent tool results matter most and a summary of
    /// them would be lossy in exactly the wrong way.
    ElideThenSummarizeTail { keep_recent: usize },
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
            "tail" | "default" => Some(Self::ElideThenSummarizeTail {
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

/// Whether the slice `[1..split)` contains any tool results. Used by the
/// elide-then-summarize-tail strategy to decide whether the "old" region is
/// tool-heavy (elide it, keep the skeleton) or conversational (summarize it).
#[allow(dead_code)]
pub(crate) fn has_tool_results(messages: &[Message], split: usize) -> bool {
    let up_to = split.min(messages.len());
    messages[1..up_to]
        .iter()
        .flat_map(|m| &m.content)
        .any(|c| matches!(c, Content::ToolResult { .. }))
}

/// The "old" conversational tail (pure Q&A user turns) that the
/// elide-then-summarize-tail strategy summarizes. A user turn counts as
/// conversational iff the assistant reply that follows it made **no tool
/// calls** — a user turn that triggered tool use is part of a tool-bearing
/// turn and gets elided (skeleton kept), not summarized. Returns the messages
/// of those conversational old turns (in `[1..split)`) in order, so the
/// summarizer sees the actual Q&A exchange, not just the prompts.
pub(crate) fn conversational_tail(messages: &[Message], split: usize) -> Vec<Message> {
    let up_to = split.min(messages.len());
    let starts = user_turn_starts(messages);
    let mut out = Vec::new();
    for (idx, &start) in starts.iter().enumerate() {
        if start >= up_to {
            break;
        }
        let end = if idx + 1 < starts.len() {
            starts[idx + 1].min(up_to)
        } else {
            up_to
        };
        // Was the assistant reply in [start..end) tool-free? If there was no
        // assistant message, treat the turn as conversational (a bare user
        // turn, e.g. the last partial turn).
        let has_tool_content = turn_has_tool_content(&messages[start..end]);
        if !has_tool_content {
            out.extend_from_slice(&messages[start..end]);
        }
    }
    out
}

/// Old turns that contain tool use/results and should stay in the transcript
/// when `ElideThenSummarizeTail` summarizes only the Q&A tail. This preserves
/// the complete user turn, not just the tool-bearing messages, so the model
/// keeps the prompt that caused each tool call and any final answer after the
/// result.
pub(crate) fn tool_bearing_turns(messages: &[Message], split: usize) -> Vec<Message> {
    let up_to = split.min(messages.len());
    let starts = user_turn_starts(messages);
    let mut out = Vec::new();

    // Preserve any legacy prefix before the first user turn if it contains
    // tool content. New transcripts should not have this shape, but older
    // compactions may have left assistant/tool skeletons at the front.
    let first_turn_start = starts.first().copied().unwrap_or(up_to).min(up_to);
    if first_turn_start > 1 && turn_has_tool_content(&messages[1..first_turn_start]) {
        out.extend_from_slice(&messages[1..first_turn_start]);
    }

    for (idx, &start) in starts.iter().enumerate() {
        if start >= up_to {
            break;
        }
        let end = if idx + 1 < starts.len() {
            starts[idx + 1].min(up_to)
        } else {
            up_to
        };
        if turn_has_tool_content(&messages[start..end]) {
            out.extend_from_slice(&messages[start..end]);
        }
    }
    out
}

fn turn_has_tool_content(messages: &[Message]) -> bool {
    messages.iter().any(|m| {
        m.content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. } | Content::ToolResult { .. }))
    })
}

/// A rough UTF-8-byte token estimate (~4 bytes/token) across all message content — used to
/// decide whether deterministic elision freed enough to skip a summary call.
pub(crate) fn estimate_tokens(messages: &[Message]) -> u64 {
    hi_ai::estimate_messages_tokens(messages)
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
    if up_to <= 1 {
        return 0;
    }
    for message in &mut messages[1..up_to] {
        for block in &mut message.content {
            match block {
                Content::ToolResult { call_id, output }
                    if output.len() > ELIDE_MIN_CHARS && !output.starts_with(ELIDED_MARK) =>
                {
                    let lines = output.lines().count();
                    let name = names.get(call_id).map_or("tool", String::as_str);
                    freed += output.len();
                    *output = format!("{ELIDED_MARK} {name} output — was {lines} lines]");
                }
                Content::ToolCall { arguments, .. } => {
                    freed += elide_old_tool_arguments_in(arguments);
                }
                Content::Thinking { text, .. } => {
                    freed += elide_old_thinking_in(text);
                }
                Content::Image { .. } => {
                    freed += elide_old_image(block);
                }
                _ => {}
            }
        }
    }
    freed
}

/// Stub bulky tool-call payloads in `messages[1..up_to]` without touching
/// results. Used on session resume so recent writes stay quoteable.
pub(crate) fn elide_old_tool_arguments(messages: &mut [Message], up_to: usize) -> usize {
    let up_to = up_to.min(messages.len());
    if up_to <= 1 {
        return 0;
    }
    let mut freed = 0;
    for message in &mut messages[1..up_to] {
        for block in &mut message.content {
            match block {
                Content::ToolCall { arguments, .. } => {
                    freed += elide_old_tool_arguments_in(arguments);
                }
                Content::Thinking { text, .. } => {
                    freed += elide_old_thinking_in(text);
                }
                Content::Image { .. } => {
                    freed += elide_old_image(block);
                }
                _ => {}
            }
        }
    }
    freed
}

const ELIDE_THINKING_MIN_CHARS: usize = 400;

fn elide_old_thinking_in(text: &mut String) -> usize {
    if text.chars().count() <= ELIDE_THINKING_MIN_CHARS || text.starts_with("[elided thinking") {
        return 0;
    }
    let n = text.chars().count();
    let freed = text.len();
    *text = format!("[elided thinking — was {n} chars]");
    freed
}

fn elide_old_image(block: &mut Content) -> usize {
    let Content::Image { data, .. } = block else {
        return 0;
    };
    let n = data.chars().count();
    let freed = data.len();
    *block = Content::Text(format!("[elided image — was {n} chars]"));
    freed
}

fn elide_old_tool_arguments_in(arguments: &mut String) -> usize {
    let Some(shrunk) = shrink_tool_arguments(arguments) else {
        return 0;
    };
    let freed = arguments.len().saturating_sub(shrunk.len());
    *arguments = shrunk;
    freed
}

/// Replace bulky tool-result outputs anywhere in the conversation except the
/// newest `keep_recent_results` tool results. This is used inside a single long
/// turn, where there may be no old user-turn boundary yet but repeated model
/// rounds would otherwise resend every previous tool payload.
pub(crate) fn elide_tool_outputs_except_recent(
    messages: &mut [Message],
    keep_recent_results: usize,
) -> usize {
    if messages.len() <= 1 {
        return 0;
    }

    let names = tool_names(messages);
    let mut recent_ids = std::collections::HashSet::new();
    let mut kept = 0usize;
    'outer: for message in messages.iter().rev() {
        for block in message.content.iter().rev() {
            if let Content::ToolResult { call_id, .. } = block {
                recent_ids.insert(call_id.clone());
                kept += 1;
                if kept >= keep_recent_results {
                    break 'outer;
                }
            }
        }
    }

    let mut seen = 0usize;
    for message in messages.iter().rev() {
        for block in message.content.iter().rev() {
            if matches!(block, Content::ToolResult { .. }) {
                seen += 1;
            }
        }
    }

    let mut eligible = seen.saturating_sub(keep_recent_results);
    let mut freed = 0usize;
    for message in &mut messages[1..] {
        let keep_thinking = message
            .content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { id, .. } if recent_ids.contains(id)));
        for block in &mut message.content {
            match block {
                Content::ToolResult { call_id, output } if eligible > 0 => {
                    eligible -= 1;
                    if output.len() > ELIDE_MIN_CHARS && !output.starts_with(ELIDED_MARK) {
                        let lines = output.lines().count();
                        let name = names.get(call_id).map_or("tool", String::as_str);
                        freed += output.len();
                        *output = format!("{ELIDED_MARK} {name} output — was {lines} lines]");
                    }
                }
                Content::ToolCall { id, arguments, .. } if !recent_ids.contains(id) => {
                    freed += elide_old_tool_arguments_in(arguments);
                }
                Content::Thinking { text, .. } if !keep_thinking => {
                    freed += elide_old_thinking_in(text);
                }
                _ => {}
            }
        }
    }
    freed
}

/// Shrink bulky string fields on an executed tool call so the payload is not
/// resent on every later model round. Keeps JSON valid and leaves identifiers
/// (`path`, `id`, `name`, …) intact.
pub(crate) fn shrink_tool_arguments(arguments: &str) -> Option<String> {
    if arguments.len() <= ELIDE_ARG_MIN_CHARS {
        return None;
    }
    let mut value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let mut changed = false;
    shrink_json_strings(&mut value, &mut changed);
    if !changed {
        return None;
    }
    serde_json::to_string(&value).ok()
}

fn shrink_json_strings(value: &mut serde_json::Value, changed: &mut bool) {
    match value {
        serde_json::Value::String(s) => {
            if s.chars().count() > ELIDE_ARG_MIN_CHARS && !s.starts_with("[elided") {
                let n = s.chars().count();
                *s = format!("[elided — {n} chars]");
                *changed = true;
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                shrink_json_strings(item, changed);
            }
        }
        serde_json::Value::Object(map) => {
            for (key, item) in map.iter_mut() {
                if matches!(
                    key.as_str(),
                    "path"
                        | "paths"
                        | "id"
                        | "name"
                        | "server"
                        | "tool"
                        | "status"
                        | "title"
                        | "glob"
                        | "pattern"
                ) {
                    continue;
                }
                shrink_json_strings(item, changed);
            }
        }
        _ => {}
    }
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
    fn shrink_tool_arguments_stubs_write_payload_and_keeps_path() {
        let args = serde_json::json!({
            "path": "src/lib.rs",
            "content": "fn main() {}\n".repeat(80),
        })
        .to_string();
        let shrunk = shrink_tool_arguments(&args).expect("large write should shrink");
        let value: serde_json::Value = serde_json::from_str(&shrunk).unwrap();
        assert_eq!(value["path"], "src/lib.rs");
        let content = value["content"].as_str().unwrap();
        assert!(content.starts_with("[elided"), "{content}");
        assert!(content.contains("chars"), "{content}");
        assert!(
            shrunk.len() < args.len() / 4,
            "payload must drop: {} vs {}",
            shrunk.len(),
            args.len()
        );
        assert!(shrink_tool_arguments(&shrunk).is_none(), "idempotent");
        assert!(shrink_tool_arguments(r#"{"path":"a.rs","content":"short"}"#).is_none());
    }

    #[test]
    fn elide_shrinks_old_write_arguments_with_results() {
        let mut m = vec![
            Message::system("sys"),
            Message::user("write it"),
            Message::assistant(vec![Content::ToolCall {
                id: "w1".into(),
                name: "write".into(),
                arguments: serde_json::json!({"path":"a.rs","content":"x".repeat(800)}).to_string(),
            }]),
            Message::tool_result("w1", "wrote a.rs"),
        ];
        let len = m.len();
        let freed = elide_tool_outputs(&mut m, len);
        assert!(freed > 0, "should reclaim write payload");
        let Content::ToolCall { arguments, .. } = &m[2].content[0] else {
            panic!("expected tool call");
        };
        assert!(arguments.contains("a.rs"), "{arguments}");
        assert!(arguments.contains("[elided"), "{arguments}");
        assert!(!arguments.contains(&"x".repeat(800)), "{arguments}");
    }

    #[test]
    fn in_turn_elide_keeps_recent_write_arguments() {
        let recent_args = serde_json::json!({
            "path": "new.rs",
            "content": "y".repeat(800),
        })
        .to_string();
        let mut m = vec![Message::system("sys"), Message::user("q")];
        m.push(Message::assistant(vec![Content::ToolCall {
            id: "old".into(),
            name: "write".into(),
            arguments: serde_json::json!({"path":"old.rs","content":"x".repeat(800)}).to_string(),
        }]));
        m.push(Message::tool_result("old", "wrote old.rs"));
        m.push(Message::assistant(vec![Content::ToolCall {
            id: "new".into(),
            name: "write".into(),
            arguments: recent_args.clone(),
        }]));
        m.push(Message::tool_result("new", "wrote new.rs"));
        let freed = elide_tool_outputs_except_recent(&mut m, 1);
        assert!(freed > 0);
        let Content::ToolCall { arguments: old, .. } = &m[2].content[0] else {
            panic!("old call");
        };
        let Content::ToolCall { arguments: new, .. } = &m[4].content[0] else {
            panic!("new call");
        };
        assert!(old.contains("[elided"), "{old}");
        assert_eq!(new, &recent_args, "newest write must stay quoteable");
    }

    fn user_with_image(text: &str, data: String) -> Message {
        Message {
            role: Role::User,
            content: vec![
                Content::Text(text.into()),
                Content::Image {
                    data,
                    media_type: "image/png".into(),
                },
            ],
        }
    }

    #[test]
    fn elide_stubs_old_images_and_keeps_recent() {
        let mut m = vec![
            Message::system("sys"),
            user_with_image("old shot", "A".repeat(2_000)),
            Message::assistant(vec![Content::Text("ok".into())]),
            user_with_image("new shot", "B".repeat(400)),
            Message::assistant(vec![Content::Text("done".into())]),
        ];
        let split = recent_split(&m, 1).unwrap();
        let freed = elide_tool_outputs(&mut m, split);
        assert!(freed > 0);
        assert!(
            matches!(&m[1].content[1], Content::Text(text) if text.starts_with("[elided image")),
            "old image stubbed: {:?}",
            m[1].content[1]
        );
        assert!(
            matches!(&m[3].content[1], Content::Image { data, .. } if data == &"B".repeat(400)),
            "recent image stays"
        );
        assert_eq!(elide_tool_outputs(&mut m, split), 0, "idempotent");
    }

    #[test]
    fn elide_stubs_old_thinking_and_keeps_recent() {
        let mut m = vec![
            Message::system("sys"),
            Message::user("q1"),
            Message::assistant(vec![Content::Thinking {
                text: "T".repeat(800),
                signature: Some("sig-old".into()),
            }]),
            Message::user("q2"),
            Message::assistant(vec![
                Content::Thinking {
                    text: "U".repeat(800),
                    signature: Some("sig-new".into()),
                },
                Content::ToolCall {
                    id: "c-new".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            ]),
            Message::tool_result("c-new", "ok"),
        ];
        let split = recent_split(&m, 1).unwrap();
        let freed = elide_tool_outputs(&mut m, split);
        assert!(freed > 0);
        let Content::Thinking { text, signature } = &m[2].content[0] else {
            panic!("old thinking");
        };
        assert!(text.starts_with("[elided thinking"), "{text}");
        assert_eq!(signature.as_deref(), Some("sig-old"));
        let Content::Thinking { text, .. } = &m[4].content[0] else {
            panic!("new thinking");
        };
        assert_eq!(text, &"U".repeat(800), "recent thinking stays");
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
    fn in_turn_elide_keeps_newest_tool_results() {
        let mut m = vec![Message::system("sys"), Message::user("q")];
        for i in 1..=4 {
            let id = format!("c{i}");
            m.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
            m.push(Message::tool_result(
                &id,
                format!("{i}\n{}", "x".repeat(500)),
            ));
        }

        let freed = elide_tool_outputs_except_recent(&mut m, 2);
        assert!(freed >= 1000, "reclaimed old tool outputs: {freed}");

        let outputs: Vec<String> = m
            .iter()
            .flat_map(|msg| &msg.content)
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.clone()),
                _ => None,
            })
            .collect();
        assert!(outputs[0].starts_with(ELIDED_MARK), "{outputs:?}");
        assert!(outputs[1].starts_with(ELIDED_MARK), "{outputs:?}");
        assert!(outputs[2].starts_with("3\n"), "{outputs:?}");
        assert!(outputs[3].starts_with("4\n"), "{outputs:?}");
        assert_eq!(elide_tool_outputs_except_recent(&mut m, 2), 0);
    }

    #[test]
    fn estimate_counts_outputs_and_args() {
        let m = vec![
            Message::user("a".repeat(40)),             // 10 tokens
            Message::tool_result("c", "b".repeat(40)), // 10 tokens
        ];
        assert_eq!(estimate_tokens(&m), 21);
    }

    #[test]
    fn has_tool_results_and_conversational_tail_partition_the_old_region() {
        let m = convo(); // system, q1, read call, big result, q2, answer
        let split = recent_split(&m, 1).unwrap(); // q2 onward is recent → split at q2
        // The old region [1..split) is turn one: q1 + read call + big result.
        assert!(
            has_tool_results(&m, split),
            "old region has the read result"
        );
        // Turn one's assistant reply made a tool call, so it's NOT part of the
        // conversational tail — the tail is empty for this conversation.
        let tail = conversational_tail(&m, split);
        assert!(
            tail.is_empty(),
            "tool turn excluded from Q&A tail: {tail:?}"
        );

        // A conversation with a real Q&A turn: system, q1 + text answer (Q&A),
        // q2 + tool turn (recent).
        let m2 = vec![
            Message::system("sys"),
            Message::user("q1"),
            Message::assistant(vec![Content::Text("a1".into())]), // no tool call → Q&A
            Message::user("q2"),
            Message::assistant(vec![Content::Text("a2".into())]),
        ];
        let split2 = recent_split(&m2, 1).unwrap(); // q2 is recent
        let tail2 = conversational_tail(&m2, split2);
        assert_eq!(tail2.len(), 2, "q1 + a1 form the Q&A tail: {tail2:?}");
        assert_eq!(tail2[0].role, Role::User);
        assert_eq!(tail2[1].role, Role::Assistant);
    }

    #[test]
    fn tool_bearing_turns_preserves_the_complete_tool_turn() {
        let m = vec![
            Message::system("sys"),
            Message::user("q1: inspect file"),
            Message::assistant(vec![Content::ToolCall {
                id: "c1".into(),
                name: "read".into(),
                arguments: "{}".into(),
            }]),
            Message::tool_result("c1", "file contents"),
            Message::assistant(vec![Content::Text("a1: found the issue".into())]),
            Message::user("q2: explain rust ownership"),
            Message::assistant(vec![Content::Text("a2: ownership answer".into())]),
            Message::user("q3: fix it"),
        ];
        let split = recent_split(&m, 1).unwrap();

        let tool_turns = tool_bearing_turns(&m, split);

        assert_eq!(
            tool_turns.iter().map(|msg| msg.role).collect::<Vec<_>>(),
            vec![Role::User, Role::Assistant, Role::Tool, Role::Assistant]
        );
        assert_eq!(tool_turns[0].text(), "q1: inspect file");
        assert!(matches!(
            &tool_turns[2].content[0],
            Content::ToolResult { output, .. } if output == "file contents"
        ));
        assert_eq!(tool_turns[3].text(), "a1: found the issue");
        assert!(
            tool_turns.iter().all(|msg| !msg.text().contains("q2")),
            "tool-free Q&A turn should be summarized instead: {tool_turns:?}"
        );
    }
}
