//! Context compaction strategies.
//!
//! When a session's history grows toward the model's context window, the agent
//! reclaims room. The cheap, deterministic strategy ([`elide_tool_outputs`])
//! shrinks the bulky tool output that dominates a coding session; the richer
//! ones (summarize / hybrid) make a model call. The agent wires these into a
//! two-tier auto policy (elide first, summarize only if still heavy) — see
//! `Agent::compact_with`.

use std::collections::{HashMap, HashSet};

use hi_ai::{Content, Message, Role};

mod elision;
pub(crate) use elision::repair_legacy_elided_thinking;
use elision::{elide_old_image, elide_old_thinking_in};

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
    /// Drop conversation history without a summary. Keeps the stable system
    /// prompt, goal/decisions/memory, and the current user task.
    FreshWindow,
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
            "window" | "fresh" => Some(Self::FreshWindow),
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

/// Newest read/write/edit result for each path stays verbatim even when it
/// falls outside the global keep-recent window. Eliding `index.html` so the
/// model "cannot safely edit" it is the live inspection stall.
fn newest_file_result_ids(messages: &[Message]) -> HashSet<String> {
    let mut calls = HashMap::new();
    for message in messages {
        for block in &message.content {
            if let Content::ToolCall {
                id,
                name,
                arguments,
            } = block
            {
                calls.insert(id.clone(), (name.clone(), arguments.clone()));
            }
        }
    }
    let mut seen_paths = HashSet::new();
    let mut keep = HashSet::new();
    for message in messages.iter().rev() {
        for block in message.content.iter().rev() {
            let Content::ToolResult { call_id, .. } = block else {
                continue;
            };
            let Some((name, arguments)) = calls.get(call_id) else {
                continue;
            };
            if !matches!(
                name.as_str(),
                "read" | "write" | "edit" | "multi_edit" | "apply_patch"
            ) {
                continue;
            }
            for path in hi_tools::target_paths(name, arguments) {
                if path.is_empty() {
                    continue;
                }
                if seen_paths.insert(path) {
                    keep.insert(call_id.clone());
                }
            }
        }
    }
    keep
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
                Content::Thinking { .. } => {
                    freed += elide_old_thinking_in(block);
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
                Content::Thinking { .. } => {
                    freed += elide_old_thinking_in(block);
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
    let newest_file_ids = newest_file_result_ids(messages);
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
                Content::ToolResult { call_id, .. } if newest_file_ids.contains(call_id) => {}
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
                Content::Thinking { .. } if !keep_thinking => {
                    freed += elide_old_thinking_in(block);
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
mod tests;
