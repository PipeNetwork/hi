//! Grok-style conversation compact: summarize at 85% of the context window.
//!
//! Manual `/compact [context]` and auto-compact both ask the model (no tools)
//! for a structured summary, then replace live history with the original user
//! query plus that summary. The session file is rewritten to the compacted
//! conversation, matching Grok's live session after compact.

use hi_ai::{Content, Message, Role};

use crate::prompt::SYSTEM_PROMPT;

/// Grok's default `[session] auto_compact_threshold_percent`.
pub(crate) const AUTO_COMPACT_THRESHOLD_PERCENT: u64 = 85;

/// Stub older tool bodies once occupancy hits this percent of the window.
/// Matches the handbook's "elide bulky tools" threshold.
pub(crate) const CHEAP_SHRINK_THRESHOLD_PERCENT: u64 = 45;

/// Newest tool-result bodies kept verbatim; older ones become stubs.
pub(crate) const KEEP_LAST_TOOL_RESULTS: usize = 6;

/// Current-turn tool bodies above this (chars/4) get stubbed oldest-first so
/// a 2M advertised window cannot keep six 15k-token reads verbatim.
/// Unique current-turn `read` paths are exempt: a live ~/chat review lost
/// 24 distinct files to this cap, then re-fetched them until inspect-repeat
/// stopped the turn with no verdict.
pub(crate) const CURRENT_TURN_TOOL_BUDGET_TOKENS: u64 = 24_000;

const COMPACT_PROMPT: &str = "\
Your task is to produce a faithful, concise summary of the conversation so far \
so that a successor assistant can continue the work seamlessly after the earlier \
turns are discarded. The successor will see the user's original query plus this \
summary. Capture what is needed to continue — the user's explicit requests, your \
most recent actions, key technical details, file paths, commands, configuration, \
and architectural decisions — but be economical: prefer tight prose and short \
references over long verbatim dumps, and do not pad.

CRITICAL: If earlier turns include a prior compaction summary (marked with \
<conversation_summary> tags or a \"This session is being continued\" preamble), \
treat it as authoritative for the early history and carry its still-relevant \
information forward into your new summary.

Think through the conversation in your private reasoning before writing. Output \
the final summary inside a single <summary>...</summary> block, organized into \
the following numbered sections. Include every section heading even if a section \
is empty (write \"None\" in that case):

1. Primary Request and Intent
2. Key Technical Concepts
3. Files and Code Sections
4. Errors and Fixes
5. Problem Solving
6. All User Messages: list every user message in order (not this compaction prompt)
7. Pending Tasks
8. Current Work
9. Optional Next Step

IMPORTANT: Do NOT call or use any tools. Respond with ONLY the \
<summary>...</summary> block.";

pub(crate) fn compact_prompt(user_context: Option<&str>) -> String {
    match user_context.map(str::trim).filter(|text| !text.is_empty()) {
        Some(context) => format!(
            "{COMPACT_PROMPT}\n\n**User-provided context for this compaction:**\n{context}\n\n\
Please incorporate this context into your summary, ensuring it is prominently \
addressed in the relevant sections."
        ),
        None => COMPACT_PROMPT.to_string(),
    }
}

pub(crate) fn parse_summary(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if let Some(inner) =
        extract_tag(trimmed, "summary").or_else(|| extract_tag(trimmed, "conversation_summary"))
    {
        let inner = inner.trim();
        if inner.is_empty() {
            return None;
        }
        return Some(inner.to_string());
    }
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn extract_tag<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text[start..].find(&close)? + start;
    Some(&text[start..end])
}

/// Stub summary used when the compact model fails and occupancy is already
/// over the window. Keeps the first and latest user lines so a successor
/// (or this turn after rewrite) can continue without a 50M-token prompt.
pub(crate) fn emergency_summary(occupancy_percent: u64) -> String {
    format!(
        "Local compact: conversation was at {occupancy_percent}% of the context \
window and model compact failed. Earlier tool transcripts were dropped. \
Continue from the latest user request."
    )
}

pub(crate) fn apply_summary(messages: &[Message], summary: &str) -> Vec<Message> {
    let first_user = messages
        .iter()
        .find(|message| message.role == Role::User)
        .cloned();
    let last_user_index = messages
        .iter()
        .rposition(|message| message.role == Role::User);
    let body = format!(
        "This session is being continued from a previous conversation that ran \
out of context. The summary below covers the earlier portion of the conversation.\n\n\
<conversation_summary>\n{summary}\n</conversation_summary>"
    );
    let mut out = Vec::new();
    if let Some(first) = first_user {
        out.push(first);
    }
    out.push(Message::assistant(vec![Content::Text(body)]));
    if let Some(index) = last_user_index {
        let already = out
            .first()
            .is_some_and(|message| message.text() == messages[index].text());
        if !already {
            out.push(messages[index].clone());
        }
    }
    out
}

fn is_tool_stub(output: &str) -> bool {
    output.ends_with(" · omitted")
}

fn tool_names(messages: &[Message]) -> std::collections::HashMap<String, String> {
    let mut names = std::collections::HashMap::new();
    for message in messages {
        for block in &message.content {
            if let Content::ToolCall { id, name, .. } = block {
                names.insert(id.clone(), name.clone());
            }
        }
    }
    names
}

fn tool_call_path(arguments: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()?
        .get("path")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
}

fn tool_paths(messages: &[Message]) -> std::collections::HashMap<String, String> {
    let mut paths = std::collections::HashMap::new();
    for message in messages {
        for block in &message.content {
            if let Content::ToolCall { id, arguments, .. } = block
                && let Some(path) = tool_call_path(arguments)
            {
                paths.insert(id.clone(), path);
            }
        }
    }
    paths
}

/// Keep the newest body per `read` path; stub older fetches of the same file
/// so cheap-shrink does not hide a unique file while leaving duplicate
/// `server.rs` reads intact (which makes the model fetch it again).
fn keep_newest_unique_read_paths(
    messages: &[Message],
    eligible: &[(usize, usize)],
    stub: &mut [bool],
    names: &std::collections::HashMap<String, String>,
) {
    let paths = tool_paths(messages);
    let mut seen = std::collections::HashSet::new();
    for i in (0..eligible.len()).rev() {
        let (mi, ci) = eligible[i];
        let Content::ToolResult { call_id, .. } = &messages[mi].content[ci] else {
            continue;
        };
        if names.get(call_id).map(String::as_str) != Some("read") {
            continue;
        }
        let Some(path) = paths.get(call_id) else {
            continue;
        };
        if seen.insert(path.clone()) {
            stub[i] = false;
        }
    }
}

fn stub_older_duplicate_paths(
    messages: &[Message],
    eligible: &[(usize, usize)],
    stub: &mut [bool],
    names: &std::collections::HashMap<String, String>,
) {
    let paths = tool_paths(messages);
    let mut seen = std::collections::HashSet::new();
    for i in (0..eligible.len()).rev() {
        if stub[i] {
            continue;
        }
        let (mi, ci) = eligible[i];
        let Content::ToolResult { call_id, .. } = &messages[mi].content[ci] else {
            continue;
        };
        if names.get(call_id).map(String::as_str) != Some("read") {
            continue;
        }
        let Some(path) = paths.get(call_id) else {
            continue;
        };
        if !seen.insert(path.clone()) {
            stub[i] = true;
        }
    }
}

fn stub_tool_result(name: &str, output: &str, path: Option<&str>) -> String {
    match path {
        Some(path) if !path.is_empty() => {
            format!("{name} {path} · {} chars · omitted", output.len())
        }
        _ => format!("{name} · {} chars · omitted", output.len()),
    }
}

/// Deterministic history shrink: stub old tool bodies and drop stale thinking.
/// Never drops a user line. Idempotent.
///
/// `preserve_current_turn`: when true (the 45% path), tool results after the
/// latest user message stay verbatim so an in-progress review can still see
/// the files it just read. When false (the 85% path, just before model
/// compact), older current-turn results may be stubbed too, keeping the last
/// [`KEEP_LAST_TOOL_RESULTS`].
#[cfg(test)]
pub(crate) fn cheap_shrink(messages: &[Message]) -> (Vec<Message>, bool) {
    cheap_shrink_with(messages, true)
}

pub(crate) fn cheap_shrink_with(
    messages: &[Message],
    preserve_current_turn: bool,
) -> (Vec<Message>, bool) {
    let names = tool_names(messages);
    let last_user = messages
        .iter()
        .rposition(|message| message.role == Role::User);
    let mut eligible = Vec::new();
    for (mi, message) in messages.iter().enumerate() {
        for (ci, block) in message.content.iter().enumerate() {
            if !matches!(block, Content::ToolResult { .. }) {
                continue;
            }
            let current_turn = last_user.is_some_and(|user| mi > user);
            if preserve_current_turn && current_turn {
                continue;
            }
            eligible.push((mi, ci));
        }
    }
    let keep_from = eligible.len().saturating_sub(KEEP_LAST_TOOL_RESULTS);
    let mut stub = vec![false; eligible.len()];
    for (i, flag) in stub.iter_mut().enumerate() {
        if i < keep_from {
            *flag = true;
        }
    }
    keep_newest_unique_read_paths(messages, &eligible, &mut stub, &names);
    stub_older_duplicate_paths(messages, &eligible, &mut stub, &names);
    // Prefer dropping bulky inspect bodies (read/list/grep) so a later
    // `bash`/`cargo test` result still fits in the current-turn budget.
    // Never drop the newest body of a unique `read` path to stay under
    // the budget — those files are why the model issued the reads.
    let mut current_kept_tokens = current_turn_kept_tokens(messages, &eligible, &stub, last_user);
    while current_kept_tokens > CURRENT_TURN_TOOL_BUDGET_TOKENS {
        let Some(i) = next_current_turn_stub_index(messages, &eligible, &stub, last_user, &names)
        else {
            break;
        };
        stub[i] = true;
        current_kept_tokens = current_turn_kept_tokens(messages, &eligible, &stub, last_user);
    }
    let last_assistant = messages
        .iter()
        .rposition(|message| message.role == Role::Assistant);

    let mut out = messages.to_vec();
    let mut shrunk = false;
    for (mi, message) in out.iter_mut().enumerate() {
        if Some(mi) == last_assistant {
            continue;
        }
        let before = message.content.len();
        message
            .content
            .retain(|block| !matches!(block, Content::Thinking { .. }));
        if message.content.len() != before {
            shrunk = true;
        }
    }
    let paths = tool_paths(&out);
    for (i, &(mi, ci)) in eligible.iter().enumerate() {
        if !stub[i] {
            continue;
        }
        let Some(block) = out.get_mut(mi).and_then(|m| m.content.get_mut(ci)) else {
            continue;
        };
        let Content::ToolResult { call_id, output } = block else {
            continue;
        };
        if is_tool_stub(output) {
            continue;
        }
        let name = names.get(call_id).map(String::as_str).unwrap_or("tool");
        let path = paths.get(call_id).map(String::as_str);
        *output = stub_tool_result(name, output, path);
        shrunk = true;
    }
    (out, shrunk)
}

fn tool_result_tokens(block: &Content) -> u64 {
    match block {
        Content::ToolResult { output, .. } => output.len() as u64 / 4,
        _ => 0,
    }
}

fn bulky_inspect_tool(name: &str) -> bool {
    matches!(
        name,
        "read" | "list" | "grep" | "glob" | "repo_map" | "find_symbol"
    )
}

fn current_turn_kept_tokens(
    messages: &[Message],
    eligible: &[(usize, usize)],
    stub: &[bool],
    last_user: Option<usize>,
) -> u64 {
    let mut tokens = 0u64;
    for (i, &(mi, ci)) in eligible.iter().enumerate() {
        if stub[i] || last_user.is_none_or(|user| mi <= user) {
            continue;
        }
        tokens = tokens.saturating_add(tool_result_tokens(&messages[mi].content[ci]));
    }
    tokens
}

fn unique_path_keeper_indices(
    messages: &[Message],
    eligible: &[(usize, usize)],
    stub: &[bool],
    names: &std::collections::HashMap<String, String>,
) -> std::collections::HashSet<usize> {
    let paths = tool_paths(messages);
    let mut seen = std::collections::HashSet::new();
    let mut keepers = std::collections::HashSet::new();
    for i in (0..eligible.len()).rev() {
        if stub[i] {
            continue;
        }
        let (mi, ci) = eligible[i];
        let Content::ToolResult { call_id, .. } = &messages[mi].content[ci] else {
            continue;
        };
        if names.get(call_id).map(String::as_str) != Some("read") {
            continue;
        }
        let Some(path) = paths.get(call_id) else {
            continue;
        };
        if seen.insert(path.clone()) {
            keepers.insert(i);
        }
    }
    keepers
}

fn next_current_turn_stub_index(
    messages: &[Message],
    eligible: &[(usize, usize)],
    stub: &[bool],
    last_user: Option<usize>,
    names: &std::collections::HashMap<String, String>,
) -> Option<usize> {
    let newest = eligible
        .iter()
        .rposition(|&(mi, _)| last_user.is_some_and(|user| mi > user))?;
    let keepers = unique_path_keeper_indices(messages, eligible, stub, names);
    let inspect = eligible.iter().enumerate().position(|(i, &(mi, ci))| {
        if i == newest || stub[i] || keepers.contains(&i) || last_user.is_none_or(|user| mi <= user)
        {
            return false;
        }
        let Content::ToolResult { call_id, .. } = &messages[mi].content[ci] else {
            return false;
        };
        bulky_inspect_tool(names.get(call_id).map(String::as_str).unwrap_or("tool"))
    });
    inspect.or_else(|| {
        eligible.iter().enumerate().position(|(i, &(mi, _))| {
            i != newest
                && !stub[i]
                && !keepers.contains(&i)
                && last_user.is_some_and(|user| mi > user)
        })
    })
}

/// Fingerprints of inspect tools whose bodies are now omitted stubs, so the
/// inspect-repeat ledger can allow a re-fetch of a file the model can no
/// longer see.
pub(crate) fn omitted_inspect_fingerprints(messages: &[Message]) -> Vec<String> {
    let mut calls = std::collections::HashMap::new();
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
    let mut out = Vec::new();
    for message in messages {
        for block in &message.content {
            let Content::ToolResult { call_id, output } = block else {
                continue;
            };
            if !is_tool_stub(output) {
                continue;
            }
            let Some((name, arguments)) = calls.get(call_id) else {
                continue;
            };
            if !hi_tools::is_inspect_tool(name) {
                continue;
            }
            out.push(crate::liveness::tool_fingerprint(name, arguments));
        }
    }
    out
}

/// Fingerprints of inspect results that still have a live (non-stub) body.
/// Cheap-shrink must not forget these — they are the only copy of that file.
pub(crate) fn live_inspect_fingerprints(messages: &[Message]) -> std::collections::HashSet<String> {
    let mut calls = std::collections::HashMap::new();
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
    let mut out = std::collections::HashSet::new();
    for message in messages {
        for block in &message.content {
            let Content::ToolResult { call_id, output } = block else {
                continue;
            };
            if is_tool_stub(output) {
                continue;
            }
            let Some((name, arguments)) = calls.get(call_id) else {
                continue;
            };
            if !hi_tools::is_inspect_tool(name) {
                continue;
            }
            out.insert(crate::liveness::tool_fingerprint(name, arguments));
        }
    }
    out
}

/// Inspect fingerprints that were live or stubbed before a rewrite and are no
/// longer live after it (stubbed or dropped). Re-fetching those files is allowed.
pub(crate) fn inspect_keys_to_forget(previous: &[Message], next: &[Message]) -> Vec<String> {
    let mut before = live_inspect_fingerprints(previous);
    before.extend(omitted_inspect_fingerprints(previous));
    let live = live_inspect_fingerprints(next);
    before
        .into_iter()
        .filter(|key| !live.contains(key))
        .collect()
}

pub(crate) fn current_turn_tool_tokens(messages: &[Message]) -> u64 {
    let Some(last_user) = messages
        .iter()
        .rposition(|message| message.role == Role::User)
    else {
        return 0;
    };
    let mut tokens = 0u64;
    for message in &messages[last_user + 1..] {
        for block in &message.content {
            if let Content::ToolResult { output, .. } = block
                && !is_tool_stub(output)
            {
                tokens = tokens.saturating_add(output.len() as u64 / 4);
            }
        }
    }
    tokens
}

pub(crate) fn estimate_message_tokens(messages: &[Message]) -> u64 {
    let mut chars = SYSTEM_PROMPT.len();
    for message in messages {
        for block in &message.content {
            chars += match block {
                Content::Text(text) | Content::Thinking { text, .. } => text.len(),
                Content::ToolCall {
                    name, arguments, ..
                } => name.len() + arguments.len(),
                Content::ToolResult { output, .. } => output.len(),
                Content::Image { data, .. } => data.len(),
                Content::ProviderReplay { .. } => 0,
            };
        }
    }
    chars as u64 / 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipe::test_support::{MockPipe, Scripted, text_chunk, usage_chunk};
    use crate::ui::TestUi;
    use crate::{
        Harness, HarnessConfig, JsonlSession, LoadedSession, PendingTurn, TurnCancellation,
        TypesafeSettings,
    };
    use hi_ai::Usage;
    use hi_tools::ProcessRunner;
    use hi_tools::sandbox::SandboxPolicy;
    use std::fs;
    use std::path::PathBuf;

    fn test_harness(url: &str, workspace: PathBuf) -> Harness {
        let state = workspace.join(".hi");
        let runner =
            ProcessRunner::new_with_policy(&workspace, SandboxPolicy::Off).expect("runner");
        let tools =
            crate::ToolHost::new_with_runner(workspace.clone(), state.clone(), runner).unwrap();
        let mut config = HarnessConfig::pipe(workspace, "pk_test");
        config.base_url = url.to_string();
        config.state_root = state;
        Harness::new_with_tools(config, tools).unwrap()
    }

    fn test_harness_jev(pipe_url: &str, typesafe_url: &str, workspace: PathBuf) -> Harness {
        let state = workspace.join(".hi");
        let runner =
            ProcessRunner::new_with_policy(&workspace, SandboxPolicy::Off).expect("runner");
        let tools =
            crate::ToolHost::new_with_runner(workspace.clone(), state.clone(), runner).unwrap();
        let mut config = HarnessConfig::pipe(workspace, "pk_test");
        config.base_url = pipe_url.to_string();
        config.state_root = state;
        config.jev_compact = true;
        config.typesafe = TypesafeSettings {
            api_key: Some("test-key".into()),
            base_url: typesafe_url.to_string(),
            model: "jev-latest".into(),
            min_confidence: 0.55,
            auto: true,
            effort: true,
        };
        Harness::new_with_tools(config, tools).unwrap()
    }

    fn jev_history() -> Vec<Message> {
        let mut messages = vec![Message::user("review this")];
        for i in 0..8 {
            let id = format!("c{i}");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: format!(r#"{{"path":"src/f{i}.rs"}}"#),
            }]));
            messages.push(Message::tool_result(
                &id,
                format!("body-{i}-{}", "x".repeat(400)),
            ));
        }
        messages.push(Message::user("keep going"));
        messages
    }

    #[test]
    fn parse_summary_reads_tagged_block() {
        let text = "noise\n<summary>\n1. Primary Request and Intent: ship it\n</summary>\n";
        assert_eq!(
            parse_summary(text).as_deref(),
            Some("1. Primary Request and Intent: ship it")
        );
    }

    #[test]
    fn apply_summary_keeps_first_user_and_latest_prompt() {
        let messages = vec![
            Message::user("review this"),
            Message::tool_result("c1", "x".repeat(1000)),
            Message::user("build all of that"),
        ];
        let compacted = apply_summary(&messages, "did a review, then started building");
        assert_eq!(compacted.len(), 3);
        assert_eq!(compacted[0].text(), "review this");
        assert!(compacted[1].text().contains("<conversation_summary>"));
        assert!(compacted[1].text().contains("started building"));
        assert_eq!(compacted[2].text(), "build all of that");
    }

    #[test]
    fn cheap_shrink_stubs_old_tool_bodies_keeps_last_six() {
        let mut messages = vec![Message::user("review this")];
        for i in 0..8 {
            let id = format!("c{i}");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
            messages.push(Message::tool_result(&id, "x".repeat(100)));
        }
        messages.push(Message::user("keep going"));
        let (shrunk, changed) = cheap_shrink(&messages);
        assert!(changed);
        let outputs: Vec<_> = shrunk
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(outputs.len(), 8);
        assert!(outputs[0].contains("omitted"), "{}", outputs[0]);
        assert!(outputs[1].contains("omitted"));
        assert_eq!(outputs[2], &"x".repeat(100));
        assert_eq!(outputs[7], &"x".repeat(100));
        assert_eq!(shrunk[0].text(), "review this");
        assert_eq!(shrunk.last().unwrap().text(), "keep going");
        let (_, again) = cheap_shrink(&shrunk);
        assert!(!again, "second shrink must be idempotent");
    }

    #[test]
    fn cheap_shrink_does_not_stub_current_turn_reads() {
        let mut messages = vec![Message::user("review this")];
        for i in 0..8 {
            let id = format!("c{i}");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
            messages.push(Message::tool_result(&id, "x".repeat(100)));
        }
        let (shrunk, changed) = cheap_shrink(&messages);
        assert!(
            !changed
                || shrunk.iter().flat_map(|m| m.content.iter()).all(|c| {
                    !matches!(c, Content::ToolResult { output, .. } if output.contains("omitted"))
                }),
            "in-progress turn reads must stay verbatim so the model can see the files"
        );
        let (aggressive, aggro) = cheap_shrink_with(&messages, false);
        assert!(aggro);
        let omitted = aggressive
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert_eq!(omitted, 2, "aggressive shrink still keeps the last six");
    }

    #[test]
    fn cheap_shrink_caps_current_turn_tool_tokens() {
        let mut messages = vec![Message::user("review this")];
        for i in 0..6 {
            let id = format!("big{i}");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
            messages.push(Message::tool_result(&id, "y".repeat(60_000)));
        }
        assert!(current_turn_tool_tokens(&messages) > CURRENT_TURN_TOOL_BUDGET_TOKENS);
        let (shrunk, changed) = cheap_shrink_with(&messages, false);
        assert!(changed);
        let omitted = shrunk
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert_eq!(
            omitted, 5,
            "token budget must stub older huge current-turn reads, got {omitted}"
        );
        assert!(current_turn_tool_tokens(&shrunk) <= CURRENT_TURN_TOOL_BUDGET_TOKENS);
    }

    #[test]
    fn cheap_shrink_stubs_older_duplicate_path_reads_first() {
        let mut messages = vec![Message::user("review this")];
        for (i, path) in ["src/server.rs", "src/server.rs", "src/state.rs"]
            .iter()
            .enumerate()
        {
            let id = format!("r{i}");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: format!(r#"{{"path":"{path}"}}"#),
            }]));
            messages.push(Message::tool_result(
                &id,
                format!("body-{path}-").repeat(200),
            ));
        }
        let (shrunk, changed) = cheap_shrink_with(&messages, false);
        assert!(changed);
        let outputs: Vec<&str> = shrunk
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| match c {
                Content::ToolResult { output, .. } => Some(output.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            outputs[0].contains("omitted"),
            "older server.rs read must stub, got {}",
            outputs[0]
        );
        assert!(
            outputs[0].contains("src/server.rs"),
            "stub should name the omitted path, got {}",
            outputs[0]
        );
        assert!(
            !outputs[1].contains("omitted"),
            "newest server.rs read must stay, got {}",
            outputs[1]
        );
        assert!(
            !outputs[2].contains("omitted"),
            "unique state.rs must stay, got {}",
            outputs[2]
        );
    }

    #[test]
    fn cheap_shrink_keeps_unique_current_turn_read_paths() {
        let mut messages = vec![Message::user("build all of that")];
        for i in 0..8 {
            let id = format!("r{i}");
            let path = format!("src/f{i}.rs");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: format!(r#"{{"path":"{path}"}}"#),
            }]));
            messages.push(Message::tool_result(&id, format!("body-{path}")));
        }
        let (shrunk, _) = cheap_shrink_with(&messages, false);
        let omitted = shrunk
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert_eq!(
            omitted, 0,
            "KEEP_LAST_6 must not hide unique in-progress files"
        );
    }

    #[test]
    fn cheap_shrink_does_not_budget_stub_unique_current_turn_paths() {
        let mut messages = vec![Message::user("review for any major issues and fix")];
        for i in 0..8 {
            let id = format!("r{i}");
            let path = format!("src/f{i}.rs");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: format!(r#"{{"path":"{path}"}}"#),
            }]));
            messages.push(Message::tool_result(&id, "x".repeat(20_000)));
        }
        assert!(current_turn_tool_tokens(&messages) > CURRENT_TURN_TOOL_BUDGET_TOKENS);
        let (shrunk, _) = cheap_shrink_with(&messages, false);
        let omitted = shrunk
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert_eq!(
            omitted, 0,
            "token budget must not hide unique current-turn files, got {omitted}"
        );
        assert!(
            current_turn_tool_tokens(&shrunk) > CURRENT_TURN_TOOL_BUDGET_TOKENS,
            "unique-path exemption may leave the turn over the inspect budget"
        );
    }

    #[test]
    fn cheap_shrink_keeps_current_turn_bash_over_reads() {
        let mut messages = vec![Message::user("review this")];
        for i in 0..6 {
            let id = format!("read{i}");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
            messages.push(Message::tool_result(&id, "y".repeat(60_000)));
        }
        messages.push(Message::assistant(vec![Content::ToolCall {
            id: "bash1".into(),
            name: "bash".into(),
            arguments: r#"{"command":"cargo test"}"#.into(),
        }]));
        messages.push(Message::tool_result(
            "bash1",
            "test planted_review_fix_e2e ... FAILED\nassert_eq!(2 + 2, 5)",
        ));
        let (shrunk, changed) = cheap_shrink_with(&messages, false);
        assert!(changed);
        let bash = shrunk
            .iter()
            .flat_map(|m| m.content.iter())
            .find_map(|c| match c {
                Content::ToolResult { call_id, output } if call_id == "bash1" => {
                    Some(output.as_str())
                }
                _ => None,
            })
            .expect("bash result present");
        assert!(
            bash.contains("planted_review_fix_e2e"),
            "cargo test output must survive inspect shrinking: {bash}"
        );
        assert!(!bash.contains("omitted"));
    }

    #[test]
    fn cheap_shrink_drops_stale_thinking() {
        let messages = vec![
            Message::user("go"),
            Message::assistant(vec![
                Content::Thinking {
                    text: "old thought".into(),
                    signature: None,
                },
                Content::Text("first".into()),
            ]),
            Message::assistant(vec![
                Content::Thinking {
                    text: "new thought".into(),
                    signature: None,
                },
                Content::Text("second".into()),
            ]),
        ];
        let (shrunk, changed) = cheap_shrink(&messages);
        assert!(changed);
        assert!(
            shrunk[1]
                .content
                .iter()
                .all(|c| !matches!(c, Content::Thinking { .. }))
        );
        assert!(
            shrunk[2]
                .content
                .iter()
                .any(|c| matches!(c, Content::Thinking { .. }))
        );
    }

    #[test]
    fn emergency_summary_mentions_occupancy() {
        let text = emergency_summary(400);
        assert!(text.contains("400%"));
        assert!(text.contains("compact failed"));
    }

    #[test]
    fn apply_summary_does_not_duplicate_the_only_user_prompt() {
        let messages = vec![
            Message::user("fix the bug"),
            Message::tool_result("c1", "log"),
            Message::tool_result("c2", "log2"),
        ];
        let compacted = apply_summary(&messages, "found the bug");
        assert_eq!(compacted.len(), 2);
        assert_eq!(compacted[0].text(), "fix the bug");
        assert!(compacted[1].text().contains("found the bug"));
    }

    #[test]
    fn auto_compact_ignores_cumulative_session_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        harness.apply_loaded_session(LoadedSession {
            messages: vec![
                Message::user("go"),
                Message::tool_result("c1", "tiny"),
                Message::assistant(vec![Content::Text("ok".into())]),
                Message::user("again"),
            ],
            usage: Usage {
                input_tokens: 200_000,
                output_tokens: 10,
                ..Usage::default()
            },
            ..LoadedSession::default()
        });
        assert!(
            !harness.should_auto_compact(),
            "lifetime session tokens must not trigger compact"
        );
    }

    #[test]
    fn occupancy_percent_uses_last_request_not_lifetime() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        harness.record_context_occupancy(Usage {
            input_tokens: 10_000,
            context_occupancy: 10_000,
            ..Usage::default()
        });
        assert!(harness.occupancy_percent() < AUTO_COMPACT_THRESHOLD_PERCENT);
        harness.record_context_occupancy(Usage {
            input_tokens: 120_000,
            context_occupancy: 120_000,
            ..Usage::default()
        });
        harness.apply_loaded_session(LoadedSession {
            messages: vec![
                Message::user("a"),
                Message::tool_result("c1", "b"),
                Message::assistant(vec![Content::Text("c".into())]),
                Message::user("d"),
            ],
            ..LoadedSession::default()
        });
        harness.record_context_occupancy(Usage {
            input_tokens: 120_000,
            context_occupancy: 120_000,
            ..Usage::default()
        });
        assert!(harness.should_auto_compact());
    }

    #[test]
    fn occupancy_floor_from_estimate_triggers_cheap_shrink() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        let mut messages = vec![Message::user("review")];
        for i in 0..12 {
            messages.push(Message::tool_result(format!("c{i}"), "y".repeat(20_000)));
        }
        messages.push(Message::user("again"));
        harness.apply_loaded_session(LoadedSession {
            messages,
            ..LoadedSession::default()
        });
        assert!(
            harness.should_cheap_shrink(),
            "bulky tool bodies must count even with no Pipe occupancy yet"
        );
        let mut ui = TestUi::default();
        assert!(harness.apply_cheap_shrink_if_needed(&mut ui));
        assert!(ui.statuses.iter().any(|s| s.contains("shrunk")));
        let omitted = harness
            .messages()
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert_eq!(omitted, 6, "12 results keep the last 6, stub the older 6");
        assert!(
            !harness.should_cheap_shrink(),
            "after stubbing, estimate should drop below 45%"
        );
    }

    #[test]
    fn cheap_shrink_persists_and_keeps_pending_turn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let state = dir.path().join(".hi");
        let runner =
            ProcessRunner::new_with_policy(dir.path(), SandboxPolicy::Off).expect("runner");
        let tools =
            crate::ToolHost::new_with_runner(dir.path().to_path_buf(), state.clone(), runner)
                .unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = state;
        config.session_path = Some(path.clone());
        let mut harness = Harness::new_with_tools(config, tools).unwrap();
        let mut messages = vec![Message::user("review this")];
        for i in 0..8 {
            messages.push(Message::tool_result(format!("c{i}"), "z".repeat(40_000)));
        }
        messages.push(Message::user("keep going"));
        harness.apply_loaded_session(LoadedSession {
            messages,
            pending_turn: Some(PendingTurn {
                turn_index: 4,
                started_unix_ms: 99,
                pre_checkpoint: Some("pre".into()),
            }),
            ..LoadedSession::default()
        });
        let mut ui = TestUi::default();
        assert!(harness.apply_cheap_shrink_if_needed(&mut ui));
        assert!(harness.pending_turn().is_some());
        let loaded = JsonlSession::load(&path).unwrap();
        assert!(loaded.pending_turn.is_some(), "pending_turn must survive");
        let omitted = loaded
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert!(
            omitted >= 2,
            "session file must contain stubbed tool bodies"
        );
    }

    #[test]
    fn cheap_shrink_failed_rewrite_restores_messages_and_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let state = dir.path().join(".hi");
        let runner =
            ProcessRunner::new_with_policy(dir.path(), SandboxPolicy::Off).expect("runner");
        let tools =
            crate::ToolHost::new_with_runner(dir.path().to_path_buf(), state.clone(), runner)
                .unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = state;
        config.session_path = Some(path.clone());
        let mut harness = Harness::new_with_tools(config, tools).unwrap();
        let mut messages = vec![Message::user("review this")];
        for i in 0..8 {
            messages.push(Message::tool_result(format!("c{i}"), "z".repeat(40_000)));
        }
        messages.push(Message::user("keep going"));
        harness.apply_loaded_session(LoadedSession {
            messages,
            ..LoadedSession::default()
        });
        assert!(harness.persist_snapshot());

        let mut tmp = path.as_os_str().to_os_string();
        tmp.push(".tmp");
        fs::create_dir(&tmp).unwrap();

        let mut ui = TestUi::default();
        assert!(
            !harness.apply_cheap_shrink_if_needed(&mut ui),
            "rewrite behind a directory tmp must fail closed"
        );
        assert!(
            harness.should_cheap_shrink(),
            "failed persist must restore occupancy so shrink can retry"
        );
        let omitted = harness
            .messages()
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert_eq!(omitted, 0, "in-memory shrink must roll back");
        let loaded = JsonlSession::load(&path).unwrap();
        let file_omitted = loaded
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter(
                |c| matches!(c, Content::ToolResult { output, .. } if output.contains("omitted")),
            )
            .count();
        assert_eq!(
            file_omitted, 0,
            "session file must stay the unshrunk original"
        );
        assert_ne!(
            harness.liveness().snapshot().invariant.map(|inv| inv.code),
            Some(hi_liveness::InvariantCode::SessionAppendFailed),
            "a rolled-back rewrite must not auto-repair the live turn"
        );
    }

    #[tokio::test]
    async fn cheap_shrink_runs_mid_turn_without_stopping() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("still going"),
            usage_chunk(20, 4),
        ])]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut harness = test_harness(&server.url, dir.path().to_path_buf());
        let mut messages = vec![Message::user("review")];
        for i in 0..12 {
            messages.push(Message::tool_result(format!("c{i}"), "y".repeat(20_000)));
        }
        messages.push(Message::user("again"));
        harness.apply_loaded_session(LoadedSession {
            messages,
            ..LoadedSession::default()
        });
        let mut ui = TestUi::default();
        let outcome = harness
            .run_turn_cancellable("wrap up", &mut ui, TurnCancellation::new())
            .await
            .unwrap();
        assert_eq!(outcome.stop_reason, crate::TurnStopReason::Completed);
        assert!(ui.texts.join("").contains("still going"));
        assert!(
            ui.statuses.iter().any(|s| s.contains("shrunk")),
            "expected cheap shrink before the Pipe round, got {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn compact_replaces_history_with_model_summary() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("<summary>\nreviewed src/main.rs and cargo test is green\n</summary>"),
            usage_chunk(20, 8),
        ])]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut harness = test_harness(&server.url, dir.path().to_path_buf());
        harness.apply_loaded_session(LoadedSession {
            messages: vec![
                Message::user("review this"),
                Message::tool_result("c1", "fn main"),
                Message::assistant(vec![Content::Text("looking".into())]),
                Message::user("keep going"),
            ],
            ..LoadedSession::default()
        });
        let mut ui = TestUi::default();
        let changed = harness
            .compact(None, &mut ui, &TurnCancellation::new())
            .await
            .unwrap();
        assert!(changed);
        assert_eq!(harness.messages()[0].text(), "review this");
        assert!(harness.messages()[1].text().contains("cargo test is green"));
        assert_eq!(harness.messages()[2].text(), "keep going");
        assert!(
            harness
                .messages()
                .iter()
                .all(|message| message.role != Role::Tool)
        );
    }

    fn served(id: &str, window: Option<u32>) -> hi_ai::ServedModel {
        hi_ai::ServedModel {
            id: id.into(),
            context_window: window,
            max_output_tokens: None,
            price: None,
            provider_label: None,
            status: None,
            available: true,
            availability_reason: None,
            capabilities: Vec::new(),
        }
    }

    #[test]
    fn context_window_defaults_to_128k() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let harness = Harness::new(config).unwrap();
        assert_eq!(harness.context_window(), 128_000);
        assert_eq!(harness.context_window_source(), "default");
        assert_eq!(harness.max_tokens(), crate::DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn completed_plan_is_dismissed_for_the_next_turn() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        let mut snapshot = harness.snapshot();
        snapshot.plan = vec![
            hi_tools::PlanStep {
                title: "read the file".into(),
                status: hi_tools::PlanStatus::Done,
            },
            hi_tools::PlanStep {
                title: "edit the file".into(),
                status: hi_tools::PlanStatus::Done,
            },
        ];
        harness.restore_snapshot(snapshot);
        assert!(harness.dismiss_completed_plan());
        assert!(harness.current_plan().is_empty());
    }

    #[test]
    fn in_progress_plan_survives_dismiss() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        let mut snapshot = harness.snapshot();
        snapshot.plan = vec![hi_tools::PlanStep {
            title: "still going".into(),
            status: hi_tools::PlanStatus::Active,
        }];
        harness.restore_snapshot(snapshot);
        assert!(!harness.dismiss_completed_plan());
        assert_eq!(harness.current_plan().len(), 1);
    }

    #[test]
    fn advertised_output_cap_drives_implicit_coding_max_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        let mut model = served(&harness.model(), Some(212_992));
        model.max_output_tokens = Some(16_384);
        harness.remember_model_windows(&[model]);
        assert_eq!(harness.max_tokens(), 16_384);
    }

    #[test]
    fn explicit_max_tokens_is_clamped_to_advertised_output_cap() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        config.max_tokens = 2_048;
        config.max_tokens_explicit = true;
        let mut harness = Harness::new(config).unwrap();
        let mut model = served(&harness.model(), Some(212_992));
        model.max_output_tokens = Some(16_384);
        harness.remember_model_windows(&[model]);
        assert_eq!(harness.max_tokens(), 2_048);

        harness.set_max_tokens(65_536);
        assert_eq!(harness.max_tokens(), 16_384);
    }

    #[test]
    fn working_window_not_advertised_window_drives_shrink() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        harness.remember_model_windows(&[served(&harness.model(), Some(2_000_000))]);
        assert_eq!(harness.context_window(), 2_000_000);
        assert_eq!(harness.working_context_window(), 128_000);
        harness.record_context_occupancy(Usage {
            input_tokens: 120_000,
            context_occupancy: 120_000,
            ..Usage::default()
        });
        harness.apply_loaded_session(LoadedSession {
            messages: vec![
                Message::user("a"),
                Message::tool_result("c1", "b"),
                Message::assistant(vec![Content::Text("c".into())]),
                Message::user("d"),
            ],
            ..LoadedSession::default()
        });
        harness.record_context_occupancy(Usage {
            input_tokens: 120_000,
            context_occupancy: 120_000,
            ..Usage::default()
        });
        assert!(
            harness.should_auto_compact(),
            "120k is 93% of the 128k working window even when the provider advertises 2M"
        );
        assert!(harness.should_cheap_shrink());
    }

    #[test]
    fn large_current_turn_reads_shrink_under_2m_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        harness.remember_model_windows(&[served(&harness.model(), Some(2_000_000))]);
        let mut messages = vec![Message::user("review")];
        for i in 0..8 {
            let id = format!("r{i}");
            messages.push(Message::assistant(vec![Content::ToolCall {
                id: id.clone(),
                name: "read".into(),
                arguments: "{}".into(),
            }]));
            messages.push(Message::tool_result(&id, "z".repeat(60_000)));
        }
        harness.apply_loaded_session(LoadedSession {
            messages,
            ..LoadedSession::default()
        });
        assert!(harness.should_stub_current_turn());
        let mut ui = TestUi::default();
        assert!(harness.apply_cheap_shrink_if_needed(&mut ui));
        assert!(
            current_turn_tool_tokens(harness.messages()) <= CURRENT_TURN_TOOL_BUDGET_TOKENS,
            "current-turn reads must shrink even under a 2M advertised window"
        );
    }

    #[test]
    fn unknown_model_keeps_previous_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        let mut harness = Harness::new(config).unwrap();
        let current = harness.model();
        harness.remember_model_windows(&[served(&current, Some(2_000_000))]);
        harness.set_model("pipe/unknown-model".into());
        assert_eq!(
            harness.context_window(),
            2_000_000,
            "missing metadata must not snap back to 128k"
        );
        assert_eq!(harness.context_window_source(), "default");
        harness.set_model(current);
        assert_eq!(harness.context_window(), 2_000_000);
        assert_eq!(harness.context_window_source(), "models");
    }

    #[test]
    fn no_auto_compact_skips_shrink_in_the_turn_loop() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.state_root = dir.path().join(".hi");
        config.auto_compact = false;
        let mut harness = Harness::new(config).unwrap();
        let mut messages = vec![Message::user("review")];
        for i in 0..12 {
            messages.push(Message::tool_result(format!("c{i}"), "y".repeat(20_000)));
        }
        messages.push(Message::user("again"));
        harness.apply_loaded_session(LoadedSession {
            messages,
            ..LoadedSession::default()
        });
        assert!(harness.should_cheap_shrink());
        assert!(!harness.should_auto_compact());
        assert!(!harness.auto_compact());
    }

    #[tokio::test]
    async fn resume_incomplete_shrinks_before_pipe() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("resumed"),
            usage_chunk(20, 4),
        ])]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut harness = test_harness(&server.url, dir.path().to_path_buf());
        let mut messages = vec![Message::user("review")];
        for i in 0..12 {
            messages.push(Message::tool_result(format!("c{i}"), "y".repeat(20_000)));
        }
        messages.push(Message::user("keep going"));
        harness.apply_loaded_session(LoadedSession {
            messages,
            pending_turn: Some(PendingTurn {
                turn_index: 2,
                started_unix_ms: 1,
                pre_checkpoint: None,
            }),
            ..LoadedSession::default()
        });
        let mut ui = TestUi::default();
        let outcome = harness
            .resume_incomplete_turn(&mut ui, TurnCancellation::new())
            .await
            .unwrap()
            .expect("pending turn");
        assert_eq!(outcome.stop_reason, crate::TurnStopReason::Completed);
        assert!(ui.texts.join("").contains("resumed"));
        assert!(
            ui.statuses.iter().any(|s| s.contains("shrunk")),
            "resume must cheap-shrink before the first Pipe call, got {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn compact_then_load_keeps_unmatched_pending_turn() {
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("<summary>\nreviewed src/main.rs and cargo test is green\n</summary>"),
            usage_chunk(20, 8),
        ])]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let state = dir.path().join(".hi");
        let runner =
            ProcessRunner::new_with_policy(dir.path(), SandboxPolicy::Off).expect("runner");
        let tools =
            crate::ToolHost::new_with_runner(dir.path().to_path_buf(), state.clone(), runner)
                .unwrap();
        let mut config = HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test");
        config.base_url = server.url.clone();
        config.state_root = state;
        config.session_path = Some(path.clone());
        let mut harness = Harness::new_with_tools(config, tools).unwrap();
        harness.apply_loaded_session(LoadedSession {
            messages: vec![
                Message::user("review this"),
                Message::tool_result("c1", "fn main"),
                Message::assistant(vec![Content::Text("looking".into())]),
                Message::user("keep going"),
            ],
            pending_turn: Some(PendingTurn {
                turn_index: 4,
                started_unix_ms: 99,
                pre_checkpoint: Some("pre".into()),
            }),
            ..LoadedSession::default()
        });
        let mut ui = TestUi::default();
        assert!(
            harness
                .compact(None, &mut ui, &TurnCancellation::new())
                .await
                .unwrap()
        );
        assert!(harness.pending_turn().is_some());
        let loaded = JsonlSession::load(&path).unwrap();
        let pending = loaded
            .pending_turn
            .expect("compact rewrite must keep unmatched PendingTurn");
        assert_eq!(pending.turn_index, 4);
        assert_eq!(pending.pre_checkpoint.as_deref(), Some("pre"));
    }

    #[test]
    fn inspect_keys_to_forget_only_omitted_or_dropped_reads() {
        let messages = vec![
            Message::user("review this"),
            Message::assistant(vec![Content::ToolCall {
                id: "keep".into(),
                name: "read".into(),
                arguments: r#"{"path":"src/keep.rs"}"#.into(),
            }]),
            Message::tool_result("keep", "keep-body"),
            Message::assistant(vec![Content::ToolCall {
                id: "drop".into(),
                name: "read".into(),
                arguments: r#"{"path":"src/drop.rs"}"#.into(),
            }]),
            Message::tool_result("drop", "x".repeat(500)),
            Message::assistant(vec![Content::ToolCall {
                id: "trunc".into(),
                name: "read".into(),
                arguments: r#"{"path":"src/trunc.rs"}"#.into(),
            }]),
            Message::tool_result("trunc", "y".repeat(500)),
            Message::assistant(vec![Content::ToolCall {
                id: "bash".into(),
                name: "bash".into(),
                arguments: r#"{"command":"ls"}"#.into(),
            }]),
            Message::tool_result("bash", "z".repeat(500)),
        ];
        let next = crate::jev_compact::apply_decisions(
            &messages,
            &[
                crate::jev_compact::CallDecision {
                    id: "t1".into(),
                    call_id: "keep".into(),
                    tool: "read".into(),
                    action: crate::jev_compact::CallAction::Keep,
                },
                crate::jev_compact::CallDecision {
                    id: "t2".into(),
                    call_id: "drop".into(),
                    tool: "read".into(),
                    action: crate::jev_compact::CallAction::DropCall,
                },
                crate::jev_compact::CallDecision {
                    id: "t3".into(),
                    call_id: "trunc".into(),
                    tool: "read".into(),
                    action: crate::jev_compact::CallAction::DropResult,
                },
                crate::jev_compact::CallDecision {
                    id: "t4".into(),
                    call_id: "bash".into(),
                    tool: "bash".into(),
                    action: crate::jev_compact::CallAction::DropCall,
                },
            ],
        );
        let forget = inspect_keys_to_forget(&messages, &next);
        let drop_fp = crate::liveness::tool_fingerprint("read", r#"{"path":"src/drop.rs"}"#);
        let trunc_fp = crate::liveness::tool_fingerprint("read", r#"{"path":"src/trunc.rs"}"#);
        let keep_fp = crate::liveness::tool_fingerprint("read", r#"{"path":"src/keep.rs"}"#);
        assert!(forget.contains(&drop_fp), "{forget:?}");
        assert!(forget.contains(&trunc_fp), "{forget:?}");
        assert!(
            !forget.contains(&keep_fp),
            "kept live read must stay pinned: {forget:?}"
        );
        assert_eq!(
            forget.len(),
            2,
            "bash is not an inspect fingerprint: {forget:?}"
        );
    }

    #[test]
    fn jev_compact_on_without_key_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut harness =
            Harness::new(HarnessConfig::pipe(dir.path().to_path_buf(), "pk_test")).unwrap();
        assert!(harness.set_jev_compact(true).is_err());
        assert!(!harness.jev_compact());
        let msg = harness.apply_jev_compact_arg("on");
        assert!(msg.contains("TYPESAFE_API_KEY"), "{msg}");
        assert!(!harness.jev_compact());
        assert!(
            harness
                .apply_jev_compact_arg("")
                .contains("jev-compact: off")
        );
    }

    #[tokio::test]
    async fn jev_prune_skips_summary_when_occupancy_drops() {
        let body = JEV_DROP_ALL;
        let typesafe = serve_json_loop(200, body, 4).await;
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("<summary>\nshould not be used\n</summary>"),
            usage_chunk(20, 8),
        ])]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut harness = test_harness_jev(&server.url, &typesafe, dir.path().to_path_buf());
        harness.apply_loaded_session(LoadedSession {
            messages: jev_history(),
            ..LoadedSession::default()
        });
        let mut ui = TestUi::default();
        let changed = harness
            .compact(None, &mut ui, &TurnCancellation::new())
            .await
            .unwrap();
        assert!(changed);
        assert!(
            harness
                .messages()
                .iter()
                .all(|message| !message.text().contains("<conversation_summary>")),
            "Jev prune under 85% must not rewrite to a summary"
        );
        assert!(
            ui.statuses.iter().any(|s| s.contains("jev-compact: kept")),
            "expected jev status, got {:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn jev_failure_falls_back_to_summary() {
        let typesafe = serve_json_loop(500, r#"{"error":"nope"}"#, 4).await;
        let Some(server) = MockPipe::new(vec![Scripted::Sse(vec![
            text_chunk("<summary>\nreviewed src/main.rs and cargo test is green\n</summary>"),
            usage_chunk(20, 8),
        ])]) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut harness = test_harness_jev(&server.url, &typesafe, dir.path().to_path_buf());
        harness.apply_loaded_session(LoadedSession {
            messages: jev_history(),
            ..LoadedSession::default()
        });
        let mut ui = TestUi::default();
        let changed = harness
            .compact(None, &mut ui, &TurnCancellation::new())
            .await
            .unwrap();
        assert!(changed);
        assert!(
            ui.statuses
                .iter()
                .any(|s| s.contains("jev-compact skipped")),
            "expected skip status, got {:?}",
            ui.statuses
        );
        assert!(
            harness
                .messages()
                .iter()
                .any(|message| message.text().contains("<conversation_summary>"))
        );
    }

    const JEV_DROP_ALL: &str = r#"{"answers":{"call_t1":{"noul":0.1},"result_t1":{"noul":0.1},"call_t2":{"noul":0.1},"result_t2":{"noul":0.1},"call_t3":{"noul":0.1},"result_t3":{"noul":0.1},"call_t4":{"noul":0.1},"result_t4":{"noul":0.1},"call_t5":{"noul":0.1},"result_t5":{"noul":0.1},"call_t6":{"noul":0.1},"result_t6":{"noul":0.1},"call_t7":{"noul":0.1},"result_t7":{"noul":0.1},"call_t8":{"noul":0.1},"result_t8":{"noul":0.1}}}"#;

    async fn serve_json_loop(status: u16, body: &'static str, n: usize) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for _ in 0..n {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = vec![0u8; 8192];
                let mut data = Vec::new();
                loop {
                    let Ok(n) = stream.read(&mut buf).await else {
                        return;
                    };
                    if n == 0 {
                        break;
                    }
                    data.extend_from_slice(&buf[..n]);
                    let headers_end = data.windows(4).position(|window| window == b"\r\n\r\n");
                    let Some(headers_end) = headers_end else {
                        continue;
                    };
                    let headers = &data[..headers_end];
                    let content_length = std::str::from_utf8(headers)
                        .ok()
                        .and_then(|headers| {
                            headers.lines().find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                        })
                        .unwrap_or(0);
                    if data.len() >= headers_end + 4 + content_length {
                        break;
                    }
                }
                let reason = if status == 200 { "OK" } else { "Error" };
                let response = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{addr}")
    }
}
