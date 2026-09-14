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
    };
    use hi_ai::Usage;
    use hi_tools::ProcessRunner;
    use hi_tools::sandbox::SandboxPolicy;
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
}
