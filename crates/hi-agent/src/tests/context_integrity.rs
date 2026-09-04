use super::common::*;
use super::*;

#[tokio::test]
async fn failed_reread_does_not_erase_the_last_usable_source() {
    let workspace = IsolatedWorkspace::new("failed-reread-context");
    let arguments = r#"{"path":"src/parser.rs"}"#;
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "retry-read".into(),
                    name: "read".into(),
                    arguments: arguments.into(),
                }],
                1,
                1,
            ),
            completion(
                vec![Content::Text(
                    "The source is no longer present on disk.".into(),
                )],
                1,
                1,
            ),
        ],
        workspace.config(),
    );
    agent.messages.push_user("Inspect the parser");
    agent.messages.push_assistant_with_results(
        vec![Content::ToolCall {
            id: "previous-read".into(),
            name: "read".into(),
            arguments: arguments.into(),
        }],
        vec![(
            "previous-read".into(),
            "1\tpub fn parse(input: &str) -> Result<Node> { /* latest known source */ }".into(),
        )],
    );
    agent
        .run_turn("Check whether the file is still there", &mut NullUi)
        .await
        .unwrap();
    let previous = agent
        .messages()
        .iter()
        .flat_map(|message| &message.content)
        .find_map(|block| match block {
            Content::ToolResult { call_id, output } if call_id == "previous-read" => Some(output),
            _ => None,
        })
        .unwrap();
    assert!(
        previous.contains("latest known source"),
        "a failed reread replaced usable source: {previous}"
    );
}

#[test]
fn resume_repairs_already_elided_signed_thinking_even_in_the_recent_window() {
    let transcript = crate::Transcript::new(vec![
        Message::system("rules"),
        Message::user("resume the code review"),
        Message::assistant(vec![Content::Thinking {
            text: "[elided thinking — was 900 chars]".into(),
            signature: Some("original signature no longer matches".into()),
        }]),
    ]);
    assert!(
        matches!(&transcript.as_slice()[2].content[0], Content::Text(text) if text.starts_with("[elided thinking"))
    );
    transcript.validate_for_provider().unwrap();
}

#[test]
fn retired_signatures_reduce_request_context_without_changing_the_cached_prefix() {
    let mut agent = agent(Vec::new(), config());
    for round in 0..8 {
        agent.messages_mut().extend([
            Message::user(format!("completed coding round {round}")),
            Message::assistant(vec![
                Content::Thinking {
                    text: "private reasoning ".repeat(100),
                    signature: Some("s".repeat(4096)),
                },
                Content::Text(format!("Completed round {round}.")),
            ]),
        ]);
    }
    agent.messages_mut().extend([
        Message::user("Keep the public API stable and fix the parser"),
        Message::assistant(vec![Content::ToolCall {
            id: "latest-source".into(),
            name: "read".into(),
            arguments: r#"{"path":"src/parser.rs"}"#.into(),
        }]),
        Message::tool_result("latest-source", "current source and test evidence"),
    ]);
    let before = agent.messages().to_vec();
    let split = crate::compaction::recent_split(&before, 1).unwrap();
    let mut after = before.clone();
    crate::compaction::elide_tool_outputs(&mut after, split);
    // Reconstruct the prior compactor's exact representation: it changed the
    // text but retained every retired signature in the outgoing request.
    let mut previous = after.clone();
    for (message, original) in previous.iter_mut().zip(&before) {
        for (block, original) in message.content.iter_mut().zip(&original.content) {
            if let (Content::Text(marker), Content::Thinking { signature, .. }) =
                (&*block, original)
                && marker.starts_with("[elided thinking")
            {
                *block = Content::Thinking {
                    text: marker.clone(),
                    signature: signature.clone(),
                };
            }
        }
    }
    let previous_tokens = hi_ai::estimate_messages_tokens(&previous);
    let after_tokens = hi_ai::estimate_messages_tokens(&after);
    let previous_bytes = serde_json::to_vec(&previous).unwrap().len();
    let after_bytes = serde_json::to_vec(&after).unwrap().len();
    eprintln!(
        "compacted request fixture: estimated input tokens {previous_tokens} -> {after_tokens}; message JSON bytes {previous_bytes} -> {after_bytes}"
    );
    assert!(previous_tokens - after_tokens >= 8192);
    assert!(previous_bytes - after_bytes >= 8 * 4096);
    assert_eq!(
        serde_json::to_value(&after[0]).unwrap(),
        serde_json::to_value(&before[0]).unwrap(),
        "stable cached system prefix is unchanged"
    );
    assert_eq!(
        serde_json::to_value(&after[split..]).unwrap(),
        serde_json::to_value(&before[split..]).unwrap(),
        "recent request, source, and evidence stay byte-identical"
    );
    let snapshot = serde_json::to_value(&after).unwrap();
    assert_eq!(crate::compaction::elide_tool_outputs(&mut after, split), 0);
    assert_eq!(
        serde_json::to_value(&after).unwrap(),
        snapshot,
        "settled elision must not keep rewriting the request prefix"
    );
    crate::Transcript::new(after)
        .validate_for_provider()
        .unwrap();
}

#[test]
fn context_elision_never_sends_modified_thinking_with_an_original_signature() {
    let original = vec![
        Message::system("rules"),
        Message::user("fix the implementation"),
        Message::assistant(vec![
            Content::Thinking {
                text: "reasoning ".repeat(300),
                signature: Some("opaque-signature".repeat(200)),
            },
            Content::ToolCall {
                id: "old".into(),
                name: "read".into(),
                arguments: "{}".into(),
            },
        ]),
        Message::tool_result("old", "old source\n".repeat(100)),
        Message::user("continue"),
        Message::assistant(vec![
            Content::Thinking {
                text: "current reasoning".into(),
                signature: Some("current signature".into()),
            },
            Content::ToolCall {
                id: "recent".into(),
                name: "read".into(),
                arguments: "{}".into(),
            },
        ]),
        Message::tool_result("recent", "latest source evidence"),
    ];
    for mode in 0..3 {
        let mut messages = original.clone();
        match mode {
            0 => {
                crate::compaction::elide_tool_outputs(&mut messages, 4);
            }
            1 => {
                crate::compaction::elide_old_tool_arguments(&mut messages, 4);
            }
            _ => {
                crate::compaction::elide_tool_outputs_except_recent(&mut messages, 1);
            }
        }
        assert!(
            matches!(&messages[2].content[0], Content::Text(text) if text.starts_with("[elided thinking")),
            "elided signed reasoning must become ordinary context, not a forged thinking block"
        );
        assert_eq!(
            serde_json::to_value(&messages[5]).unwrap(),
            serde_json::to_value(&original[5]).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&messages[6]).unwrap(),
            serde_json::to_value(&original[6]).unwrap()
        );
        crate::Transcript::new(messages)
            .validate_for_provider()
            .unwrap();
    }
}

#[test]
fn background_poll_folding_preserves_each_unique_incremental_log_chunk() {
    let mut transcript = crate::Transcript::new(vec![
        Message::system("rules"),
        Message::user("run the tests"),
    ]);
    for (id, output) in [
        (
            "first",
            "[sh_1 · cargo test: still running]\nFAIL parser::rejects_invalid_input at parser.rs:42",
        ),
        ("idle", "[sh_1 · cargo test: still running — no new output]"),
        (
            "last",
            "[sh_1 · cargo test: exited with code 1]\n1 failed; 99 passed",
        ),
    ] {
        transcript.push_assistant_with_results(
            vec![Content::ToolCall {
                id: id.into(),
                name: "bash_output".into(),
                arguments: r#"{"id":"sh_1"}"#.into(),
            }],
            vec![(id.into(), output.into())],
        );
        transcript.fold_superseded_background_polls("sh_1");
    }
    let outputs = transcript
        .as_slice()
        .iter()
        .flat_map(|message| &message.content)
        .filter_map(|block| match block {
            Content::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(outputs.contains("FAIL parser::rejects_invalid_input at parser.rs:42"));
    assert!(outputs.contains("1 failed; 99 passed"));
    transcript.validate_for_provider().unwrap();
    let stored = serde_json::to_string(transcript.as_slice()).unwrap();
    let mut resumed = crate::Transcript::new(serde_json::from_str(&stored).unwrap());
    resumed.fold_superseded_background_polls("sh_1");
    assert_eq!(
        serde_json::to_string(resumed.as_slice()).unwrap(),
        stored,
        "serialized resume must retain all incremental logs without rewriting settled context"
    );
}

#[tokio::test]
async fn retained_user_images_survive_hybrid_tail_and_overflow_compaction() {
    for mode in 0..3 {
        let mut agent = agent(
            vec![completion(
                vec![Content::Text("Earlier work summary".into())],
                1,
                1,
            )],
            config(),
        );
        agent.messages_mut().extend([
            Message::user("old task"),
            Message::assistant(vec![Content::Text("old answer".into())]),
            Message {
                role: Role::User,
                content: vec![
                    Content::Text("Match this screenshot".into()),
                    Content::Image {
                        media_type: "image/png".into(),
                        data: "attached-screenshot".into(),
                    },
                    Content::Text("Keep the footer exactly as shown".into()),
                ],
            },
        ]);
        match mode {
            0 => agent
                .compact_with(CompactionKind::Hybrid { keep_recent: 1 }, &mut NullUi)
                .await
                .unwrap(),
            1 => agent
                .compact_with(
                    CompactionKind::ElideThenSummarizeTail { keep_recent: 1 },
                    &mut NullUi,
                )
                .await
                .unwrap(),
            _ => {
                assert!(
                    agent
                        .retry_after_request_too_large_compact(&mut NullUi)
                        .unwrap()
                );
            }
        }
        let current = agent
            .messages()
            .iter()
            .find(|message| message.text().contains("Match this screenshot"))
            .unwrap();
        assert!(current.content.iter().any(|block| matches!(block, Content::Image { data, .. } if data == "attached-screenshot")), "mode {mode} discarded the recent user attachment");
        assert!(current.text().contains("Keep the footer exactly as shown"));
        agent.messages.validate_for_provider().unwrap();
    }
}
