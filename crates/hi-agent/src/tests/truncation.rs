use super::common::*;
use super::*;

#[tokio::test]
async fn truncation_continues_instead_of_ending_early() {
    // The model's first response is truncated (stop_reason = "length") —
    // cut off mid-generation. The agent should nudge it to continue rather
    // than treating the truncation as a natural stop. The model then
    // finishes on the second response.
    let mut cfg = config();
    cfg.loop_limits.max_truncation_retries = 2;
    let responses = vec![
        Completion {
            content: vec![Content::Text("Here is the first half of my".into())],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        },
        completion(vec![Content::Text(" answer. Done.".into())], 10, 50),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("explain it", &mut ui).await.unwrap();
    assert!(
        ui.statuses.iter().any(|s| s.contains("output token limit")),
        "should warn about truncation, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed after continuation");
    // The final assistant message in history should include the second
    // (non-truncated) response, proving the turn didn't end on the
    // truncated first half.
    let last_assistant = agent
        .messages()
        .iter()
        .rev()
        .find(|m| m.role == hi_ai::Role::Assistant)
        .expect("there is a final assistant message");
    let text = last_assistant
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(
        text.contains("Done."),
        "model continued past truncation, got: {text}"
    );
}

#[tokio::test]
async fn truncation_recovery_is_internal_not_user_visible_status() {
    let mut cfg = config();
    cfg.loop_limits.max_truncation_retries = 2;
    let responses = vec![
        Completion {
            content: vec![Content::Text("Here is the first half".into())],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        },
        completion(vec![Content::Text(" and the finish.".into())], 10, 50),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = SplitUi::default();
    agent.run_turn("explain it", &mut ui).await.unwrap();

    assert!(
        ui.statuses
            .iter()
            .all(|s| !s.contains("output token limit")),
        "truncation recovery must not be user-visible status: {:?}",
        ui.statuses
    );
    assert!(
        ui.nudges.iter().any(|s| s.contains("output token limit")),
        "internal recovery telemetry should remain available to tests: {:?}",
        ui.nudges
    );
    assert!(
        ui.turn_end.is_some(),
        "turn completed after hidden continuation"
    );
}

#[tokio::test]
async fn truncation_gives_up_after_retry_budget() {
    // The model keeps hitting the output token cap every round. After the
    // truncation-retry budget is exhausted, the turn ends with the truncated
    // output rather than looping forever.
    let mut cfg = config();
    cfg.loop_limits.max_truncation_retries = 1;
    // max_truncation_retries=1 → one retry, then give up. So 2 truncated
    // responses: the original + the one retry.
    let responses = vec![
        Completion {
            content: vec![Content::Text("truncated...".into())],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("max_tokens".into()),
        },
        Completion {
            content: vec![Content::Text("truncated...".into())],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("max_tokens".into()),
        },
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("big task", &mut ui).await.unwrap();
    // One "continuing" retry warning, then one exhaustion warning.
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("output token limit — continuing"))
            .count(),
        1,
        "exactly one truncation retry warning, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("task may be incomplete")),
        "should warn about exhaustion, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn ended after budget exhausted");
}

#[tokio::test]
async fn truncation_exhaustion_does_not_finalize_as_done() {
    let mut cfg = config();
    cfg.memory.finalize = true;
    cfg.loop_limits.max_truncation_retries = 1;
    let path = temp_file("truncation-no-finalize");
    let p = path.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        Completion {
            content: vec![Content::Text("truncated once".into())],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        },
        Completion {
            content: vec![Content::Text("truncated twice".into())],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        },
        // Would be consumed by finalize_turn if truncation exhaustion were
        // incorrectly treated as a completed changed-files turn.
        completion(
            vec![Content::Text("FINALIZE RECAP SHOULD NOT RUN".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("write the big file", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&path);

    assert!(
        agent.last_turn_telemetry().stalled_unfinished,
        "truncation exhaustion should be an unfinished turn"
    );
    assert_eq!(
        agent.last_turn_telemetry().truncation_retries,
        1,
        "telemetry should retain the retry count before exhaustion"
    );
    assert!(
        !ui.assistant.contains("FINALIZE RECAP SHOULD NOT RUN"),
        "truncation exhaustion must not trigger finalization, assistant was: {}",
        ui.assistant
    );
}

#[tokio::test]
async fn truncation_budget_is_separate_from_empty_retries() {
    // Truncation recovery has its own budget, separate from the empty-retry
    // budget. A big task that hits the output token cap multiple times
    // should keep going (up to its own budget) even if it would have
    // exhausted the shared empty-retry budget under the old design.
    let mut cfg = config();
    cfg.loop_limits.max_empty_retries = 1; // small empty-retry budget
    cfg.loop_limits.max_truncation_retries = 4; // generous truncation budget
    // 4 truncated responses, then a clean finish — the turn should survive
    // all 4 truncations (using the dedicated budget) and complete.
    let mut responses: Vec<Completion> = (0..4)
        .map(|_| Completion {
            content: vec![Content::Text("truncated...".into())],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        })
        .collect();
    responses.push(completion(
        vec![Content::Text("Finally done.".into())],
        10,
        50,
    ));
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("big task", &mut ui).await.unwrap();
    // Should have warned about truncation 4 times (one per retry).
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("output token limit — continuing"))
            .count(),
        4,
        "4 truncation retry warnings (one per retry), got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed after truncations");
    // The final assistant message should be the clean finish, not a
    // truncated fragment.
    let last_assistant = agent
        .messages()
        .iter()
        .rev()
        .find(|m| m.role == hi_ai::Role::Assistant)
        .expect("there is a final assistant message");
    let text = last_assistant
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(
        text.contains("Finally done."),
        "model finished past truncations, got: {text}"
    );
}

#[tokio::test]
async fn truncation_budget_resets_after_tool_progress() {
    // The truncation budget is a consecutive-stall budget, not a whole-turn
    // lifetime budget. A long task can hit the output cap, make real tool
    // progress, and later hit the cap again. That later truncation should get a
    // fresh continuation budget instead of ending the harness immediately.
    let mut cfg = config();
    cfg.loop_limits.max_truncation_retries = 1;
    let path = temp_file("truncation-reset-progress");
    let p = path.to_string_lossy().to_string();
    let responses = vec![
        Completion {
            content: vec![Content::Text(
                "Let me rewrite the first section with a small patch:".into(),
            )],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        },
        write_completion(&p),
        Completion {
            content: vec![Content::Text(
                "Now I will inspect the result and apply the next patch:".into(),
            )],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("max_tokens".into()),
        },
        completion(vec![Content::Text("Finally done.".into())], 10, 50),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("big task", &mut ui).await.unwrap();

    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("output token limit — continuing"))
            .count(),
        2,
        "each truncation separated by tool progress should get a retry, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("task may be incomplete")),
        "should not exhaust the truncation budget after intervening progress: {:?}",
        ui.statuses
    );
    assert!(
        ui.turn_end.is_some(),
        "turn completed after later truncation"
    );
    assert_eq!(
        agent.last_turn_telemetry().truncation_retries,
        2,
        "telemetry should retain cumulative truncation nudges"
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn truncation_during_announced_edit_forces_next_tool_call() {
    // A model often gets cut off while narrating a large edit ("Let me replace
    // this section...") rather than while emitting JSON. Continuing prose just
    // burns the truncation budget. For active tool work, the retry should force
    // a compact complete tool call.
    let mut cfg = config();
    cfg.loop_limits.max_truncation_retries = 1;
    let path = temp_file("truncation-force-tool");
    let p = path.to_string_lossy().to_string();
    let responses = vec![
        Completion {
            content: vec![Content::Text(
                "Let me replace src/intra/mod.rs with a compact edit:".into(),
            )],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        },
        write_completion(&p),
        completion(vec![Content::Text("Done.".into())], 10, 50),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();
    agent.run_turn("big task", &mut ui).await.unwrap();

    let modes = modes.lock().unwrap().clone();
    assert!(
        matches!(
            modes.as_slice(),
            [hi_ai::ToolMode::Auto, hi_ai::ToolMode::Required, ..]
        ),
        "truncation retry should force a tool call, got modes: {:?}",
        modes
    );
    let transcript_text = agent
        .messages()
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript_text.contains("Issue one fresh, complete tool call now"),
        "active-work truncation should get the tool-call nudge: {transcript_text}"
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn truncation_with_partial_tool_call_does_not_orphan() {
    // The model's response is truncated mid-tool-call — the ToolCall block
    // has partial/malformed JSON arguments. The truncation recovery must
    // strip the partial tool call (it was never executed, so it has no
    // matching tool_result) and record only the text. Without stripping,
    // the next provider request would carry an orphan tool_use and be
    // rejected — the turn would stall.
    let mut cfg = config();
    cfg.loop_limits.max_truncation_retries = 2;
    let responses = vec![
        Completion {
            content: vec![
                Content::Text("Let me write the file".into()),
                Content::ToolCall {
                    id: "call_1".into(),
                    name: "write".into(),
                    arguments: "{\"path\":\"main.rs\",\"content\":\"fn main() { // trun".into(),
                },
            ],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("length".into()),
        },
        // Second response: truncation recovery continues; "write …" is
        // expected_mutation, so a finished text-only answer would enter the
        // no-change edit cascade. Land the write instead, then recap.
        write_completion("main.rs"),
        completion(vec![Content::Text("Done writing the file.".into())], 10, 50),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("write main.rs", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // The partial tool call should NOT appear in history — it was stripped
    // (it was never executed, so it has no matching tool_result; leaving it
    // would create an orphan tool_use that providers reject).
    let has_partial_call = agent.messages().iter().any(|m| {
        m.content.iter().any(|c| {
            matches!(c, Content::ToolCall { name, arguments, .. }
                if name == "write" && arguments.contains("trun"))
        })
    });
    assert!(
        !has_partial_call,
        "partial tool call should be stripped from history"
    );
    // Also verify no orphan tool_use: every ToolCall in history has a
    // matching ToolResult somewhere.
    let mut call_ids: Vec<&str> = Vec::new();
    let mut answered: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for m in agent.messages().iter() {
        for c in &m.content {
            match c {
                Content::ToolCall { id, .. } => call_ids.push(id),
                Content::ToolResult { call_id, .. } => {
                    answered.insert(call_id);
                }
                _ => {}
            }
        }
    }
    for id in &call_ids {
        assert!(
            answered.contains(*id),
            "orphan tool_use {id} has no matching tool_result"
        );
    }
}

#[tokio::test]
async fn truncation_with_partial_text_tool_call_strips_raw_protocol() {
    let mut cfg = config();
    cfg.loop_limits.max_truncation_retries = 2;
    let responses = vec![
        Completion {
            content: vec![Content::Text(
                "Let me write it.\n<tool_call>write<arg_key>path</arg_key><arg_value>main.py</arg_value><arg_key>content</arg_key><arg_value>print("
                    .into(),
            )],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 100,
                ..Default::default()
            },
            stop_reason: Some("max_tokens".into()),
        },
        // Same as truncation_with_partial_tool_call_does_not_orphan: after
        // truncation recovery, land the write rather than a text-only finish
        // that would re-enter the expected_mutation no-change cascade.
        write_completion("main.py"),
        completion(vec![Content::Text("Done after retry.".into())], 10, 50),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("write main.py", &mut ui).await.unwrap();

    let transcript_text = agent
        .messages()
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !transcript_text.contains("<tool_call>"),
        "partial XML-ish tool call must be stripped: {transcript_text}"
    );
    assert!(
        transcript_text.contains("Issue one fresh, complete tool call"),
        "fresh-tool-call nudge should be recorded: {transcript_text}"
    );
}
