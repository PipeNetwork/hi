use super::*;
use super::common::*;

#[tokio::test]
async fn truncation_continues_instead_of_ending_early() {
    // The model's first response is truncated (stop_reason = "length") —
    // cut off mid-generation. The agent should nudge it to continue rather
    // than treating the truncation as a natural stop. The model then
    // finishes on the second response.
    let mut cfg = config();
    cfg.max_truncation_retries = 2;
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
async fn truncation_gives_up_after_retry_budget() {
    // The model keeps hitting the output token cap every round. After the
    // truncation-retry budget is exhausted, the turn ends with the truncated
    // output rather than looping forever.
    let mut cfg = config();
    cfg.max_truncation_retries = 1;
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
async fn truncation_budget_is_separate_from_empty_retries() {
    // Truncation recovery has its own budget, separate from the empty-retry
    // budget. A big task that hits the output token cap multiple times
    // should keep going (up to its own budget) even if it would have
    // exhausted the shared empty-retry budget under the old design.
    let mut cfg = config();
    cfg.max_empty_retries = 1; // small empty-retry budget
    cfg.max_truncation_retries = 4; // generous truncation budget
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
async fn truncation_with_partial_tool_call_does_not_orphan() {
    // The model's response is truncated mid-tool-call — the ToolCall block
    // has partial/malformed JSON arguments. The truncation recovery must
    // strip the partial tool call (it was never executed, so it has no
    // matching tool_result) and record only the text. Without stripping,
    // the next provider request would carry an orphan tool_use and be
    // rejected — the turn would stall.
    let mut cfg = config();
    cfg.max_truncation_retries = 2;
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
        // Second response: the model continues and finishes cleanly.
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
    cfg.max_truncation_retries = 2;
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
        transcript_text.contains("Re-issue one fresh, complete tool call"),
        "fresh-tool-call nudge should be recorded: {transcript_text}"
    );
}

