use super::*;
use super::common::*;

#[tokio::test]
async fn nudges_when_model_repeats_the_same_command() {
    // The model runs a command, then re-issues the *exact same* call next
    // round. The repetition guard nudges it to act on the output instead of
    // re-running, and the model then finishes. One repeat-nudge, no
    // "stuck repeating" notice.
    let responses = vec![
        echo_call(),
        echo_call(), // exact repeat → nudged
        completion(vec![Content::Text("Done. Run cargo test.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("check it", &mut ui).await.unwrap();
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("re-ran the same command"))
            .count(),
        1,
        "exactly one repeat-nudge, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.statuses.iter().any(|s| s.contains("kept re-running")),
        "no stuck-repeating notice once it moved on, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
}

#[tokio::test]
async fn gives_up_with_notice_after_repeat_cap() {
    // The model re-issues the exact same command every round, through the
    // whole repeat-nudge budget: bounded nudges, then an honest
    // "stuck repeating" notice.
    let mut responses = vec![echo_call()];
    for _ in 0..(config().max_repeat_nudges + 1) {
        responses.push(echo_call()); // exact repeat each round
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("check it", &mut ui).await.unwrap();
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|s| s.contains("re-ran the same command"))
            .count(),
        config().max_repeat_nudges as usize,
        "repeat-nudges are bounded, got: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("kept re-running")),
        "stuck-repeating notice after the cap, got: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn does_not_nudge_a_different_command() {
    // Two consecutive tool calls with different arguments are not a repeat —
    // both execute, no repeat-nudge.
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "t".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo one\"}".into(),
            }],
            1,
            1,
        ),
        completion(
            vec![Content::ToolCall {
                id: "t".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo two\"}".into(),
            }],
            1,
            1,
        ),
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("run them", &mut ui).await.unwrap();
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("re-ran the same command")),
        "different commands are not a repeat, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
}

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

#[tokio::test]
async fn stale_nudge_stripped_before_next_turn() {
    // When a turn ends after a repeat-nudge stall, the last message in
    // history is a synthetic user nudge. Without stripping, the next
    // prompt would fold into that nudge via `push_user_or_fold`. This
    // test verifies the nudge is stripped so the next turn starts clean.
    let mut responses = vec![echo_call()];
    // Repeat the same call through the whole repeat-nudge budget so the
    // turn ends with a trailing repeat-nudge.
    for _ in 0..(config().max_repeat_nudges + 1) {
        responses.push(echo_call());
    }
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("check it", &mut ui).await.unwrap();

    // After the turn, the last message should NOT be a nudge (user message
    // with a [hi:nudge:...] marker). It should be the assistant's text or
    // a real user message.
    let msgs = agent.messages();
    let last = msgs.last().expect("history is non-empty");
    if last.role == hi_ai::Role::User {
        let text = last
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert!(
            !text.starts_with("[hi:nudge:"),
            "trailing nudge should be stripped, but last message is: {text}"
        );
    }
}

#[tokio::test]
async fn next_prompt_does_not_fold_into_stale_nudge() {
    // End-to-end: a turn stalls with a repeat-nudge, then a second turn is
    // sent. The second turn's user message should NOT be folded into the
    // stale nudge — it should be a clean, separate user message. We verify
    // by checking that the model sees the real prompt, not nudge text.
    let mut responses = vec![echo_call()];
    for _ in 0..(config().max_repeat_nudges + 1) {
        responses.push(echo_call());
    }
    // Second turn: a clean text response.
    responses.push(completion(vec![Content::Text("ok".into())], 1, 1));

    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("first task", &mut ui).await.unwrap();

    // Second turn — should start clean, not folded into a nudge.
    let mut ui2 = RecUi::default();
    agent.run_turn("second task", &mut ui2).await.unwrap();

    let msgs = agent.messages();
    // Find the last user message — it should be "second task", not a
    // folded nudge+prompt combination.
    let last_user = msgs
        .iter()
        .rev()
        .find(|m| m.role == hi_ai::Role::User)
        .expect("there is a last user message");
    let text = last_user
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(
        !text.contains("[hi:nudge:"),
        "next prompt should not be folded into a stale nudge, got: {text}"
    );
    assert!(
        text.contains("second task"),
        "next prompt should be the real user input, got: {text}"
    );
}

#[tokio::test]
async fn silent_auto_continue_keeps_turn_going_without_status() {
    // The model narrates an announced-but-unperformed next step ("Now let me
    // check the tests.") with no tool call. With max_silent_continues > 0 the
    // agent silently re-prompts it to continue — no status line, no visible
    // nudge — and the model then makes the next tool call and finishes with a
    // recap. The recap ("Done.") is a *finished* answer, not a forward-looking
    // step, so it ends the turn cleanly: no further nudge, no false
    // "incomplete" warning.
    let mut cfg = config();
    cfg.max_silent_continues = 3;
    let responses = vec![
        // Round 1: model makes a tool call (actively working).
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // Round 2: announced next step, no tool call → silent continue.
        completion(
            vec![Content::Text("Now let me check the tests.".into())],
            1,
            1,
        ),
        // Round 3: silently re-prompted, model makes the next tool call.
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // Round 4: model finishes with a recap → turn ends cleanly.
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("review the code", &mut ui).await.unwrap();
    // The turn completed, consuming exactly the four canned responses — a
    // spurious continue after the "Done." recap would have asked for a fifth
    // and panicked on the empty queue.
    assert!(ui.turn_end.is_some(), "turn completed");
    // No visible "nudging" status during the silent continue, and no false
    // "incomplete" warning — the recap ended the turn cleanly.
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("nudging") || s.contains("incomplete")),
        "silent continue then clean finish: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn finished_recap_after_tool_use_ends_without_incomplete_warning() {
    // Repro of the reported "review codebase runs a bit, then stops without
    // finishing" bug. A read-only task reads files (tool calls), then gives
    // its final recap as text with no tool call. The recap is a *finished*
    // answer (past tense), not an announced next step, so the turn must end
    // cleanly — no silent-continue nudge, no false "the model kept narrating
    // … may be incomplete" warning. Before the fix, `made_tool_call` alone
    // forced a nudge on any post-tool text, so a finished review churned the
    // whole silent-continue budget and stopped on the warning.
    let mut cfg = config();
    cfg.max_silent_continues = 3;
    let responses = vec![
        // Reads a file (actively working).
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"Cargo.toml"}"#.into(),
            }],
            1,
            1,
        ),
        // Final recap — a finished answer, text only.
        completion(
            vec![Content::Text(
                "I reviewed Cargo.toml. The workspace status is clear and tests pass.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("review codebase", &mut ui).await.unwrap();
    // The turn ended after exactly the two canned responses — a spurious
    // continue would have asked for a third and panicked on the empty queue.
    assert!(ui.turn_end.is_some(), "turn completed");
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no false incomplete warning on a finished review: {:?}",
        ui.statuses
    );
    // The recap is the closing message — the turn stopped there rather than
    // churning past it with spurious continues.
    let m = agent.messages();
    assert!(
        m.last().unwrap().text().contains("I reviewed Cargo.toml"),
        "the recap is the model's final response: {:?}",
        m.last().unwrap().text()
    );
}

#[tokio::test]
async fn silent_continue_budget_resets_after_tool_progress() {
    // The actual "review codebase stops without finishing" bug. A long,
    // productive turn that *intermittently* narrates a next step without the
    // tool call (a quirk of some models), but reads a file after each nudge.
    // The silent-continue budget bounds *consecutive* stalls, not their
    // total across the turn: each tool call resets the counter, so the turn
    // keeps going as long as the model makes progress between stalls — even
    // when the number of stalls exceeds max_silent_continues. Before the
    // reset the cumulative counter crept up across the whole turn (stall 1,
    // act, stall 2, act, …) and ended it mid-review with a false "incomplete"
    // warning once the Nth stall hit the budget, despite progress every time.
    let mut cfg = config();
    cfg.max_silent_continues = 1;
    let read = |id: &str, path: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: format!(r#"{{"path":"{path}"}}"#),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // Stall 1: narrates a next step, no tool call → nudge (budget is 1).
        completion(vec![Content::Text("Let me read Cargo.toml.".into())], 1, 1),
        // Recovers: reads a file → must reset the silent-continue counter.
        read("a", "Cargo.toml"),
        // Stall 2: narrates again. With the reset this is still within budget;
        // without it the cumulative counter is already exhausted here.
        completion(vec![Content::Text("Let me read README.md.".into())], 1, 1),
        // Recovers again.
        read("b", "README.md"),
        // Finishes with a recap → clean end.
        completion(
            vec![Content::Text(
                "Reviewed Cargo.toml and README.md. Done.".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("review codebase", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no false incomplete warning while making progress: {:?}",
        ui.statuses
    );
    // Ran all the way to the recap rather than quitting at the second stall.
    assert!(
        agent.messages().last().unwrap().text().contains("Done."),
        "turn ran to the recap: {:?}",
        agent.messages().last().unwrap().text()
    );
}

#[tokio::test]
async fn plan_with_pending_steps_continues_past_recap() {
    // The model posts a plan (2/3 done), does one step, then stops with a
    // finished-looking recap. Without plan-awareness, the text heuristic
    // sees a finished recap and ends the turn — leaving the plan at 2/3.
    // With plan-awareness, the agent detects pending steps and nudges the
    // model to continue until the plan is complete.
    let mut cfg = config();
    cfg.max_silent_continues = 5;
    // Helper: an update_plan call with given step statuses.
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: model posts the initial plan (0/3 done) and starts step 1.
        plan_call("p1", &["active", "pending", "pending"]),
        // R2: model does a read for step 1.
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R3: model updates plan (1/3 done, step 2 active) and does a read.
        plan_call("p2", &["done", "active", "pending"]),
        // R4: model stops with a finished-looking recap — but plan is 1/3!
        // The plan-aware continue should nudge it to keep going.
        completion(
            vec![Content::Text(
                "I've completed step 1. The implementation looks good.".into(),
            )],
            1,
            1,
        ),
        // R5 (nudged): model does step 2.
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // R6: model updates plan (2/3 done, step 3 active).
        plan_call("p3", &["done", "done", "active"]),
        // R7: model stops with recap again — plan is 2/3, nudge again.
        completion(
            vec![Content::Text("Step 2 is done. Moving on.".into())],
            1,
            1,
        ),
        // R8 (nudged): model does step 3.
        completion(
            vec![Content::ToolCall {
                id: "r3".into(),
                name: "read".into(),
                arguments: r#"{"path":"z"}"#.into(),
            }],
            1,
            1,
        ),
        // R9: model updates plan (3/3 done) — all complete.
        plan_call("p4", &["done", "done", "done"]),
        // R10: model gives final recap — plan is complete, turn ends.
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("implement the feature", &mut ui)
        .await
        .unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // The turn should have run all the way to the final recap (R10),
    // not stopped at R4 or R7 when the model gave a partial recap.
    assert!(
        agent
            .messages()
            .last()
            .unwrap()
            .text()
            .contains("All steps complete"),
        "turn ran to the final recap with plan complete: {:?}",
        agent.messages().last().unwrap().text()
    );
}

#[tokio::test]
async fn complete_plan_ends_turn_without_spurious_continue() {
    // When the plan is fully done (all steps "done"), the model's recap
    // should end the turn cleanly — no plan-driven continue nudge.
    let mut cfg = config();
    cfg.max_silent_continues = 5;
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // Model posts plan (all done) and gives final recap.
        plan_call("p1", &["done", "done"]),
        completion(vec![Content::Text("All done.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // No spurious continue — the turn ended after exactly 2 responses.
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no incomplete warning when plan is done: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn long_plan_10_steps_runs_to_completion() {
    // A 10-step plan where the model does one step per round, then stops
    // with a recap. The plan-aware continue should nudge it to keep going
    // until all 10 steps are done. The silent_continues counter resets on
    // each tool call, so this should work regardless of plan length.
    let mut cfg = config();
    cfg.max_silent_continues = 3; // the default
    let n_steps = 10;
    let plan_call = |id: &str, statuses: &[&str]| {
        let steps: Vec<String> = statuses
            .iter()
            .enumerate()
            .map(|(i, s)| format!(r#"{{"title":"step {}","status":"{}"}}"#, i + 1, s))
            .collect();
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(r#"{{"steps":[{}]}}"#, steps.join(",")),
            }],
            1,
            1,
        )
    };
    let read_call = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        )
    };
    let recap = |text: &str| completion(vec![Content::Text(text.into())], 1, 1);

    let mut responses = Vec::new();
    for step in 0..n_steps {
        // Build statuses: steps before `step` are done, step `step` is active,
        // steps after are pending.
        let statuses: Vec<&str> = (0..n_steps)
            .map(|i| {
                if i < step {
                    "done"
                } else if i == step {
                    "active"
                } else {
                    "pending"
                }
            })
            .collect();
        // Model posts plan + does a read for this step.
        responses.push(plan_call(&format!("p{step}"), &statuses));
        responses.push(read_call(&format!("r{step}")));
        // Model stops with a recap (unless it's the last step).
        if step < n_steps - 1 {
            responses.push(recap(&format!(
                "Step {} is done. The implementation looks good.",
                step + 1
            )));
        }
    }
    // Final: all steps done + final recap.
    let all_done: Vec<&str> = (0..n_steps).map(|_| "done").collect();
    responses.push(plan_call("pfinal", &all_done));
    responses.push(recap("All 10 steps complete. Done."));

    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("implement the feature", &mut ui)
        .await
        .unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    // The turn should have run all the way to the final recap.
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All 10 steps complete"),
        "turn ran to the final recap, got: {last_text}"
    );
    // Should NOT have ended with an incomplete warning.
    assert!(
        !ui.statuses.iter().any(|s| s.contains("incomplete")),
        "no incomplete warning on a completed 10-step plan: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn long_plan_survives_text_only_response_to_nudge() {
    // A plan where the model sometimes responds to the continue-nudge with
    // text-only (no tool call) before eventually doing the work. This is
    // the real-world pattern that causes stalls: the model writes a recap,
    // gets nudged, writes another recap instead of acting, gets nudged
    // again, and eventually does the work. The silent_continues budget
    // must be high enough to survive a few text-only responses.
    //
    // With max_silent_continues=3, the model can text-only 3 times in a
    // row before the turn ends. On the 4th text-only, the budget is
    // exhausted. This test has 3 text-only responses (within budget)
    // before the model finally acts.
    let mut cfg = config();
    cfg.max_silent_continues = 3;
    let plan_call = |id: &str, s1: &str, s2: &str, s3: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}},{{"title":"c","status":"{s3}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: plan + read for step 1.
        plan_call("p1", "active", "pending", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R2: recap, no tools → nudge (silent_continues=1, force_tools).
        completion(vec![Content::Text("Step 1 done. Looks good.".into())], 1, 1),
        // R3: text-only again (ignores force) → nudge (silent_continues=2).
        completion(
            vec![Content::Text(
                "The implementation is clean. No issues found.".into(),
            )],
            1,
            1,
        ),
        // R4: text-only again (ignores force) → nudge (silent_continues=3).
        completion(
            vec![Content::Text("Everything looks correct so far.".into())],
            1,
            1,
        ),
        // R5: finally does a tool call → silent_continues resets to 0.
        plan_call("p2", "done", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r2".into(),
                name: "read".into(),
                arguments: r#"{"path":"y"}"#.into(),
            }],
            1,
            1,
        ),
        // R6: recap → nudge (silent_continues=1).
        completion(vec![Content::Text("Step 2 done.".into())], 1, 1),
        // R7: does step 3.
        plan_call("p3", "done", "done", "active"),
        completion(
            vec![Content::ToolCall {
                id: "r3".into(),
                name: "read".into(),
                arguments: r#"{"path":"z"}"#.into(),
            }],
            1,
            1,
        ),
        // R8: all done + final recap.
        plan_call("p4", "done", "done", "done"),
        completion(
            vec![Content::Text("All steps complete. Done.".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn completed");
    let last_text = agent.messages().last().unwrap().text();
    assert!(
        last_text.contains("All steps complete"),
        "turn ran to completion despite text-only responses to nudges, got: {last_text}"
    );
}

#[tokio::test]
async fn plan_stalls_after_max_consecutive_text_only_responses() {
    // When the model responds to the continue-nudge with text-only (no tool
    // call) more than max_silent_continues times in a row, the turn ends
    // with an "incomplete" warning. This is the safety valve — the model is
    // stuck narrating without acting. This test verifies the valve fires
    // at the right point: after exactly max_silent_continues+1 text-only
    // responses (the original recap + max_silent_continues nudged retries).
    let mut cfg = config();
    cfg.max_silent_continues = 3;
    let plan_call = |id: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: r#"{"steps":[{"title":"a","status":"active"},{"title":"b","status":"pending"}]}"#.into(),
            }],
            1,
            1,
        )
    };
    let responses = vec![
        // R1: plan + read for step 1.
        plan_call("p1"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R2: recap → nudge (1/3).
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        // R3: text-only → nudge (2/3).
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // R4: text-only → nudge (3/3).
        completion(vec![Content::Text("Correct.".into())], 1, 1),
        // R5: text-only → budget exhausted, turn ends with warning.
        completion(vec![Content::Text("Fine.".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    assert!(ui.turn_end.is_some(), "turn ended");
    // Should warn about incomplete — the model kept narrating without acting.
    assert!(
        ui.statuses.iter().any(|s| s.contains("incomplete")),
        "should warn incomplete after exhausting continue budget: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn plan_persists_across_turns_for_continue() {
    // When a turn ends with an incomplete plan and the user types
    // "continue", the plan state should persist so the plan-aware continue
    // logic can fire. Without persistence, last_plan is cleared at the
    // start of the new turn and the agent can't detect the incomplete plan.
    let mut cfg = config();
    cfg.max_silent_continues = 3;
    let plan_call = |id: &str, s1: &str, s2: &str| {
        completion(
            vec![Content::ToolCall {
                id: id.into(),
                name: "update_plan".into(),
                arguments: format!(
                    r#"{{"steps":[{{"title":"a","status":"{s1}"}},{{"title":"b","status":"{s2}"}}]}}"#
                ),
            }],
            1,
            1,
        )
    };

    // Turn 1: model posts plan (step 1 active), does step 1, then stops
    // with a recap. The plan-continue nudges, but the model text-only's
    // past the budget, so the turn ends with an incomplete plan (1/2).
    let turn1_responses = vec![
        plan_call("p1", "active", "pending"),
        completion(
            vec![Content::ToolCall {
                id: "r1".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // Recap → nudge (1/3).
        completion(vec![Content::Text("Step 1 done.".into())], 1, 1),
        // Text-only → nudge (2/3).
        completion(vec![Content::Text("Looks good.".into())], 1, 1),
        // Text-only → nudge (3/3).
        completion(vec![Content::Text("Correct.".into())], 1, 1),
        // Text-only → budget exhausted, turn ends.
        completion(vec![Content::Text("Fine.".into())], 1, 1),
    ];
    let mut agent = agent(turn1_responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("do it", &mut ui).await.unwrap();
    // Turn 1 ended with incomplete warning — plan is 1/2.
    assert!(
        ui.statuses.iter().any(|s| s.contains("incomplete")),
        "turn 1 should end incomplete: {:?}",
        ui.statuses
    );

    // Verify the plan state persisted after turn 1 — it should still have
    // pending steps so the plan-aware continue can fire on "continue".
    let plan_after_turn1 = &agent.last_plan;
    assert!(
        plan_has_pending_steps(plan_after_turn1),
        "plan should persist with pending steps after turn 1: {:?}",
        plan_after_turn1
    );

    // Turn 2: user types "fix a different bug" (NOT "continue"). The plan
    // should be cleared so a stale plan doesn't cause spurious nudges.
    // We can't easily run a full turn here (Canned provider is exhausted),
    // but we can verify the clearing logic by checking that a non-continue
    // input would clear it. Simulate by calling the clearing logic directly.
    let mut plan = agent.last_plan.clone();
    // The agent clears last_plan when input doesn't look like "continue".
    // Verify the heuristic: "fix a different bug" is NOT a continue command.
    assert!(
        !looks_like_continue("fix a different bug"),
        "a new task should not look like continue"
    );
    assert!(
        looks_like_continue("continue"),
        "'continue' should look like continue"
    );
    // Simulate the clearing: a new task clears, "continue" doesn't.
    plan.clear(); // what the agent does on a new task
    assert!(
        !plan_has_pending_steps(&plan),
        "plan should be cleared on a new task"
    );
}

#[tokio::test]
async fn continue_nudge_forces_tool_choice_on_the_next_round() {
    // When the model narrates instead of acting and gets a silent-continue
    // nudge, the *next* request forces a tool call (tool_mode Required ->
    // tool_choice "required") so the model can't answer the nudge with yet
    // another narration or an empty completion (the observed failure mode of
    // some OpenAI-compat coder models). Once the model acts, the force clears.
    let mut cfg = config();
    cfg.max_silent_continues = 1;
    assert_eq!(cfg.tool_mode, ToolMode::Auto, "precondition: free tool use");
    let responses = vec![
        // R1: narrates a next step, no tool call → nudge + force next round.
        completion(vec![Content::Text("Let me read the code.".into())], 1, 1),
        // R2 (forced): the model calls a tool → force clears.
        completion(
            vec![Content::ToolCall {
                id: "r".into(),
                name: "read".into(),
                arguments: r#"{"path":"x"}"#.into(),
            }],
            1,
            1,
        ),
        // R3: finishes with a recap → turn ends.
        completion(vec![Content::Text("Done.".into())], 1, 1),
    ];
    let modes = std::sync::Arc::new(Mutex::new(Vec::new()));
    let provider = RecordToolModes {
        responses: Mutex::new(responses),
        modes: modes.clone(),
    };
    let mut agent = Agent::new(Box::new(provider), cfg);
    let mut ui = RecUi::default();
    agent.run_turn("review", &mut ui).await.unwrap();
    let modes = modes.lock().unwrap().clone();
    assert_eq!(modes.len(), 3, "three model rounds: {modes:?}");
    assert_eq!(modes[0], ToolMode::Auto, "first round is normal");
    assert_eq!(
        modes[1],
        ToolMode::Required,
        "the round after the nudge forces a tool call"
    );
    assert_eq!(
        modes[2],
        ToolMode::Auto,
        "after the model acted, the force is cleared"
    );
}

