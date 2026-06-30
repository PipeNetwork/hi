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

#[tokio::test]
async fn runs_a_tool_then_finishes() {
    let responses = vec![
        completion(
            vec![Content::ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: "{\"command\":\"echo hi\"}".into(),
            }],
            5,
            1,
        ),
        completion(vec![Content::Text("all done".into())], 6, 2),
    ];
    let mut agent = agent(responses, config());
    agent.run_turn("do it", &mut NullUi).await.unwrap();

    let roles: Vec<Role> = agent.messages().iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        vec![
            Role::System,
            Role::User,
            Role::Assistant, // tool call
            Role::Tool,      // tool result
            Role::Assistant, // final text
        ]
    );
    // Token totals accumulate across both model calls.
    assert_eq!(agent.totals().input_tokens, 11);
    assert_eq!(agent.totals().output_tokens, 3);
    assert_eq!(agent.messages().last().unwrap().text(), "all done");
}

#[tokio::test]
async fn batched_read_only_tools_run_and_preserve_order() {
    // One round emits two read-only calls; both run (concurrently) and their
    // results are recorded back in call order. Reads resolve against the
    // crate dir (cargo sets cwd to the manifest dir).
    let responses = vec![
        completion(
            vec![
                Content::ToolCall {
                    id: "1".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"Cargo.toml"}"#.into(),
                },
                Content::ToolCall {
                    id: "2".into(),
                    name: "read".into(),
                    arguments: r#"{"path":"src/lib.rs"}"#.into(),
                },
            ],
            5,
            1,
        ),
        completion(vec![Content::Text("done".into())], 6, 2),
    ];
    let mut agent = agent(responses, config());
    agent.run_turn("scan", &mut NullUi).await.unwrap();

    let outputs: Vec<String> = agent
        .messages()
        .iter()
        .flat_map(|m| &m.content)
        .filter_map(|c| match c {
            Content::ToolResult { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 2, "both tool results recorded");
    assert!(
        outputs[0].contains("hi-agent"),
        "first result is Cargo.toml"
    );
    assert!(
        // The file's top-of-module doc comment — stable in the kept head even
        // after the per-result cap clips this (large) file's middle.
        outputs[1].contains("The agent loop"),
        "second result is lib.rs"
    );
}

