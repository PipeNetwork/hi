use super::common::*;
use super::*;

#[tokio::test]
async fn does_not_nudge_a_plain_answer() {
    // No tool call this turn (a Q&A-style reply) — never nudge, never warn,
    // even though the text isn't an action.
    let responses = vec![completion(
        vec![Content::Text("The answer is 42.".into())],
        1,
        1,
    )];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("what is 6*7?", &mut ui).await.unwrap();
    assert!(
        !ui.statuses
            .iter()
            .any(|s| s.contains("nudging") || s.contains("incomplete")),
        "plain answer is left alone, got: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "turn completed");
}

#[tokio::test]
async fn finalizes_with_a_recap_when_files_changed() {
    // A turn that changes a file ends with a dedicated recap call. The recap
    // is emitted to the UI (so the user sees it) and its usage is counted,
    // but the [user: finalize-nudge][assistant: recap] pair is stripped from
    // the persisted transcript at turn end — the FINALIZE_PROMPT's "don't
    // take any further action" instruction must not bleed into the next turn.
    // Holds the workspace lock: this test writes a temp file, which would
    // otherwise perturb the file-change detection of the verify tests.
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let mut cfg = config();
    cfg.finalize = true;
    let tmp = visible_temp_file("finalize");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text(
                "## Summary\n- Created the file.\n\nRun `cargo test`.".into(),
            )],
            3,
            4,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("make a file", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&tmp);

    // The recap was emitted to the UI (the user sees it).
    assert!(
        ui.assistant.contains("## Summary"),
        "recap is emitted to the UI: {}",
        ui.assistant
    );

    let m = agent.messages();
    // The finalize nudge + recap are stripped from history. The last message
    // is the assistant's "done" from the turn work, not the recap.
    let last = m.last().expect("history is non-empty");
    assert_eq!(last.role, Role::Assistant);
    assert!(
        !last.text().contains("[hi:nudge:finalize]"),
        "no finalize nudge marker in history, got: {}",
        last.text()
    );
    // No finalize nudge anywhere in the transcript.
    assert!(
        !m.iter().any(|msg| {
            msg.role == Role::User
                && msg
                    .content
                    .iter()
                    .any(|c| matches!(c, Content::Text(t) if t.contains("[hi:nudge:finalize]")))
        }),
        "finalize nudge should be stripped from history"
    );
    // Roles alternate (no two assistants in a row → provider-safe next turn).
    assert!(
        m.windows(2).all(|w| w[0].role != w[1].role),
        "roles must alternate"
    );
    // The recap call's usage (3/4) is folded into the running totals.
    assert_eq!(agent.totals().input_tokens, 1 + 1 + 3);
    assert_eq!(agent.totals().output_tokens, 1 + 1 + 4);
}

#[tokio::test]
async fn finalize_recap_is_emitted_to_the_ui() {
    // The Canned provider never calls the stream sink — it returns text
    // only in the completion object. The finalize fallback must emit that
    // text through ui.assistant_text so the user sees the recap, not just
    // record it silently in history. (This is the "ending doesn't show"
    // bug: the recap was recorded but never displayed.)
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let mut cfg = config();
    cfg.finalize = true;
    let tmp = visible_temp_file("finalize_ui");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text("## Summary\n- Created the file.".into())],
            3,
            4,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("make a file", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&tmp);

    // The recap text must have been emitted to the UI, not just recorded.
    assert!(
        ui.assistant.contains("## Summary"),
        "recap text should be emitted to the UI, got assistant: {:?}",
        ui.assistant
    );
}

#[tokio::test]
async fn finalize_nudge_does_not_bleed_into_next_turn() {
    // Regression: after a finalized turn, the FINALIZE_PROMPT ("don't take
    // any further action") was left in history. On the next turn the model
    // saw it above the new prompt and emitted more summary text instead of
    // executing the request. The fix strips the [user: finalize-nudge]
    // [assistant: recap] pair at turn end. This test verifies the nudge is
    // gone from history before the second turn starts, so the model's
    // context for turn 2 contains only real conversation.
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let mut cfg = config();
    cfg.finalize = true;
    let tmp = visible_temp_file("finalize_bleed");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        // Turn 1: write a file, then a "done" text, then the recap.
        write_completion(&p),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(
            vec![Content::Text("## Summary\n- Created the file.".into())],
            3,
            4,
        ),
        // Turn 2: a clean text response to the second prompt.
        completion(vec![Content::Text("ok second".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("make a file", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&tmp);

    // After turn 1: no finalize nudge or recap in history.
    let msgs = agent.messages();
    assert!(
        !msgs.iter().any(|m| {
            m.content.iter().any(|c| {
                matches!(
                    c,
                    Content::Text(t) if t.contains("[hi:nudge:finalize]")
                )
            })
        }),
        "finalize nudge must be stripped from history after turn 1"
    );
    assert!(
        !msgs.iter().any(|m| m.text().contains("## Summary")),
        "recap must be stripped from history after turn 1"
    );

    // Turn 2: the model should see the new prompt without the stale
    // "don't take any further action" instruction. We verify by checking
    // the last user message is the real second prompt, not folded nudge text.
    let mut ui2 = RecUi::default();
    agent
        .run_turn("now do something else", &mut ui2)
        .await
        .unwrap();

    let msgs = agent.messages();
    let last_user = msgs
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
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
        text.contains("now do something else"),
        "second prompt is the real user message, got: {text}"
    );
    assert!(
        !text.contains("don't take any further action"),
        "stale finalize instruction must not be in the second prompt context, got: {text}"
    );
}

#[tokio::test]
async fn does_not_finalize_a_plain_answer() {
    // Finalization on, but the turn changed no files (a Q&A reply) — no extra
    // recap call fires. (The canned provider has exactly one completion; a
    // stray finalization call would panic trying to pop a second.)
    let mut cfg = config();
    cfg.finalize = true;
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("The answer is 42.".into())],
            1,
            1,
        )],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("what is 6*7?", &mut ui).await.unwrap();
    let assistants = agent
        .messages()
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .count();
    assert_eq!(assistants, 1, "no extra recap message");
    assert_eq!(agent.totals().output_tokens, 1, "no extra recap call");
}

#[tokio::test]
async fn turn_end_reports_prompt_and_generated_not_context_as_input() {
    // Two rounds (5/1 then 6/2). The done line must show the cumulative
    // session total (11/3/14), matching the live counter — not just the
    // last round (6/2/8).
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
        completion(vec![Content::Text("done".into())], 6, 2),
    ];
    let mut agent = agent(responses, config());
    let mut ui = RecUi::default();
    agent.run_turn("go", &mut ui).await.unwrap();
    let summary = ui.turn_end.expect("turn_end emitted");
    // The primary input is the raw user prompt estimate, not the full request
    // context. Generated output remains the current-turn total.
    assert!(
        summary.contains("prompt↑1 gen↓3"),
        "turn-local prompt/output, got: {summary}"
    );
}
