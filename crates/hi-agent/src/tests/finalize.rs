use super::common::*;
use super::*;
use hi_ai::{ChatRequest, Provider, StreamEvent};

struct HangAfterTwoCalls {
    path: String,
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl Provider for HangAfterTwoCalls {
    async fn stream(
        &self,
        _request: ChatRequest,
        _sink: &mut (dyn FnMut(StreamEvent) + Send),
    ) -> anyhow::Result<Completion> {
        match self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
            0 => Ok(write_completion(&self.path)),
            1 => Ok(completion(
                vec![Content::Text(
                    "[answer retry: generic completion placeholder rejected; provide the actual result]"
                        .into(),
                )],
                1,
                1,
            )),
            _ => std::future::pending().await,
        }
    }

    native_tool_test_provider!();
}

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
async fn closes_changed_files_without_an_additional_recap_request() {
    // A private answer marker must produce a visible deterministic closeout;
    // the extra scripted completion remains unused and unbilled.
    let workspace = IsolatedWorkspace::new("finalize-recap");
    let mut cfg = workspace.config();
    cfg.memory.finalize = true;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(
            vec![Content::Text(
                "[answer retry: generic completion placeholder rejected; provide the actual result]"
                    .into(),
            )],
            1,
            1,
        ),
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

    // The recap was emitted to the UI (the user sees it).
    assert!(
        ui.assistant
            .contains("The turn is closed with retained changes"),
        "recap is emitted to the UI: {}",
        ui.assistant
    );

    let m = agent.messages();
    // The finalize nudge + recap are stripped from history. The last message
    // is the assistant's private repair marker from the turn work, not the recap.
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
    // Only the two execution requests consume usage.
    assert_eq!(agent.totals().input_tokens, 2, "no recap input is charged");
    assert_eq!(
        agent.totals().output_tokens,
        2,
        "no recap output is charged"
    );
}

#[tokio::test]
async fn deterministic_closeout_is_emitted_and_persisted() {
    // The Canned provider never calls the stream sink — it returns text
    // only in the completion object. The finalize fallback must emit that
    // text through ui.assistant_text so the user sees the recap, not just
    // record it silently in history. (This is the "ending doesn't show"
    // bug: the recap was recorded but never displayed.)
    let workspace = IsolatedWorkspace::new("finalize-ui");
    let mut cfg = workspace.config();
    cfg.memory.finalize = true;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(
            vec![Content::Text(
                "[answer retry: generic completion placeholder rejected; provide the actual result]"
                    .into(),
            )],
            1,
            1,
        ),
        completion(
            vec![Content::Text("## Summary\n- Created the file.".into())],
            3,
            4,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("make a file", &mut ui).await.unwrap();

    // The recap text must have been emitted to the UI, not just recorded.
    assert!(
        ui.assistant
            .contains("The turn is closed with retained changes"),
        "recap text should be emitted to the UI, got assistant: {:?}",
        ui.assistant
    );
}

#[tokio::test]
async fn a_hanging_third_request_is_never_started_after_settlement() {
    missing_answer_never_starts_aftercare(false).await;
    missing_answer_never_starts_aftercare(true).await;
}

async fn missing_answer_never_starts_aftercare(capped: bool) {
    let workspace = IsolatedWorkspace::new("hanging-finalize");
    let mut cfg = workspace.config();
    cfg.memory.finalize = true;
    cfg.memory.suggest_next_prompt = true;
    if capped {
        cfg.loop_limits.max_steps = 1;
    }
    let path = workspace.path("changed.rs").to_string_lossy().to_string();
    let provider = std::sync::Arc::new(HangAfterTwoCalls {
        path,
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let mut agent = Agent::new(provider.clone(), cfg).unwrap();
    agent.side_call_timeout = Some(std::time::Duration::from_millis(25));
    let mut ui = RecUi::default();

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent.run_turn("make a file", &mut ui),
    )
    .await
    .expect("a hanging recap must be bounded")
    .expect("the primary turn should still settle normally");

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_ne!(agent.answer_state, crate::recovery::AnswerState::Accepted);
    assert_eq!(
        outcome.stop_reason,
        crate::TurnStopReason::VerificationUnavailable
    );
    if capped {
        assert!(ui.statuses.iter().any(|s| s.contains("reached step limit")));
    }
    assert!(ui.assistant.contains("The turn is closed"));
    assert_eq!(
        provider.calls.load(std::sync::atomic::Ordering::SeqCst),
        2,
        "missing-final closeout never requests a recap or suggestion, capped={capped}"
    );
}

#[tokio::test]
async fn generic_answer_completes_without_a_synthetic_failure_status() {
    let workspace = IsolatedWorkspace::new("generic-terminal-closeout");
    let mut cfg = workspace.config();
    cfg.memory.finalize = true;
    let generic = || {
        completion(
            vec![Content::Text("Completed the requested action.".into())],
            1,
            1,
        )
    };
    let provider = StreamingCanned(std::sync::Mutex::new(vec![
        echo_call(),
        generic(),
        generic(),
        generic(),
    ]));
    let mut agent = Agent::new(std::sync::Arc::new(provider), cfg).unwrap();
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn(
            "inspect the workspace and report the concrete result",
            &mut ui,
        )
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::NoApplicableVerification
    );
    assert!(
        ui.assistant.contains("Completed the requested action"),
        "the model's answer should remain visible: {}",
        ui.assistant
    );
    assert!(
        !ui.statuses
            .iter()
            .any(|status| status.contains("incomplete") || status.contains("stalled")),
        "generic answers must not manufacture legacy failure status: {:?}",
        ui.statuses
    );
    assert!(ui.turn_end.is_some(), "the turn emitted its terminal event");
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
    let workspace = IsolatedWorkspace::new("finalize-history");
    let mut cfg = workspace.config();
    cfg.memory.finalize = true;
    let tmp = workspace.path("changed.rs");
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
    cfg.memory.finalize = true;
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
async fn answerless_tool_turn_returns_provider_error_without_finalizing() {
    // Tools ran and the normal empty-response retry path still produced no
    // answer. Close-out finalize must still fire without synthetic
    // keep-working rounds.
    let mut cfg = config();
    cfg.memory.finalize = true;
    let mut responses = vec![echo_call()];
    let empty_attempts = cfg.loop_limits.max_empty_retries + 1;
    for _ in 0..empty_attempts {
        responses.push(completion(Vec::new(), 1, 0));
    }
    responses.push(completion(
        vec![Content::Text(
            "## Summary\n- Inspected the 403 notice; no edit landed.".into(),
        )],
        3,
        4,
    ));
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let error = agent.run_turn("check it", &mut ui).await.unwrap_err();
    assert!(
        format!("{error:#}").contains("model returned no response"),
        "unexpected error: {error:#}"
    );
    assert!(
        !ui.assistant
            .contains("The turn is closed with retained changes"),
        "provider failure must not run the success recap: {:?}",
        ui.assistant
    );
}

#[tokio::test]
async fn empty_recap_response_cannot_hide_an_answerless_provider_failure() {
    let workspace = IsolatedWorkspace::new("empty-closeout");
    let mut cfg = workspace.config();
    cfg.memory.finalize = true;
    let mut responses = vec![echo_call()];
    let empty_attempts = cfg.loop_limits.max_empty_retries + 1;
    for _ in 0..empty_attempts {
        responses.push(completion(Vec::new(), 3, 0));
    }
    // The final item is consumed by the ChatOnly recap side call.
    responses.push(completion(Vec::new(), 3, 0));
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let error = agent.run_turn("check it", &mut ui).await.unwrap_err();

    assert!(
        format!("{error:#}").contains("model returned no response"),
        "unexpected error: {error:#}"
    );
    assert!(
        ui.assistant.contains("No accepted final answer"),
        "provider failure needs a deterministic non-success closeout: {:?}",
        ui.assistant
    );
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
        summary.contains("user prompt estimate 1 · output across all model calls 3"),
        "turn-local prompt/output, got: {summary}"
    );
}
