use super::*;
use super::common::*;

#[tokio::test]
async fn layered_verify_stops_at_first_failing_stage() {
    let _guard = VERIFY_TEST_LOCK.lock().await;
    // The compile gate fails, so the later (passing) test stage must NOT run
    // — and the feedback should be the compile-error guidance, not the test one.
    let mut cfg = config();
    cfg.verify = vec![
        VerifyStage::new("check", "false"), // "compile" fails
        VerifyStage::new("test", "true"),   // would pass, must be skipped
    ];
    cfg.max_verify_iterations = 1;
    // The model edits (so verification runs), then stops; after the failing
    // verify it re-prompts once more before the cap is reached.
    let tmp = temp_file("stop");
    let p = tmp.to_string_lossy().to_string();
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("attempt 1".into())], 1, 1),
            completion(vec![Content::Text("attempt 2".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("x", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(agent.last_verify(), Some(false));
    // The failing stage is named…
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("check") && s.contains("failed")),
        "names the failing stage: {:?}",
        ui.statuses
    );
    // …and the later test stage never ran (no status line for it).
    assert!(
        !ui.statuses.iter().any(|s| s.contains("· test:")),
        "test stage must be skipped after the gate fails: {:?}",
        ui.statuses
    );
    // …and the feedback to the model is the compile-error nudge.
    let fed_back = agent
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text().contains("fix its root cause"));
    assert!(fed_back, "compile-stage guidance fed back");
    // The `false` command's output isn't a parseable diagnostic, so the
    // attribution layer adds no "Likely cause" section — the nudge keeps
    // its original shape (enrich-only contract).
    let has_cause = agent
        .messages()
        .iter()
        .any(|m| m.role == Role::User && m.text().contains("Likely cause"));
    assert!(!has_cause, "no attribution section for unparseable output");
}

#[tokio::test]
async fn layered_verify_passes_when_all_stages_pass() {
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let mut cfg = config();
    cfg.verify = vec![
        VerifyStage::new("check", "true"),
        VerifyStage::new("test", "true"),
    ];
    let tmp = temp_file("pass");
    let p = tmp.to_string_lossy().to_string();
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    agent.run_turn("x", &mut NullUi).await.unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(agent.last_verify(), Some(true));
}

#[tokio::test]
async fn verify_failure_exhausts_retries() {
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let mut cfg = config();
    cfg.verify = vec![VerifyStage::new("test", "false")]; // always fails
    cfg.max_verify_iterations = 2;
    // The model edits once (so verify runs), then keeps finishing without
    // tool calls; verify fails each round until the cap.
    let tmp = temp_file("exhaust");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("attempt 1".into())], 1, 1),
        completion(vec![Content::Text("attempt 2".into())], 1, 1),
        completion(vec![Content::Text("attempt 3".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    agent.run_turn("x", &mut NullUi).await.unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(agent.last_verify(), Some(false));
    // PROBE: with max_verify_iterations=2 the verifier should iterate twice.
    let tel = agent.last_turn_telemetry();
    eprintln!(
        "PROBE verify_rounds={} telemetry={:?}",
        tel.verify_rounds, tel
    );
}

#[tokio::test]
async fn verify_failure_nudge_carries_attribution() {
    let _guard = VERIFY_TEST_LOCK.lock().await;
    // A verify stage that emits a real rustc-style diagnostic should yield a
    // "Likely cause" section in the nudge pointing at the parsed file:line,
    // while the raw `Output:` block is preserved (enrich-only).
    let mut cfg = config();
    cfg.verify = vec![VerifyStage::new(
        "check",
        "printf 'error[E0308]: mismatched types\\n  --> src/lib.rs:42:18\\n' >&2; exit 1",
    )];
    cfg.max_verify_iterations = 1;
    let tmp = temp_file("attr");
    let p = tmp.to_string_lossy().to_string();
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("attempt 1".into())], 1, 1),
            completion(vec![Content::Text("attempt 2".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("x", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&tmp);
    // The attribution section is present and points at the parsed location.
    let nudge = agent
        .messages()
        .iter()
        .find(|m| m.role == Role::User && m.text().contains("Likely cause"))
        .expect("attribution section present");
    let body = nudge.text();
    assert!(
        body.contains("Likely cause (verify and fix first)"),
        "section header: {body}"
    );
    assert!(
        body.contains("src/lib.rs:42:18"),
        "parsed location in attribution: {body}"
    );
    assert!(body.contains("[compile]"), "compile kind label: {body}");
    // Enrich-only: the raw output block is still there alongside it.
    assert!(
        body.contains("Output:\n"),
        "raw Output block preserved: {body}"
    );
    assert!(
        body.contains("mismatched types"),
        "raw error message preserved in Output block: {body}"
    );
}

#[tokio::test]
async fn verify_skipped_when_no_files_changed() {
    let _guard = VERIFY_TEST_LOCK.lock().await;
    // A turn that only answers (no edits) must not run verification, even
    // when configured — so a red test suite can't hijack a question.
    let mut cfg = config();
    cfg.verify = vec![VerifyStage::new("test", "false")];
    let mut agent = agent(
        vec![completion(
            vec![Content::Text("just answering".into())],
            1,
            1,
        )],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("what does this do?", &mut ui).await.unwrap();
    assert_eq!(agent.last_verify(), None, "verify must not have run");
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("skipped — no files changed")),
        "skip is surfaced: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn verify_runs_when_bash_changes_files() {
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let tmp = temp_file("bash");
    let p = tmp.to_string_lossy().to_string();
    let mut cfg = config();
    cfg.verify = vec![VerifyStage::new("test", "true")];
    let mut agent = agent(
        vec![
            completion(
                vec![Content::ToolCall {
                    id: "b".into(),
                    name: "bash".into(),
                    arguments: format!("{{\"command\":\"printf x > '{}'\"}}", p),
                }],
                1,
                1,
            ),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    agent.run_turn("x", &mut NullUi).await.unwrap();
    let _ = std::fs::remove_file(&tmp);
    assert_eq!(agent.last_verify(), Some(true));
}

#[tokio::test]
async fn proactive_verify_surfaces_a_per_edit_check_failure() {
    // With proactive_verify on, a write to a .py file with a syntax error
    // triggers a background `python3 -m py_compile` whose failure surfaces
    // as a status line during the turn (before turn-end verify). Skipped if
    // python3 isn't on PATH (the check just won't run).
    if std::process::Command::new("sh")
        .arg("-c")
        .arg("command -v python3")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: python3 not on PATH");
        return;
    }
    let _guard = VERIFY_TEST_LOCK.lock().await;
    let mut cfg = config();
    cfg.proactive_verify = true;
    let tmp = temp_file("proactive");
    let py = tmp.with_extension("py");
    let p = py.to_string_lossy().to_string();
    // Write invalid Python so py_compile fails.
    let responses = vec![
        Completion {
            content: vec![Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!(r#"{{"path":{p:?},"content":"def (\n"}}"#),
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                context_occupancy: 1,
                ..Default::default()
            },
            stop_reason: None,
        },
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("write it", &mut ui).await.unwrap();
    let _ = std::fs::remove_file(&py);
    // A proactive-check failure status line names the file.
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("proactive check failed") && s.contains(&p)),
        "proactive failure surfaced: {:?}",
        ui.statuses
    );
}

