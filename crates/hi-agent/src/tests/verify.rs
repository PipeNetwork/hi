use super::common::*;
use super::*;

#[tokio::test]
async fn layered_verify_stops_at_first_failing_stage() {
    let workspace = IsolatedWorkspace::new("verify-stop");
    // The compile gate fails, so the later (passing) test stage must NOT run
    // — and the feedback should be the compile-error guidance, not the test one.
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Explicit(vec![
        VerifyStage::new("check", "false"), // "compile" fails
        VerifyStage::new("test", "true"),   // would pass, must be skipped
    ]);
    cfg.max_verify_repairs = 0;
    // The model edits (so verification runs), then stops; after the failing
    // verify it re-prompts once more before the cap is reached.
    let tmp = workspace.path("changed.rs");
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
    // Automatic checkpoints let the verifier rerun the failed stage against
    // the pre-turn workspace. Even though `false` has no diagnostic body, the
    // nudge should accurately identify this as a pre-existing failure.
    let cause = agent
        .messages()
        .iter()
        .find(|m| m.role == Role::User && m.text().contains("Likely cause"))
        .expect("pre-turn attribution section");
    assert!(
        cause
            .text()
            .contains("also failed in an isolated pre-turn workspace")
    );
}

#[tokio::test]
async fn layered_verify_passes_when_all_stages_pass() {
    let workspace = IsolatedWorkspace::new("verify-pass");
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Explicit(vec![
        VerifyStage::new("check", "true"),
        VerifyStage::new("test", "true"),
    ]);
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    agent.run_turn("x", &mut NullUi).await.unwrap();
    assert_eq!(agent.last_verify(), Some(true));
}

#[tokio::test]
async fn verify_failure_exhausts_retries() {
    let workspace = IsolatedWorkspace::new("verify-exhaust");
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]); // always fails
    cfg.max_verify_repairs = 1;
    // The model edits once (so verify runs), then keeps finishing without
    // tool calls; verify fails each round until the cap.
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("attempt 1".into())], 1, 1),
        completion(vec![Content::Text("attempt 2".into())], 1, 1),
        completion(vec![Content::Text("attempt 3".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    agent.run_turn("x", &mut NullUi).await.unwrap();
    assert_eq!(agent.last_verify(), Some(false));
    assert_eq!(agent.last_turn_telemetry().verify_rounds, 2);
    assert!(agent.last_turn_telemetry().stalled_unfinished);
}

#[tokio::test]
async fn verify_failure_exhaustion_does_not_finalize_as_done() {
    let workspace = IsolatedWorkspace::new("verify-no-finalize");
    let mut cfg = workspace.config();
    cfg.finalize = true;
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.max_verify_repairs = 0;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("attempt 1".into())], 1, 1),
        completion(vec![Content::Text("attempt 2".into())], 1, 1),
        // Would be consumed by finalize_turn if failed verification were
        // incorrectly treated as a completed turn.
        completion(
            vec![Content::Text("FINALIZE RECAP SHOULD NOT RUN".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("x", &mut ui).await.unwrap();

    assert_eq!(agent.last_verify(), Some(false));
    assert!(
        agent.last_turn_telemetry().stalled_unfinished,
        "failed verification exhaustion should be an unfinished turn"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("verification still failed after the retry budget")),
        "expected explicit exhausted-verify status, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.assistant.contains("FINALIZE RECAP SHOULD NOT RUN"),
        "failed verification must not trigger finalization, assistant was: {}",
        ui.assistant
    );
}

#[tokio::test]
async fn verify_failure_nudge_carries_attribution() {
    let workspace = IsolatedWorkspace::new("verify-attribution");
    // A verify stage that emits a real rustc-style diagnostic should yield a
    // "Likely cause" section in the nudge pointing at the parsed file:line,
    // while the raw `Output:` block is preserved (enrich-only).
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new(
        "check",
        "printf 'error[E0308]: mismatched types\\n  --> src/lib.rs:42:18\\n' >&2; exit 1",
    )]);
    cfg.max_verify_repairs = 0;
    let tmp = workspace.path("changed.rs");
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
    let workspace = IsolatedWorkspace::new("verify-no-changes");
    // A turn that only answers (no edits) must not run verification, even
    // when configured — so a red test suite can't hijack a question.
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
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
async fn auto_verify_skips_prose_only_changes() {
    let workspace = IsolatedWorkspace::new("verify-auto-docs");
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"docs-only\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    let tmp = workspace.path("README.md");
    let p = tmp.to_string_lossy().to_string();
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Auto;
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("docs updated".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();

    agent.run_turn("update docs", &mut ui).await.unwrap();

    assert_eq!(agent.last_verify(), None, "automatic code checks may skip");
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("skipped — prose-only")),
        "automatic prose-only skip is surfaced: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn explicit_verify_runs_for_prose_only_changes() {
    let workspace = IsolatedWorkspace::new("verify-docs");
    let tmp = workspace.path("README.md");
    let p = tmp.to_string_lossy().to_string();
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("docs", "true")]);
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("docs updated".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("update docs", &mut ui).await.unwrap();
    assert_eq!(
        agent.last_verify(),
        Some(true),
        "explicit verifier must run"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("verifying") && s.contains("· docs: true")),
        "explicit documentation verifier result is surfaced: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn verify_runs_when_bash_changes_files() {
    let workspace = IsolatedWorkspace::new("verify-bash");
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let mut cfg = workspace.config();
    cfg.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
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
    let workspace = IsolatedWorkspace::new("verify-proactive");
    let mut cfg = workspace.config();
    cfg.proactive_verify = true;
    let py = workspace.path("invalid.py");
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
    // A proactive-check failure status line names the file.
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("proactive check failed") && s.contains(&p)),
        "proactive failure surfaced: {:?}",
        ui.statuses
    );
}
