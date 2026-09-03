use super::common::*;
use super::*;

#[tokio::test]
async fn layered_verify_stops_at_first_failing_stage() {
    let workspace = IsolatedWorkspace::new("verify-stop");
    // The compile gate fails, so the later (passing) test stage must NOT run
    // — and the feedback should be the compile-error guidance, not the test one.
    let mut cfg = workspace.config();
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![
        VerifyStage::new("check", "false"), // "compile" fails
        VerifyStage::new("test", "true"),   // would pass, must be skipped
    ]);
    cfg.gates.max_verify_repairs = 0;
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
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![
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
async fn green_turn_records_coding_facts_into_decisions() {
    let workspace = IsolatedWorkspace::new("coding-facts");
    let mut cfg = workspace.config();
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("check", "true")]);
    let tmp = workspace.path("src/lib.rs");
    std::fs::create_dir_all(tmp.parent().unwrap()).unwrap();
    let p = tmp.to_string_lossy().to_string();
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent
        .run_turn("fix the helper and keep tests green", &mut ui)
        .await
        .unwrap();
    assert_eq!(agent.last_verify(), Some(true));
    assert!(
        !agent.decisions().is_empty(),
        "expected auto coding facts in decision log"
    );
    assert!(
        agent
            .decisions()
            .entries()
            .iter()
            .any(|d| d.summary.starts_with("verify:") || d.summary.starts_with("stack:")),
        "facts: {:?}",
        agent.decisions().entries()
    );
    assert!(
        ui.statuses.iter().any(|s| s.contains("coding memory")),
        "status: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn verify_failure_exhausts_retries() {
    let workspace = IsolatedWorkspace::new("verify-exhaust");
    let mut cfg = workspace.config();
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]); // always fails
    cfg.gates.max_verify_repairs = 1;
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
    let outcome = agent.run_turn("x", &mut NullUi).await.unwrap();
    assert_eq!(agent.last_verify(), Some(false));
    assert_eq!(agent.last_turn_telemetry().verify_rounds, 2);
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationFailed);
}

#[tokio::test]
async fn default_verification_repairs_continue_past_two_productive_cycles() {
    let workspace = IsolatedWorkspace::new("verify-unlimited-default");
    let mut cfg = workspace.config();
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new(
        "test",
        "test \"$(cat changed.rs)\" = 3",
    )]);
    cfg.gates.max_verify_repairs = AgentGates::default().max_verify_repairs;
    assert_eq!(cfg.gates.max_verify_repairs, crate::UNLIMITED_REPAIR_CYCLES);

    let responses = vec![
        write_content_completion("changed.rs", "0\n"),
        completion(vec![Content::Text("initial attempt".into())], 1, 1),
        write_content_completion("changed.rs", "1\n"),
        completion(vec![Content::Text("repair one".into())], 1, 1),
        write_content_completion("changed.rs", "2\n"),
        completion(vec![Content::Text("repair two".into())], 1, 1),
        write_content_completion("changed.rs", "3\n"),
        completion(vec![Content::Text("repair three".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("make changed.rs pass its verification gate", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(agent.last_verify(), Some(true));
    assert_eq!(agent.last_turn_telemetry().verify_rounds, 4);
    assert_eq!(
        std::fs::read_to_string(workspace.path("changed.rs")).unwrap(),
        "3\n"
    );
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("verifying (4/unlimited)")),
        "unlimited repair status should remain human-readable: {:?}",
        ui.statuses
    );
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("4294967295")),
        "the unlimited sentinel must not leak into UI text: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn verify_failure_remains_failed_without_a_duplicate_recap() {
    let workspace = IsolatedWorkspace::new("verify-no-finalize");
    let mut cfg = workspace.config();
    cfg.memory.finalize = true;
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.gates.max_verify_repairs = 0;
    let tmp = workspace.path("changed.rs");
    let p = tmp.to_string_lossy().to_string();
    let responses = vec![
        write_completion(&p),
        completion(vec![Content::Text("attempt 1".into())], 1, 1),
        completion(vec![Content::Text("attempt 2".into())], 1, 1),
        completion(
            vec![Content::Text("FINALIZE RECAP DESCRIBES FAILURE".into())],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let outcome = agent.run_turn("x", &mut ui).await.unwrap();

    assert_eq!(agent.last_verify(), Some(false));
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::VerificationFailed);
    assert!(
        ui.statuses
            .iter()
            .any(|s| s.contains("verification still failed after repair")),
        "expected explicit exhausted-verify status, got: {:?}",
        ui.statuses
    );
    assert!(
        !ui.assistant.contains("FINALIZE RECAP DESCRIBES FAILURE"),
        "a visible failed answer must not trigger a duplicate recap request: {}",
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
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new(
        "check",
        "printf 'error[E0308]: mismatched types\\n  --> src/lib.rs:42:18\\n' >&2; exit 1",
    )]);
    cfg.gates.max_verify_repairs = 0;
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
async fn obligation_nudges_when_mutation_never_verified() {
    // Explicit verify stages exist, the model mutates, then stops with a text
    // answer — but verification is somehow never sealed green (Disabled would
    // skip obligation; here stages are present and Auto would run). Force the
    // gap by using Explicit stages that are never reached: the model finishes
    // after write with no further tool use while we stub verification off after
    // the write via Disabled… that must NOT fire. Instead: Auto on a tree with
    // no Cargo.toml so Auto resolves empty stages → NotApplicable, not obligation.
    // Real gap: mutation + Explicit stages + SkippedNoChanges is rare. Cover the
    // unit via coding_verify_obligation tests; here assert Failed still structures.
    let workspace = IsolatedWorkspace::new("verify-obligation-structure");
    let mut cfg = workspace.config();
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![VerifyStage::new(
        "check",
        "printf 'error[E0425]: cannot find value `foo`\\n  --> src/main.rs:3:5\\n' >&2; exit 1",
    )]);
    cfg.gates.max_verify_repairs = 0;
    let tmp = workspace.path("src/main.rs");
    std::fs::create_dir_all(tmp.parent().unwrap()).unwrap();
    let p = tmp.to_string_lossy().to_string();
    let mut agent = agent(
        vec![
            write_completion(&p),
            completion(vec![Content::Text("attempt".into())], 1, 1),
            completion(vec![Content::Text("done".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();
    agent.run_turn("fix foo", &mut ui).await.unwrap();
    let body = agent
        .messages()
        .iter()
        .find(|m| m.role == Role::User && m.text().contains("Verification stage"))
        .map(|m| m.text())
        .unwrap_or_default();
    assert!(
        body.contains("Likely cause") && body.contains("src/main.rs:3:5"),
        "structured verify failure: {body}"
    );
}

#[tokio::test]
async fn verify_skipped_when_no_files_changed() {
    let workspace = IsolatedWorkspace::new("verify-no-changes");
    // A turn that only answers (no edits) must not run verification, even
    // when configured — so a red test suite can't hijack a question.
    let mut cfg = workspace.config();
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
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
    cfg.gates.verification = crate::VerificationMode::Auto;
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
            .any(|s| s.contains("not required — prose-only")),
        "automatic prose-only non-requirement is surfaced: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn explicit_verify_runs_for_prose_only_changes() {
    let workspace = IsolatedWorkspace::new("verify-docs");
    let tmp = workspace.path("README.md");
    let p = tmp.to_string_lossy().to_string();
    let mut cfg = workspace.config();
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("docs", "true")]);
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
    cfg.gates.verification =
        crate::VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
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
    cfg.gates.proactive_verify = true;
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
            ..Completion::default()
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

#[tokio::test]
async fn proactive_verify_replays_a_successful_check_to_the_model() {
    // A passing per-edit check is deliberately not emitted as user-facing
    // status. It must still be attached to the tool result that is replayed
    // to the model, otherwise a reasoning model may run the same validation
    // again in a needless shell round. Skipped when the check can't actually
    // run here (no python3, or a sandbox blocking its bytecode cache).
    let workspace = IsolatedWorkspace::new("verify-proactive-pass");
    if !python_fast_check_works(&workspace.path("")) {
        eprintln!("skipping: python3 -m py_compile cannot run in this environment");
        return;
    }
    let mut cfg = workspace.config();
    cfg.gates.proactive_verify = true;
    let py = workspace.path("valid.py");
    let p = py.to_string_lossy().to_string();
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write_content_completion(
                &p,
                "def greeting(name):\n    return f'Hi, {name}!'\n",
            )),
            ProviderStep::Completion(completion(vec![Content::Text("done".into())], 1, 1)),
        ],
        cfg,
    );

    agent
        .run_turn("write valid Python", &mut NullUi)
        .await
        .unwrap();

    let requests = requests.lock().unwrap();
    let second_request = requests
        .get(1)
        .expect("the model should receive a follow-up request after the edit");
    assert!(
        second_request.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    Content::ToolResult { output, .. }
                        if output.contains("✓ fast check passed")
                )
            })
        }),
        "successful fast check must be replayed in the next tool result: {second_request:?}"
    );
}

#[tokio::test]
async fn mid_turn_pytest_runs_when_task_is_test_gated() {
    if std::process::Command::new("pytest")
        .arg("--version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: pytest not on PATH");
        return;
    }
    let workspace = IsolatedWorkspace::new("verify-py-fast-test");
    std::fs::write(
        workspace.path("pyproject.toml"),
        "[project]\nname = \"demo\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path("test_demo.py"),
        "def test_ok():\n    assert True\n",
    )
    .unwrap();

    let mut cfg = workspace.config();
    cfg.gates.lsp_mode = crate::LspMode::Off;
    cfg.gates.verification = crate::VerificationMode::Disabled;
    cfg.gates.max_verify_repairs = 0;

    let path = workspace.path("test_demo.py");
    let p = path.to_string_lossy().to_string();
    let broken = "def test_ok():\\n    assert False\\n";
    let responses = vec![
        Completion {
            content: vec![Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!(r#"{{"path":{p:?},"content":"{broken}"}}"#),
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                context_occupancy: 1,
                ..Default::default()
            },
            stop_reason: None,
            ..Completion::default()
        },
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("fix the failing unit tests", &mut ui)
        .await
        .unwrap();

    let saw = ui
        .statuses
        .iter()
        .any(|s| s.contains("pytest") || s.contains("package test") || s.starts_with('✗'))
        || agent.messages().iter().any(|m| {
            let t = m.text();
            t.contains("pytest") || t.contains("AssertionError") || t.contains("assert")
        });
    assert!(
        saw,
        "mid-turn pytest failure should surface; statuses={:?} messages={:?}",
        ui.statuses,
        agent
            .messages()
            .iter()
            .map(|m| m.text())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn mid_turn_cargo_test_runs_when_task_is_test_gated() {
    // LSP off → check then test on a test-gated prompt; a failing unit test
    // must surface mid-turn (before WorkspaceRepair).
    if std::process::Command::new("cargo")
        .arg("--version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: cargo not on PATH");
        return;
    }
    let workspace = IsolatedWorkspace::new("verify-rust-fast-test");
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "pub fn ok() -> i32 { 1 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn it_works() { assert_eq!(super::ok(), 1); }\n}\n",
    )
    .unwrap();

    let mut cfg = workspace.config();
    cfg.gates.lsp_mode = crate::LspMode::Off;
    cfg.gates.verification = crate::VerificationMode::Disabled;
    cfg.gates.max_verify_repairs = 0;

    let path = workspace.path("src/lib.rs");
    let p = path.to_string_lossy().to_string();
    // Break the unit test assertion.
    let broken = "pub fn ok() -> i32 { 1 }\\n\\n#[cfg(test)]\\nmod tests {\\n    #[test]\\n    fn it_works() { assert_eq!(super::ok(), 99); }\\n}\\n";
    let responses = vec![
        Completion {
            content: vec![Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!(r#"{{"path":{p:?},"content":"{broken}"}}"#),
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                context_occupancy: 1,
                ..Default::default()
            },
            stop_reason: None,
            ..Completion::default()
        },
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent
        .run_turn("fix the failing unit tests in demo", &mut ui)
        .await
        .unwrap();

    let saw_test = ui
        .statuses
        .iter()
        .any(|s| s.contains("cargo test") || (s.starts_with('✗') && s.contains("assert")))
        || agent.messages().iter().any(|m| {
            let t = m.text();
            t.contains("cargo test") || t.contains("it_works") || t.contains("assertion")
        });
    assert!(
        saw_test,
        "mid-turn cargo test failure should surface; statuses={:?} messages={:?}",
        ui.statuses,
        agent
            .messages()
            .iter()
            .map(|m| m.text())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn mid_turn_cargo_fast_check_surfaces_on_broken_rust() {
    // LSP is off so Tier 2 cargo check always runs after a .rs mutation.
    if std::process::Command::new("cargo")
        .arg("--version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        eprintln!("skipping: cargo not on PATH");
        return;
    }
    let workspace = IsolatedWorkspace::new("verify-rust-fast");
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(workspace.path("src/lib.rs"), "pub fn ok() {}\n").unwrap();

    let mut cfg = workspace.config();
    cfg.gates.lsp_mode = crate::LspMode::Off;
    // Turn-end verify off so we only assert mid-turn feedback.
    cfg.gates.verification = crate::VerificationMode::Disabled;
    cfg.gates.max_verify_repairs = 0;

    let p = workspace.path("src/lib.rs");
    let path = p.to_string_lossy().to_string();
    let responses = vec![
        Completion {
            content: vec![Content::ToolCall {
                id: "w".into(),
                name: "write".into(),
                arguments: format!(r#"{{"path":{path:?},"content":"pub fn broken( -> {{}}\n"}}"#),
            }],
            usage: Usage {
                input_tokens: 1,
                output_tokens: 1,
                context_occupancy: 1,
                ..Default::default()
            },
            stop_reason: None,
            ..Completion::default()
        },
        completion(vec![Content::Text("done".into())], 1, 1),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    agent.run_turn("break the rust", &mut ui).await.unwrap();

    let saw_status = ui.statuses.iter().any(|s| {
        (s.contains("fast check") && s.contains("cargo"))
            || s.contains("unclosed delimiter")
            || (s.starts_with('✗') && s.contains("src/lib.rs"))
    });
    let saw_transcript = agent.messages().iter().any(|m| {
        let t = m.text();
        t.contains("fast check") || t.contains("Likely cause") || t.contains("unclosed delimiter")
    });
    assert!(
        saw_status || saw_transcript,
        "mid-turn cargo failure should surface in status or transcript; statuses={:?} messages={:?}",
        ui.statuses,
        agent
            .messages()
            .iter()
            .map(|m| m.text())
            .collect::<Vec<_>>()
    );
}
