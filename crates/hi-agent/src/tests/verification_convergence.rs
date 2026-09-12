use super::common::*;
use super::*;

fn independent_review_cfg(workspace: &IsolatedWorkspace) -> AgentConfig {
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "true")]);
    cfg.gates.review = ReviewPolicy::Always;
    cfg.gates.lsp_mode = LspMode::Off;
    cfg.gates.allow_no_checkpoint = true;
    cfg
}

fn write_file_completion(id: &str, path: &str, content: &str) -> Completion {
    completion(
        vec![Content::ToolCall {
            id: id.into(),
            name: "write".into(),
            arguments: serde_json::json!({ "path": path, "content": content }).to_string(),
        }],
        1,
        1,
    )
}

#[tokio::test]
async fn default_unlimited_verification_stops_after_an_unchanged_repair() {
    let workspace = IsolatedWorkspace::new("verify-unlimited-no-progress");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    cfg.gates.max_verify_repairs = AgentGates::default().max_verify_repairs;
    assert_eq!(cfg.gates.max_verify_repairs, crate::UNLIMITED_REPAIR_CYCLES);

    let path = workspace.path("changed.rs").to_string_lossy().to_string();
    let responses = vec![
        write_completion(&path),
        completion(vec![Content::Text("initial attempt".into())], 1, 1),
        completion(
            vec![Content::Text(
                "the repair attempt made no workspace change".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("make changed.rs pass its verification gate", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::NoProgress);
    assert_eq!(agent.last_turn_telemetry().verify_rounds, 2);
    assert!(
        agent.task_recovery().exhausted,
        "an unchanged failed revision exhausts the shared recovery episode"
    );
}

#[tokio::test]
async fn ignored_runtime_database_reset_requires_only_memory_revalidation() {
    let workspace = IsolatedWorkspace::new("outcome-hygiene-ignored-runtime-db");
    let git = |args: &[&str]| {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(workspace.path(""))
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    };
    git(&["init", "-q"]);
    git(&["config", "user.email", "test@example.com"]);
    git(&["config", "user.name", "Test"]);
    std::fs::create_dir_all(workspace.path("src")).unwrap();
    std::fs::write(
        workspace.path("Cargo.toml"),
        "[package]\nname = \"ignored-runtime-db\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path("src/lib.rs"),
        "pub fn answer() -> u8 { 42 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn answers() { assert_eq!(super::answer(), 42); }\n}\n",
    )
    .unwrap();
    std::fs::write(
        workspace.path(".gitignore"),
        "/target/\n/.hi/\nalovewtf.db\n",
    )
    .unwrap();
    let cargo = std::process::Command::new("cargo")
        .args(["generate-lockfile", "--quiet"])
        .current_dir(workspace.path(""))
        .output()
        .unwrap();
    assert!(
        cargo.status.success(),
        "cargo generate-lockfile failed: {}",
        String::from_utf8_lossy(&cargo.stderr)
    );
    git(&[
        "add",
        ".gitignore",
        "Cargo.toml",
        "Cargo.lock",
        "src/lib.rs",
    ]);
    git(&["commit", "-qm", "baseline"]);

    // Match the live failure: the pre-turn ignored database exceeded the
    // checkpoint ignored-input cap, then reset to a tiny fresh database. The
    // durability ledger must retain that 17 MiB delta, while diff hygiene must
    // not mistake it for a reviewable source rewrite.
    let database = std::fs::File::create(workspace.path("alovewtf.db")).unwrap();
    database.set_len(17 * 1024 * 1024).unwrap();

    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![
        VerifyStage::new("check", "cargo check --quiet"),
        VerifyStage::new("test", "cargo test --quiet"),
    ]);
    cfg.gates.review = ReviewPolicy::Off;
    cfg.gates.max_verify_repairs = crate::UNLIMITED_REPAIR_CYCLES;
    cfg.gates.max_independent_review_repairs = crate::UNLIMITED_REPAIR_CYCLES;
    let mut agent = agent(
        vec![
            bash_completion("rm -f alovewtf.db && printf 'fresh\\n' > alovewtf.db"),
            completion(vec![Content::Text("database reset complete".into())], 1, 1),
        ],
        cfg,
    );
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("reset alovewtf.db and start the program again", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert!(
        ui.statuses
            .iter()
            .all(|status| !status.contains("diff hygiene")),
        "ignored runtime data must not enter merge-quality repair: {:?}",
        ui.statuses
    );
    let verification = ui
        .statuses
        .iter()
        .filter(|status| status.contains("verifying ("))
        .collect::<Vec<_>>();
    assert_eq!(
        verification.len(),
        2,
        "one check and one test run for the edit; prose memory does not spend another pipeline: {verification:?}"
    );
    assert_eq!(
        verification
            .iter()
            .filter(|status| status.contains("verifying (1/unlimited)"))
            .count(),
        2,
        "the attested revision should execute one pipeline: {verification:?}"
    );
    assert_eq!(
        outcome.verified_workspace_revision,
        Some(agent.runtime.ledger().workspace_revision())
    );
    let database_change = agent
        .last_file_changes()
        .iter()
        .find(|change| change.path == "alovewtf.db")
        .expect("durability accounting must retain the ignored database effect");
    assert_eq!(database_change.before_len, Some(17 * 1024 * 1024));
    assert_eq!(database_change.after_len, Some(6));
    let status = std::process::Command::new("git")
        .arg("-C")
        .arg(workspace.path(""))
        .args(["status", "--porcelain"])
        .output()
        .unwrap();
    assert!(status.status.success());
    assert!(
        status.stdout.is_empty(),
        "the ignored database reset is absent from the repository diff: {}",
        String::from_utf8_lossy(&status.stdout)
    );
}

#[tokio::test]
async fn default_unlimited_hygiene_stops_after_an_unchanged_repair() {
    let workspace = IsolatedWorkspace::new("outcome-hygiene-unlimited-no-progress");
    let mut cfg = independent_review_cfg(&workspace);
    cfg.gates.review = ReviewPolicy::Off;
    assert_eq!(
        cfg.gates.max_independent_review_repairs,
        crate::UNLIMITED_REPAIR_CYCLES
    );
    let large_source = format!(
        "// generated source fixture\n// {}\n",
        "x".repeat(crate::hygiene::LARGE_FILE_BYTES as usize + 1)
    );
    let responses = vec![
        write_file_completion("large-source", "src/generated.rs", &large_source),
        bash_completion("python3 -c 'assert 2 + 2 == 4'"),
        completion(vec![Content::Text("initial implementation".into())], 1, 1),
        bash_completion("true # hygiene repair"),
        completion(
            vec![Content::Text(
                "the repair attempt made no workspace change".into(),
            )],
            1,
            1,
        ),
    ];
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("create a generated source fixture", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Objected);
    assert_eq!(outcome.stop_reason, TurnStopReason::ReviewObjected);
    assert_eq!(agent.last_turn_telemetry().verify_rounds, 2);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("diff hygiene still objects without a workspace change")),
        "an unchanged hygiene objection must settle instead of consuming unlimited rounds: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn default_unlimited_completion_review_stops_without_a_workspace_change() {
    let workspace = IsolatedWorkspace::new("outcome-review-unlimited-no-progress");
    let responses = vec![
        write_file_completion("write-review", "reviewed.txt", "v1\n"),
        bash_completion("python3 -c 'assert 2 + 2 == 4'"),
        completion(vec![Content::Text("initial implementation".into())], 1, 1),
        completion(
            vec![Content::Text("OBJECT\n- concrete defect".into())],
            1,
            1,
        ),
        completion(
            vec![Content::Text(
                "the repair attempt made no workspace change".into(),
            )],
            1,
            1,
        ),
    ];
    let cfg = independent_review_cfg(&workspace);
    assert_eq!(
        cfg.gates.max_independent_review_repairs,
        crate::UNLIMITED_REPAIR_CYCLES
    );
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();

    let outcome = agent
        .run_turn("implement the reviewed file", &mut ui)
        .await
        .unwrap();

    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(outcome.review, ReviewStatus::Objected);
    assert_eq!(outcome.stop_reason, TurnStopReason::ReviewObjected);
    assert_eq!(agent.last_turn_telemetry().verify_rounds, 2);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status
                .contains("completion review still objects without a workspace change")),
        "an unchanged review objection must settle instead of being re-reviewed forever: {:?}",
        ui.statuses
    );
}

#[tokio::test]
async fn genuinely_reducing_failures_continues_beyond_three_repairs() {
    let workspace = IsolatedWorkspace::new("shared-recovery-improves");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new(
        "required",
        "cat remaining.rs; test ! -s remaining.rs",
    )]);
    cfg.gates.max_verify_repairs = crate::UNLIMITED_REPAIR_CYCLES;
    cfg.memory.tool_set = ToolSet::Full;
    cfg.gates.lsp_mode = LspMode::Off;
    let path = workspace.path("remaining.rs").to_string_lossy().to_string();
    let mut responses = Vec::new();
    for remaining in (0..=5).rev() {
        let output = (0..remaining)
            .map(|i| format!("error: failure_{i}\n"))
            .collect::<String>();
        responses.push(write_file_completion(
            &format!("write-{remaining}"),
            &path,
            &output,
        ));
        responses.push(completion(
            vec![Content::Text(format!("Applied repair {remaining}."))],
            1,
            1,
        ));
    }
    let mut agent = agent(responses, cfg);
    let mut ui = RecUi::default();
    let outcome = agent
        .run_turn(
            "/build remaining.rs, reducing its diagnostic failures until the required check passes",
            &mut ui,
        )
        .await
        .unwrap();
    assert_eq!(
        outcome.status,
        TurnStatus::Completed,
        "{outcome:?}; {:?}; {:?}",
        ui.statuses,
        agent.task_recovery()
    );
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert!(agent.last_turn_telemetry().verify_rounds > 3);
    assert!(!agent.task_recovery().exhausted);
}
