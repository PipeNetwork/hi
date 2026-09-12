//! Regression coverage for unbounded per-session coding-fact recording.

use super::common::*;
use super::*;

#[tokio::test]
async fn coding_facts_continue_after_the_previous_session_cap() {
    let cfg = config();
    std::fs::write(
        cfg.paths.workspace_root.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::create_dir_all(cfg.paths.workspace_root.join("src")).unwrap();
    std::fs::write(
        cfg.paths.workspace_root.join("src/lib.rs"),
        "pub fn ok() {}\n",
    )
    .unwrap();

    let mut agent = agent(Vec::new(), cfg);
    agent.subagents.coding_facts_written = 8;
    agent.report.verify = crate::domain::VerifyEvidence::pass(1, "digest".into());
    agent.workspace.last_changed_files = vec!["src/lib.rs".into()];

    agent.record_coding_facts_turn_end(&mut NullUi).await;

    assert!(
        agent.subagents.coding_facts_written > 8,
        "the old per-session count must remain telemetry, not a stop condition"
    );
    assert!(
        agent
            .decisions
            .entries()
            .iter()
            .any(|decision| decision.summary.contains("Rust")),
        "a fact discovered after the old boundary must still be recorded"
    );
}

async fn run_enrichment_turn(
    workspace: &IsolatedWorkspace,
    command: &str,
    repairs: u32,
    curate: bool,
) -> (Agent, TurnOutcome, RecUi) {
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("check", command)]);
    cfg.gates.max_verify_repairs = repairs;
    cfg.gates.review = ReviewPolicy::Off;
    cfg.memory.curate_skills = curate;
    let mut responses = vec![
        completion(vec![Content::ToolCall {
            id: "patch".into(),
            name: "apply_patch".into(),
            arguments: serde_json::json!({
                "patch": "*** Begin Patch\n*** Add File: changed.rs\n+pub fn value() -> u32 { 42 }\n*** End Patch"
            }).to_string(),
        }], 1, 1),
        completion(vec![Content::Text("Updated changed.rs with value(), which returns 42. The check completed successfully.".into())], 1, 1),
    ];
    if curate {
        responses.push(completion(vec![Content::Text(
            "---\nname: Preserve Checked Inputs\ndescription: Bind validation to exact bytes.\nscope: project\n---\n# Preserve Checked Inputs\n\nRecheck after changing validation inputs.".into()
        )], 1, 1));
    }
    let mut subject = agent(responses, cfg);
    let mut ui = RecUi::default();
    let outcome = subject
        .run_turn("update changed.rs", &mut ui)
        .await
        .unwrap_or_else(|error| {
            panic!(
                "{error:#}; statuses={:?}; messages={:?}",
                ui.statuses,
                subject
                    .messages()
                    .iter()
                    .map(Message::text)
                    .collect::<Vec<_>>()
            )
        });
    (subject, outcome, ui)
}

#[tokio::test]
async fn enrichment_revalidation_failure_settles_without_another_model_request() {
    let workspace = IsolatedWorkspace::new("enrichment-check-fails");
    let (subject, outcome, ui) = run_enrichment_turn(
        &workspace,
        "test ! -e .hi/memory.md",
        crate::UNLIMITED_REPAIR_CYCLES,
        false,
    )
    .await;

    assert!(workspace.path(".hi/memory.md").exists());
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.verification, VerificationStatus::Failed);
    assert_eq!(subject.last_turn_telemetry().verify_rounds, 2);
    assert_eq!(subject.last_turn_telemetry().model_requests, 2);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("without another model request"))
    );
}

#[tokio::test]
async fn enrichment_revalidation_respects_the_explicit_verification_ceiling() {
    let workspace = IsolatedWorkspace::new("enrichment-check-cap");
    let (subject, outcome, _) = run_enrichment_turn(&workspace, "true", 0, false).await;

    assert!(workspace.path(".hi/memory.md").exists());
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(subject.last_turn_telemetry().verify_rounds, 1);
    assert_eq!(subject.last_turn_telemetry().model_requests, 2);
}

#[tokio::test]
async fn curated_skill_and_coding_memory_share_one_revalidation() {
    let workspace = IsolatedWorkspace::new("enrichment-curation");
    let (subject, outcome, ui) = run_enrichment_turn(&workspace, "true", 1, true).await;

    assert_eq!(outcome.status, TurnStatus::Completed, "{:?}", ui.statuses);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(subject.subagents.auto_skills_written, 1);
    assert!(
        workspace
            .path(".hi/skills/preserve-checked-inputs/SKILL.md")
            .exists()
    );
    assert!(workspace.path(".hi/memory.md").exists());
    assert_eq!(subject.last_turn_telemetry().verify_rounds, 1);
    assert_eq!(subject.last_turn_telemetry().model_requests, 2);
    assert_eq!(
        outcome.verified_workspace_revision,
        Some(subject.runtime.ledger().workspace_revision())
    );
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|status| status.as_str() == "curating skill…")
            .count(),
        1
    );
}

#[tokio::test]
async fn current_coding_memory_does_not_spend_another_verification_round() {
    let workspace = IsolatedWorkspace::new("enrichment-current-memory");
    let (first, outcome, _) = run_enrichment_turn(&workspace, "true", 1, false).await;
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    drop(first);
    std::fs::remove_file(workspace.path("changed.rs")).unwrap();
    // The internal undo sidecar participates in the canonical input too. Make
    // both planned write targets current so this exercises an actual no-op.
    std::fs::copy(
        workspace.path(".hi/memory.md"),
        workspace.path(".hi/memory.undo.md"),
    )
    .unwrap();

    let (subject, outcome, ui) = run_enrichment_turn(&workspace, "true", 0, false).await;

    assert_eq!(outcome.status, TurnStatus::Completed, "{:?}", ui.statuses);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(subject.last_turn_telemetry().verify_rounds, 1);
    assert!(
        ui.statuses
            .iter()
            .any(|status| status.contains("memory already current"))
    );
    assert_eq!(
        outcome.verified_workspace_revision,
        Some(subject.runtime.ledger().workspace_revision())
    );
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
    let outcome = agent
        .run_turn("fix the helper and keep checks green", &mut ui)
        .await
        .unwrap();
    assert_eq!(outcome.status, TurnStatus::Completed);
    assert_eq!(outcome.verification, VerificationStatus::Passed);
    assert_eq!(agent.last_verify(), Some(true));
    assert_eq!(agent.last_turn_telemetry().verify_rounds, 1);
    assert_eq!(agent.last_turn_telemetry().model_requests, 2);
    assert_eq!(
        outcome.verified_workspace_revision,
        Some(agent.runtime.ledger().workspace_revision()),
        "the final seal must attest the bytes including the new coding memory"
    );
    assert_eq!(
        ui.statuses
            .iter()
            .filter(|status| status.starts_with("coding memory ·"))
            .count(),
        1,
        "revalidation must not repeat enrichment"
    );
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
async fn task_review_survives_only_acknowledged_metadata_outside_its_scope() {
    struct LearningUi {
        mutation: Option<std::path::PathBuf>,
        during_recheck: bool,
        checks: usize,
        statuses: Vec<String>,
    }
    impl LearningUi {
        fn mutate(&mut self) {
            if let Some(path) = self.mutation.take() {
                std::fs::write(path, "changed outside the approved task diff\n").unwrap();
            }
        }
    }
    impl Ui for LearningUi {
        fn assistant_text(&mut self, _: &str) {}
        fn assistant_reasoning(&mut self, _: &str) {}
        fn assistant_end(&mut self) {}
        fn tool_call(&mut self, _: &str, _: &str) {}
        fn tool_result(&mut self, _: &str, _: &str) {}
        fn turn_end(&mut self, _: &str) {}
        fn status(&mut self, status: &str) {
            if !self.during_recheck && status.starts_with("coding memory ·") {
                self.mutate();
            }
            self.statuses.push(status.to_owned());
        }
        fn semantic_event(&mut self, event: hi_events::RunEvent) {
            if event.kind == hi_events::EventKind::VerificationStarted {
                self.checks += 1;
                if self.during_recheck && self.checks == 2 {
                    self.mutate();
                }
            }
        }
    }
    for (scenario, mutation, during_recheck, references_memory) in [
        ("owned-only", None, false, false),
        ("reviewed-source", Some("reviewed.txt"), false, false),
        ("new-source", Some("external.txt"), false, false),
        ("requested-memory", None, false, true),
        ("source-before-recheck", Some("reviewed.txt"), true, false),
    ] {
        let workspace = IsolatedWorkspace::new(scenario);
        let mut cfg = workspace.config();
        // Recheck is skipped for prose-only memory unless the stage names that
        // file. This scenario injects a source mutation at the second
        // VerificationStarted, so the stage must still require revalidation.
        let verify_command = if scenario == "source-before-recheck" {
            "true # .hi/memory.md"
        } else {
            "true"
        };
        cfg.gates.verification =
            VerificationMode::Explicit(vec![VerifyStage::new("test", verify_command)]);
        cfg.gates.review = ReviewPolicy::Always;
        cfg.memory.curate_skills = scenario == "owned-only";
        let mut responses = vec![
            write_content_completion("reviewed.txt", "reviewed\n"),
            bash_completion("python3 -c 'assert 2 + 2 == 4'"),
            completion(
                vec![Content::Text(
                    "Created reviewed.txt with the requested content and ran python3 -c 'assert 2 + 2 == 4'."
                        .into(),
                )],
                1,
                1,
            ),
            completion(vec![Content::Text("APPROVE".into())], 1, 1),
        ];
        if cfg.memory.curate_skills {
            responses.push(completion(vec![Content::Text(
                "---\nname: Preserve Task Review\ndescription: Keep review bound to unchanged task inputs.\nscope: project\n---\n# Preserve Task Review\n\nRecheck metadata separately.".into()
            )], 1, 1));
        }
        let mut subject = agent(responses, cfg);
        let mut ui = LearningUi {
            mutation: mutation.map(|path| workspace.path(path)),
            during_recheck,
            checks: 0,
            statuses: Vec::new(),
        };
        let prompt = if references_memory {
            "create reviewed.txt and consult .hi/memory.md"
        } else {
            "create the reviewed file"
        };
        let outcome = subject
            .run_turn(prompt, &mut ui)
            .await
            .unwrap_or_else(|error| panic!("{scenario}: {error:#}; {:?}", ui.statuses));
        assert_eq!(
            outcome.status,
            TurnStatus::Completed,
            "{scenario}: {:?}",
            ui.statuses
        );
        assert_eq!(
            outcome.verification,
            VerificationStatus::Passed,
            "{scenario}"
        );
        assert_eq!(
            outcome.review,
            if scenario == "owned-only" {
                ReviewStatus::Passed
            } else {
                ReviewStatus::Unavailable
            },
            "{scenario}: {:?}",
            ui.statuses
        );
        let expected_verify_rounds = if scenario == "owned-only" { 1 } else { 2 };
        assert_eq!(
            subject.last_turn_telemetry().verify_rounds,
            expected_verify_rounds,
            "{scenario}"
        );
        assert_eq!(
            subject.last_turn_telemetry().model_requests,
            3,
            "{scenario}"
        );
        assert_eq!(
            outcome.verified_workspace_revision,
            Some(subject.runtime.ledger().workspace_revision()),
            "{scenario}"
        );
        assert!(workspace.path(".hi/memory.md").exists(), "{scenario}");
        if scenario == "owned-only" {
            assert_eq!(subject.subagents.auto_skills_written, 1);
        }
        assert!(
            ui.mutation.is_none(),
            "mutation callback did not run: {scenario}"
        );
    }
}
