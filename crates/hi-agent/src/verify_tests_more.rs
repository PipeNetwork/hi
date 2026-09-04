use super::*;

use super::verify_test_support::{NullUi, checkpoint, roots};
use crate::snapshot::workspace_snapshot;

#[test]
fn root_cargo_changes_do_not_duplicate_the_root_pipeline() {
    let (base, root, _) = roots("root-cargo");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"single\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let configured = vec![VerifyStage::new("test", "cargo test --quiet")];
    assert_eq!(
        effective_stages(&root, &["src/lib.rs".into()], &configured, true),
        configured
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn affected_javascript_package_precedes_root_pipeline_and_deduplicates_changes() {
    let (base, root, _) = roots("affected-javascript");
    std::fs::write(
        root.join("package.json"),
        r#"{"scripts":{"test":"root-test"}}"#,
    )
    .unwrap();
    let package = root.join("apps/web");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(
        package.join("package.json"),
        r#"{"scripts":{"typecheck":"tsc --noEmit","test":"vitest"}}"#,
    )
    .unwrap();
    std::fs::write(package.join("tsconfig.json"), "{}\n").unwrap();

    let stages = effective_stages(
        &root,
        &[
            "apps/web/src/index.ts".into(),
            "apps/web/src/other.ts".into(),
        ],
        &[
            VerifyStage::new("typecheck", "npx --no-install tsc --noEmit"),
            VerifyStage::new("test", "npm test --silent"),
        ],
        true,
    );

    assert_eq!(
        stages
            .iter()
            .map(|stage| (stage.name.as_str(), stage.command.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (
                "affected-typecheck:apps/web",
                "npm --prefix 'apps/web' run typecheck --silent",
            ),
            (
                "affected-test:apps/web",
                "npm --prefix 'apps/web' test --silent",
            ),
            ("typecheck", "npx --no-install tsc --noEmit"),
            ("test", "npm test --silent"),
        ]
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn affected_go_modules_are_sorted_before_the_root_pipeline() {
    let (base, root, _) = roots("affected-go");
    std::fs::write(root.join("go.mod"), "module example.test/root\n").unwrap();
    for module in ["services/zeta", "services/alpha"] {
        let module_root = root.join(module);
        std::fs::create_dir_all(module_root.join("pkg")).unwrap();
        std::fs::write(
            module_root.join("go.mod"),
            format!("module example.test/{module}\n"),
        )
        .unwrap();
    }

    let stages = effective_stages(
        &root,
        &[
            "services/zeta/pkg/z.go".into(),
            "services/alpha/pkg/a.go".into(),
        ],
        &[VerifyStage::new("test", "go test ./...")],
        true,
    );

    assert_eq!(
        stages
            .iter()
            .map(|stage| stage.command.as_str())
            .collect::<Vec<_>>(),
        vec![
            // Package-local `go build` is dropped when `go test` for the
            // same module will run — test already compiles.
            "go -C 'services/alpha' test ./...",
            "go -C 'services/zeta' test ./...",
            "go test ./...",
        ]
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn affected_python_package_uses_pyproject_tools_before_root_pipeline() {
    let (base, root, _) = roots("affected-python");
    std::fs::write(root.join("pyproject.toml"), "[project]\nname='root'\n").unwrap();
    let package = root.join("packages/service");
    std::fs::create_dir_all(package.join("service")).unwrap();
    std::fs::write(
        package.join("pyproject.toml"),
        "[project]\nname='service'\n[tool.ruff]\nline-length=100\n",
    )
    .unwrap();
    std::fs::write(package.join("service").join("test_api.py"), "\n").unwrap();

    let stages = effective_stages(
        &root,
        &["packages/service/service/api.py".into()],
        &[
            VerifyStage::new("lint", "ruff check ."),
            VerifyStage::new("test", "pytest -q"),
        ],
        true,
    );

    assert_eq!(
        stages
            .iter()
            .map(|stage| (stage.name.as_str(), stage.command.as_str()))
            .collect::<Vec<_>>(),
        vec![
            (
                "affected-lint:packages/service",
                "ruff check 'packages/service'",
            ),
            (
                "affected-test:packages/service",
                "pytest -q 'packages/service'",
            ),
            ("lint", "ruff check ."),
            ("test", "pytest -q"),
        ]
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn python_setup_and_pytest_markers_define_nested_package_roots() {
    let (base, root, _) = roots("python-markers");
    for (package, marker) in [
        ("packages/legacy", "setup.py"),
        ("packages/tests-only", "pytest.ini"),
    ] {
        let package_root = root.join(package);
        std::fs::create_dir_all(package_root.join("src")).unwrap();
        std::fs::write(package_root.join(marker), "\n").unwrap();
        std::fs::write(package_root.join("src").join("test_module.py"), "\n").unwrap();
    }

    let stages = effective_stages(
        &root,
        &[
            "packages/tests-only/src/test_api.py".into(),
            "packages/legacy/src/module.py".into(),
        ],
        &[],
        true,
    );

    assert_eq!(
        stages
            .iter()
            .map(|stage| stage.command.as_str())
            .collect::<Vec<_>>(),
        vec![
            "pytest -q 'packages/legacy'",
            "pytest -q 'packages/tests-only'",
        ]
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn affected_python_package_without_tests_omits_pytest_stage() {
    let (base, root, _) = roots("python-no-tests");
    std::fs::write(root.join("pyproject.toml"), "[project]\nname='root'\n").unwrap();
    let package = root.join("packages/adapter");
    std::fs::create_dir_all(package.join("src/hi_terminal_bench")).unwrap();
    // pyproject.toml with a build backend but NO test files — mirrors
    // bench/terminal-bench, which must not generate a pytest stage.
    std::fs::write(
        package.join("pyproject.toml"),
        "[project]\nname='adapter'\n[build-system]\nrequires=['hatchling']\n",
    )
    .unwrap();
    std::fs::write(
        package.join("src/hi_terminal_bench/__init__.py"),
        "\"\"\"adapter package\"\"\"\n",
    )
    .unwrap();
    std::fs::write(package.join("src/hi_terminal_bench/agent.py"), "x = 1\n").unwrap();

    let stages = effective_stages(
        &root,
        &["packages/adapter/src/hi_terminal_bench/agent.py".into()],
        &[VerifyStage::new("test", "pytest -q")],
        true,
    );

    // No affected-test stage for the testless package; the root pipeline
    // still runs.
    assert!(
        !stages
            .iter()
            .any(|stage| stage.name.starts_with("affected-test:")),
        "testless Python package should not emit an affected-test stage: {:?}",
        stages.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    assert!(
        stages
            .iter()
            .any(|stage| stage.name == "test" && stage.command == "pytest -q")
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn root_javascript_go_and_python_changes_do_not_duplicate_root_stages() {
    let (base, root, _) = roots("root-polyglot");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("package.json"), "{}\n").unwrap();
    std::fs::write(root.join("go.mod"), "module example.test/root\n").unwrap();
    std::fs::write(root.join("pyproject.toml"), "[project]\nname='root'\n").unwrap();
    let configured = vec![VerifyStage::new("root", "./root-verify")];

    assert_eq!(
        effective_stages(
            &root,
            &[
                "src/index.ts".into(),
                "src/main.go".into(),
                "src/main.py".into(),
            ],
            &configured,
            true,
        ),
        configured
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn root_cargo_package_changes_check_member_consumers() {
    let (base, root, _) = roots("root-cargo-dependent");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("crates/consumer/src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn api() {}\n").unwrap();
    std::fs::write(
        root.join("crates/consumer/src/lib.rs"),
        "pub fn use_api() {}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname='root-app'\nversion='0.1.0'\n\n[workspace]\nmembers=['crates/consumer']\n",
    )
    .unwrap();
    std::fs::write(
        root.join("crates/consumer/Cargo.toml"),
        "[package]\nname='consumer'\nversion='0.1.0'\n[dependencies]\nroot-app={path='../..'}\n",
    )
    .unwrap();

    let stages = affected_cargo_stages(&root, &["src/lib.rs".into()]);
    assert!(
        stages
            .iter()
            .any(|stage| stage.name == "affected-dependent-check:crates/consumer"),
        "root-package consumers should be compile-checked: {stages:?}"
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn automatic_mode_finds_nested_packages_without_a_root_manifest() {
    let (base, root, _) = roots("nested-only");
    let package = root.join("nested/app");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(package.join("package.json"), "{}\n").unwrap();

    let stages = effective_stages(&root, &["nested/app/src/index.js".into()], &[], true);

    assert_eq!(
        stages,
        vec![VerifyStage::new(
            "affected-test:nested/app",
            "npm --prefix 'nested/app' test --silent",
        )]
    );
    assert!(RepairVerifier::automatic(Vec::new(), 1).is_on());
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn explicit_pipeline_is_exact_even_for_nested_package_changes() {
    let (base, root, _) = roots("explicit-exact");
    let package = root.join("apps/web");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::write(package.join("package.json"), "{}\n").unwrap();
    let explicit = vec![VerifyStage::new("custom", "./verify-exactly")];

    assert_eq!(
        effective_stages(&root, &["apps/web/src/index.ts".into()], &explicit, false,),
        explicit
    );
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn explicit_documentation_pipeline_is_not_skipped_as_prose_only() {
    let (base, root, state) = roots("explicit-docs");
    std::fs::write(root.join("README.md"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("README.md"), "after\n").unwrap();
    let mut verifier = RepairVerifier::new(vec![VerifyStage::new("docs", "false")], 1);
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;

    assert!(matches!(outcome, VerifyOutcome::Failed { .. }));
    assert_eq!(verifier.executions().len(), 1);
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn failed_stage_output_carries_digest_and_convergence_note() {
    let (base, root, state) = roots("digest-convergence");
    std::fs::write(root.join("main.rs"), "fn main() {}\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("main.rs"), "fn main() { broken }\n").unwrap();
    std::fs::write(
        root.join("diag.txt"),
        "error[E0425]: cannot find value `broken` in this scope\n  --> main.rs:1:13\n",
    )
    .unwrap();
    let stage = VerifyStage::new("check", "cat diag.txt >&2; false");
    let mut verifier = RepairVerifier::new(vec![stage], 3);
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    let workspace = VerifyWorkspace::new(&root, &state, None, &lsp);

    let first = verifier
        .check(&workspace, &turn_snapshot, &mut cache, None, &mut ui)
        .await;
    let VerifyOutcome::Failed { output, .. } = first else {
        panic!("expected failure, got {first:?}");
    };
    assert!(output.contains("failure digest"), "{output}");
    assert!(output.contains("error[E0425]"), "{output}");
    assert!(
        output.contains("source (main.rs:"),
        "digest should inline the span: {output}"
    );
    assert!(
        !output.contains("No progress"),
        "first round has no history: {output}"
    );

    let second = verifier
        .check(&workspace, &turn_snapshot, &mut cache, None, &mut ui)
        .await;
    let VerifyOutcome::Failed { output, .. } = second else {
        panic!("expected failure, got {second:?}");
    };
    assert!(
        output.contains("No progress since the previous repair attempt"),
        "identical failure set should be called out: {output}"
    );
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn applied_net_zero_mutation_still_runs_explicit_verification() {
    let (base, root, state) = roots("net-zero-mutation");
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    let mut verifier = RepairVerifier::new(vec![VerifyStage::new("test", "false")], 1);
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp)
                .with_changed_files(&[])
                .with_mutation_seen(true),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;

    assert!(matches!(outcome, VerifyOutcome::Failed { .. }));
    assert_eq!(verifier.executions().len(), 1);
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn gitignored_inputs_still_trigger_verification() {
    let (base, root, state) = roots("ignored-input");
    std::fs::write(root.join(".gitignore"), ".env\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join(".env"), "MODE=test\n").unwrap();
    let mut verifier = RepairVerifier::new(vec![VerifyStage::new("test", "test -f .env")], 1);
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;

    assert!(matches!(outcome, VerifyOutcome::Passed));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn final_failure_is_classified_against_internal_pre_turn_checkpoint() {
    let (base, root, state) = roots("preexisting");
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    let checkpoint = checkpoint(&root, &state).await;
    std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();

    let mut verifier = RepairVerifier::new(
        vec![VerifyStage::new(
            "test",
            "printf 'baseline failure\\n' >&2; exit 7",
        )],
        1,
    );
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    let VerifyOutcome::Failed { output, round, .. } = outcome else {
        panic!("expected classified failure");
    };
    assert_eq!(round, 1);
    assert!(output.contains("already failed this verification stage before the turn"));
    assert!(output.contains("Baseline output:\nbaseline failure"));
    assert_eq!(
        std::fs::read_to_string(root.join("source.rs")).unwrap(),
        "current changed contents\n",
        "baseline attribution must never restore over the destination"
    );
    assert!(!state.join("verification-sandboxes").exists());
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn final_failure_absent_from_pre_turn_checkpoint_is_identified() {
    let (base, root, state) = roots("introduced");
    std::fs::write(root.join("state.toml"), "ok\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    let checkpoint = checkpoint(&root, &state).await;
    std::fs::write(root.join("state.toml"), "broken now\n").unwrap();

    let mut verifier = RepairVerifier::new(
        vec![VerifyStage::new("test", "test \"$(cat state.toml)\" = ok")],
        1,
    );
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    let VerifyOutcome::Failed { output, .. } = outcome else {
        panic!("expected classified failure");
    };
    assert!(output.contains("current failure was not present at the turn baseline"));
    assert_eq!(
        std::fs::read_to_string(root.join("state.toml")).unwrap(),
        "broken now\n"
    );
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn baseline_attribution_runs_only_after_last_allowed_check() {
    let (base, root, state) = roots("final-only");
    std::fs::write(root.join("state.toml"), "ok\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    let checkpoint = checkpoint(&root, &state).await;
    std::fs::write(root.join("state.toml"), "broken now\n").unwrap();

    let mut verifier = RepairVerifier::new(
        vec![VerifyStage::new("test", "test \"$(cat state.toml)\" = ok")],
        2,
    );
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    let first = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    let VerifyOutcome::Failed { output, round, .. } = first else {
        panic!("expected first failure");
    };
    assert_eq!(round, 1);
    assert!(!output.contains("Pre-turn attribution"));

    let second = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    let VerifyOutcome::Failed { output, round, .. } = second else {
        panic!("expected final failure");
    };
    assert_eq!(round, 2);
    assert!(output.contains("Pre-turn attribution"));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn repair_budgets_zero_one_two_run_one_two_three_checks() {
    for repairs in 0..=2 {
        let (base, root, state) = roots(&format!("budget-{repairs}"));
        let counter = base.join("runs");
        std::fs::write(root.join("source.rs"), "before\n").unwrap();
        let turn_snapshot = workspace_snapshot(&root).await.unwrap();
        std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();
        let command = format!("printf x >> {}; exit 1", counter.display());
        let mut verifier =
            RepairVerifier::new(vec![VerifyStage::new("test", command)], repairs + 1);
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        for expected_round in 1..=(repairs + 1) {
            let outcome = verifier
                .check(
                    &VerifyWorkspace::new(&root, &state, None, &lsp),
                    &turn_snapshot,
                    &mut cache,
                    None,
                    &mut ui,
                )
                .await;
            assert!(matches!(
                outcome,
                VerifyOutcome::Failed { round, .. } if round == expected_round
            ));
        }
        assert!(matches!(
            verifier
                .check(
                    &VerifyWorkspace::new(&root, &state, None, &lsp),
                    &turn_snapshot,
                    &mut cache,
                    None,
                    &mut ui,
                )
                .await,
            VerifyOutcome::NotRun
        ));
        assert_eq!(
            std::fs::read(&counter).unwrap().len(),
            (repairs + 1) as usize
        );
        assert_eq!(verifier.executions().len(), (repairs + 1) as usize);
        for (index, execution) in verifier.executions().iter().enumerate() {
            assert_eq!(execution.round, index as u32 + 1);
            assert_eq!(execution.name, "test");
            assert_eq!(execution.status, hi_tools::ToolStatus::Failed);
            assert_eq!(
                execution
                    .process
                    .as_ref()
                    .and_then(|process| process.exit_code),
                Some(1)
            );
            assert_eq!(
                execution.truncation,
                Some(hi_tools::TruncationState::Complete)
            );
        }
        let _ = std::fs::remove_dir_all(base);
    }
}

#[tokio::test]
async fn late_mutation_requires_a_fresh_current_revision_pass() {
    let (base, root, state) = roots("late-mutation");
    std::fs::write(root.join("state.txt"), "ok\n").unwrap();
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    let checkpoint = checkpoint(&root, &state).await;
    std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();

    let mut verifier = RepairVerifier::new(
        vec![VerifyStage::new("test", "test \"$(cat state.txt)\" = ok")],
        1,
    );
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    assert!(matches!(
        verifier
            .check(
                &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
                &turn_snapshot,
                &mut cache,
                None,
                &mut ui,
            )
            .await,
        VerifyOutcome::Passed
    ));

    std::fs::write(root.join("state.txt"), "late mutation broke it\n").unwrap();
    cache.invalidate();
    verifier.allow_review_revalidation();
    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(
        matches!(outcome, VerifyOutcome::Failed { round: 2, .. }),
        "expected the late mutation to fail a fresh verification round, got {outcome:?}"
    );
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn broken_attribution_checkpoint_is_infrastructure_error() {
    let (base, root, state) = roots("infra");
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();
    let mut verifier = RepairVerifier::new(vec![VerifyStage::new("test", "exit 1")], 1);
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    let outcome = verifier
        .check(
            &VerifyWorkspace::new(
                &root,
                &state,
                Some("internal:v1:not-this-workspace:missing"),
                &lsp,
            ),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(matches!(
        outcome,
        VerifyOutcome::InfrastructureError { round: 1, .. }
    ));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn repeatedly_mutating_verification_stage_is_unstable_not_a_pass() {
    let (base, root, state) = roots("unstable");
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("source.rs"), "current changed contents\n").unwrap();
    let mut verifier = RepairVerifier::new(
        vec![VerifyStage::new(
            "formatter",
            "printf mutation >> source.rs; exit 0",
        )],
        2,
    );
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    let first = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(matches!(
        first,
        VerifyOutcome::Failed {
            round: 1,
            ref output,
            ..
        } if output.contains("modified relevant source files")
    ));
    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(matches!(
        outcome,
        VerifyOutcome::Unstable {
            round: 2,
            ref changed_files,
            ..
        } if changed_files == &["source.rs"]
    ));
    let _ = std::fs::remove_dir_all(base);
}
