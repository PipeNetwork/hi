use super::*;

use super::verify_test_support::{NullUi, roots};
use crate::snapshot::workspace_snapshot;

#[derive(Default)]
struct RecordingWorkspaceDurability {
    records: std::sync::Mutex<Vec<crate::WorkspaceTranscriptExecution>>,
}

#[async_trait::async_trait]
impl crate::WorkspaceDurability for RecordingWorkspaceDurability {
    async fn mutation_started(&self, _dirty_paths: Option<Vec<String>>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> anyhow::Result<()> {
        self.records
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(record.clone());
        Ok(())
    }
}

#[test]
fn verification_execution_diagnostics_are_bounded_but_test_evidence_is_sticky() {
    let mut verifier = WorkspaceRepairVerifier::new(Vec::new(), u32::MAX);
    for index in 0..300_u32 {
        verifier.record_execution(VerificationExecution {
            round: index + 1,
            name: if index == 100 {
                "unit-test".into()
            } else {
                "check".into()
            },
            command: if index == 100 {
                "cargo test".into()
            } else {
                "cargo check".into()
            },
            status: hi_tools::ToolStatus::Succeeded,
            process: None,
            truncation: None,
        });
    }

    assert_eq!(verifier.executions().len(), VERIFICATION_EXECUTION_LIMIT);
    assert_eq!(verifier.executions_dropped(), 44);
    assert_eq!(verifier.execution_count(), 300);
    assert!(
        verifier.successful_test_stage(),
        "the successful test was compacted from the middle but must remain correctness evidence"
    );
    assert_eq!(verifier.executions().first().unwrap().round, 1);
    assert_eq!(verifier.executions().last().unwrap().round, 300);

    let mut telemetry = crate::TurnTelemetry::default();
    telemetry.replace_verification_diagnostics(
        verifier.executions(),
        verifier.executions_dropped(),
        verifier.execution_count(),
        verifier.successful_test_stage(),
    );
    assert_eq!(
        telemetry.verification_executions,
        verifier.executions(),
        "immediate post-check telemetry retains the bounded execution trail"
    );
    assert_eq!(
        telemetry
            .diagnostic_retention
            .verification_executions_dropped,
        44
    );
    assert_eq!(
        telemetry.diagnostic_retention.verification_executions_total,
        300
    );
    assert!(telemetry.diagnostic_retention.successful_test_verification);
}

#[tokio::test]
async fn saturated_unlimited_round_continues_without_final_attribution() {
    let (base, root, state) = roots("unlimited-saturated-round");
    std::fs::write(root.join("state.txt"), "changed\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    let changed = vec!["state.txt".to_string()];
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut verifier = WorkspaceRepairVerifier::new(
        vec![VerifyStage::new("test", "exit 7")],
        crate::UNLIMITED_REPAIR_CYCLES,
    );
    verifier.round = u32::MAX;
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp).with_changed_files(&changed),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;

    match outcome {
        VerifyOutcome::Failed { output, round, .. } => {
            assert_eq!(round, u32::MAX);
            assert!(
                !output.contains("Pre-turn attribution"),
                "the unlimited sentinel must not be treated as the final repair round"
            );
        }
        other => panic!("expected a productive verification attempt, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn sandbox_denial_is_not_treated_as_a_preexisting_code_failure() {
    let execution = hi_tools::ProcessExecution {
        status: hi_tools::ToolStatus::Failed,
        outcome: hi_tools::ProcessOutcome {
            exit_code: Some(1),
            stdout_summary: String::new(),
            stderr_summary: "cargo: Operation not permitted while creating target".into(),
            duration_ms: 1,
        },
        truncation: hi_tools::TruncationState::Complete,
    };

    assert!(baseline_failure_is_infrastructure(&execution));
}

#[test]
fn verifier_is_off_when_no_stages() {
    let v = RepairVerifier::new(Vec::new(), 2);
    assert!(!v.is_on());
    assert_eq!(v.round(), 0);
}

#[test]
fn verifier_is_on_with_stages() {
    let v = RepairVerifier::new(vec![VerifyStage::new("check", "true")], 2);
    assert!(v.is_on());
}

#[test]
fn native_verifier_policy_only_bypasses_admission_for_proven_reads() {
    assert!(native_verifier_intent(&VerifyStage::new("inspect", "rg TODO src")).is_none());

    let opaque = native_verifier_intent(&VerifyStage::new("test", "cargo test"))
        .expect("cache-writing verifier must require admission");
    assert_eq!(opaque.effect_scope, hi_workspace::EffectScope::LiveWriter);
    assert_eq!(
        opaque.replay_class,
        hi_workspace::ReplayClass::NonReplayableExternal
    );

    let dynamic = native_verifier_intent(&VerifyStage::new("dynamic", "eval \"$VERIFY\""))
        .expect("dynamic verifier must fail closed");
    assert_eq!(dynamic.effect_scope, hi_workspace::EffectScope::LiveWriter);
    assert_eq!(
        dynamic.replay_class,
        hi_workspace::ReplayClass::NonReplayableExternal
    );
}

#[test]
fn native_verifier_report_preserves_terminal_process_disposition() {
    let stage = VerifyStage::new("test", "cargo test");
    let intent = native_verifier_intent(&stage).unwrap();
    let execution = |status| hi_tools::ProcessExecution {
        status,
        outcome: hi_tools::ProcessOutcome {
            exit_code: None,
            stdout_summary: String::new(),
            stderr_summary: String::new(),
            duration_ms: 1,
        },
        truncation: hi_tools::TruncationState::Complete,
    };

    for (status, disposition) in [
        (
            hi_tools::ToolStatus::Succeeded,
            hi_workspace::ExecutionDisposition::Succeeded,
        ),
        (
            hi_tools::ToolStatus::Failed,
            hi_workspace::ExecutionDisposition::Failed,
        ),
        (
            hi_tools::ToolStatus::Cancelled,
            hi_workspace::ExecutionDisposition::Cancelled,
        ),
        (
            hi_tools::ToolStatus::TimedOut,
            hi_workspace::ExecutionDisposition::Cancelled,
        ),
    ] {
        let execution = execution(status);
        let report =
            native_verifier_execution_report(&stage, &intent, Some(&execution), vec![], None);
        assert_eq!(report.disposition, disposition, "status {status:?}");
        assert!(report.workspace_may_have_changed);
        assert!(report.external_effect_may_have_occurred);
    }

    let indeterminate = native_verifier_execution_report(
        &stage,
        &intent,
        None,
        vec![],
        Some("spawn acknowledgement lost"),
    );
    assert_eq!(
        indeterminate.disposition,
        hi_workspace::ExecutionDisposition::Indeterminate
    );
}

#[tokio::test]
async fn closed_admission_blocks_opaque_verifier_but_allows_proven_read() {
    use hi_workspace::WorkspaceController;

    let (base, root, state) = roots("closed-admission");
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("source.rs"), "after\n").unwrap();
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let controller = std::sync::Arc::new(hi_workspace::InMemoryWorkspaceController::new_local(
        "closed-admission",
        &root,
        &state,
    ));
    coordination.install_controller(controller.clone()).unwrap();
    let _blocking_permit = controller
        .begin(hi_workspace::MutationIntent::workspace("existing writer"))
        .await
        .unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let mut opaque = RepairVerifier::new(
        vec![VerifyStage::new("opaque", "touch should-not-exist")],
        1,
    );
    let outcome = opaque
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp)
                .with_workspace_coordination(coordination.clone(), None),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(matches!(outcome, VerifyOutcome::InfrastructureError { .. }));
    assert!(
        !root.join("should-not-exist").exists(),
        "denied verifier must not reach process spawn"
    );

    let mut read_only = RepairVerifier::new(vec![VerifyStage::new("read", "true")], 1);
    let outcome = read_only
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp)
                .with_workspace_coordination(coordination, None),
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
async fn failed_native_verifier_is_recorded_as_failed_execution_in_journal() {
    let (base, root, state) = roots("failed-journal");
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("source.rs"), "after\n").unwrap();
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let binding_id = coordination.binding().binding_id.to_string();
    let mut verifier = RepairVerifier::new(vec![VerifyStage::new("failing", "sh -c 'exit 7'")], 1);
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp)
                .with_workspace_coordination(coordination, None),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(matches!(outcome, VerifyOutcome::Failed { .. }));

    let store = hi_control::ControlStore::open_for_state(&state).unwrap();
    let operations = store.operations_for_binding(&binding_id).unwrap();
    assert_eq!(operations.len(), 1);
    let operation = &operations[0];
    assert!(operation.execution_ref.is_some());
    assert!(operation.settlement_ref.is_some());
    assert_eq!(
        operation.replay_class,
        hi_control::OperationReplayClass::NonReplayableExternal
    );
    assert!(
        operation
            .error
            .as_deref()
            .is_some_and(|detail| detail.contains("finished with Failed")),
        "journal must retain failed execution rather than invent success: {operation:?}"
    );
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn pipefs_verifier_stages_exact_command_result_and_failed_report() {
    let (base, root, state) = roots("pipefs-stage");
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("source.rs"), "after\n").unwrap();
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    coordination
        .install_controller(std::sync::Arc::new(
            hi_workspace::InMemoryWorkspaceController::new_pipefs(
                "pipefs-stage",
                "session",
                2,
                true,
                &root,
                &state,
            ),
        ))
        .unwrap();
    let durability = std::sync::Arc::new(RecordingWorkspaceDurability::default());
    let command = "printf 'verifier-sentinel' >&2; exit 7";
    let mut verifier = RepairVerifier::new(vec![VerifyStage::new("failing", command)], 1);
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp)
                .with_workspace_coordination(coordination, Some(durability.clone())),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(matches!(outcome, VerifyOutcome::Failed { .. }));

    let records = durability
        .records
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(
        record.execution.disposition,
        hi_workspace::ExecutionDisposition::Failed
    );
    let hi_ai::Content::ToolCall {
        name, arguments, ..
    } = &record.assistant_content[0]
    else {
        panic!("native verifier evidence must carry a synthetic tool call");
    };
    assert_eq!(name, "native_verify");
    let arguments: serde_json::Value = serde_json::from_str(arguments).unwrap();
    assert_eq!(arguments["command"], command);
    assert_eq!(record.calls[0].name, "native_verify");
    assert!(record.calls[0].result.contains("verifier-sentinel"));
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn pipefs_verifier_without_execution_stager_fails_closed() {
    let (base, root, state) = roots("pipefs-no-stager");
    std::fs::write(root.join("source.rs"), "before\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    std::fs::write(root.join("source.rs"), "after\n").unwrap();
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let controller = std::sync::Arc::new(hi_workspace::InMemoryWorkspaceController::new_pipefs(
        "pipefs-no-stager",
        "session",
        2,
        true,
        &root,
        &state,
    ));
    coordination.install_controller(controller.clone()).unwrap();
    let mut verifier =
        RepairVerifier::new(vec![VerifyStage::new("writes", "touch verifier-output")], 1);
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;

    let outcome = verifier
        .check(
            &VerifyWorkspace::new(&root, &state, None, &lsp)
                .with_workspace_coordination(coordination, None),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        )
        .await;
    assert!(matches!(outcome, VerifyOutcome::InfrastructureError { .. }));
    assert_eq!(
        hi_workspace::WorkspaceController::status(controller.as_ref()).state,
        hi_workspace::WorkspaceState::RecoveryRequired,
        "executed PipeFS verification without transcript evidence must retain recovery"
    );
    assert!(root.join("verifier-output").exists());
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn lsp_execution_evidence_does_not_invent_process_data() {
    let record = VerificationExecution::lsp(3, hi_tools::ToolStatus::Failed);
    let value = serde_json::to_value(&record).unwrap();

    assert_eq!(value["round"], 3);
    assert_eq!(value["name"], "lsp");
    assert_eq!(value["command"], "diagnostics");
    assert_eq!(value["status"], "failed");
    assert!(value.get("process").is_none());
    assert!(value.get("truncation").is_none());
}

#[test]
fn shell_execution_evidence_strips_terminal_control_sequences() {
    let execution = hi_tools::ProcessExecution {
        status: hi_tools::ToolStatus::Succeeded,
        outcome: hi_tools::ProcessOutcome {
            exit_code: Some(0),
            stdout_summary: "\u{1b}[32mok\u{1b}[0m".into(),
            stderr_summary: String::new(),
            duration_ms: 4,
        },
        truncation: hi_tools::TruncationState::Complete,
    };
    let record =
        VerificationExecution::shell(1, &VerifyStage::new("check", "cargo check"), &execution);

    assert_eq!(record.process.unwrap().stdout_summary, "ok");
}

#[test]
fn stage_guidance_differs_tests_vs_compile() {
    let test_stage = VerifyStage::new("test", "pytest");
    let compile_stage = VerifyStage::new("check", "cargo check");
    assert_ne!(stage_guidance(&test_stage), stage_guidance(&compile_stage));
    assert!(stage_guidance(&test_stage).contains("required behavior"));
    assert!(stage_guidance(&compile_stage).contains("root cause"));
}

#[test]
fn prose_only_path_detection_is_conservative() {
    assert!(is_prose_only_path("README.md"));
    assert!(is_prose_only_path("docs/guide.rst"));
    assert!(is_prose_only_path("LICENSE"));
    assert!(is_prose_only_path(".hi/memory.md"));
    assert!(is_prose_only_path(".hi/memory.undo.md"));
    assert!(!is_prose_only_path("package.json"));
    assert!(!is_prose_only_path("docs/example.ts"));
    assert!(!is_prose_only_path(".github/workflows/test.yml"));
}

#[test]
fn only_known_runtime_artifacts_are_hidden_from_project_changes() {
    assert!(is_internal_runtime_artifact_path(".hi/history"));
    assert!(is_internal_runtime_artifact_path("./.hi/memory.undo.md"));
    assert!(is_internal_runtime_artifact_path(".hi\\history"));
    assert!(!is_internal_runtime_artifact_path(".hi/memory.md"));
    assert!(!is_internal_runtime_artifact_path(".hi/config.toml"));
    assert!(!is_internal_runtime_artifact_path("src/main.rs"));
}

#[test]
fn verification_mutation_filter_keeps_source_and_ignores_generated_noise() {
    assert!(verification_relevant_path("src/lib.rs"));
    assert!(verification_relevant_path("Cargo.lock"));
    assert!(!verification_relevant_path("pkg/__pycache__/mod.pyc"));
    assert!(!verification_relevant_path("README.md"));
    assert!(!verification_relevant_path(".hi/state.json"));
    // Caches the verification stages write themselves must never read as
    // workspace mutation — that path reports stable stages as unstable.
    assert!(!verification_relevant_path(
        "vectorops/.pytest_cache/v/cache/nodeids"
    ));
    assert!(!verification_relevant_path(
        "vectorops/.pytest_cache/CACHEDIR.TAG"
    ));
    assert!(!verification_relevant_path(
        "vectorops/vectorops.egg-info/PKG-INFO"
    ));
    assert!(!verification_relevant_path("app/node_modules/pkg/index.js"));
    assert!(!verification_relevant_path(
        ".cargo-home/registry/src/dep/Cargo.lock"
    ));
    assert!(!verification_relevant_path(".mypy_cache/3.12/foo.json"));
    // A source file that merely lives near a cache stays relevant.
    assert!(verification_relevant_path("src/pytest_cache_helper.py"));
}

#[test]
fn dependent_crates_get_compile_check_stages() {
    let root = std::env::temp_dir().join(format!("hi-verify-deps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for member in ["core", "consumer"] {
        std::fs::create_dir_all(root.join(format!("crates/{member}/src"))).unwrap();
        std::fs::write(root.join(format!("crates/{member}/src/lib.rs")), "").unwrap();
    }
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/*\"]\n",
    )
    .unwrap();
    std::fs::write(
        root.join("crates/core/Cargo.toml"),
        "[package]\nname = \"core\"\nversion = \"0.0.1\"\n",
    )
    .unwrap();
    std::fs::write(
        root.join("crates/consumer/Cargo.toml"),
        "[package]\nname = \"consumer\"\nversion = \"0.0.1\"\n[dependencies]\ncore = { path = \"../core\" }\n",
    )
    .unwrap();

    let changed = vec!["crates/core/src/lib.rs".to_string()];
    let stages = affected_cargo_stages(&root, &changed);
    let names: Vec<&str> = stages.iter().map(|stage| stage.name.as_str()).collect();
    assert!(names.contains(&"affected-test:crates/core"), "{names:?}");
    assert!(
        names.contains(&"affected-dependent-check:crates/consumer"),
        "consumer compile-checks after core changes: {names:?}"
    );
    assert!(
        !names
            .iter()
            .any(|n| n.contains("dependent-check:crates/core")),
        "the changed crate is not its own dependent: {names:?}"
    );
    // Dependent checks run after the changed package's own stages.
    let own = names
        .iter()
        .position(|n| *n == "affected-test:crates/core")
        .unwrap();
    let dependent = names
        .iter()
        .position(|n| *n == "affected-dependent-check:crates/consumer")
        .unwrap();
    assert!(own < dependent);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn skip_affected_stage_matches_sealed_package_labels() {
    let mut checks = std::collections::BTreeSet::new();
    checks.insert("crates/library".into());
    checks.insert("web".into());
    checks.insert("svc".into());
    let tests = std::collections::BTreeSet::new();
    assert!(should_skip_affected_stage(
        &VerifyStage::new(
            "affected-check:crates/library",
            "cargo check --quiet --manifest-path 'crates/library/Cargo.toml'",
        ),
        &checks,
        &tests,
    ));
    // Phase O: polyglot check seals cover typecheck/build/lint stages.
    assert!(should_skip_affected_stage(
        &VerifyStage::new(
            "affected-typecheck:web",
            "npm --prefix 'web' exec -- tsc --noEmit",
        ),
        &checks,
        &tests,
    ));
    assert!(should_skip_affected_stage(
        &VerifyStage::new("affected-build:svc", "go -C 'svc' build ./..."),
        &checks,
        &tests,
    ));
    assert!(should_skip_affected_stage(
        &VerifyStage::new("affected-lint:pkg", "ruff check 'pkg'"),
        &{
            let mut c = checks.clone();
            c.insert("pkg".into());
            c
        },
        &tests,
    ));
    assert!(!should_skip_affected_stage(
        &VerifyStage::new(
            "affected-test:crates/library",
            "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'",
        ),
        &checks,
        &tests,
    ));
    // Root pipeline is never skipped via this path.
    assert!(!should_skip_affected_stage(
        &VerifyStage::new("check", "cargo check --quiet"),
        &checks,
        &tests,
    ));
    let mut test_set = std::collections::BTreeSet::new();
    test_set.insert("crates/library".into());
    assert!(should_skip_affected_stage(
        &VerifyStage::new(
            "affected-test:crates/library",
            "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'",
        ),
        &checks,
        &test_set,
    ));
}

#[test]
fn affected_cargo_packages_precede_the_root_pipeline() {
    let (base, root, _) = roots("affected-cargo");
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/*\"]\ndefault-members = [\"crates/app\"]\n",
    )
    .unwrap();
    for package in ["app", "library"] {
        let package_root = root.join("crates").join(package);
        std::fs::create_dir_all(package_root.join("src")).unwrap();
        std::fs::write(
            package_root.join("Cargo.toml"),
            format!("[package]\nname = \"{package}\"\nversion = \"0.1.0\"\n"),
        )
        .unwrap();
    }
    let stages = effective_stages(
        &root,
        &[
            "crates/library/src/lib.rs".into(),
            "crates/library/src/other.rs".into(),
        ],
        &[
            VerifyStage::new("check", "cargo check --quiet"),
            VerifyStage::new("test", "cargo test --quiet"),
        ],
        true,
    );
    assert_eq!(
        stages
            .iter()
            .map(|stage| (stage.name.as_str(), stage.command.as_str()))
            .collect::<Vec<_>>(),
        vec![(
            "affected-test:crates/library",
            "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'",
        ),]
    );
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn package_local_tests_supersede_the_whole_workspace_test_run() {
    // Measured on a 24-crate workspace: `cargo test` 811s vs `cargo check`
    // 114s, against a 600s stage timeout. Every turn ended unjudged however
    // small the edit, because the gate's cost tracked the project rather
    // than the change.
    let (base, root, _) = roots("supersede-workspace-test");
    std::fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/*\"]\n",
    )
    .unwrap();
    let package_root = root.join("crates").join("library");
    std::fs::create_dir_all(package_root.join("src")).unwrap();
    std::fs::write(
        package_root.join("Cargo.toml"),
        "[package]\nname = \"library\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let configured = vec![
        VerifyStage::new("check", "cargo check --quiet"),
        VerifyStage::new("test", "cargo test --quiet"),
    ];
    let changed = ["crates/library/src/lib.rs".to_string()];

    let auto = effective_stages(&root, &changed, &configured, true);
    assert!(
        !auto.iter().any(|s| s.command == "cargo test --quiet"),
        "the whole-workspace test run must be superseded: {auto:?}"
    );
    assert!(
        !auto.iter().any(|s| s.command == "cargo check --quiet"),
        "the whole-workspace check is also superseded when package tests cover the edit: {auto:?}"
    );
    assert!(
        auto.iter()
            .any(|s| s.name == "affected-test:crates/library"),
        "package-local coverage must actually be present: {auto:?}"
    );

    // Explicit configuration is the user's decision and is run as written —
    // this refinement applies only to the auto-detected pipeline.
    let explicit = effective_stages(&root, &changed, &configured, false);
    assert_eq!(explicit, configured, "explicit stages must be untouched");
    let _ = std::fs::remove_dir_all(base);
}

#[test]
fn only_unscoped_cargo_test_commands_are_superseded() {
    assert!(is_whole_workspace_cargo_test("cargo test"));
    assert!(is_whole_workspace_cargo_test("cargo test --quiet"));
    assert!(is_whole_workspace_cargo_test("  cargo test --workspace  "));
    // Already narrowed by the caller — leave it alone.
    assert!(!is_whole_workspace_cargo_test("cargo test -p library"));
    assert!(!is_whole_workspace_cargo_test(
        "cargo test --package library"
    ));
    assert!(!is_whole_workspace_cargo_test(
        "cargo test --manifest-path 'a/Cargo.toml'"
    ));
    assert!(!is_whole_workspace_cargo_test(
        "cargo test --test integration"
    ));
    // Not a plain `cargo test` at all.
    assert!(!is_whole_workspace_cargo_test("cargo testsuite"));
    assert!(!is_whole_workspace_cargo_test("cargo test && ./extra.sh"));
    assert!(!is_whole_workspace_cargo_test("cargo check --quiet"));
    assert!(!is_whole_workspace_cargo_test("make test"));

    assert!(is_whole_workspace_cargo_check("cargo check"));
    assert!(is_whole_workspace_cargo_check("cargo check --quiet"));
    assert!(!is_whole_workspace_cargo_check("cargo check -p library"));
    assert!(!is_whole_workspace_cargo_check(
        "cargo check --manifest-path 'a/Cargo.toml'"
    ));
    assert!(!is_whole_workspace_cargo_check("cargo test --quiet"));
    assert!(is_package_local_cargo_test(
        "cargo test --quiet --manifest-path 'crates/library/Cargo.toml'"
    ));
    assert!(!is_package_local_cargo_test("cargo test --quiet"));
}
