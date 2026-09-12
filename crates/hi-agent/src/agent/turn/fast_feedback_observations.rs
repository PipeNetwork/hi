//! Translate physical fast checks into the task's canonical recovery evidence.

use hi_tools::infra::CargoCommandOutcome;

use crate::recovery::{ValidationObservation, ValidationResult};
use crate::workspace_runtime::WorkspaceRuntime;

pub(super) async fn input_revision(runtime: &WorkspaceRuntime) -> Option<String> {
    runtime.ensure_ledger_scan_complete_async().await.ok()?;
    runtime.reconcile_ledger_async().await.ok()?;
    Some(runtime.ledger().workspace_revision())
}

pub(super) async fn unchanged(runtime: &WorkspaceRuntime, before: Option<&str>) -> bool {
    let Some(before) = before else { return false };
    input_revision(runtime).await.as_deref() == Some(before)
}

/// Preserve physical exit evidence, including semantic failures such as gofmt.
pub(super) fn process_status(output: &hi_tools::ToolOutcome, stable: bool) -> ValidationResult {
    if !stable {
        return ValidationResult::Deferred;
    }
    let code = output
        .process
        .as_ref()
        .and_then(|process| process.exit_code);
    match (output.status, code) {
        (hi_tools::ToolStatus::Succeeded, Some(0)) => ValidationResult::Passed,
        (hi_tools::ToolStatus::Failed, Some(_))
            if !crate::verify::validation_failure_is_infrastructure(code, &output.content) =>
        {
            ValidationResult::Failed
        }
        (hi_tools::ToolStatus::Denied | hi_tools::ToolStatus::Cancelled, _) => {
            ValidationResult::Deferred
        }
        _ => ValidationResult::Infrastructure,
    }
}

pub(super) fn file_check(
    check: &str,
    path: &str,
    input: &str,
    output: &hi_tools::ToolOutcome,
    stable: bool,
    root: &std::path::Path,
) -> ValidationObservation {
    let status = process_status(output, stable);
    ValidationObservation::command(
        format!("fast-file:{}", uuid::Uuid::new_v4()),
        &format!("{check} [file:{path}]"),
        input.into(),
        status,
        &output.content,
        root,
        false,
    )
}

pub(super) async fn package_outcome(
    runtime: &WorkspaceRuntime,
    observations: &mut Vec<ValidationObservation>,
    outcome: CargoCommandOutcome,
    input: Option<String>,
) -> CargoCommandOutcome {
    let (command, packages, output, status) = match &outcome {
        CargoCommandOutcome::Passed { command, packages } => {
            (*command, packages.clone(), "", ValidationResult::Passed)
        }
        CargoCommandOutcome::Failed {
            command,
            package,
            output,
        } => (
            *command,
            vec![package.clone()],
            output.as_str(),
            if crate::verify::validation_failure_is_infrastructure(None, output) {
                ValidationResult::Infrastructure
            } else {
                ValidationResult::Failed
            },
        ),
        CargoCommandOutcome::TimedOut { command, package } => (
            *command,
            vec![package.clone()],
            "",
            ValidationResult::Infrastructure,
        ),
        CargoCommandOutcome::Skipped | CargoCommandOutcome::Unavailable { .. } => return outcome,
    };
    let stable = unchanged(runtime, input.as_deref()).await;
    let execution = uuid::Uuid::new_v4();
    for package in packages {
        let mut observation = ValidationObservation::command(
            format!("fast-package:{execution}:{package}"),
            &format!("{command} [package:{package}]"),
            input.clone().unwrap_or_default(),
            if stable {
                status
            } else {
                ValidationResult::Deferred
            },
            output,
            runtime.root(),
            false,
        );
        crate::verify::cargo_scope::canonicalize_package(
            &mut observation,
            runtime.root(),
            command,
            &package,
        );
        observations.push(observation);
    }
    if !stable && matches!(status, ValidationResult::Passed | ValidationResult::Failed) {
        CargoCommandOutcome::Unavailable {
            detail:
                "fast check input changed during execution; current revision remains unverified"
                    .into(),
        }
    } else if status == ValidationResult::Infrastructure
        && matches!(outcome, CargoCommandOutcome::Failed { .. })
    {
        CargoCommandOutcome::Unavailable {
            detail: format!("fast check infrastructure unavailable: {output}"),
        }
    } else {
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::turn::helpers::synthetic_tool_outcome;
    use crate::tests::common::{Canned, IsolatedWorkspace};
    use std::sync::{Arc, Mutex};

    #[test]
    fn file_check_preserves_exit_evidence_and_semantic_format_failure() {
        let mut output = synthetic_tool_outcome("src/main.go".into(), hi_tools::ToolStatus::Failed);
        assert_eq!(
            file_check(
                "gofmt -l",
                "src/main.go",
                "rev",
                &output,
                true,
                std::path::Path::new(".")
            )
            .status,
            ValidationResult::Infrastructure
        );
        output.process = Some(hi_tools::ProcessOutcome {
            exit_code: Some(0),
            stdout_summary: "src/main.go".into(),
            stderr_summary: String::new(),
            duration_ms: 0,
        });
        assert_eq!(
            file_check(
                "gofmt -l",
                "src/main.go",
                "rev",
                &output,
                true,
                std::path::Path::new(".")
            )
            .status,
            ValidationResult::Failed
        );
        assert_eq!(
            file_check(
                "gofmt -l",
                "src/main.go",
                "rev",
                &output,
                false,
                std::path::Path::new(".")
            )
            .status,
            ValidationResult::Deferred
        );
    }

    #[tokio::test]
    async fn multi_edit_implementation_survives_intermediate_cargo_failures() {
        use crate::tests::common::{RecordingUi, completion};
        use hi_ai::Content;

        let workspace = IsolatedWorkspace::new("chat-multi-edit-checks");
        let root = workspace.path("");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"multi_edit_fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        let mut source = "pub fn answer() -> u32 { 0 }\n".to_owned();
        std::fs::write(root.join("src/lib.rs"), &source).unwrap();
        let preparation = tokio::process::Command::new("cargo")
            .args(["generate-lockfile", "--offline"])
            .current_dir(&root)
            .output()
            .await
            .unwrap();
        assert!(
            preparation.status.success(),
            "{}",
            String::from_utf8_lossy(&preparation.stderr)
        );
        let mut responses = vec![completion(
            vec![Content::ToolCall {
                id: "read-source".into(),
                name: "read".into(),
                arguments: r#"{"path":"src/lib.rs"}"#.into(),
            }],
            1,
            1,
        )];
        // Five successful edits whose automatic checks fail exceeds the old
        // task recovery allowance. Only the final edit completes the feature.
        for index in 0..6 {
            let next = if index == 5 {
                "pub fn answer() -> u32 { 42 }\n".to_owned()
            } else {
                format!("pub fn answer() -> u32 {{ pending_{index}() }}\n")
            };
            responses.push(completion(vec![Content::ToolCall {
                id: format!("edit-{index}"), name: "edit".into(),
                arguments: serde_json::json!({"path":"src/lib.rs", "old_string":source, "new_string":next}).to_string(),
            }], 1, 1));
            source = next;
        }
        responses.push(completion(
            vec![Content::Text(
                "Implemented answer in src/lib.rs; cargo check passed.".into(),
            )],
            1,
            1,
        ));
        let mut cfg = workspace.config();
        cfg.gates.lsp_mode = crate::LspMode::Off;
        cfg.gates.verification =
            crate::VerificationMode::Explicit(vec![crate::config::VerifyStage::new(
                "check",
                "cargo check --quiet",
            )]);
        let mut agent = crate::Agent::new(Arc::new(Canned(Mutex::new(responses))), cfg).unwrap();
        let mut ui = RecordingUi::default();
        let outcome = agent
            .run_turn("Build the answer function in src/lib.rs", &mut ui)
            .await
            .unwrap();
        assert_eq!(
            outcome.status,
            crate::TurnStatus::Completed,
            "{outcome:?}: {:?}",
            ui.statuses
        );
        assert_eq!(outcome.verification, crate::VerificationStatus::Passed);
        assert!(!agent.task_recovery.exhausted);
        assert_eq!(agent.task_recovery.interventions, 0);
        assert!(
            !ui.statuses
                .iter()
                .any(|s| s.contains("missing validation") || s.contains("without validation")),
            "{:?}",
            ui.statuses
        );
        assert_eq!(
            std::fs::read_to_string(root.join("src/lib.rs")).unwrap(),
            source
        );
        assert!(
            ui.statuses
                .iter()
                .filter(|status| status.contains("cannot find function `pending_"))
                .count()
                >= 5,
            "{:?}",
            ui.statuses
        );
    }

    #[tokio::test]
    async fn actual_final_cargo_check_clears_repaired_fast_check_failure() {
        use crate::config::VerifyStage;
        use crate::verify::{VerifyOutcome, VerifyWorkspace, WorkspaceRepairVerifier};
        use std::collections::BTreeSet;

        let workspace = IsolatedWorkspace::new("native-cargo-scope");
        let root = workspace.path("");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"native_scope_fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        std::fs::write(
            root.join("src/lib.rs"),
            "pub fn answer() -> u32 { \"broken\" }\n",
        )
        .unwrap();
        // Let Cargo write its exact lock format before capturing validation
        // inputs. Rewriting a hand-authored lock correctly defers the check.
        let preparation = tokio::process::Command::new("cargo")
            .args(["generate-lockfile", "--offline"])
            .current_dir(&root)
            .output()
            .await
            .unwrap();
        assert!(
            preparation.status.success(),
            "{}",
            String::from_utf8_lossy(&preparation.stderr)
        );
        let lockfile = std::fs::read(root.join("Cargo.lock")).unwrap();
        let mut cfg = workspace.config();
        cfg.gates.lsp_mode = crate::LspMode::Off;
        let mut agent = crate::Agent::new(Arc::new(Canned(Mutex::new(Vec::new()))), cfg).unwrap();
        let changed = vec!["src/lib.rs".to_owned()];
        agent
            .runtime
            .ensure_ledger_scan_complete_async()
            .await
            .unwrap();
        let input = input_revision(&agent.runtime).await;
        let failed =
            hi_tools::infra::run_affected_cargo_checks(&root, &changed, &mut BTreeSet::new()).await;
        assert!(
            matches!(failed, CargoCommandOutcome::Failed { .. }),
            "{failed:?}"
        );
        assert_eq!(
            std::fs::read(root.join("Cargo.lock")).unwrap(),
            lockfile,
            "scope regression must keep Cargo's validation inputs stable"
        );
        let mut observations = Vec::new();
        package_outcome(&agent.runtime, &mut observations, failed, input).await;
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].status, ValidationResult::Failed);
        agent.task_recovery.observe_feedback(&observations[0]);
        let baseline = crate::snapshot::workspace_snapshot(&root).await.unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn answer() -> u32 { 42 }\n").unwrap();
        let revision = input_revision(&agent.runtime).await.unwrap();
        assert!(
            agent
                .task_recovery
                .unresolved_validation_status(&revision)
                .is_some()
        );
        let mut verifier =
            WorkspaceRepairVerifier::new(vec![VerifyStage::new("check", "cargo check --quiet")], 1);
        verifier.timeout_override = Some(std::time::Duration::from_secs(20));
        let lsp = agent.runtime.lsp();
        let outcome = verifier
            .check(
                &VerifyWorkspace::new(&root, agent.runtime.state_root(), None, &lsp)
                    .with_process_runner(agent.runtime.process_runner())
                    .with_changed_files(&changed)
                    .with_mutation_seen(true),
                &baseline,
                &mut crate::snapshot::SnapshotCache::default(),
                Some(agent.runtime.ledger_arc()),
                &mut crate::tests::common::NullUi,
            )
            .await;
        assert!(
            matches!(outcome, VerifyOutcome::Passed { .. }),
            "{outcome:?}"
        );
        let final_observations = verifier.take_observations();
        assert_eq!(final_observations.len(), 1);
        assert_eq!(final_observations[0].scope, observations[0].scope);
        agent.task_recovery.observe(&final_observations[0]);
        assert!(
            agent
                .task_recovery
                .unresolved_validation_status(&revision)
                .is_none()
        );
    }

    #[tokio::test]
    async fn package_result_cannot_seal_a_revision_that_changed_during_check() {
        let workspace = IsolatedWorkspace::new("fast-feedback-revision");
        let agent = crate::Agent::new(Arc::new(Canned(Mutex::new(Vec::new()))), workspace.config())
            .unwrap();
        let path = agent.runtime.root().join("value.rs");
        std::fs::write(&path, "one").unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        let input = input_revision(&agent.runtime).await;
        std::fs::write(&path, "two").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        let mut observations = Vec::new();
        let result = package_outcome(
            &agent.runtime,
            &mut observations,
            CargoCommandOutcome::Passed {
                command: "cargo test",
                packages: vec!["core".into()],
            },
            input,
        )
        .await;
        assert!(matches!(result, CargoCommandOutcome::Unavailable { .. }));
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].status, ValidationResult::Deferred);
        assert_eq!(observations[0].scope, "cargo test [package:core]");
        assert!(!observations[0].required_stage);
        let input = input_revision(&agent.runtime).await;
        let result = package_outcome(
            &agent.runtime,
            &mut observations,
            CargoCommandOutcome::Failed {
                command: "cargo test",
                package: "core".into(),
                output: "Operation not permitted".into(),
            },
            input,
        )
        .await;
        assert!(matches!(result, CargoCommandOutcome::Unavailable { .. }));
        assert_eq!(
            observations.last().unwrap().status,
            ValidationResult::Infrastructure
        );
    }
}
