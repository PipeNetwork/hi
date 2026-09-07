use super::verify_test_support::{NullUi, roots};
use super::*;
use crate::snapshot::workspace_snapshot;

#[tokio::test]
async fn explicit_current_revision_checks_unchanged_passing_and_failing_bytes() {
    for (bytes, passed) in [("ready\n", true), ("broken\n", false)] {
        let (base, root, state) = roots("current-revision-explicit");
        std::fs::write(root.join("input.rs"), bytes).unwrap();
        let baseline = workspace_snapshot(&root).await.unwrap();
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let mut verifier = WorkspaceRepairVerifier::new(
            vec![VerifyStage::new(
                "input",
                "test \"$(cat input.rs)\" = ready",
            )],
            1,
        );
        let mut cache = SnapshotCache::default();
        let mut ui = NullUi;
        let workspace = VerifyWorkspace::new(&root, &state, None, &lsp).with_changed_files(&[]);
        assert!(matches!(
            verifier
                .check(&workspace, &baseline, &mut cache, None, &mut ui)
                .await,
            VerifyOutcome::SkippedNoChanges { .. }
        ));
        assert_eq!(verifier.execution_count(), 0);

        let workspace = workspace.with_admission(VerificationAdmission::CurrentRevision);
        let outcome = verifier
            .check(&workspace, &baseline, &mut cache, None, &mut ui)
            .await;
        if passed {
            assert!(
                matches!(outcome, VerifyOutcome::Passed { .. }),
                "{outcome:?}"
            );
        } else {
            assert!(
                matches!(outcome, VerifyOutcome::Failed { .. }),
                "{outcome:?}"
            );
        }
        assert_eq!(verifier.execution_count(), 1);
        assert_eq!(
            verifier.executions()[0].process.as_ref().unwrap().exit_code,
            Some(if passed { 0 } else { 1 })
        );
        assert_eq!(workspace_snapshot(&root).await.unwrap(), baseline);
        assert_eq!(verifier.take_observations().len(), 1);
        assert!(
            matches!(
                verifier
                    .check(&workspace, &baseline, &mut cache, None, &mut ui)
                    .await,
                VerifyOutcome::NotRun
            ),
            "explicit admission must not bypass the round cap"
        );
        assert_eq!(verifier.execution_count(), 1);
        let _ = std::fs::remove_dir_all(base);
    }
}

#[tokio::test]
async fn current_revision_auto_discovers_real_root_checks_without_changed_paths() {
    for (bytes, passed) in [("ready\n", true), ("broken\n", false)] {
        let (base, root, state) = roots("current-revision-auto");
        std::fs::write(root.join("input.rs"), bytes).unwrap();
        std::fs::write(
            root.join("Makefile"),
            "check:\n\t@test \"$$(cat input.rs)\" = ready\n",
        )
        .unwrap();
        let baseline = workspace_snapshot(&root).await.unwrap();
        let lsp = hi_lsp::LspManager::new(&root).unwrap();
        let workspace = VerifyWorkspace::new(&root, &state, None, &lsp)
            .with_changed_files(&[])
            .with_admission(VerificationAdmission::CurrentRevision);
        let mut verifier = WorkspaceRepairVerifier::automatic(Vec::new(), 1);
        let outcome = verifier
            .check(
                &workspace,
                &baseline,
                &mut SnapshotCache::default(),
                None,
                &mut NullUi,
            )
            .await;
        if passed {
            assert!(
                matches!(outcome, VerifyOutcome::Passed { .. }),
                "{outcome:?}"
            );
        } else {
            assert!(
                matches!(outcome, VerifyOutcome::Failed { .. }),
                "{outcome:?}"
            );
        }
        assert_eq!(verifier.execution_count(), 1);
        assert_eq!(verifier.executions()[0].command, "make check");
        assert_eq!(workspace_snapshot(&root).await.unwrap(), baseline);
        let _ = std::fs::remove_dir_all(base);
    }
}

#[tokio::test]
async fn current_revision_admission_keeps_off_empty_caps_and_writer_fences() {
    let (base, root, state) = roots("current-revision-gates");
    std::fs::write(root.join("input.rs"), "ready\n").unwrap();
    let baseline = workspace_snapshot(&root).await.unwrap();
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let workspace = VerifyWorkspace::new(&root, &state, None, &lsp)
        .with_changed_files(&[])
        .with_admission(VerificationAdmission::CurrentRevision);
    let stage = VerifyStage::new("input", "test \"$(cat input.rs)\" = ready");
    for mut verifier in [
        WorkspaceRepairVerifier::new(Vec::new(), 1),
        WorkspaceRepairVerifier::automatic(Vec::new(), 1),
        WorkspaceRepairVerifier::new(vec![stage.clone()], 0),
    ] {
        let outcome = verifier
            .check(
                &workspace,
                &baseline,
                &mut SnapshotCache::default(),
                None,
                &mut NullUi,
            )
            .await;
        assert!(matches!(outcome, VerifyOutcome::NotRun), "{outcome:?}");
        assert_eq!(verifier.execution_count(), 0);
    }

    let mut verifier = WorkspaceRepairVerifier::new(vec![stage], 1);
    let workspace = workspace.with_active_managed_live_writer(true);
    let outcome = verifier
        .check(
            &workspace,
            &baseline,
            &mut SnapshotCache::default(),
            None,
            &mut NullUi,
        )
        .await;
    assert!(
        matches!(outcome, VerifyOutcome::DeferredActiveWriter { .. }),
        "{outcome:?}"
    );
    assert_eq!(verifier.execution_count(), 0);
    assert_eq!(verifier.round(), 0);
    let _ = std::fs::remove_dir_all(base);
}

#[tokio::test]
async fn current_revision_check_cannot_seal_a_stage_modified_input() {
    let (base, root, state) = roots("current-revision-unstable");
    std::fs::write(root.join("input.rs"), "ready\n").unwrap();
    let baseline = workspace_snapshot(&root).await.unwrap();
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let workspace = VerifyWorkspace::new(&root, &state, None, &lsp)
        .with_changed_files(&[])
        .with_admission(VerificationAdmission::CurrentRevision);
    let mut verifier = WorkspaceRepairVerifier::new(
        vec![VerifyStage::new(
            "rewrites-input",
            "printf changed > input.rs",
        )],
        1,
    );
    let outcome = verifier
        .check(
            &workspace,
            &baseline,
            &mut SnapshotCache::default(),
            None,
            &mut NullUi,
        )
        .await;
    assert!(
        matches!(outcome, VerifyOutcome::Unstable { .. }),
        "{outcome:?}"
    );
    assert_eq!(verifier.execution_count(), 1);
    assert_eq!(
        std::fs::read_to_string(root.join("input.rs")).unwrap(),
        "changed"
    );
    let _ = std::fs::remove_dir_all(base);
}

#[cfg(test)]
#[test]
fn only_healthy_active_writer_admission_is_a_verifier_deferral() {
    let denied = |reason, state| {
        anyhow::Error::from(hi_workspace::AdmissionDenied {
            reason,
            state,
            detail: "test admission denial".into(),
        })
    };
    assert!(native_verifier_deferred_by_active_writer(&denied(
        hi_workspace::AdmissionDeniedReason::ActiveWriter,
        hi_workspace::WorkspaceState::Ready,
    )));
    for (reason, state) in [
        (
            hi_workspace::AdmissionDeniedReason::NotReady,
            hi_workspace::WorkspaceState::RecoveryRequired,
        ),
        (
            hi_workspace::AdmissionDeniedReason::NotReady,
            hi_workspace::WorkspaceState::LeaseLost,
        ),
        (
            hi_workspace::AdmissionDeniedReason::NotReady,
            hi_workspace::WorkspaceState::Conflict,
        ),
        (
            hi_workspace::AdmissionDeniedReason::Incompatible,
            hi_workspace::WorkspaceState::Incompatible,
        ),
        (
            hi_workspace::AdmissionDeniedReason::ActiveMutation,
            hi_workspace::WorkspaceState::Ready,
        ),
    ] {
        assert!(!native_verifier_deferred_by_active_writer(&denied(
            reason, state
        )));
    }
}
