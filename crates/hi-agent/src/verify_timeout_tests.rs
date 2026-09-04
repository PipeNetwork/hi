use super::verify_test_support::{NullUi, checkpoint, roots};
use super::*;
use crate::snapshot::workspace_snapshot;
use std::time::Duration;

#[tokio::test]
async fn baseline_attribution_respects_the_callers_verification_timeout() {
    let (base, root, state) = roots("baseline-timeout");
    std::fs::write(root.join("state.toml"), "baseline\n").unwrap();
    let turn_snapshot = workspace_snapshot(&root).await.unwrap();
    let checkpoint = checkpoint(&root, &state).await;
    std::fs::write(root.join("state.toml"), "changed\n").unwrap();

    // The edited revision fails immediately, while the baseline is slow.
    // Attribution must inherit the caller's cap instead of reverting to an
    // unbounded command when HI_VERIFY_TIMEOUT_SECS is unset.
    let mut verifier = RepairVerifier::new(
        vec![VerifyStage::new(
            "test",
            "if test \"$(cat state.toml)\" = baseline; then sleep 10; else exit 1; fi",
        )],
        1,
    );
    verifier.timeout_override = Some(Duration::from_millis(200));
    let lsp = hi_lsp::LspManager::new(&root).unwrap();
    let mut cache = SnapshotCache::default();
    let mut ui = NullUi;
    let outcome = tokio::time::timeout(
        Duration::from_secs(3),
        verifier.check(
            &VerifyWorkspace::new(&root, &state, Some(&checkpoint), &lsp),
            &turn_snapshot,
            &mut cache,
            None,
            &mut ui,
        ),
    )
    .await
    .expect("baseline attribution ignored the caller's verification timeout");

    let VerifyOutcome::Failed { output, round, .. } = outcome else {
        panic!("the current revision still has a real failure: {outcome:?}");
    };
    assert_eq!(round, 1);
    assert!(
        output.contains("isolated baseline command timed out"),
        "{output}"
    );
    assert!(!output.contains("already failed this verification stage"));
    assert_eq!(
        std::fs::read_to_string(root.join("state.toml")).unwrap(),
        "changed\n"
    );
    assert!(!state.join("verification-sandboxes").exists());
    std::fs::remove_dir_all(base).unwrap();
}
