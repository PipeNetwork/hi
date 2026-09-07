use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

fn roots(label: &str) -> (PathBuf, PathBuf) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir().join(format!(
        "hi-runtime-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let root = base.join("workspace");
    let state = base.join("state");
    std::fs::create_dir_all(&root).unwrap();
    (root, state)
}

async fn wait_for_gate_count(
    gate: &crate::change_ledger::ScanTestGate,
    expected: usize,
    exited: bool,
) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let actual = if exited {
                gate.exited()
            } else {
                gate.entered()
            };
            if actual >= expected {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
    })
    .await
    .expect("ledger scan test gate was not reached");
}

#[test]
fn agents_in_different_roots_have_independent_state() {
    let (first_root, first_state) = roots("one");
    let (second_root, second_state) = roots("two");
    let first = WorkspaceRuntime::new(&first_root, &first_state, LspMode::Off).unwrap();
    let second = WorkspaceRuntime::new(&second_root, &second_state, LspMode::Off).unwrap();
    assert_ne!(first.root(), second.root());
    assert_ne!(first.state_root(), second.state_root());
    assert!(!Arc::ptr_eq(&first.lsp(), &second.lsp()));
    assert!(!std::ptr::eq(first.read_cache(), second.read_cache()));
    first.invalidate_context();
    assert_eq!(first.context_generation(), 1);
    assert_eq!(second.context_generation(), 0);
    first.invalidate_context_after_compaction();
    assert_eq!(first.context_generation(), 2);
    assert_eq!(second.context_generation(), 0);
    let _ = std::fs::remove_dir_all(first_root.parent().unwrap());
    let _ = std::fs::remove_dir_all(second_root.parent().unwrap());
}

#[tokio::test]
async fn background_registries_are_workspace_local() {
    let (first_root, first_state) = roots("background-one");
    let (second_root, second_state) = roots("background-two");
    let first = WorkspaceRuntime::new(&first_root, &first_state, LspMode::Off).unwrap();
    let second = WorkspaceRuntime::new(&second_root, &second_state, LspMode::Off).unwrap();

    let id = first
        .background()
        .spawn(first.process_runner(), "sleep 600")
        .unwrap();
    assert_eq!(first.background().ids(), vec![id.clone()]);
    assert!(second.background().ids().is_empty());
    assert!(second.background().poll(&id).is_err());
    first.background().kill(&id).unwrap();

    let _ = std::fs::remove_dir_all(first_root.parent().unwrap());
    let _ = std::fs::remove_dir_all(second_root.parent().unwrap());
}

#[tokio::test]
async fn dropping_full_reconcile_cancels_worker_and_releases_ledger() {
    let (root, state) = roots("cancel-full-reconcile");
    let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
    std::fs::write(root.join("changed.txt"), "after baseline\n").unwrap();
    let gate = crate::change_ledger::install_scan_test_gate(&root);

    let worker_runtime = Arc::clone(&runtime);
    let reconcile = tokio::spawn(async move { worker_runtime.reconcile_ledger_async().await });
    wait_for_gate_count(&gate, 1, false).await;
    reconcile.abort();
    assert!(reconcile.await.unwrap_err().is_cancelled());
    wait_for_gate_count(&gate, 1, true).await;
    assert!(
        runtime
            .wait_for_ledger_available(std::time::Duration::from_millis(250))
            .await,
        "cancelled reconcile retained the ledger mutex"
    );

    gate.release();
    let changes = runtime.reconcile_ledger_async().await.unwrap();
    assert_eq!(
        changes
            .iter()
            .map(|change| change.path.as_str())
            .collect::<Vec<_>>(),
        vec!["changed.txt"]
    );
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[tokio::test]
async fn cancellation_after_staging_preserves_the_change_for_retry() {
    let (root, state) = roots("cancel-reconcile-before-commit");
    let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
    std::fs::write(root.join("changed.txt"), "after baseline\n").unwrap();
    let gate = crate::change_ledger::install_reconcile_commit_test_gate(&root);

    let worker_runtime = Arc::clone(&runtime);
    let reconcile = tokio::spawn(async move { worker_runtime.reconcile_ledger_async().await });
    wait_for_gate_count(&gate, 1, false).await;
    reconcile.abort();
    assert!(reconcile.await.unwrap_err().is_cancelled());
    wait_for_gate_count(&gate, 1, true).await;
    assert!(
        runtime
            .wait_for_ledger_available(std::time::Duration::from_millis(250))
            .await,
        "cancelled pre-commit reconcile retained the ledger mutex"
    );

    gate.release();
    let changes = runtime.reconcile_ledger_async().await.unwrap();
    assert_eq!(
        changes
            .iter()
            .map(|change| change.path.as_str())
            .collect::<Vec<_>>(),
        vec!["changed.txt"],
        "a cancelled worker must not consume the change before retry"
    );
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_waits_for_a_commit_that_already_owns_publication() {
    let (root, state) = roots("cancel-owned-reconcile-commit");
    let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
    let baseline = runtime.ledger().revision();
    std::fs::write(root.join("changed.txt"), "after baseline\n").unwrap();
    let gate = crate::change_ledger::install_owned_reconcile_commit_test_gate(&root);

    let worker_runtime = Arc::clone(&runtime);
    let reconcile = tokio::spawn(async move { worker_runtime.reconcile_ledger_async().await });
    wait_for_gate_count(&gate, 1, false).await;
    reconcile.abort();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert!(
        !reconcile.is_finished(),
        "dropping the future returned while an owned commit was paused"
    );

    gate.release();
    let cancellation = tokio::time::timeout(std::time::Duration::from_secs(2), reconcile)
        .await
        .expect("owned commit did not release its cancellation waiter")
        .expect_err("aborted reconcile task unexpectedly returned normally");
    assert!(cancellation.is_cancelled());
    wait_for_gate_count(&gate, 1, true).await;
    assert!(runtime.try_ledger().is_some());
    assert_eq!(
        runtime
            .ledger()
            .changes_since(baseline)
            .into_iter()
            .map(|change| change.path)
            .collect::<Vec<_>>(),
        vec!["changed.txt"],
        "the owned commit must publish before cancellation settles"
    );
    assert!(
        runtime.reconcile_ledger_async().await.unwrap().is_empty(),
        "an owned commit must not leave its change for duplicate reconciliation"
    );
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[tokio::test]
async fn dropping_exact_path_reconcile_cancels_hash_and_releases_ledger() {
    let (root, state) = roots("cancel-path-reconcile");
    std::fs::write(root.join("tracked.txt"), "baseline\n").unwrap();
    let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
    std::fs::write(root.join("tracked.txt"), "changed\n").unwrap();
    let gate = crate::change_ledger::install_scan_test_gate(&root);

    let worker_runtime = Arc::clone(&runtime);
    let reconcile = tokio::spawn(async move {
        worker_runtime
            .reconcile_dirty_paths_async(vec!["tracked.txt".into()])
            .await
    });
    wait_for_gate_count(&gate, 1, false).await;
    reconcile.abort();
    assert!(reconcile.await.unwrap_err().is_cancelled());
    wait_for_gate_count(&gate, 1, true).await;
    assert!(
        runtime
            .wait_for_ledger_available(std::time::Duration::from_millis(250))
            .await,
        "cancelled exact-path reconcile retained the ledger mutex"
    );

    gate.release();
    let changes = runtime
        .reconcile_dirty_paths_async(vec!["tracked.txt".into()])
        .await
        .unwrap();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].path, "tracked.txt");
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[tokio::test]
async fn cancelled_startup_scan_wait_never_owns_the_ledger_mutex() {
    let (root, state) = roots("cancel-startup-scan");
    std::fs::write(root.join("tracked.txt"), "baseline\n").unwrap();
    let gate = crate::change_ledger::install_scan_test_gate(&root);
    let scan = crate::BackgroundScan::start(&root, &[], &BTreeSet::new()).unwrap();
    wait_for_gate_count(&gate, 1, false).await;
    let runtime =
        Arc::new(WorkspaceRuntime::new_with_scan(&root, &state, LspMode::Off, Some(scan)).unwrap());

    let waiter_runtime = Arc::clone(&runtime);
    let waiter =
        tokio::spawn(async move { waiter_runtime.ensure_ledger_scan_complete_async().await });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    waiter.abort();
    assert!(waiter.await.unwrap_err().is_cancelled());
    assert!(
        runtime.try_ledger().is_some(),
        "waiting for the startup scan held the ledger mutex"
    );

    drop(runtime);
    wait_for_gate_count(&gate, 1, true).await;
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[tokio::test]
async fn cancelled_goal_input_registration_cannot_publish_a_partial_baseline() {
    let (root, state) = roots("cancel-goal-input");
    let goal = crate::Goal::new("runtime view", vec!["step".into()]);
    goal.export_markdown_to(&root).unwrap();
    let runtime = Arc::new(WorkspaceRuntime::new(&root, &state, LspMode::Off).unwrap());
    runtime.ensure_ledger_scan_complete_async().await.unwrap();
    let before = runtime.ledger().workspace_revision();
    let gate = crate::change_ledger::install_reconcile_commit_test_gate(&root);
    let worker = runtime.clone();
    let registration = tokio::spawn(async move { worker.register_goal_validation_input().await });
    wait_for_gate_count(&gate, 1, false).await;
    registration.abort();
    assert!(registration.await.unwrap_err().is_cancelled());
    wait_for_gate_count(&gate, 1, true).await;
    assert!(
        runtime
            .wait_for_ledger_available(std::time::Duration::from_millis(250))
            .await
    );
    gate.release();
    std::fs::write(
        root.join(crate::goal::GOAL_EXPORT_PATH),
        "later generated view",
    )
    .unwrap();
    assert!(runtime.reconcile_ledger_async().await.unwrap().is_empty());
    assert_eq!(runtime.ledger().workspace_revision(), before);
    runtime.register_goal_validation_input().await.unwrap();
    assert_ne!(runtime.ledger().workspace_revision(), before);
    assert!(runtime.ledger().changed_paths_since(0).is_empty());
    std::fs::remove_dir_all(root.parent().unwrap()).unwrap();
}
