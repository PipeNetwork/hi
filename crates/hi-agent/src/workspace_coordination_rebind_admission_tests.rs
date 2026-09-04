use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use hi_tools::{
    BackgroundJobEffect, BackgroundJobKind, BackgroundJobLifecycle, BackgroundJobRegistration,
    BackgroundJobTerminal, BackgroundTaskOutcome, BackgroundTaskRegistry, BackgroundTaskState,
};
use hi_workspace::{
    BarrierKind, ExecutionReport, InMemoryWorkspaceController, WorkspaceController, WorkspaceState,
};

use super::jobs::WorkspaceJobLifecycleBridge;
use super::{WorkspaceCoordination, WorkspaceDurability};

struct CountingDurability(Arc<AtomicUsize>);

#[async_trait::async_trait]
impl WorkspaceDurability for CountingDurability {
    async fn mutation_started(&self, _dirty_paths: Option<Vec<String>>) -> anyhow::Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn read_registration(handle: &str) -> BackgroundJobRegistration {
    BackgroundJobRegistration {
        id: hi_tools::BackgroundJobId {
            source_id: "rebind-test".into(),
            handle: handle.into(),
        },
        kind: BackgroundJobKind::ReadAgent,
        effect: BackgroundJobEffect::ReadOnly,
        name: "rebind race reader".into(),
    }
}

#[tokio::test]
async fn rebind_gate_drains_terminals_and_rejects_a_waiting_stale_registry_admission() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    let replacement_root = directory.path().join("replacement-workspace");
    let replacement_state = directory.path().join("replacement-state");
    for path in [&root, &state, &replacement_root, &replacement_state] {
        std::fs::create_dir_all(path).unwrap();
    }

    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let old_controller = coordination.job_controller();

    // Terminal callbacks deliberately do not acquire admission. They must be
    // able to drain jobs after the exclusive rebind side has closed new work.
    let draining_bridge = WorkspaceJobLifecycleBridge::new(coordination.clone());
    let draining = read_registration("draining");
    draining_bridge.register(draining.clone()).await.unwrap();
    let rebind = coordination.close_admission_for_rebind().await;
    draining_bridge
        .observe_terminal(
            &draining.id,
            BackgroundJobTerminal::Cancelled,
            Some("drained for rebind".into()),
        )
        .await
        .unwrap();

    let old_processes = hi_tools::BackgroundRegistry::default();
    let tasks = Arc::new(BackgroundTaskRegistry::new());
    coordination.bind_background_registries(&old_processes, &tasks);
    let ran = Arc::new(AtomicBool::new(false));
    let worker_ran = ran.clone();
    let cloned_registry = tasks.clone();
    let spawning = tokio::spawn(async move {
        cloned_registry
            .spawn(
                "racing admission",
                "explore",
                Box::new(move || {
                    Box::pin(async move {
                        worker_ran.store(true, Ordering::SeqCst);
                        BackgroundTaskOutcome {
                            id: String::new(),
                            description: String::new(),
                            subagent_type: "explore".into(),
                            state: BackgroundTaskState::Completed,
                            output: "must not run".into(),
                            applied: false,
                            changed_files: Vec::new(),
                        }
                    })
                }),
            )
            .await
    });
    let durability_calls = Arc::new(AtomicUsize::new(0));
    let durability: Arc<dyn WorkspaceDurability> =
        Arc::new(CountingDurability(durability_calls.clone()));
    let waiting_coordination = coordination.clone();
    let waiting_durability = durability.clone();
    let beginning = tokio::spawn(async move {
        waiting_coordination
            .begin(Some(waiting_durability), None)
            .await
    });

    tokio::time::timeout(Duration::from_secs(2), async {
        while coordination.admission_waiting_readers() < 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("cloned registry should reach the closed admission gate");
    assert!(old_controller.status().active_jobs.is_empty());

    // This is the real publication order: final unified barrier, pointer
    // replacement, lifecycle rebinding, then reopening admission.
    coordination
        .require_barrier(BarrierKind::Rebind)
        .await
        .unwrap();
    let replacement: Arc<dyn WorkspaceController> =
        Arc::new(InMemoryWorkspaceController::new_local_at_epoch(
            "replacement",
            &replacement_root,
            &replacement_state,
            old_controller.binding().epoch.saturating_add(1),
        ));
    coordination
        .replace_during_rebind(replacement.clone(), true, &rebind)
        .unwrap();
    let replacement_processes = hi_tools::BackgroundRegistry::default();
    coordination.bind_background_registries(&replacement_processes, &tasks);
    drop(rebind);

    let error = tokio::time::timeout(Duration::from_secs(2), spawning)
        .await
        .expect("waiting registry admission should be released")
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("stale workspace binding"));
    assert!(!ran.load(Ordering::SeqCst));
    assert!(old_controller.status().active_jobs.is_empty());
    assert!(replacement.status().active_jobs.is_empty());

    beginning
        .await
        .expect("foreground admission task should not panic")
        .expect("foreground admission should use the replacement controller");
    assert_eq!(durability_calls.load(Ordering::SeqCst), 0);
    coordination
        .checkpoint(Some(durability), ExecutionReport::succeeded(None))
        .await
        .unwrap();
    assert_eq!(durability_calls.load(Ordering::SeqCst), 0);
    assert_eq!(coordination.status().state, WorkspaceState::Ready);
}
