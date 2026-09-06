use super::*;
use hi_control::{ControlJobState, ControlStore};

fn live_writer(handle: &str) -> BackgroundJobRegistration {
    BackgroundJobRegistration {
        id: BackgroundJobId {
            source_id: "process-registry".into(),
            handle: handle.into(),
        },
        kind: BackgroundJobKind::Process,
        effect: BackgroundJobEffect::LiveWriter,
        name: "test background process".into(),
    }
}

#[tokio::test]
async fn keep_background_orphans_only_running_writers_and_settles_an_exit_race() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let store = ControlStore::open_for_state(&state).unwrap();
    let bridge = WorkspaceJobLifecycleBridge::new(coordination.clone());

    let running = live_writer("server_1");
    bridge.register(running.clone()).await.unwrap();
    let running_job_id = bridge.jobs.lock().await[&running.id].job_id.clone();
    assert_eq!(
        bridge
            .observe_terminal(&running.id, BackgroundJobTerminal::Orphaned, None)
            .await
            .unwrap(),
        BackgroundJobPublication::Published
    );
    assert_eq!(
        store
            .get_job(running_job_id.as_str())
            .unwrap()
            .unwrap()
            .state,
        ControlJobState::Orphaned
    );

    let raced = live_writer("raced_server_2");
    bridge.register(raced.clone()).await.unwrap();
    let raced_job_id = bridge.jobs.lock().await[&raced.id].job_id.clone();
    assert_eq!(
        bridge
            .observe_terminal(&raced.id, BackgroundJobTerminal::Succeeded, None)
            .await
            .unwrap(),
        BackgroundJobPublication::DurabilityPending
    );
    assert_eq!(
        bridge
            .observe_terminal(&raced.id, BackgroundJobTerminal::Orphaned, None)
            .await
            .unwrap(),
        BackgroundJobPublication::DurabilityPending
    );
    assert_eq!(
        store.get_job(raced_job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::DurabilityPending
    );
    let pending = bridge.pending(&raced.id.source_id).await;
    assert_eq!(pending.as_slice(), std::slice::from_ref(&raced.id));
    coordination
        .begin_intent(None, hi_workspace::MutationIntent::reconciliation())
        .await
        .unwrap();
    coordination
        .checkpoint(None, hi_workspace::ExecutionReport::succeeded(None))
        .await
        .unwrap();
    bridge.settle_after_workspace(&pending).await.unwrap();
    assert_eq!(
        store.get_job(raced_job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::Succeeded
    );

    drop(bridge);
    let restarted = WorkspaceCoordination::new_local(&root, &state);
    let restart_status = restarted.status();
    assert_eq!(restart_status.state, hi_workspace::WorkspaceState::Ready);
    assert!(restart_status.recovery_id.is_none());
    assert!(restart_status.active_jobs.is_empty());
    restarted
        .begin(None, None)
        .await
        .expect("a durable orphan handoff must not fence restart admission");
    restarted
        .checkpoint(None, hi_workspace::ExecutionReport::succeeded(None))
        .await
        .unwrap();
}
