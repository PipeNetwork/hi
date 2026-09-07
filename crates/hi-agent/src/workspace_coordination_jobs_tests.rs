use super::*;
use hi_control::{ControlJobState, ControlStore};

fn coordination_with_candidates(
    root: &std::path::Path,
    state: &std::path::Path,
) -> WorkspaceCoordination {
    let mut harness = hi_workspace::ResolvedHarnessSettings::default();
    harness.features.candidate_jobs_v2 = true;
    WorkspaceCoordination::new_local_with_settings(root, state, harness)
}

fn registration(effect: BackgroundJobEffect) -> BackgroundJobRegistration {
    BackgroundJobRegistration {
        id: BackgroundJobId {
            source_id: "process-registry".into(),
            handle: "server_1".into(),
        },
        kind: match effect {
            BackgroundJobEffect::ReadOnly => BackgroundJobKind::ReadAgent,
            BackgroundJobEffect::CandidateOnly => BackgroundJobKind::WriteCandidate,
            BackgroundJobEffect::LiveWriter => BackgroundJobKind::Process,
        },
        effect,
        name: "test background process".into(),
    }
}

#[test]
fn already_sealed_success_does_not_acknowledge_recovery_required() {
    let job_id = JobId::new("job-1");
    let outcome = JobSealOutcome {
        job_id: job_id.clone(),
        status: JobSealStatus::AlreadySealed,
        state: Some(JobState::Succeeded),
        recovery_id: None,
        detail: Some("inner controller had already published success".into()),
    };

    let error = acknowledge_seal(&job_id, JobState::RecoveryRequired, &outcome).unwrap_err();
    assert!(error.contains("AlreadySealed"));
    assert!(error.contains("Succeeded"));
    assert!(error.contains("RecoveryRequired"));
}

#[tokio::test]
async fn live_writer_success_waits_for_the_workspace_receipt_exactly_once() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let store = ControlStore::open_for_state(&state).unwrap();
    let bridge = WorkspaceJobLifecycleBridge::new(coordination.clone());
    coordination.begin(None, None).await.unwrap();

    let registration = registration(BackgroundJobEffect::LiveWriter);
    bridge.register(registration.clone()).await.unwrap();
    let job_id = bridge
        .jobs
        .lock()
        .await
        .get(&registration.id)
        .unwrap()
        .job_id
        .clone();
    assert_eq!(
        store.get_job(job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::Running
    );

    assert_eq!(
        bridge
            .observe_terminal(&registration.id, BackgroundJobTerminal::Succeeded, None)
            .await
            .unwrap(),
        BackgroundJobPublication::DurabilityPending
    );
    let pending_record = store.get_job(job_id.as_str()).unwrap().unwrap();
    assert_eq!(pending_record.state, ControlJobState::DurabilityPending);
    assert_eq!(
        bridge
            .observe_terminal(&registration.id, BackgroundJobTerminal::Succeeded, None)
            .await
            .unwrap(),
        BackgroundJobPublication::DurabilityPending
    );
    assert_eq!(
        store.get_job(job_id.as_str()).unwrap().unwrap().revision,
        pending_record.revision,
        "a repeated process callback must not write a second transition"
    );

    let pending = bridge.pending(&registration.id.source_id).await;
    coordination
        .checkpoint(
            None,
            hi_workspace::ExecutionReport::succeeded(Some("workspace-receipt".into())),
        )
        .await
        .unwrap();
    assert_eq!(
        store.get_job(job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::DurabilityPending,
        "workspace settlement alone must not publish a job omitted from the frozen set"
    );
    bridge.settle_after_workspace(&pending).await.unwrap();
    let succeeded = store.get_job(job_id.as_str()).unwrap().unwrap();
    assert_eq!(succeeded.state, ControlJobState::Succeeded);

    bridge.settle_after_workspace(&pending).await.unwrap();
    assert_eq!(
        store.get_job(job_id.as_str()).unwrap().unwrap().revision,
        succeeded.revision,
        "repeated settlement must be idempotent"
    );
}

#[tokio::test]
async fn mixed_batch_children_inherit_the_active_parent_operation() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    let coordination = coordination_with_candidates(&root, &state);
    let store = ControlStore::open_for_state(&state).unwrap();
    let bridge = WorkspaceJobLifecycleBridge::new(coordination.clone());
    coordination.begin(None, None).await.unwrap();
    let reader = registration(BackgroundJobEffect::ReadOnly);
    bridge.register(reader.clone()).await.unwrap();
    let job_id = bridge.jobs.lock().await[&reader.id].job_id.clone();

    assert_eq!(
        bridge
            .observe_terminal(&reader.id, BackgroundJobTerminal::Succeeded, None)
            .await
            .unwrap(),
        BackgroundJobPublication::Published
    );
    assert_eq!(
        store.get_job(job_id.as_str()).unwrap().unwrap().state,
        ControlJobState::Succeeded
    );

    let mut candidate = registration(BackgroundJobEffect::CandidateOnly);
    candidate.id.handle = "candidate_1".into();
    bridge.register(candidate.clone()).await.unwrap();
    let candidate_id = bridge.jobs.lock().await[&candidate.id].job_id.clone();
    assert_eq!(
        bridge
            .observe_terminal(&candidate.id, BackgroundJobTerminal::Succeeded, None)
            .await
            .unwrap(),
        BackgroundJobPublication::DurabilityPending
    );
    assert_eq!(
        store.get_job(candidate_id.as_str()).unwrap().unwrap().state,
        ControlJobState::ReadyToMerge
    );
    coordination
        .checkpoint(None, hi_workspace::ExecutionReport::succeeded(None))
        .await
        .unwrap();
    for (transition, expected) in [
        (
            BackgroundCandidateTransition::Merging,
            ControlJobState::Merging,
        ),
        (
            BackgroundCandidateTransition::Settling,
            ControlJobState::Settling,
        ),
        (
            BackgroundCandidateTransition::Succeeded,
            ControlJobState::Succeeded,
        ),
    ] {
        bridge
            .transition_candidate(&candidate.id, transition, None)
            .await
            .unwrap();
        assert_eq!(
            store.get_job(candidate_id.as_str()).unwrap().unwrap().state,
            expected
        );
    }
}

#[tokio::test]
async fn cancelled_candidate_cannot_reenter_merging() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    let coordination = coordination_with_candidates(&root, &state);
    let bridge = WorkspaceJobLifecycleBridge::new(coordination);
    let candidate = registration(BackgroundJobEffect::CandidateOnly);
    bridge.register(candidate.clone()).await.unwrap();
    bridge
        .observe_terminal(&candidate.id, BackgroundJobTerminal::Succeeded, None)
        .await
        .unwrap();
    bridge
        .observe_terminal(&candidate.id, BackgroundJobTerminal::Cancelled, None)
        .await
        .unwrap();

    let error = bridge
        .transition_candidate(&candidate.id, BackgroundCandidateTransition::Merging, None)
        .await
        .unwrap_err();
    assert!(error.contains("already terminal"));
}

#[tokio::test]
async fn resolved_limits_gate_admission_and_populate_job_deadlines() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    let mut harness = hi_workspace::ResolvedHarnessSettings::default();
    harness.jobs.max_active = 1;
    harness.jobs.queue_timeout = std::time::Duration::from_millis(11);
    harness.jobs.candidate_timeout = std::time::Duration::from_millis(22);
    harness.jobs.verifier_timeout = std::time::Duration::from_millis(33);
    harness.features.candidate_jobs_v2 = true;
    let coordination = WorkspaceCoordination::new_local_with_settings(&root, &state, harness);
    let bridge = WorkspaceJobLifecycleBridge::new(coordination);

    let first = registration(BackgroundJobEffect::ReadOnly);
    bridge.register(first.clone()).await.unwrap();
    let mut second = registration(BackgroundJobEffect::ReadOnly);
    second.id.handle = "reader_2".into();
    let error = bridge.register(second).await.unwrap_err();
    assert!(error.contains("concurrency reached (1)"));

    let settings = bridge.coordination.harness_settings();
    assert_eq!(
        managed_limits(&settings.jobs, BackgroundJobKind::WriteCandidate),
        JobLimits {
            queue_ms: Some(11),
            execution_ms: Some(22),
            verification_ms: Some(33),
            output_bytes: None,
        }
    );
}

#[tokio::test]
async fn pipefs_protocol_one_rejects_background_candidates_in_the_lifecycle_bridge() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    let coordination = coordination_with_candidates(&root, &state);
    coordination
        .install_pipefs("protocol-one", 1, false, &root, &state)
        .unwrap();
    let bridge = WorkspaceJobLifecycleBridge::new(coordination);

    let error = bridge
        .register(registration(BackgroundJobEffect::CandidateOnly))
        .await
        .unwrap_err();
    assert!(error.contains("unavailable"));
}

#[tokio::test]
async fn local_candidates_are_also_closed_until_the_rollout_gate_is_enabled() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let bridge = WorkspaceJobLifecycleBridge::new(coordination);

    let error = bridge
        .register(registration(BackgroundJobEffect::CandidateOnly))
        .await
        .unwrap_err();
    assert!(error.contains("unavailable"));
}

#[tokio::test]
async fn resolved_limits_reach_the_local_controller_for_direct_jobs() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    let mut harness = hi_workspace::ResolvedHarnessSettings::default();
    harness.jobs.max_preparations = 1;
    harness.jobs.max_active = 2;
    let coordination = WorkspaceCoordination::new_local_with_settings(&root, &state, harness);
    let controller = coordination.job_controller();
    let candidate = |name: &str| JobSpec {
        kind: JobKind::WriteCandidate,
        effect_scope: EffectScope::CandidateOnly,
        name: name.into(),
        limits: JobLimits::default(),
        parent_operation: None,
    };

    controller.register_job(candidate("first")).await.unwrap();
    let preparation_error = controller
        .register_job(candidate("second"))
        .await
        .unwrap_err();
    assert!(
        preparation_error
            .detail
            .contains("candidate preparation limit reached (1)")
    );
    controller
        .register_job(JobSpec {
            kind: JobKind::ReadAgent,
            effect_scope: EffectScope::ReadOnly,
            name: "reader".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();
    let active_error = controller
        .register_job(JobSpec {
            kind: JobKind::ReadAgent,
            effect_scope: EffectScope::ReadOnly,
            name: "overflow".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap_err();
    assert!(active_error.detail.contains("active job limit reached (2)"));
}

#[tokio::test]
async fn frozen_job_drain_does_not_serialize_independent_publications() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("workspace");
    let state = directory.path().join("state");
    std::fs::create_dir_all(&root).unwrap();
    let coordination = WorkspaceCoordination::new_local(&root, &state);
    let bridge = Arc::new(WorkspaceJobLifecycleBridge::new(coordination.clone()));
    coordination.begin(None, None).await.unwrap();
    let first = registration(BackgroundJobEffect::LiveWriter);
    let mut second = first.clone();
    second.id.handle = "second".into();
    for registration in [&first, &second] {
        bridge.register(registration.clone()).await.unwrap();
        bridge
            .observe_terminal(&registration.id, BackgroundJobTerminal::Succeeded, None)
            .await
            .unwrap();
    }
    let jobs = bridge.jobs.lock().await;
    let blocked = jobs[&first.id].clone();
    let independent = jobs[&second.id].clone();
    drop(jobs);
    let guard = blocked.gate.lock().await;
    coordination
        .checkpoint(
            None,
            hi_workspace::ExecutionReport::succeeded(Some("receipt".into())),
        )
        .await
        .unwrap();
    let owner = bridge.clone();
    let drain =
        tokio::spawn(async move { owner.settle_after_workspace(&[first.id, second.id]).await });
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while independent.state().unwrap() != JobState::Succeeded {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the second frozen job must publish while the first gate is held");
    assert!(!drain.is_finished());
    drop(guard);
    drain.await.unwrap().unwrap();
    assert_eq!(blocked.state().unwrap(), JobState::Succeeded);
}
