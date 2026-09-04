use std::time::{Duration, Instant};

use crate::{
    BarrierKind, BarrierStatus, BindingId, EffectScope, JobCompletion, JobFence, JobKind,
    JobLimits, JobRegistryError, JobRegistryLimits, JobSpec, JobState, JobTerminal,
    JobWaitCondition, WorkspaceBinding, WorkspaceJobRegistry,
};

fn binding() -> WorkspaceBinding {
    WorkspaceBinding::new_local(
        "controller".into(),
        "workspace".into(),
        "/work".into(),
        "/state".into(),
    )
}

fn spec(kind: JobKind, effect_scope: EffectScope, name: &str) -> JobSpec {
    JobSpec {
        kind,
        effect_scope,
        name: name.to_owned(),
        limits: JobLimits::default(),
        parent_operation: None,
    }
}

fn terminal(completion: JobCompletion) -> JobTerminal {
    JobTerminal {
        completion,
        detail: None,
        artifacts: Vec::new(),
    }
}

fn advance(
    registry: &WorkspaceJobRegistry,
    fence: &JobFence,
    job_id: &crate::JobId,
    from: JobState,
    to: JobState,
) {
    registry
        .transition(fence, job_id, from, to, None, Vec::new())
        .unwrap();
}

#[test]
fn defaults_and_active_limit_are_enforced() {
    let binding = binding();
    let registry = WorkspaceJobRegistry::new(binding);
    let fence = registry.fence();
    assert_eq!(registry.snapshot().limits.max_preparations, 4);
    assert_eq!(registry.snapshot().limits.max_active_jobs, 16);

    for index in 0..16 {
        registry
            .register(
                &fence,
                spec(
                    JobKind::ReadAgent,
                    EffectScope::ReadOnly,
                    &format!("r{index}"),
                ),
            )
            .unwrap();
    }
    assert_eq!(
        registry
            .register(
                &fence,
                spec(JobKind::ReadAgent, EffectScope::ReadOnly, "overflow"),
            )
            .unwrap_err(),
        JobRegistryError::ActiveLimitReached { limit: 16 }
    );
}

#[test]
fn only_four_candidates_prepare_concurrently() {
    let registry = WorkspaceJobRegistry::new(binding());
    let fence = registry.fence();
    let jobs: Vec<_> = (0..5)
        .map(|index| {
            registry
                .register(
                    &fence,
                    spec(
                        JobKind::WriteCandidate,
                        EffectScope::CandidateOnly,
                        &format!("candidate-{index}"),
                    ),
                )
                .unwrap()
        })
        .collect();
    for job in jobs.iter().take(4) {
        advance(
            &registry,
            &fence,
            &job.job_id,
            JobState::Queued,
            JobState::Starting,
        );
    }
    assert_eq!(
        registry
            .transition(
                &fence,
                &jobs[4].job_id,
                JobState::Queued,
                JobState::Starting,
                None,
                Vec::new(),
            )
            .unwrap_err(),
        JobRegistryError::PreparationLimitReached { limit: 4 }
    );
    assert_eq!(
        registry
            .seal(&fence, &jobs[0].job_id, terminal(JobCompletion::Failed))
            .status,
        crate::JobSealStatus::Sealed
    );
    advance(
        &registry,
        &fence,
        &jobs[4].job_id,
        JobState::Queued,
        JobState::Starting,
    );
}

#[test]
fn stale_epoch_callbacks_cannot_register_or_transition() {
    let registry = WorkspaceJobRegistry::new(binding());
    let fence = registry.fence();
    let job = registry
        .register(
            &fence,
            spec(JobKind::ReadAgent, EffectScope::ReadOnly, "reader"),
        )
        .unwrap();
    let mut stale = fence.clone();
    stale.epoch = stale.epoch.saturating_add(1);
    assert_eq!(
        registry
            .register(
                &stale,
                spec(JobKind::ReadAgent, EffectScope::ReadOnly, "stale"),
            )
            .unwrap_err(),
        JobRegistryError::StaleFence
    );
    assert_eq!(
        registry
            .transition(
                &stale,
                &job.job_id,
                JobState::Queued,
                JobState::Starting,
                None,
                Vec::new(),
            )
            .unwrap_err(),
        JobRegistryError::StaleFence
    );
}

#[test]
fn candidate_success_requires_merge_and_settlement_and_seals_once() {
    let registry = WorkspaceJobRegistry::new(binding());
    let fence = registry.fence();
    let job = registry
        .register(
            &fence,
            spec(
                JobKind::WriteCandidate,
                EffectScope::CandidateOnly,
                "writer",
            ),
        )
        .unwrap();
    advance(
        &registry,
        &fence,
        &job.job_id,
        JobState::Queued,
        JobState::Starting,
    );
    advance(
        &registry,
        &fence,
        &job.job_id,
        JobState::Starting,
        JobState::Running,
    );
    assert_eq!(
        registry
            .seal(&fence, &job.job_id, terminal(JobCompletion::Succeeded))
            .status,
        crate::JobSealStatus::Rejected
    );
    assert_eq!(
        registry
            .seal(&fence, &job.job_id, terminal(JobCompletion::ReadyToMerge))
            .state,
        Some(JobState::ReadyToMerge)
    );
    advance(
        &registry,
        &fence,
        &job.job_id,
        JobState::ReadyToMerge,
        JobState::Merging,
    );
    advance(
        &registry,
        &fence,
        &job.job_id,
        JobState::Merging,
        JobState::Settling,
    );
    assert_eq!(
        registry
            .seal(&fence, &job.job_id, terminal(JobCompletion::Succeeded))
            .status,
        crate::JobSealStatus::Sealed
    );
    assert_eq!(
        registry
            .seal(&fence, &job.job_id, terminal(JobCompletion::Succeeded))
            .status,
        crate::JobSealStatus::AlreadySealed
    );
}

#[test]
fn restart_requires_recovery_for_writers_and_orphans_readers() {
    let original_binding = binding();
    let registry = WorkspaceJobRegistry::new(original_binding.clone());
    let fence = registry.fence();
    let candidate = registry
        .register(
            &fence,
            spec(
                JobKind::WriteCandidate,
                EffectScope::CandidateOnly,
                "candidate",
            ),
        )
        .unwrap();
    let stale_candidate = registry
        .register(
            &fence,
            spec(
                JobKind::WriteCandidate,
                EffectScope::CandidateOnly,
                "stale candidate",
            ),
        )
        .unwrap();
    let writer = registry
        .register(
            &fence,
            spec(JobKind::Process, EffectScope::LiveWriter, "writer"),
        )
        .unwrap();
    let reader = registry
        .register(
            &fence,
            spec(JobKind::ReadAgent, EffectScope::ReadOnly, "reader"),
        )
        .unwrap();
    for job_id in [
        &candidate.job_id,
        &stale_candidate.job_id,
        &writer.job_id,
        &reader.job_id,
    ] {
        advance(
            &registry,
            &fence,
            job_id,
            JobState::Queued,
            JobState::Starting,
        );
        advance(
            &registry,
            &fence,
            job_id,
            JobState::Starting,
            JobState::Running,
        );
    }

    let mut jobs = registry.snapshot().jobs;
    jobs.iter_mut()
        .find(|job| job.permit.job_id == stale_candidate.job_id)
        .unwrap()
        .permit
        .binding_id = BindingId::new("old-binding");
    let (restored, report) =
        WorkspaceJobRegistry::restore(original_binding, JobRegistryLimits::default(), jobs)
            .unwrap();
    let restored_fence = restored.fence();

    assert_eq!(
        restored
            .status(&restored_fence, &candidate.job_id)
            .unwrap()
            .state,
        JobState::RecoveryRequired
    );
    assert_eq!(
        restored
            .status(&restored_fence, &writer.job_id)
            .unwrap()
            .state,
        JobState::RecoveryRequired
    );
    assert_eq!(
        restored
            .status(&restored_fence, &reader.job_id)
            .unwrap()
            .state,
        JobState::Orphaned
    );
    // Stale records remain visible in the aggregate projection, but their old
    // permit cannot be addressed through the current fence.
    assert!(
        restored
            .snapshot()
            .jobs
            .iter()
            .any(|job| job.permit.job_id == stale_candidate.job_id && job.state == JobState::Stale)
    );
    assert!(report.recovery_required.contains(&candidate.job_id));
    assert!(report.recovery_required.contains(&writer.job_id));
    assert!(report.orphaned.contains(&reader.job_id));
    assert!(report.stale.contains(&stale_candidate.job_id));
    assert_eq!(
        restored
            .barrier(
                &restored_fence,
                BarrierKind::Exit,
                Instant::now() + Duration::from_secs(1),
            )
            .unwrap()
            .status,
        BarrierStatus::RecoveryRequired
    );
}

#[test]
fn recovery_required_job_has_identity_and_explicit_idempotent_resolution() {
    let registry = WorkspaceJobRegistry::new(binding());
    let fence = registry.fence();
    let job = registry
        .register(
            &fence,
            spec(JobKind::Process, EffectScope::LiveWriter, "writer"),
        )
        .unwrap();
    advance(
        &registry,
        &fence,
        &job.job_id,
        JobState::Queued,
        JobState::Starting,
    );
    advance(
        &registry,
        &fence,
        &job.job_id,
        JobState::Starting,
        JobState::Running,
    );

    let sealed = registry.seal(
        &fence,
        &job.job_id,
        terminal(JobCompletion::RecoveryRequired),
    );
    let recovery_id = sealed
        .recovery_id
        .expect("recovery-required jobs need an addressable recovery");
    let blocked = registry
        .barrier(
            &fence,
            BarrierKind::Exit,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    assert_eq!(blocked.status, BarrierStatus::RecoveryRequired);
    assert_eq!(blocked.recovery_id.as_ref(), Some(&recovery_id));

    let resolved = registry
        .reconcile_recovery(
            &fence,
            &recovery_id,
            Some("operator inspected effects".into()),
        )
        .unwrap();
    assert_eq!(resolved.state, JobState::Failed);
    assert_eq!(resolved.recovery_id.as_ref(), Some(&recovery_id));
    assert_eq!(resolved.terminal.unwrap().completion, JobCompletion::Failed);
    assert_eq!(
        registry
            .barrier(&fence, BarrierKind::Exit, Instant::now())
            .unwrap()
            .status,
        BarrierStatus::Passed
    );

    let revision = resolved.revision;
    assert_eq!(
        registry
            .reconcile_recovery(&fence, &recovery_id, None)
            .unwrap()
            .revision,
        revision,
        "repeated reconciliation must not write a second terminal transition"
    );
}

#[tokio::test]
async fn wait_cancel_and_bounded_output_use_the_same_projection() {
    let registry = WorkspaceJobRegistry::new(binding());
    let fence = registry.fence();
    let mut output_spec = spec(JobKind::Process, EffectScope::ReadOnly, "process");
    output_spec.limits.output_bytes = Some(4);
    let job = registry.register(&fence, output_spec).unwrap();
    registry
        .append_output(&fence, &job.job_id, crate::JobOutputStream::Stdout, "aéz")
        .unwrap();
    let output = registry.job_output(&fence, &job.job_id).unwrap();
    assert_eq!(output.output_bytes, 4);
    assert_eq!(output.output[0].text, "aéz");
    registry
        .append_output(&fence, &job.job_id, crate::JobOutputStream::Stderr, "tail")
        .unwrap();
    let output = registry.job_output(&fence, &job.job_id).unwrap();
    assert!(output.output_truncated);
    assert_eq!(output.output[0].text, "tail");

    let waiting_registry = registry.clone();
    let waiting_fence = fence.clone();
    let waiting_job = job.job_id.clone();
    let waiter = tokio::spawn(async move {
        waiting_registry
            .wait(
                &waiting_fence,
                &waiting_job,
                JobWaitCondition::Terminal,
                Some(Instant::now() + Duration::from_secs(1)),
            )
            .await
    });
    tokio::task::yield_now().await;
    registry.job_cancel(&fence, &job.job_id).unwrap();
    assert_eq!(waiter.await.unwrap().unwrap().state, JobState::Cancelled);
}
