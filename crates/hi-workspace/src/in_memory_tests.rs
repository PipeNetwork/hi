use std::sync::Arc;

use super::*;
use crate::{EffectScope, IdempotencyKey, JobLimits, ReplayClass};

fn controller() -> InMemoryWorkspaceController {
    InMemoryWorkspaceController::new_local("workspace", "/work/one", "/state/one")
}

#[test]
fn causal_pipefs_still_withholds_live_background_writers() {
    let capabilities = WorkspaceCapabilities::pipefs(true);
    assert!(capabilities.causal_commit);
    assert!(capabilities.candidate_apply);
    assert!(!capabilities.background_writers);
}

#[tokio::test]
async fn admits_one_mutation_and_advances_version_after_settlement() {
    let controller = controller();
    let permit = controller
        .begin(MutationIntent::workspace("edit"))
        .await
        .unwrap();
    assert_eq!(controller.status().state, WorkspaceState::Mutating);
    let denied = controller
        .begin(MutationIntent::workspace("second edit"))
        .await
        .unwrap_err();
    assert_eq!(denied.reason, AdmissionDeniedReason::NotReady);

    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(Some("digest-1".into())))
        .await;
    assert_eq!(outcome.status, SettlementStatus::Durable);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert_eq!(
        controller.binding().version,
        crate::WorkspaceVersion::Local {
            generation: 1,
            content_digest: Some("digest-1".into())
        }
    );
}

#[tokio::test]
async fn known_failed_execution_can_settle_changed_bytes_without_recovery() {
    let controller = controller();
    let permit = controller
        .begin(MutationIntent::workspace("partially failing edit"))
        .await
        .unwrap();
    let outcome = controller
        .settle(
            permit,
            ExecutionReport {
                disposition: ExecutionDisposition::Failed,
                workspace_may_have_changed: true,
                external_effect_may_have_occurred: false,
                content_digest: Some("failed-effect-digest".into()),
                changed_paths: vec![PathBuf::from("partial.txt")],
                artifacts: Vec::new(),
                detail: Some("command exited 1 after writing partial.txt".into()),
            },
        )
        .await;

    assert_eq!(outcome.status, SettlementStatus::Durable);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
    assert!(outcome.recovery_id.is_none());
    assert_eq!(
        controller.binding().version,
        crate::WorkspaceVersion::Local {
            generation: 1,
            content_digest: Some("failed-effect-digest".into()),
        }
    );
}

#[tokio::test]
async fn idempotent_external_key_is_the_operation_publication_key() {
    let controller = controller();
    let key = IdempotencyKey::new("effect-key-1");
    let permit = controller
        .begin(MutationIntent {
            effect_scope: EffectScope::LiveWriter,
            replay_class: ReplayClass::IdempotentExternal { key: key.clone() },
            dirty_paths: None,
            description: Some("idempotent external effect".into()),
        })
        .await
        .unwrap();
    assert_eq!(permit.record().idempotency_key, key);
    controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
}

#[tokio::test]
async fn abandoned_permit_synchronously_fences_new_mutations() {
    let controller = controller();
    let permit = controller
        .begin(MutationIntent::workspace("edit"))
        .await
        .unwrap();
    let serialized = serde_json::to_value(&permit).unwrap();
    assert_eq!(
        serialized["operation_id"],
        permit.record().operation_id.as_str()
    );
    drop(permit);

    let status = controller.status();
    assert_eq!(status.state, WorkspaceState::RecoveryRequired);
    let recovery_id = status.recovery_id.unwrap();
    assert_eq!(
        controller.recovery(&recovery_id).unwrap().kind,
        RecoveryKind::AbandonedMutation
    );
    assert!(
        controller
            .begin(MutationIntent::workspace("blocked"))
            .await
            .is_err()
    );

    let outcome = controller.reconcile(recovery_id).await;
    assert_eq!(outcome.status, RecoveryStatus::Recovered);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn seeded_recoveries_keep_admission_closed_until_each_is_resolved() {
    let controller = controller();
    let binding = controller.binding();
    let first = make_recovery(
        &binding,
        RecoveryKind::CrashedWriterJob,
        None,
        Some(JobId::new("job-one")),
        "first crashed writer",
    );
    let second = make_recovery(
        &binding,
        RecoveryKind::CrashedWriterJob,
        None,
        Some(JobId::new("job-two")),
        "second crashed writer",
    );
    controller.require_recovery(first.clone()).unwrap();
    controller.require_recovery(second.clone()).unwrap();
    assert_eq!(
        controller.status().recovery_id.as_ref(),
        Some(&second.recovery_id)
    );

    controller.reconcile(second.recovery_id).await;
    assert_eq!(controller.status().state, WorkspaceState::RecoveryRequired);
    assert_eq!(
        controller.status().recovery_id.as_ref(),
        Some(&first.recovery_id)
    );
    controller.reconcile(first.recovery_id).await;
    assert_eq!(controller.status().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn rebind_increments_epoch_and_changes_binding_id() {
    let controller = controller();
    let permit = controller
        .begin(MutationIntent::workspace("edit before rebind"))
        .await
        .unwrap();
    controller
        .settle(permit, ExecutionReport::succeeded(Some("old-root".into())))
        .await;
    let before = controller.binding();
    let after = controller.rebind("/work/two", "/state/two").unwrap();
    assert_eq!(after.epoch, before.epoch + 1);
    assert_ne!(after.binding_id, before.binding_id);
    assert_eq!(
        after.version,
        crate::WorkspaceVersion::Local {
            generation: 0,
            content_digest: None,
        }
    );
    assert_eq!(controller.status().epoch, after.epoch);
}

#[tokio::test]
async fn watch_subscription_publishes_state_transitions() {
    let controller = controller();
    let mut status_rx = controller.subscribe();
    let permit = controller
        .begin(MutationIntent::workspace("watched edit"))
        .await
        .unwrap();
    status_rx.changed().await.unwrap();
    assert_eq!(status_rx.borrow().state, WorkspaceState::Mutating);

    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(None))
        .await;
    assert_eq!(outcome.status, SettlementStatus::NoChange);
    status_rx.changed().await.unwrap();
    assert_eq!(status_rx.borrow().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn live_writer_job_blocks_mutations_and_requires_settlement_before_success() {
    let controller = controller();
    let job = controller
        .register_job(JobSpec {
            kind: crate::JobKind::Process,
            effect_scope: EffectScope::LiveWriter,
            name: "writer".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();
    let denied = controller
        .begin(MutationIntent {
            effect_scope: EffectScope::LiveWriter,
            replay_class: ReplayClass::PureWorkspace,
            dirty_paths: None,
            description: None,
        })
        .await
        .unwrap_err();
    assert_eq!(denied.reason, AdmissionDeniedReason::ActiveWriter);

    let succeeded = JobTerminal {
        completion: JobCompletion::Succeeded,
        detail: None,
        artifacts: Vec::new(),
    };
    let premature = controller
        .seal_job(job.job_id.clone(), succeeded.clone())
        .await;
    assert_eq!(premature.status, JobSealStatus::Rejected);
    assert_eq!(controller.job_state(&job.job_id), Some(JobState::Running));

    let denied_reconciliation = controller
        .begin(MutationIntent::reconciliation())
        .await
        .unwrap_err();
    assert_eq!(
        denied_reconciliation.reason,
        AdmissionDeniedReason::ActiveWriter,
        "a reconciliation label must not authorize overlap with a running writer"
    );

    let pending = controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::DurabilityPending,
                detail: None,
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(pending.status, JobSealStatus::Sealed);
    assert_eq!(
        controller
            .begin(MutationIntent::workspace("unrelated writer"))
            .await
            .unwrap_err()
            .reason,
        AdmissionDeniedReason::ActiveWriter
    );
    let reconciliation = controller
        .begin(MutationIntent::reconciliation())
        .await
        .expect("a durability-pending writer must admit its receipt boundary");
    assert_eq!(
        controller
            .settle(reconciliation, ExecutionReport::succeeded(None))
            .await
            .status,
        SettlementStatus::NoChange
    );

    for completion in [JobCompletion::Settling, JobCompletion::Succeeded] {
        let outcome = controller
            .seal_job(
                job.job_id.clone(),
                JobTerminal {
                    completion,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await;
        assert_eq!(outcome.status, JobSealStatus::Sealed);
    }
    let duplicate = controller.seal_job(job.job_id, succeeded).await;
    assert_eq!(duplicate.status, JobSealStatus::AlreadySealed);
}

#[tokio::test]
async fn direct_job_admission_enforces_configured_active_and_preparation_limits() {
    let candidate_controller = InMemoryWorkspaceController::new_local_at_epoch_with_job_limits(
        "candidate-limits",
        "/work/candidates",
        "/state/candidates",
        0,
        JobRegistryLimits {
            max_preparations: 1,
            max_active_jobs: 4,
        },
    );
    let candidate_spec = |name: &str| JobSpec {
        kind: crate::JobKind::WriteCandidate,
        effect_scope: EffectScope::CandidateOnly,
        name: name.into(),
        limits: JobLimits::default(),
        parent_operation: None,
    };
    let first = candidate_controller
        .register_job(candidate_spec("first"))
        .await
        .unwrap();
    let preparation_denial = candidate_controller
        .register_job(candidate_spec("second"))
        .await
        .unwrap_err();
    assert!(
        preparation_denial
            .detail
            .contains("candidate preparation limit reached (1)")
    );
    let ready = candidate_controller
        .seal_job(
            first.job_id,
            JobTerminal {
                completion: JobCompletion::ReadyToMerge,
                detail: None,
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(ready.status, JobSealStatus::Sealed);
    candidate_controller
        .register_job(candidate_spec("second"))
        .await
        .expect("ready-to-merge work no longer consumes a preparation slot");

    let active_controller = InMemoryWorkspaceController::new_local_at_epoch_with_job_limits(
        "active-limits",
        "/work/active",
        "/state/active",
        0,
        JobRegistryLimits {
            max_preparations: 2,
            max_active_jobs: 2,
        },
    );
    let read_spec = |name: &str| JobSpec {
        kind: crate::JobKind::ReadAgent,
        effect_scope: EffectScope::ReadOnly,
        name: name.into(),
        limits: JobLimits::default(),
        parent_operation: None,
    };
    active_controller
        .register_job(read_spec("one"))
        .await
        .unwrap();
    active_controller
        .register_job(read_spec("two"))
        .await
        .unwrap();
    let active_denial = active_controller
        .register_job(read_spec("three"))
        .await
        .unwrap_err();
    assert!(
        active_denial
            .detail
            .contains("active job limit reached (2)")
    );
}

#[tokio::test]
async fn candidate_job_cannot_publish_success_before_parent_apply_settles() {
    let controller = controller();
    let job = controller
        .register_job(JobSpec {
            kind: crate::JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "detached candidate".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();

    let premature = controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::Succeeded,
                detail: Some("child claimed success".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(premature.status, JobSealStatus::Rejected);
    assert_eq!(controller.job_state(&job.job_id), Some(JobState::Running));

    for completion in [
        JobCompletion::ReadyToMerge,
        JobCompletion::Merging,
        JobCompletion::Settling,
        JobCompletion::Succeeded,
    ] {
        assert_eq!(
            controller
                .seal_job(
                    job.job_id.clone(),
                    JobTerminal {
                        completion,
                        detail: None,
                        artifacts: Vec::new(),
                    },
                )
                .await
                .status,
            JobSealStatus::Sealed
        );
    }
}

#[tokio::test]
async fn reconciling_job_recovery_finalizes_job_and_unblocks_barrier() {
    let controller = controller();
    let job = controller
        .register_job(JobSpec {
            kind: crate::JobKind::Process,
            effect_scope: EffectScope::LiveWriter,
            name: "writer needing recovery".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();
    let sealed = controller
        .seal_job(
            job.job_id.clone(),
            JobTerminal {
                completion: JobCompletion::RecoveryRequired,
                detail: Some("writer effects need inspection".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    let recovery_id = sealed.recovery_id.unwrap();
    let blocked = controller.barrier(BarrierKind::Exit, Instant::now()).await;
    assert_eq!(blocked.status, BarrierStatus::RecoveryRequired);
    assert_eq!(blocked.recovery_id.as_ref(), Some(&recovery_id));

    let recovered = controller.reconcile(recovery_id).await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(controller.job_state(&job.job_id), Some(JobState::Failed));
    assert_eq!(
        controller
            .barrier(BarrierKind::Exit, Instant::now())
            .await
            .status,
        BarrierStatus::Passed
    );
}

#[tokio::test]
async fn active_mutation_can_handoff_to_its_matching_child_jobs() {
    let controller = controller();
    let mutation = controller
        .begin(MutationIntent::workspace("launch background writer"))
        .await
        .unwrap();
    let mut child_jobs = Vec::new();
    for (kind, effect_scope, name) in [
        (crate::JobKind::ReadAgent, EffectScope::ReadOnly, "reader"),
        (
            crate::JobKind::WriteCandidate,
            EffectScope::CandidateOnly,
            "candidate",
        ),
    ] {
        child_jobs.push(
            controller
                .register_job(JobSpec {
                    kind,
                    effect_scope,
                    name: name.into(),
                    limits: JobLimits::default(),
                    parent_operation: Some(mutation.record().operation_id.clone()),
                })
                .await
                .unwrap(),
        );
    }
    let foreign = controller
        .register_job(JobSpec {
            kind: crate::JobKind::ReadAgent,
            effect_scope: EffectScope::ReadOnly,
            name: "foreign reader".into(),
            limits: JobLimits::default(),
            parent_operation: Some(crate::OperationId::new("different-operation")),
        })
        .await
        .unwrap_err();
    assert_eq!(foreign.reason, AdmissionDeniedReason::NotReady);
    let own_job = controller
        .register_job(JobSpec {
            kind: crate::JobKind::Process,
            effect_scope: EffectScope::LiveWriter,
            name: "writer".into(),
            limits: JobLimits::default(),
            parent_operation: Some(mutation.record().operation_id.clone()),
        })
        .await
        .unwrap();
    let denied = controller
        .register_job(JobSpec {
            kind: crate::JobKind::Process,
            effect_scope: EffectScope::LiveWriter,
            name: "second writer".into(),
            limits: JobLimits::default(),
            parent_operation: Some(mutation.record().operation_id.clone()),
        })
        .await
        .unwrap_err();
    assert_eq!(denied.reason, AdmissionDeniedReason::ActiveWriter);

    controller
        .settle(mutation, ExecutionReport::succeeded(None))
        .await;
    child_jobs.push(own_job);
    for child in child_jobs {
        controller
            .seal_job(
                child.job_id,
                JobTerminal {
                    completion: JobCompletion::Cancelled,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await;
    }
}

#[tokio::test]
async fn pipefs_settlement_never_changes_the_authority_version_variant() {
    let controller = InMemoryWorkspaceController::new_pipefs(
        WorkspaceId::new("workspace"),
        "session",
        1,
        false,
        PathBuf::from("/workspace"),
        PathBuf::from("/state"),
    );
    let permit = controller
        .begin(MutationIntent::workspace("change"))
        .await
        .unwrap();
    let outcome = controller
        .settle(permit, ExecutionReport::succeeded(Some("observed".into())))
        .await;
    assert_eq!(outcome.status, SettlementStatus::Durable);
    assert!(matches!(
        controller.binding().version,
        crate::WorkspaceVersion::PipeFs {
            manifest_digest: Some(ref digest),
            ..
        } if digest == "observed"
    ));
}

#[test]
fn job_transition_table_rejects_terminal_reentry() {
    assert!(JobState::Queued.can_transition_to(JobState::Starting));
    assert!(JobState::Starting.can_transition_to(JobState::Running));
    assert!(JobState::Running.can_transition_to(JobState::ReadyToMerge));
    assert!(JobState::ReadyToMerge.can_transition_to(JobState::Merging));
    assert!(JobState::Merging.can_transition_to(JobState::Settling));
    assert!(JobState::Settling.can_transition_to(JobState::Succeeded));
    assert!(!JobState::Succeeded.can_transition_to(JobState::Running));
    assert!(!JobState::Running.can_transition_to(JobState::Queued));
}

#[test]
fn trait_is_object_safe() {
    let concrete = controller();
    let controller: Arc<dyn WorkspaceController> = Arc::new(concrete);
    assert_eq!(controller.status().state, WorkspaceState::Ready);
}
