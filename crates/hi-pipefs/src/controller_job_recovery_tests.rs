use std::time::Instant;

use super::*;
use hi_workspace::{BarrierKind, BarrierStatus, JobState};

#[tokio::test]
async fn job_recovery_is_addressable_and_unblocks_pipefs_barriers() {
    let (_temporary, controller, _session, _server) = subject(false).await;
    let job = controller
        .register_job(JobSpec {
            kind: JobKind::WriteCandidate,
            effect_scope: EffectScope::CandidateOnly,
            name: "candidate requiring recovery".into(),
            limits: JobLimits::default(),
            parent_operation: None,
        })
        .await
        .unwrap();
    let sealed = controller
        .seal_job(
            job.job_id,
            JobTerminal {
                completion: JobCompletion::RecoveryRequired,
                detail: Some("candidate receipt was ambiguous".into()),
                artifacts: Vec::new(),
            },
        )
        .await;
    assert_eq!(sealed.state, Some(JobState::RecoveryRequired));
    let recovery_id = sealed.recovery_id.unwrap();
    let blocked = controller.barrier(BarrierKind::Exit, Instant::now()).await;
    assert_eq!(blocked.status, BarrierStatus::RecoveryRequired);
    assert_eq!(blocked.recovery_id.as_ref(), Some(&recovery_id));

    let recovered = controller.reconcile(recovery_id.clone()).await;
    assert_eq!(recovered.status, RecoveryStatus::Recovered);
    assert_eq!(
        controller
            .barrier(BarrierKind::Exit, Instant::now())
            .await
            .status,
        BarrierStatus::Passed
    );
    assert_eq!(
        controller.reconcile(recovery_id).await.status,
        RecoveryStatus::Recovered
    );
}
