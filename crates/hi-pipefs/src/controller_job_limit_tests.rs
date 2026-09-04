use hi_workspace::{JobRegistryLimits, WorkspaceController};

use super::*;

#[tokio::test]
async fn preparation_limit_failure_rejects_pipefs_job_admission_without_stranding_it() {
    let (_temporary, source, session, _server) = subject(false).await;
    let binding = source.binding();
    let controller = PipeFsWorkspaceController::new_with_job_limits(
        source.inner.workspace.clone(),
        session,
        PipeFsControllerConfig {
            workspace_id: binding.workspace_id,
            session_id: "session-1".into(),
            writer_protocol: 2,
            causal_commit_available: true,
            writes_available: true,
            workspace_root: binding.workspace_root,
            state_root: binding.state_root,
            epoch: binding.epoch.saturating_add(1),
            allow_protocol_one_writes: false,
        },
        JobRegistryLimits {
            max_preparations: 1,
            max_active_jobs: 4,
        },
    )
    .await;
    let candidate = |name: &str| JobSpec {
        kind: JobKind::WriteCandidate,
        effect_scope: EffectScope::CandidateOnly,
        name: name.into(),
        limits: JobLimits::default(),
        parent_operation: None,
    };

    let first = controller.register_job(candidate("first")).await.unwrap();
    let error = controller
        .register_job(candidate("second"))
        .await
        .unwrap_err();
    assert!(
        error
            .detail
            .contains("candidate preparation limit reached (1)")
    );
    assert_eq!(controller.status().active_jobs, vec![first.job_id.clone()]);

    assert_eq!(
        controller
            .seal_job(
                first.job_id,
                JobTerminal {
                    completion: JobCompletion::Failed,
                    detail: None,
                    artifacts: Vec::new(),
                },
            )
            .await
            .status,
        JobSealStatus::Sealed
    );
    controller
        .register_job(candidate("after-release"))
        .await
        .expect("failed admission must release its preparation slot");
}
