use super::common::{IsolatedWorkspace, NullUi, ProviderStep, completion, scripted_agent};
use super::*;
use hi_workspace::WorkspaceController;

fn provider_503() -> ProviderStep {
    ProviderStep::ErrorMessage(
        ProviderErrorKind::ModelUnavailable,
        "API error 503 Service Unavailable: upstream temporarily unavailable".into(),
    )
}

#[tokio::test]
async fn provider_outage_after_failed_verification_is_not_verifier_infrastructure_failure() {
    let workspace = IsolatedWorkspace::new("provider-after-verification-failure");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new("test", "false")]);
    let write = completion(
        vec![Content::ToolCall {
            id: "write".into(),
            name: "write".into(),
            arguments: serde_json::json!({
                "path": "src.rs",
                "content": "fn checked() {}\n"
            })
            .to_string(),
        }],
        1,
        1,
    );
    let done = completion(vec![Content::Text("implemented".into())], 1, 1);
    // One route retry is allowed. The second 503 escapes the normal turn loop,
    // matching an outage that begins when the model is asked to repair a
    // deterministic verification failure.
    let (mut agent, requests) = scripted_agent(
        vec![
            ProviderStep::Completion(write),
            ProviderStep::Completion(done),
            provider_503(),
            provider_503(),
        ],
        cfg,
    );

    let error = agent
        .run_turn("implement src.rs", &mut NullUi)
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("503 Service Unavailable"));
    assert_eq!(requests.lock().unwrap().len(), 4);

    let outcome = &crate::TurnFailure::from_error(&error)
        .expect("Agent returns the reconciled failed-turn receipt")
        .outcome;
    assert_eq!(outcome.status, TurnStatus::Failed);
    assert_eq!(outcome.stop_reason, TurnStopReason::InfrastructureFailure);
    assert_eq!(outcome.verification, VerificationStatus::Failed);
    assert_ne!(
        outcome.verification,
        VerificationStatus::InfrastructureError
    );
    assert!(outcome.verified_workspace_revision.is_none());
    assert_eq!(outcome.exit_code(false), 3);
}

#[test]
fn generic_provider_failure_constructor_does_not_claim_verifier_failure() {
    let outcome = TurnOutcome::infrastructure_failure(
        "pipe/deepseek-v4-flash-0731",
        Some("pipe".into()),
        Vec::new(),
    );

    assert_eq!(outcome.stop_reason, TurnStopReason::InfrastructureFailure);
    assert_eq!(outcome.verification, VerificationStatus::Unverified);
    assert_eq!(outcome.exit_code(false), 3);
}

#[test]
fn workspace_admission_errors_point_at_status_and_recovery() {
    for (state, expected_kind, expected_stop_reason) in [
        (
            hi_workspace::WorkspaceState::Mutating,
            "workspace",
            TurnStopReason::WorkspaceNotReady,
        ),
        (
            hi_workspace::WorkspaceState::RecoveryRequired,
            "recovery",
            TurnStopReason::WorkspaceRecoveryRequired,
        ),
        (
            hi_workspace::WorkspaceState::Incompatible,
            "recovery",
            TurnStopReason::WorkspaceRecoveryRequired,
        ),
    ] {
        let err = anyhow::Error::new(hi_workspace::AdmissionDenied {
            reason: hi_workspace::AdmissionDeniedReason::NotReady,
            state,
            detail: "state is fenced; recovery_id=recovery-1".into(),
        })
        .context("tool batch admission failed");
        let (kind, guidance) = crate::classify_error(&err);
        assert_eq!(kind, expected_kind);
        assert!(guidance.contains("hi workspace status"));
        assert!(guidance.contains("hi workspace recover inspect RECOVERY_ID"));
        assert!(!crate::ui::error_counts_as_model_issue(&err));
        assert_eq!(TurnStopReason::for_error(&err), expected_stop_reason);
    }
}

#[tokio::test]
async fn recovery_admission_cleanup_blocks_without_settling_the_existing_fence() {
    let workspace = IsolatedWorkspace::new("workspace-recovery-admission-cleanup");
    let (mut agent, _) = scripted_agent(Vec::new(), workspace.config());
    let controller = std::sync::Arc::new(hi_workspace::InMemoryWorkspaceController::new_local(
        "recovery-workspace",
        workspace.path(""),
        workspace.path(".hi/state"),
    ));
    agent
        .install_workspace_controller(controller.clone())
        .unwrap();
    let binding = controller.binding();
    let recovery_id = hi_workspace::RecoveryId::new("recovery-1");
    controller
        .require_recovery(hi_workspace::RecoveryRecord {
            schema_version: hi_workspace::WORKSPACE_CONTRACT_SCHEMA_VERSION,
            recovery_id: recovery_id.clone(),
            kind: hi_workspace::RecoveryKind::CrashedWriterJob,
            binding_id: binding.binding_id,
            epoch: binding.epoch,
            operation_id: None,
            job_id: None,
            detail: "writer lifecycle is unknown".into(),
            created_at_ms: 1,
            resolved: false,
        })
        .unwrap();
    let status_before = controller.status();
    let denied: anyhow::Error = hi_workspace::AdmissionDenied {
        reason: hi_workspace::AdmissionDeniedReason::NotReady,
        state: status_before.state,
        detail: status_before.admission_block_detail("workspace is not ready"),
    }
    .into();

    let outcome = agent
        .cleanup_turn(crate::TurnCleanupKind::for_error(&denied))
        .await
        .unwrap()
        .outcome;

    assert_eq!(outcome.status, TurnStatus::Blocked);
    assert_eq!(
        outcome.stop_reason,
        TurnStopReason::WorkspaceRecoveryRequired
    );
    assert_eq!(outcome.exit_code(false), 1);
    assert!(crate::plan_drive::outcome_blocks_automatic_drive(&outcome));
    agent
        .quiesce_after_blocked_workspace_admission()
        .await
        .unwrap();
    let status_after = controller.status();
    assert_eq!(
        status_after.state,
        hi_workspace::WorkspaceState::RecoveryRequired
    );
    assert_eq!(status_after.recovery_id, Some(recovery_id));
    assert_eq!(status_after.sequence, status_before.sequence);
}
