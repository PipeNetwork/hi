use super::common::{IsolatedWorkspace, NullUi, ProviderStep, completion, scripted_agent};
use super::*;

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
    assert!(error.to_string().contains("503 Service Unavailable"));
    assert_eq!(requests.lock().unwrap().len(), 4);

    let outcome = agent.finalize_failed_turn();
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
