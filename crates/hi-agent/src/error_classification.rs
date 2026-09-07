pub(crate) fn local_request_guidance(
    reason: hi_ai::RequestFailureReason,
) -> (&'static str, &'static str) {
    match reason {
        hi_ai::RequestFailureReason::AttemptsExhausted => (
            "request_limit",
            "hi stopped after this model request used its bounded retry allowance; completed tool results were retained — /retry starts a new request",
        ),
        hi_ai::RequestFailureReason::BackoffExhausted => (
            "request_limit",
            "hi stopped after this model request used its recovery wait allowance; completed tool results were retained — /retry starts a new request",
        ),
        hi_ai::RequestFailureReason::DeadlineExceeded => (
            "timeout",
            "this model request reached its configured HTTP deadline; completed tool results were retained — /retry starts a new request",
        ),
    }
}

pub fn error_counts_as_model_issue(err: &anyhow::Error) -> bool {
    let err = crate::TurnFailure::from_error(err).map_or(err, |failure| &failure.original);
    if err
        .downcast_ref::<hi_workspace::AdmissionDenied>()
        .is_some()
    {
        return false;
    }
    !matches!(
        hi_ai::provider_error_kind(err),
        Some(
            hi_ai::ProviderErrorKind::CapacityUnavailable
                | hi_ai::ProviderErrorKind::ModelUnavailable
                | hi_ai::ProviderErrorKind::Outage
                | hi_ai::ProviderErrorKind::QualityRejected
                | hi_ai::ProviderErrorKind::ToolProtocol
                | hi_ai::ProviderErrorKind::PaymentRequired
        )
    )
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn local_limits_keep_their_reason_through_the_settled_error_chain() {
        let execution = hi_ai::RequestExecution::new(hi_ai::RequestExecutionPolicy {
            max_attempts: 0,
            max_backoff: std::time::Duration::ZERO,
            ..Default::default()
        });
        for (original, reason) in [
            (
                anyhow::Error::from(execution.ensure_available().unwrap_err()),
                hi_ai::RequestFailureReason::AttemptsExhausted,
            ),
            (
                execution
                    .backoff(std::time::Duration::from_secs(1))
                    .await
                    .unwrap_err(),
                hi_ai::RequestFailureReason::BackoffExhausted,
            ),
        ] {
            let outcome = crate::TurnOutcome::infrastructure_failure("model", None, vec![]);
            let settled: anyhow::Error =
                crate::TurnFailure::new(original.context("model round"), outcome).into();
            let settled = settled.context("turn execution");
            let details = hi_ai::provider_error_details(&settled).unwrap();
            assert_eq!(details.retryable, Some(false));
            assert_eq!(details.request_failure.as_ref().unwrap().reason, reason);
            let (kind, guidance) = crate::ui::classify_error(&settled);
            assert_eq!(kind, "request_limit");
            assert!(guidance.contains("allowance"));
            assert!(!guidance.contains("rejected"));
            assert!(!guidance.contains("provider route"));
            assert!(!super::error_counts_as_model_issue(&settled));
        }
        // A server-provided string cannot impersonate locally typed evidence.
        let upstream: anyhow::Error =
            hi_ai::ProviderError::new(hi_ai::ProviderErrorKind::Outage, "rejected")
                .with_api_contract(Some("request_attempts_exhausted".into()), Some(false), None)
                .into();
        assert!(
            crate::ui::classify_error(&upstream)
                .1
                .contains("will not succeed unchanged")
        );
    }

    #[test]
    fn settled_failure_preserves_original_provider_guidance_and_health_classification() {
        for kind in [
            hi_ai::ProviderErrorKind::Auth,
            hi_ai::ProviderErrorKind::CapacityUnavailable,
            hi_ai::ProviderErrorKind::Outage,
        ] {
            let original: anyhow::Error =
                hi_ai::ProviderError::new(kind, "provider rejected request").into();
            let guidance = crate::ui::classify_error(&original);
            let model_issue = super::error_counts_as_model_issue(&original);
            let outcome = crate::TurnOutcome::infrastructure_failure("model", None, vec![]);
            let settled: anyhow::Error = crate::TurnFailure::new(original, outcome).into();
            assert_eq!(crate::ui::classify_error(&settled), guidance);
            assert_eq!(super::error_counts_as_model_issue(&settled), model_issue);
        }
    }
}
