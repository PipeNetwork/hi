//! Failed-turn result after Agent-owned cleanup has completed or been fenced.

use crate::TurnOutcome;

/// Every error returned by `run_turn` carries this receipt. Frontends render
/// it; they must not run a second cleanup based on mutable Agent report state.
#[derive(Debug)]
pub struct TurnFailure {
    pub outcome: TurnOutcome,
    pub original: anyhow::Error,
    pub cleanup_diagnostics: Vec<String>,
    /// Accepted settlement remains owned in the background, or durable
    /// recovery evidence prevents reuse until its ambiguity is resolved.
    pub settlement_pending: bool,
    body_settled: bool,
}

impl TurnFailure {
    pub fn new(original: anyhow::Error, outcome: TurnOutcome) -> Self {
        Self {
            outcome,
            original,
            cleanup_diagnostics: Vec::new(),
            settlement_pending: false,
            body_settled: false,
        }
    }

    /// Only core's final exit can certify that its retained work was settled.
    pub(crate) fn from_settled_body(original: anyhow::Error, outcome: TurnOutcome) -> Self {
        let mut failure = Self::new(original, outcome);
        failure.body_settled = true;
        failure
    }

    pub(crate) fn body_settled(&self) -> bool {
        self.body_settled
    }

    pub(crate) fn invalidate_terminal_verification(&mut self) {
        if self.outcome.verification == crate::VerificationStatus::Passed {
            self.outcome.verification = crate::VerificationStatus::Unverified;
            self.outcome.verified_workspace_revision = None;
        }
    }

    pub fn from_error(error: &anyhow::Error) -> Option<&Self> {
        error.downcast_ref::<Self>()
    }
}

impl std::fmt::Display for TurnFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("turn failed")?;
        for diagnostic in &self.cleanup_diagnostics {
            write!(formatter, "; cleanup: {diagnostic}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TurnFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.original.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_final_reconciliation_revokes_pass_without_replacing_provider_failure() {
        let original: anyhow::Error =
            hi_ai::ProviderError::new(hi_ai::ProviderErrorKind::Outage, "provider unavailable")
                .into();
        let mut outcome = TurnOutcome::infrastructure_failure("model", None, Vec::new());
        outcome.verification = crate::VerificationStatus::Passed;
        outcome.verified_workspace_revision = Some("checked-input".into());
        let mut failure = TurnFailure::from_settled_body(original, outcome);
        failure.invalidate_terminal_verification();
        failure
            .cleanup_diagnostics
            .push("final reconciliation deadline exceeded".into());
        assert_eq!(failure.outcome.status, crate::TurnStatus::Failed);
        assert_eq!(
            failure.outcome.stop_reason,
            crate::TurnStopReason::InfrastructureFailure
        );
        assert_eq!(
            failure.outcome.verification,
            crate::VerificationStatus::Unverified
        );
        assert_eq!(failure.outcome.verified_workspace_revision, None);
        let error: anyhow::Error = failure.into();
        assert_eq!(
            hi_ai::provider_error_details(&error).unwrap().kind,
            hi_ai::ProviderErrorKind::Outage
        );
        assert!(format!("{error:#}").contains("final reconciliation deadline exceeded"));
    }

    #[test]
    fn outer_error_chain_renders_original_once_and_keeps_provider_details() {
        let original: anyhow::Error = hi_ai::ProviderError::new(
            hi_ai::ProviderErrorKind::Other,
            "physical attempt limit reached",
        )
        .with_api_contract(Some("request_attempts_exhausted".into()), Some(false), None)
        .into();
        let outcome = TurnOutcome::infrastructure_failure("model", None, Vec::new());
        let mut failure = TurnFailure::new(original.context("model request"), outcome);
        assert!(!failure.body_settled());
        failure.cleanup_diagnostics.push("late diagnostic".into());
        let error: anyhow::Error = failure.into();
        let rendered = format!("{error:#}");
        assert_eq!(
            rendered.matches("physical attempt limit reached").count(),
            1
        );
        assert_eq!(rendered.matches("model request").count(), 1);
        assert!(rendered.contains("cleanup: late diagnostic"));
        assert_eq!(
            hi_ai::provider_error_details(&error)
                .unwrap()
                .code
                .as_deref(),
            Some("request_attempts_exhausted")
        );
    }
}
