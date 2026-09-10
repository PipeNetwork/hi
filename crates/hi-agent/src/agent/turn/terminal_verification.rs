//! Exhaustion stops automatic corrections, without discarding the final check.

use crate::Ui;
use crate::domain::VerifyEvidence;
use crate::verify::VerifyOutcome;

use super::state::TurnState;

pub(super) fn protocol_format_exhausted(recovery: &crate::recovery::TaskRecoveryState) -> bool {
    recovery
        .last_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("invalid tool turns exhausted"))
}

pub(super) fn productive_stall_is_leftover(
    recovery: &crate::recovery::TaskRecoveryState,
    turn_had_mutation: bool,
    stationarity_ended: bool,
) -> bool {
    turn_had_mutation && (protocol_format_exhausted(recovery) || stationarity_ended)
}

/// Invalid tool JSON is a model-format failure, not a down provider.
/// Grok-build injects a reminder and starts a new generation; burning the
/// shared physical-send budget on the same malformed call must not become
/// `infrastructure_failure`.
pub(super) fn tool_protocol_allowance_exhausted(error: &anyhow::Error) -> bool {
    let Some(details) = hi_ai::provider_error_details(error) else {
        return false;
    };
    if details.kind == hi_ai::ProviderErrorKind::ToolProtocol {
        return true;
    }
    details.request_failure.as_ref().is_some_and(|failure| {
        failure.reason == hi_ai::RequestFailureReason::AttemptsExhausted
            && failure
                .recent_attempts
                .last()
                .and_then(|attempt| attempt.failure_kind.as_deref())
                == Some("tool_protocol")
    })
}

impl crate::Agent {
    pub(super) async fn retain_terminal_provider_error(
        &mut self,
        error: anyhow::Error,
        turn: &mut TurnState,
        ui: &mut dyn Ui,
    ) -> anyhow::Result<()> {
        if let Some(evidence) =
            hi_ai::provider_error_details(&error).and_then(|error| error.request_failure.as_deref())
        {
            self.report
                .last_turn_telemetry
                .record_wire_audit(serde_json::json!({
                    "kind": "request_failure", "request_failure": evidence,
                }));
        }
        turn.flags.provider_exhausted = true;
        self.task_recovery
            .stop(format!("provider terminated recovery: {error}"));
        // Retain the cause before any fallible settlement operation.
        turn.pending_provider_error = Some(error);
        self.answer_state = crate::recovery::AnswerState::Commentary;
        self.persist_task_recovery_async().await?;
        ui.status("provider requests stopped; settling retained tool work and verification");
        Ok(())
    }

    /// Protocol retries ran out. Settle as no-progress when nothing was
    /// edited; verify retained mutations without branding a provider outage.
    pub(super) async fn settle_invalid_tool_budget(
        &mut self,
        error: anyhow::Error,
        turn: &mut TurnState,
        ui: &mut dyn Ui,
    ) -> anyhow::Result<super::ModelLoopDecision> {
        if let Some(evidence) =
            hi_ai::provider_error_details(&error).and_then(|error| error.request_failure.as_deref())
        {
            self.report
                .last_turn_telemetry
                .record_wire_audit(serde_json::json!({
                    "kind": "request_failure", "request_failure": evidence,
                }));
        }
        self.task_recovery
            .stop(format!("invalid tool turns exhausted: {error}"));
        self.persist_task_recovery_async().await?;
        ui.status(
            "invalid tool turns exhausted; settling without treating it as a provider outage",
        );
        if self
            .runtime
            .ledger()
            .had_mutation_since(turn.turn_ledger_revision)
        {
            Ok(super::ModelLoopDecision::VerifyAfterRecoveryExhaustion)
        } else {
            Ok(super::ModelLoopDecision::Settle(
                crate::TurnStopReason::NoProgress,
            ))
        }
    }

    pub(super) fn recovery_terminal_decision(
        &self,
        turn_revision: u64,
    ) -> super::ModelLoopDecision {
        let mut ledger = self.runtime.ledger();
        let digest = ledger.workspace_revision();
        let unresolved = self.task_recovery.unresolved_validation_status(&digest);
        if (ledger.had_mutation_since(turn_revision)
            || unresolved == Some(crate::recovery::ValidationResult::Deferred))
            && (self.report.verify.digest() != Some(digest.as_str()) || unresolved.is_some())
        {
            super::ModelLoopDecision::VerifyAfterRecoveryExhaustion
        } else {
            super::ModelLoopDecision::Settle(crate::TurnStopReason::NoProgress)
        }
    }

    pub(super) fn terminal_verification_required(&self, turn: &TurnState) -> bool {
        turn.verifier.is_on()
            && (turn.pending_provider_error.is_some()
                || self.recovery_terminal_decision(turn.turn_ledger_revision)
                    == super::ModelLoopDecision::VerifyAfterRecoveryExhaustion)
    }

    /// Store the physical result without the normal review/repair/enrichment
    /// callbacks. Recovery observations have already retained their evidence;
    /// an exhausted allowance remains absorbing even when this check passes.
    pub(super) fn record_terminal_verification(
        &mut self,
        outcome: VerifyOutcome,
        turn: &mut TurnState,
        ui: &mut dyn Ui,
    ) {
        match outcome {
            VerifyOutcome::Passed { revision, digest } => {
                self.report.verify = VerifyEvidence::pass(revision, digest);
                self.runtime.ledger().retain_verification_baseline(revision);
                ui.status("✓ final verification passed; automatic recovery remains stopped");
            }
            VerifyOutcome::Failed { stage, output, .. } => {
                self.report.verify = VerifyEvidence::fail();
                turn.last_verify_attributions = hi_tools::format_structured_failure(
                    &format!("Final verification stage `{}` failed.", stage.name),
                    &output,
                    None,
                )
                .attributions;
                ui.status(&format!(
                    "✗ {} failed for the current workspace; automatic repair remains stopped",
                    stage.name
                ));
            }
            VerifyOutcome::InfrastructureError { stage, output, .. } => {
                self.report.verify = VerifyEvidence::none();
                turn.verification_infrastructure_error = true;
                ui.status(&format!(
                    "final verification unavailable at {}: {output}",
                    stage.name
                ));
            }
            VerifyOutcome::DeferredActiveWriter { stage, detail } => {
                self.report.verify = VerifyEvidence::none();
                turn.verification_deferred_active_writer = true;
                ui.status(&format!(
                    "final verification deferred at {}: {detail}",
                    stage.name
                ));
            }
            VerifyOutcome::Unstable {
                stage,
                changed_files,
                ..
            } => {
                self.report.verify = VerifyEvidence::fail();
                turn.verification_unstable = true;
                ui.status(&format!(
                    "final verification is unstable: {} modified {}",
                    stage.name,
                    changed_files.join(", ")
                ));
            }
            VerifyOutcome::NotRun => {
                // The explicit stage budget stays authoritative. Its earlier
                // result cannot describe edits made after the checked input.
                let current_digest = self.runtime.ledger().workspace_revision();
                if self
                    .task_recovery
                    .unresolved_validation_status(&current_digest)
                    == Some(crate::recovery::ValidationResult::Failed)
                {
                    self.report.verify = VerifyEvidence::fail();
                    ui.status("final verification was not run within the configured limits; the latest applicable check still fails for the current workspace");
                } else {
                    self.report.verify = VerifyEvidence::none();
                    ui.status("final verification was not run within the configured limits; retained edits remain unverified");
                }
            }
            VerifyOutcome::SkippedNoChanges { .. } | VerifyOutcome::SkippedProseOnly { .. } => {}
        }
    }
}

/// A settlement failure cannot replace the provider failure that stopped work.
/// Cancellation retains its private control marker so Entry still owns rollback.
pub(super) fn finish_provider_settlement(
    result: anyhow::Result<crate::TurnOutcome>,
    original: Option<anyhow::Error>,
) -> anyhow::Result<crate::TurnOutcome> {
    match (result, original) {
        (Ok(outcome), Some(original)) => {
            Err(crate::TurnFailure::from_settled_body(original, outcome).into())
        }
        (Err(error), Some(original)) if error.is::<super::entry::TurnCancellationRequested>() => {
            Err(error.context(format!("provider had already stopped: {original:#}")))
        }
        (Err(error), Some(original)) => {
            Err(original.context(format!("terminal settlement failed: {error:#}")))
        }
        (result, None) => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_format_reason_is_not_a_generic_stall() {
        let mut recovery = crate::recovery::TaskRecoveryState::new("task".into(), 3);
        assert!(!super::protocol_format_exhausted(&recovery));
        recovery.stop("invalid tool turns exhausted: invalid tool JSON");
        assert!(super::protocol_format_exhausted(&recovery));
        recovery = crate::recovery::TaskRecoveryState::new("task".into(), 3);
        recovery.stop("automatic recovery exhausted");
        assert!(!super::protocol_format_exhausted(&recovery));
    }

    #[test]
    fn tool_protocol_budget_is_not_a_provider_outage() {
        let protocol: anyhow::Error =
            hi_ai::ProviderError::new(hi_ai::ProviderErrorKind::ToolProtocol, "invalid tool JSON")
                .into();
        assert!(super::tool_protocol_allowance_exhausted(&protocol));
        let outage: anyhow::Error =
            hi_ai::ProviderError::new(hi_ai::ProviderErrorKind::Outage, "upstream unavailable")
                .into();
        assert!(!super::tool_protocol_allowance_exhausted(&outage));
    }

    #[test]
    fn failed_terminal_settlement_retains_provider_cause_without_certifying_body() {
        let original: anyhow::Error = hi_ai::ProviderError::new(
            hi_ai::ProviderErrorKind::Other,
            "physical allowance exhausted",
        )
        .with_api_contract(Some("request_attempts_exhausted".into()), Some(false), None)
        .into();
        let error = finish_provider_settlement(
            Err(anyhow::anyhow!("session append failed")),
            Some(original),
        )
        .unwrap_err();
        assert!(crate::TurnFailure::from_error(&error).is_none());
        assert!(format!("{error:#}").contains("session append failed"));
        assert_eq!(
            hi_ai::provider_error_details(&error)
                .unwrap()
                .code
                .as_deref(),
            Some("request_attempts_exhausted")
        );
    }

    #[test]
    fn cancelled_terminal_settlement_keeps_entry_cancellation_control() {
        let error = finish_provider_settlement(
            Err(super::super::entry::TurnCancellationRequested.into()),
            Some(anyhow::anyhow!("provider stopped")),
        )
        .unwrap_err();
        assert!(error.is::<super::super::entry::TurnCancellationRequested>());
        assert!(format!("{error:#}").contains("provider stopped"));
    }
}
