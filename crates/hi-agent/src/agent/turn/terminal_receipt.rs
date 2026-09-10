//! The outer turn owner publishes exactly one diagnostic outcome.

use crate::{TurnFailure, TurnOutcome, TurnStatus, TurnStopReason};
use anyhow::Result;
use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventKind, RunEvent, SemanticActivity,
};

impl crate::Agent {
    pub(super) async fn publish_terminal_receipt(
        &mut self,
        result: Result<TurnOutcome>,
        deadline: tokio::time::Instant,
        recovery_credit: Option<crate::TaskRecoveryState>,
    ) -> Result<TurnOutcome> {
        let result = result.map_err(|error| {
            let mut failure = error
                .downcast::<TurnFailure>()
                .expect("entry owns a typed failure receipt");
            if failure.outcome.status == TurnStatus::Completed {
                failure.outcome.status = TurnStatus::Failed;
                failure.outcome.stop_reason = TurnStopReason::InfrastructureFailure;
            }
            failure.settlement_pending |= self.session_recovery_pending;
            anyhow::Error::from(failure)
        });
        let outcome = match &result {
            Ok(outcome) => outcome.clone(),
            Err(error) => TurnFailure::from_error(error)
                .expect("entry owns a typed failure receipt")
                .outcome
                .clone(),
        };
        self.report.set_outcome(outcome.clone());
        let reason = self
            .report
            .last_turn_telemetry
            .review_unavailable_reason
            .clone();
        // Accepted work remains owned if the deadline drops this waiter. Never
        // append a second, contradictory outcome after that ambiguity.
        let committed_recovery = recovery_credit.clone();
        let settled_goal = (outcome.status == TurnStatus::Completed
            && outcome.verification == crate::VerificationStatus::Passed)
            .then(|| self.goals.structured.clone())
            .flatten()
            .filter(|goal| self.goal_for_persistence(goal) != *goal);
        let write = self.write_session(move |session| {
            session.record_turn_settlement(
                &outcome,
                reason.as_deref(),
                recovery_credit.as_ref(),
                settled_goal.as_ref(),
            )
        });
        let barrier = self.session_barrier();
        let publication = tokio::time::timeout_at(deadline, async move {
            write.await?;
            barrier.await
        })
        .await;
        let (error, pending) = match publication {
            Ok(Ok(())) => {
                if let Some(recovery) = committed_recovery {
                    self.task_recovery = recovery;
                }
                return result;
            }
            Ok(Err(error)) => (error.context("persisting terminal turn receipt"), false),
            Err(_) => (
                anyhow::anyhow!(
                    "terminal turn receipt exceeded the shared settlement deadline; accepted writes remain owned"
                ),
                true,
            ),
        };
        let mut failure = match result {
            Ok(mut outcome) => {
                outcome.status = TurnStatus::Failed;
                outcome.stop_reason = TurnStopReason::InfrastructureFailure;
                TurnFailure::new(error, outcome)
            }
            Err(original) => {
                let mut failure = original
                    .downcast::<TurnFailure>()
                    .expect("entry owns a typed failure receipt");
                failure.cleanup_diagnostics.push(format!("{error:#}"));
                failure
            }
        };
        failure.settlement_pending |= pending;
        self.session_recovery_pending |= pending && self.has_session_io();
        if failure.outcome.status == TurnStatus::Completed {
            failure.outcome.status = TurnStatus::Failed;
            failure.outcome.stop_reason = TurnStopReason::InfrastructureFailure;
        }
        if let Some(before) = &self.report.provisional_goal_baseline
            && self
                .goals
                .structured
                .as_mut()
                .is_some_and(|goal| goal.revoke_unsupported_completion(before.as_ref()))
        {
            // The pre-receipt goal record is already conservative. Keep the
            // live cursor conservative too if the receipt was not acknowledged,
            // without erasing authored steps or unrelated recovery progress.
            self.refresh_system_message();
            failure.outcome.leftover = self.goals.leftover_work();
            failure.outcome.plan_leftover = self.goals.plan_leftover_work();
        }
        self.report.set_outcome(failure.outcome.clone());
        Err(failure.into())
    }
    pub(super) fn emit_terminal_result_event(
        &mut self,
        ui: &mut dyn crate::Ui,
        result: &Result<TurnOutcome>,
        event_turn: u32,
    ) {
        let outcome = result.as_ref().ok().or_else(|| {
            result
                .as_ref()
                .err()
                .and_then(TurnFailure::from_error)
                .map(|failure| &failure.outcome)
        });
        let (event_kind, state, verb) = match outcome {
            Some(outcome) => match outcome.status {
                TurnStatus::Completed => (
                    EventKind::RunCompleted,
                    ActivityState::Succeeded,
                    ActivityVerb::Complete,
                ),
                TurnStatus::Cancelled => (
                    EventKind::RunCancelled,
                    ActivityState::Cancelled,
                    ActivityVerb::Cancel,
                ),
                TurnStatus::Failed => (
                    EventKind::RunFailed,
                    ActivityState::Failed,
                    ActivityVerb::Fail,
                ),
                TurnStatus::Blocked => (
                    EventKind::RunCompleted,
                    ActivityState::Failed,
                    ActivityVerb::Complete,
                ),
            },
            None => (
                EventKind::RunFailed,
                ActivityState::Failed,
                ActivityVerb::Fail,
            ),
        };
        let title = match state {
            ActivityState::Succeeded => "Run finished",
            ActivityState::Cancelled => "Run cancelled",
            _ => "Run failed",
        };
        let mut run_event = RunEvent::new(
            event_kind,
            self.event_context(),
            SemanticActivity {
                verb,
                object: ActivityObject::Run,
                state,
                group_key: format!("run:turn:{event_turn}"),
                title: title.into(),
                detail: None,
                refs: Vec::new(),
                progress: None,
            },
        );
        if let Some(outcome) = outcome {
            run_event = run_event.with_field("status", serde_json::json!(outcome.status));
            run_event = run_event.with_field("stop_reason", serde_json::json!(outcome.stop_reason));
            if !outcome.changed_files.is_empty() {
                ui.semantic_event(hi_events::RunEvent::new(
                    hi_events::EventKind::GitChanged,
                    self.event_context(),
                    hi_events::SemanticActivity {
                        verb: hi_events::ActivityVerb::Change,
                        object: hi_events::ActivityObject::Git,
                        state: hi_events::ActivityState::Succeeded,
                        group_key: format!("workspace:turn:{event_turn}"),
                        title: "workspace changed".into(),
                        detail: Some(format!("{} file(s) changed", outcome.changed_files.len())),
                        refs: Vec::new(),
                        progress: None,
                    },
                ));
            }
        }
        ui.semantic_event(run_event);
    }
}
