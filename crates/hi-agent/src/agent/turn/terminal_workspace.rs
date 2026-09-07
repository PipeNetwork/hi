//! Check the exact input after every late callback, before admitting a receipt.

use anyhow::Result;

use crate::verify::is_internal_runtime_artifact_path;
use crate::{
    AgentStateSnapshot, ReviewStatus, TaskRecoveryState, TurnFailure, TurnOutcome, TurnStatus,
    TurnStopReason, Ui, VerificationStatus,
};

impl crate::Agent {
    pub(super) async fn reconcile_terminal_workspace(
        &mut self,
        result: &mut Result<TurnOutcome>,
        before: &AgentStateSnapshot,
        body_digest: &str,
        ui: &mut dyn Ui,
    ) -> Result<Option<TaskRecoveryState>> {
        // Cancellation/error cleanup already reconciled or returned explicit
        // pending evidence. Terminal hooks do not run after cancellation, and
        // a second scan could reenter the same blocked worker after its cancel
        // token was cleared by cleanup. Such a receipt cannot claim success.
        if result
            .as_ref()
            .is_ok_and(|outcome| outcome.status == TurnStatus::Cancelled)
            || result.as_ref().is_err_and(|error| {
                TurnFailure::from_error(error).is_some_and(|failure| {
                    failure.settlement_pending || failure.outcome.status == TurnStatus::Cancelled
                })
            })
        {
            return Ok(None);
        }
        self.reconcile_workspace_changes().await?;
        let digest = self.runtime.ledger().workspace_revision();
        let failed = result.is_err();
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(error) => {
                &mut error
                    .downcast_mut::<TurnFailure>()
                    .expect("entry owns a typed failure receipt")
                    .outcome
            }
        };
        self.refresh_terminal_files(outcome);
        let changed = digest != body_digest
            || outcome
                .verified_workspace_revision
                .as_ref()
                .is_some_and(|verified| verified != &digest);
        if changed {
            let mut review = outcome.review;
            super::settlement::reconcile_verified_revision(
                &mut self.report.verify,
                &mut review,
                digest,
                ui,
            );
            if review == ReviewStatus::Unavailable && outcome.review == ReviewStatus::Passed {
                self.report.last_turn_telemetry.review_unavailable_reason =
                    Some("workspace changed in a terminal callback after review".into());
            }
            outcome.review = review;
            outcome.verified_workspace_revision = None;
            if !failed && outcome.status != TurnStatus::Cancelled {
                outcome.verification = VerificationStatus::Unverified;
                if outcome.status == TurnStatus::Completed {
                    outcome.status = TurnStatus::Failed;
                    outcome.stop_reason = TurnStopReason::VerificationUnavailable;
                }
            } else if outcome.verification == VerificationStatus::Passed {
                outcome.verification = VerificationStatus::Unverified;
            }
            // Goal completion was provisional until the final input remained
            // applicable. Restore only that goal, preserving actual task edits
            // and their model/validation recovery history.
            let corrected_goal = self.goals.structured.as_mut().is_some_and(|goal| {
                goal.revoke_unsupported_completion(before.structured_goal.as_ref())
            });
            if corrected_goal {
                self.refresh_system_message();
                let goal = self.goals.structured.clone().expect("goal corrected above");
                self.write_session(move |sink| sink.record_goal(&goal))
                    .await?;
            }
            outcome.leftover = self.goals.leftover_work();
            outcome.plan_leftover = self.goals.plan_leftover_work();
            // Corrective persistence/status callbacks can produce more effects.
            // Once invalidated, this path can never promote the verdict again.
            self.reconcile_workspace_changes().await?;
            self.checkpoint_durable_workspace().await?;
            self.reconcile_workspace_changes().await?;
            self.refresh_terminal_files(outcome);
            self.report.set_outcome(outcome.clone());
            return Ok(None);
        }
        self.report.set_outcome(outcome.clone());
        if failed
            || outcome.status != TurnStatus::Completed
            || outcome.verification != VerificationStatus::Passed
            || self.task_recovery.exhausted
        {
            return Ok(None);
        }
        let mut credit = self.task_recovery.clone();
        if let (Some(before), Some(after)) = (&before.structured_goal, &self.goals.structured)
            && before.objective == after.objective
        {
            for (index, previous) in before.sub_goals.iter().enumerate() {
                if previous.status != crate::goal::GoalStatus::Done
                    && after.sub_goals.get(index).is_some_and(|current| {
                        current.description == previous.description
                            && current.status == crate::goal::GoalStatus::Done
                    })
                {
                    credit.observe_required_effect(format!("goal:{}:{index}", before.objective));
                }
            }
        }
        Ok((credit != self.task_recovery).then_some(credit))
    }

    fn refresh_terminal_files(&mut self, outcome: &mut TurnOutcome) {
        self.workspace
            .last_file_changes
            .retain(|change| !is_internal_runtime_artifact_path(&change.path));
        self.workspace.last_changed_files = self
            .workspace
            .last_file_changes
            .iter()
            .map(|change| change.path.clone())
            .collect();
        outcome.changed_files = self.workspace.last_changed_files.clone();
    }
}
