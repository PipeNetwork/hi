//! Await durable drive transitions while reusing the synchronous state policy.

use super::*;

#[derive(Clone, PartialEq)]
struct DriveState {
    pause: crate::plan_drive::PlanDrivePause,
    plan_stall: u32,
    plan_evidence: crate::plan_drive::DriveEvidenceLedger,
    goal_stall: u32,
    goal_evidence: crate::plan_drive::DriveEvidenceLedger,
    goal: Option<crate::Goal>,
    pending: bool,
    consumed: bool,
}

impl DriveState {
    fn capture(agent: &crate::Agent) -> Self {
        Self {
            pause: agent.plan_drive_pause,
            plan_stall: agent.plan_drive_stall,
            plan_evidence: agent.plan_drive_evidence.clone(),
            goal_stall: agent.goal_drive_stall,
            goal_evidence: agent.goal_drive_evidence.clone(),
            goal: agent.goals.structured.clone(),
            pending: agent.pending_plan_interruption_resume,
            consumed: agent.turn_consumed_plan_interruption,
        }
    }
    fn restore(self, agent: &mut crate::Agent) {
        agent.plan_drive_pause = self.pause;
        agent.plan_drive_stall = self.plan_stall;
        agent.plan_drive_evidence = self.plan_evidence;
        agent.goal_drive_stall = self.goal_stall;
        agent.goal_drive_evidence = self.goal_evidence;
        agent.goals.structured = self.goal;
        agent.pending_plan_interruption_resume = self.pending;
        agent.turn_consumed_plan_interruption = self.consumed;
        agent.refresh_system_message();
    }
}

impl crate::Agent {
    pub(crate) async fn runtime_drive_transition<T>(
        &mut self,
        transition: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let before = DriveState::capture(self);
        // Prepare the same policy transition with no external writes. There is
        // no await while the session is detached. Dependent execution starts
        // only after the complete resulting drive state is durably recorded.
        let session = self.session.take();
        let result = transition(self);
        self.session = session;
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                before.restore(self);
                return Err(error);
            }
        };
        let after = DriveState::capture(self);
        if before == after {
            return Ok(result);
        }
        let prior = before.clone();
        let persisted_goal = after
            .goal
            .as_ref()
            .map(|goal| self.goal_for_persistence(goal));
        let paused = self.durable_plan_drive_paused();
        let resume_on_input = self.plan_drive_resumes_on_user_input();
        let write = self
            .write_session(move |sink| {
                if prior.pause != after.pause
                    || prior.plan_stall != after.plan_stall
                    || prior.plan_evidence != after.plan_evidence
                {
                    sink.record_plan_drive_state_with_policy(
                        paused,
                        after.plan_stall,
                        resume_on_input,
                        true,
                        &after.plan_evidence.snapshot(),
                    )?;
                }
                if prior.goal_stall != after.goal_stall
                    || prior.goal_evidence != after.goal_evidence
                {
                    sink.record_goal_drive_state(
                        after.goal_stall,
                        true,
                        &after.goal_evidence.snapshot(),
                    )?;
                }
                if prior.goal != after.goal {
                    if let Some(goal) = persisted_goal {
                        sink.record_goal(&goal)?;
                    } else {
                        sink.clear_goal()?;
                    }
                }
                Ok(())
            })
            .await;
        if let Err(error) = write {
            // Several legacy metadata records form this transition. A failed
            // later append can leave an earlier durable prefix; only replay can
            // reconcile that prefix with the restored live state.
            self.session_recovery_pending |= self.has_session_io();
            before.restore(self);
            return Err(error);
        }
        Ok(result)
    }

    pub(crate) async fn begin_drive_turn_async(&mut self, kind: crate::DriveKind) -> Result<()> {
        self.runtime_drive_transition(|agent| agent.begin_drive_turn(kind))
            .await
    }
    pub(crate) async fn pause_plan_drive_until_user_input_async(&mut self) -> Result<bool> {
        self.runtime_drive_transition(Self::pause_plan_drive_until_user_input)
            .await
    }
    pub(crate) async fn settle_plan_interruption_resume_async(
        &mut self,
        success: bool,
    ) -> Result<()> {
        self.runtime_drive_transition(|agent| agent.settle_plan_interruption_resume(success))
            .await
    }
    pub(crate) async fn note_approval_parked_async(
        &mut self,
        ui: &mut dyn crate::Ui,
    ) -> Result<()> {
        self.runtime_drive_transition(|agent| {
            agent.note_approval_parked(ui);
            Ok(())
        })
        .await
    }
    pub(crate) async fn maybe_requeue_goal_second_pass_async(&mut self) -> Result<Option<usize>> {
        if self.task_recovery.exhausted {
            return Ok(None);
        }
        self.runtime_drive_transition(|agent| Ok(agent.maybe_requeue_goal_second_pass()))
            .await
    }
}
