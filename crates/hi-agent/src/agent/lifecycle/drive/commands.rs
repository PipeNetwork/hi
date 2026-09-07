//! Explicit durable plan pause and resume transitions.

use super::*;

impl crate::Agent {
    pub fn set_plan_drive_paused(&mut self, paused: bool) {
        let _ = self.try_set_plan_drive_paused(paused);
    }

    /// Persist an explicit manual pause/unpause, reverting the live state when
    /// the session sink rejects the transition.
    pub fn try_set_plan_drive_paused(&mut self, paused: bool) -> Result<bool> {
        let pause = if paused {
            crate::plan_drive::PlanDrivePause::Manual
        } else {
            crate::plan_drive::PlanDrivePause::Running
        };
        if self.plan_drive_pause == pause
            && !self.pending_plan_interruption_resume
            && !self.turn_consumed_plan_interruption
        {
            return Ok(false);
        }
        let previous_pause = self.plan_drive_pause;
        let previous_pending = self.pending_plan_interruption_resume;
        let previous_consumed = self.turn_consumed_plan_interruption;
        self.plan_drive_pause = pause;
        self.pending_plan_interruption_resume = false;
        self.turn_consumed_plan_interruption = false;
        if let Err(error) = self.try_persist_plan_drive_evidence_delta(false, &[]) {
            self.plan_drive_pause = previous_pause;
            self.pending_plan_interruption_resume = previous_pending;
            self.turn_consumed_plan_interruption = previous_consumed;
            return Err(error);
        }
        Ok(true)
    }

    /// Resume explicit pause/park state in one durable record so restart can
    /// never observe `paused=false` with the old parked stall ledger.
    pub fn resume_plan_drive(&mut self) -> Result<bool> {
        self.restart_task_recovery()?;
        let previous_pause = self.plan_drive_pause;
        let previous_stall = self.plan_drive_stall;
        let previous_evidence = self.plan_drive_evidence.clone();
        let previous_pending = self.pending_plan_interruption_resume;
        let previous_consumed = self.turn_consumed_plan_interruption;
        let reset_evidence = !self.plan_drive_evidence.is_empty();
        let changed = self.durable_plan_drive_paused()
            || self.plan_drive_stall != 0
            || reset_evidence
            || previous_pending
            || previous_consumed;
        if !changed {
            return Ok(false);
        }
        self.plan_drive_pause = crate::plan_drive::PlanDrivePause::Running;
        self.plan_drive_stall = 0;
        self.plan_drive_evidence.clear();
        self.pending_plan_interruption_resume = false;
        self.turn_consumed_plan_interruption = false;
        if let Err(error) = self.try_persist_plan_drive_evidence_delta(reset_evidence, &[]) {
            self.plan_drive_pause = previous_pause;
            self.plan_drive_stall = previous_stall;
            self.plan_drive_evidence = previous_evidence;
            self.pending_plan_interruption_resume = previous_pending;
            self.turn_consumed_plan_interruption = previous_consumed;
            return Err(error);
        }
        Ok(true)
    }
}
