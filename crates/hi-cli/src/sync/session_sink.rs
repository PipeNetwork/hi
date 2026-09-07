//! Local-first session records and their remote mirror.

use super::*;

impl SessionSink for SyncSession {
    fn id(&self) -> Option<String> {
        self.local.id()
    }

    fn requires_local_workspace_execution_stage(&self) -> bool {
        true
    }

    fn record(&mut self, messages: &[Message], usage: Usage) -> Result<()> {
        self.local.record(messages, usage)?;
        self.remote.observe_messages(messages);
        self.remote.observe_context_used(usage.context_occupancy);
        self.remote.reconcile_message_prefix(self.local.path())
    }

    fn stage_workspace_execution(
        &mut self,
        record: &hi_agent::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        self.remote.stage_workspace_execution(record)
    }

    fn stage_local_workspace_execution(
        &mut self,
        record: &hi_agent::WorkspaceTranscriptExecution,
        visible_on_resume: bool,
    ) -> Result<()> {
        self.local
            .stage_local_workspace_execution(record, visible_on_resume)
    }

    fn settle_local_workspace_execution(
        &mut self,
        operation_id: &hi_workspace::OperationId,
    ) -> Result<()> {
        self.local.settle_local_workspace_execution(operation_id)
    }

    fn record_model_context(&mut self, model: &str, context_window: Option<u32>) {
        self.remote.set_model_context(model, context_window);
    }

    fn record_compaction(&mut self, messages: &[Message]) -> Result<()> {
        self.local.record_compaction(messages)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_state_replacement(
        &mut self,
        messages: &[Message],
        goal: Option<&hi_agent::Goal>,
        decisions: &hi_agent::DecisionLog,
        plan: &[hi_agent::PlanStep],
    ) -> Result<()> {
        self.local
            .record_state_replacement(messages, goal, decisions, plan)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_checkpoints(&mut self, refs: &[String]) -> Result<()> {
        self.local.record_checkpoints(refs)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_pipefs_mode(&mut self, enabled: bool) -> Result<()> {
        self.local.record_pipefs_mode(enabled)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_goal(&mut self, goal: &hi_agent::Goal) -> Result<()> {
        self.local.record_goal(goal)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn clear_goal(&mut self) -> Result<()> {
        self.local.clear_goal()?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan(&mut self, plan: &[hi_agent::PlanStep]) -> Result<()> {
        self.local.record_plan(plan)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn clear_plan(&mut self) -> Result<()> {
        self.local.clear_plan()?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan_drive(&mut self, paused: bool, stall: u32) -> Result<()> {
        self.record_plan_drive_state(paused, stall, false, &[])
    }

    fn record_plan_drive_state(
        &mut self,
        paused: bool,
        stall: u32,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.local
            .record_plan_drive_state(paused, stall, evidence_reset, evidence_add)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan_drive_state_with_policy(
        &mut self,
        paused: bool,
        stall: u32,
        resume_on_user_input: bool,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.local.record_plan_drive_state_with_policy(
            paused,
            stall,
            resume_on_user_input,
            evidence_reset,
            evidence_add,
        )?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_plan_approval_parked(&mut self, parked: bool) -> Result<()> {
        self.local.record_plan_approval_parked(parked)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_task_recovery(&mut self, state: &hi_agent::TaskRecoveryState) -> Result<()> {
        self.local.record_task_recovery(state)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_goal_drive(&mut self, stall: u32) -> Result<()> {
        self.record_goal_drive_state(stall, false, &[])
    }

    fn record_goal_drive_state(
        &mut self,
        stall: u32,
        evidence_reset: bool,
        evidence_add: &[String],
    ) -> Result<()> {
        self.local
            .record_goal_drive_state(stall, evidence_reset, evidence_add)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_decisions(&mut self, decisions: &hi_agent::DecisionLog) -> Result<()> {
        self.local.record_decisions(decisions)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_turn_outcome(
        &mut self,
        outcome: &hi_agent::TurnOutcome,
        review_unavailable_reason: Option<&str>,
    ) -> Result<()> {
        self.local
            .record_turn_outcome(outcome, review_unavailable_reason)?;
        self.reconcile_best_effort();
        Ok(())
    }

    fn record_turn_settlement(
        &mut self,
        outcome: &hi_agent::TurnOutcome,
        review_unavailable_reason: Option<&str>,
        task_recovery: Option<&hi_agent::TaskRecoveryState>,
        settled_goal: Option<&hi_agent::Goal>,
    ) -> Result<()> {
        self.local.record_turn_settlement(
            outcome,
            review_unavailable_reason,
            task_recovery,
            settled_goal,
        )?;
        self.reconcile_best_effort();
        Ok(())
    }
}
