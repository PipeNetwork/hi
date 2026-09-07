//! Runtime context replacement waits for its durable session boundary.
use super::*;

impl crate::Agent {
    pub(crate) async fn replace_history_with_compaction_async(
        &mut self,
        messages: Vec<hi_ai::Message>,
    ) -> Result<()> {
        let record = messages.clone();
        self.write_session(move |sink| sink.record_compaction(&record))
            .await?;
        self.messages.replace_all(messages);
        self.persisted = self.messages.len();
        Ok(())
    }

    pub(super) async fn wipe_conversation_keep_identity_async(
        &mut self,
        reset_drive: bool,
    ) -> Result<()> {
        self.replace_history_with_compaction_async(vec![self.system_message()])
            .await?;
        self.token_budget.advance_window();
        self.runtime.invalidate_context_after_compaction();
        self.report.context_used = 0;
        self.token_budget
            .begin_turn(0, self.config.routing.context_window);
        if reset_drive {
            self.runtime_drive_transition(|agent| {
                agent.reset_goal_drive_stall();
                agent.reset_plan_drive_stall();
                Ok(())
            })
            .await?;
        }
        Ok(())
    }

    pub(crate) async fn apply_fresh_window_async(
        &mut self,
        ui: &mut dyn Ui,
        current_task: Option<&str>,
    ) -> Result<()> {
        let task = current_task
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
            .or_else(|| self.current_window_task());
        self.wipe_conversation_keep_identity_async(true).await?;
        if let Some(task) = task {
            self.messages.push_user(wrap_current_task(
                self.volatile_context_block().as_deref(),
                &task,
            ));
        }
        ui.status("fresh context window — conversation dropped, goal/decisions kept");
        Ok(())
    }

    pub(crate) async fn apply_plan_recovery_window_async(&mut self, ui: &mut dyn Ui) -> Result<()> {
        self.wipe_conversation_keep_identity_async(false).await?;
        ui.status("fresh plan-recovery context — prior tool output dropped; plan, decisions, and no-progress evidence kept as fingerprints");
        Ok(())
    }

    pub(crate) async fn finish_fresh_window_compaction_async(&mut self) -> Result<()> {
        self.runtime_drive_transition(|agent| {
            agent.finish_fresh_window_compaction();
            Ok(())
        })
        .await
    }
}
