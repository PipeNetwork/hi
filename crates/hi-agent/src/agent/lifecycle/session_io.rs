//! Runtime persistence awaits owned session commits before publishing state.

use crate::Ui;
use anyhow::{Context, Result};

impl crate::Agent {
    /// Preserve authored goal structure while withholding unsupported Done
    /// transitions from crash-replayable metadata until the terminal receipt.
    pub(crate) fn goal_for_persistence(&self, goal: &crate::Goal) -> crate::Goal {
        let mut persisted = goal.clone();
        if let Some(before) = &self.report.provisional_goal_baseline {
            persisted.revoke_unsupported_completion(before.as_ref());
        }
        persisted
    }
    /// Refuse commands which would reuse state while an accepted append can
    /// still change the authoritative transcript behind this live Agent.
    pub fn ensure_session_reusable(&self) -> Result<()> {
        let guidance = if self.session.is_some() {
            "session persistence remains indeterminate; restart hi, or wait for pending writes to settle before reopening the session from its authoritative transcript"
        } else {
            "session metadata publication remains indeterminate; wait for pending writes to settle, then restart hi before starting another turn"
        };
        anyhow::ensure!(!self.session_recovery_pending, "{guidance}");
        Ok(())
    }

    pub async fn record_workspace_checkpoint_boundary_async(&mut self) -> Result<()> {
        self.write_session(|session| session.record_checkpoints(&[]))
            .await
            .context("persisting the workspace checkpoint-generation boundary")
    }

    pub(crate) async fn persist_async(&mut self) -> Result<()> {
        if self.pending_legacy_goal_budget_migration {
            if let Some(goal) = self.goals.structured.clone() {
                if self.session.is_some() {
                    self.write_session(move |sink| sink.record_goal(&goal))
                        .await
                        .context("persisting normalized legacy goal budget")?;
                    self.pending_legacy_goal_budget_migration = false;
                }
            } else {
                self.pending_legacy_goal_budget_migration = false;
            }
        }
        if self.session.is_some() {
            let start = self.persisted.min(self.messages.len());
            let messages = self.messages.as_slice()[start..].to_vec();
            let usage = self.totals;
            self.write_session(move |sink| sink.record(&messages, usage))
                .await?;
            self.persisted = self.messages.len();
        }
        Ok(())
    }

    pub(crate) async fn persist_durable_boundary_async(&mut self, boundary: &str) -> Result<()> {
        let requires_anchor = self
            .session
            .as_ref()
            .is_some_and(|sink| sink.requires_local_workspace_execution_stage());
        if self.config.execution.is_durable() || requires_anchor {
            self.persist_async().await.with_context(|| {
                format!("durable execution checkpoint failed at {boundary} boundary")
            })?;
        }
        Ok(())
    }

    pub(crate) async fn persist_goal_async(&mut self, ui: &mut dyn Ui) {
        if let Some(goal) = self.goals.structured.clone() {
            let persisted_goal = self.goal_for_persistence(&goal);
            let root = (!self.pipefs_workspace_active()).then(|| self.runtime.root().to_path_buf());
            if self.session.is_none() && root.is_none() {
                return;
            }
            if let Err(error) = self
                .write_session_metadata(move |sink| {
                    sink.record_goal(&persisted_goal)?;
                    // The view reflects the live candidate goal. The saved
                    // snapshot withholds Done until the terminal receipt; this
                    // export happens first so the final input guard can see it.
                    if let Some(root) = root {
                        let _ = goal.export_markdown_to(&root);
                        let _ = crate::goal::scratch::ensure(&root);
                    }
                    Ok(())
                })
                .await
            {
                ui.status(&format!("(couldn't persist goal: {error})"));
            }
        }
    }

    pub(crate) async fn rewind_to_snapshot_durable_with_workspace_rollback_async(
        &mut self,
        len: usize,
        snapshot: &crate::AgentStateSnapshot,
        workspace_rolled_back: bool,
    ) -> Result<()> {
        let prepared = self.prepare_snapshot_rewind(len, snapshot, workspace_rolled_back);
        let durable = prepared.clone();
        self.write_session(move |session| durable.record(session))
            .await?;
        prepared.apply(self);
        Ok(())
    }
}
