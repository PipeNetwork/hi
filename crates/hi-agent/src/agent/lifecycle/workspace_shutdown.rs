//! Ordered frontend shutdown for the authoritative workspace.

use anyhow::{Context, Result};
use hi_workspace::BarrierReceipt;

impl crate::Agent {
    /// Reap work owned by this process without attempting admission, a
    /// checkpoint, or a clean barrier against a pre-existing controller fence.
    /// Used when a turn was blocked before it acquired a workspace permit.
    pub async fn quiesce_after_blocked_workspace_admission(&mut self) -> Result<()> {
        self.kill_background_processes();
        let process_reap = self
            .ensure_background_processes_quiescent()
            .await
            .context("waiting for background processes to be fully reaped during shutdown");

        // Request task cancellation even if native process reaping failed.
        // These registries contain only work started by this Agent instance;
        // durable pre-restart recovery evidence lives in the controller and is
        // intentionally untouched.
        self.background_task_registry().kill_all().await;
        process_reap
    }

    /// Stop session-owned work and prove its workspace effects are settled.
    ///
    /// A stopped live writer is durability-pending until the final workspace
    /// image has been reconciled and acknowledged. Keep this ordering in one
    /// place so frontend exit paths cannot publish a clean exit after merely
    /// signalling their children. `--keep-background` uses a distinct sequence:
    /// settle terminal owned jobs, durably orphan requested processes that are
    /// still running, and then require the same exit barrier.
    pub async fn settle_workspace_for_exit(&mut self) -> Result<BarrierReceipt> {
        self.quiesce_after_blocked_workspace_admission().await?;

        self.reconcile_workspace_changes()
            .await
            .context("reconciling final workspace bytes during shutdown")?;
        self.checkpoint_durable_workspace()
            .await
            .context("settling final workspace effects during shutdown")?;
        self.require_workspace_barrier(hi_workspace::BarrierKind::Exit)
            .await
            .context("waiting for the unified workspace exit barrier")
    }
}
