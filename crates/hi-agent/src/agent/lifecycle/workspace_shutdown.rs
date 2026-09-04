//! Ordered frontend shutdown for the authoritative workspace.

use anyhow::{Context, Result};
use hi_workspace::BarrierReceipt;

impl crate::Agent {
    /// Stop session-owned work and prove its workspace effects are settled.
    ///
    /// A stopped live writer is durability-pending until the final workspace
    /// image has been reconciled and acknowledged. Keep this ordering in one
    /// place so frontend exit paths cannot publish a clean exit after merely
    /// signalling their children. Callers implementing `--keep-background`
    /// must not use this method or claim its receipt: releasing a deliberate
    /// live writer is explicitly not a barrier-compliant shutdown.
    pub async fn settle_workspace_for_exit(&mut self) -> Result<BarrierReceipt> {
        self.kill_background_processes();
        let process_reap = self
            .ensure_background_processes_quiescent()
            .await
            .context("waiting for background processes to be fully reaped during shutdown");

        // Always request task cancellation even when native process reaping
        // failed. Their lifecycle callbacks preserve any recovery evidence;
        // the error below still prevents this shutdown from being reported as
        // barrier-compliant.
        self.background_task_registry().kill_all().await;
        process_reap?;

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
