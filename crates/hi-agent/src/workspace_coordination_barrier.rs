//! Unified lifecycle barriers for workspace rebind, mode switch, and exit.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail, ensure};
use hi_workspace::{BarrierKind, BarrierReceipt, BarrierStatus};

use super::WorkspaceCoordination;

impl WorkspaceCoordination {
    pub(crate) async fn require_barrier(&self, reason: BarrierKind) -> Result<BarrierReceipt> {
        let deadline = Instant::now()
            .checked_add(self.harness.settlement_pending_after)
            .unwrap_or_else(Instant::now);
        self.require_barrier_before(reason, deadline).await
    }

    async fn require_barrier_before(
        &self,
        reason: BarrierKind,
        deadline: Instant,
    ) -> Result<BarrierReceipt> {
        let controller = self.controller();
        let expected = controller.binding();
        let mut changes = controller.subscribe();
        loop {
            ensure!(
                Arc::ptr_eq(&controller, &self.controller()),
                "workspace controller changed while completing the {reason:?} barrier"
            );
            let receipt = controller.barrier(reason, deadline).await;
            ensure!(
                receipt.binding_id == expected.binding_id && receipt.epoch == expected.epoch,
                "workspace barrier changed binding from {}@{} to {}@{}",
                expected.binding_id,
                expected.epoch,
                receipt.binding_id,
                receipt.epoch
            );
            match receipt.status {
                BarrierStatus::Passed => {
                    let current_controller = self.controller();
                    ensure!(
                        Arc::ptr_eq(&controller, &current_controller),
                        "workspace controller changed while completing the {reason:?} barrier"
                    );
                    let current = current_controller.binding();
                    let current_status = current_controller.status();
                    ensure!(
                        current.controller_id == expected.controller_id
                            && current.binding_id == expected.binding_id
                            && current.epoch == expected.epoch,
                        "workspace binding changed while completing the {reason:?} barrier"
                    );
                    ensure!(
                        matches!(
                            current_status.state,
                            hi_workspace::WorkspaceState::Ready
                                | hi_workspace::WorkspaceState::LocalAuditDegraded
                        ),
                        "workspace {reason:?} barrier cannot pass in {:?} state",
                        current_status.state
                    );
                    return Ok(receipt);
                }
                BarrierStatus::RecoveryRequired => bail!(barrier_error(&receipt)),
                BarrierStatus::TimedOut => bail!(barrier_error(&receipt)),
                BarrierStatus::Blocked => {}
            }

            let wait = changes.changed();
            if tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), wait)
                .await
                .is_err()
            {
                continue;
            }
            ensure!(
                changes.has_changed().is_ok(),
                "workspace status subscription closed during the {reason:?} barrier"
            );
        }
    }
}

impl crate::Agent {
    pub fn ensure_workspace_rebind_ready(&self) -> Result<()> {
        self.workspace_coordination.ensure_replace_ready()
    }

    pub fn require_workspace_barrier(
        &self,
        reason: BarrierKind,
    ) -> impl std::future::Future<Output = Result<BarrierReceipt>> + Send + 'static {
        let coordination = self.workspace_coordination.clone();
        async move { coordination.require_barrier(reason).await }
    }
}

fn barrier_error(receipt: &BarrierReceipt) -> String {
    let mut evidence = Vec::new();
    if let Some(operation) = &receipt.active_operation {
        evidence.push(format!("operation {operation}"));
    }
    if !receipt.pending_jobs.is_empty() {
        evidence.push(format!(
            "jobs {}",
            receipt
                .pending_jobs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(recovery) = &receipt.recovery_id {
        evidence.push(format!("recovery {recovery}"));
    }
    let evidence = if evidence.is_empty() {
        String::new()
    } else {
        format!(" ({})", evidence.join("; "))
    };
    format!(
        "workspace {:?} barrier {:?}{evidence}: {}",
        receipt.kind,
        receipt.status,
        receipt
            .detail
            .as_deref()
            .unwrap_or("workspace work is unsettled")
    )
}

#[cfg(test)]
#[path = "workspace_coordination_barrier_tests.rs"]
mod tests;
