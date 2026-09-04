use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use hi_workspace::{ExecutionDisposition, ExecutionReport, WorkspaceState};

use super::{ActiveMutation, WorkspaceCoordination, settlement_result};
use crate::WorkspaceDurability;

impl WorkspaceCoordination {
    /// Settle only an operation which was already admitted by the interrupted
    /// turn. Unlike `checkpoint`, this never manufactures a final
    /// reconciliation operation when cancellation happened before admission.
    pub(crate) async fn settle_active(
        &self,
        mut durability: Option<Arc<dyn WorkspaceDurability>>,
        execution: ExecutionReport,
    ) -> Result<bool> {
        if self.controller_settles_backend.load(Ordering::Acquire) {
            durability = None;
        }
        let active = self.lock_active()?.take();
        let Some(active) = active else {
            return Ok(false);
        };
        self.settle_owned(active, durability, execution).await?;
        Ok(true)
    }

    /// Wait for a settlement task which already took ownership of the permit
    /// to publish either a durable terminal state or explicit recovery state.
    pub(crate) async fn await_known_settlement(&self, deadline: Duration) -> Result<()> {
        let controller = self.controller();
        let mut status = controller.subscribe();
        let wait = async {
            loop {
                let current = status.borrow().clone();
                if !matches!(
                    current.state,
                    WorkspaceState::Mutating | WorkspaceState::Settling
                ) {
                    return Ok(());
                }
                status.changed().await.map_err(|_| {
                    anyhow!("workspace controller closed while settlement was in flight")
                })?;
            }
        };
        tokio::time::timeout(deadline, wait).await.map_err(|_| {
            anyhow!(
                "workspace settlement did not reach a durable or recovery state within {}ms",
                deadline.as_millis()
            )
        })?
    }

    pub(super) async fn settle_owned(
        &self,
        active: ActiveMutation,
        durability: Option<Arc<dyn WorkspaceDurability>>,
        execution: ExecutionReport,
    ) -> Result<()> {
        let settlement = tokio::spawn(async move {
            let (report, backend_error) = match durability {
                Some(durability) => match durability.checkpoint().await {
                    Ok(()) => (execution, None),
                    Err(error) => {
                        let mut report = execution;
                        let execution_detail = report.detail.take();
                        report.disposition = ExecutionDisposition::Indeterminate;
                        report.content_digest = None;
                        report.detail = Some(match execution_detail {
                            Some(detail) => {
                                format!("{detail}; workspace durability is ambiguous: {error:#}")
                            }
                            None => format!("workspace durability is ambiguous: {error:#}"),
                        });
                        (report, Some(error))
                    }
                },
                None => (execution, None),
            };
            let outcome = active.controller.settle(active.permit, report).await;
            (outcome, backend_error)
        });

        let pending_after = self.harness.settlement_pending_after;
        match tokio::time::timeout(pending_after, settlement).await {
            Ok(Ok((outcome, backend_error))) => settlement_result(outcome, backend_error),
            Ok(Err(error)) => Err(anyhow!("workspace settlement task failed: {error}")),
            Err(_) => bail!(
                "workspace settlement is still pending after {} seconds; recovery continues in the background",
                pending_after.as_secs()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use hi_workspace::{ExecutionDisposition, ExecutionReport, WorkspaceState, WorkspaceVersion};

    use super::WorkspaceCoordination;

    #[tokio::test]
    async fn cancelled_active_operation_is_settled_without_starting_a_replacement() {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state");
        std::fs::create_dir_all(&state).unwrap();
        let subject = WorkspaceCoordination::new_local(root.path(), &state);
        let report = |changed| ExecutionReport {
            disposition: ExecutionDisposition::Cancelled,
            workspace_may_have_changed: changed,
            external_effect_may_have_occurred: false,
            content_digest: changed.then(|| "restored-digest".into()),
            changed_paths: Vec::new(),
            artifacts: Vec::new(),
            detail: Some("process reaped and workspace reconciled".into()),
        };

        assert!(!subject.settle_active(None, report(false)).await.unwrap());
        assert_eq!(subject.status().state, WorkspaceState::Ready);
        assert!(matches!(
            subject.binding().version,
            WorkspaceVersion::Local { generation: 0, .. }
        ));

        subject.begin(None, None).await.unwrap();
        assert!(subject.settle_active(None, report(true)).await.unwrap());
        assert_eq!(subject.status().state, WorkspaceState::Ready);
        assert!(subject.status().active_operation.is_none());
    }
}
