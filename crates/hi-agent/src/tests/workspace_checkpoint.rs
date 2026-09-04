use std::sync::{Arc, Mutex};

use anyhow::Result;
use hi_workspace::{
    InMemoryWorkspaceController, OperationId, WorkspaceBinding, WorkspaceController, WorkspaceState,
};

use super::common::{agent, config};

#[derive(Clone)]
struct StageObservation {
    record: crate::WorkspaceTranscriptExecution,
    active_operation: Option<OperationId>,
    binding: WorkspaceBinding,
}

struct CheckpointDurability {
    controller: Arc<InMemoryWorkspaceController>,
    observations: Arc<Mutex<Vec<StageObservation>>>,
    fail_stage: bool,
}

#[async_trait::async_trait]
impl crate::WorkspaceDurability for CheckpointDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> Result<()> {
        Ok(())
    }

    fn stage_workspace_execution(
        &self,
        record: &crate::WorkspaceTranscriptExecution,
    ) -> Result<()> {
        self.observations.lock().unwrap().push(StageObservation {
            record: record.clone(),
            active_operation: self.controller.status().active_operation,
            binding: self.controller.binding(),
        });
        if self.fail_stage {
            anyhow::bail!("synthetic stage unavailable");
        }
        Ok(())
    }
}

fn pipefs_subject(
    fail_stage: bool,
) -> (
    crate::Agent,
    Arc<InMemoryWorkspaceController>,
    Arc<Mutex<Vec<StageObservation>>>,
) {
    let cfg = config();
    let root = cfg.paths.workspace_root.clone();
    let state = cfg.paths.state_root.clone();
    let mut subject = agent(Vec::new(), cfg);
    let controller = Arc::new(InMemoryWorkspaceController::new_pipefs(
        "checkpoint-workspace",
        "checkpoint-session",
        2,
        true,
        root,
        state,
    ));
    subject
        .install_workspace_controller(controller.clone())
        .unwrap();
    let observations = Arc::new(Mutex::new(Vec::new()));
    subject.set_workspace_durability(Some(Arc::new(CheckpointDurability {
        controller: controller.clone(),
        observations: observations.clone(),
        fail_stage,
    })));
    (subject, controller, observations)
}

#[tokio::test]
async fn synthetic_reconciliation_is_admitted_before_exact_pipefs_staging() {
    let (subject, controller, observations) = pipefs_subject(false);
    let initial_binding = controller.binding();

    subject.checkpoint_durable_workspace().await.unwrap();

    let observations = observations.lock().unwrap();
    assert_eq!(observations.len(), 1);
    let observed = &observations[0];
    assert_eq!(
        observed.active_operation.as_ref(),
        Some(&observed.record.operation_id),
        "the staged record must name the permit active during staging"
    );
    assert_eq!(observed.binding.binding_id, initial_binding.binding_id);
    assert_eq!(observed.binding.epoch, initial_binding.epoch);
    assert_eq!(observed.binding.version, initial_binding.version);
    assert!(observed.record.calls.is_empty());
    assert_eq!(
        observed.record.execution.disposition,
        hi_workspace::ExecutionDisposition::Succeeded
    );
    assert_eq!(controller.status().state, WorkspaceState::Ready);
}

#[tokio::test]
async fn synthetic_reconciliation_stage_failure_cannot_return_success() {
    let (subject, controller, observations) = pipefs_subject(true);

    let error = subject.checkpoint_durable_workspace().await.unwrap_err();

    assert!(
        format!("{error:#}").contains("synthetic stage unavailable"),
        "{error:#}"
    );
    assert_eq!(observations.lock().unwrap().len(), 1);
    assert_eq!(
        controller.status().state,
        WorkspaceState::RecoveryRequired,
        "failed exact staging must leave remote publication recovery-blocked"
    );
}
