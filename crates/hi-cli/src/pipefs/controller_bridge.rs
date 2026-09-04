use std::future::Future;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use hi_control::{
    ControlStore, OperationReplayClass, WorkspaceBindingRecord, WorkspaceOperationRecord,
    WorkspaceProjectionJournal, WorkspaceRecoveryRecord,
};
use hi_pipefs::{CausalOperationReceipt, PipeFsWorkspaceController};
use hi_workspace::{
    BindingId, OperationId, RecoveryKind, RecoveryRecord, ReplayClass, WorkspaceAuthority,
    WorkspaceBinding, WorkspaceController, WorkspaceVersion, restart_operation_recovery_id,
};

use super::{PipeFsDurability, RemoteSessionSink, lease_monitor, refresh_pipefs_lease};

pub(super) struct PreparedController {
    controller: Arc<dyn WorkspaceController>,
    durability: Arc<PipeFsDurability>,
}

impl PreparedController {
    pub(super) fn install(self, agent: &mut hi_agent::Agent) -> Result<Arc<PipeFsDurability>> {
        agent.install_workspace_controller(self.controller)?;
        agent.set_workspace_durability(Some(self.durability.clone()));
        Ok(self.durability)
    }
}

pub(super) fn prepare(
    agent: &hi_agent::Agent,
    workspace: hi_pipefs::PipeFsWorkspace,
    sync: Arc<RemoteSessionSink>,
    activation: &hi_pipefs::Activation,
) -> impl Future<Output = Result<PreparedController>> + Send + 'static {
    let minimum_epoch = agent.workspace_controller_binding().epoch.saturating_add(1);
    let harness = agent.harness_settings().clone();
    let background_processes = agent.background_process_registry();
    let foreground_processes = agent.foreground_process_registry();
    let activation = activation.clone();
    async move {
        let durability = lease_monitor::build_durability(
            workspace.clone(),
            sync,
            background_processes,
            foreground_processes,
        );
        let controller = build_controller(
            minimum_epoch,
            workspace,
            durability.clone(),
            &activation,
            &harness,
        )
        .await?;
        Ok(PreparedController {
            controller,
            durability,
        })
    }
}

pub(super) async fn retry(
    agent: &mut hi_agent::Agent,
    workspace: &hi_pipefs::PipeFsWorkspace,
    durability: Option<&Arc<PipeFsDurability>>,
) -> Result<Option<String>> {
    if agent.workspace_controller_status().recovery_id.is_none() {
        return Ok(None);
    }
    agent
        .acknowledge_workspace_recovery()
        .await
        .context("retrying native PipeFS workspace settlement")?;
    if let Some(durability) = durability {
        durability.clear_failure();
    }
    Ok(Some(workspace.status().await.to_string()))
}

async fn build_controller(
    minimum_epoch: u64,
    workspace: hi_pipefs::PipeFsWorkspace,
    session: Arc<PipeFsDurability>,
    activation: &hi_pipefs::Activation,
    harness: &hi_workspace::ResolvedHarnessSettings,
) -> Result<Arc<dyn WorkspaceController>> {
    let session_id = session.sync.session_id().to_owned();
    let store = ControlStore::open_for_state(&activation.state_root)
        .context("opening the PipeFS control journal")?;
    let historical = store.unsettled_pipefs_bindings(&session_id)?;
    let evidence = workspace.persisted_operation_recovery_evidence().await;
    let epoch = store
        .latest_pipefs_binding(&session_id)?
        .map_or(minimum_epoch, |binding| {
            minimum_epoch.max(binding.epoch.saturating_add(1))
        });
    let workspace_id = uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        activation.workspace_root.to_string_lossy().as_bytes(),
    )
    .to_string();
    let bridge: Arc<dyn hi_pipefs::PipeFsSessionBridge> = session;
    let raw = Arc::new(
        PipeFsWorkspaceController::new_with_job_limits(
            workspace,
            bridge,
            hi_pipefs::PipeFsControllerConfig {
                workspace_id: workspace_id.into(),
                session_id,
                writer_protocol: activation.writer_protocol,
                causal_commit_available: activation.causal_commit_available
                    && harness.features.pipefs_causal_commit_v1,
                writes_available: activation.writes_available,
                workspace_root: activation.workspace_root.clone(),
                state_root: activation.state_root.clone(),
                epoch,
                // This binary implements the legacy CAS + deterministic
                // transcript-flush fallback; protocol-1 clients themselves
                // remain unable to mutate protocol-2-only sessions.
                allow_protocol_one_writes: true,
            },
            hi_workspace::JobRegistryLimits {
                max_preparations: harness.jobs.max_preparations,
                max_active_jobs: harness.jobs.max_active,
            },
        )
        .await,
    );
    seed_restart_recoveries(&raw, &store, &historical, evidence.as_ref())?;
    let inner: Arc<dyn WorkspaceController> = raw;
    Ok(Arc::new(
        hi_control::JournaledWorkspaceController::attach_store(inner, store)
            .context("attaching the fail-closed PipeFS control journal")?,
    ))
}

fn seed_restart_recoveries(
    controller: &PipeFsWorkspaceController,
    store: &ControlStore,
    historical: &[WorkspaceBindingRecord],
    evidence: Option<&CausalOperationReceipt>,
) -> Result<()> {
    let current = controller.binding();
    let expected_real = evidence.map(|evidence| {
        restart_operation_recovery_id(
            &BindingId::new(evidence.binding_id.clone()),
            evidence.binding_epoch,
            &OperationId::new(evidence.operation_id.clone()),
        )
    });
    if let Some(expected) = &expected_real {
        ensure!(
            controller.status().recovery_id.as_ref() == Some(expected),
            "PipeFS pending operation recovery identity does not match its persisted evidence"
        );
    }
    let plan = restart_recovery_plan(store, historical, &current, evidence)?;
    if let Some(matched) = plan.matched_real {
        ensure!(
            expected_real.as_ref() == Some(&matched),
            "PipeFS and journal restart recovery identities diverged"
        );
    }
    for recovery in plan.unmatched {
        controller.require_restart_recovery(recovery)?;
    }
    Ok(())
}

#[derive(Debug, Default)]
struct RestartRecoveryPlan {
    matched_real: Option<hi_workspace::RecoveryId>,
    unmatched: Vec<RecoveryRecord>,
}

fn restart_recovery_plan(
    store: &ControlStore,
    historical: &[WorkspaceBindingRecord],
    current: &WorkspaceBinding,
    evidence: Option<&CausalOperationReceipt>,
) -> Result<RestartRecoveryPlan> {
    let journal = WorkspaceProjectionJournal::from_control_store(store.clone());
    let mut plan = RestartRecoveryPlan::default();
    for persisted in historical {
        let binding = historical_binding(current, persisted);
        let report = journal.reconcile_jobs_after_restart(&binding)?;
        for recovery_id in report.recovery_ids {
            let recovery = store
                .get_workspace_recovery(recovery_id.as_str())?
                .ok_or_else(|| anyhow!("restart recovery {recovery_id} was not persisted"))?;
            if recovery_matches_evidence(store, &recovery, evidence)? {
                ensure!(
                    plan.matched_real.replace(recovery_id.clone()).is_none(),
                    "one PipeFS pending operation matched multiple journal recoveries"
                );
            } else {
                if let Some(evidence) = evidence {
                    let evidence_id = restart_operation_recovery_id(
                        &BindingId::new(evidence.binding_id.clone()),
                        evidence.binding_epoch,
                        &OperationId::new(evidence.operation_id.clone()),
                    );
                    ensure!(
                        recovery.recovery_id != evidence_id.as_str(),
                        "PipeFS pending operation conflicts with journal recovery {}; exact identity fences do not match",
                        recovery.recovery_id
                    );
                }
                plan.unmatched.push(contract_recovery(&binding, recovery));
            }
        }
    }
    Ok(plan)
}

fn historical_binding(
    current: &WorkspaceBinding,
    persisted: &WorkspaceBindingRecord,
) -> WorkspaceBinding {
    let mut binding = current.clone();
    binding.binding_id = persisted.binding_id.clone().into();
    binding.workspace_id = persisted.workspace_id.clone().into();
    binding.epoch = persisted.epoch;
    binding.authority = match persisted.authority {
        hi_control::WorkspaceAuthority::Local => WorkspaceAuthority::Local,
        hi_control::WorkspaceAuthority::PipeFs => current.authority.clone(),
    };
    binding.version = persisted
        .workspace_version
        .as_deref()
        .and_then(|version| serde_json::from_str(version).ok())
        .unwrap_or(WorkspaceVersion::Unknown);
    binding
}

fn recovery_matches_evidence(
    store: &ControlStore,
    recovery: &WorkspaceRecoveryRecord,
    evidence: Option<&CausalOperationReceipt>,
) -> Result<bool> {
    let (Some(operation_id), Some(evidence)) = (&recovery.operation_id, evidence) else {
        return Ok(false);
    };
    let operation = store
        .get_workspace_operation(operation_id)?
        .ok_or_else(|| {
            anyhow!(
                "recovery {0} references a missing operation",
                recovery.recovery_id
            )
        })?;
    Ok(operation_matches_evidence(&operation, recovery, evidence))
}

fn operation_matches_evidence(
    operation: &WorkspaceOperationRecord,
    recovery: &WorkspaceRecoveryRecord,
    evidence: &CausalOperationReceipt,
) -> bool {
    let expected = restart_operation_recovery_id(
        &BindingId::new(evidence.binding_id.clone()),
        evidence.binding_epoch,
        &OperationId::new(evidence.operation_id.clone()),
    );
    recovery.recovery_id == expected.as_str()
        && recovery.job_id.is_none()
        && recovery.operation_id.as_deref() == Some(evidence.operation_id.as_str())
        && recovery.binding_id.as_deref() == Some(evidence.binding_id.as_str())
        && operation.operation_id == evidence.operation_id
        && operation.binding_id == evidence.binding_id
        && operation.epoch == evidence.binding_epoch
        && operation.idempotency_key == evidence.idempotency_key
        && replay_class_matches(
            operation.replay_class,
            &evidence.replay_class,
            &evidence.idempotency_key,
        )
}

fn replay_class_matches(
    control: OperationReplayClass,
    evidence: &ReplayClass,
    idempotency_key: &str,
) -> bool {
    match (control, evidence) {
        (OperationReplayClass::PureWorkspace, ReplayClass::PureWorkspace)
        | (OperationReplayClass::NonReplayableExternal, ReplayClass::NonReplayableExternal) => true,
        (OperationReplayClass::IdempotentExternal, ReplayClass::IdempotentExternal { key }) => {
            key.as_str() == idempotency_key
        }
        _ => false,
    }
}

fn contract_recovery(
    binding: &WorkspaceBinding,
    recovery: WorkspaceRecoveryRecord,
) -> RecoveryRecord {
    RecoveryRecord {
        schema_version: hi_workspace::WORKSPACE_CONTRACT_SCHEMA_VERSION,
        recovery_id: recovery.recovery_id.into(),
        kind: if recovery.operation_id.is_some() {
            RecoveryKind::AbandonedMutation
        } else {
            RecoveryKind::CrashedWriterJob
        },
        binding_id: binding.binding_id.clone(),
        epoch: binding.epoch,
        operation_id: recovery.operation_id.map(Into::into),
        job_id: recovery.job_id.map(Into::into),
        detail: recovery
            .detail
            .unwrap_or_else(|| "workspace work was unsettled when the harness restarted".into()),
        created_at_ms: recovery.created_at_ms,
        resolved: false,
    }
}

#[cfg(test)]
#[path = "controller_bridge_tests.rs"]
mod tests;

#[async_trait::async_trait]
impl hi_pipefs::PipeFsSessionBridge for PipeFsDurability {
    fn subscribe_lease_status(&self) -> tokio::sync::watch::Receiver<hi_pipefs::PipeFsLeaseStatus> {
        self.sync.subscribe_writer_lease_status()
    }

    async fn refresh_lease(&self) -> Result<hi_pipefs::PipeFsLease> {
        self.ensure_mutations_unblocked()?;
        refresh_pipefs_lease(&self.workspace, &self.sync).await?;
        let token = self.sync.writer_lease_token().ok_or_else(|| {
            anyhow!("the shared HI writer lease is unavailable; recovery cache retained")
        })?;
        Ok(hi_pipefs::PipeFsLease {
            token,
            generation: self.sync.writer_lease_generation(),
        })
    }

    async fn prepare_causal_mutation(&self) -> Result<()> {
        self.ensure_mutations_unblocked()?;
        self.sync.prepare_causal_pipefs_mutation().await
    }

    async fn causal_transcript_batch(&self) -> Result<hi_pipefs::CausalTranscriptBatch> {
        self.sync.ensure_workspace_execution_staged()?;
        self.sync.causal_pipefs_transcript_batch()
    }

    async fn acknowledge_causal_transcript(
        &self,
        batch: &hi_pipefs::CausalTranscriptBatch,
        cursor: u64,
    ) -> Result<()> {
        self.sync
            .acknowledge_causal_pipefs_transcript(batch, cursor)
    }

    async fn flush_compatibility_transcript(
        &self,
        operation: &hi_pipefs::CausalOperationReceipt,
    ) -> Result<Option<u64>> {
        self.sync.ensure_workspace_execution_staged()?;
        self.sync
            .ensure_compatibility_workspace_execution(operation)?;
        self.sync.flush_required().await?;
        self.sync
            .compatibility_workspace_execution_cursor(operation)
            .map(Some)
    }
}
