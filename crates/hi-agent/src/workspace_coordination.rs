//! Always-present workspace admission and settlement coordination.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result, anyhow, bail};
use hi_workspace::{
    ExecutionDisposition, ExecutionReport, InMemoryWorkspaceController, MutationIntent,
    MutationPermit, RecoveryKind, RecoveryRecord, RecoveryStatus, SettlementOutcome,
    SettlementStatus, WorkspaceAuthority, WorkspaceBinding, WorkspaceCapabilities,
    WorkspaceController, WorkspaceState, WorkspaceStatus, WorkspaceVersion,
};

use crate::WorkspaceDurability;

#[path = "workspace_candidate_recovery.rs"]
mod candidate_recovery;

pub(super) struct ActiveMutation {
    controller: Arc<dyn WorkspaceController>,
    permit: MutationPermit,
}

/// Owns the non-cloneable permit while tools execute. The controller itself is
/// replaceable only at a quiescent rebind boundary.
#[derive(Clone)]
pub(crate) struct WorkspaceCoordination {
    controller: Arc<RwLock<Arc<dyn WorkspaceController>>>,
    active: Arc<Mutex<Option<ActiveMutation>>>,
    admission: Arc<admission::WorkspaceAdmissionGate>,
    controller_settles_backend: Arc<AtomicBool>,
    harness: hi_workspace::ResolvedHarnessSettings,
}

impl WorkspaceCoordination {
    pub(crate) fn binding(&self) -> WorkspaceBinding {
        self.controller().binding()
    }

    pub(crate) fn capabilities(&self) -> WorkspaceCapabilities {
        self.controller().capabilities()
    }

    pub(crate) fn status(&self) -> WorkspaceStatus {
        self.controller().status()
    }

    pub(crate) fn install_local(&self, workspace_root: &Path, state_root: &Path) -> Result<()> {
        let epoch = self.controller().binding().epoch.saturating_add(1);
        self.replace(
            local_controller(
                workspace_root,
                state_root,
                epoch,
                resolved_job_limits(&self.harness),
            ),
            false,
        )
    }

    pub(crate) fn install_pipefs(
        &self,
        session_id: &str,
        writer_protocol: u16,
        causal_commit: bool,
        workspace_root: &Path,
        state_root: &Path,
    ) -> Result<()> {
        let epoch = self.controller().binding().epoch.saturating_add(1);
        let controller = pipefs_controller(
            session_id,
            writer_protocol,
            causal_commit,
            workspace_root,
            state_root,
            epoch,
            resolved_job_limits(&self.harness),
        )?;
        self.replace(controller, false)
    }

    pub(crate) fn install_controller(
        &self,
        controller: Arc<dyn WorkspaceController>,
    ) -> Result<()> {
        self.replace(controller, true)
    }

    fn replace(
        &self,
        controller: Arc<dyn WorkspaceController>,
        controller_settles_backend: bool,
    ) -> Result<()> {
        self.replace_with_admission_closed(controller, controller_settles_backend)
    }

    fn replace_while_admission_closed(
        &self,
        controller: Arc<dyn WorkspaceController>,
        controller_settles_backend: bool,
    ) -> Result<()> {
        self.ensure_replace_ready()?;
        *self
            .controller
            .write()
            .map_err(|_| anyhow!("workspace controller lock is poisoned"))? = controller;
        self.controller_settles_backend
            .store(controller_settles_backend, Ordering::Release);
        Ok(())
    }

    pub(crate) fn ensure_replace_ready(&self) -> Result<()> {
        if self.lock_active()?.is_some() {
            bail!("cannot replace a workspace controller while a mutation is unsettled");
        }
        let current = self.controller().status();
        if current.state != WorkspaceState::Ready {
            bail!(
                "cannot replace a workspace controller in {:?} state",
                current.state
            );
        }
        if !current.active_jobs.is_empty() {
            bail!(
                "cannot replace a workspace controller while jobs remain unsettled: {}",
                current
                    .active_jobs
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Ok(())
    }

    pub(crate) async fn begin(
        &self,
        durability: Option<Arc<dyn WorkspaceDurability>>,
        dirty_paths: Option<Vec<String>>,
    ) -> Result<()> {
        let mut intent = MutationIntent::workspace("tool or lifecycle workspace mutation");
        intent.dirty_paths = dirty_paths.map(|paths| paths.into_iter().map(Into::into).collect());
        self.begin_intent(durability, intent).await
    }

    pub(crate) async fn begin_intent(
        &self,
        mut durability: Option<Arc<dyn WorkspaceDurability>>,
        intent: MutationIntent,
    ) -> Result<()> {
        let _admission = self.acquire_admission().await;
        if self.lock_active()?.is_some() {
            bail!("a workspace mutation is already admitted and awaiting settlement");
        }
        let controller = self.controller();
        let dirty_paths = intent.dirty_paths.as_ref().map(|paths| {
            paths
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        });
        if !self.harness.features.workspace_controller_v2 {
            let status = controller.status();
            if status.recovery_id.is_some()
                || !matches!(
                    status.state,
                    WorkspaceState::Ready | WorkspaceState::LeaseUncertain
                )
            {
                bail!(
                    "workspace recovery remains required while controller-v2 admission is disabled: {:?}",
                    status.state
                );
            }
            match durability {
                Some(durability) => durability
                    .mutation_started(dirty_paths)
                    .await
                    .context("legacy workspace mutation admission failed")?,
                None if matches!(
                    controller.binding().authority,
                    WorkspaceAuthority::PipeFs { .. }
                ) =>
                {
                    bail!(
                        "PipeFS mutation admission requires the legacy durability fence while controller-v2 admission is disabled"
                    )
                }
                None => {}
            }
            return Ok(());
        }
        if self.controller_settles_backend.load(Ordering::Acquire) {
            durability = None;
        }
        let permit = controller.begin(intent).await?;
        if let Some(durability) = durability
            && let Err(error) = durability.mutation_started(dirty_paths).await
        {
            let report = ExecutionReport {
                disposition: ExecutionDisposition::Failed,
                workspace_may_have_changed: false,
                external_effect_may_have_occurred: false,
                content_digest: None,
                changed_paths: Vec::new(),
                artifacts: Vec::new(),
                detail: Some(format!("mutation admission backend failed: {error:#}")),
            };
            let _ = controller.settle(permit, report).await;
            return Err(
                error.context("workspace mutation admission backend rejected the operation")
            );
        }
        *self.lock_active()? = Some(ActiveMutation { controller, permit });
        Ok(())
    }

    /// Shield accepted settlement from cancellation. Once the bounded caller
    /// wait expires the task remains detached and admission stays closed until
    /// it publishes a terminal controller state.
    pub(crate) async fn checkpoint(
        &self,
        mut durability: Option<Arc<dyn WorkspaceDurability>>,
        execution: ExecutionReport,
    ) -> Result<()> {
        let admitted = {
            let mut active = self.lock_active()?;
            active.take()
        };
        if let Some(active) = admitted {
            if self.controller_settles_backend.load(Ordering::Acquire) {
                durability = None;
            }
            return self.settle_owned(active, durability, execution).await;
        }
        if !self.harness.features.workspace_controller_v2 {
            let status = self.controller().status();
            if status.recovery_id.is_some() {
                bail!("workspace recovery remains required: {:?}", status.state);
            }
            if let Some(durability) = durability {
                durability
                    .checkpoint()
                    .await
                    .context("legacy workspace checkpoint failed")?;
            }
            return Ok(());
        }
        if self.controller_settles_backend.load(Ordering::Acquire) {
            durability = None;
        }
        let controller = self.controller();
        let permit = controller.begin(MutationIntent::reconciliation()).await?;
        self.settle_owned(ActiveMutation { controller, permit }, durability, execution)
            .await
    }

    /// Use only after the authoritative backend has proved an ambiguous
    /// operation durable (for example after `/pipefs retry`).
    pub(crate) async fn reconcile_after_external_proof(&self) -> Result<()> {
        let controller = self.controller();
        let Some(recovery_id) = controller.status().recovery_id else {
            return Ok(());
        };
        let outcome = controller.reconcile(recovery_id).await;
        match outcome.status {
            RecoveryStatus::Recovered => Ok(()),
            other => bail!(
                "workspace controller rejected authoritative recovery proof ({other:?}): {}",
                outcome.detail.as_deref().unwrap_or("no detail")
            ),
        }
    }

    fn controller(&self) -> Arc<dyn WorkspaceController> {
        self.controller
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(super) fn job_controller(&self) -> Arc<dyn WorkspaceController> {
        self.controller()
    }

    pub(super) fn active_parent_operation(&self) -> Option<hi_workspace::OperationId> {
        self.active_mutation_record()
            .map(|record| record.operation_id)
    }

    pub(super) fn active_mutation_record(&self) -> Option<hi_workspace::MutationPermitRecord> {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|active| active.permit.record().clone())
    }

    pub(super) fn active_intent(&self) -> Option<MutationIntent> {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|active| active.permit.record().intent.clone())
    }

    pub(super) fn abandon_active(&self) -> Result<()> {
        let active = self.lock_active()?.take();
        drop(active);
        Ok(())
    }

    fn lock_active(&self) -> Result<std::sync::MutexGuard<'_, Option<ActiveMutation>>> {
        self.active
            .lock()
            .map_err(|_| anyhow!("workspace mutation permit lock is poisoned"))
    }
}

#[path = "workspace_coordination_admission.rs"]
mod admission;
#[path = "workspace_coordination_barrier.rs"]
mod barrier;
#[path = "workspace_coordination_cancel.rs"]
mod cancellation;
#[path = "workspace_coordination_controller_factory.rs"]
mod controller_factory;
#[path = "workspace_coordination_jobs.rs"]
mod jobs;
#[cfg(test)]
#[path = "workspace_coordination_rebind_admission_tests.rs"]
mod rebind_admission_tests;
#[path = "workspace_coordination_runtime_settings.rs"]
mod runtime_settings;
#[cfg(test)]
#[path = "workspace_coordination_settings_tests.rs"]
mod settings_tests;
use controller_factory::{local_controller, pipefs_controller, resolved_job_limits};

fn seed_restart_recoveries(
    controller: &InMemoryWorkspaceController,
    journal: &hi_control::WorkspaceProjectionJournal,
    store: &hi_control::ControlStore,
    historical: &[hi_control::WorkspaceBindingRecord],
) -> Result<()> {
    let current = controller.binding();
    for persisted in historical {
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
        let report = journal.reconcile_jobs_after_restart(&binding)?;
        for recovery_id in report.recovery_ids {
            let persisted_recovery = store
                .get_workspace_recovery(recovery_id.as_str())?
                .ok_or_else(|| anyhow!("restart recovery {recovery_id} was not persisted"))?;
            controller.require_recovery(RecoveryRecord {
                schema_version: hi_workspace::WORKSPACE_CONTRACT_SCHEMA_VERSION,
                recovery_id,
                kind: if persisted_recovery.operation_id.is_some() {
                    RecoveryKind::AbandonedMutation
                } else {
                    RecoveryKind::CrashedWriterJob
                },
                binding_id: current.binding_id.clone(),
                epoch: current.epoch,
                operation_id: persisted_recovery.operation_id.map(Into::into),
                job_id: persisted_recovery.job_id.map(Into::into),
                detail: persisted_recovery.detail.unwrap_or_else(|| {
                    "workspace work was unsettled when the harness restarted".to_owned()
                }),
                created_at_ms: persisted_recovery.created_at_ms,
                resolved: false,
            })?;
        }
    }
    Ok(())
}

fn workspace_id(root: &Path) -> String {
    uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        root.to_string_lossy().as_bytes(),
    )
    .to_string()
}

fn settlement_result(
    outcome: SettlementOutcome,
    backend_error: Option<anyhow::Error>,
) -> Result<()> {
    if let Some(error) = backend_error {
        return Err(error.context(format!(
            "workspace operation {} was not proven durable ({:?})",
            outcome.operation_id, outcome.status
        )));
    }
    match outcome.status {
        SettlementStatus::Durable
        | SettlementStatus::NoChange
        | SettlementStatus::LocalAuditDegraded => Ok(()),
        status => bail!(
            "workspace operation {} did not settle ({status:?}): {}",
            outcome.operation_id,
            outcome.detail.as_deref().unwrap_or("no detail")
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Result, anyhow};
    use async_trait::async_trait;
    use hi_workspace::{
        EffectScope, ExecutionReport, InMemoryWorkspaceController, JobKind, JobLimits, JobSpec,
        WorkspaceController, WorkspaceState, WorkspaceVersion,
    };

    use super::{WorkspaceCoordination, workspace_id};
    use crate::WorkspaceDurability;

    struct FailingDurability;

    #[async_trait]
    impl WorkspaceDurability for FailingDurability {
        async fn mutation_started(&self, _dirty_paths: Option<Vec<String>>) -> Result<()> {
            Ok(())
        }

        async fn checkpoint(&self) -> Result<()> {
            Err(anyhow!("lost acknowledgement"))
        }
    }

    struct GatedDurability {
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl WorkspaceDurability for GatedDurability {
        async fn mutation_started(&self, _dirty_paths: Option<Vec<String>>) -> Result<()> {
            Ok(())
        }

        async fn checkpoint(&self) -> Result<()> {
            self.entered.notify_one();
            self.release.notified().await;
            Ok(())
        }
    }

    fn subject() -> (tempfile::TempDir, Arc<WorkspaceCoordination>) {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state");
        std::fs::create_dir_all(&state).unwrap();
        let coordination = Arc::new(WorkspaceCoordination::new_local(root.path(), &state));
        (root, coordination)
    }

    #[test]
    fn every_backend_rebind_advances_the_epoch() {
        let (root, subject) = subject();
        let state = root.path().join("next-state");
        std::fs::create_dir_all(&state).unwrap();
        assert_eq!(subject.binding().epoch, 0);
        subject.install_local(root.path(), &state).unwrap();
        assert_eq!(subject.binding().epoch, 1);
        subject
            .install_pipefs("session-1", 2, false, root.path(), &state)
            .unwrap();
        assert_eq!(subject.binding().epoch, 2);
    }

    #[tokio::test]
    async fn rebind_rejects_unsettled_unified_jobs() {
        let (root, subject) = subject();
        subject
            .job_controller()
            .register_job(JobSpec {
                kind: JobKind::ReadAgent,
                effect_scope: EffectScope::ReadOnly,
                name: "reader crossing rebind".into(),
                limits: JobLimits::default(),
                parent_operation: None,
            })
            .await
            .unwrap();

        let next_state = root.path().join("next-state");
        std::fs::create_dir_all(&next_state).unwrap();
        let error = subject.install_local(root.path(), &next_state).unwrap_err();
        assert!(error.to_string().contains("jobs remain unsettled"));
        assert_eq!(subject.binding().epoch, 0);
    }

    #[tokio::test]
    async fn local_mutation_uses_the_same_admission_and_settlement_path() {
        let (_root, subject) = subject();
        subject
            .begin(None, Some(vec!["a.rs".into()]))
            .await
            .unwrap();
        assert_eq!(subject.status().state, WorkspaceState::Mutating);
        subject
            .checkpoint(None, ExecutionReport::succeeded(Some("digest-a".into())))
            .await
            .unwrap();
        assert_eq!(
            subject.status().state,
            WorkspaceState::Ready,
            "{:?}",
            subject.status()
        );
        assert!(matches!(
            subject.binding().version,
            WorkspaceVersion::Local { generation: 1, .. }
        ));
    }
    #[tokio::test]
    async fn abandoning_after_admission_enters_recovery_before_tool_execution() {
        let (_root, subject) = subject();
        subject.begin(None, None).await.unwrap();
        subject.abandon_active().unwrap();
        assert_eq!(subject.status().state, WorkspaceState::RecoveryRequired);
        assert!(subject.begin(None, None).await.is_err());
    }

    #[tokio::test]
    async fn ambiguous_backend_failure_blocks_until_authoritative_proof() {
        let (root, subject) = subject();
        let durability: Arc<dyn WorkspaceDurability> = Arc::new(FailingDurability);
        subject.begin(Some(durability.clone()), None).await.unwrap();
        assert!(
            subject
                .checkpoint(
                    Some(durability),
                    ExecutionReport::succeeded(Some("digest-a".into())),
                )
                .await
                .is_err()
        );
        assert_eq!(subject.status().state, WorkspaceState::RecoveryRequired);
        assert!(
            subject
                .install_local(root.path(), &root.path().join("state"))
                .is_err()
        );
        subject.reconcile_after_external_proof().await.unwrap();
        assert_eq!(subject.status().state, WorkspaceState::Ready);
    }

    #[tokio::test]
    async fn native_controller_does_not_double_invoke_legacy_durability() {
        let (root, subject) = subject();
        let state = root.path().join("state");
        let controller: Arc<dyn WorkspaceController> =
            Arc::new(InMemoryWorkspaceController::new_local_at_epoch(
                workspace_id(root.path()),
                root.path(),
                &state,
                1,
            ));
        subject.install_controller(controller).unwrap();
        let durability: Arc<dyn WorkspaceDurability> = Arc::new(FailingDurability);
        subject.begin(Some(durability.clone()), None).await.unwrap();
        subject
            .checkpoint(Some(durability), ExecutionReport::succeeded(None))
            .await
            .unwrap();
        assert_eq!(subject.status().state, WorkspaceState::Ready);
    }

    #[tokio::test]
    async fn cancelling_the_caller_does_not_cancel_accepted_settlement() {
        let (_root, subject) = subject();
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let durability: Arc<dyn WorkspaceDurability> = Arc::new(GatedDurability {
            entered: entered.clone(),
            release: release.clone(),
        });
        subject.begin(Some(durability.clone()), None).await.unwrap();
        let worker_subject = subject.clone();
        let worker = tokio::spawn(async move {
            worker_subject
                .checkpoint(
                    Some(durability),
                    ExecutionReport::succeeded(Some("digest-a".into())),
                )
                .await
        });
        entered.notified().await;
        worker.abort();
        assert!(worker.await.unwrap_err().is_cancelled());
        release.notify_one();
        tokio::time::timeout(Duration::from_secs(2), async {
            while subject.status().state != WorkspaceState::Ready {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn local_host_persists_binding_and_mutation_in_the_shared_control_store() {
        let (root, subject) = subject();
        subject.begin(None, None).await.unwrap();
        subject
            .checkpoint(
                None,
                ExecutionReport::succeeded(Some("persisted-digest".into())),
            )
            .await
            .unwrap();

        let store = hi_control::ControlStore::open_for_state(root.path().join("state")).unwrap();
        let binding = store
            .latest_workspace_binding(&workspace_id(root.path()))
            .unwrap()
            .unwrap();
        assert_eq!(binding.state, hi_control::WorkspaceProjectionState::Ready);
        assert!(store.max_event_sequence().unwrap() >= 6);
    }

    #[tokio::test]
    async fn active_writer_is_recovered_before_restart_admission_and_not_refenced() {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state");
        std::fs::create_dir_all(&state).unwrap();
        let store = hi_control::ControlStore::open_for_state(&state).unwrap();
        let raw: Arc<dyn WorkspaceController> = Arc::new(InMemoryWorkspaceController::new_local(
            workspace_id(root.path()),
            root.path(),
            &state,
        ));
        let original =
            hi_control::JournaledWorkspaceController::attach_store(raw, store.clone()).unwrap();
        let job = original
            .register_job(JobSpec {
                kind: JobKind::WriteCandidate,
                effect_scope: EffectScope::CandidateOnly,
                name: "interrupted candidate".into(),
                limits: JobLimits::default(),
                parent_operation: None,
            })
            .await
            .unwrap();
        drop(original);

        let restarted = WorkspaceCoordination::new_local(root.path(), &state);
        assert_eq!(restarted.status().state, WorkspaceState::RecoveryRequired);
        assert!(restarted.begin(None, None).await.is_err());
        restarted.reconcile_after_external_proof().await.unwrap();
        assert_eq!(restarted.status().state, WorkspaceState::Ready);
        assert_eq!(
            store.get_job(job.job_id.as_str()).unwrap().unwrap().state,
            hi_control::ControlJobState::Failed
        );

        let second_restart = WorkspaceCoordination::new_local(root.path(), &state);
        assert_eq!(second_restart.status().state, WorkspaceState::Ready);
    }

    #[tokio::test]
    async fn unavailable_local_journal_keeps_foreground_work_but_disables_writers() {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state");
        std::fs::create_dir_all(state.join("events.sqlite3")).unwrap();
        let subject = WorkspaceCoordination::new_local(root.path(), &state);

        assert_eq!(subject.status().state, WorkspaceState::LocalAuditDegraded);
        assert!(!subject.capabilities().background_writers);
        subject.begin(None, None).await.unwrap();
        subject
            .checkpoint(None, ExecutionReport::succeeded(None))
            .await
            .unwrap();
        assert_eq!(subject.status().state, WorkspaceState::LocalAuditDegraded);
    }

    #[test]
    fn unavailable_pipefs_journal_fails_installation_closed() {
        let (root, subject) = subject();
        let bad_state = root.path().join("bad-state");
        std::fs::create_dir_all(bad_state.join("events.sqlite3")).unwrap();

        let error = subject
            .install_pipefs("session-closed", 2, true, root.path(), &bad_state)
            .unwrap_err();
        assert!(error.to_string().contains("PipeFS control journal"));
        assert!(matches!(
            subject.binding().authority,
            hi_workspace::WorkspaceAuthority::Local
        ));
    }
}
