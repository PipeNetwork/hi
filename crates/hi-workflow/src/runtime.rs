use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::{
    Journal, StoredRunStatus, WorkflowHostRequest, WorkflowOutcome, WorkflowRunManifest,
    WorkflowRunOwnership, WorkflowRunParams, WorkflowRunStore,
};
use hi_events::{
    ActivityObject, ActivityState, ActivityVerb, EventContext, EventKind, EventSink, RunEvent,
    SemanticActivity,
};

const MAX_ACTIVE_RUNS: usize = 4;
static NEXT_RUN_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("workflow run not found: {0}")]
    NotFound(String),
    #[error("workflow run is not resumable from status {0:?}")]
    NotResumable(StoredRunStatus),
    #[error("workflow run is active: {0}")]
    Active(String),
    #[error("workflow run is owned by another process: {0}")]
    Owned(String),
    #[error("workflow engine {0:?} cannot be resumed by this runtime")]
    UnsupportedEngine(crate::WorkflowEngineKind),
    #[error("too many active workflow runs (maximum {MAX_ACTIVE_RUNS})")]
    AtCapacity,
    #[error(transparent)]
    Store(#[from] crate::StoreError),
    #[error(transparent)]
    Journal(#[from] crate::JournalError),
    #[error("workflow task failed: {0}")]
    Task(String),
    #[error("raised workflow budget {raised} is below spent budget {spent}")]
    InvalidBudget { raised: u64, spent: u64 },
    #[error("workflow approval rejected: {0}")]
    Approval(String),
}

pub struct ManagedWorkflowRun {
    pub manifest: WorkflowRunManifest,
    pub host_rx: mpsc::UnboundedReceiver<WorkflowHostRequest>,
    cancel: CancellationToken,
    task: JoinHandle<WorkflowOutcome>,
    ownership: WorkflowRunOwnership,
}

impl ManagedWorkflowRun {
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    pub fn into_parts(
        self,
    ) -> (
        WorkflowRunManifest,
        mpsc::UnboundedReceiver<WorkflowHostRequest>,
        CancellationToken,
        JoinHandle<WorkflowOutcome>,
        WorkflowRunOwnership,
    ) {
        (
            self.manifest,
            self.host_rx,
            self.cancel,
            self.task,
            self.ownership,
        )
    }
}

pub struct WorkflowRuntimeManager {
    store: WorkflowRunStore,
    active: HashMap<String, ManagedWorkflowRun>,
    event_sink: Option<Arc<dyn EventSink>>,
}

impl WorkflowRuntimeManager {
    pub fn new(store: WorkflowRunStore) -> Self {
        Self {
            store,
            active: HashMap::new(),
            event_sink: None,
        }
    }

    pub fn with_event_sink(store: WorkflowRunStore, event_sink: Arc<dyn EventSink>) -> Self {
        Self {
            store,
            active: HashMap::new(),
            event_sink: Some(event_sink),
        }
    }

    pub fn store(&self) -> &WorkflowRunStore {
        &self.store
    }

    /// Reconcile runs left non-terminal by a previous process. This must be
    /// called before accepting new work after startup.
    pub fn recover_interrupted(&self) -> Result<Vec<crate::StoredWorkflowRun>, RuntimeError> {
        let runs = self.store.list()?;
        let mut recovered = Vec::new();
        for run in runs {
            // Paused is an intentional durable state (including approval
            // pauses), not evidence that the previous process died mid-run.
            // Reclassifying it as Interrupted would let the generic resume
            // path bypass the approval boundary after a restart.
            if run.manifest.status == StoredRunStatus::Running {
                let Some(_ownership) = self.store.try_claim(&run.manifest.run_id)? else {
                    continue;
                };
                recovered.push(self.store.recover(&run.manifest.run_id)?);
            }
        }
        Ok(recovered)
    }

    pub fn start(
        &mut self,
        workflow_name: String,
        script: String,
        args: serde_json::Value,
        agent_budget: u64,
    ) -> Result<String, RuntimeError> {
        if self.active.len() >= MAX_ACTIVE_RUNS {
            return Err(RuntimeError::AtCapacity);
        }
        let id = self.allocate_run_id();
        let manifest = WorkflowRunManifest::new(id.clone(), workflow_name.clone(), agent_budget)?;
        self.store.register(&manifest, &script, &args)?;
        let ownership = self
            .store
            .try_claim(&id)?
            .ok_or_else(|| RuntimeError::Owned(id.clone()))?;
        let run = spawn_run(
            manifest,
            script,
            args,
            Journal::load(self.store.journal_path(&id)?)?,
            ownership,
        );
        self.active.insert(id.clone(), run);
        self.emit_workflow_event(
            EventKind::WorkflowStarted,
            &id,
            &workflow_name,
            ActivityState::Running,
            ActivityVerb::Start,
        );
        Ok(id)
    }

    /// Register a workflow in a durable approval-paused state without
    /// starting its engine task. This gives background triggers a recoverable
    /// manifest before any side effect or agent work is allowed.
    pub fn start_paused_for_approval(
        &mut self,
        id: String,
        workflow_name: String,
        script: String,
        args: serde_json::Value,
        agent_budget: u64,
        approval: (&str, &str),
    ) -> Result<String, RuntimeError> {
        if self.active.len() >= MAX_ACTIVE_RUNS {
            return Err(RuntimeError::AtCapacity);
        }
        let mut manifest =
            WorkflowRunManifest::new(id.clone(), workflow_name.clone(), agent_budget)?;
        manifest.set_pending_approval(approval.0, approval.1);
        self.store.register(&manifest, &script, &args)?;
        self.emit_workflow_event(
            EventKind::WorkflowPaused,
            &id,
            &workflow_name,
            ActivityState::Waiting,
            ActivityVerb::Wait,
        );
        Ok(id)
    }

    /// Allocate the durable run identity before creating an approval so the
    /// approval request can be bound to this exact run.
    pub fn allocate_run_id(&self) -> String {
        new_run_id()
    }

    pub fn resume(&mut self, run_id: &str, raised_budget: Option<u64>) -> Result<(), RuntimeError> {
        if self.active.contains_key(run_id) {
            return Err(RuntimeError::Active(run_id.into()));
        }
        if self.active.len() >= MAX_ACTIVE_RUNS {
            return Err(RuntimeError::AtCapacity);
        }
        let ownership = self
            .store
            .try_claim(run_id)?
            .ok_or_else(|| RuntimeError::Owned(run_id.into()))?;
        let stored = self.store.load(run_id)?;
        if stored.manifest.engine != crate::WorkflowEngineKind::Rhai {
            return Err(RuntimeError::UnsupportedEngine(stored.manifest.engine));
        }
        if !matches!(
            stored.manifest.status,
            StoredRunStatus::Paused
                | StoredRunStatus::BudgetExceeded
                | StoredRunStatus::Interrupted
        ) {
            return Err(RuntimeError::NotResumable(stored.manifest.status));
        }
        if stored.manifest.pending_approval_id.is_some()
            || stored.manifest.pending_operation_digest.is_some()
        {
            return Err(RuntimeError::Approval(
                "workflow has a pending approval; use resume_with_approval".into(),
            ));
        }
        // Validate the durable journal before publishing Running. Otherwise a
        // corrupt/unreadable journal leaves a manifest that claims to be
        // active even though no engine task was spawned.
        let journal = Journal::load(stored.journal_path)?;
        let mut manifest = stored.manifest;
        if let Some(budget) = raised_budget {
            if budget < manifest.agent_spent {
                return Err(RuntimeError::InvalidBudget {
                    raised: budget,
                    spent: manifest.agent_spent,
                });
            }
            manifest.agent_budget = budget.max(manifest.agent_budget);
        }
        manifest.status = StoredRunStatus::Running;
        manifest.outcome = None;
        self.store.persist(&manifest)?;
        let workflow_name = manifest.workflow_name.clone();
        let run = spawn_run(manifest, stored.script, stored.args, journal, ownership);
        self.active.insert(run_id.into(), run);
        self.emit_workflow_event(
            EventKind::WorkflowResumed,
            run_id,
            &workflow_name,
            ActivityState::Running,
            ActivityVerb::Resume,
        );
        Ok(())
    }

    /// Atomically validate and consume the exact approval recorded in a
    /// paused workflow manifest before resuming it. Approval consumption is
    /// one-shot; a changed digest cannot resume the workflow.
    pub fn resume_with_approval(
        &mut self,
        run_id: &str,
        approval_store: &dyn hi_policy::ApprovalStore,
        approval_id: &str,
        operation_digest: &str,
        raised_budget: Option<u64>,
    ) -> Result<(), RuntimeError> {
        if self.active.contains_key(run_id) {
            return Err(RuntimeError::Active(run_id.into()));
        }
        if self.active.len() >= MAX_ACTIVE_RUNS {
            return Err(RuntimeError::AtCapacity);
        }
        let ownership = self
            .store
            .try_claim(run_id)?
            .ok_or_else(|| RuntimeError::Owned(run_id.into()))?;
        let stored = self.store.load(run_id)?;
        if stored.manifest.engine != crate::WorkflowEngineKind::Rhai {
            return Err(RuntimeError::UnsupportedEngine(stored.manifest.engine));
        }
        if !matches!(
            stored.manifest.status,
            StoredRunStatus::Paused
                | StoredRunStatus::ApprovalClaiming
                | StoredRunStatus::BudgetExceeded
                | StoredRunStatus::Interrupted
        ) {
            return Err(RuntimeError::NotResumable(stored.manifest.status));
        }
        if stored.manifest.pending_approval_id.as_deref() != Some(approval_id)
            || stored.manifest.pending_operation_digest.as_deref() != Some(operation_digest)
        {
            return Err(RuntimeError::Approval(
                "approval does not match the workflow's pending operation".into(),
            ));
        }
        let mut manifest = stored.manifest;
        if let Some(budget) = raised_budget {
            if budget < manifest.agent_spent {
                return Err(RuntimeError::InvalidBudget {
                    raised: budget,
                    spent: manifest.agent_spent,
                });
            }
            manifest.agent_budget = budget.max(manifest.agent_budget);
        }
        let journal = Journal::load(stored.journal_path)?;

        // Publish a fail-closed intermediate before consuming the one-shot
        // approval. If the process dies after claim(), a later explicit
        // approval resume can reconcile the consumed record and finish this
        // transition; plain resume never accepts ApprovalClaiming.
        let was_claiming = manifest.status == StoredRunStatus::ApprovalClaiming;
        let approval_id = hi_policy::ApprovalId(approval_id.to_string());
        let operation_digest = hi_policy::OperationDigest(operation_digest.to_string());
        let current = approval_store
            .get(&approval_id)
            .map_err(|error| RuntimeError::Approval(error.to_string()))?
            .ok_or_else(|| RuntimeError::Approval("approval request not found".into()))?;
        validate_workflow_approval(
            &current,
            run_id,
            &manifest.workflow_name,
            &approval_id,
            &operation_digest,
        )?;
        if (!was_claiming && current.state != hi_policy::ApprovalState::Approved)
            || (was_claiming
                && !matches!(
                    current.state,
                    hi_policy::ApprovalState::Approved | hi_policy::ApprovalState::Consumed
                ))
        {
            return Err(RuntimeError::Approval(format!(
                "approval is not approved: {:?}",
                current.state
            )));
        }
        if !was_claiming {
            manifest.begin_approval_claim();
            self.store.persist(&manifest)?;
        }
        let record = if was_claiming && current.state == hi_policy::ApprovalState::Consumed {
            current
        } else {
            approval_store
                .claim(&approval_id, &operation_digest)
                .map_err(|error| RuntimeError::Approval(error.to_string()))?
        };
        validate_workflow_approval(
            &record,
            run_id,
            &manifest.workflow_name,
            &approval_id,
            &operation_digest,
        )?;
        if record.state != hi_policy::ApprovalState::Consumed
            || record
                .consumed_at_ms
                .is_none_or(|consumed| consumed < manifest.updated_at_ms)
        {
            return Err(RuntimeError::Approval(
                "consumed approval record does not match the durable workflow claim".into(),
            ));
        }
        manifest.pending_approval_id = None;
        manifest.pending_operation_digest = None;
        manifest.status = StoredRunStatus::Running;
        manifest.outcome = None;
        self.store.persist(&manifest)?;
        let workflow_name = manifest.workflow_name.clone();
        let run = spawn_run(manifest, stored.script, stored.args, journal, ownership);
        self.active.insert(run_id.into(), run);
        self.emit_workflow_event(
            EventKind::WorkflowResumed,
            run_id,
            &workflow_name,
            ActivityState::Running,
            ActivityVerb::Resume,
        );
        Ok(())
    }

    /// Persist a durable approval pause for a workflow that has not yet been
    /// taken into the active runtime. This is the recovery boundary used by
    /// background dispatchers; no approval is auto-consumed here.
    pub fn pause_for_approval(
        &self,
        run_id: &str,
        approval_id: &str,
        operation_digest: &str,
    ) -> Result<(), RuntimeError> {
        if self.active.contains_key(run_id) {
            return Err(RuntimeError::Active(run_id.into()));
        }
        let _ownership = self
            .store
            .try_claim(run_id)?
            .ok_or_else(|| RuntimeError::Owned(run_id.into()))?;
        let mut stored = self.store.load(run_id)?.manifest;
        stored.set_pending_approval(approval_id, operation_digest);
        self.store.persist(&stored)?;
        Ok(())
    }

    pub fn cancel(&self, run_id: &str) -> Result<(), RuntimeError> {
        let run = self
            .active
            .get(run_id)
            .ok_or_else(|| RuntimeError::NotFound(run_id.into()))?;
        run.cancel();
        Ok(())
    }

    pub fn active_mut(&mut self, run_id: &str) -> Option<&mut ManagedWorkflowRun> {
        self.active.get_mut(run_id)
    }

    /// Transfer an active run to a UI/service owner while preserving its host
    /// receiver, cancellation token, and join handle. This avoids dropping a
    /// short-lived manager immediately after `resume`.
    pub fn take_active(&mut self, run_id: &str) -> Result<ManagedWorkflowRun, RuntimeError> {
        self.active
            .remove(run_id)
            .ok_or_else(|| RuntimeError::NotFound(run_id.into()))
    }

    pub fn active_ids(&self) -> impl Iterator<Item = &str> {
        self.active.keys().map(String::as_str)
    }

    pub async fn join(&mut self, run_id: &str) -> Result<WorkflowOutcome, RuntimeError> {
        let run = self
            .active
            .remove(run_id)
            .ok_or_else(|| RuntimeError::NotFound(run_id.into()))?;
        self.finish_run(run).await
    }

    async fn finish_run(&self, run: ManagedWorkflowRun) -> Result<WorkflowOutcome, RuntimeError> {
        let ManagedWorkflowRun {
            mut manifest,
            task,
            ownership,
            ..
        } = run;
        let outcome = match task.await {
            Ok(outcome) => outcome,
            Err(error) => WorkflowOutcome::Failed {
                error: format!("workflow task failed: {error}"),
            },
        };
        manifest.finish(outcome.clone());
        self.store.persist(&manifest)?;
        drop(ownership);
        let (kind, state, verb) = match &outcome {
            WorkflowOutcome::Completed { .. } => (
                EventKind::WorkflowCompleted,
                ActivityState::Succeeded,
                ActivityVerb::Complete,
            ),
            WorkflowOutcome::Paused { .. } => (
                EventKind::WorkflowPaused,
                ActivityState::Waiting,
                ActivityVerb::Wait,
            ),
            WorkflowOutcome::Cancelled => (
                EventKind::WorkflowFailed,
                ActivityState::Cancelled,
                ActivityVerb::Cancel,
            ),
            WorkflowOutcome::BudgetExceeded { .. } | WorkflowOutcome::Failed { .. } => (
                EventKind::WorkflowFailed,
                ActivityState::Failed,
                ActivityVerb::Fail,
            ),
        };
        self.emit_workflow_event(kind, &manifest.run_id, &manifest.workflow_name, state, verb);
        Ok(outcome)
    }

    fn emit_workflow_event(
        &self,
        kind: EventKind,
        run_id: &str,
        workflow_name: &str,
        state: ActivityState,
        verb: ActivityVerb,
    ) {
        let Some(sink) = &self.event_sink else { return };
        let _ = sink.publish(RunEvent::new(
            kind,
            EventContext {
                workflow_id: Some(run_id.to_string()),
                run_id: Some(run_id.to_string()),
                ..EventContext::default()
            },
            SemanticActivity {
                verb,
                object: ActivityObject::Workflow,
                state,
                group_key: format!("workflow:{run_id}"),
                title: workflow_name.to_string(),
                detail: None,
                refs: vec![],
                progress: None,
            },
        ));
    }

    pub fn list(&self) -> Result<Vec<crate::StoredWorkflowRun>, RuntimeError> {
        Ok(self.store.list()?)
    }

    pub fn delete(&self, run_id: &str) -> Result<(), RuntimeError> {
        if self.active.contains_key(run_id) {
            return Err(RuntimeError::Active(run_id.into()));
        }
        self.store.delete(run_id)?;
        Ok(())
    }

    pub async fn shutdown(&mut self, timeout: Duration) {
        for run in self.active.values() {
            run.cancel();
        }
        let deadline = tokio::time::Instant::now() + timeout;
        let ids = self.active.keys().cloned().collect::<Vec<_>>();
        for id in ids {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            let Some(run) = self.active.remove(&id) else {
                continue;
            };
            let mut manifest = run.manifest.clone();
            let abort_handle = run.task.abort_handle();
            match tokio::time::timeout(remaining, self.finish_run(run)).await {
                Ok(_) => {}
                Err(_) => {
                    abort_handle.abort();
                    manifest.finish(WorkflowOutcome::Failed {
                        error: "workflow shutdown timed out".into(),
                    });
                    let _ = self.store.persist(&manifest);
                    break;
                }
            }
        }
        for (_, run) in self.active.drain() {
            run.task.abort();
            let mut manifest = run.manifest;
            manifest.finish(WorkflowOutcome::Failed {
                error: "workflow shutdown timed out".into(),
            });
            let _ = self.store.persist(&manifest);
        }
    }
}

fn validate_workflow_approval(
    record: &hi_policy::ApprovalRecord,
    run_id: &str,
    workflow_name: &str,
    approval_id: &hi_policy::ApprovalId,
    operation_digest: &hi_policy::OperationDigest,
) -> Result<(), RuntimeError> {
    let scope_matches = matches!(
        &record.request.scope,
        hi_policy::ResourceScope::Workflow {
            workflow_id,
            run_id: scope_run_id,
        } if workflow_id == workflow_name && scope_run_id == run_id
    );
    if record.request.approval_id != *approval_id
        || record.request.operation_digest != *operation_digest
        || record.request.run_id.as_deref() != Some(run_id)
        || record.request.capability != hi_policy::CapabilityKind::WorkflowExecution
        || record.request.tool != "workflow"
        || !scope_matches
    {
        return Err(RuntimeError::Approval(
            "approval request is not bound to this exact workflow run".into(),
        ));
    }
    Ok(())
}

fn spawn_run(
    manifest: WorkflowRunManifest,
    script: String,
    args: serde_json::Value,
    journal: Journal,
    ownership: WorkflowRunOwnership,
) -> ManagedWorkflowRun {
    let (host_tx, host_rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let params = WorkflowRunParams {
        script,
        args,
        journal,
        host_tx,
        cancel: cancel.clone(),
        max_ops: WorkflowRunParams::DEFAULT_MAX_OPS,
    };
    let task = tokio::task::spawn_blocking(move || crate::run_workflow(params));
    ManagedWorkflowRun {
        manifest,
        host_rx,
        cancel,
        task,
        ownership,
    }
}

fn new_run_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("run-{now}-{}", NEXT_RUN_ID.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hi_policy::{
        ApprovalDecision, ApprovalId, ApprovalRecord, ApprovalState, ApprovalStore, CapabilityKind,
        CapabilityRequest, OperationDigest, ResourceScope,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    struct CountingApprovalStore {
        claims: AtomicUsize,
        record: Mutex<ApprovalRecord>,
        fail_after_claim: AtomicBool,
    }

    impl CountingApprovalStore {
        fn approved_for_run(run_id: &str, workflow_name: &str) -> Self {
            let now = hi_policy::now_ms();
            Self {
                claims: AtomicUsize::new(0),
                record: Mutex::new(ApprovalRecord {
                    request: CapabilityRequest {
                        approval_id: ApprovalId("approval-1".into()),
                        capability: CapabilityKind::WorkflowExecution,
                        scope: ResourceScope::Workflow {
                            workflow_id: workflow_name.into(),
                            run_id: run_id.into(),
                        },
                        operation_digest: OperationDigest("digest-1".into()),
                        tool: "workflow".into(),
                        run_id: Some(run_id.into()),
                        session_id: None,
                        title: "resume workflow".into(),
                        redacted_detail: String::new(),
                        created_at_ms: now,
                        expires_at_ms: now.saturating_add(60_000),
                    },
                    state: ApprovalState::Approved,
                    decided_at_ms: Some(now),
                    consumed_at_ms: None,
                }),
                fail_after_claim: AtomicBool::new(false),
            }
        }

        fn claim_count(&self) -> usize {
            self.claims.load(Ordering::SeqCst)
        }

        fn fail_next_finalize_after_claim(&self) {
            self.fail_after_claim.store(true, Ordering::SeqCst);
        }
    }

    impl ApprovalStore for CountingApprovalStore {
        fn create(&self, _request: CapabilityRequest) -> anyhow::Result<ApprovalRecord> {
            anyhow::bail!("unused")
        }

        fn get(&self, id: &ApprovalId) -> anyhow::Result<Option<ApprovalRecord>> {
            let record = self.record.lock().expect("approval test mutex");
            Ok((record.request.approval_id == *id).then(|| record.clone()))
        }

        fn decide(
            &self,
            _id: &ApprovalId,
            _decision: ApprovalDecision,
        ) -> anyhow::Result<ApprovalRecord> {
            anyhow::bail!("unused")
        }

        fn claim(
            &self,
            id: &ApprovalId,
            digest: &OperationDigest,
        ) -> anyhow::Result<ApprovalRecord> {
            let mut record = self.record.lock().expect("approval test mutex");
            if record.request.approval_id != *id || record.request.operation_digest != *digest {
                anyhow::bail!("approval request mismatch");
            }
            if record.state != ApprovalState::Approved {
                anyhow::bail!("approval is not approved: {:?}", record.state);
            }
            let now = hi_policy::now_ms();
            record.state = ApprovalState::Consumed;
            record.consumed_at_ms = Some(now);
            let claimed = record.clone();
            self.claims.fetch_add(1, Ordering::SeqCst);
            if self.fail_after_claim.swap(false, Ordering::SeqCst) {
                crate::store::fail_nth_atomic_write_for_test(1);
            }
            Ok(claimed)
        }

        fn abandon_run(&self, _run_id: &str) -> anyhow::Result<u64> {
            Ok(0)
        }

        fn pending(&self) -> anyhow::Result<Vec<ApprovalRecord>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn take_active_transfers_run_ownership() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store);
        let id = manager
            .start(
                "test".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
            )
            .unwrap();

        let run = manager.take_active(&id).unwrap();
        assert!(manager.active_ids().next().is_none());
        let (manifest, _host_rx, _cancel, _task, _ownership) = run.into_parts();
        assert_eq!(manifest.run_id, id);
    }

    #[tokio::test]
    async fn join_persists_task_failure() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store.clone());
        let manifest = WorkflowRunManifest::new("panic-run".into(), "test".into(), 8).unwrap();
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        let (host_tx, host_rx) = mpsc::unbounded_channel();
        drop(host_tx);
        let task = tokio::task::spawn_blocking(|| -> WorkflowOutcome { panic!("boom") });
        let ownership = store.try_claim("panic-run").unwrap().unwrap();
        manager.active.insert(
            "panic-run".into(),
            ManagedWorkflowRun {
                manifest,
                host_rx,
                cancel: CancellationToken::new(),
                task,
                ownership,
            },
        );

        assert!(matches!(
            manager.join("panic-run").await.unwrap(),
            WorkflowOutcome::Failed { .. }
        ));
        let loaded = store.load("panic-run").unwrap();
        assert_eq!(loaded.manifest.status, StoredRunStatus::Failed);
        assert!(matches!(
            loaded.manifest.outcome,
            Some(WorkflowOutcome::Failed { .. })
        ));
        assert!(manager.active.is_empty());
    }

    #[test]
    fn resume_rejects_budget_below_spend_without_mutating_state() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest = WorkflowRunManifest::new("budget".into(), "test".into(), 10).unwrap();
        manifest.agent_spent = 7;
        manifest.status = StoredRunStatus::Paused;
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        let mut manager = WorkflowRuntimeManager::new(store.clone());
        assert!(matches!(
            manager.resume("budget", Some(6)),
            Err(RuntimeError::InvalidBudget {
                raised: 6,
                spent: 7
            })
        ));
        assert_eq!(
            store.load("budget").unwrap().manifest.status,
            StoredRunStatus::Paused
        );
    }

    #[test]
    fn resume_rejects_a_corrupt_journal_before_publishing_running() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest = WorkflowRunManifest::new("corrupt".into(), "test".into(), 8).unwrap();
        manifest.status = StoredRunStatus::Paused;
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        let journal_path = store.load("corrupt").unwrap().journal_path;
        std::fs::write(journal_path, b"not-json\n").unwrap();
        let mut manager = WorkflowRuntimeManager::new(store.clone());

        assert!(matches!(
            manager.resume("corrupt", None),
            Err(RuntimeError::Journal(_))
        ));
        assert_eq!(
            store.load("corrupt").unwrap().manifest.status,
            StoredRunStatus::Paused
        );
        assert!(manager.active.is_empty());
    }

    #[test]
    fn declarative_manifest_resume_fails_closed_without_mutating_state() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest =
            WorkflowRunManifest::new("declarative".into(), "test".into(), 8).unwrap();
        manifest.engine = crate::WorkflowEngineKind::Declarative;
        manifest.status = StoredRunStatus::Paused;
        store
            .register(
                &manifest,
                r#"{"metadata":{"name":"test"}}"#,
                &serde_json::json!({}),
            )
            .unwrap();
        let mut manager = WorkflowRuntimeManager::new(store.clone());

        assert!(matches!(
            manager.resume("declarative", None),
            Err(RuntimeError::UnsupportedEngine(
                crate::WorkflowEngineKind::Declarative
            ))
        ));
        assert_eq!(
            store.load("declarative").unwrap().manifest.status,
            StoredRunStatus::Paused
        );
    }

    #[tokio::test]
    async fn a_second_manager_cannot_resume_a_run_owned_by_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manifest = WorkflowRunManifest::new("exclusive".into(), "test".into(), 8).unwrap();
        manifest.status = StoredRunStatus::Paused;
        store
            .register(&manifest, "complete(1);", &serde_json::json!({}))
            .unwrap();
        let mut first = WorkflowRuntimeManager::new(store.clone());
        let mut second = WorkflowRuntimeManager::new(store.clone());

        first.resume("exclusive", None).unwrap();
        assert!(matches!(
            second.resume("exclusive", None),
            Err(RuntimeError::Owned(id)) if id == "exclusive"
        ));
        assert!(matches!(
            first.join("exclusive").await.unwrap(),
            WorkflowOutcome::Completed { .. }
        ));
    }

    #[tokio::test]
    async fn recovery_does_not_interrupt_a_run_owned_by_another_manager() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut owner = WorkflowRuntimeManager::new(store.clone());
        let id = owner
            .start(
                "test".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
            )
            .unwrap();
        let observer = WorkflowRuntimeManager::new(store.clone());

        assert!(observer.recover_interrupted().unwrap().is_empty());
        assert_eq!(
            store.load(&id).unwrap().manifest.status,
            StoredRunStatus::Running
        );
        owner.join(&id).await.unwrap();
    }

    #[test]
    fn startup_recovery_interrupts_running_runs_only() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        for (id, status) in [
            ("running", StoredRunStatus::Running),
            ("paused", StoredRunStatus::Paused),
            ("done", StoredRunStatus::Completed),
        ] {
            let mut manifest = WorkflowRunManifest::new(id.into(), "test".into(), 8).unwrap();
            manifest.status = status;
            store
                .register(&manifest, "complete(1);", &serde_json::json!({}))
                .unwrap();
        }
        let manager = WorkflowRuntimeManager::new(store.clone());
        let recovered = manager.recover_interrupted().unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(
            store.load("running").unwrap().manifest.status,
            StoredRunStatus::Interrupted
        );
        assert_eq!(
            store.load("paused").unwrap().manifest.status,
            StoredRunStatus::Paused
        );
        assert_eq!(
            store.load("done").unwrap().manifest.status,
            StoredRunStatus::Completed
        );
        assert!(manager.recover_interrupted().unwrap().is_empty());
    }

    #[test]
    fn approval_pause_survives_restart_and_plain_resume_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut first_manager = WorkflowRuntimeManager::new(store.clone());
        let run_id = first_manager.allocate_run_id();
        let id = first_manager
            .start_paused_for_approval(
                run_id,
                "test".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
                ("approval-1", "digest-1"),
            )
            .unwrap();
        drop(first_manager);

        let mut restored_manager = WorkflowRuntimeManager::new(store.clone());
        assert!(restored_manager.recover_interrupted().unwrap().is_empty());
        let restored = store.load(&id).unwrap().manifest;
        assert_eq!(restored.status, StoredRunStatus::Paused);
        assert_eq!(restored.pending_approval_id.as_deref(), Some("approval-1"));
        assert_eq!(
            restored.pending_operation_digest.as_deref(),
            Some("digest-1")
        );
        assert!(matches!(
            restored_manager.resume(&id, None),
            Err(RuntimeError::Approval(_))
        ));
        assert!(!restored_manager.active.contains_key(&id));
    }

    #[test]
    fn approval_gate_is_part_of_the_first_published_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store.clone());

        // Registration publishes script, args, and state. A former fourth
        // write changed Running to Paused and could fail after exposing an
        // ungated run. The gated implementation needs only the first three.
        crate::store::fail_nth_atomic_write_for_test(4);
        let run_id = manager.allocate_run_id();
        let result = manager.start_paused_for_approval(
            run_id,
            "test".into(),
            "complete(1);".into(),
            serde_json::json!({}),
            8,
            ("approval-1", "digest-1"),
        );
        crate::store::clear_atomic_write_failure_for_test();
        let id = result.unwrap();
        let manifest = store.load(&id).unwrap().manifest;
        assert_eq!(manifest.status, StoredRunStatus::Paused);
        assert_eq!(manifest.pending_approval_id.as_deref(), Some("approval-1"));
        assert_eq!(
            manifest.pending_operation_digest.as_deref(),
            Some("digest-1")
        );
    }

    #[test]
    fn approval_is_not_claimed_when_resume_preflight_fails() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store.clone());
        let run_id = manager.allocate_run_id();
        let id = manager
            .start_paused_for_approval(
                run_id,
                "test".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
                ("approval-1", "digest-1"),
            )
            .unwrap();
        let approvals = CountingApprovalStore::approved_for_run(&id, "test");

        let mut manifest = store.load(&id).unwrap().manifest;
        manifest.agent_spent = 7;
        store.persist(&manifest).unwrap();
        assert!(matches!(
            manager.resume_with_approval(&id, &approvals, "approval-1", "digest-1", Some(6),),
            Err(RuntimeError::InvalidBudget {
                raised: 6,
                spent: 7
            })
        ));
        assert_eq!(approvals.claim_count(), 0);

        crate::store::fail_nth_atomic_write_for_test(1);
        assert!(matches!(
            manager.resume_with_approval(&id, &approvals, "approval-1", "digest-1", Some(8),),
            Err(RuntimeError::Store(_))
        ));
        assert_eq!(approvals.claim_count(), 0);
    }

    #[test]
    fn approval_for_another_run_is_rejected_before_claim() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store.clone());
        let run_id = manager.allocate_run_id();
        let id = manager
            .start_paused_for_approval(
                run_id,
                "test".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
                ("approval-1", "digest-1"),
            )
            .unwrap();
        let approvals = CountingApprovalStore::approved_for_run("another-run", "test");

        assert!(matches!(
            manager.resume_with_approval(&id, &approvals, "approval-1", "digest-1", None),
            Err(RuntimeError::Approval(_))
        ));
        assert_eq!(approvals.claim_count(), 0);
        assert_eq!(
            store.load(&id).unwrap().manifest.status,
            StoredRunStatus::Paused
        );
    }

    #[tokio::test]
    async fn consumed_approval_reconciles_after_final_manifest_write_fails() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store.clone());
        let run_id = manager.allocate_run_id();
        let id = manager
            .start_paused_for_approval(
                run_id,
                "test".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
                ("approval-1", "digest-1"),
            )
            .unwrap();
        let approvals = CountingApprovalStore::approved_for_run(&id, "test");

        approvals.fail_next_finalize_after_claim();
        assert!(matches!(
            manager.resume_with_approval(&id, &approvals, "approval-1", "digest-1", None),
            Err(RuntimeError::Store(_))
        ));
        assert_eq!(approvals.claim_count(), 1);
        let claiming = store.load(&id).unwrap().manifest;
        assert_eq!(claiming.status, StoredRunStatus::ApprovalClaiming);
        assert_eq!(claiming.pending_approval_id.as_deref(), Some("approval-1"));

        manager
            .resume_with_approval(&id, &approvals, "approval-1", "digest-1", None)
            .unwrap();
        assert_eq!(approvals.claim_count(), 1, "claim remains one-shot");
        let running = store.load(&id).unwrap().manifest;
        assert_eq!(running.status, StoredRunStatus::Running);
        assert!(running.pending_approval_id.is_none());
        assert!(matches!(
            manager.join(&id).await.unwrap(),
            WorkflowOutcome::Completed { .. }
        ));
    }

    #[tokio::test]
    async fn approval_is_not_claimed_at_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store);
        let run_id = manager.allocate_run_id();
        let paused = manager
            .start_paused_for_approval(
                run_id,
                "guarded".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
                ("approval-1", "digest-1"),
            )
            .unwrap();
        let approvals = CountingApprovalStore::approved_for_run(&paused, "guarded");
        for index in 0..MAX_ACTIVE_RUNS {
            manager
                .start(
                    format!("active-{index}"),
                    "complete(1);".into(),
                    serde_json::json!({}),
                    8,
                )
                .unwrap();
        }

        assert!(matches!(
            manager.resume_with_approval(&paused, &approvals, "approval-1", "digest-1", None,),
            Err(RuntimeError::AtCapacity)
        ));
        assert_eq!(approvals.claim_count(), 0);
    }

    #[tokio::test]
    async fn approved_resume_consumes_once_and_clears_pending_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store.clone());
        let run_id = manager.allocate_run_id();
        let id = manager
            .start_paused_for_approval(
                run_id,
                "test".into(),
                "complete(1);".into(),
                serde_json::json!({}),
                8,
                ("approval-1", "digest-1"),
            )
            .unwrap();
        let approvals = CountingApprovalStore::approved_for_run(&id, "test");

        manager
            .resume_with_approval(&id, &approvals, "approval-1", "digest-1", None)
            .unwrap();
        assert_eq!(approvals.claim_count(), 1);
        let running = store.load(&id).unwrap().manifest;
        assert_eq!(running.status, StoredRunStatus::Running);
        assert!(running.pending_approval_id.is_none());
        assert!(running.pending_operation_digest.is_none());
        assert!(matches!(
            manager.join(&id).await.unwrap(),
            WorkflowOutcome::Completed { .. }
        ));
    }

    #[tokio::test]
    async fn shutdown_uses_one_deadline_and_persists_timeouts() {
        let dir = tempfile::tempdir().unwrap();
        let store = WorkflowRunStore::new(dir.path());
        let mut manager = WorkflowRuntimeManager::new(store.clone());
        for id in ["slow-a", "slow-b"] {
            let manifest = WorkflowRunManifest::new(id.into(), "test".into(), 8).unwrap();
            store
                .register(&manifest, "complete(1);", &serde_json::json!({}))
                .unwrap();
            let (host_tx, host_rx) = mpsc::unbounded_channel();
            drop(host_tx);
            let task = tokio::task::spawn_blocking(|| {
                std::thread::sleep(Duration::from_millis(200));
                WorkflowOutcome::Completed {
                    result: serde_json::Value::Null,
                }
            });
            let ownership = store.try_claim(id).unwrap().unwrap();
            manager.active.insert(
                id.into(),
                ManagedWorkflowRun {
                    manifest,
                    host_rx,
                    cancel: CancellationToken::new(),
                    task,
                    ownership,
                },
            );
        }
        let started = tokio::time::Instant::now();
        manager.shutdown(Duration::from_millis(20)).await;
        assert!(started.elapsed() < Duration::from_millis(100));
        assert!(manager.active.is_empty());
        for id in ["slow-a", "slow-b"] {
            assert_eq!(
                store.load(id).unwrap().manifest.status,
                StoredRunStatus::Failed
            );
        }
    }
}
