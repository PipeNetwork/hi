use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::{
    Journal, StoredRunStatus, WorkflowHostRequest, WorkflowOutcome, WorkflowRunManifest,
    WorkflowRunParams, WorkflowRunStore,
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
}

pub struct ManagedWorkflowRun {
    pub manifest: WorkflowRunManifest,
    pub host_rx: mpsc::UnboundedReceiver<WorkflowHostRequest>,
    cancel: CancellationToken,
    task: JoinHandle<WorkflowOutcome>,
}

impl ManagedWorkflowRun {
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

pub struct WorkflowRuntimeManager {
    store: WorkflowRunStore,
    active: HashMap<String, ManagedWorkflowRun>,
}

impl WorkflowRuntimeManager {
    pub fn new(store: WorkflowRunStore) -> Self {
        Self {
            store,
            active: HashMap::new(),
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
            if matches!(run.manifest.status, StoredRunStatus::Running | StoredRunStatus::Paused) {
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
        let id = new_run_id();
        let manifest = WorkflowRunManifest::new(id.clone(), workflow_name, agent_budget)?;
        self.store.register(&manifest, &script, &args)?;
        let run = spawn_run(
            manifest,
            script,
            args,
            Journal::load(self.store.journal_path(&id)?)?,
        );
        self.active.insert(id.clone(), run);
        Ok(id)
    }

    pub fn resume(&mut self, run_id: &str, raised_budget: Option<u64>) -> Result<(), RuntimeError> {
        if self.active.contains_key(run_id) {
            return Err(RuntimeError::Active(run_id.into()));
        }
        if self.active.len() >= MAX_ACTIVE_RUNS {
            return Err(RuntimeError::AtCapacity);
        }
        let stored = self.store.load(run_id)?;
        if !matches!(
            stored.manifest.status,
            StoredRunStatus::Paused
                | StoredRunStatus::BudgetExceeded
                | StoredRunStatus::Interrupted
        ) {
            return Err(RuntimeError::NotResumable(stored.manifest.status));
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
        manifest.status = StoredRunStatus::Running;
        manifest.outcome = None;
        self.store.persist(&manifest)?;
        let run = spawn_run(
            manifest,
            stored.script,
            stored.args,
            Journal::load(stored.journal_path)?,
        );
        self.active.insert(run_id.into(), run);
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
        let mut manifest = run.manifest;
        let outcome = match run.task.await {
            Ok(outcome) => outcome,
            Err(error) => WorkflowOutcome::Failed {
                error: format!("workflow task failed: {error}"),
            },
        };
        manifest.finish(outcome.clone());
        self.store.persist(&manifest)?;
        Ok(outcome)
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

fn spawn_run(
    manifest: WorkflowRunManifest,
    script: String,
    args: serde_json::Value,
    journal: Journal,
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
        manager.active.insert(
            "panic-run".into(),
            ManagedWorkflowRun {
                manifest,
                host_rx,
                cancel: CancellationToken::new(),
                task,
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
    fn startup_recovery_reconciles_nonterminal_runs_only() {
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
        assert_eq!(recovered.len(), 2);
        assert_eq!(store.load("running").unwrap().manifest.status, StoredRunStatus::Interrupted);
        assert_eq!(store.load("paused").unwrap().manifest.status, StoredRunStatus::Interrupted);
        assert_eq!(store.load("done").unwrap().manifest.status, StoredRunStatus::Completed);
        assert!(manager.recover_interrupted().unwrap().is_empty());
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
            manager.active.insert(
                id.into(),
                ManagedWorkflowRun {
                    manifest,
                    host_rx,
                    cancel: CancellationToken::new(),
                    task,
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
