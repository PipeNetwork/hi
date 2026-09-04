use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Semaphore;

use super::*;
use crate::{
    BackgroundJobId, BackgroundJobLifecycle, BackgroundJobPublication, BackgroundJobRegistration,
    BackgroundJobTerminal,
};

struct GatedLifecycle {
    registrations: Mutex<Vec<BackgroundJobRegistration>>,
    terminals: Mutex<Vec<(BackgroundJobId, BackgroundJobTerminal)>>,
    entered: Semaphore,
    release: Semaphore,
}

impl Default for GatedLifecycle {
    fn default() -> Self {
        Self {
            registrations: Mutex::new(Vec::new()),
            terminals: Mutex::new(Vec::new()),
            entered: Semaphore::new(0),
            release: Semaphore::new(0),
        }
    }
}

#[async_trait]
impl BackgroundJobLifecycle for GatedLifecycle {
    async fn register(&self, registration: BackgroundJobRegistration) -> Result<(), String> {
        self.registrations.lock().unwrap().push(registration);
        Ok(())
    }

    async fn observe_terminal(
        &self,
        id: &BackgroundJobId,
        terminal: BackgroundJobTerminal,
        _detail: Option<String>,
    ) -> Result<BackgroundJobPublication, String> {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        self.terminals.lock().unwrap().push((id.clone(), terminal));
        Ok(BackgroundJobPublication::Published)
    }

    async fn pending(&self, _source_id: &str) -> Vec<BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(&self, _pending: &[BackgroundJobId]) -> Result<(), String> {
        Ok(())
    }
}

async fn wait_entered(lifecycle: &GatedLifecycle) {
    tokio::time::timeout(Duration::from_secs(2), lifecycle.entered.acquire())
        .await
        .expect("lifecycle callback should start")
        .unwrap()
        .forget();
}

fn completed() -> BackgroundTaskOutcome {
    BackgroundTaskOutcome {
        id: String::new(),
        description: String::new(),
        subagent_type: String::new(),
        state: BackgroundTaskState::Completed,
        output: "done".into(),
        applied: false,
        changed_files: Vec::new(),
    }
}

#[tokio::test]
async fn task_success_is_not_visible_before_lifecycle_settlement() {
    let registry = BackgroundTaskRegistry::new();
    let lifecycle = Arc::new(GatedLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let id = registry
        .spawn(
            "read",
            "explore",
            Box::new(|| Box::pin(async { completed() })),
        )
        .await
        .unwrap();

    wait_entered(&lifecycle).await;
    assert_eq!(
        registry.poll(&id, Duration::ZERO).await.unwrap().state,
        BackgroundTaskState::Running
    );
    lifecycle.release.add_permits(1);
    assert_eq!(
        registry
            .poll(&id, Duration::from_secs(2))
            .await
            .unwrap()
            .state,
        BackgroundTaskState::Completed
    );
    assert_eq!(lifecycle.registrations.lock().unwrap().len(), 1);
    assert_eq!(lifecycle.terminals.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn task_cancel_is_not_visible_before_abort_and_lifecycle_settlement() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let lifecycle = Arc::new(GatedLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let id = registry
        .spawn(
            "cancel",
            "explore",
            Box::new(move || {
                Box::pin(async move {
                    let _ = started_tx.send(());
                    std::future::pending::<BackgroundTaskOutcome>().await
                })
            }),
        )
        .await
        .unwrap();
    started_rx.await.unwrap();

    let kill_registry = registry.clone();
    let kill_id = id.clone();
    let kill = tokio::spawn(async move { kill_registry.kill(&kill_id).await.unwrap() });
    wait_entered(&lifecycle).await;
    assert_eq!(
        registry.poll(&id, Duration::ZERO).await.unwrap().state,
        BackgroundTaskState::Running
    );
    lifecycle.release.add_permits(1);
    assert_eq!(kill.await.unwrap().state, BackgroundTaskState::Cancelled);
    assert_eq!(
        lifecycle.terminals.lock().unwrap()[0].1,
        BackgroundJobTerminal::Cancelled
    );
    assert_eq!(lifecycle.terminals.lock().unwrap().len(), 1);
}
