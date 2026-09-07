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

struct ExecutionDropped(Option<tokio::sync::oneshot::Sender<()>>);
impl Drop for ExecutionDropped {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}

#[tokio::test]
async fn cancellation_signals_execution_before_waiting_for_publication_gate() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let lifecycle = Arc::new(GatedLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let (started, running) = tokio::sync::oneshot::channel();
    let (dropped, stopped) = tokio::sync::oneshot::channel();
    let id = registry
        .spawn(
            "gate",
            "explore",
            Box::new(move || {
                Box::pin(async move {
                    let _drop = ExecutionDropped(Some(dropped));
                    let _ = started.send(());
                    std::future::pending::<BackgroundTaskOutcome>().await
                })
            }),
        )
        .await
        .unwrap();
    running.await.unwrap();
    let gate = registry
        .tasks
        .lock()
        .await
        .get(&id)
        .unwrap()
        .lifecycle_gate
        .clone();
    let held = gate.lock().await;
    let request = registry.request_kill(&id).await.unwrap();
    tokio::time::timeout(Duration::from_millis(200), stopped)
        .await
        .expect("publication lock must not delay execution abort")
        .unwrap();
    assert_eq!(
        registry.poll(&id, Duration::ZERO).await.unwrap().state,
        BackgroundTaskState::Running
    );
    drop(held);
    wait_entered(&lifecycle).await;
    lifecycle.release.add_permits(1);
    let result = registry
        .finish_kill(
            &id,
            request,
            tokio::time::Instant::now() + Duration::from_secs(2),
        )
        .await;
    assert_eq!(result.state, BackgroundTaskState::Cancelled);
}

#[tokio::test]
async fn kill_all_signals_every_execution_before_draining_a_slow_callback() {
    let registry = Arc::new(BackgroundTaskRegistry::new());
    let lifecycle = Arc::new(GatedLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let mut stopped = Vec::new();
    for _ in 0..3 {
        let (started, running) = tokio::sync::oneshot::channel();
        let (dropped, done) = tokio::sync::oneshot::channel();
        registry
            .spawn(
                "drain",
                "explore",
                Box::new(move || {
                    Box::pin(async move {
                        let _drop = ExecutionDropped(Some(dropped));
                        let _ = started.send(());
                        std::future::pending::<BackgroundTaskOutcome>().await
                    })
                }),
            )
            .await
            .unwrap();
        running.await.unwrap();
        stopped.push(done);
    }
    let owner = registry.clone();
    let drain = tokio::spawn(async move {
        owner
            .kill_all_before(tokio::time::Instant::now() + Duration::from_millis(100))
            .await
    });
    tokio::time::timeout(Duration::from_millis(200), async {
        for done in stopped {
            done.await.unwrap();
        }
    })
    .await
    .expect("all execution aborts must precede publication draining");
    let pending = drain.await.unwrap();
    assert_eq!(pending.len(), 3);
    assert!(
        pending
            .iter()
            .all(|outcome| outcome.state == BackgroundTaskState::Running)
    );
    lifecycle.release.add_permits(3);
    let ids = pending
        .iter()
        .map(|outcome| outcome.id.clone())
        .collect::<Vec<_>>();
    let settled = registry.wait_all(&ids, Duration::from_secs(2)).await;
    assert!(
        settled
            .iter()
            .all(|outcome| outcome.state == BackgroundTaskState::Cancelled)
    );
}

#[tokio::test]
async fn cancelling_after_execution_does_not_drop_an_accepted_publication() {
    let registry = BackgroundTaskRegistry::new();
    let lifecycle = Arc::new(GatedLifecycle::default());
    registry.set_job_lifecycle(lifecycle.clone());
    let id = registry
        .spawn(
            "publishing",
            "explore",
            Box::new(|| Box::pin(async { completed() })),
        )
        .await
        .unwrap();
    wait_entered(&lifecycle).await;
    let request = registry.request_kill(&id).await.unwrap();
    let pending = registry
        .finish_kill(
            &id,
            request,
            tokio::time::Instant::now() + Duration::from_millis(20),
        )
        .await;
    assert_eq!(pending.state, BackgroundTaskState::Running);
    lifecycle.release.add_permits(1);
    let result = registry.wait_all(&[id], Duration::from_secs(2)).await;
    assert_eq!(result[0].state, BackgroundTaskState::Completed);
    assert_eq!(lifecycle.terminals.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn turn_cancellation_preserves_tasks_from_the_previous_turn() {
    let registry = BackgroundTaskRegistry::new();
    let prior = registry
        .spawn(
            "prior",
            "explore",
            Box::new(|| Box::pin(std::future::pending::<BackgroundTaskOutcome>())),
        )
        .await
        .unwrap();
    let baseline = registry.list().await;
    let current = registry
        .spawn(
            "current",
            "explore",
            Box::new(|| Box::pin(std::future::pending::<BackgroundTaskOutcome>())),
        )
        .await
        .unwrap();
    let settled = registry
        .kill_started_after_before(
            &baseline,
            tokio::time::Instant::now() + Duration::from_secs(2),
        )
        .await;
    assert_eq!(settled.len(), 1);
    assert_eq!(settled[0].id, current);
    assert_eq!(settled[0].state, BackgroundTaskState::Cancelled);
    assert_eq!(
        registry.poll(&prior, Duration::ZERO).await.unwrap().state,
        BackgroundTaskState::Running
    );
    registry.kill_all().await;
}
