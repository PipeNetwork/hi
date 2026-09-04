use std::sync::Arc;

use super::common::{IsolatedWorkspace, NullUi, agent, bash_completion, write_completion};
use super::*;

struct CancelAfterResult {
    cancellation: TurnCancellation,
}

impl Ui for CancelAfterResult {
    fn assistant_text(&mut self, _: &str) {}
    fn assistant_reasoning(&mut self, _: &str) {}
    fn assistant_end(&mut self) {}
    fn tool_call(&mut self, _: &str, _: &str) {}
    fn tool_result(&mut self, _: &str, _: &str) {
        self.cancellation.cancel();
    }
    fn status(&mut self, _: &str) {}
    fn turn_end(&mut self, _: &str) {}
}

struct GatedDurability {
    entered: Arc<std::sync::atomic::AtomicUsize>,
    completed: Arc<std::sync::atomic::AtomicUsize>,
}

#[async_trait::async_trait]
impl WorkspaceDurability for GatedDurability {
    async fn mutation_started(&self, _: Option<Vec<String>>) -> anyhow::Result<()> {
        Ok(())
    }

    async fn checkpoint(&self) -> anyhow::Result<()> {
        self.entered
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        tokio::time::sleep(std::time::Duration::from_millis(75)).await;
        self.completed
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        Ok(())
    }
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_foreground_shell_reaps_before_workspace_settlement() {
    let workspace = IsolatedWorkspace::new("cancel-foreground-reap-settlement");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut agent = agent(
        vec![bash_completion(
            "printf started > foreground-started; sleep 600",
        )],
        cfg,
    );
    let foreground = agent.foreground_process_registry();
    let cancellation = TurnCancellation::new();
    let cancel_when_running = {
        let foreground = foreground.clone();
        let cancellation = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while foreground.active_count() == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("foreground shell never registered");
            cancellation.cancel();
        })
    };

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent.run_turn_cancellable("run a long command", &mut NullUi, cancellation),
    )
    .await
    .expect("foreground cancellation exceeded its reap/settlement bound")
    .expect("reaped foreground cancellation should produce a typed outcome");
    cancel_when_running.await.unwrap();

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    assert_eq!(
        foreground.active_count(),
        0,
        "foreground child was not reaped"
    );
    assert!(
        !workspace.path("foreground-started").exists(),
        "rollback raced or skipped the cancelled foreground writer"
    );
    let controller = agent.workspace_controller_status();
    assert_eq!(controller.state, hi_workspace::WorkspaceState::Ready);
    assert!(controller.active_operation.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_outcome_waits_for_durability_acknowledgement() {
    let workspace = IsolatedWorkspace::new("cancel-await-durability");
    let mut cfg = workspace.config();
    cfg.gates.verification = VerificationMode::Disabled;
    cfg.memory.finalize = false;
    cfg.memory.suggest_next_prompt = false;
    let mut agent = agent(vec![write_completion("cancelled.txt")], cfg);
    let entered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let completed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    agent.set_workspace_durability(Some(Arc::new(GatedDurability {
        entered: entered.clone(),
        completed: completed.clone(),
    })));
    let cancellation = TurnCancellation::new();
    let mut ui = CancelAfterResult {
        cancellation: cancellation.clone(),
    };
    let outcome = agent
        .run_turn_cancellable("write then cancel", &mut ui, cancellation)
        .await
        .expect("acknowledged cancellation should settle");

    assert_eq!(outcome.status, TurnStatus::Cancelled);
    let entered = entered.load(std::sync::atomic::Ordering::Acquire);
    assert!(entered > 0, "cancellation never attempted settlement");
    assert_eq!(
        completed.load(std::sync::atomic::Ordering::Acquire),
        entered,
        "Cancelled was returned with durability acknowledgement still pending"
    );
    assert_eq!(
        agent.workspace_controller_status().state,
        hi_workspace::WorkspaceState::Ready
    );
}
