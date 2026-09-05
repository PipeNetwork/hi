use super::*;
use crate::tests::common::{agent, config};
use std::sync::Arc;
use std::time::Duration;

struct SettlementGate {
    entered: tokio::sync::Semaphore,
    release: tokio::sync::Semaphore,
}

#[async_trait::async_trait]
impl hi_tools::BackgroundJobLifecycle for SettlementGate {
    async fn register(&self, _: hi_tools::BackgroundJobRegistration) -> Result<(), String> {
        Ok(())
    }

    async fn observe_terminal(
        &self,
        _: &hi_tools::BackgroundJobId,
        _: hi_tools::BackgroundJobTerminal,
        _: Option<String>,
    ) -> Result<hi_tools::BackgroundJobPublication, String> {
        self.entered.add_permits(1);
        self.release.acquire().await.unwrap().forget();
        Ok(hi_tools::BackgroundJobPublication::Published)
    }

    async fn pending(&self, _: &str) -> Vec<hi_tools::BackgroundJobId> {
        Vec::new()
    }

    async fn settle_after_workspace(&self, _: &[hi_tools::BackgroundJobId]) -> Result<(), String> {
        Ok(())
    }
}

#[tokio::test]
async fn verification_waits_for_auto_background_reap_and_settlement() {
    // Auto-backgrounding is an opt-in process setting. Isolate that setting
    // from concurrent tests instead of mutating the shared test environment.
    if std::env::var("HI_TEST_VERIFY_REAP_CHILD").as_deref() != Ok("1") {
        let output = tokio::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "agent::turn::verify_run::tests::verification_waits_for_auto_background_reap_and_settlement", "--test-threads=1", "--nocapture"])
            .env("HI_TEST_VERIFY_REAP_CHILD", "1")
            .env("HI_BASH_AUTO_BACKGROUND", "1")
            .output().await.unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let mut subject = agent(vec![], config());
    let background = subject.runtime.background_arc();
    let gate = Arc::new(SettlementGate {
        entered: tokio::sync::Semaphore::new(0),
        release: tokio::sync::Semaphore::new(0),
    });
    background.set_job_lifecycle(gate.clone());
    let output = hi_tools::execute_in_runtime_shared_with_runner(
        subject.runtime.process_runner(),
        subject.runtime.root(),
        subject.runtime.state_root(),
        &subject.runtime.lsp(),
        &background,
        subject.runtime.read_cache(),
        &subject.runtime.repo_map_arc(),
        None,
        None,
        "bash",
        r#"{"command":"sleep 600","timeout":1}"#,
    )
    .await;
    let id = output
        .background
        .expect("foreground overrun must be adopted")
        .id;
    subject.set_turn_phase(TurnPhase::Model);
    subject.set_turn_phase(TurnPhase::WorkspaceRepair);
    let mut verifier = WorkspaceRepairVerifier::new(Vec::new(), 0);
    let mut snapshot = None;
    let fast_feedback = crate::agent::turn::fast_feedback::FastFeedbackState::default();
    let mut ui = crate::tests::common::NullUi;
    let mut verification = Box::pin(subject.run_workspace_repair_verification(
        &mut verifier,
        &[],
        &mut snapshot,
        false,
        0,
        &fast_feedback,
        &mut ui,
    ));
    let completed_before_settlement =
        tokio::time::timeout(Duration::from_millis(100), &mut verification)
            .await
            .is_ok();
    tokio::time::timeout(Duration::from_secs(3), gate.entered.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    gate.release.add_permits(1);
    if !completed_before_settlement {
        let outcome = tokio::time::timeout(Duration::from_secs(3), &mut verification)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(outcome, VerifyOutcome::NotRun));
    }
    drop(verification);
    // Also clean up the failing-before case before reporting the regression.
    background.kill_and_reap(&id).await.unwrap();
    assert!(
        !completed_before_settlement,
        "verification continued while the terminated writer's settlement callback was still blocked"
    );
}
