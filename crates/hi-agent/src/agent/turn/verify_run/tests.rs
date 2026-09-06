use super::*;
use crate::tests::common::{Canned, RecUi, agent, completion, config};
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
    // Exercise the unlimited-command handoff path without paying the
    // production 30-second foreground attachment budget. A positive `timeout`
    // is a hard process-lifetime deadline and is deliberately never adopted.
    background.set_foreground_handoff_budget(Some(Duration::from_millis(25)));
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
        r#"{"command":"sleep 600"}"#,
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

async fn local_service_verification_case() -> (crate::TurnOutcome, Vec<String>) {
    let mut cfg = config();
    cfg.gates.review = crate::ReviewPolicy::Off;
    cfg.gates.max_verify_repairs = 0;
    cfg.gates.verification = crate::VerificationMode::Explicit(vec![crate::VerifyStage::new(
        "test",
        "printf verifier-ran > verifier-ran.txt",
    )]);
    let source = cfg.paths.workspace_root.join("source.rs");
    let provider = Arc::new(Canned(std::sync::Mutex::new(vec![
        completion(
            vec![hi_ai::Content::ToolCall {
                id: "write-source".into(),
                name: "write".into(),
                arguments: serde_json::json!({
                    "path": source,
                    "content": "pub fn answer() -> u32 { 42 }\n",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![hi_ai::Content::ToolCall {
                id: "start-service".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "sleep 600",
                    "run_in_background": true,
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![hi_ai::Content::Text("The service is running.".into())],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider.clone(), cfg).unwrap();
    let verifier_marker = subject.runtime.root().join("verifier-ran.txt");
    let mut ui = RecUi::default();

    let outcome = subject
        .run_turn("perform these workspace operations", &mut ui)
        .await
        .unwrap_or_else(|error| {
            panic!(
                "turn failed: {error:#}; statuses={:?}; messages={:?}",
                ui.statuses,
                subject
                    .messages()
                    .iter()
                    .map(hi_ai::Message::text)
                    .collect::<Vec<_>>()
            )
        });

    assert!(
        provider.0.lock().unwrap().is_empty(),
        "successful verification must not re-enter the model obligation loop"
    );
    assert_eq!(outcome.verification, crate::VerificationStatus::Passed);
    assert_eq!(outcome.stop_reason, crate::TurnStopReason::Completed);
    assert!(subject.last_verify().is_some());
    assert_eq!(subject.last_turn_telemetry().verify_rounds, 1);
    assert_eq!(subject.last_verification_executions().len(), 1);
    assert!(verifier_marker.exists(), "the local verifier must execute");
    assert!(!ui.statuses.iter().any(|line| {
        line.contains("verification deferred")
            || line.contains("verification infrastructure failed")
            || line.contains("verification obligation")
    }));
    assert_eq!(subject.active_background_process_ids().len(), 1);
    assert_eq!(
        subject.workspace_controller_status().state,
        hi_workspace::WorkspaceState::Ready
    );
    assert_eq!(subject.workspace_controller_status().active_jobs.len(), 1);

    subject.settle_workspace_for_exit().await.unwrap();
    (outcome, ui.statuses)
}

#[tokio::test]
async fn local_service_does_not_defer_verification_or_reenter_obligation_loop() {
    let (outcome, _) = local_service_verification_case().await;
    assert_eq!(outcome.status, crate::TurnStatus::Completed);
}

#[tokio::test]
async fn no_change_build_and_background_service_complete_without_verification_failure() {
    let mut cfg = config();
    cfg.gates.verification = crate::VerificationMode::Auto;
    cfg.gates.review = crate::ReviewPolicy::Off;
    let provider = Arc::new(Canned(std::sync::Mutex::new(vec![
        completion(
            vec![hi_ai::Content::ToolCall {
                id: "build".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "sh -c 'printf build-ok'",
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![hi_ai::Content::ToolCall {
                id: "start-service".into(),
                name: "bash".into(),
                arguments: serde_json::json!({
                    "command": "sleep 600",
                    "run_in_background": true,
                })
                .to_string(),
            }],
            1,
            1,
        ),
        completion(
            vec![hi_ai::Content::Text(
                "The build passed and the service is running.".into(),
            )],
            1,
            1,
        ),
    ])));
    let mut subject = crate::Agent::new(provider.clone(), cfg).unwrap();
    let mut ui = RecUi::default();

    let outcome = subject
        .run_turn(
            "run the project check and launch its existing server",
            &mut ui,
        )
        .await
        .unwrap_or_else(|error| {
            panic!(
                "turn failed: {error:#}; statuses={:?}; messages={:?}",
                ui.statuses,
                subject
                    .messages()
                    .iter()
                    .map(hi_ai::Message::text)
                    .collect::<Vec<_>>()
            )
        });

    assert!(provider.0.lock().unwrap().is_empty());
    assert_eq!(outcome.status, crate::TurnStatus::Completed);
    assert_eq!(
        outcome.verification,
        crate::VerificationStatus::NotApplicable
    );
    assert_eq!(
        outcome.stop_reason,
        crate::TurnStopReason::NoApplicableVerification
    );
    assert!(outcome.changed_files.is_empty());
    assert!(!ui.statuses.iter().any(|line| {
        line.contains("infrastructure")
            || line.contains("workspace recovery")
            || line.contains("workspace admission")
    }));
    assert_eq!(subject.active_background_process_ids().len(), 1);
    subject.settle_workspace_for_exit().await.unwrap();
}
