use super::common::{IsolatedWorkspace, RecUi};
use super::*;
use hi_ai::test_support::{FakeOpenAiServer, Response};
use std::sync::Arc;

struct ReceiptSink(Arc<Mutex<Vec<TurnOutcome>>>);

impl SessionSink for ReceiptSink {
    fn record(&mut self, _: &[Message], _: Usage) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_compaction(&mut self, _: &[Message]) -> anyhow::Result<()> {
        Ok(())
    }
    fn record_turn_outcome(
        &mut self,
        outcome: &TurnOutcome,
        _: Option<&str>,
    ) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(outcome.clone());
        Ok(())
    }
}

#[tokio::test]
async fn physical_provider_exhaustion_after_tool_work_settles_actual_verification_once() {
    assert_provider_failure_settlement(true).await;
}

#[tokio::test]
async fn terminal_retryable_outage_after_tool_work_settles_without_extra_requests() {
    assert_provider_failure_settlement(false).await;
}

async fn assert_provider_failure_settlement(physical_limit: bool) {
    for check_passes in [true, false] {
        let workspace = IsolatedWorkspace::new("provider-terminal-verification");
        let checks = tempfile::NamedTempFile::new().unwrap();
        let check_path = serde_json::to_string(checks.path().to_str().unwrap()).unwrap();
        std::fs::write(workspace.path("validate.py"), format!(
            "from pathlib import Path\nimport sys\nassert Path('changed.rs').read_text() == 'pub fn answer() -> u8 {{ 42 }}\\n'\nwith open({check_path}, 'a') as receipt: receipt.write('checked\\n')\nprint('test result: ok. 18 passed; 0 failed' if {check_passes} else 'test gate::retained_edit ... FAILED')\nsys.exit(0 if {check_passes} else 1)\n",
            check_passes = if check_passes { "True" } else { "False" }
        )).unwrap();
        let tool = serde_json::json!({"choices":[{"delta":{"tool_calls":[{
            "index":0,"id":"one-effect","type":"function","function":{
                "name":"apply_patch",
                "arguments":serde_json::json!({"patch":"*** Begin Patch\n*** Add File: changed.rs\n+pub fn answer() -> u8 { 42 }\n*** End Patch"}).to_string()
            }
        }]},"finish_reason":"tool_calls"}]});
        let mut responses = vec![Response::sse(format!("data: {tool}\n\ndata: [DONE]\n\n"))];
        let expected_failed_sends = if physical_limit { 4 } else { 2 };
        responses.extend((0..expected_failed_sends).map(|_| {
            if physical_limit {
                Response::json(400, r#"{"error":{"message":"invalid tool JSON","code":"tool_protocol_error","retryable":true}}"#)
            } else {
                Response::json(503, r#"{"error":{"message":"upstream unavailable","code":"service_unavailable","retryable":true}}"#)
            }
        }));
        let Some(server) = FakeOpenAiServer::new(responses) else {
            return;
        };
        let mut cfg = workspace.config();
        cfg.loop_limits.max_recovery_interventions = 16;
        cfg.loop_limits.max_keep_working = 2;
        cfg.gates.verification = VerificationMode::Explicit(vec![VerifyStage::new(
            "retained source",
            "python3 validate.py",
        )]);
        cfg.gates.max_verify_repairs = 0;
        cfg.gates.review = ReviewPolicy::Always;
        cfg.memory.curate_skills = true;
        cfg.memory.suggest_next_prompt = true;
        let provider = hi_ai::OpenAiProvider::new(server.url().to_owned(), "test".into());
        let mut subject = Agent::new(Arc::new(provider), cfg).unwrap();
        subject.restore_plan(vec![PlanStep {
            title: "finish the complete requested implementation".into(),
            status: PlanStatus::Pending,
        }]);
        let receipts = Arc::new(Mutex::new(Vec::new()));
        subject.set_session(Box::new(ReceiptSink(receipts.clone())));
        let mut ui = RecUi::default();
        let error = subject
            .run_turn("build all of the requested implementation", &mut ui)
            .await
            .expect_err("missing completion remains a provider failure");
        let failure = TurnFailure::from_error(&error).expect("typed settled receipt");
        assert!(
            failure.body_settled(),
            "entry must not wrap or clean up the already-settled body again"
        );
        assert!(!failure.settlement_pending);
        assert_eq!(failure.outcome.status, TurnStatus::Failed);
        assert_eq!(
            failure.outcome.stop_reason,
            TurnStopReason::InfrastructureFailure
        );
        assert_eq!(
            failure.outcome.verification,
            if check_passes {
                VerificationStatus::Passed
            } else {
                VerificationStatus::Failed
            }
        );
        assert_eq!(
            server.bodies().len(),
            1 + expected_failed_sends,
            "one accepted tool operation plus the existing bounded failed sends"
        );
        assert_eq!(
            ui.tool_results
                .iter()
                .filter(|(name, _)| name == "apply_patch")
                .count(),
            1
        );
        assert_eq!(
            std::fs::read_to_string(workspace.path("changed.rs")).unwrap(),
            "pub fn answer() -> u8 { 42 }\n"
        );
        assert_eq!(
            std::fs::read_to_string(checks.path()).unwrap(),
            "checked\n",
            "one check of the final retained revision"
        );
        assert_eq!(subject.last_turn_telemetry().verify_rounds, 1);
        assert_eq!(receipts.lock().unwrap().len(), 1);
        assert!(
            subject.plan_incomplete(),
            "passing tests cannot complete unfinished plan work"
        );
        assert!(ui.turn_end.is_some(), "normal body settlement must run");
        let provider_error = hi_ai::provider_error_details(&error)
            .expect("original provider cause survives settlement");
        if physical_limit {
            let evidence = provider_error
                .request_failure
                .as_ref()
                .expect("physical evidence retained");
            assert_eq!(
                evidence.reason,
                hi_ai::RequestFailureReason::AttemptsExhausted
            );
            assert_eq!(evidence.attempts, 4);
            assert_eq!(evidence.attempt_limit, 4);
            assert_eq!(
                format!("{error:#}")
                    .matches("model request exhausted its physical attempt allowance")
                    .count(),
                1
            );
            assert!(
                subject
                    .last_turn_telemetry()
                    .wire_audit
                    .iter()
                    .any(|event| event
                        .get("request_failure")
                        .is_some_and(|failure| failure["attempts"] == 4))
            );
            assert!(
                subject
                    .task_recovery()
                    .last_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("4/4 sends"))
            );
        } else {
            assert_eq!(provider_error.kind, hi_ai::ProviderErrorKind::Outage);
            assert!(provider_error.request_failure.is_none());
            assert_eq!(
                format!("{error:#}").matches("upstream unavailable").count(),
                1
            );
        }
    }
}
